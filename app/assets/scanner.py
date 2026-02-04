import contextlib
import logging
import os
import time
from typing import Literal

import folder_paths
from app.assets.database.queries import (
    add_missing_tag_for_asset_id,
    bulk_set_needs_verify,
    delete_cache_states_by_ids,
    delete_orphaned_seed_asset,
    ensure_tags_exist,
    get_cache_states_for_prefixes,
    remove_missing_tag_for_asset_id,
)
from app.assets.services.bulk_ingest import (
    batch_insert_seed_assets,
    prune_orphaned_assets,
)
from app.assets.services.file_utils import (
    get_mtime_ns,
    list_files_recursively,
    verify_file_unchanged,
)
from app.assets.services.path_utils import (
    compute_relative_filename,
    get_comfy_models_folders,
    get_name_and_tags_from_asset_path,
)
from app.database.db import create_session, dependencies_available

RootType = Literal["models", "input", "output"]


def get_prefixes_for_root(root: RootType) -> list[str]:
    if root == "models":
        bases: list[str] = []
        for _bucket, paths in get_comfy_models_folders():
            bases.extend(paths)
        return [os.path.abspath(p) for p in bases]
    if root == "input":
        return [os.path.abspath(folder_paths.get_input_directory())]
    if root == "output":
        return [os.path.abspath(folder_paths.get_output_directory())]
    return []


def collect_models_files() -> list[str]:
    out: list[str] = []
    for folder_name, bases in get_comfy_models_folders():
        rel_files = folder_paths.get_filename_list(folder_name) or []
        for rel_path in rel_files:
            abs_path = folder_paths.get_full_path(folder_name, rel_path)
            if not abs_path:
                continue
            abs_path = os.path.abspath(abs_path)
            allowed = False
            for b in bases:
                base_abs = os.path.abspath(b)
                with contextlib.suppress(Exception):
                    if os.path.commonpath([abs_path, base_abs]) == base_abs:
                        allowed = True
                        break
            if allowed:
                out.append(abs_path)
    return out


def sync_cache_states_with_filesystem(
    session,
    root: RootType,
    collect_existing_paths: bool = False,
    update_missing_tags: bool = False,
) -> set[str] | None:
    """Reconcile cache states with filesystem for a root.

    - Toggle needs_verify per state using fast mtime/size check
    - For hashed assets with at least one fast-ok state in this root: delete stale missing states
    - For seed assets with all states missing: delete Asset and its AssetInfos
    - Optionally add/remove 'missing' tags based on fast-ok in this root
    - Optionally return surviving absolute paths

    Args:
        session: Database session
        root: Root type to scan
        collect_existing_paths: If True, return set of surviving file paths
        update_missing_tags: If True, update 'missing' tags based on file status

    Returns:
        Set of surviving absolute paths if collect_existing_paths=True, else None
    """
    prefixes = get_prefixes_for_root(root)
    if not prefixes:
        return set() if collect_existing_paths else None

    rows = get_cache_states_for_prefixes(session, prefixes)

    by_asset: dict[str, dict] = {}
    for row in rows:
        acc = by_asset.get(row.asset_id)
        if acc is None:
            acc = {"hash": row.asset_hash, "size_db": row.size_bytes, "states": []}
            by_asset[row.asset_id] = acc

        fast_ok = False
        try:
            exists = True
            fast_ok = verify_file_unchanged(
                mtime_db=row.mtime_ns,
                size_db=acc["size_db"],
                stat_result=os.stat(row.file_path, follow_symlinks=True),
            )
        except FileNotFoundError:
            exists = False
        except OSError:
            exists = False

        acc["states"].append(
            {
                "sid": row.state_id,
                "fp": row.file_path,
                "exists": exists,
                "fast_ok": fast_ok,
                "needs_verify": row.needs_verify,
            }
        )

    to_set_verify: list[int] = []
    to_clear_verify: list[int] = []
    stale_state_ids: list[int] = []
    survivors: set[str] = set()

    for aid, acc in by_asset.items():
        a_hash = acc["hash"]
        states = acc["states"]
        any_fast_ok = any(s["fast_ok"] for s in states)
        all_missing = all(not s["exists"] for s in states)

        for s in states:
            if not s["exists"]:
                continue
            if s["fast_ok"] and s["needs_verify"]:
                to_clear_verify.append(s["sid"])
            if not s["fast_ok"] and not s["needs_verify"]:
                to_set_verify.append(s["sid"])

        if a_hash is None:
            if states and all_missing:
                delete_orphaned_seed_asset(session, aid)
            else:
                for s in states:
                    if s["exists"]:
                        survivors.add(os.path.abspath(s["fp"]))
            continue

        if any_fast_ok:
            for s in states:
                if not s["exists"]:
                    stale_state_ids.append(s["sid"])
            if update_missing_tags:
                with contextlib.suppress(Exception):
                    remove_missing_tag_for_asset_id(session, asset_id=aid)
        elif update_missing_tags:
            with contextlib.suppress(Exception):
                add_missing_tag_for_asset_id(session, asset_id=aid, origin="automatic")

        for s in states:
            if s["exists"]:
                survivors.add(os.path.abspath(s["fp"]))

    delete_cache_states_by_ids(session, stale_state_ids)
    bulk_set_needs_verify(session, to_set_verify, value=True)
    bulk_set_needs_verify(session, to_clear_verify, value=False)

    return survivors if collect_existing_paths else None


def _sync_root_safely(root: RootType) -> set[str]:
    """Sync a single root's cache states with the filesystem.

    Returns survivors (existing paths) or empty set on failure.
    """
    try:
        with create_session() as sess:
            survivors = sync_cache_states_with_filesystem(
                sess,
                root,
                collect_existing_paths=True,
                update_missing_tags=True,
            )
            sess.commit()
            return survivors or set()
    except Exception as e:
        logging.exception("fast DB scan failed for %s: %s", root, e)
        return set()


def _prune_orphans_safely(prefixes: list[str]) -> int:
    """Prune orphaned assets outside the given prefixes.

    Returns count pruned or 0 on failure.
    """
    try:
        with create_session() as sess:
            count = prune_orphaned_assets(sess, prefixes)
            sess.commit()
            return count
    except Exception as e:
        logging.exception("orphan pruning failed: %s", e)
        return 0


def _collect_paths_for_roots(roots: tuple[RootType, ...]) -> list[str]:
    """Collect all file paths for the given roots."""
    paths: list[str] = []
    if "models" in roots:
        paths.extend(collect_models_files())
    if "input" in roots:
        paths.extend(list_files_recursively(folder_paths.get_input_directory()))
    if "output" in roots:
        paths.extend(list_files_recursively(folder_paths.get_output_directory()))
    return paths


def _build_asset_specs(
    paths: list[str],
    existing_paths: set[str],
) -> tuple[list[dict], set[str], int]:
    """Build asset specs from paths, returning (specs, tag_pool, skipped_count)."""
    specs: list[dict] = []
    tag_pool: set[str] = set()
    skipped = 0

    for p in paths:
        abs_p = os.path.abspath(p)
        if abs_p in existing_paths:
            skipped += 1
            continue
        try:
            stat_p = os.stat(abs_p, follow_symlinks=False)
        except OSError:
            continue
        if not stat_p.st_size:
            continue
        name, tags = get_name_and_tags_from_asset_path(abs_p)
        specs.append(
            {
                "abs_path": abs_p,
                "size_bytes": stat_p.st_size,
                "mtime_ns": get_mtime_ns(stat_p),
                "info_name": name,
                "tags": tags,
                "fname": compute_relative_filename(abs_p),
            }
        )
        tag_pool.update(tags)

    return specs, tag_pool, skipped


def _insert_asset_specs(specs: list[dict], tag_pool: set[str]) -> int:
    """Insert asset specs into database, returning count of created infos."""
    if not specs:
        return 0
    with create_session() as sess:
        if tag_pool:
            ensure_tags_exist(sess, tag_pool, tag_type="user")
        result = batch_insert_seed_assets(sess, specs=specs, owner_id="")
        sess.commit()
        return result.inserted_infos


def seed_assets(roots: tuple[RootType, ...], enable_logging: bool = False) -> None:
    """Scan the given roots and seed the assets into the database."""
    if not dependencies_available():
        if enable_logging:
            logging.warning("Database dependencies not available, skipping assets scan")
        return

    t_start = time.perf_counter()

    existing_paths: set[str] = set()
    for r in roots:
        existing_paths.update(_sync_root_safely(r))

    all_prefixes = [os.path.abspath(p) for r in roots for p in get_prefixes_for_root(r)]
    orphans_pruned = _prune_orphans_safely(all_prefixes)

    paths = _collect_paths_for_roots(roots)
    specs, tag_pool, skipped_existing = _build_asset_specs(paths, existing_paths)
    created = _insert_asset_specs(specs, tag_pool)

    if enable_logging:
        logging.info(
            "Assets scan(roots=%s) completed in %.3fs (created=%d, skipped_existing=%d, orphans_pruned=%d, total_seen=%d)",
            roots,
            time.perf_counter() - t_start,
            created,
            skipped_existing,
            orphans_pruned,
            len(paths),
        )
