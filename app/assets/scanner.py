import contextlib
import logging
import os
import time
import uuid
from typing import Literal

from sqlalchemy.orm import Session

import folder_paths
from app.assets.database.queries import (
    add_missing_tag_for_asset_id,
    bulk_insert_asset_infos_ignore_conflicts,
    bulk_insert_assets,
    bulk_insert_cache_states_ignore_conflicts,
    bulk_insert_tags_and_meta,
    bulk_set_needs_verify,
    delete_assets_by_ids,
    delete_cache_states_by_ids,
    delete_cache_states_outside_prefixes,
    delete_orphaned_seed_asset,
    ensure_tags_exist,
    get_asset_info_ids_by_ids,
    get_cache_states_by_paths_and_asset_ids,
    get_cache_states_for_prefixes,
    get_orphaned_seed_asset_ids,
    remove_missing_tag_for_asset_id,
)
from app.assets.helpers import get_utc_now
from app.assets.services.path_utils import (
    compute_relative_filename,
    get_comfy_models_folders,
    get_name_and_tags_from_asset_path,
)
from app.database.db import create_session, dependencies_available

RootType = Literal["models", "input", "output"]


def verify_asset_file_unchanged(
    mtime_db: int | None,
    size_db: int | None,
    stat_result: os.stat_result,
) -> bool:
    if mtime_db is None:
        return False
    actual_mtime_ns = getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000))
    if int(mtime_db) != int(actual_mtime_ns):
        return False
    sz = int(size_db or 0)
    if sz > 0:
        return int(stat_result.st_size) == sz
    return True


def list_files_recursively(base_dir: str) -> list[str]:
    out: list[str] = []
    base_abs = os.path.abspath(base_dir)
    if not os.path.isdir(base_abs):
        return out
    for dirpath, _subdirs, filenames in os.walk(base_abs, topdown=True, followlinks=False):
        for name in filenames:
            out.append(os.path.abspath(os.path.join(dirpath, name)))
    return out


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


def _batch_insert_assets_from_paths(
    session: Session,
    specs: list[dict],
    owner_id: str = "",
) -> dict:
    """Seed assets from filesystem specs in batch.

    Each spec is a dict with keys:
      - abs_path: str
      - size_bytes: int
      - mtime_ns: int
      - info_name: str
      - tags: list[str]
      - fname: Optional[str]

    This function orchestrates:
    1. Insert seed Assets (hash=NULL)
    2. Claim cache states with ON CONFLICT DO NOTHING
    3. Query to find winners (paths where our asset_id was inserted)
    4. Delete Assets for losers (path already claimed by another asset)
    5. Insert AssetInfo for winners
    6. Insert tags and metadata for successfully inserted AssetInfos

    Returns:
        dict with keys: inserted_infos, won_states, lost_states
    """
    if not specs:
        return {"inserted_infos": 0, "won_states": 0, "lost_states": 0}

    now = get_utc_now()
    asset_rows: list[dict] = []
    state_rows: list[dict] = []
    path_to_asset: dict[str, str] = {}
    asset_to_info: dict[str, dict] = {}
    path_list: list[str] = []

    for sp in specs:
        ap = os.path.abspath(sp["abs_path"])
        aid = str(uuid.uuid4())
        iid = str(uuid.uuid4())
        path_list.append(ap)
        path_to_asset[ap] = aid

        asset_rows.append({
            "id": aid,
            "hash": None,
            "size_bytes": sp["size_bytes"],
            "mime_type": None,
            "created_at": now,
        })
        state_rows.append({
            "asset_id": aid,
            "file_path": ap,
            "mtime_ns": sp["mtime_ns"],
        })
        asset_to_info[aid] = {
            "id": iid,
            "owner_id": owner_id,
            "name": sp["info_name"],
            "asset_id": aid,
            "preview_id": None,
            "user_metadata": {"filename": sp["fname"]} if sp["fname"] else None,
            "created_at": now,
            "updated_at": now,
            "last_access_time": now,
            "_tags": sp["tags"],
            "_filename": sp["fname"],
        }

    bulk_insert_assets(session, asset_rows)
    bulk_insert_cache_states_ignore_conflicts(session, state_rows)
    winners_by_path = get_cache_states_by_paths_and_asset_ids(session, path_to_asset)

    all_paths_set = set(path_list)
    losers_by_path = all_paths_set - winners_by_path
    lost_assets = [path_to_asset[p] for p in losers_by_path]

    if lost_assets:
        delete_assets_by_ids(session, lost_assets)

    if not winners_by_path:
        return {"inserted_infos": 0, "won_states": 0, "lost_states": len(losers_by_path)}

    winner_info_rows = [asset_to_info[path_to_asset[p]] for p in winners_by_path]
    db_info_rows = [
        {
            "id": row["id"],
            "owner_id": row["owner_id"],
            "name": row["name"],
            "asset_id": row["asset_id"],
            "preview_id": row["preview_id"],
            "user_metadata": row["user_metadata"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_access_time": row["last_access_time"],
        }
        for row in winner_info_rows
    ]
    bulk_insert_asset_infos_ignore_conflicts(session, db_info_rows)

    all_info_ids = [row["id"] for row in winner_info_rows]
    inserted_info_ids = get_asset_info_ids_by_ids(session, all_info_ids)

    tag_rows: list[dict] = []
    meta_rows: list[dict] = []
    if inserted_info_ids:
        for row in winner_info_rows:
            iid = row["id"]
            if iid not in inserted_info_ids:
                continue
            for t in row["_tags"]:
                tag_rows.append({
                    "asset_info_id": iid,
                    "tag_name": t,
                    "origin": "automatic",
                    "added_at": now,
                })
            if row["_filename"]:
                meta_rows.append({
                    "asset_info_id": iid,
                    "key": "filename",
                    "ordinal": 0,
                    "val_str": row["_filename"],
                    "val_num": None,
                    "val_bool": None,
                    "val_json": None,
                })

    bulk_insert_tags_and_meta(session, tag_rows=tag_rows, meta_rows=meta_rows)

    return {
        "inserted_infos": len(inserted_info_ids),
        "won_states": len(winners_by_path),
        "lost_states": len(losers_by_path),
    }


def prune_orphaned_assets(session, valid_prefixes: list[str]) -> int:
    """Prune cache states outside valid prefixes, then delete orphaned seed assets.

    Args:
        session: Database session
        valid_prefixes: List of absolute directory prefixes that are valid

    Returns:
        Number of orphaned assets deleted
    """
    delete_cache_states_outside_prefixes(session, valid_prefixes)
    orphan_ids = get_orphaned_seed_asset_ids(session)
    return delete_assets_by_ids(session, orphan_ids)


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
            fast_ok = verify_asset_file_unchanged(
                mtime_db=row.mtime_ns,
                size_db=acc["size_db"],
                stat_result=os.stat(row.file_path, follow_symlinks=True),
            )
        except FileNotFoundError:
            exists = False
        except OSError:
            exists = False

        acc["states"].append({
            "sid": row.state_id,
            "fp": row.file_path,
            "exists": exists,
            "fast_ok": fast_ok,
            "needs_verify": row.needs_verify,
        })

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
        specs.append({
            "abs_path": abs_p,
            "size_bytes": stat_p.st_size,
            "mtime_ns": getattr(stat_p, "st_mtime_ns", int(stat_p.st_mtime * 1_000_000_000)),
            "info_name": name,
            "tags": tags,
            "fname": compute_relative_filename(abs_p),
        })
        tag_pool.update(tags)

    return specs, tag_pool, skipped


def _insert_asset_specs(specs: list[dict], tag_pool: set[str]) -> int:
    """Insert asset specs into database, returning count of created infos."""
    if not specs:
        return 0
    with create_session() as sess:
        if tag_pool:
            ensure_tags_exist(sess, tag_pool, tag_type="user")
        result = _batch_insert_assets_from_paths(sess, specs=specs, owner_id="")
        sess.commit()
        return result["inserted_infos"]


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

    all_prefixes = [
        os.path.abspath(p) for r in roots for p in get_prefixes_for_root(r)
    ]
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
