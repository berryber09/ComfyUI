import contextlib
import logging
import os
import time
from typing import Literal, TypedDict

import folder_paths
from app.assets.database.queries import (
    add_missing_tag_for_asset_id,
    bulk_update_enrichment_level,
    bulk_update_is_missing,
    bulk_update_needs_verify,
    delete_orphaned_seed_asset,
    delete_references_by_ids,
    ensure_tags_exist,
    get_asset_by_hash,
    get_references_for_prefixes,
    get_unenriched_references,
    reassign_asset_references,
    remove_missing_tag_for_asset_id,
    set_reference_metadata,
    update_asset_hash_and_mime,
)
from app.assets.services.bulk_ingest import (
    SeedAssetSpec,
    batch_insert_seed_assets,
    mark_assets_missing_outside_prefixes,
)
from app.assets.services.file_utils import (
    get_mtime_ns,
    list_files_recursively,
    verify_file_unchanged,
)
from app.assets.services.hashing import compute_blake3_hash
from app.assets.services.metadata_extract import extract_file_metadata
from app.assets.services.path_utils import (
    compute_relative_filename,
    get_comfy_models_folders,
    get_name_and_tags_from_asset_path,
)
from app.database.db import create_session, dependencies_available


class _RefInfo(TypedDict):
    ref_id: str
    fp: str
    exists: bool
    fast_ok: bool
    needs_verify: bool


class _AssetAccumulator(TypedDict):
    hash: str | None
    size_db: int
    refs: list[_RefInfo]


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


def get_all_known_prefixes() -> list[str]:
    """Get all known asset prefixes across all root types."""
    all_roots: tuple[RootType, ...] = ("models", "input", "output")
    return [
        os.path.abspath(p) for root in all_roots for p in get_prefixes_for_root(root)
    ]


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
                with contextlib.suppress(ValueError):
                    if os.path.commonpath([abs_path, base_abs]) == base_abs:
                        allowed = True
                        break
            if allowed:
                out.append(abs_path)
    return out


def sync_references_with_filesystem(
    session,
    root: RootType,
    collect_existing_paths: bool = False,
    update_missing_tags: bool = False,
) -> set[str] | None:
    """Reconcile asset references with filesystem for a root.

    - Toggle needs_verify per reference using fast mtime/size check
    - For hashed assets with at least one fast-ok ref: delete stale missing refs
    - For seed assets with all refs missing: delete Asset and its references
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

    rows = get_references_for_prefixes(
        session, prefixes, include_missing=update_missing_tags
    )

    by_asset: dict[str, _AssetAccumulator] = {}
    for row in rows:
        acc = by_asset.get(row.asset_id)
        if acc is None:
            acc = {"hash": row.asset_hash, "size_db": row.size_bytes, "refs": []}
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
        except PermissionError:
            exists = True
            logging.debug("Permission denied accessing %s", row.file_path)
        except OSError as e:
            exists = False
            logging.debug("OSError checking %s: %s", row.file_path, e)

        acc["refs"].append(
            {
                "ref_id": row.reference_id,
                "fp": row.file_path,
                "exists": exists,
                "fast_ok": fast_ok,
                "needs_verify": row.needs_verify,
            }
        )

    to_set_verify: list[str] = []
    to_clear_verify: list[str] = []
    stale_ref_ids: list[str] = []
    to_mark_missing: list[str] = []
    to_clear_missing: list[str] = []
    survivors: set[str] = set()

    for aid, acc in by_asset.items():
        a_hash = acc["hash"]
        refs = acc["refs"]
        any_fast_ok = any(r["fast_ok"] for r in refs)
        all_missing = all(not r["exists"] for r in refs)

        for r in refs:
            if not r["exists"]:
                to_mark_missing.append(r["ref_id"])
                continue
            if r["fast_ok"]:
                to_clear_missing.append(r["ref_id"])
                if r["needs_verify"]:
                    to_clear_verify.append(r["ref_id"])
            if not r["fast_ok"] and not r["needs_verify"]:
                to_set_verify.append(r["ref_id"])

        if a_hash is None:
            if refs and all_missing:
                delete_orphaned_seed_asset(session, aid)
            else:
                for r in refs:
                    if r["exists"]:
                        survivors.add(os.path.abspath(r["fp"]))
            continue

        if any_fast_ok:
            for r in refs:
                if not r["exists"]:
                    stale_ref_ids.append(r["ref_id"])
            if update_missing_tags:
                try:
                    remove_missing_tag_for_asset_id(session, asset_id=aid)
                except Exception as e:
                    logging.warning(
                        "Failed to remove missing tag for asset %s: %s", aid, e
                    )
        elif update_missing_tags:
            try:
                add_missing_tag_for_asset_id(session, asset_id=aid, origin="automatic")
            except Exception as e:
                logging.warning("Failed to add missing tag for asset %s: %s", aid, e)

        for r in refs:
            if r["exists"]:
                survivors.add(os.path.abspath(r["fp"]))

    delete_references_by_ids(session, stale_ref_ids)
    stale_set = set(stale_ref_ids)
    to_mark_missing = [ref_id for ref_id in to_mark_missing if ref_id not in stale_set]
    bulk_update_is_missing(session, to_mark_missing, value=True)
    bulk_update_is_missing(session, to_clear_missing, value=False)
    bulk_update_needs_verify(session, to_set_verify, value=True)
    bulk_update_needs_verify(session, to_clear_verify, value=False)

    return survivors if collect_existing_paths else None


def sync_root_safely(root: RootType) -> set[str]:
    """Sync a single root's references with the filesystem.

    Returns survivors (existing paths) or empty set on failure.
    """
    try:
        with create_session() as sess:
            survivors = sync_references_with_filesystem(
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


def mark_missing_outside_prefixes_safely(prefixes: list[str]) -> int:
    """Mark references as missing when outside the given prefixes.

    This is a non-destructive soft-delete. Returns count marked or 0 on failure.
    """
    try:
        with create_session() as sess:
            count = mark_assets_missing_outside_prefixes(sess, prefixes)
            sess.commit()
            return count
    except Exception as e:
        logging.exception("marking missing assets failed: %s", e)
        return 0


def collect_paths_for_roots(roots: tuple[RootType, ...]) -> list[str]:
    """Collect all file paths for the given roots."""
    paths: list[str] = []
    if "models" in roots:
        paths.extend(collect_models_files())
    if "input" in roots:
        paths.extend(list_files_recursively(folder_paths.get_input_directory()))
    if "output" in roots:
        paths.extend(list_files_recursively(folder_paths.get_output_directory()))
    return paths


def build_asset_specs(
    paths: list[str],
    existing_paths: set[str],
    enable_metadata_extraction: bool = True,
    compute_hashes: bool = False,
) -> tuple[list[SeedAssetSpec], set[str], int]:
    """Build asset specs from paths, returning (specs, tag_pool, skipped_count).

    Args:
        paths: List of file paths to process
        existing_paths: Set of paths that already exist in the database
        enable_metadata_extraction: If True, extract tier 1 & 2 metadata
        compute_hashes: If True, compute blake3 hashes (slow for large files)
    """
    specs: list[SeedAssetSpec] = []
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
        rel_fname = compute_relative_filename(abs_p)

        # Extract metadata (tier 1: filesystem, tier 2: safetensors header)
        metadata = None
        if enable_metadata_extraction:
            metadata = extract_file_metadata(
                abs_p,
                stat_result=stat_p,
                enable_safetensors=True,
                relative_filename=rel_fname,
            )

        # Compute hash if requested
        asset_hash: str | None = None
        if compute_hashes:
            try:
                digest = compute_blake3_hash(abs_p)
                asset_hash = "blake3:" + digest
            except Exception as e:
                logging.warning("Failed to hash %s: %s", abs_p, e)

        mime_type = metadata.content_type if metadata else None
        if mime_type is None:
            pass
        specs.append(
            {
                "abs_path": abs_p,
                "size_bytes": stat_p.st_size,
                "mtime_ns": get_mtime_ns(stat_p),
                "info_name": name,
                "tags": tags,
                "fname": rel_fname,
                "metadata": metadata,
                "hash": asset_hash,
                "mime_type": mime_type,
            }
        )
        tag_pool.update(tags)

    return specs, tag_pool, skipped


def build_stub_specs(
    paths: list[str],
    existing_paths: set[str],
) -> tuple[list[SeedAssetSpec], set[str], int]:
    """Build minimal stub specs for fast phase scanning.

    Only collects filesystem metadata (stat), no file content reading.
    This is the fastest possible scan to populate the asset database.

    Args:
        paths: List of file paths to process
        existing_paths: Set of paths that already exist in the database

    Returns:
        Tuple of (specs, tag_pool, skipped_count)
    """
    specs: list[SeedAssetSpec] = []
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
        rel_fname = compute_relative_filename(abs_p)

        specs.append(
            {
                "abs_path": abs_p,
                "size_bytes": stat_p.st_size,
                "mtime_ns": get_mtime_ns(stat_p),
                "info_name": name,
                "tags": tags,
                "fname": rel_fname,
                "metadata": None,
                "hash": None,
                "mime_type": None,
            }
        )
        tag_pool.update(tags)

    return specs, tag_pool, skipped


def insert_asset_specs(specs: list[SeedAssetSpec], tag_pool: set[str]) -> int:
    """Insert asset specs into database, returning count of created refs."""
    if not specs:
        return 0
    with create_session() as sess:
        if tag_pool:
            ensure_tags_exist(sess, tag_pool, tag_type="user")
        result = batch_insert_seed_assets(sess, specs=specs, owner_id="")
        sess.commit()
        return result.inserted_refs


def seed_assets(
    roots: tuple[RootType, ...],
    enable_logging: bool = False,
    compute_hashes: bool = False,
) -> None:
    """Scan the given roots and seed the assets into the database.

    Args:
        roots: Tuple of root types to scan (models, input, output)
        enable_logging: If True, log progress and completion messages
        compute_hashes: If True, compute blake3 hashes (slow for large files)

    Note: This function does not mark missing assets.
    Call mark_missing_outside_prefixes_safely separately if cleanup is needed.
    """
    if not dependencies_available():
        if enable_logging:
            logging.warning("Database dependencies not available, skipping assets scan")
        return

    t_start = time.perf_counter()

    existing_paths: set[str] = set()
    for r in roots:
        existing_paths.update(sync_root_safely(r))

    paths = collect_paths_for_roots(roots)
    specs, tag_pool, skipped_existing = build_asset_specs(
        paths, existing_paths, compute_hashes=compute_hashes
    )
    created = insert_asset_specs(specs, tag_pool)

    if enable_logging:
        logging.info(
            "Assets scan(roots=%s) completed in %.3fs "
            "(created=%d, skipped_existing=%d, total_seen=%d)",
            roots,
            time.perf_counter() - t_start,
            created,
            skipped_existing,
            len(paths),
        )


# Enrichment level constants
ENRICHMENT_STUB = 0  # Fast scan: path, size, mtime only
ENRICHMENT_METADATA = 1  # Metadata extracted (safetensors header, mime type)
ENRICHMENT_HASHED = 2  # Hash computed (blake3)


def get_unenriched_assets_for_roots(
    roots: tuple[RootType, ...],
    max_level: int = ENRICHMENT_STUB,
    limit: int = 1000,
) -> list:
    """Get assets that need enrichment for the given roots.

    Args:
        roots: Tuple of root types to scan
        max_level: Maximum enrichment level to include
        limit: Maximum number of rows to return

    Returns:
        List of UnenrichedReferenceRow
    """
    prefixes: list[str] = []
    for root in roots:
        prefixes.extend(get_prefixes_for_root(root))

    if not prefixes:
        return []

    with create_session() as sess:
        return get_unenriched_references(
            sess, prefixes, max_level=max_level, limit=limit
        )


def enrich_asset(
    file_path: str,
    reference_id: str,
    asset_id: str,
    extract_metadata: bool = True,
    compute_hash: bool = False,
) -> int:
    """Enrich a single asset with metadata and/or hash.

    Args:
        file_path: Absolute path to the file
        reference_id: ID of the reference to update
        asset_id: ID of the asset to update (for mime_type and hash)
        extract_metadata: If True, extract safetensors header and mime type
        compute_hash: If True, compute blake3 hash

    Returns:
        New enrichment level achieved
    """
    new_level = ENRICHMENT_STUB

    try:
        stat_p = os.stat(file_path, follow_symlinks=True)
    except OSError:
        return new_level

    rel_fname = compute_relative_filename(file_path)
    mime_type: str | None = None

    if extract_metadata:
        metadata = extract_file_metadata(
            file_path,
            stat_result=stat_p,
            enable_safetensors=True,
            relative_filename=rel_fname,
        )
        if metadata:
            mime_type = metadata.content_type
            new_level = ENRICHMENT_METADATA

    full_hash: str | None = None
    if compute_hash:
        try:
            digest = compute_blake3_hash(file_path)
            full_hash = f"blake3:{digest}"
            new_level = ENRICHMENT_HASHED
        except Exception as e:
            logging.warning("Failed to hash %s: %s", file_path, e)

    with create_session() as sess:
        if extract_metadata and metadata:
            user_metadata = metadata.to_user_metadata()
            set_reference_metadata(sess, reference_id, user_metadata)

        if full_hash:
            existing = get_asset_by_hash(sess, full_hash)
            if existing and existing.id != asset_id:
                reassign_asset_references(sess, asset_id, existing.id, reference_id)
                delete_orphaned_seed_asset(sess, asset_id)
                if mime_type:
                    update_asset_hash_and_mime(sess, existing.id, mime_type=mime_type)
            else:
                update_asset_hash_and_mime(sess, asset_id, full_hash, mime_type)
        elif mime_type:
            update_asset_hash_and_mime(sess, asset_id, mime_type=mime_type)

        bulk_update_enrichment_level(sess, [reference_id], new_level)
        sess.commit()

    return new_level


def enrich_assets_batch(
    rows: list,
    extract_metadata: bool = True,
    compute_hash: bool = False,
) -> tuple[int, int]:
    """Enrich a batch of assets.

    Args:
        rows: List of UnenrichedReferenceRow from get_unenriched_assets_for_roots
        extract_metadata: If True, extract metadata for each asset
        compute_hash: If True, compute hash for each asset

    Returns:
        Tuple of (enriched_count, failed_count)
    """
    enriched = 0
    failed = 0

    for row in rows:
        try:
            new_level = enrich_asset(
                file_path=row.file_path,
                reference_id=row.reference_id,
                asset_id=row.asset_id,
                extract_metadata=extract_metadata,
                compute_hash=compute_hash,
            )
            if new_level > row.enrichment_level:
                enriched += 1
            else:
                failed += 1
        except Exception as e:
            logging.warning("Failed to enrich %s: %s", row.file_path, e)
            failed += 1

    return enriched, failed
