import os
from typing import NamedTuple, Sequence

import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.dialects import sqlite
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetCacheState, AssetInfo
from app.assets.database.queries.common import (
    MAX_BIND_PARAMS,
    calculate_rows_per_statement,
    iter_chunks,
)
from app.assets.helpers import escape_sql_like_string


class CacheStateRow(NamedTuple):
    """Row from cache state query with joined asset data."""

    state_id: int
    file_path: str
    mtime_ns: int | None
    needs_verify: bool
    asset_id: str
    asset_hash: str | None
    size_bytes: int


def list_cache_states_by_asset_id(
    session: Session, *, asset_id: str
) -> Sequence[AssetCacheState]:
    return (
        (
            session.execute(
                select(AssetCacheState)
                .where(AssetCacheState.asset_id == asset_id)
                .order_by(AssetCacheState.id.asc())
            )
        )
        .scalars()
        .all()
    )


def upsert_cache_state(
    session: Session,
    asset_id: str,
    file_path: str,
    mtime_ns: int,
) -> tuple[bool, bool]:
    """Upsert a cache state by file_path. Returns (created, updated).

    Also restores cache states that were previously marked as missing.
    """
    vals = {
        "asset_id": asset_id,
        "file_path": file_path,
        "mtime_ns": int(mtime_ns),
        "is_missing": False,
    }
    ins = (
        sqlite.insert(AssetCacheState)
        .values(**vals)
        .on_conflict_do_nothing(index_elements=[AssetCacheState.file_path])
    )
    res = session.execute(ins)
    created = int(res.rowcount or 0) > 0

    if created:
        return True, False

    upd = (
        sa.update(AssetCacheState)
        .where(AssetCacheState.file_path == file_path)
        .where(
            sa.or_(
                AssetCacheState.asset_id != asset_id,
                AssetCacheState.mtime_ns.is_(None),
                AssetCacheState.mtime_ns != int(mtime_ns),
                AssetCacheState.is_missing == True,  # noqa: E712
            )
        )
        .values(asset_id=asset_id, mtime_ns=int(mtime_ns), is_missing=False)
    )
    res2 = session.execute(upd)
    updated = int(res2.rowcount or 0) > 0
    return False, updated


def mark_cache_states_missing_outside_prefixes(
    session: Session, valid_prefixes: list[str]
) -> int:
    """Mark cache states as missing when file_path doesn't match any valid prefix.

    This is a non-destructive soft-delete that preserves user metadata.
    Cache states can be restored if the file reappears in a future scan.

    Args:
        session: Database session
        valid_prefixes: List of absolute directory prefixes that are valid

    Returns:
        Number of cache states marked as missing
    """
    if not valid_prefixes:
        return 0

    def make_prefix_condition(prefix: str):
        base = prefix if prefix.endswith(os.sep) else prefix + os.sep
        escaped, esc = escape_sql_like_string(base)
        return AssetCacheState.file_path.like(escaped + "%", escape=esc)

    matches_valid_prefix = sa.or_(*[make_prefix_condition(p) for p in valid_prefixes])
    result = session.execute(
        sa.update(AssetCacheState)
        .where(~matches_valid_prefix)
        .where(AssetCacheState.is_missing == False)  # noqa: E712
        .values(is_missing=True)
    )
    return result.rowcount


def restore_cache_states_by_paths(session: Session, file_paths: list[str]) -> int:
    """Restore cache states that were previously marked as missing.

    Called when a file path is re-scanned and found to exist.

    Args:
        session: Database session
        file_paths: List of file paths that exist and should be restored

    Returns:
        Number of cache states restored
    """
    if not file_paths:
        return 0

    result = session.execute(
        sa.update(AssetCacheState)
        .where(AssetCacheState.file_path.in_(file_paths))
        .where(AssetCacheState.is_missing == True)  # noqa: E712
        .values(is_missing=False)
    )
    return result.rowcount


def get_unreferenced_unhashed_asset_ids(session: Session) -> list[str]:
    """Get IDs of unhashed assets (hash=None) with no active cache states.

    An asset is considered unreferenced if it has no cache states,
    or all its cache states are marked as missing.

    Returns:
        List of asset IDs that are unreferenced
    """
    active_cache_state_exists = (
        sa.select(sa.literal(1))
        .where(AssetCacheState.asset_id == Asset.id)
        .where(AssetCacheState.is_missing == False)  # noqa: E712
        .correlate(Asset)
        .exists()
    )
    unreferenced_subq = sa.select(Asset.id).where(
        Asset.hash.is_(None), ~active_cache_state_exists
    )
    return [row[0] for row in session.execute(unreferenced_subq).all()]


def delete_assets_by_ids(session: Session, asset_ids: list[str]) -> int:
    """Delete assets and their AssetInfos by ID.

    Args:
        session: Database session
        asset_ids: List of asset IDs to delete

    Returns:
        Number of assets deleted
    """
    if not asset_ids:
        return 0
    session.execute(sa.delete(AssetInfo).where(AssetInfo.asset_id.in_(asset_ids)))
    result = session.execute(sa.delete(Asset).where(Asset.id.in_(asset_ids)))
    return result.rowcount


def get_cache_states_for_prefixes(
    session: Session,
    prefixes: list[str],
    *,
    include_missing: bool = False,
) -> list[CacheStateRow]:
    """Get all cache states with paths matching any of the given prefixes.

    Args:
        session: Database session
        prefixes: List of absolute directory prefixes to match
        include_missing: If False (default), exclude cache states marked as missing

    Returns:
        List of cache state rows with joined asset data, ordered by asset_id, state_id
    """
    if not prefixes:
        return []

    conds = []
    for p in prefixes:
        base = os.path.abspath(p)
        if not base.endswith(os.sep):
            base += os.sep
        escaped, esc = escape_sql_like_string(base)
        conds.append(AssetCacheState.file_path.like(escaped + "%", escape=esc))

    query = (
        sa.select(
            AssetCacheState.id,
            AssetCacheState.file_path,
            AssetCacheState.mtime_ns,
            AssetCacheState.needs_verify,
            AssetCacheState.asset_id,
            Asset.hash,
            Asset.size_bytes,
        )
        .join(Asset, Asset.id == AssetCacheState.asset_id)
        .where(sa.or_(*conds))
    )

    if not include_missing:
        query = query.where(AssetCacheState.is_missing == False)  # noqa: E712

    rows = session.execute(
        query.order_by(AssetCacheState.asset_id.asc(), AssetCacheState.id.asc())
    ).all()

    return [
        CacheStateRow(
            state_id=row[0],
            file_path=row[1],
            mtime_ns=row[2],
            needs_verify=row[3],
            asset_id=row[4],
            asset_hash=row[5],
            size_bytes=int(row[6] or 0),
        )
        for row in rows
    ]


def bulk_update_needs_verify(session: Session, state_ids: list[int], value: bool) -> int:
    """Set needs_verify flag for multiple cache states.

    Returns: Number of rows updated
    """
    if not state_ids:
        return 0
    result = session.execute(
        sa.update(AssetCacheState)
        .where(AssetCacheState.id.in_(state_ids))
        .values(needs_verify=value)
    )
    return result.rowcount


def bulk_update_is_missing(session: Session, state_ids: list[int], value: bool) -> int:
    """Set is_missing flag for multiple cache states.

    Returns: Number of rows updated
    """
    if not state_ids:
        return 0
    result = session.execute(
        sa.update(AssetCacheState)
        .where(AssetCacheState.id.in_(state_ids))
        .values(is_missing=value)
    )
    return result.rowcount


def delete_cache_states_by_ids(session: Session, state_ids: list[int]) -> int:
    """Delete cache states by their IDs.

    Returns: Number of rows deleted
    """
    if not state_ids:
        return 0
    result = session.execute(
        sa.delete(AssetCacheState).where(AssetCacheState.id.in_(state_ids))
    )
    return result.rowcount


def delete_orphaned_seed_asset(session: Session, asset_id: str) -> bool:
    """Delete a seed asset (hash is None) and its AssetInfos.

    Returns: True if asset was deleted, False if not found
    """
    session.execute(sa.delete(AssetInfo).where(AssetInfo.asset_id == asset_id))
    asset = session.get(Asset, asset_id)
    if asset:
        session.delete(asset)
        return True
    return False


def bulk_insert_cache_states_ignore_conflicts(
    session: Session,
    rows: list[dict],
) -> None:
    """Bulk insert cache state rows with ON CONFLICT DO NOTHING on file_path.

    Each dict should have: asset_id, file_path, mtime_ns
    The is_missing field is automatically set to False for new inserts.
    """
    if not rows:
        return
    enriched_rows = [{**row, "is_missing": False} for row in rows]
    ins = sqlite.insert(AssetCacheState).on_conflict_do_nothing(
        index_elements=[AssetCacheState.file_path]
    )
    for chunk in iter_chunks(enriched_rows, calculate_rows_per_statement(4)):
        session.execute(ins, chunk)


def get_cache_states_by_paths_and_asset_ids(
    session: Session,
    path_to_asset: dict[str, str],
) -> set[str]:
    """Query cache states to find paths where our asset_id won the insert.

    Args:
        path_to_asset: Mapping of file_path -> asset_id we tried to insert

    Returns:
        Set of file_paths where our asset_id is present
    """
    if not path_to_asset:
        return set()

    paths = list(path_to_asset.keys())
    winners: set[str] = set()

    for chunk in iter_chunks(paths, MAX_BIND_PARAMS):
        result = session.execute(
            select(AssetCacheState.file_path).where(
                AssetCacheState.file_path.in_(chunk),
                AssetCacheState.asset_id.in_([path_to_asset[p] for p in chunk]),
            )
        )
        winners.update(result.scalars().all())

    return winners
