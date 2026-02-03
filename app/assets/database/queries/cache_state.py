import os
from typing import Sequence

import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetCacheState, AssetInfo
from app.assets.helpers import escape_like_prefix


def list_cache_states_by_asset_id(
    session: Session, *, asset_id: str
) -> Sequence[AssetCacheState]:
    return (
        session.execute(
            select(AssetCacheState)
            .where(AssetCacheState.asset_id == asset_id)
            .order_by(AssetCacheState.id.asc())
        )
    ).scalars().all()


def pick_best_live_path(states: Sequence[AssetCacheState]) -> str:
    """
    Return the best on-disk path among cache states:
      1) Prefer a path that exists with needs_verify == False (already verified).
      2) Otherwise, pick the first path that exists.
      3) Otherwise return empty string.
    """
    alive = [s for s in states if getattr(s, "file_path", None) and os.path.isfile(s.file_path)]
    if not alive:
        return ""
    for s in alive:
        if not getattr(s, "needs_verify", False):
            return s.file_path
    return alive[0].file_path


def prune_orphaned_assets(session: Session, roots: tuple[str, ...], prefixes_for_root_fn) -> int:
    """Prune cache states outside configured prefixes, then delete orphaned seed assets.

    Args:
        session: Database session
        roots: Tuple of root types to prune
        prefixes_for_root_fn: Function to get prefixes for a root type

    Returns:
        Number of orphaned assets deleted
    """
    all_prefixes = [os.path.abspath(p) for r in roots for p in prefixes_for_root_fn(r)]
    if not all_prefixes:
        return 0

    def make_prefix_condition(prefix: str):
        base = prefix if prefix.endswith(os.sep) else prefix + os.sep
        escaped, esc = escape_like_prefix(base)
        return AssetCacheState.file_path.like(escaped + "%", escape=esc)

    matches_valid_prefix = sa.or_(*[make_prefix_condition(p) for p in all_prefixes])

    orphan_subq = (
        sa.select(Asset.id)
        .outerjoin(AssetCacheState, AssetCacheState.asset_id == Asset.id)
        .where(Asset.hash.is_(None), AssetCacheState.id.is_(None))
    ).scalar_subquery()

    session.execute(sa.delete(AssetCacheState).where(~matches_valid_prefix))
    session.execute(sa.delete(AssetInfo).where(AssetInfo.asset_id.in_(orphan_subq)))
    result = session.execute(sa.delete(Asset).where(Asset.id.in_(orphan_subq)))
    return result.rowcount


def fast_db_consistency_pass(
    session: Session,
    root: str,
    *,
    prefixes_for_root_fn,
    escape_like_prefix_fn,
    fast_asset_file_check_fn,
    add_missing_tag_fn,
    remove_missing_tag_fn,
    collect_existing_paths: bool = False,
    update_missing_tags: bool = False,
) -> set[str] | None:
    """Fast DB+FS pass for a root:
      - Toggle needs_verify per state using fast check
      - For hashed assets with at least one fast-ok state in this root: delete stale missing states
      - For seed assets with all states missing: delete Asset and its AssetInfos
      - Optionally add/remove 'missing' tags based on fast-ok in this root
      - Optionally return surviving absolute paths
    """
    import contextlib

    prefixes = prefixes_for_root_fn(root)
    if not prefixes:
        return set() if collect_existing_paths else None

    conds = []
    for p in prefixes:
        base = os.path.abspath(p)
        if not base.endswith(os.sep):
            base += os.sep
        escaped, esc = escape_like_prefix_fn(base)
        conds.append(AssetCacheState.file_path.like(escaped + "%", escape=esc))

    rows = (
        session.execute(
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
            .order_by(AssetCacheState.asset_id.asc(), AssetCacheState.id.asc())
        )
    ).all()

    by_asset: dict[str, dict] = {}
    for sid, fp, mtime_db, needs_verify, aid, a_hash, a_size in rows:
        acc = by_asset.get(aid)
        if acc is None:
            acc = {"hash": a_hash, "size_db": int(a_size or 0), "states": []}
            by_asset[aid] = acc

        fast_ok = False
        try:
            exists = True
            fast_ok = fast_asset_file_check_fn(
                mtime_db=mtime_db,
                size_db=acc["size_db"],
                stat_result=os.stat(fp, follow_symlinks=True),
            )
        except FileNotFoundError:
            exists = False
        except OSError:
            exists = False

        acc["states"].append({
            "sid": sid,
            "fp": fp,
            "exists": exists,
            "fast_ok": fast_ok,
            "needs_verify": bool(needs_verify),
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
                session.execute(sa.delete(AssetInfo).where(AssetInfo.asset_id == aid))
                asset = session.get(Asset, aid)
                if asset:
                    session.delete(asset)
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
                    remove_missing_tag_fn(session, asset_id=aid)
        elif update_missing_tags:
            with contextlib.suppress(Exception):
                add_missing_tag_fn(session, asset_id=aid, origin="automatic")

        for s in states:
            if s["exists"]:
                survivors.add(os.path.abspath(s["fp"]))

    if stale_state_ids:
        session.execute(sa.delete(AssetCacheState).where(AssetCacheState.id.in_(stale_state_ids)))
    if to_set_verify:
        session.execute(
            sa.update(AssetCacheState)
            .where(AssetCacheState.id.in_(to_set_verify))
            .values(needs_verify=True)
        )
    if to_clear_verify:
        session.execute(
            sa.update(AssetCacheState)
            .where(AssetCacheState.id.in_(to_clear_verify))
            .values(needs_verify=False)
        )
    return survivors if collect_existing_paths else None
