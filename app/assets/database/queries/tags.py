from typing import Iterable, Sequence

import sqlalchemy as sa
from sqlalchemy import select, delete, func
from sqlalchemy.dialects import sqlite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.assets.database.models import AssetInfo, AssetInfoMeta, AssetInfoTag, Tag
from app.assets.helpers import escape_like_prefix, normalize_tags, utcnow

MAX_BIND_PARAMS = 800


def _rows_per_stmt(cols: int) -> int:
    return max(1, MAX_BIND_PARAMS // max(1, cols))


def _chunk_rows(rows: list[dict], cols_per_row: int) -> Iterable[list[dict]]:
    if not rows:
        return []
    rows_per_stmt = max(1, MAX_BIND_PARAMS // max(1, cols_per_row))
    for i in range(0, len(rows), rows_per_stmt):
        yield rows[i : i + rows_per_stmt]


def _visible_owner_clause(owner_id: str) -> sa.sql.ClauseElement:
    """Build owner visibility predicate for reads. Owner-less rows are visible to everyone."""
    owner_id = (owner_id or "").strip()
    if owner_id == "":
        return AssetInfo.owner_id == ""
    return AssetInfo.owner_id.in_(["", owner_id])


def ensure_tags_exist(session: Session, names: Iterable[str], tag_type: str = "user") -> None:
    wanted = normalize_tags(list(names))
    if not wanted:
        return
    rows = [{"name": n, "tag_type": tag_type} for n in list(dict.fromkeys(wanted))]
    ins = (
        sqlite.insert(Tag)
        .values(rows)
        .on_conflict_do_nothing(index_elements=[Tag.name])
    )
    session.execute(ins)


def get_asset_tags(session: Session, asset_info_id: str) -> list[str]:
    return [
        tag_name for (tag_name,) in (
            session.execute(
                select(AssetInfoTag.tag_name).where(AssetInfoTag.asset_info_id == asset_info_id)
            )
        ).all()
    ]


def set_asset_info_tags(
    session: Session,
    asset_info_id: str,
    tags: Sequence[str],
    origin: str = "manual",
) -> dict:
    desired = normalize_tags(tags)

    current = set(
        tag_name for (tag_name,) in (
            session.execute(select(AssetInfoTag.tag_name).where(AssetInfoTag.asset_info_id == asset_info_id))
        ).all()
    )

    to_add = [t for t in desired if t not in current]
    to_remove = [t for t in current if t not in desired]

    if to_add:
        ensure_tags_exist(session, to_add, tag_type="user")
        session.add_all([
            AssetInfoTag(asset_info_id=asset_info_id, tag_name=t, origin=origin, added_at=utcnow())
            for t in to_add
        ])
        session.flush()

    if to_remove:
        session.execute(
            delete(AssetInfoTag)
            .where(AssetInfoTag.asset_info_id == asset_info_id, AssetInfoTag.tag_name.in_(to_remove))
        )
        session.flush()

    return {"added": to_add, "removed": to_remove, "total": desired}


def add_tags_to_asset_info(
    session: Session,
    asset_info_id: str,
    tags: Sequence[str],
    origin: str = "manual",
    create_if_missing: bool = True,
    asset_info_row = None,
) -> dict:
    if not asset_info_row:
        info = session.get(AssetInfo, asset_info_id)
        if not info:
            raise ValueError(f"AssetInfo {asset_info_id} not found")

    norm = normalize_tags(tags)
    if not norm:
        total = get_asset_tags(session, asset_info_id=asset_info_id)
        return {"added": [], "already_present": [], "total_tags": total}

    if create_if_missing:
        ensure_tags_exist(session, norm, tag_type="user")

    current = {
        tag_name
        for (tag_name,) in (
            session.execute(
                sa.select(AssetInfoTag.tag_name).where(AssetInfoTag.asset_info_id == asset_info_id)
            )
        ).all()
    }

    want = set(norm)
    to_add = sorted(want - current)

    if to_add:
        with session.begin_nested() as nested:
            try:
                session.add_all(
                    [
                        AssetInfoTag(
                            asset_info_id=asset_info_id,
                            tag_name=t,
                            origin=origin,
                            added_at=utcnow(),
                        )
                        for t in to_add
                    ]
                )
                session.flush()
            except IntegrityError:
                nested.rollback()

    after = set(get_asset_tags(session, asset_info_id=asset_info_id))
    return {
        "added": sorted(((after - current) & want)),
        "already_present": sorted(want & current),
        "total_tags": sorted(after),
    }


def remove_tags_from_asset_info(
    session: Session,
    asset_info_id: str,
    tags: Sequence[str],
) -> dict:
    info = session.get(AssetInfo, asset_info_id)
    if not info:
        raise ValueError(f"AssetInfo {asset_info_id} not found")

    norm = normalize_tags(tags)
    if not norm:
        total = get_asset_tags(session, asset_info_id=asset_info_id)
        return {"removed": [], "not_present": [], "total_tags": total}

    existing = {
        tag_name
        for (tag_name,) in (
            session.execute(
                sa.select(AssetInfoTag.tag_name).where(AssetInfoTag.asset_info_id == asset_info_id)
            )
        ).all()
    }

    to_remove = sorted(set(t for t in norm if t in existing))
    not_present = sorted(set(t for t in norm if t not in existing))

    if to_remove:
        session.execute(
            delete(AssetInfoTag)
            .where(
                AssetInfoTag.asset_info_id == asset_info_id,
                AssetInfoTag.tag_name.in_(to_remove),
            )
        )
        session.flush()

    total = get_asset_tags(session, asset_info_id=asset_info_id)
    return {"removed": to_remove, "not_present": not_present, "total_tags": total}


def add_missing_tag_for_asset_id(
    session: Session,
    asset_id: str,
    origin: str = "automatic",
) -> None:
    select_rows = (
        sa.select(
            AssetInfo.id.label("asset_info_id"),
            sa.literal("missing").label("tag_name"),
            sa.literal(origin).label("origin"),
            sa.literal(utcnow()).label("added_at"),
        )
        .where(AssetInfo.asset_id == asset_id)
        .where(
            sa.not_(
                sa.exists().where((AssetInfoTag.asset_info_id == AssetInfo.id) & (AssetInfoTag.tag_name == "missing"))
            )
        )
    )
    session.execute(
        sqlite.insert(AssetInfoTag)
        .from_select(
            ["asset_info_id", "tag_name", "origin", "added_at"],
            select_rows,
        )
        .on_conflict_do_nothing(index_elements=[AssetInfoTag.asset_info_id, AssetInfoTag.tag_name])
    )


def remove_missing_tag_for_asset_id(
    session: Session,
    asset_id: str,
) -> None:
    session.execute(
        sa.delete(AssetInfoTag).where(
            AssetInfoTag.asset_info_id.in_(sa.select(AssetInfo.id).where(AssetInfo.asset_id == asset_id)),
            AssetInfoTag.tag_name == "missing",
        )
    )


def list_tags_with_usage(
    session: Session,
    prefix: str | None = None,
    limit: int = 100,
    offset: int = 0,
    include_zero: bool = True,
    order: str = "count_desc",
    owner_id: str = "",
) -> tuple[list[tuple[str, str, int]], int]:
    counts_sq = (
        select(
            AssetInfoTag.tag_name.label("tag_name"),
            func.count(AssetInfoTag.asset_info_id).label("cnt"),
        )
        .select_from(AssetInfoTag)
        .join(AssetInfo, AssetInfo.id == AssetInfoTag.asset_info_id)
        .where(_visible_owner_clause(owner_id))
        .group_by(AssetInfoTag.tag_name)
        .subquery()
    )

    q = (
        select(
            Tag.name,
            Tag.tag_type,
            func.coalesce(counts_sq.c.cnt, 0).label("count"),
        )
        .select_from(Tag)
        .join(counts_sq, counts_sq.c.tag_name == Tag.name, isouter=True)
    )

    if prefix:
        escaped, esc = escape_like_prefix(prefix.strip().lower())
        q = q.where(Tag.name.like(escaped + "%", escape=esc))

    if not include_zero:
        q = q.where(func.coalesce(counts_sq.c.cnt, 0) > 0)

    if order == "name_asc":
        q = q.order_by(Tag.name.asc())
    else:
        q = q.order_by(func.coalesce(counts_sq.c.cnt, 0).desc(), Tag.name.asc())

    total_q = select(func.count()).select_from(Tag)
    if prefix:
        escaped, esc = escape_like_prefix(prefix.strip().lower())
        total_q = total_q.where(Tag.name.like(escaped + "%", escape=esc))
    if not include_zero:
        total_q = total_q.where(
            Tag.name.in_(select(AssetInfoTag.tag_name).group_by(AssetInfoTag.tag_name))
        )

    rows = (session.execute(q.limit(limit).offset(offset))).all()
    total = (session.execute(total_q)).scalar_one()

    rows_norm = [(name, ttype, int(count or 0)) for (name, ttype, count) in rows]
    return rows_norm, int(total or 0)


def bulk_insert_tags_and_meta(
    session: Session,
    tag_rows: list[dict],
    meta_rows: list[dict],
) -> None:
    """Batch insert into asset_info_tags and asset_info_meta with ON CONFLICT DO NOTHING.

    Args:
        session: Database session
        tag_rows: List of dicts with keys: asset_info_id, tag_name, origin, added_at
        meta_rows: List of dicts with keys: asset_info_id, key, ordinal, val_str, val_num, val_bool, val_json
    """
    if tag_rows:
        ins_tags = sqlite.insert(AssetInfoTag).on_conflict_do_nothing(
            index_elements=[AssetInfoTag.asset_info_id, AssetInfoTag.tag_name]
        )
        for chunk in _chunk_rows(tag_rows, cols_per_row=4):
            session.execute(ins_tags, chunk)

    if meta_rows:
        ins_meta = sqlite.insert(AssetInfoMeta).on_conflict_do_nothing(
            index_elements=[AssetInfoMeta.asset_info_id, AssetInfoMeta.key, AssetInfoMeta.ordinal]
        )
        for chunk in _chunk_rows(meta_rows, cols_per_row=7):
            session.execute(ins_meta, chunk)
