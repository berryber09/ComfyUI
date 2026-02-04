from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Sequence

import sqlalchemy as sa
from sqlalchemy import delete, exists, select
from sqlalchemy.dialects import sqlite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, contains_eager, noload

from app.assets.database.models import (
    Asset,
    AssetInfo,
    AssetInfoMeta,
    AssetInfoTag,
    Tag,
)
from app.assets.helpers import escape_sql_like_string, get_utc_now, normalize_tags


def check_is_scalar(v):
    if v is None:
        return True
    if isinstance(v, bool):
        return True
    if isinstance(v, (int, float, Decimal, str)):
        return True
    return False


def _scalar_to_row(key: str, ordinal: int, value) -> dict:
    """Convert a scalar value to a typed projection row."""
    if value is None:
        return {
            "key": key, "ordinal": ordinal,
            "val_str": None, "val_num": None, "val_bool": None, "val_json": None
        }
    if isinstance(value, bool):
        return {"key": key, "ordinal": ordinal, "val_bool": bool(value)}
    if isinstance(value, (int, float, Decimal)):
        num = value if isinstance(value, Decimal) else Decimal(str(value))
        return {"key": key, "ordinal": ordinal, "val_num": num}
    if isinstance(value, str):
        return {"key": key, "ordinal": ordinal, "val_str": value}
    return {"key": key, "ordinal": ordinal, "val_json": value}


def convert_metadata_to_rows(key: str, value) -> list[dict]:
    """
    Turn a metadata key/value into typed projection rows.
    Returns list[dict] with keys:
      key, ordinal, and one of val_str / val_num / val_bool / val_json (others None)
    """
    if value is None:
        return [_scalar_to_row(key, 0, None)]

    if check_is_scalar(value):
        return [_scalar_to_row(key, 0, value)]

    if isinstance(value, list):
        if all(check_is_scalar(x) for x in value):
            return [_scalar_to_row(key, i, x) for i, x in enumerate(value)]
        return [{"key": key, "ordinal": i, "val_json": x} for i, x in enumerate(value)]

    return [{"key": key, "ordinal": 0, "val_json": value}]

MAX_BIND_PARAMS = 800


def _calculate_rows_per_statement(cols: int) -> int:
    return max(1, MAX_BIND_PARAMS // max(1, cols))


def _iter_chunks(seq, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _build_visible_owner_clause(owner_id: str) -> sa.sql.ClauseElement:
    """Build owner visibility predicate for reads. Owner-less rows are visible to everyone."""
    owner_id = (owner_id or "").strip()
    if owner_id == "":
        return AssetInfo.owner_id == ""
    return AssetInfo.owner_id.in_(["", owner_id])


def _apply_tag_filters(
    stmt: sa.sql.Select,
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
) -> sa.sql.Select:
    """include_tags: every tag must be present; exclude_tags: none may be present."""
    include_tags = normalize_tags(include_tags)
    exclude_tags = normalize_tags(exclude_tags)

    if include_tags:
        for tag_name in include_tags:
            stmt = stmt.where(
                exists().where(
                    (AssetInfoTag.asset_info_id == AssetInfo.id)
                    & (AssetInfoTag.tag_name == tag_name)
                )
            )

    if exclude_tags:
        stmt = stmt.where(
            ~exists().where(
                (AssetInfoTag.asset_info_id == AssetInfo.id)
                & (AssetInfoTag.tag_name.in_(exclude_tags))
            )
        )
    return stmt


def _apply_metadata_filter(
    stmt: sa.sql.Select,
    metadata_filter: dict | None = None,
) -> sa.sql.Select:
    """Apply filters using asset_info_meta projection table."""
    if not metadata_filter:
        return stmt

    def _exists_for_pred(key: str, *preds) -> sa.sql.ClauseElement:
        return sa.exists().where(
            AssetInfoMeta.asset_info_id == AssetInfo.id,
            AssetInfoMeta.key == key,
            *preds,
        )

    def _exists_clause_for_value(key: str, value) -> sa.sql.ClauseElement:
        if value is None:
            no_row_for_key = sa.not_(
                sa.exists().where(
                    AssetInfoMeta.asset_info_id == AssetInfo.id,
                    AssetInfoMeta.key == key,
                )
            )
            null_row = _exists_for_pred(
                key,
                AssetInfoMeta.val_json.is_(None),
                AssetInfoMeta.val_str.is_(None),
                AssetInfoMeta.val_num.is_(None),
                AssetInfoMeta.val_bool.is_(None),
            )
            return sa.or_(no_row_for_key, null_row)

        if isinstance(value, bool):
            return _exists_for_pred(key, AssetInfoMeta.val_bool == bool(value))
        if isinstance(value, (int, float)):
            num = value if isinstance(value, Decimal) else Decimal(str(value))
            return _exists_for_pred(key, AssetInfoMeta.val_num == num)
        if isinstance(value, str):
            return _exists_for_pred(key, AssetInfoMeta.val_str == value)
        return _exists_for_pred(key, AssetInfoMeta.val_json == value)

    for k, v in metadata_filter.items():
        if isinstance(v, list):
            ors = [_exists_clause_for_value(k, elem) for elem in v]
            if ors:
                stmt = stmt.where(sa.or_(*ors))
        else:
            stmt = stmt.where(_exists_clause_for_value(k, v))
    return stmt


def asset_info_exists_for_asset_id(
    session: Session,
    asset_id: str,
) -> bool:
    q = (
        select(sa.literal(True))
        .select_from(AssetInfo)
        .where(AssetInfo.asset_id == asset_id)
        .limit(1)
    )
    return (session.execute(q)).first() is not None


def get_asset_info_by_id(
    session: Session,
    asset_info_id: str,
) -> AssetInfo | None:
    return session.get(AssetInfo, asset_info_id)


def insert_asset_info(
    session: Session,
    asset_id: str,
    owner_id: str,
    name: str,
    preview_id: str | None = None,
) -> AssetInfo | None:
    """Insert a new AssetInfo. Returns None if unique constraint violated."""
    now = get_utc_now()
    try:
        with session.begin_nested():
            info = AssetInfo(
                owner_id=owner_id,
                name=name,
                asset_id=asset_id,
                preview_id=preview_id,
                created_at=now,
                updated_at=now,
                last_access_time=now,
            )
            session.add(info)
            session.flush()
            return info
    except IntegrityError:
        return None


def get_or_create_asset_info(
    session: Session,
    asset_id: str,
    owner_id: str,
    name: str,
    preview_id: str | None = None,
) -> tuple[AssetInfo, bool]:
    """Get existing or create new AssetInfo. Returns (info, created)."""
    info = insert_asset_info(
        session,
        asset_id=asset_id,
        owner_id=owner_id,
        name=name,
        preview_id=preview_id,
    )
    if info:
        return info, True

    existing = session.execute(
        select(AssetInfo)
        .where(
            AssetInfo.asset_id == asset_id,
            AssetInfo.name == name,
            AssetInfo.owner_id == owner_id,
        )
        .limit(1)
    ).unique().scalar_one_or_none()
    if not existing:
        raise RuntimeError("Failed to find AssetInfo after insert conflict.")
    return existing, False


def update_asset_info_timestamps(
    session: Session,
    asset_info: AssetInfo,
    preview_id: str | None = None,
) -> None:
    """Update timestamps and optionally preview_id on existing AssetInfo."""
    now = get_utc_now()
    if preview_id and asset_info.preview_id != preview_id:
        asset_info.preview_id = preview_id
    asset_info.updated_at = now
    if asset_info.last_access_time < now:
        asset_info.last_access_time = now
    session.flush()


def list_asset_infos_page(
    session: Session,
    owner_id: str = "",
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    name_contains: str | None = None,
    metadata_filter: dict | None = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "created_at",
    order: str = "desc",
) -> tuple[list[AssetInfo], dict[str, list[str]], int]:
    base = (
        select(AssetInfo)
        .join(Asset, Asset.id == AssetInfo.asset_id)
        .options(contains_eager(AssetInfo.asset), noload(AssetInfo.tags))
        .where(_build_visible_owner_clause(owner_id))
    )

    if name_contains:
        escaped, esc = escape_sql_like_string(name_contains)
        base = base.where(AssetInfo.name.ilike(f"%{escaped}%", escape=esc))

    base = _apply_tag_filters(base, include_tags, exclude_tags)
    base = _apply_metadata_filter(base, metadata_filter)

    sort = (sort or "created_at").lower()
    order = (order or "desc").lower()
    sort_map = {
        "name": AssetInfo.name,
        "created_at": AssetInfo.created_at,
        "updated_at": AssetInfo.updated_at,
        "last_access_time": AssetInfo.last_access_time,
        "size": Asset.size_bytes,
    }
    sort_col = sort_map.get(sort, AssetInfo.created_at)
    sort_exp = sort_col.desc() if order == "desc" else sort_col.asc()

    base = base.order_by(sort_exp).limit(limit).offset(offset)

    count_stmt = (
        select(sa.func.count())
        .select_from(AssetInfo)
        .join(Asset, Asset.id == AssetInfo.asset_id)
        .where(_build_visible_owner_clause(owner_id))
    )
    if name_contains:
        escaped, esc = escape_sql_like_string(name_contains)
        count_stmt = count_stmt.where(AssetInfo.name.ilike(f"%{escaped}%", escape=esc))
    count_stmt = _apply_tag_filters(count_stmt, include_tags, exclude_tags)
    count_stmt = _apply_metadata_filter(count_stmt, metadata_filter)

    total = int((session.execute(count_stmt)).scalar_one() or 0)

    infos = (session.execute(base)).unique().scalars().all()

    id_list: list[str] = [i.id for i in infos]
    tag_map: dict[str, list[str]] = defaultdict(list)
    if id_list:
        rows = session.execute(
            select(AssetInfoTag.asset_info_id, Tag.name)
            .join(Tag, Tag.name == AssetInfoTag.tag_name)
            .where(AssetInfoTag.asset_info_id.in_(id_list))
            .order_by(AssetInfoTag.added_at)
        )
        for aid, tag_name in rows.all():
            tag_map[aid].append(tag_name)

    return infos, tag_map, total


def fetch_asset_info_asset_and_tags(
    session: Session,
    asset_info_id: str,
    owner_id: str = "",
) -> tuple[AssetInfo, Asset, list[str]] | None:
    stmt = (
        select(AssetInfo, Asset, Tag.name)
        .join(Asset, Asset.id == AssetInfo.asset_id)
        .join(AssetInfoTag, AssetInfoTag.asset_info_id == AssetInfo.id, isouter=True)
        .join(Tag, Tag.name == AssetInfoTag.tag_name, isouter=True)
        .where(
            AssetInfo.id == asset_info_id,
            _build_visible_owner_clause(owner_id),
        )
        .options(noload(AssetInfo.tags))
        .order_by(Tag.name.asc())
    )

    rows = (session.execute(stmt)).all()
    if not rows:
        return None

    first_info, first_asset, _ = rows[0]
    tags: list[str] = []
    seen: set[str] = set()
    for _info, _asset, tag_name in rows:
        if tag_name and tag_name not in seen:
            seen.add(tag_name)
            tags.append(tag_name)
    return first_info, first_asset, tags


def fetch_asset_info_and_asset(
    session: Session,
    asset_info_id: str,
    owner_id: str = "",
) -> tuple[AssetInfo, Asset] | None:
    stmt = (
        select(AssetInfo, Asset)
        .join(Asset, Asset.id == AssetInfo.asset_id)
        .where(
            AssetInfo.id == asset_info_id,
            _build_visible_owner_clause(owner_id),
        )
        .limit(1)
        .options(noload(AssetInfo.tags))
    )
    row = session.execute(stmt)
    pair = row.first()
    if not pair:
        return None
    return pair[0], pair[1]


def update_asset_info_access_time(
    session: Session,
    asset_info_id: str,
    ts: datetime | None = None,
    only_if_newer: bool = True,
) -> None:
    ts = ts or get_utc_now()
    stmt = sa.update(AssetInfo).where(AssetInfo.id == asset_info_id)
    if only_if_newer:
        stmt = stmt.where(
            sa.or_(AssetInfo.last_access_time.is_(None), AssetInfo.last_access_time < ts)
        )
    session.execute(stmt.values(last_access_time=ts))


def update_asset_info_name(
    session: Session,
    asset_info_id: str,
    name: str,
) -> None:
    """Update the name of an AssetInfo."""
    now = get_utc_now()
    session.execute(
        sa.update(AssetInfo)
        .where(AssetInfo.id == asset_info_id)
        .values(name=name, updated_at=now)
    )


def update_asset_info_updated_at(
    session: Session,
    asset_info_id: str,
    ts: datetime | None = None,
) -> None:
    """Update the updated_at timestamp of an AssetInfo."""
    ts = ts or get_utc_now()
    session.execute(
        sa.update(AssetInfo)
        .where(AssetInfo.id == asset_info_id)
        .values(updated_at=ts)
    )


def set_asset_info_metadata(
    session: Session,
    asset_info_id: str,
    user_metadata: dict | None = None,
) -> None:
    info = session.get(AssetInfo, asset_info_id)
    if not info:
        raise ValueError(f"AssetInfo {asset_info_id} not found")

    info.user_metadata = user_metadata or {}
    info.updated_at = get_utc_now()
    session.flush()

    session.execute(delete(AssetInfoMeta).where(AssetInfoMeta.asset_info_id == asset_info_id))
    session.flush()

    if not user_metadata:
        return

    rows: list[AssetInfoMeta] = []
    for k, v in user_metadata.items():
        for r in convert_metadata_to_rows(k, v):
            rows.append(
                AssetInfoMeta(
                    asset_info_id=asset_info_id,
                    key=r["key"],
                    ordinal=int(r["ordinal"]),
                    val_str=r.get("val_str"),
                    val_num=r.get("val_num"),
                    val_bool=r.get("val_bool"),
                    val_json=r.get("val_json"),
                )
            )
    if rows:
        session.add_all(rows)
        session.flush()


def delete_asset_info_by_id(
    session: Session,
    asset_info_id: str,
    owner_id: str,
) -> bool:
    stmt = sa.delete(AssetInfo).where(
        AssetInfo.id == asset_info_id,
        _build_visible_owner_clause(owner_id),
    )
    return int((session.execute(stmt)).rowcount or 0) > 0


def set_asset_info_preview(
    session: Session,
    asset_info_id: str,
    preview_asset_id: str | None = None,
) -> None:
    """Set or clear preview_id and bump updated_at. Raises on unknown IDs."""
    info = session.get(AssetInfo, asset_info_id)
    if not info:
        raise ValueError(f"AssetInfo {asset_info_id} not found")

    if preview_asset_id is None:
        info.preview_id = None
    else:
        if not session.get(Asset, preview_asset_id):
            raise ValueError(f"Preview Asset {preview_asset_id} not found")
        info.preview_id = preview_asset_id

    info.updated_at = get_utc_now()
    session.flush()


def bulk_insert_asset_infos_ignore_conflicts(
    session: Session,
    rows: list[dict],
) -> None:
    """Bulk insert AssetInfo rows with ON CONFLICT DO NOTHING.

    Each dict should have: id, owner_id, name, asset_id, preview_id,
    user_metadata, created_at, updated_at, last_access_time
    """
    if not rows:
        return
    ins = sqlite.insert(AssetInfo).on_conflict_do_nothing(
        index_elements=[AssetInfo.asset_id, AssetInfo.owner_id, AssetInfo.name]
    )
    for chunk in _iter_chunks(rows, _calculate_rows_per_statement(9)):
        session.execute(ins, chunk)


def get_asset_info_ids_by_ids(
    session: Session,
    info_ids: list[str],
) -> set[str]:
    """Query to find which AssetInfo IDs exist in the database."""
    if not info_ids:
        return set()

    found: set[str] = set()
    for chunk in _iter_chunks(info_ids, MAX_BIND_PARAMS):
        result = session.execute(
            select(AssetInfo.id).where(AssetInfo.id.in_(chunk))
        )
        found.update(result.scalars().all())
    return found
