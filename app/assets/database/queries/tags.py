from typing import Iterable, Sequence, TypedDict

import sqlalchemy as sa
from sqlalchemy import delete, func, select
from sqlalchemy.dialects import sqlite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.assets.database.models import (
    AssetReference,
    AssetReferenceMeta,
    AssetReferenceTag,
    Tag,
)
from app.assets.database.queries.common import (
    build_visible_owner_clause,
    iter_row_chunks,
)
from app.assets.helpers import escape_sql_like_string, get_utc_now, normalize_tags


class AddTagsDict(TypedDict):
    added: list[str]
    already_present: list[str]
    total_tags: list[str]


class RemoveTagsDict(TypedDict):
    removed: list[str]
    not_present: list[str]
    total_tags: list[str]


class SetTagsDict(TypedDict):
    added: list[str]
    removed: list[str]
    total: list[str]


def ensure_tags_exist(
    session: Session, names: Iterable[str], tag_type: str = "user"
) -> None:
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


def get_reference_tags(session: Session, reference_id: str) -> list[str]:
    return [
        tag_name
        for (tag_name,) in (
            session.execute(
                select(AssetReferenceTag.tag_name).where(
                    AssetReferenceTag.asset_reference_id == reference_id
                )
            )
        ).all()
    ]


def set_reference_tags(
    session: Session,
    reference_id: str,
    tags: Sequence[str],
    origin: str = "manual",
) -> SetTagsDict:
    desired = normalize_tags(tags)

    current = set(
        tag_name
        for (tag_name,) in (
            session.execute(
                select(AssetReferenceTag.tag_name).where(
                    AssetReferenceTag.asset_reference_id == reference_id
                )
            )
        ).all()
    )

    to_add = [t for t in desired if t not in current]
    to_remove = [t for t in current if t not in desired]

    if to_add:
        ensure_tags_exist(session, to_add, tag_type="user")
        session.add_all(
            [
                AssetReferenceTag(
                    asset_reference_id=reference_id,
                    tag_name=t,
                    origin=origin,
                    added_at=get_utc_now(),
                )
                for t in to_add
            ]
        )
        session.flush()

    if to_remove:
        session.execute(
            delete(AssetReferenceTag).where(
                AssetReferenceTag.asset_reference_id == reference_id,
                AssetReferenceTag.tag_name.in_(to_remove),
            )
        )
        session.flush()

    return {"added": to_add, "removed": to_remove, "total": desired}


def add_tags_to_reference(
    session: Session,
    reference_id: str,
    tags: Sequence[str],
    origin: str = "manual",
    create_if_missing: bool = True,
    reference_row: AssetReference | None = None,
) -> AddTagsDict:
    if not reference_row:
        ref = session.get(AssetReference, reference_id)
        if not ref:
            raise ValueError(f"AssetReference {reference_id} not found")

    norm = normalize_tags(tags)
    if not norm:
        total = get_reference_tags(session, reference_id=reference_id)
        return {"added": [], "already_present": [], "total_tags": total}

    if create_if_missing:
        ensure_tags_exist(session, norm, tag_type="user")

    current = {
        tag_name
        for (tag_name,) in (
            session.execute(
                sa.select(AssetReferenceTag.tag_name).where(
                    AssetReferenceTag.asset_reference_id == reference_id
                )
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
                        AssetReferenceTag(
                            asset_reference_id=reference_id,
                            tag_name=t,
                            origin=origin,
                            added_at=get_utc_now(),
                        )
                        for t in to_add
                    ]
                )
                session.flush()
            except IntegrityError:
                nested.rollback()

    after = set(get_reference_tags(session, reference_id=reference_id))
    return {
        "added": sorted(((after - current) & want)),
        "already_present": sorted(want & current),
        "total_tags": sorted(after),
    }


def remove_tags_from_reference(
    session: Session,
    reference_id: str,
    tags: Sequence[str],
) -> RemoveTagsDict:
    ref = session.get(AssetReference, reference_id)
    if not ref:
        raise ValueError(f"AssetReference {reference_id} not found")

    norm = normalize_tags(tags)
    if not norm:
        total = get_reference_tags(session, reference_id=reference_id)
        return {"removed": [], "not_present": [], "total_tags": total}

    existing = {
        tag_name
        for (tag_name,) in (
            session.execute(
                sa.select(AssetReferenceTag.tag_name).where(
                    AssetReferenceTag.asset_reference_id == reference_id
                )
            )
        ).all()
    }

    to_remove = sorted(set(t for t in norm if t in existing))
    not_present = sorted(set(t for t in norm if t not in existing))

    if to_remove:
        session.execute(
            delete(AssetReferenceTag).where(
                AssetReferenceTag.asset_reference_id == reference_id,
                AssetReferenceTag.tag_name.in_(to_remove),
            )
        )
        session.flush()

    total = get_reference_tags(session, reference_id=reference_id)
    return {"removed": to_remove, "not_present": not_present, "total_tags": total}


def add_missing_tag_for_asset_id(
    session: Session,
    asset_id: str,
    origin: str = "automatic",
) -> None:
    select_rows = (
        sa.select(
            AssetReference.id.label("asset_reference_id"),
            sa.literal("missing").label("tag_name"),
            sa.literal(origin).label("origin"),
            sa.literal(get_utc_now()).label("added_at"),
        )
        .where(AssetReference.asset_id == asset_id)
        .where(
            sa.not_(
                sa.exists().where(
                    (AssetReferenceTag.asset_reference_id == AssetReference.id)
                    & (AssetReferenceTag.tag_name == "missing")
                )
            )
        )
    )
    session.execute(
        sqlite.insert(AssetReferenceTag)
        .from_select(
            ["asset_reference_id", "tag_name", "origin", "added_at"],
            select_rows,
        )
        .on_conflict_do_nothing(
            index_elements=[
                AssetReferenceTag.asset_reference_id,
                AssetReferenceTag.tag_name,
            ]
        )
    )


def remove_missing_tag_for_asset_id(
    session: Session,
    asset_id: str,
) -> None:
    session.execute(
        sa.delete(AssetReferenceTag).where(
            AssetReferenceTag.asset_reference_id.in_(
                sa.select(AssetReference.id).where(AssetReference.asset_id == asset_id)
            ),
            AssetReferenceTag.tag_name == "missing",
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
            AssetReferenceTag.tag_name.label("tag_name"),
            func.count(AssetReferenceTag.asset_reference_id).label("cnt"),
        )
        .select_from(AssetReferenceTag)
        .join(AssetReference, AssetReference.id == AssetReferenceTag.asset_reference_id)
        .where(build_visible_owner_clause(owner_id))
        .group_by(AssetReferenceTag.tag_name)
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
        escaped, esc = escape_sql_like_string(prefix.strip().lower())
        q = q.where(Tag.name.like(escaped + "%", escape=esc))

    if not include_zero:
        q = q.where(func.coalesce(counts_sq.c.cnt, 0) > 0)

    if order == "name_asc":
        q = q.order_by(Tag.name.asc())
    else:
        q = q.order_by(func.coalesce(counts_sq.c.cnt, 0).desc(), Tag.name.asc())

    total_q = select(func.count()).select_from(Tag)
    if prefix:
        escaped, esc = escape_sql_like_string(prefix.strip().lower())
        total_q = total_q.where(Tag.name.like(escaped + "%", escape=esc))
    if not include_zero:
        total_q = total_q.where(
            Tag.name.in_(
                select(AssetReferenceTag.tag_name).group_by(AssetReferenceTag.tag_name)
            )
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
    """Batch insert into asset_reference_tags and asset_reference_meta.

    Uses ON CONFLICT DO NOTHING.

    Args:
        session: Database session
        tag_rows: Dicts with: asset_reference_id, tag_name, origin, added_at
        meta_rows: Dicts with: asset_reference_id, key, ordinal, val_*
    """
    if tag_rows:
        ins_tags = sqlite.insert(AssetReferenceTag).on_conflict_do_nothing(
            index_elements=[
                AssetReferenceTag.asset_reference_id,
                AssetReferenceTag.tag_name,
            ]
        )
        for chunk in iter_row_chunks(tag_rows, cols_per_row=4):
            session.execute(ins_tags, chunk)

    if meta_rows:
        ins_meta = sqlite.insert(AssetReferenceMeta).on_conflict_do_nothing(
            index_elements=[
                AssetReferenceMeta.asset_reference_id,
                AssetReferenceMeta.key,
                AssetReferenceMeta.ordinal,
            ]
        )
        for chunk in iter_row_chunks(meta_rows, cols_per_row=7):
            session.execute(ins_meta, chunk)
