import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.dialects import sqlite
from sqlalchemy.orm import Session

from app.assets.database.models import Asset
from app.assets.database.queries.common import calculate_rows_per_statement, iter_chunks


def asset_exists_by_hash(
    session: Session,
    asset_hash: str,
) -> bool:
    """
    Check if an asset with a given hash exists in database.
    """
    row = (
        session.execute(
            select(sa.literal(True))
            .select_from(Asset)
            .where(Asset.hash == asset_hash)
            .limit(1)
        )
    ).first()
    return row is not None


def get_asset_by_hash(
    session: Session,
    asset_hash: str,
) -> Asset | None:
    return (
        (session.execute(select(Asset).where(Asset.hash == asset_hash).limit(1)))
        .scalars()
        .first()
    )


def upsert_asset(
    session: Session,
    asset_hash: str,
    size_bytes: int,
    mime_type: str | None = None,
) -> tuple[Asset, bool, bool]:
    """Upsert an Asset by hash. Returns (asset, created, updated)."""
    vals = {"hash": asset_hash, "size_bytes": int(size_bytes)}
    if mime_type:
        vals["mime_type"] = mime_type

    ins = (
        sqlite.insert(Asset)
        .values(**vals)
        .on_conflict_do_nothing(index_elements=[Asset.hash])
    )
    res = session.execute(ins)
    created = int(res.rowcount or 0) > 0

    asset = (
        session.execute(select(Asset).where(Asset.hash == asset_hash).limit(1))
        .scalars()
        .first()
    )
    if not asset:
        raise RuntimeError("Asset row not found after upsert.")

    updated = False
    if not created:
        changed = False
        if asset.size_bytes != int(size_bytes) and int(size_bytes) > 0:
            asset.size_bytes = int(size_bytes)
            changed = True
        if mime_type and asset.mime_type != mime_type:
            asset.mime_type = mime_type
            changed = True
        if changed:
            updated = True

    return asset, created, updated


def bulk_insert_assets(
    session: Session,
    rows: list[dict],
) -> None:
    """Bulk insert Asset rows. Each dict should have: id, hash, size_bytes, mime_type, created_at."""
    if not rows:
        return
    ins = sqlite.insert(Asset).on_conflict_do_nothing(index_elements=[Asset.hash])
    for chunk in iter_chunks(rows, calculate_rows_per_statement(5)):
        session.execute(ins, chunk)


def get_existing_asset_ids(
    session: Session,
    asset_ids: list[str],
) -> set[str]:
    """Return the subset of asset_ids that exist in the database."""
    if not asset_ids:
        return set()
    rows = session.execute(
        select(Asset.id).where(Asset.id.in_(asset_ids))
    ).fetchall()
    return {row[0] for row in rows}
