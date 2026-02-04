from app.assets.database.queries import (
    add_tags_to_asset_info,
    get_asset_info_by_id,
    list_tags_with_usage,
    remove_tags_from_asset_info,
)
from app.database.db import create_session


def apply_tags(
    asset_info_id: str,
    tags: list[str],
    origin: str = "manual",
    owner_id: str = "",
) -> dict:
    """
    Add tags to an asset.
    Returns dict with added, already_present, and total_tags lists.
    """
    with create_session() as session:
        info_row = get_asset_info_by_id(session, asset_info_id=asset_info_id)
        if not info_row:
            raise ValueError(f"AssetInfo {asset_info_id} not found")
        if info_row.owner_id and info_row.owner_id != owner_id:
            raise PermissionError("not owner")

        data = add_tags_to_asset_info(
            session,
            asset_info_id=asset_info_id,
            tags=tags,
            origin=origin,
            create_if_missing=True,
            asset_info_row=info_row,
        )
        session.commit()

    return data


def remove_tags(
    asset_info_id: str,
    tags: list[str],
    owner_id: str = "",
) -> dict:
    """
    Remove tags from an asset.
    Returns dict with removed, not_present, and total_tags lists.
    """
    with create_session() as session:
        info_row = get_asset_info_by_id(session, asset_info_id=asset_info_id)
        if not info_row:
            raise ValueError(f"AssetInfo {asset_info_id} not found")
        if info_row.owner_id and info_row.owner_id != owner_id:
            raise PermissionError("not owner")

        data = remove_tags_from_asset_info(
            session,
            asset_info_id=asset_info_id,
            tags=tags,
        )
        session.commit()

    return data


def list_tags(
    prefix: str | None = None,
    limit: int = 100,
    offset: int = 0,
    order: str = "count_desc",
    include_zero: bool = True,
    owner_id: str = "",
) -> tuple[list[tuple[str, str, int]], int]:
    """
    List tags with usage counts.
    Returns (rows, total) where rows are (name, tag_type, count) tuples.
    """
    limit = max(1, min(1000, limit))
    offset = max(0, offset)

    with create_session() as session:
        rows, total = list_tags_with_usage(
            session,
            prefix=prefix,
            limit=limit,
            offset=offset,
            include_zero=include_zero,
            order=order,
            owner_id=owner_id,
        )

    return rows, total
