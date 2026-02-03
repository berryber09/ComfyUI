"""
Asset management services - CRUD operations on assets.

Business logic for:
- get_asset_detail: Fetch full asset details with tags
- update_asset_metadata: Update name, tags, and/or metadata
- delete_asset_reference: Delete AssetInfo and optionally orphaned content
- set_asset_preview: Set or clear preview on an asset
"""
import contextlib
import os
from typing import Sequence

from app.assets.database.models import Asset
from app.database.db import create_session
from app.assets.helpers import pick_best_live_path, get_utc_now
from app.assets.services.path_utils import compute_relative_filename
from app.assets.database.queries import (
    asset_info_exists_for_asset_id,
    delete_asset_info_by_id,
    fetch_asset_info_asset_and_tags,
    get_asset_info_by_id,
    list_cache_states_by_asset_id,
    replace_asset_info_metadata_projection,
    set_asset_info_preview,
    set_asset_info_tags,
)


def get_asset_detail(
    asset_info_id: str,
    owner_id: str = "",
) -> dict | None:
    """
    Fetch full asset details including tags.
    Returns dict with info, asset, and tags, or None if not found.
    """
    with create_session() as session:
        result = fetch_asset_info_asset_and_tags(
            session,
            asset_info_id=asset_info_id,
            owner_id=owner_id,
        )
        if not result:
            return None

        info, asset, tags = result
        return {
            "info": info,
            "asset": asset,
            "tags": tags,
        }


def update_asset_metadata(
    asset_info_id: str,
    name: str | None = None,
    tags: Sequence[str] | None = None,
    user_metadata: dict | None = None,
    tag_origin: str = "manual",
    owner_id: str = "",
) -> dict:
    """
    Update name, tags, and/or metadata on an AssetInfo.
    Returns updated info dict with tags.
    """
    with create_session() as session:
        info = get_asset_info_by_id(session, asset_info_id=asset_info_id)
        if not info:
            raise ValueError(f"AssetInfo {asset_info_id} not found")
        if info.owner_id and info.owner_id != owner_id:
            raise PermissionError("not owner")

        touched = False
        if name is not None and name != info.name:
            info.name = name
            touched = True

        # Compute filename from best live path
        computed_filename = _compute_filename_for_asset(session, info.asset_id)

        if user_metadata is not None:
            new_meta = dict(user_metadata)
            if computed_filename:
                new_meta["filename"] = computed_filename
            replace_asset_info_metadata_projection(
                session, asset_info_id=asset_info_id, user_metadata=new_meta
            )
            touched = True
        else:
            if computed_filename:
                current_meta = info.user_metadata or {}
                if current_meta.get("filename") != computed_filename:
                    new_meta = dict(current_meta)
                    new_meta["filename"] = computed_filename
                    replace_asset_info_metadata_projection(
                        session, asset_info_id=asset_info_id, user_metadata=new_meta
                    )
                    touched = True

        if tags is not None:
            set_asset_info_tags(
                session,
                asset_info_id=asset_info_id,
                tags=tags,
                origin=tag_origin,
            )
            touched = True

        if touched and user_metadata is None:
            info.updated_at = get_utc_now()
            session.flush()

        # Fetch updated info with tags
        result = fetch_asset_info_asset_and_tags(
            session,
            asset_info_id=asset_info_id,
            owner_id=owner_id,
        )
        session.commit()

        if not result:
            raise RuntimeError("State changed during update")

        info, asset, tag_list = result
        return {
            "info": info,
            "asset": asset,
            "tags": tag_list,
        }


def delete_asset_reference(
    asset_info_id: str,
    owner_id: str,
    delete_content_if_orphan: bool = True,
) -> bool:
    """
    Delete an AssetInfo reference.
    If delete_content_if_orphan is True and no other AssetInfos reference the asset,
    also delete the Asset and its cached files.
    """
    with create_session() as session:
        info_row = get_asset_info_by_id(session, asset_info_id=asset_info_id)
        asset_id = info_row.asset_id if info_row else None

        deleted = delete_asset_info_by_id(session, asset_info_id=asset_info_id, owner_id=owner_id)
        if not deleted:
            session.commit()
            return False

        if not delete_content_if_orphan or not asset_id:
            session.commit()
            return True

        still_exists = asset_info_exists_for_asset_id(session, asset_id=asset_id)
        if still_exists:
            session.commit()
            return True

        # Orphaned asset - delete it and its files
        states = list_cache_states_by_asset_id(session, asset_id=asset_id)
        file_paths = [s.file_path for s in (states or []) if getattr(s, "file_path", None)]

        asset_row = session.get(Asset, asset_id)
        if asset_row is not None:
            session.delete(asset_row)

        session.commit()

        # Delete files after commit
        for p in file_paths:
            with contextlib.suppress(Exception):
                if p and os.path.isfile(p):
                    os.remove(p)

    return True


def set_asset_preview(
    asset_info_id: str,
    preview_asset_id: str | None = None,
    owner_id: str = "",
) -> dict:
    """
    Set or clear preview_id on an AssetInfo.
    Returns updated asset detail dict.
    """
    with create_session() as session:
        info_row = get_asset_info_by_id(session, asset_info_id=asset_info_id)
        if not info_row:
            raise ValueError(f"AssetInfo {asset_info_id} not found")
        if info_row.owner_id and info_row.owner_id != owner_id:
            raise PermissionError("not owner")

        set_asset_info_preview(
            session,
            asset_info_id=asset_info_id,
            preview_asset_id=preview_asset_id,
        )

        result = fetch_asset_info_asset_and_tags(
            session, asset_info_id=asset_info_id, owner_id=owner_id
        )
        if not result:
            raise RuntimeError("State changed during preview update")

        info, asset, tags = result
        session.commit()

        return {
            "info": info,
            "asset": asset,
            "tags": tags,
        }


def _compute_filename_for_asset(session, asset_id: str) -> str | None:
    """Compute the relative filename for an asset from its cache states."""
    primary_path = pick_best_live_path(list_cache_states_by_asset_id(session, asset_id=asset_id))
    return compute_relative_filename(primary_path) if primary_path else None
