"""
Ingest services - handles ingesting files into the asset database.

Business logic for:
- ingest_file_from_path: Ingest a file from filesystem path (upsert asset, cache state, info)
- register_existing_asset: Create AssetInfo for an asset that already exists by hash
"""
import logging
import os
from typing import Sequence

from sqlalchemy import select

from app.assets.database.models import Asset, Tag
from app.database.db import create_session
from app.assets.helpers import (
    compute_relative_filename,
    normalize_tags,
    pick_best_live_path,
    utcnow,
)
from app.assets.database.queries import (
    get_asset_by_hash,
    get_or_create_asset_info,
    list_cache_states_by_asset_id,
    remove_missing_tag_for_asset_id,
    replace_asset_info_metadata_projection,
    set_asset_info_tags,
    update_asset_info_timestamps,
    upsert_asset,
    upsert_cache_state,
    add_tags_to_asset_info,
    ensure_tags_exist,
    get_asset_tags,
)


def ingest_file_from_path(
    abs_path: str,
    asset_hash: str,
    size_bytes: int,
    mtime_ns: int,
    mime_type: str | None = None,
    info_name: str | None = None,
    owner_id: str = "",
    preview_id: str | None = None,
    user_metadata: dict | None = None,
    tags: Sequence[str] = (),
    tag_origin: str = "manual",
    require_existing_tags: bool = False,
) -> dict:
    """
    Idempotently upsert:
      - Asset by content hash (create if missing)
      - AssetCacheState(file_path) pointing to asset_id
      - Optionally AssetInfo + tag links and metadata projection
    Returns flags and ids.
    """
    locator = os.path.abspath(abs_path)

    out: dict = {
        "asset_created": False,
        "asset_updated": False,
        "state_created": False,
        "state_updated": False,
        "asset_info_id": None,
    }

    with create_session() as session:
        # Validate preview_id if provided
        if preview_id:
            if not session.get(Asset, preview_id):
                preview_id = None

        # 1. Upsert Asset
        asset, created, updated = upsert_asset(
            session,
            asset_hash=asset_hash,
            size_bytes=size_bytes,
            mime_type=mime_type,
        )
        out["asset_created"] = created
        out["asset_updated"] = updated

        # 2. Upsert CacheState
        state_created, state_updated = upsert_cache_state(
            session,
            asset_id=asset.id,
            file_path=locator,
            mtime_ns=mtime_ns,
        )
        out["state_created"] = state_created
        out["state_updated"] = state_updated

        # 3. Optionally create/update AssetInfo
        if info_name:
            info, info_created = get_or_create_asset_info(
                session,
                asset_id=asset.id,
                owner_id=owner_id,
                name=info_name,
                preview_id=preview_id,
            )
            if info_created:
                out["asset_info_id"] = info.id
            else:
                update_asset_info_timestamps(session, asset_info=info, preview_id=preview_id)
                out["asset_info_id"] = info.id

            # 4. Handle tags
            norm = normalize_tags(list(tags))
            if norm and out["asset_info_id"]:
                if require_existing_tags:
                    _validate_tags_exist(session, norm)
                add_tags_to_asset_info(
                    session,
                    asset_info_id=out["asset_info_id"],
                    tags=norm,
                    origin=tag_origin,
                    create_if_missing=not require_existing_tags,
                )

            # 5. Update metadata with computed filename
            if out["asset_info_id"]:
                _update_metadata_with_filename(
                    session,
                    asset_info_id=out["asset_info_id"],
                    asset_id=asset.id,
                    info=info,
                    user_metadata=user_metadata,
                )

        # 6. Remove missing tag
        try:
            remove_missing_tag_for_asset_id(session, asset_id=asset.id)
        except Exception:
            logging.exception("Failed to clear 'missing' tag for asset %s", asset.id)

        session.commit()

    return out


def register_existing_asset(
    asset_hash: str,
    name: str,
    user_metadata: dict | None = None,
    tags: list[str] | None = None,
    tag_origin: str = "manual",
    owner_id: str = "",
) -> dict:
    """
    Create or return existing AssetInfo for an asset that already exists by hash.
    
    Returns dict with asset and info details, or raises ValueError if hash not found.
    """
    with create_session() as session:
        asset = get_asset_by_hash(session, asset_hash=asset_hash)
        if not asset:
            raise ValueError(f"No asset with hash {asset_hash}")

        info, info_created = get_or_create_asset_info(
            session,
            asset_id=asset.id,
            owner_id=owner_id,
            name=name,
            preview_id=None,
        )

        if not info_created:
            # Return existing info
            tag_names = get_asset_tags(session, asset_info_id=info.id)
            session.commit()
            return {
                "info": info,
                "asset": asset,
                "tags": tag_names,
                "created": False,
            }

        # New info - apply metadata and tags
        new_meta = dict(user_metadata or {})
        computed_filename = _compute_filename_for_asset(session, asset.id)
        if computed_filename:
            new_meta["filename"] = computed_filename

        if new_meta:
            replace_asset_info_metadata_projection(
                session,
                asset_info_id=info.id,
                user_metadata=new_meta,
            )

        if tags is not None:
            set_asset_info_tags(
                session,
                asset_info_id=info.id,
                tags=tags,
                origin=tag_origin,
            )

        tag_names = get_asset_tags(session, asset_info_id=info.id)
        session.commit()

        return {
            "info": info,
            "asset": asset,
            "tags": tag_names,
            "created": True,
        }


def _validate_tags_exist(session, tags: list[str]) -> None:
    """Raise ValueError if any tags don't exist."""
    existing_tag_names = set(
        name for (name,) in session.execute(select(Tag.name).where(Tag.name.in_(tags))).all()
    )
    missing = [t for t in tags if t not in existing_tag_names]
    if missing:
        raise ValueError(f"Unknown tags: {missing}")


def _compute_filename_for_asset(session, asset_id: str) -> str | None:
    """Compute the relative filename for an asset from its cache states."""
    primary_path = pick_best_live_path(list_cache_states_by_asset_id(session, asset_id=asset_id))
    return compute_relative_filename(primary_path) if primary_path else None


def _update_metadata_with_filename(
    session,
    asset_info_id: str,
    asset_id: str,
    info,
    user_metadata: dict | None,
) -> None:
    """Update metadata projection with computed filename."""
    computed_filename = _compute_filename_for_asset(session, asset_id)

    current_meta = info.user_metadata or {}
    new_meta = dict(current_meta)
    if user_metadata:
        for k, v in user_metadata.items():
            new_meta[k] = v
    if computed_filename:
        new_meta["filename"] = computed_filename

    if new_meta != current_meta:
        replace_asset_info_metadata_projection(
            session,
            asset_info_id=asset_info_id,
            user_metadata=new_meta,
        )
