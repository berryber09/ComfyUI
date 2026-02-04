import contextlib
import logging
import mimetypes
import os
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

import app.assets.services.hashing as hashing
from app.assets.database.models import Asset, AssetInfo, Tag
from app.assets.database.queries import (
    add_tags_to_asset_info,
    fetch_asset_info_and_asset,
    get_asset_by_hash,
    get_asset_tags,
    get_or_create_asset_info,
    remove_missing_tag_for_asset_id,
    set_asset_info_metadata,
    set_asset_info_tags,
    update_asset_info_timestamps,
    upsert_asset,
    upsert_cache_state,
)
from app.assets.helpers import normalize_tags
from app.assets.services.file_utils import get_size_and_mtime_ns
from app.assets.services.path_utils import (
    compute_filename_for_asset,
    resolve_destination_from_tags,
    validate_path_within_base,
)
from app.assets.services.schemas import (
    IngestResult,
    RegisterAssetResult,
    UploadResult,
    UserMetadata,
    extract_asset_data,
    extract_info_data,
)
from app.database.db import create_session


def ingest_file_from_path(
    abs_path: str,
    asset_hash: str,
    size_bytes: int,
    mtime_ns: int,
    mime_type: str | None = None,
    info_name: str | None = None,
    owner_id: str = "",
    preview_id: str | None = None,
    user_metadata: UserMetadata = None,
    tags: Sequence[str] = (),
    tag_origin: str = "manual",
    require_existing_tags: bool = False,
) -> IngestResult:
    locator = os.path.abspath(abs_path)

    asset_created = False
    asset_updated = False
    state_created = False
    state_updated = False
    asset_info_id: str | None = None

    with create_session() as session:
        if preview_id:
            if not session.get(Asset, preview_id):
                preview_id = None

        asset, asset_created, asset_updated = upsert_asset(
            session,
            asset_hash=asset_hash,
            size_bytes=size_bytes,
            mime_type=mime_type,
        )

        state_created, state_updated = upsert_cache_state(
            session,
            asset_id=asset.id,
            file_path=locator,
            mtime_ns=mtime_ns,
        )

        if info_name:
            info, info_created = get_or_create_asset_info(
                session,
                asset_id=asset.id,
                owner_id=owner_id,
                name=info_name,
                preview_id=preview_id,
            )
            if info_created:
                asset_info_id = info.id
            else:
                update_asset_info_timestamps(
                    session, asset_info=info, preview_id=preview_id
                )
                asset_info_id = info.id

            norm = normalize_tags(list(tags))
            if norm and asset_info_id:
                if require_existing_tags:
                    _validate_tags_exist(session, norm)
                add_tags_to_asset_info(
                    session,
                    asset_info_id=asset_info_id,
                    tags=norm,
                    origin=tag_origin,
                    create_if_missing=not require_existing_tags,
                )

            if asset_info_id:
                _update_metadata_with_filename(
                    session,
                    asset_info_id=asset_info_id,
                    asset_id=asset.id,
                    info=info,
                    user_metadata=user_metadata,
                )

        try:
            remove_missing_tag_for_asset_id(session, asset_id=asset.id)
        except Exception:
            logging.exception("Failed to clear 'missing' tag for asset %s", asset.id)

        session.commit()

    return IngestResult(
        asset_created=asset_created,
        asset_updated=asset_updated,
        state_created=state_created,
        state_updated=state_updated,
        asset_info_id=asset_info_id,
    )


def register_existing_asset(
    asset_hash: str,
    name: str,
    user_metadata: UserMetadata = None,
    tags: list[str] | None = None,
    tag_origin: str = "manual",
    owner_id: str = "",
) -> RegisterAssetResult:
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
            tag_names = get_asset_tags(session, asset_info_id=info.id)
            result = RegisterAssetResult(
                info=extract_info_data(info),
                asset=extract_asset_data(asset),
                tags=tag_names,
                created=False,
            )
            session.commit()
            return result

        new_meta = dict(user_metadata or {})
        computed_filename = compute_filename_for_asset(session, asset.id)
        if computed_filename:
            new_meta["filename"] = computed_filename

        if new_meta:
            set_asset_info_metadata(
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
        session.refresh(info)
        result = RegisterAssetResult(
            info=extract_info_data(info),
            asset=extract_asset_data(asset),
            tags=tag_names,
            created=True,
        )
        session.commit()

        return result


def _validate_tags_exist(session: Session, tags: list[str]) -> None:
    existing_tag_names = set(
        name
        for (name,) in session.execute(select(Tag.name).where(Tag.name.in_(tags))).all()
    )
    missing = [t for t in tags if t not in existing_tag_names]
    if missing:
        raise ValueError(f"Unknown tags: {missing}")


def _update_metadata_with_filename(
    session: Session,
    asset_info_id: str,
    asset_id: str,
    info: AssetInfo,
    user_metadata: UserMetadata,
) -> None:
    computed_filename = compute_filename_for_asset(session, asset_id)

    current_meta = info.user_metadata or {}
    new_meta = dict(current_meta)
    if user_metadata:
        for k, v in user_metadata.items():
            new_meta[k] = v
    if computed_filename:
        new_meta["filename"] = computed_filename

    if new_meta != current_meta:
        set_asset_info_metadata(
            session,
            asset_info_id=asset_info_id,
            user_metadata=new_meta,
        )


def _sanitize_filename(name: str | None, fallback: str) -> str:
    n = os.path.basename((name or "").strip() or fallback)
    return n if n else fallback


class HashMismatchError(Exception):
    pass


class DependencyMissingError(Exception):
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


def upload_from_temp_path(
    temp_path: str,
    name: str | None = None,
    tags: list[str] | None = None,
    user_metadata: dict | None = None,
    client_filename: str | None = None,
    owner_id: str = "",
    expected_hash: str | None = None,
) -> UploadResult:
    try:
        digest = hashing.compute_blake3_hash(temp_path)
    except ImportError as e:
        raise DependencyMissingError(str(e))
    except Exception as e:
        raise RuntimeError(f"failed to hash uploaded file: {e}")
    asset_hash = "blake3:" + digest

    if expected_hash and asset_hash != expected_hash.strip().lower():
        raise HashMismatchError("Uploaded file hash does not match provided hash.")

    with create_session() as session:
        existing = get_asset_by_hash(session, asset_hash=asset_hash)

    if existing is not None:
        with contextlib.suppress(Exception):
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        display_name = _sanitize_filename(name or client_filename, fallback=digest)
        result = register_existing_asset(
            asset_hash=asset_hash,
            name=display_name,
            user_metadata=user_metadata or {},
            tags=tags or [],
            tag_origin="manual",
            owner_id=owner_id,
        )
        return UploadResult(
            info=result.info,
            asset=result.asset,
            tags=result.tags,
            created_new=False,
        )

    base_dir, subdirs = resolve_destination_from_tags(tags)
    dest_dir = os.path.join(base_dir, *subdirs) if subdirs else base_dir
    os.makedirs(dest_dir, exist_ok=True)

    src_for_ext = (client_filename or name or "").strip()
    _ext = os.path.splitext(os.path.basename(src_for_ext))[1] if src_for_ext else ""
    ext = _ext if 0 < len(_ext) <= 16 else ""
    hashed_basename = f"{digest}{ext}"
    dest_abs = os.path.abspath(os.path.join(dest_dir, hashed_basename))
    validate_path_within_base(dest_abs, base_dir)

    content_type = (
        mimetypes.guess_type(os.path.basename(src_for_ext), strict=False)[0]
        or mimetypes.guess_type(hashed_basename, strict=False)[0]
        or "application/octet-stream"
    )

    try:
        os.replace(temp_path, dest_abs)
    except Exception as e:
        raise RuntimeError(f"failed to move uploaded file into place: {e}")

    try:
        size_bytes, mtime_ns = get_size_and_mtime_ns(dest_abs)
    except OSError as e:
        raise RuntimeError(f"failed to stat destination file: {e}")

    ingest_result = ingest_file_from_path(
        asset_hash=asset_hash,
        abs_path=dest_abs,
        size_bytes=size_bytes,
        mtime_ns=mtime_ns,
        mime_type=content_type,
        info_name=_sanitize_filename(name or client_filename, fallback=digest),
        owner_id=owner_id,
        preview_id=None,
        user_metadata=user_metadata or {},
        tags=tags,
        tag_origin="manual",
        require_existing_tags=False,
    )
    info_id = ingest_result.asset_info_id
    if not info_id:
        raise RuntimeError("failed to create asset metadata")

    with create_session() as session:
        pair = fetch_asset_info_and_asset(
            session, asset_info_id=info_id, owner_id=owner_id
        )
        if not pair:
            raise RuntimeError("inconsistent DB state after ingest")
        info, asset = pair
        tag_names = get_asset_tags(session, asset_info_id=info.id)

    return UploadResult(
        info=extract_info_data(info),
        asset=extract_asset_data(asset),
        tags=tag_names,
        created_new=ingest_result.asset_created,
    )


def create_from_hash(
    hash_str: str,
    name: str,
    tags: list[str] | None = None,
    user_metadata: dict | None = None,
    owner_id: str = "",
) -> UploadResult | None:
    canonical = hash_str.strip().lower()

    with create_session() as session:
        asset = get_asset_by_hash(session, asset_hash=canonical)
        if not asset:
            return None

    result = register_existing_asset(
        asset_hash=canonical,
        name=_sanitize_filename(
            name, fallback=canonical.split(":", 1)[1] if ":" in canonical else canonical
        ),
        user_metadata=user_metadata or {},
        tags=tags or [],
        tag_origin="manual",
        owner_id=owner_id,
    )

    return UploadResult(
        info=result.info,
        asset=result.asset,
        tags=result.tags,
        created_new=False,
    )
