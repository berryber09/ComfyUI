import contextlib
import mimetypes
import os
from typing import Sequence

from pydantic import ValidationError

import app.assets.services.hashing as hashing
import folder_paths
from app.assets.api import schemas_in, schemas_out
from app.assets.api.schemas_in import (
    AssetNotFoundError,
    AssetValidationError,
    DependencyMissingError,
    HashMismatchError,
    ParsedUpload,
)
from app.assets.api.upload import _delete_temp_file_if_exists
from app.assets.database.queries import (
    asset_exists_by_hash,
    fetch_asset_info_and_asset,
    get_asset_by_hash,
    get_asset_tags,
    list_asset_infos_page,
    list_cache_states_by_asset_id,
    update_asset_info_access_time,
)
from app.assets.helpers import select_best_live_path
from app.assets.services import (
    apply_tags,
    get_asset_detail,
    ingest_file_from_path,
    register_existing_asset,
    remove_tags,
    update_asset_metadata,
)
from app.assets.services import (
    delete_asset_reference as svc_delete_asset_reference,
)
from app.assets.services import (
    set_asset_preview as svc_set_asset_preview,
)
from app.assets.services.path_utils import (
    resolve_destination_from_tags,
    validate_path_within_base,
)
from app.assets.services.tagging import list_tags as svc_list_tags
from app.database.db import create_session


def _validate_sort_field(requested: str | None) -> str:
    if not requested:
        return "created_at"
    v = requested.lower()
    if v in {"name", "created_at", "updated_at", "size", "last_access_time"}:
        return v
    return "created_at"


def _get_size_mtime_ns(path: str) -> tuple[int, int]:
    st = os.stat(path, follow_symlinks=True)
    return st.st_size, getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))


def _sanitize_filename(name: str | None, fallback: str) -> str:
    n = os.path.basename((name or "").strip() or fallback)
    if n:
        return n
    return fallback


def asset_exists(asset_hash: str) -> bool:
    with create_session() as session:
        return asset_exists_by_hash(session, asset_hash=asset_hash)


def list_assets(
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    name_contains: str | None = None,
    metadata_filter: dict | None = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "created_at",
    order: str = "desc",
    owner_id: str = "",
) -> schemas_out.AssetsList:
    sort = _validate_sort_field(sort)
    order = "desc" if (order or "desc").lower() not in {"asc", "desc"} else order.lower()

    with create_session() as session:
        infos, tag_map, total = list_asset_infos_page(
            session,
            owner_id=owner_id,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            name_contains=name_contains,
            metadata_filter=metadata_filter,
            limit=limit,
            offset=offset,
            sort=sort,
            order=order,
        )

    summaries: list[schemas_out.AssetSummary] = []
    for info in infos:
        asset = info.asset
        tags = tag_map.get(info.id, [])
        summaries.append(
            schemas_out.AssetSummary(
                id=info.id,
                name=info.name,
                asset_hash=asset.hash if asset else None,
                size=int(asset.size_bytes) if asset else None,
                mime_type=asset.mime_type if asset else None,
                tags=tags,
                created_at=info.created_at,
                updated_at=info.updated_at,
                last_access_time=info.last_access_time,
            )
        )

    return schemas_out.AssetsList(
        assets=summaries,
        total=total,
        has_more=(offset + len(summaries)) < total,
    )


def get_asset(
    asset_info_id: str,
    owner_id: str = "",
) -> schemas_out.AssetDetail:
    result = get_asset_detail(asset_info_id=asset_info_id, owner_id=owner_id)
    if not result:
        raise ValueError(f"AssetInfo {asset_info_id} not found")

    info = result.info
    asset = result.asset

    return schemas_out.AssetDetail(
        id=info.id,
        name=info.name,
        asset_hash=asset.hash if asset else None,
        size=int(asset.size_bytes) if asset and asset.size_bytes is not None else None,
        mime_type=asset.mime_type if asset else None,
        tags=result.tags,
        user_metadata=info.user_metadata or {},
        preview_id=info.preview_id,
        created_at=info.created_at,
        last_access_time=info.last_access_time,
    )


def resolve_asset_content_for_download(
    asset_info_id: str,
    owner_id: str = "",
) -> tuple[str, str, str]:
    with create_session() as session:
        pair = fetch_asset_info_and_asset(session, asset_info_id=asset_info_id, owner_id=owner_id)
        if not pair:
            raise ValueError(f"AssetInfo {asset_info_id} not found")

        info, asset = pair
        states = list_cache_states_by_asset_id(session, asset_id=asset.id)
        abs_path = select_best_live_path(states)
        if not abs_path:
            raise FileNotFoundError

        update_asset_info_access_time(session, asset_info_id=asset_info_id)
        session.commit()

        ctype = asset.mime_type or mimetypes.guess_type(info.name or abs_path)[0] or "application/octet-stream"
        download_name = info.name or os.path.basename(abs_path)
        return abs_path, ctype, download_name


def upload_asset_from_temp_path(
    spec: schemas_in.UploadAssetSpec,
    temp_path: str,
    client_filename: str | None = None,
    owner_id: str = "",
    expected_asset_hash: str | None = None,
) -> schemas_out.AssetCreated:
    try:
        digest = hashing.compute_blake3_hash(temp_path)
    except ImportError as e:
        raise DependencyMissingError(str(e))
    except Exception as e:
        raise RuntimeError(f"failed to hash uploaded file: {e}")
    asset_hash = "blake3:" + digest

    if expected_asset_hash and asset_hash != expected_asset_hash.strip().lower():
        raise ValueError("HASH_MISMATCH")

    # Check if asset already exists by hash
    with create_session() as session:
        existing = get_asset_by_hash(session, asset_hash=asset_hash)

    if existing is not None:
        with contextlib.suppress(Exception):
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        display_name = _sanitize_filename(spec.name or (client_filename or ""), fallback=digest)
        result = register_existing_asset(
            asset_hash=asset_hash,
            name=display_name,
            user_metadata=spec.user_metadata or {},
            tags=spec.tags or [],
            tag_origin="manual",
            owner_id=owner_id,
        )
        info = result.info
        asset = result.asset

        return schemas_out.AssetCreated(
            id=info.id,
            name=info.name,
            asset_hash=asset.hash,
            size=int(asset.size_bytes) if asset.size_bytes is not None else None,
            mime_type=asset.mime_type,
            tags=result.tags,
            user_metadata=info.user_metadata or {},
            preview_id=info.preview_id,
            created_at=info.created_at,
            last_access_time=info.last_access_time,
            created_new=False,
        )

    # New asset - move file to destination
    base_dir, subdirs = resolve_destination_from_tags(spec.tags)
    dest_dir = os.path.join(base_dir, *subdirs) if subdirs else base_dir
    os.makedirs(dest_dir, exist_ok=True)

    src_for_ext = (client_filename or spec.name or "").strip()
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
        size_bytes, mtime_ns = _get_size_mtime_ns(dest_abs)
    except OSError as e:
        raise RuntimeError(f"failed to stat destination file: {e}")

    result = ingest_file_from_path(
        asset_hash=asset_hash,
        abs_path=dest_abs,
        size_bytes=size_bytes,
        mtime_ns=mtime_ns,
        mime_type=content_type,
        info_name=_sanitize_filename(spec.name or (client_filename or ""), fallback=digest),
        owner_id=owner_id,
        preview_id=None,
        user_metadata=spec.user_metadata or {},
        tags=spec.tags,
        tag_origin="manual",
        require_existing_tags=False,
    )
    info_id = result["asset_info_id"]
    if not info_id:
        raise RuntimeError("failed to create asset metadata")

    with create_session() as session:
        pair = fetch_asset_info_and_asset(session, asset_info_id=info_id, owner_id=owner_id)
        if not pair:
            raise RuntimeError("inconsistent DB state after ingest")
        info, asset = pair
        tag_names = get_asset_tags(session, asset_info_id=info.id)

    return schemas_out.AssetCreated(
        id=info.id,
        name=info.name,
        asset_hash=asset.hash,
        size=int(asset.size_bytes),
        mime_type=asset.mime_type,
        tags=tag_names,
        user_metadata=info.user_metadata or {},
        preview_id=info.preview_id,
        created_at=info.created_at,
        last_access_time=info.last_access_time,
        created_new=result["asset_created"],
    )


def process_upload(
    parsed: ParsedUpload,
    owner_id: str = "",
) -> schemas_out.AssetCreated:
    """
    Process a parsed multipart upload.

    Args:
        parsed: The parsed upload data from parse_multipart_upload
        owner_id: The owner ID for the asset

    Returns:
        AssetCreated response (check created_new to determine if new asset was created)

    Raises:
        UploadError: On validation or processing errors
    """
    try:
        spec = schemas_in.UploadAssetSpec.model_validate({
            "tags": parsed.tags_raw,
            "name": parsed.provided_name,
            "user_metadata": parsed.user_metadata_raw,
            "hash": parsed.provided_hash,
        })
    except ValidationError as ve:
        _delete_temp_file_if_exists(parsed.tmp_path)
        raise AssetValidationError("INVALID_BODY", f"Validation failed: {ve.json()}")

    # Validate models category against configured folders
    if spec.tags and spec.tags[0] == "models":
        if len(spec.tags) < 2 or spec.tags[1] not in folder_paths.folder_names_and_paths:
            _delete_temp_file_if_exists(parsed.tmp_path)
            category = spec.tags[1] if len(spec.tags) >= 2 else ""
            raise AssetValidationError("INVALID_BODY", f"unknown models category '{category}'")

    # Fast path: if a valid provided hash exists, create AssetInfo without writing anything
    if spec.hash and parsed.provided_hash_exists is True:
        result = create_asset_from_hash(
            hash_str=spec.hash,
            name=spec.name or (spec.hash.split(":", 1)[1]),
            tags=spec.tags,
            user_metadata=spec.user_metadata or {},
            owner_id=owner_id,
        )

        if result is None:
            raise AssetNotFoundError(f"Asset content {spec.hash} does not exist")

        # Drain temp if we accidentally saved (e.g., hash field came after file)
        _delete_temp_file_if_exists(parsed.tmp_path)
        return result

    # Otherwise, we must have a temp file path to ingest
    if not parsed.tmp_path or not os.path.exists(parsed.tmp_path):
        raise AssetNotFoundError("Provided hash not found and no file uploaded.")

    try:
        return upload_asset_from_temp_path(
            spec,
            temp_path=parsed.tmp_path,
            client_filename=parsed.file_client_name,
            owner_id=owner_id,
            expected_asset_hash=spec.hash,
        )
    except ValueError as e:
        _delete_temp_file_if_exists(parsed.tmp_path)
        msg = str(e)
        if "HASH_MISMATCH" in msg or msg.strip().upper() == "HASH_MISMATCH":
            raise HashMismatchError("Uploaded file hash does not match provided hash.")
        raise AssetValidationError("BAD_REQUEST", "Invalid inputs.")
    except Exception:
        _delete_temp_file_if_exists(parsed.tmp_path)
        raise


def update_asset(
    asset_info_id: str,
    name: str | None = None,
    tags: list[str] | None = None,
    user_metadata: dict | None = None,
    owner_id: str = "",
) -> schemas_out.AssetUpdated:
    result = update_asset_metadata(
        asset_info_id=asset_info_id,
        name=name,
        tags=tags,
        user_metadata=user_metadata,
        tag_origin="manual",
        owner_id=owner_id,
    )
    info = result.info
    asset = result.asset

    return schemas_out.AssetUpdated(
        id=info.id,
        name=info.name,
        asset_hash=asset.hash if asset else None,
        tags=result.tags,
        user_metadata=info.user_metadata or {},
        updated_at=info.updated_at,
    )


def set_asset_preview(
    asset_info_id: str,
    preview_asset_id: str | None = None,
    owner_id: str = "",
) -> schemas_out.AssetDetail:
    result = svc_set_asset_preview(
        asset_info_id=asset_info_id,
        preview_asset_id=preview_asset_id,
        owner_id=owner_id,
    )
    info = result.info
    asset = result.asset

    return schemas_out.AssetDetail(
        id=info.id,
        name=info.name,
        asset_hash=asset.hash if asset else None,
        size=int(asset.size_bytes) if asset and asset.size_bytes is not None else None,
        mime_type=asset.mime_type if asset else None,
        tags=result.tags,
        user_metadata=info.user_metadata or {},
        preview_id=info.preview_id,
        created_at=info.created_at,
        last_access_time=info.last_access_time,
    )


def delete_asset_reference(asset_info_id: str, owner_id: str, delete_content_if_orphan: bool = True) -> bool:
    return svc_delete_asset_reference(
        asset_info_id=asset_info_id,
        owner_id=owner_id,
        delete_content_if_orphan=delete_content_if_orphan,
    )


def create_asset_from_hash(
    hash_str: str,
    name: str,
    tags: list[str] | None = None,
    user_metadata: dict | None = None,
    owner_id: str = "",
) -> schemas_out.AssetCreated | None:
    canonical = hash_str.strip().lower()

    with create_session() as session:
        asset = get_asset_by_hash(session, asset_hash=canonical)
        if not asset:
            return None

    result = register_existing_asset(
        asset_hash=canonical,
        name=_sanitize_filename(name, fallback=canonical.split(":", 1)[1] if ":" in canonical else canonical),
        user_metadata=user_metadata or {},
        tags=tags or [],
        tag_origin="manual",
        owner_id=owner_id,
    )
    info = result.info
    asset = result.asset

    # created_new indicates whether new CONTENT was created, not whether a new AssetInfo was created.
    # Since we're referencing an existing hash, the content already exists.
    return schemas_out.AssetCreated(
        id=info.id,
        name=info.name,
        asset_hash=asset.hash,
        size=int(asset.size_bytes),
        mime_type=asset.mime_type,
        tags=result.tags,
        user_metadata=info.user_metadata or {},
        preview_id=info.preview_id,
        created_at=info.created_at,
        last_access_time=info.last_access_time,
        created_new=False,
    )


def add_tags_to_asset(
    asset_info_id: str,
    tags: list[str],
    origin: str = "manual",
    owner_id: str = "",
) -> schemas_out.TagsAdd:
    data = apply_tags(
        asset_info_id=asset_info_id,
        tags=tags,
        origin=origin,
        owner_id=owner_id,
    )
    return schemas_out.TagsAdd(**data)


def remove_tags_from_asset(
    asset_info_id: str,
    tags: list[str],
    owner_id: str = "",
) -> schemas_out.TagsRemove:
    data = remove_tags(
        asset_info_id=asset_info_id,
        tags=tags,
        owner_id=owner_id,
    )
    return schemas_out.TagsRemove(**data)


def list_tags(
    prefix: str | None = None,
    limit: int = 100,
    offset: int = 0,
    order: str = "count_desc",
    include_zero: bool = True,
    owner_id: str = "",
) -> schemas_out.TagsList:
    rows, total = svc_list_tags(
        prefix=prefix,
        limit=limit,
        offset=offset,
        order=order,
        include_zero=include_zero,
        owner_id=owner_id,
    )

    tags = [schemas_out.TagUsage(name=name, count=count, type=tag_type) for (name, tag_type, count) in rows]
    return schemas_out.TagsList(tags=tags, total=total, has_more=(offset + len(tags)) < total)
