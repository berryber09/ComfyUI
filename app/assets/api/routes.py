import logging
import uuid
import urllib.parse
import os
from aiohttp import web

from pydantic import ValidationError

import app.assets.manager as manager
from app import user_manager
from app.assets.api import schemas_in
from app.assets.api.schemas_in import (
    AssetNotFoundError,
    AssetValidationError,
    HashMismatchError,
    UploadError,
)
from app.assets.api.upload import parse_multipart_upload
from app.assets.services.scanner import seed_assets as scanner_seed_assets
from typing import Any


ROUTES = web.RouteTableDef()
USER_MANAGER: user_manager.UserManager | None = None

# UUID regex (canonical hyphenated form, case-insensitive)
UUID_RE = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"

def get_query_dict(request: web.Request) -> dict[str, Any]:
    """
    Gets a dictionary of query parameters from the request.

    'request.query' is a MultiMapping[str], needs to be converted to a dictionary to be validated by Pydantic.
    """
    query_dict = {
        key: request.query.getall(key) if len(request.query.getall(key)) > 1 else request.query.get(key)
        for key in request.query.keys()
    }
    return query_dict

# Note to any custom node developers reading this code:
# The assets system is not yet fully implemented, do not rely on the code in /app/assets remaining the same.

def register_assets_system(app: web.Application, user_manager_instance: user_manager.UserManager) -> None:
    global USER_MANAGER
    USER_MANAGER = user_manager_instance
    app.add_routes(ROUTES)

def _build_error_response(status: int, code: str, message: str, details: dict | None = None) -> web.Response:
    return web.json_response({"error": {"code": code, "message": message, "details": details or {}}}, status=status)


def _build_validation_error_response(code: str, ve: ValidationError) -> web.Response:
    return _build_error_response(400, code, "Validation failed.", {"errors": ve.json()})


@ROUTES.head("/api/assets/hash/{hash}")
async def head_asset_by_hash(request: web.Request) -> web.Response:
    hash_str = request.match_info.get("hash", "").strip().lower()
    if not hash_str or ":" not in hash_str:
        return _build_error_response(400, "INVALID_HASH", "hash must be like 'blake3:<hex>'")
    algo, digest = hash_str.split(":", 1)
    if algo != "blake3" or not digest or any(c for c in digest if c not in "0123456789abcdef"):
        return _build_error_response(400, "INVALID_HASH", "hash must be like 'blake3:<hex>'")
    exists = manager.asset_exists(asset_hash=hash_str)
    return web.Response(status=200 if exists else 404)


@ROUTES.get("/api/assets")
async def list_assets(request: web.Request) -> web.Response:
    """
    GET request to list assets.
    """
    query_dict = get_query_dict(request)
    try:
        q = schemas_in.ListAssetsQuery.model_validate(query_dict)
    except ValidationError as ve:
        return _build_validation_error_response("INVALID_QUERY", ve)

    payload = manager.list_assets(
        include_tags=q.include_tags,
        exclude_tags=q.exclude_tags,
        name_contains=q.name_contains,
        metadata_filter=q.metadata_filter,
        limit=q.limit,
        offset=q.offset,
        sort=q.sort,
        order=q.order,
        owner_id=USER_MANAGER.get_request_user_id(request),
    )
    return web.json_response(payload.model_dump(mode="json", exclude_none=True))


@ROUTES.get(f"/api/assets/{{id:{UUID_RE}}}")
async def get_asset(request: web.Request) -> web.Response:
    """
    GET request to get an asset's info as JSON.
    """
    asset_info_id = str(uuid.UUID(request.match_info["id"]))
    try:
        result = manager.get_asset(
            asset_info_id=asset_info_id,
            owner_id=USER_MANAGER.get_request_user_id(request),
        )
    except ValueError as e:
        return _build_error_response(404, "ASSET_NOT_FOUND", str(e), {"id": asset_info_id})
    except Exception:
        logging.exception(
            "get_asset failed for asset_info_id=%s, owner_id=%s",
            asset_info_id,
            USER_MANAGER.get_request_user_id(request),
        )
        return _build_error_response(500, "INTERNAL", "Unexpected server error.")
    return web.json_response(result.model_dump(mode="json"), status=200)


@ROUTES.get(f"/api/assets/{{id:{UUID_RE}}}/content")
async def download_asset_content(request: web.Request) -> web.Response:
    # question: do we need disposition? could we just stick with one of these?
    disposition = request.query.get("disposition", "attachment").lower().strip()
    if disposition not in {"inline", "attachment"}:
        disposition = "attachment"

    try:
        abs_path, content_type, filename = manager.resolve_asset_content_for_download(
            asset_info_id=str(uuid.UUID(request.match_info["id"])),
            owner_id=USER_MANAGER.get_request_user_id(request),
        )
    except ValueError as ve:
        return _build_error_response(404, "ASSET_NOT_FOUND", str(ve))
    except NotImplementedError as nie:
        return _build_error_response(501, "BACKEND_UNSUPPORTED", str(nie))
    except FileNotFoundError:
        return _build_error_response(404, "FILE_NOT_FOUND", "Underlying file not found on disk.")

    quoted = (filename or "").replace("\r", "").replace("\n", "").replace('"', "'")
    cd = f'{disposition}; filename="{quoted}"; filename*=UTF-8\'\'{urllib.parse.quote(filename)}'

    file_size = os.path.getsize(abs_path)
    logging.info(
        "download_asset_content: path=%s, size=%d bytes (%.2f MB), content_type=%s, filename=%s",
        abs_path,
        file_size,
        file_size / (1024 * 1024),
        content_type,
        filename,
    )

    async def stream_file_chunks():
        chunk_size = 64 * 1024
        with open(abs_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    return web.Response(
        body=stream_file_chunks(),
        content_type=content_type,
        headers={
            "Content-Disposition": cd,
            "Content-Length": str(file_size),
        },
    )


@ROUTES.post("/api/assets/from-hash")
async def create_asset_from_hash(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        body = schemas_in.CreateFromHashBody.model_validate(payload)
    except ValidationError as ve:
        return _build_validation_error_response("INVALID_BODY", ve)
    except Exception:
        return _build_error_response(400, "INVALID_JSON", "Request body must be valid JSON.")

    result = manager.create_asset_from_hash(
        hash_str=body.hash,
        name=body.name,
        tags=body.tags,
        user_metadata=body.user_metadata,
        owner_id=USER_MANAGER.get_request_user_id(request),
    )
    if result is None:
        return _build_error_response(404, "ASSET_NOT_FOUND", f"Asset content {body.hash} does not exist")
    return web.json_response(result.model_dump(mode="json"), status=201)


@ROUTES.post("/api/assets")
async def upload_asset(request: web.Request) -> web.Response:
    """Multipart/form-data endpoint for Asset uploads."""
    try:
        parsed = await parse_multipart_upload(request, check_hash_exists=manager.asset_exists)
    except UploadError as e:
        return _build_error_response(e.status, e.code, e.message)

    owner_id = USER_MANAGER.get_request_user_id(request)

    try:
        result = manager.process_upload(parsed=parsed, owner_id=owner_id)
    except AssetValidationError as e:
        return _build_error_response(400, e.code, str(e))
    except AssetNotFoundError as e:
        return _build_error_response(404, "ASSET_NOT_FOUND", str(e))
    except HashMismatchError as e:
        return _build_error_response(400, "HASH_MISMATCH", str(e))
    except Exception:
        logging.exception("process_upload failed for owner_id=%s", owner_id)
        return _build_error_response(500, "INTERNAL", "Unexpected server error.")

    status = 201 if result.created_new else 200
    return web.json_response(result.model_dump(mode="json"), status=status)


@ROUTES.put(f"/api/assets/{{id:{UUID_RE}}}")
async def update_asset(request: web.Request) -> web.Response:
    asset_info_id = str(uuid.UUID(request.match_info["id"]))
    try:
        body = schemas_in.UpdateAssetBody.model_validate(await request.json())
    except ValidationError as ve:
        return _build_validation_error_response("INVALID_BODY", ve)
    except Exception:
        return _build_error_response(400, "INVALID_JSON", "Request body must be valid JSON.")

    try:
        result = manager.update_asset(
            asset_info_id=asset_info_id,
            name=body.name,
            user_metadata=body.user_metadata,
            owner_id=USER_MANAGER.get_request_user_id(request),
        )
    except (ValueError, PermissionError) as ve:
        return _build_error_response(404, "ASSET_NOT_FOUND", str(ve), {"id": asset_info_id})
    except Exception:
        logging.exception(
            "update_asset failed for asset_info_id=%s, owner_id=%s",
            asset_info_id,
            USER_MANAGER.get_request_user_id(request),
        )
        return _build_error_response(500, "INTERNAL", "Unexpected server error.")
    return web.json_response(result.model_dump(mode="json"), status=200)


@ROUTES.delete(f"/api/assets/{{id:{UUID_RE}}}")
async def delete_asset(request: web.Request) -> web.Response:
    asset_info_id = str(uuid.UUID(request.match_info["id"]))
    delete_content = request.query.get("delete_content")
    delete_content = True if delete_content is None else delete_content.lower() not in {"0", "false", "no"}

    try:
        deleted = manager.delete_asset_reference(
            asset_info_id=asset_info_id,
            owner_id=USER_MANAGER.get_request_user_id(request),
            delete_content_if_orphan=delete_content,
        )
    except Exception:
        logging.exception(
            "delete_asset_reference failed for asset_info_id=%s, owner_id=%s",
            asset_info_id,
            USER_MANAGER.get_request_user_id(request),
        )
        return _build_error_response(500, "INTERNAL", "Unexpected server error.")

    if not deleted:
        return _build_error_response(404, "ASSET_NOT_FOUND", f"AssetInfo {asset_info_id} not found.")
    return web.Response(status=204)


@ROUTES.get("/api/tags")
async def get_tags(request: web.Request) -> web.Response:
    """
    GET request to list all tags based on query parameters.
    """
    query_map = dict(request.rel_url.query)

    try:
        query = schemas_in.TagsListQuery.model_validate(query_map)
    except ValidationError as e:
        return web.json_response(
            {"error": {"code": "INVALID_QUERY", "message": "Invalid query parameters", "details": e.errors()}},
            status=400,
        )

    result = manager.list_tags(
        prefix=query.prefix,
        limit=query.limit,
        offset=query.offset,
        order=query.order,
        include_zero=query.include_zero,
        owner_id=USER_MANAGER.get_request_user_id(request),
    )
    return web.json_response(result.model_dump(mode="json"))


@ROUTES.post(f"/api/assets/{{id:{UUID_RE}}}/tags")
async def add_asset_tags(request: web.Request) -> web.Response:
    asset_info_id = str(uuid.UUID(request.match_info["id"]))
    try:
        payload = await request.json()
        data = schemas_in.TagsAdd.model_validate(payload)
    except ValidationError as ve:
        return _build_error_response(400, "INVALID_BODY", "Invalid JSON body for tags add.", {"errors": ve.errors()})
    except Exception:
        return _build_error_response(400, "INVALID_JSON", "Request body must be valid JSON.")

    try:
        result = manager.add_tags_to_asset(
            asset_info_id=asset_info_id,
            tags=data.tags,
            origin="manual",
            owner_id=USER_MANAGER.get_request_user_id(request),
        )
    except (ValueError, PermissionError) as ve:
        return _build_error_response(404, "ASSET_NOT_FOUND", str(ve), {"id": asset_info_id})
    except Exception:
        logging.exception(
            "add_tags_to_asset failed for asset_info_id=%s, owner_id=%s",
            asset_info_id,
            USER_MANAGER.get_request_user_id(request),
        )
        return _build_error_response(500, "INTERNAL", "Unexpected server error.")

    return web.json_response(result.model_dump(mode="json"), status=200)


@ROUTES.delete(f"/api/assets/{{id:{UUID_RE}}}/tags")
async def delete_asset_tags(request: web.Request) -> web.Response:
    asset_info_id = str(uuid.UUID(request.match_info["id"]))
    try:
        payload = await request.json()
        data = schemas_in.TagsRemove.model_validate(payload)
    except ValidationError as ve:
        return _build_error_response(400, "INVALID_BODY", "Invalid JSON body for tags remove.", {"errors": ve.errors()})
    except Exception:
        return _build_error_response(400, "INVALID_JSON", "Request body must be valid JSON.")

    try:
        result = manager.remove_tags_from_asset(
            asset_info_id=asset_info_id,
            tags=data.tags,
            owner_id=USER_MANAGER.get_request_user_id(request),
        )
    except ValueError as ve:
        return _build_error_response(404, "ASSET_NOT_FOUND", str(ve), {"id": asset_info_id})
    except Exception:
        logging.exception(
            "remove_tags_from_asset failed for asset_info_id=%s, owner_id=%s",
            asset_info_id,
            USER_MANAGER.get_request_user_id(request),
        )
        return _build_error_response(500, "INTERNAL", "Unexpected server error.")

    return web.json_response(result.model_dump(mode="json"), status=200)


@ROUTES.post("/api/assets/seed")
async def seed_assets(request: web.Request) -> web.Response:
    """Trigger asset seeding for specified roots (models, input, output)."""
    try:
        payload = await request.json()
        roots = payload.get("roots", ["models", "input", "output"])
    except Exception:
        roots = ["models", "input", "output"]

    valid_roots = [r for r in roots if r in ("models", "input", "output")]
    if not valid_roots:
        return _build_error_response(400, "INVALID_BODY", "No valid roots specified")

    try:
        scanner_seed_assets(tuple(valid_roots))
    except Exception:
        logging.exception("scanner_seed_assets failed for roots=%s", valid_roots)
        return _build_error_response(500, "INTERNAL", "Seed operation failed")

    return web.json_response({"seeded": valid_roots}, status=200)
