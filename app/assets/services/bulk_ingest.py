from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypedDict

from sqlalchemy.orm import Session

from app.assets.database.queries import (
    bulk_insert_asset_infos_ignore_conflicts,
    bulk_insert_assets,
    bulk_insert_cache_states_ignore_conflicts,
    bulk_insert_tags_and_meta,
    delete_assets_by_ids,
    get_asset_info_ids_by_ids,
    get_cache_states_by_paths_and_asset_ids,
    get_existing_asset_ids,
    get_unreferenced_unhashed_asset_ids,
    mark_cache_states_missing_outside_prefixes,
    restore_cache_states_by_paths,
)
from app.assets.helpers import get_utc_now

if TYPE_CHECKING:
    from app.assets.services.metadata_extract import ExtractedMetadata


class SeedAssetSpec(TypedDict):
    """Spec for seeding an asset from filesystem."""

    abs_path: str
    size_bytes: int
    mtime_ns: int
    info_name: str
    tags: list[str]
    fname: str
    metadata: ExtractedMetadata | None
    hash: str | None


class AssetRow(TypedDict):
    """Row data for inserting an Asset."""

    id: str
    hash: str | None
    size_bytes: int
    mime_type: str | None
    created_at: datetime


class CacheStateRow(TypedDict):
    """Row data for inserting a CacheState."""

    asset_id: str
    file_path: str
    mtime_ns: int


class AssetInfoRow(TypedDict):
    """Row data for inserting an AssetInfo."""

    id: str
    owner_id: str
    name: str
    asset_id: str
    preview_id: str | None
    user_metadata: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    last_access_time: datetime


class AssetInfoRowInternal(TypedDict):
    """Internal row data for AssetInfo with extra tracking fields."""

    id: str
    owner_id: str
    name: str
    asset_id: str
    preview_id: str | None
    user_metadata: dict[str, Any] | None
    created_at: datetime
    updated_at: datetime
    last_access_time: datetime
    _tags: list[str]
    _filename: str
    _extracted_metadata: ExtractedMetadata | None


class TagRow(TypedDict):
    """Row data for inserting a Tag."""

    asset_info_id: str
    tag_name: str
    origin: str
    added_at: datetime


class MetadataRow(TypedDict):
    """Row data for inserting asset metadata."""

    asset_info_id: str
    key: str
    ordinal: int
    val_str: str | None
    val_num: float | None
    val_bool: bool | None
    val_json: dict[str, Any] | None


@dataclass
class BulkInsertResult:
    """Result of bulk asset insertion."""

    inserted_infos: int
    won_states: int
    lost_states: int


def batch_insert_seed_assets(
    session: Session,
    specs: list[SeedAssetSpec],
    owner_id: str = "",
) -> BulkInsertResult:
    """Seed assets from filesystem specs in batch.

    Each spec is a dict with keys:
      - abs_path: str
      - size_bytes: int
      - mtime_ns: int
      - info_name: str
      - tags: list[str]
      - fname: Optional[str]

    This function orchestrates:
    1. Insert seed Assets (hash=NULL)
    2. Claim cache states with ON CONFLICT DO NOTHING
    3. Query to find winners (paths where our asset_id was inserted)
    4. Delete Assets for losers (path already claimed by another asset)
    5. Insert AssetInfo for winners
    6. Insert tags and metadata for successfully inserted AssetInfos

    Returns:
        BulkInsertResult with inserted_infos, won_states, lost_states
    """
    if not specs:
        return BulkInsertResult(inserted_infos=0, won_states=0, lost_states=0)

    current_time = get_utc_now()
    asset_rows: list[AssetRow] = []
    cache_state_rows: list[CacheStateRow] = []
    path_to_asset_id: dict[str, str] = {}
    asset_id_to_info: dict[str, AssetInfoRowInternal] = {}
    absolute_path_list: list[str] = []

    for spec in specs:
        absolute_path = os.path.abspath(spec["abs_path"])
        asset_id = str(uuid.uuid4())
        asset_info_id = str(uuid.uuid4())
        absolute_path_list.append(absolute_path)
        path_to_asset_id[absolute_path] = asset_id

        asset_rows.append(
            {
                "id": asset_id,
                "hash": spec.get("hash"),
                "size_bytes": spec["size_bytes"],
                "mime_type": None,
                "created_at": current_time,
            }
        )
        cache_state_rows.append(
            {
                "asset_id": asset_id,
                "file_path": absolute_path,
                "mtime_ns": spec["mtime_ns"],
            }
        )
        # Build user_metadata from extracted metadata or fallback to filename
        extracted_metadata = spec.get("metadata")
        if extracted_metadata:
            user_metadata: dict[str, Any] | None = extracted_metadata.to_user_metadata()
        elif spec["fname"]:
            user_metadata = {"filename": spec["fname"]}
        else:
            user_metadata = None

        asset_id_to_info[asset_id] = {
            "id": asset_info_id,
            "owner_id": owner_id,
            "name": spec["info_name"],
            "asset_id": asset_id,
            "preview_id": None,
            "user_metadata": user_metadata,
            "created_at": current_time,
            "updated_at": current_time,
            "last_access_time": current_time,
            "_tags": spec["tags"],
            "_filename": spec["fname"],
            "_extracted_metadata": extracted_metadata,
        }

    bulk_insert_assets(session, asset_rows)

    # Filter cache states to only those whose assets were actually inserted
    # (assets with duplicate hashes are silently dropped by ON CONFLICT DO NOTHING)
    inserted_asset_ids = get_existing_asset_ids(
        session, [r["asset_id"] for r in cache_state_rows]
    )
    cache_state_rows = [
        r for r in cache_state_rows if r["asset_id"] in inserted_asset_ids
    ]

    bulk_insert_cache_states_ignore_conflicts(session, cache_state_rows)
    restore_cache_states_by_paths(session, absolute_path_list)
    winning_paths = get_cache_states_by_paths_and_asset_ids(session, path_to_asset_id)

    all_paths_set = set(absolute_path_list)
    losing_paths = all_paths_set - winning_paths
    lost_asset_ids = [path_to_asset_id[path] for path in losing_paths]

    if lost_asset_ids:
        delete_assets_by_ids(session, lost_asset_ids)

    if not winning_paths:
        return BulkInsertResult(
            inserted_infos=0,
            won_states=0,
            lost_states=len(losing_paths),
        )

    winner_info_rows = [
        asset_id_to_info[path_to_asset_id[path]] for path in winning_paths
    ]
    database_info_rows: list[AssetInfoRow] = [
        {
            "id": info_row["id"],
            "owner_id": info_row["owner_id"],
            "name": info_row["name"],
            "asset_id": info_row["asset_id"],
            "preview_id": info_row["preview_id"],
            "user_metadata": info_row["user_metadata"],
            "created_at": info_row["created_at"],
            "updated_at": info_row["updated_at"],
            "last_access_time": info_row["last_access_time"],
        }
        for info_row in winner_info_rows
    ]
    bulk_insert_asset_infos_ignore_conflicts(session, database_info_rows)

    all_info_ids = [info_row["id"] for info_row in winner_info_rows]
    inserted_info_ids = get_asset_info_ids_by_ids(session, all_info_ids)

    tag_rows: list[TagRow] = []
    metadata_rows: list[MetadataRow] = []
    if inserted_info_ids:
        for info_row in winner_info_rows:
            info_id = info_row["id"]
            if info_id not in inserted_info_ids:
                continue
            for tag in info_row["_tags"]:
                tag_rows.append(
                    {
                        "asset_info_id": info_id,
                        "tag_name": tag,
                        "origin": "automatic",
                        "added_at": current_time,
                    }
                )

            # Use extracted metadata for meta rows if available
            extracted_metadata = info_row.get("_extracted_metadata")
            if extracted_metadata:
                metadata_rows.extend(extracted_metadata.to_meta_rows(info_id))
            elif info_row["_filename"]:
                # Fallback: just store filename
                metadata_rows.append(
                    {
                        "asset_info_id": info_id,
                        "key": "filename",
                        "ordinal": 0,
                        "val_str": info_row["_filename"],
                        "val_num": None,
                        "val_bool": None,
                        "val_json": None,
                    }
                )

    bulk_insert_tags_and_meta(session, tag_rows=tag_rows, meta_rows=metadata_rows)

    return BulkInsertResult(
        inserted_infos=len(inserted_info_ids),
        won_states=len(winning_paths),
        lost_states=len(losing_paths),
    )


def mark_assets_missing_outside_prefixes(
    session: Session, valid_prefixes: list[str]
) -> int:
    """Mark cache states as missing when outside valid prefixes.

    This is a non-destructive operation that soft-deletes cache states
    by setting is_missing=True. User metadata is preserved and assets
    can be restored if the file reappears in a future scan.

    Note: This does NOT delete
    unreferenced unhashed assets. Those are preserved so user metadata
    remains intact even when base directories change.

    Args:
        session: Database session
        valid_prefixes: List of absolute directory prefixes that are valid

    Returns:
        Number of cache states marked as missing
    """
    return mark_cache_states_missing_outside_prefixes(session, valid_prefixes)


def cleanup_unreferenced_assets(session: Session) -> int:
    """Hard-delete unhashed assets with no active cache states.

    This is a destructive operation intended for explicit cleanup.
    Only deletes assets where hash=None and all cache states are missing.

    Returns:
        Number of assets deleted
    """
    unreferenced_ids = get_unreferenced_unhashed_asset_ids(session)
    return delete_assets_by_ids(session, unreferenced_ids)
