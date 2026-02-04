import os
import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.assets.database.queries import (
    bulk_insert_asset_infos_ignore_conflicts,
    bulk_insert_assets,
    bulk_insert_cache_states_ignore_conflicts,
    bulk_insert_tags_and_meta,
    delete_assets_by_ids,
    delete_cache_states_outside_prefixes,
    get_asset_info_ids_by_ids,
    get_cache_states_by_paths_and_asset_ids,
    get_orphaned_seed_asset_ids,
)
from app.assets.helpers import get_utc_now


@dataclass
class BulkInsertResult:
    """Result of bulk asset insertion."""

    inserted_infos: int
    won_states: int
    lost_states: int


def batch_insert_seed_assets(
    session: Session,
    specs: list[dict],
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

    now = get_utc_now()
    asset_rows: list[dict] = []
    state_rows: list[dict] = []
    path_to_asset: dict[str, str] = {}
    asset_to_info: dict[str, dict] = {}
    path_list: list[str] = []

    for sp in specs:
        ap = os.path.abspath(sp["abs_path"])
        aid = str(uuid.uuid4())
        iid = str(uuid.uuid4())
        path_list.append(ap)
        path_to_asset[ap] = aid

        asset_rows.append(
            {
                "id": aid,
                "hash": None,
                "size_bytes": sp["size_bytes"],
                "mime_type": None,
                "created_at": now,
            }
        )
        state_rows.append(
            {
                "asset_id": aid,
                "file_path": ap,
                "mtime_ns": sp["mtime_ns"],
            }
        )
        asset_to_info[aid] = {
            "id": iid,
            "owner_id": owner_id,
            "name": sp["info_name"],
            "asset_id": aid,
            "preview_id": None,
            "user_metadata": {"filename": sp["fname"]} if sp["fname"] else None,
            "created_at": now,
            "updated_at": now,
            "last_access_time": now,
            "_tags": sp["tags"],
            "_filename": sp["fname"],
        }

    bulk_insert_assets(session, asset_rows)
    bulk_insert_cache_states_ignore_conflicts(session, state_rows)
    winners_by_path = get_cache_states_by_paths_and_asset_ids(session, path_to_asset)

    all_paths_set = set(path_list)
    losers_by_path = all_paths_set - winners_by_path
    lost_assets = [path_to_asset[p] for p in losers_by_path]

    if lost_assets:
        delete_assets_by_ids(session, lost_assets)

    if not winners_by_path:
        return BulkInsertResult(
            inserted_infos=0,
            won_states=0,
            lost_states=len(losers_by_path),
        )

    winner_info_rows = [asset_to_info[path_to_asset[p]] for p in winners_by_path]
    db_info_rows = [
        {
            "id": row["id"],
            "owner_id": row["owner_id"],
            "name": row["name"],
            "asset_id": row["asset_id"],
            "preview_id": row["preview_id"],
            "user_metadata": row["user_metadata"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_access_time": row["last_access_time"],
        }
        for row in winner_info_rows
    ]
    bulk_insert_asset_infos_ignore_conflicts(session, db_info_rows)

    all_info_ids = [row["id"] for row in winner_info_rows]
    inserted_info_ids = get_asset_info_ids_by_ids(session, all_info_ids)

    tag_rows: list[dict] = []
    meta_rows: list[dict] = []
    if inserted_info_ids:
        for row in winner_info_rows:
            iid = row["id"]
            if iid not in inserted_info_ids:
                continue
            for t in row["_tags"]:
                tag_rows.append(
                    {
                        "asset_info_id": iid,
                        "tag_name": t,
                        "origin": "automatic",
                        "added_at": now,
                    }
                )
            if row["_filename"]:
                meta_rows.append(
                    {
                        "asset_info_id": iid,
                        "key": "filename",
                        "ordinal": 0,
                        "val_str": row["_filename"],
                        "val_num": None,
                        "val_bool": None,
                        "val_json": None,
                    }
                )

    bulk_insert_tags_and_meta(session, tag_rows=tag_rows, meta_rows=meta_rows)

    return BulkInsertResult(
        inserted_infos=len(inserted_info_ids),
        won_states=len(winners_by_path),
        lost_states=len(losers_by_path),
    )


def prune_orphaned_assets(session: Session, valid_prefixes: list[str]) -> int:
    """Prune cache states outside valid prefixes, then delete orphaned seed assets.

    Args:
        session: Database session
        valid_prefixes: List of absolute directory prefixes that are valid

    Returns:
        Number of orphaned assets deleted
    """
    delete_cache_states_outside_prefixes(session, valid_prefixes)
    orphan_ids = get_orphaned_seed_asset_ids(session)
    return delete_assets_by_ids(session, orphan_ids)
