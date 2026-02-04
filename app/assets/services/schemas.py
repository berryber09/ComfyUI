from dataclasses import dataclass
from datetime import datetime
from typing import Any, NamedTuple

from app.assets.database.models import Asset, AssetInfo

UserMetadata = dict[str, Any] | None


@dataclass(frozen=True)
class AssetData:
    hash: str
    size_bytes: int | None
    mime_type: str | None


@dataclass(frozen=True)
class AssetInfoData:
    id: str
    name: str
    user_metadata: UserMetadata
    preview_id: str | None
    created_at: datetime
    updated_at: datetime
    last_access_time: datetime | None


@dataclass(frozen=True)
class AssetDetailResult:
    info: AssetInfoData
    asset: AssetData | None
    tags: list[str]


@dataclass(frozen=True)
class RegisterAssetResult:
    info: AssetInfoData
    asset: AssetData
    tags: list[str]
    created: bool


@dataclass(frozen=True)
class IngestResult:
    asset_created: bool
    asset_updated: bool
    state_created: bool
    state_updated: bool
    asset_info_id: str | None


@dataclass(frozen=True)
class AddTagsResult:
    added: list[str]
    already_present: list[str]
    total_tags: list[str]


@dataclass(frozen=True)
class RemoveTagsResult:
    removed: list[str]
    not_present: list[str]
    total_tags: list[str]


@dataclass(frozen=True)
class SetTagsResult:
    added: list[str]
    removed: list[str]
    total: list[str]


class TagUsage(NamedTuple):
    name: str
    tag_type: str
    count: int


def extract_info_data(info: AssetInfo) -> AssetInfoData:
    return AssetInfoData(
        id=info.id,
        name=info.name,
        user_metadata=info.user_metadata,
        preview_id=info.preview_id,
        created_at=info.created_at,
        updated_at=info.updated_at,
        last_access_time=info.last_access_time,
    )


def extract_asset_data(asset: Asset | None) -> AssetData | None:
    if asset is None:
        return None
    return AssetData(
        hash=asset.hash,
        size_bytes=asset.size_bytes,
        mime_type=asset.mime_type,
    )
