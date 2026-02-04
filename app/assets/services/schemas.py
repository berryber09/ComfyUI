from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class AssetData:
    """Plain data extracted from an Asset ORM object."""
    hash: str
    size_bytes: int | None
    mime_type: str | None


@dataclass(frozen=True)
class AssetInfoData:
    """Plain data extracted from an AssetInfo ORM object."""
    id: str
    name: str
    user_metadata: dict | None
    preview_id: str | None
    created_at: datetime
    updated_at: datetime
    last_access_time: datetime | None


@dataclass(frozen=True)
class AssetDetailResult:
    """Result from get_asset_detail and similar operations."""
    info: AssetInfoData
    asset: AssetData | None
    tags: list[str]


@dataclass(frozen=True)
class RegisterAssetResult:
    """Result from register_existing_asset."""
    info: AssetInfoData
    asset: AssetData
    tags: list[str]
    created: bool


def extract_info_data(info) -> AssetInfoData:
    """Extract plain data from an AssetInfo ORM object."""
    return AssetInfoData(
        id=info.id,
        name=info.name,
        user_metadata=info.user_metadata,
        preview_id=info.preview_id,
        created_at=info.created_at,
        updated_at=info.updated_at,
        last_access_time=info.last_access_time,
    )


def extract_asset_data(asset) -> AssetData | None:
    """Extract plain data from an Asset ORM object."""
    if asset is None:
        return None
    return AssetData(
        hash=asset.hash,
        size_bytes=asset.size_bytes,
        mime_type=asset.mime_type,
    )
