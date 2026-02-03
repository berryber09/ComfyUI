"""Tests for asset_management services."""
import pytest
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetInfo
from app.assets.database.queries import ensure_tags_exist, add_tags_to_asset_info
from app.assets.helpers import get_utc_now
from app.assets.services import (
    get_asset_detail,
    update_asset_metadata,
    delete_asset_reference,
    set_asset_preview,
)


def _make_asset(session: Session, hash_val: str = "blake3:test", size: int = 1024) -> Asset:
    asset = Asset(hash=hash_val, size_bytes=size, mime_type="application/octet-stream")
    session.add(asset)
    session.flush()
    return asset


def _make_asset_info(
    session: Session,
    asset: Asset,
    name: str = "test",
    owner_id: str = "",
) -> AssetInfo:
    now = get_utc_now()
    info = AssetInfo(
        owner_id=owner_id,
        name=name,
        asset_id=asset.id,
        created_at=now,
        updated_at=now,
        last_access_time=now,
    )
    session.add(info)
    session.flush()
    return info


class TestGetAssetDetail:
    def test_returns_none_for_nonexistent(self, mock_create_session):
        result = get_asset_detail(asset_info_id="nonexistent")
        assert result is None

    def test_returns_asset_with_tags(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        info = _make_asset_info(session, asset, name="test.bin")
        ensure_tags_exist(session, ["alpha", "beta"])
        add_tags_to_asset_info(session, asset_info_id=info.id, tags=["alpha", "beta"])
        session.commit()

        result = get_asset_detail(asset_info_id=info.id)

        assert result is not None
        assert result["info"].id == info.id
        assert result["asset"].id == asset.id
        assert set(result["tags"]) == {"alpha", "beta"}

    def test_respects_owner_visibility(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        info = _make_asset_info(session, asset, owner_id="user1")
        session.commit()

        # Wrong owner cannot see
        result = get_asset_detail(asset_info_id=info.id, owner_id="user2")
        assert result is None

        # Correct owner can see
        result = get_asset_detail(asset_info_id=info.id, owner_id="user1")
        assert result is not None


class TestUpdateAssetMetadata:
    def test_updates_name(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        info = _make_asset_info(session, asset, name="old_name.bin")
        info_id = info.id
        session.commit()

        update_asset_metadata(
            asset_info_id=info_id,
            name="new_name.bin",
        )

        # Verify by re-fetching from DB
        session.expire_all()
        updated_info = session.get(AssetInfo, info_id)
        assert updated_info.name == "new_name.bin"

    def test_updates_tags(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        info = _make_asset_info(session, asset)
        ensure_tags_exist(session, ["old"])
        add_tags_to_asset_info(session, asset_info_id=info.id, tags=["old"])
        session.commit()

        result = update_asset_metadata(
            asset_info_id=info.id,
            tags=["new1", "new2"],
        )

        assert set(result["tags"]) == {"new1", "new2"}
        assert "old" not in result["tags"]

    def test_updates_user_metadata(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        info = _make_asset_info(session, asset)
        info_id = info.id
        session.commit()

        update_asset_metadata(
            asset_info_id=info_id,
            user_metadata={"key": "value", "num": 42},
        )

        # Verify by re-fetching from DB
        session.expire_all()
        updated_info = session.get(AssetInfo, info_id)
        assert updated_info.user_metadata["key"] == "value"
        assert updated_info.user_metadata["num"] == 42

    def test_raises_for_nonexistent(self, mock_create_session):
        with pytest.raises(ValueError, match="not found"):
            update_asset_metadata(asset_info_id="nonexistent", name="fail")

    def test_raises_for_wrong_owner(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        info = _make_asset_info(session, asset, owner_id="user1")
        session.commit()

        with pytest.raises(PermissionError, match="not owner"):
            update_asset_metadata(
                asset_info_id=info.id,
                name="new",
                owner_id="user2",
            )


class TestDeleteAssetReference:
    def test_deletes_asset_info(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        info = _make_asset_info(session, asset)
        info_id = info.id
        session.commit()

        result = delete_asset_reference(
            asset_info_id=info_id,
            owner_id="",
            delete_content_if_orphan=False,
        )

        assert result is True
        assert session.get(AssetInfo, info_id) is None

    def test_returns_false_for_nonexistent(self, mock_create_session):
        result = delete_asset_reference(
            asset_info_id="nonexistent",
            owner_id="",
        )
        assert result is False

    def test_returns_false_for_wrong_owner(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        info = _make_asset_info(session, asset, owner_id="user1")
        info_id = info.id
        session.commit()

        result = delete_asset_reference(
            asset_info_id=info_id,
            owner_id="user2",
        )

        assert result is False
        assert session.get(AssetInfo, info_id) is not None

    def test_keeps_asset_if_other_infos_exist(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        info1 = _make_asset_info(session, asset, name="info1")
        _make_asset_info(session, asset, name="info2")  # Second info keeps asset alive
        asset_id = asset.id
        session.commit()

        delete_asset_reference(
            asset_info_id=info1.id,
            owner_id="",
            delete_content_if_orphan=True,
        )

        # Asset should still exist
        assert session.get(Asset, asset_id) is not None

    def test_deletes_orphaned_asset(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        info = _make_asset_info(session, asset)
        asset_id = asset.id
        info_id = info.id
        session.commit()

        delete_asset_reference(
            asset_info_id=info_id,
            owner_id="",
            delete_content_if_orphan=True,
        )

        # Both info and asset should be gone
        assert session.get(AssetInfo, info_id) is None
        assert session.get(Asset, asset_id) is None


class TestSetAssetPreview:
    def test_sets_preview(self, mock_create_session, session: Session):
        asset = _make_asset(session, hash_val="blake3:main")
        preview_asset = _make_asset(session, hash_val="blake3:preview")
        info = _make_asset_info(session, asset)
        info_id = info.id
        preview_id = preview_asset.id
        session.commit()

        set_asset_preview(
            asset_info_id=info_id,
            preview_asset_id=preview_id,
        )

        # Verify by re-fetching from DB
        session.expire_all()
        updated_info = session.get(AssetInfo, info_id)
        assert updated_info.preview_id == preview_id

    def test_clears_preview(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        preview_asset = _make_asset(session, hash_val="blake3:preview")
        info = _make_asset_info(session, asset)
        info.preview_id = preview_asset.id
        info_id = info.id
        session.commit()

        set_asset_preview(
            asset_info_id=info_id,
            preview_asset_id=None,
        )

        # Verify by re-fetching from DB
        session.expire_all()
        updated_info = session.get(AssetInfo, info_id)
        assert updated_info.preview_id is None

    def test_raises_for_nonexistent_info(self, mock_create_session):
        with pytest.raises(ValueError, match="not found"):
            set_asset_preview(asset_info_id="nonexistent")

    def test_raises_for_wrong_owner(self, mock_create_session, session: Session):
        asset = _make_asset(session)
        info = _make_asset_info(session, asset, owner_id="user1")
        session.commit()

        with pytest.raises(PermissionError, match="not owner"):
            set_asset_preview(
                asset_info_id=info.id,
                preview_asset_id=None,
                owner_id="user2",
            )
