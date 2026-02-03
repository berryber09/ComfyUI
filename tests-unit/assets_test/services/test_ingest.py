"""Tests for ingest services."""
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetCacheState, AssetInfo, Tag
from app.assets.database.queries import get_asset_tags
from app.assets.services import ingest_file_from_path, register_existing_asset


class TestIngestFileFromPath:
    def test_creates_asset_and_cache_state(self, mock_create_session, temp_dir: Path, session: Session):
        file_path = temp_dir / "test_file.bin"
        file_path.write_bytes(b"test content")

        result = ingest_file_from_path(
            abs_path=str(file_path),
            asset_hash="blake3:abc123",
            size_bytes=12,
            mtime_ns=1234567890000000000,
            mime_type="application/octet-stream",
        )

        assert result["asset_created"] is True
        assert result["state_created"] is True
        assert result["asset_info_id"] is None  # no info_name provided

        # Verify DB state
        assets = session.query(Asset).all()
        assert len(assets) == 1
        assert assets[0].hash == "blake3:abc123"

        states = session.query(AssetCacheState).all()
        assert len(states) == 1
        assert states[0].file_path == str(file_path)

    def test_creates_asset_info_when_name_provided(self, mock_create_session, temp_dir: Path, session: Session):
        file_path = temp_dir / "model.safetensors"
        file_path.write_bytes(b"model data")

        result = ingest_file_from_path(
            abs_path=str(file_path),
            asset_hash="blake3:def456",
            size_bytes=10,
            mtime_ns=1234567890000000000,
            mime_type="application/octet-stream",
            info_name="My Model",
            owner_id="user1",
        )

        assert result["asset_created"] is True
        assert result["asset_info_id"] is not None

        info = session.query(AssetInfo).first()
        assert info is not None
        assert info.name == "My Model"
        assert info.owner_id == "user1"

    def test_creates_tags_when_provided(self, mock_create_session, temp_dir: Path, session: Session):
        file_path = temp_dir / "tagged.bin"
        file_path.write_bytes(b"data")

        result = ingest_file_from_path(
            abs_path=str(file_path),
            asset_hash="blake3:ghi789",
            size_bytes=4,
            mtime_ns=1234567890000000000,
            info_name="Tagged Asset",
            tags=["models", "checkpoints"],
        )

        assert result["asset_info_id"] is not None

        # Verify tags were created and linked
        tags = session.query(Tag).all()
        tag_names = {t.name for t in tags}
        assert "models" in tag_names
        assert "checkpoints" in tag_names

        asset_tags = get_asset_tags(session, asset_info_id=result["asset_info_id"])
        assert set(asset_tags) == {"models", "checkpoints"}

    def test_idempotent_upsert(self, mock_create_session, temp_dir: Path, session: Session):
        file_path = temp_dir / "dup.bin"
        file_path.write_bytes(b"content")

        # First ingest
        r1 = ingest_file_from_path(
            abs_path=str(file_path),
            asset_hash="blake3:repeat",
            size_bytes=7,
            mtime_ns=1234567890000000000,
        )
        assert r1["asset_created"] is True

        # Second ingest with same hash - should update, not create
        r2 = ingest_file_from_path(
            abs_path=str(file_path),
            asset_hash="blake3:repeat",
            size_bytes=7,
            mtime_ns=1234567890000000001,  # different mtime
        )
        assert r2["asset_created"] is False
        assert r2["state_updated"] is True or r2["state_created"] is False

        # Still only one asset
        assets = session.query(Asset).all()
        assert len(assets) == 1

    def test_validates_preview_id(self, mock_create_session, temp_dir: Path, session: Session):
        file_path = temp_dir / "with_preview.bin"
        file_path.write_bytes(b"data")

        # Create a preview asset first
        preview_asset = Asset(hash="blake3:preview", size_bytes=100)
        session.add(preview_asset)
        session.commit()
        preview_id = preview_asset.id

        result = ingest_file_from_path(
            abs_path=str(file_path),
            asset_hash="blake3:main",
            size_bytes=4,
            mtime_ns=1234567890000000000,
            info_name="With Preview",
            preview_id=preview_id,
        )

        assert result["asset_info_id"] is not None
        info = session.query(AssetInfo).filter_by(id=result["asset_info_id"]).first()
        assert info.preview_id == preview_id

    def test_invalid_preview_id_is_cleared(self, mock_create_session, temp_dir: Path, session: Session):
        file_path = temp_dir / "bad_preview.bin"
        file_path.write_bytes(b"data")

        result = ingest_file_from_path(
            abs_path=str(file_path),
            asset_hash="blake3:badpreview",
            size_bytes=4,
            mtime_ns=1234567890000000000,
            info_name="Bad Preview",
            preview_id="nonexistent-uuid",
        )

        assert result["asset_info_id"] is not None
        info = session.query(AssetInfo).filter_by(id=result["asset_info_id"]).first()
        assert info.preview_id is None


class TestRegisterExistingAsset:
    def test_creates_info_for_existing_asset(self, mock_create_session, session: Session):
        # Create existing asset
        asset = Asset(hash="blake3:existing", size_bytes=1024, mime_type="image/png")
        session.add(asset)
        session.commit()

        result = register_existing_asset(
            asset_hash="blake3:existing",
            name="Registered Asset",
            user_metadata={"key": "value"},
            tags=["models"],
        )

        assert result["created"] is True
        assert "models" in result["tags"]

        # Verify by re-fetching from DB
        session.expire_all()
        infos = session.query(AssetInfo).filter_by(name="Registered Asset").all()
        assert len(infos) == 1

    def test_returns_existing_info(self, mock_create_session, session: Session):
        # Create asset and info
        asset = Asset(hash="blake3:withinfo", size_bytes=512)
        session.add(asset)
        session.flush()

        from app.assets.helpers import get_utc_now
        info = AssetInfo(
            owner_id="",
            name="Existing Info",
            asset_id=asset.id,
            created_at=get_utc_now(),
            updated_at=get_utc_now(),
            last_access_time=get_utc_now(),
        )
        session.add(info)
        session.flush()  # Flush to get the ID
        info_id = info.id
        session.commit()

        result = register_existing_asset(
            asset_hash="blake3:withinfo",
            name="Existing Info",
            owner_id="",
        )

        assert result["created"] is False

        # Verify only one AssetInfo exists for this name
        session.expire_all()
        infos = session.query(AssetInfo).filter_by(name="Existing Info").all()
        assert len(infos) == 1
        assert infos[0].id == info_id

    def test_raises_for_nonexistent_hash(self, mock_create_session):
        with pytest.raises(ValueError, match="No asset with hash"):
            register_existing_asset(
                asset_hash="blake3:doesnotexist",
                name="Fail",
            )

    def test_applies_tags_to_new_info(self, mock_create_session, session: Session):
        asset = Asset(hash="blake3:tagged", size_bytes=256)
        session.add(asset)
        session.commit()

        result = register_existing_asset(
            asset_hash="blake3:tagged",
            name="Tagged Info",
            tags=["alpha", "beta"],
        )

        assert result["created"] is True
        assert set(result["tags"]) == {"alpha", "beta"}
