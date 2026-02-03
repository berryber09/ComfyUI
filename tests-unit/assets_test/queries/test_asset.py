import uuid
from sqlalchemy.orm import Session

from app.assets.database.models import Asset
from app.assets.database.queries import (
    asset_exists_by_hash,
    get_asset_by_hash,
    upsert_asset,
    bulk_insert_assets,
)


class TestAssetExistsByHash:
    def test_returns_false_for_nonexistent(self, session: Session):
        assert asset_exists_by_hash(session, asset_hash="nonexistent") is False

    def test_returns_true_for_existing(self, session: Session):
        asset = Asset(hash="blake3:abc123", size_bytes=100)
        session.add(asset)
        session.commit()

        assert asset_exists_by_hash(session, asset_hash="blake3:abc123") is True

    def test_does_not_match_null_hash(self, session: Session):
        asset = Asset(hash=None, size_bytes=100)
        session.add(asset)
        session.commit()

        assert asset_exists_by_hash(session, asset_hash="") is False


class TestGetAssetByHash:
    def test_returns_none_for_nonexistent(self, session: Session):
        assert get_asset_by_hash(session, asset_hash="nonexistent") is None

    def test_returns_asset_for_existing(self, session: Session):
        asset = Asset(hash="blake3:def456", size_bytes=200, mime_type="image/png")
        session.add(asset)
        session.commit()

        result = get_asset_by_hash(session, asset_hash="blake3:def456")
        assert result is not None
        assert result.id == asset.id
        assert result.size_bytes == 200
        assert result.mime_type == "image/png"


class TestUpsertAsset:
    def test_creates_new_asset(self, session: Session):
        asset, created, updated = upsert_asset(
            session,
            asset_hash="blake3:newasset",
            size_bytes=1024,
            mime_type="application/octet-stream",
        )
        session.commit()

        assert created is True
        assert updated is False
        assert asset.hash == "blake3:newasset"
        assert asset.size_bytes == 1024
        assert asset.mime_type == "application/octet-stream"

    def test_returns_existing_asset_without_update(self, session: Session):
        # First insert
        asset1, created1, _ = upsert_asset(
            session,
            asset_hash="blake3:existing",
            size_bytes=500,
            mime_type="text/plain",
        )
        session.commit()

        # Second upsert with same values
        asset2, created2, updated2 = upsert_asset(
            session,
            asset_hash="blake3:existing",
            size_bytes=500,
            mime_type="text/plain",
        )
        session.commit()

        assert created1 is True
        assert created2 is False
        assert updated2 is False
        assert asset1.id == asset2.id

    def test_updates_existing_asset_with_new_values(self, session: Session):
        # First insert with size 0
        asset1, created1, _ = upsert_asset(
            session,
            asset_hash="blake3:toupdate",
            size_bytes=0,
        )
        session.commit()

        # Second upsert with new size and mime type
        asset2, created2, updated2 = upsert_asset(
            session,
            asset_hash="blake3:toupdate",
            size_bytes=2048,
            mime_type="image/png",
        )
        session.commit()

        assert created1 is True
        assert created2 is False
        assert updated2 is True
        assert asset2.size_bytes == 2048
        assert asset2.mime_type == "image/png"

    def test_does_not_update_if_size_zero(self, session: Session):
        # First insert
        asset1, _, _ = upsert_asset(
            session,
            asset_hash="blake3:keepsize",
            size_bytes=1000,
        )
        session.commit()

        # Second upsert with size 0 should not change size
        asset2, created2, updated2 = upsert_asset(
            session,
            asset_hash="blake3:keepsize",
            size_bytes=0,
        )
        session.commit()

        assert created2 is False
        assert updated2 is False
        assert asset2.size_bytes == 1000


class TestBulkInsertAssets:
    def test_inserts_multiple_assets(self, session: Session):
        rows = [
            {"id": str(uuid.uuid4()), "hash": "blake3:bulk1", "size_bytes": 100, "mime_type": "text/plain", "created_at": None},
            {"id": str(uuid.uuid4()), "hash": "blake3:bulk2", "size_bytes": 200, "mime_type": "image/png", "created_at": None},
            {"id": str(uuid.uuid4()), "hash": "blake3:bulk3", "size_bytes": 300, "mime_type": None, "created_at": None},
        ]
        bulk_insert_assets(session, rows)
        session.commit()

        assets = session.query(Asset).all()
        assert len(assets) == 3
        hashes = {a.hash for a in assets}
        assert hashes == {"blake3:bulk1", "blake3:bulk2", "blake3:bulk3"}

    def test_empty_list_is_noop(self, session: Session):
        bulk_insert_assets(session, [])
        session.commit()
        assert session.query(Asset).count() == 0

    def test_handles_large_batch(self, session: Session):
        """Test chunking logic with more rows than MAX_BIND_PARAMS allows."""
        rows = [
            {"id": str(uuid.uuid4()), "hash": f"blake3:large{i}", "size_bytes": i, "mime_type": None, "created_at": None}
            for i in range(200)
        ]
        bulk_insert_assets(session, rows)
        session.commit()

        assert session.query(Asset).count() == 200
