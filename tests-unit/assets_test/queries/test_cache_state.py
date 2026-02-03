"""Tests for cache_state query functions."""
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetCacheState, AssetInfo
from app.assets.database.queries import (
    list_cache_states_by_asset_id,
    upsert_cache_state,
    delete_cache_states_outside_prefixes,
    get_orphaned_seed_asset_ids,
    delete_assets_by_ids,
    get_cache_states_for_prefixes,
    bulk_set_needs_verify,
    delete_cache_states_by_ids,
    delete_orphaned_seed_asset,
    bulk_insert_cache_states_ignore_conflicts,
    get_cache_states_by_paths_and_asset_ids,
)
from app.assets.helpers import pick_best_live_path, get_utc_now


def _make_asset(session: Session, hash_val: str | None = None, size: int = 1024) -> Asset:
    asset = Asset(hash=hash_val, size_bytes=size)
    session.add(asset)
    session.flush()
    return asset


def _make_cache_state(
    session: Session,
    asset: Asset,
    file_path: str,
    mtime_ns: int | None = None,
    needs_verify: bool = False,
) -> AssetCacheState:
    state = AssetCacheState(
        asset_id=asset.id,
        file_path=file_path,
        mtime_ns=mtime_ns,
        needs_verify=needs_verify,
    )
    session.add(state)
    session.flush()
    return state


class TestListCacheStatesByAssetId:
    def test_returns_empty_for_no_states(self, session: Session):
        asset = _make_asset(session, "hash1")
        states = list_cache_states_by_asset_id(session, asset_id=asset.id)
        assert list(states) == []

    def test_returns_states_for_asset(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_cache_state(session, asset, "/path/a.bin")
        _make_cache_state(session, asset, "/path/b.bin")
        session.commit()

        states = list_cache_states_by_asset_id(session, asset_id=asset.id)
        paths = [s.file_path for s in states]
        assert set(paths) == {"/path/a.bin", "/path/b.bin"}

    def test_does_not_return_other_assets_states(self, session: Session):
        asset1 = _make_asset(session, "hash1")
        asset2 = _make_asset(session, "hash2")
        _make_cache_state(session, asset1, "/path/asset1.bin")
        _make_cache_state(session, asset2, "/path/asset2.bin")
        session.commit()

        states = list_cache_states_by_asset_id(session, asset_id=asset1.id)
        paths = [s.file_path for s in states]
        assert paths == ["/path/asset1.bin"]


class TestPickBestLivePath:
    def test_returns_empty_for_empty_list(self):
        result = pick_best_live_path([])
        assert result == ""

    def test_returns_empty_when_no_files_exist(self, session: Session):
        asset = _make_asset(session, "hash1")
        state = _make_cache_state(session, asset, "/nonexistent/path.bin")
        session.commit()

        result = pick_best_live_path([state])
        assert result == ""

    def test_prefers_verified_path(self, session: Session, tmp_path):
        """needs_verify=False should be preferred."""
        asset = _make_asset(session, "hash1")

        verified_file = tmp_path / "verified.bin"
        verified_file.write_bytes(b"data")

        unverified_file = tmp_path / "unverified.bin"
        unverified_file.write_bytes(b"data")

        state_verified = _make_cache_state(
            session, asset, str(verified_file), needs_verify=False
        )
        state_unverified = _make_cache_state(
            session, asset, str(unverified_file), needs_verify=True
        )
        session.commit()

        states = [state_unverified, state_verified]
        result = pick_best_live_path(states)
        assert result == str(verified_file)

    def test_falls_back_to_existing_unverified(self, session: Session, tmp_path):
        """If all states need verification, return first existing path."""
        asset = _make_asset(session, "hash1")

        existing_file = tmp_path / "exists.bin"
        existing_file.write_bytes(b"data")

        state = _make_cache_state(session, asset, str(existing_file), needs_verify=True)
        session.commit()

        result = pick_best_live_path([state])
        assert result == str(existing_file)


class TestPickBestLivePathWithMocking:
    def test_handles_missing_file_path_attr(self):
        """Gracefully handle states with None file_path."""

        class MockState:
            file_path = None
            needs_verify = False

        result = pick_best_live_path([MockState()])
        assert result == ""


class TestUpsertCacheState:
    def test_creates_new_state(self, session: Session):
        asset = _make_asset(session, "hash1")
        created, updated = upsert_cache_state(
            session, asset_id=asset.id, file_path="/new/path.bin", mtime_ns=12345
        )
        session.commit()

        assert created is True
        assert updated is False
        state = session.query(AssetCacheState).filter_by(file_path="/new/path.bin").one()
        assert state.asset_id == asset.id
        assert state.mtime_ns == 12345

    def test_returns_existing_without_update(self, session: Session):
        asset = _make_asset(session, "hash1")
        upsert_cache_state(session, asset_id=asset.id, file_path="/existing.bin", mtime_ns=100)
        session.commit()

        created, updated = upsert_cache_state(
            session, asset_id=asset.id, file_path="/existing.bin", mtime_ns=100
        )
        session.commit()

        assert created is False
        assert updated is False

    def test_updates_existing_with_new_mtime(self, session: Session):
        asset = _make_asset(session, "hash1")
        upsert_cache_state(session, asset_id=asset.id, file_path="/update.bin", mtime_ns=100)
        session.commit()

        created, updated = upsert_cache_state(
            session, asset_id=asset.id, file_path="/update.bin", mtime_ns=200
        )
        session.commit()

        assert created is False
        assert updated is True
        state = session.query(AssetCacheState).filter_by(file_path="/update.bin").one()
        assert state.mtime_ns == 200


class TestDeleteCacheStatesOutsidePrefixes:
    def test_deletes_states_outside_prefixes(self, session: Session, tmp_path):
        asset = _make_asset(session, "hash1")
        valid_dir = tmp_path / "valid"
        valid_dir.mkdir()
        invalid_dir = tmp_path / "invalid"
        invalid_dir.mkdir()

        valid_path = str(valid_dir / "file.bin")
        invalid_path = str(invalid_dir / "file.bin")

        _make_cache_state(session, asset, valid_path)
        _make_cache_state(session, asset, invalid_path)
        session.commit()

        deleted = delete_cache_states_outside_prefixes(session, [str(valid_dir)])
        session.commit()

        assert deleted == 1
        remaining = session.query(AssetCacheState).all()
        assert len(remaining) == 1
        assert remaining[0].file_path == valid_path

    def test_empty_prefixes_deletes_nothing(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_cache_state(session, asset, "/some/path.bin")
        session.commit()

        deleted = delete_cache_states_outside_prefixes(session, [])

        assert deleted == 0


class TestGetOrphanedSeedAssetIds:
    def test_returns_orphaned_seed_assets(self, session: Session):
        # Seed asset (hash=None) with no cache states
        orphan = _make_asset(session, hash_val=None)
        # Seed asset with cache state (not orphaned)
        with_state = _make_asset(session, hash_val=None)
        _make_cache_state(session, with_state, "/has/state.bin")
        # Regular asset (hash not None) - should not be returned
        _make_asset(session, hash_val="blake3:regular")
        session.commit()

        orphaned = get_orphaned_seed_asset_ids(session)

        assert orphan.id in orphaned
        assert with_state.id not in orphaned


class TestDeleteAssetsByIds:
    def test_deletes_assets_and_infos(self, session: Session):
        asset = _make_asset(session, "hash1")
        now = get_utc_now()
        info = AssetInfo(
            owner_id="", name="test", asset_id=asset.id,
            created_at=now, updated_at=now, last_access_time=now
        )
        session.add(info)
        session.commit()

        deleted = delete_assets_by_ids(session, [asset.id])
        session.commit()

        assert deleted == 1
        assert session.query(Asset).count() == 0
        assert session.query(AssetInfo).count() == 0

    def test_empty_list_deletes_nothing(self, session: Session):
        _make_asset(session, "hash1")
        session.commit()

        deleted = delete_assets_by_ids(session, [])

        assert deleted == 0
        assert session.query(Asset).count() == 1


class TestGetCacheStatesForPrefixes:
    def test_returns_states_matching_prefix(self, session: Session, tmp_path):
        asset = _make_asset(session, "hash1")
        dir1 = tmp_path / "dir1"
        dir1.mkdir()
        dir2 = tmp_path / "dir2"
        dir2.mkdir()

        path1 = str(dir1 / "file.bin")
        path2 = str(dir2 / "file.bin")

        _make_cache_state(session, asset, path1, mtime_ns=100)
        _make_cache_state(session, asset, path2, mtime_ns=200)
        session.commit()

        rows = get_cache_states_for_prefixes(session, [str(dir1)])

        assert len(rows) == 1
        assert rows[0].file_path == path1

    def test_empty_prefixes_returns_empty(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_cache_state(session, asset, "/some/path.bin")
        session.commit()

        rows = get_cache_states_for_prefixes(session, [])

        assert rows == []


class TestBulkSetNeedsVerify:
    def test_sets_needs_verify_flag(self, session: Session):
        asset = _make_asset(session, "hash1")
        state1 = _make_cache_state(session, asset, "/path1.bin", needs_verify=False)
        state2 = _make_cache_state(session, asset, "/path2.bin", needs_verify=False)
        session.commit()

        updated = bulk_set_needs_verify(session, [state1.id, state2.id], True)
        session.commit()

        assert updated == 2
        session.refresh(state1)
        session.refresh(state2)
        assert state1.needs_verify is True
        assert state2.needs_verify is True

    def test_empty_list_updates_nothing(self, session: Session):
        updated = bulk_set_needs_verify(session, [], True)
        assert updated == 0


class TestDeleteCacheStatesByIds:
    def test_deletes_states_by_id(self, session: Session):
        asset = _make_asset(session, "hash1")
        state1 = _make_cache_state(session, asset, "/path1.bin")
        _make_cache_state(session, asset, "/path2.bin")
        session.commit()

        deleted = delete_cache_states_by_ids(session, [state1.id])
        session.commit()

        assert deleted == 1
        assert session.query(AssetCacheState).count() == 1

    def test_empty_list_deletes_nothing(self, session: Session):
        deleted = delete_cache_states_by_ids(session, [])
        assert deleted == 0


class TestDeleteOrphanedSeedAsset:
    def test_deletes_seed_asset_and_infos(self, session: Session):
        asset = _make_asset(session, hash_val=None)
        now = get_utc_now()
        info = AssetInfo(
            owner_id="", name="test", asset_id=asset.id,
            created_at=now, updated_at=now, last_access_time=now
        )
        session.add(info)
        session.commit()

        deleted = delete_orphaned_seed_asset(session, asset.id)
        session.commit()

        assert deleted is True
        assert session.query(Asset).count() == 0
        assert session.query(AssetInfo).count() == 0

    def test_returns_false_for_nonexistent(self, session: Session):
        deleted = delete_orphaned_seed_asset(session, "nonexistent-id")
        assert deleted is False


class TestBulkInsertCacheStatesIgnoreConflicts:
    def test_inserts_multiple_states(self, session: Session):
        asset = _make_asset(session, "hash1")
        rows = [
            {"asset_id": asset.id, "file_path": "/bulk1.bin", "mtime_ns": 100},
            {"asset_id": asset.id, "file_path": "/bulk2.bin", "mtime_ns": 200},
        ]
        bulk_insert_cache_states_ignore_conflicts(session, rows)
        session.commit()

        assert session.query(AssetCacheState).count() == 2

    def test_ignores_conflicts(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_cache_state(session, asset, "/existing.bin", mtime_ns=100)
        session.commit()

        rows = [
            {"asset_id": asset.id, "file_path": "/existing.bin", "mtime_ns": 999},
            {"asset_id": asset.id, "file_path": "/new.bin", "mtime_ns": 200},
        ]
        bulk_insert_cache_states_ignore_conflicts(session, rows)
        session.commit()

        assert session.query(AssetCacheState).count() == 2
        existing = session.query(AssetCacheState).filter_by(file_path="/existing.bin").one()
        assert existing.mtime_ns == 100  # Original value preserved

    def test_empty_list_is_noop(self, session: Session):
        bulk_insert_cache_states_ignore_conflicts(session, [])
        assert session.query(AssetCacheState).count() == 0


class TestGetCacheStatesByPathsAndAssetIds:
    def test_returns_matching_paths(self, session: Session):
        asset1 = _make_asset(session, "hash1")
        asset2 = _make_asset(session, "hash2")

        _make_cache_state(session, asset1, "/path1.bin")
        _make_cache_state(session, asset2, "/path2.bin")
        session.commit()

        path_to_asset = {
            "/path1.bin": asset1.id,
            "/path2.bin": asset2.id,
        }
        winners = get_cache_states_by_paths_and_asset_ids(session, path_to_asset)

        assert winners == {"/path1.bin", "/path2.bin"}

    def test_excludes_non_matching_asset_ids(self, session: Session):
        asset1 = _make_asset(session, "hash1")
        asset2 = _make_asset(session, "hash2")

        _make_cache_state(session, asset1, "/path1.bin")
        session.commit()

        # Path exists but with different asset_id
        path_to_asset = {"/path1.bin": asset2.id}
        winners = get_cache_states_by_paths_and_asset_ids(session, path_to_asset)

        assert winners == set()

    def test_empty_dict_returns_empty(self, session: Session):
        winners = get_cache_states_by_paths_and_asset_ids(session, {})
        assert winners == set()
