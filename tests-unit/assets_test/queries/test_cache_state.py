"""Tests for cache_state query functions."""
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetCacheState
from app.assets.database.queries import (
    list_cache_states_by_asset_id,
    pick_best_live_path,
)


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
