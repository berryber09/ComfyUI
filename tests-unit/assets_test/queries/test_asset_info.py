import pytest
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetInfo
from app.assets.database.queries import (
    asset_info_exists_for_asset_id,
    get_asset_info_by_id,
    list_asset_infos_page,
    fetch_asset_info_asset_and_tags,
    fetch_asset_info_and_asset,
    touch_asset_info_by_id,
    delete_asset_info_by_id,
    set_asset_info_preview,
    ensure_tags_exist,
    add_tags_to_asset_info,
)
from app.assets.helpers import utcnow


def _make_asset(session: Session, hash_val: str | None = None, size: int = 1024) -> Asset:
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
    now = utcnow()
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


class TestAssetInfoExistsForAssetId:
    def test_returns_false_when_no_info(self, session: Session):
        asset = _make_asset(session, "hash1")
        assert asset_info_exists_for_asset_id(session, asset_id=asset.id) is False

    def test_returns_true_when_info_exists(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_asset_info(session, asset)
        assert asset_info_exists_for_asset_id(session, asset_id=asset.id) is True


class TestGetAssetInfoById:
    def test_returns_none_for_nonexistent(self, session: Session):
        assert get_asset_info_by_id(session, asset_info_id="nonexistent") is None

    def test_returns_info(self, session: Session):
        asset = _make_asset(session, "hash1")
        info = _make_asset_info(session, asset, name="myfile.txt")

        result = get_asset_info_by_id(session, asset_info_id=info.id)
        assert result is not None
        assert result.name == "myfile.txt"


class TestListAssetInfosPage:
    def test_empty_db(self, session: Session):
        infos, tag_map, total = list_asset_infos_page(session)
        assert infos == []
        assert tag_map == {}
        assert total == 0

    def test_returns_infos_with_tags(self, session: Session):
        asset = _make_asset(session, "hash1")
        info = _make_asset_info(session, asset, name="test.bin")
        ensure_tags_exist(session, ["alpha", "beta"])
        add_tags_to_asset_info(session, asset_info_id=info.id, tags=["alpha", "beta"])
        session.commit()

        infos, tag_map, total = list_asset_infos_page(session)
        assert len(infos) == 1
        assert infos[0].id == info.id
        assert set(tag_map[info.id]) == {"alpha", "beta"}
        assert total == 1

    def test_name_contains_filter(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_asset_info(session, asset, name="model_v1.safetensors")
        _make_asset_info(session, asset, name="config.json")
        session.commit()

        infos, _, total = list_asset_infos_page(session, name_contains="model")
        assert total == 1
        assert infos[0].name == "model_v1.safetensors"

    def test_owner_visibility(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_asset_info(session, asset, name="public", owner_id="")
        _make_asset_info(session, asset, name="private", owner_id="user1")
        session.commit()

        # Empty owner sees only public
        infos, _, total = list_asset_infos_page(session, owner_id="")
        assert total == 1
        assert infos[0].name == "public"

        # Owner sees both
        infos, _, total = list_asset_infos_page(session, owner_id="user1")
        assert total == 2

    def test_include_tags_filter(self, session: Session):
        asset = _make_asset(session, "hash1")
        info1 = _make_asset_info(session, asset, name="tagged")
        _make_asset_info(session, asset, name="untagged")
        ensure_tags_exist(session, ["wanted"])
        add_tags_to_asset_info(session, asset_info_id=info1.id, tags=["wanted"])
        session.commit()

        infos, _, total = list_asset_infos_page(session, include_tags=["wanted"])
        assert total == 1
        assert infos[0].name == "tagged"

    def test_exclude_tags_filter(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_asset_info(session, asset, name="keep")
        info_exclude = _make_asset_info(session, asset, name="exclude")
        ensure_tags_exist(session, ["bad"])
        add_tags_to_asset_info(session, asset_info_id=info_exclude.id, tags=["bad"])
        session.commit()

        infos, _, total = list_asset_infos_page(session, exclude_tags=["bad"])
        assert total == 1
        assert infos[0].name == "keep"

    def test_sorting(self, session: Session):
        asset = _make_asset(session, "hash1", size=100)
        asset2 = _make_asset(session, "hash2", size=500)
        _make_asset_info(session, asset, name="small")
        _make_asset_info(session, asset2, name="large")
        session.commit()

        infos, _, _ = list_asset_infos_page(session, sort="size", order="desc")
        assert infos[0].name == "large"

        infos, _, _ = list_asset_infos_page(session, sort="name", order="asc")
        assert infos[0].name == "large"


class TestFetchAssetInfoAssetAndTags:
    def test_returns_none_for_nonexistent(self, session: Session):
        result = fetch_asset_info_asset_and_tags(session, "nonexistent")
        assert result is None

    def test_returns_tuple(self, session: Session):
        asset = _make_asset(session, "hash1")
        info = _make_asset_info(session, asset, name="test.bin")
        ensure_tags_exist(session, ["tag1"])
        add_tags_to_asset_info(session, asset_info_id=info.id, tags=["tag1"])
        session.commit()

        result = fetch_asset_info_asset_and_tags(session, info.id)
        assert result is not None
        ret_info, ret_asset, ret_tags = result
        assert ret_info.id == info.id
        assert ret_asset.id == asset.id
        assert ret_tags == ["tag1"]


class TestFetchAssetInfoAndAsset:
    def test_returns_none_for_nonexistent(self, session: Session):
        result = fetch_asset_info_and_asset(session, asset_info_id="nonexistent")
        assert result is None

    def test_returns_tuple(self, session: Session):
        asset = _make_asset(session, "hash1")
        info = _make_asset_info(session, asset)
        session.commit()

        result = fetch_asset_info_and_asset(session, asset_info_id=info.id)
        assert result is not None
        ret_info, ret_asset = result
        assert ret_info.id == info.id
        assert ret_asset.id == asset.id


class TestTouchAssetInfoById:
    def test_updates_last_access_time(self, session: Session):
        asset = _make_asset(session, "hash1")
        info = _make_asset_info(session, asset)
        original_time = info.last_access_time
        session.commit()

        import time
        time.sleep(0.01)

        touch_asset_info_by_id(session, asset_info_id=info.id)
        session.commit()

        session.refresh(info)
        assert info.last_access_time > original_time


class TestDeleteAssetInfoById:
    def test_deletes_existing(self, session: Session):
        asset = _make_asset(session, "hash1")
        info = _make_asset_info(session, asset)
        session.commit()

        result = delete_asset_info_by_id(session, asset_info_id=info.id, owner_id="")
        assert result is True
        assert get_asset_info_by_id(session, asset_info_id=info.id) is None

    def test_returns_false_for_nonexistent(self, session: Session):
        result = delete_asset_info_by_id(session, asset_info_id="nonexistent", owner_id="")
        assert result is False

    def test_respects_owner_visibility(self, session: Session):
        asset = _make_asset(session, "hash1")
        info = _make_asset_info(session, asset, owner_id="user1")
        session.commit()

        result = delete_asset_info_by_id(session, asset_info_id=info.id, owner_id="user2")
        assert result is False
        assert get_asset_info_by_id(session, asset_info_id=info.id) is not None


class TestSetAssetInfoPreview:
    def test_sets_preview(self, session: Session):
        asset = _make_asset(session, "hash1")
        preview_asset = _make_asset(session, "preview_hash")
        info = _make_asset_info(session, asset)
        session.commit()

        set_asset_info_preview(session, asset_info_id=info.id, preview_asset_id=preview_asset.id)
        session.commit()

        session.refresh(info)
        assert info.preview_id == preview_asset.id

    def test_clears_preview(self, session: Session):
        asset = _make_asset(session, "hash1")
        preview_asset = _make_asset(session, "preview_hash")
        info = _make_asset_info(session, asset)
        info.preview_id = preview_asset.id
        session.commit()

        set_asset_info_preview(session, asset_info_id=info.id, preview_asset_id=None)
        session.commit()

        session.refresh(info)
        assert info.preview_id is None

    def test_raises_for_nonexistent_info(self, session: Session):
        with pytest.raises(ValueError, match="not found"):
            set_asset_info_preview(session, asset_info_id="nonexistent", preview_asset_id=None)

    def test_raises_for_nonexistent_preview(self, session: Session):
        asset = _make_asset(session, "hash1")
        info = _make_asset_info(session, asset)
        session.commit()

        with pytest.raises(ValueError, match="Preview Asset"):
            set_asset_info_preview(session, asset_info_id=info.id, preview_asset_id="nonexistent")
