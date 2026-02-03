"""Tests for metadata filtering logic in asset_info queries."""
from sqlalchemy.orm import Session

from app.assets.database.models import Asset, AssetInfo, AssetInfoMeta
from app.assets.database.queries import list_asset_infos_page
from app.assets.database.queries.asset_info import expand_metadata_to_rows
from app.assets.helpers import get_utc_now


def _make_asset(session: Session, hash_val: str) -> Asset:
    asset = Asset(hash=hash_val, size_bytes=1024)
    session.add(asset)
    session.flush()
    return asset


def _make_asset_info(
    session: Session,
    asset: Asset,
    name: str,
    metadata: dict | None = None,
) -> AssetInfo:
    now = get_utc_now()
    info = AssetInfo(
        owner_id="",
        name=name,
        asset_id=asset.id,
        user_metadata=metadata,
        created_at=now,
        updated_at=now,
        last_access_time=now,
    )
    session.add(info)
    session.flush()

    if metadata:
        for key, val in metadata.items():
            for row in expand_metadata_to_rows(key, val):
                meta_row = AssetInfoMeta(
                    asset_info_id=info.id,
                    key=row["key"],
                    ordinal=row.get("ordinal", 0),
                    val_str=row.get("val_str"),
                    val_num=row.get("val_num"),
                    val_bool=row.get("val_bool"),
                    val_json=row.get("val_json"),
                )
                session.add(meta_row)
        session.flush()

    return info


class TestMetadataFilterString:
    def test_filter_by_string_value(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_asset_info(session, asset, "match", {"category": "models"})
        _make_asset_info(session, asset, "nomatch", {"category": "images"})
        session.commit()

        infos, _, total = list_asset_infos_page(session, metadata_filter={"category": "models"})
        assert total == 1
        assert infos[0].name == "match"

    def test_filter_by_string_no_match(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_asset_info(session, asset, "item", {"category": "models"})
        session.commit()

        infos, _, total = list_asset_infos_page(session, metadata_filter={"category": "other"})
        assert total == 0


class TestMetadataFilterNumeric:
    def test_filter_by_int_value(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_asset_info(session, asset, "epoch5", {"epoch": 5})
        _make_asset_info(session, asset, "epoch10", {"epoch": 10})
        session.commit()

        infos, _, total = list_asset_infos_page(session, metadata_filter={"epoch": 5})
        assert total == 1
        assert infos[0].name == "epoch5"

    def test_filter_by_float_value(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_asset_info(session, asset, "high", {"score": 0.95})
        _make_asset_info(session, asset, "low", {"score": 0.5})
        session.commit()

        infos, _, total = list_asset_infos_page(session, metadata_filter={"score": 0.95})
        assert total == 1
        assert infos[0].name == "high"


class TestMetadataFilterBoolean:
    def test_filter_by_true(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_asset_info(session, asset, "active", {"enabled": True})
        _make_asset_info(session, asset, "inactive", {"enabled": False})
        session.commit()

        infos, _, total = list_asset_infos_page(session, metadata_filter={"enabled": True})
        assert total == 1
        assert infos[0].name == "active"

    def test_filter_by_false(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_asset_info(session, asset, "active", {"enabled": True})
        _make_asset_info(session, asset, "inactive", {"enabled": False})
        session.commit()

        infos, _, total = list_asset_infos_page(session, metadata_filter={"enabled": False})
        assert total == 1
        assert infos[0].name == "inactive"


class TestMetadataFilterNull:
    def test_filter_by_null_matches_missing_key(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_asset_info(session, asset, "has_key", {"optional": "value"})
        _make_asset_info(session, asset, "missing_key", {})
        session.commit()

        infos, _, total = list_asset_infos_page(session, metadata_filter={"optional": None})
        assert total == 1
        assert infos[0].name == "missing_key"

    def test_filter_by_null_matches_explicit_null(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_asset_info(session, asset, "explicit_null", {"nullable": None})
        _make_asset_info(session, asset, "has_value", {"nullable": "present"})
        session.commit()

        infos, _, total = list_asset_infos_page(session, metadata_filter={"nullable": None})
        assert total == 1
        assert infos[0].name == "explicit_null"


class TestMetadataFilterList:
    def test_filter_by_list_or(self, session: Session):
        """List values should match ANY of the values (OR)."""
        asset = _make_asset(session, "hash1")
        _make_asset_info(session, asset, "cat_a", {"category": "a"})
        _make_asset_info(session, asset, "cat_b", {"category": "b"})
        _make_asset_info(session, asset, "cat_c", {"category": "c"})
        session.commit()

        infos, _, total = list_asset_infos_page(session, metadata_filter={"category": ["a", "b"]})
        assert total == 2
        names = {i.name for i in infos}
        assert names == {"cat_a", "cat_b"}


class TestMetadataFilterMultipleKeys:
    def test_multiple_keys_and(self, session: Session):
        """Multiple keys should ALL match (AND)."""
        asset = _make_asset(session, "hash1")
        _make_asset_info(session, asset, "match", {"type": "model", "version": 2})
        _make_asset_info(session, asset, "wrong_type", {"type": "config", "version": 2})
        _make_asset_info(session, asset, "wrong_version", {"type": "model", "version": 1})
        session.commit()

        infos, _, total = list_asset_infos_page(
            session, metadata_filter={"type": "model", "version": 2}
        )
        assert total == 1
        assert infos[0].name == "match"


class TestMetadataFilterEmptyDict:
    def test_empty_filter_returns_all(self, session: Session):
        asset = _make_asset(session, "hash1")
        _make_asset_info(session, asset, "a", {"key": "val"})
        _make_asset_info(session, asset, "b", {})
        session.commit()

        infos, _, total = list_asset_infos_page(session, metadata_filter={})
        assert total == 2
