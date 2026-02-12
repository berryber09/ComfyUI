"""
Merge AssetInfo and AssetCacheState into unified asset_references table.

This migration:
1. Creates asset_references table with combined columns
2. Creates asset_reference_tags and asset_reference_meta tables
3. Migrates data from assets_info and asset_cache_state, merging where unambiguous
4. Migrates tags and metadata
5. Drops old tables

Revision ID: 0005_merge_to_asset_references
Revises: 0004_drop_asset_info_unique_constraint
Create Date: 2025-02-11
"""
# ruff: noqa: E501

import os
import uuid
from datetime import datetime

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

revision = "0005_merge_to_asset_references"
down_revision = "0004_drop_asset_info_unique_constraint"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # Step 1: Create asset_references table
    op.create_table(
        "asset_references",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "asset_id",
            sa.String(length=36),
            sa.ForeignKey("assets.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # From AssetCacheState
        sa.Column("file_path", sa.Text(), nullable=True),
        sa.Column("mtime_ns", sa.BigInteger(), nullable=True),
        sa.Column(
            "needs_verify",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "is_missing", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column("enrichment_level", sa.Integer(), nullable=False, server_default="0"),
        # From AssetInfo
        sa.Column("owner_id", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("name", sa.String(length=512), nullable=False),
        sa.Column(
            "preview_id",
            sa.String(length=36),
            sa.ForeignKey("assets.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("user_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=False), nullable=False),
        sa.Column("last_access_time", sa.DateTime(timezone=False), nullable=False),
        # Constraints
        sa.CheckConstraint(
            "(mtime_ns IS NULL) OR (mtime_ns >= 0)", name="ck_ar_mtime_nonneg"
        ),
        sa.CheckConstraint(
            "enrichment_level >= 0 AND enrichment_level <= 2",
            name="ck_ar_enrichment_level_range",
        ),
    )

    # Create unique index on file_path where not null (partial unique).
    # SQLite UNIQUE on nullable columns works as expected.
    op.create_index(
        "uq_asset_references_file_path",
        "asset_references",
        ["file_path"],
        unique=True,
    )
    op.create_index("ix_asset_references_asset_id", "asset_references", ["asset_id"])
    op.create_index("ix_asset_references_owner_id", "asset_references", ["owner_id"])
    op.create_index("ix_asset_references_name", "asset_references", ["name"])
    op.create_index(
        "ix_asset_references_is_missing", "asset_references", ["is_missing"]
    )
    op.create_index(
        "ix_asset_references_enrichment_level", "asset_references", ["enrichment_level"]
    )
    op.create_index(
        "ix_asset_references_created_at", "asset_references", ["created_at"]
    )
    op.create_index(
        "ix_asset_references_last_access_time", "asset_references", ["last_access_time"]
    )
    op.create_index(
        "ix_asset_references_owner_name", "asset_references", ["owner_id", "name"]
    )

    # Step 2: Create asset_reference_tags table
    op.create_table(
        "asset_reference_tags",
        sa.Column(
            "asset_reference_id",
            sa.String(length=36),
            sa.ForeignKey("asset_references.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "tag_name",
            sa.String(length=512),
            sa.ForeignKey("tags.name", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "origin", sa.String(length=32), nullable=False, server_default="manual"
        ),
        sa.Column("added_at", sa.DateTime(timezone=False), nullable=False),
        sa.PrimaryKeyConstraint(
            "asset_reference_id", "tag_name", name="pk_asset_reference_tags"
        ),
    )
    op.create_index(
        "ix_asset_reference_tags_tag_name", "asset_reference_tags", ["tag_name"]
    )
    op.create_index(
        "ix_asset_reference_tags_asset_reference_id",
        "asset_reference_tags",
        ["asset_reference_id"],
    )

    # Step 3: Create asset_reference_meta table
    op.create_table(
        "asset_reference_meta",
        sa.Column(
            "asset_reference_id",
            sa.String(length=36),
            sa.ForeignKey("asset_references.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("key", sa.String(length=256), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("val_str", sa.String(length=2048), nullable=True),
        sa.Column("val_num", sa.Numeric(38, 10), nullable=True),
        sa.Column("val_bool", sa.Boolean(), nullable=True),
        sa.Column("val_json", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint(
            "asset_reference_id", "key", "ordinal", name="pk_asset_reference_meta"
        ),
    )
    op.create_index("ix_asset_reference_meta_key", "asset_reference_meta", ["key"])
    op.create_index(
        "ix_asset_reference_meta_key_val_str",
        "asset_reference_meta",
        ["key", "val_str"],
    )
    op.create_index(
        "ix_asset_reference_meta_key_val_num",
        "asset_reference_meta",
        ["key", "val_num"],
    )
    op.create_index(
        "ix_asset_reference_meta_key_val_bool",
        "asset_reference_meta",
        ["key", "val_bool"],
    )

    # Step 4: Migrate data
    # Create mapping from cache_state to info that should absorb it.
    # Merge when: same asset_id AND exactly one cache_state AND basename == name
    now = datetime.utcnow().isoformat()

    # Find unambiguous matches: assets_info rows that have exactly one matching cache_state
    # where basename(file_path) == name AND same asset_id
    # We'll do this in Python for clarity and SQLite compatibility

    # Get all assets_info rows
    info_rows = conn.execute(
        text("""
            SELECT id, owner_id, name, asset_id, preview_id, user_metadata,
                   created_at, updated_at, last_access_time
            FROM assets_info
        """)
    ).fetchall()

    # Get all asset_cache_state rows
    cache_rows = conn.execute(
        text("""
            SELECT id, asset_id, file_path, mtime_ns, needs_verify, is_missing, enrichment_level
            FROM asset_cache_state
        """)
    ).fetchall()

    # Build mapping: asset_id -> list of cache_state rows
    cache_by_asset: dict = {}
    for row in cache_rows:
        (
            cache_id,
            asset_id,
            file_path,
            mtime_ns,
            needs_verify,
            is_missing,
            enrichment_level,
        ) = row
        if asset_id not in cache_by_asset:
            cache_by_asset[asset_id] = []
        cache_by_asset[asset_id].append(
            {
                "cache_id": cache_id,
                "file_path": file_path,
                "mtime_ns": mtime_ns,
                "needs_verify": needs_verify,
                "is_missing": is_missing,
                "enrichment_level": enrichment_level,
            }
        )

    # Track which cache_states get merged (so we don't insert them separately)
    merged_cache_ids: set = set()
    # Track info_id -> cache_data for merged rows
    info_to_cache: dict = {}

    for info_row in info_rows:
        (
            info_id,
            owner_id,
            name,
            asset_id,
            preview_id,
            user_metadata,
            created_at,
            updated_at,
            last_access,
        ) = info_row
        caches = cache_by_asset.get(asset_id, [])

        # Only merge if exactly one cache_state AND basename matches
        if len(caches) == 1:
            cache = caches[0]
            basename = os.path.basename(cache["file_path"])
            if basename == name:
                merged_cache_ids.add(cache["cache_id"])
                info_to_cache[info_id] = cache

    # Insert merged and non-merged assets_info rows into asset_references
    for info_row in info_rows:
        (
            info_id,
            owner_id,
            name,
            asset_id,
            preview_id,
            user_metadata,
            created_at,
            updated_at,
            last_access,
        ) = info_row

        cache = info_to_cache.get(info_id)
        if cache:
            # Merged row: has file_path and cache data
            conn.execute(
                text("""
                    INSERT INTO asset_references (
                        id, asset_id, file_path, mtime_ns, needs_verify, is_missing,
                        enrichment_level, owner_id, name, preview_id, user_metadata,
                        created_at, updated_at, last_access_time
                    ) VALUES (
                        :id, :asset_id, :file_path, :mtime_ns, :needs_verify, :is_missing,
                        :enrichment_level, :owner_id, :name, :preview_id, :user_metadata,
                        :created_at, :updated_at, :last_access_time
                    )
                """),
                {
                    "id": info_id,
                    "asset_id": asset_id,
                    "file_path": cache["file_path"],
                    "mtime_ns": cache["mtime_ns"],
                    "needs_verify": cache["needs_verify"],
                    "is_missing": cache["is_missing"],
                    "enrichment_level": cache["enrichment_level"],
                    "owner_id": owner_id or "",
                    "name": name,
                    "preview_id": preview_id,
                    "user_metadata": user_metadata,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "last_access_time": last_access,
                },
            )
        else:
            # Non-merged row: no file_path
            conn.execute(
                text("""
                    INSERT INTO asset_references (
                        id, asset_id, file_path, mtime_ns, needs_verify, is_missing,
                        enrichment_level, owner_id, name, preview_id, user_metadata,
                        created_at, updated_at, last_access_time
                    ) VALUES (
                        :id, :asset_id, NULL, NULL, false, false, 0,
                        :owner_id, :name, :preview_id, :user_metadata,
                        :created_at, :updated_at, :last_access_time
                    )
                """),
                {
                    "id": info_id,
                    "asset_id": asset_id,
                    "owner_id": owner_id or "",
                    "name": name,
                    "preview_id": preview_id,
                    "user_metadata": user_metadata,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "last_access_time": last_access,
                },
            )

    # Insert remaining (non-merged) cache_state rows as new asset_references
    for cache_row in cache_rows:
        (
            cache_id,
            asset_id,
            file_path,
            mtime_ns,
            needs_verify,
            is_missing,
            enrichment_level,
        ) = cache_row
        if cache_id in merged_cache_ids:
            continue

        new_id = str(uuid.uuid4())
        basename = os.path.basename(file_path) if file_path else "unknown"

        conn.execute(
            text("""
                INSERT INTO asset_references (
                    id, asset_id, file_path, mtime_ns, needs_verify, is_missing,
                    enrichment_level, owner_id, name, preview_id, user_metadata,
                    created_at, updated_at, last_access_time
                ) VALUES (
                    :id, :asset_id, :file_path, :mtime_ns, :needs_verify, :is_missing,
                    :enrichment_level, '', :name, NULL, NULL,
                    :now, :now, :now
                )
            """),
            {
                "id": new_id,
                "asset_id": asset_id,
                "file_path": file_path,
                "mtime_ns": mtime_ns,
                "needs_verify": needs_verify,
                "is_missing": is_missing,
                "enrichment_level": enrichment_level,
                "name": basename,
                "now": now,
            },
        )

    # Step 5: Migrate tags (asset_info_id maps directly to asset_reference_id since we reused IDs)
    conn.execute(
        text("""
            INSERT INTO asset_reference_tags (asset_reference_id, tag_name, origin, added_at)
            SELECT asset_info_id, tag_name, origin, added_at
            FROM asset_info_tags
            WHERE asset_info_id IN (SELECT id FROM asset_references)
        """)
    )

    # Step 6: Migrate metadata
    conn.execute(
        text("""
            INSERT INTO asset_reference_meta (asset_reference_id, key, ordinal, val_str, val_num, val_bool, val_json)
            SELECT asset_info_id, key, ordinal, val_str, val_num, val_bool, val_json
            FROM asset_info_meta
            WHERE asset_info_id IN (SELECT id FROM asset_references)
        """)
    )

    # Step 7: Drop old tables
    op.drop_index("ix_asset_info_meta_key_val_bool", table_name="asset_info_meta")
    op.drop_index("ix_asset_info_meta_key_val_num", table_name="asset_info_meta")
    op.drop_index("ix_asset_info_meta_key_val_str", table_name="asset_info_meta")
    op.drop_index("ix_asset_info_meta_key", table_name="asset_info_meta")
    op.drop_table("asset_info_meta")

    op.drop_index("ix_asset_info_tags_asset_info_id", table_name="asset_info_tags")
    op.drop_index("ix_asset_info_tags_tag_name", table_name="asset_info_tags")
    op.drop_table("asset_info_tags")

    op.drop_index("ix_asset_cache_state_asset_id", table_name="asset_cache_state")
    op.drop_index("ix_asset_cache_state_file_path", table_name="asset_cache_state")
    op.drop_index("ix_asset_cache_state_is_missing", table_name="asset_cache_state")
    op.drop_index(
        "ix_asset_cache_state_enrichment_level", table_name="asset_cache_state"
    )
    op.drop_table("asset_cache_state")

    op.drop_index("ix_assets_info_owner_name", table_name="assets_info")
    op.drop_index("ix_assets_info_last_access_time", table_name="assets_info")
    op.drop_index("ix_assets_info_created_at", table_name="assets_info")
    op.drop_index("ix_assets_info_name", table_name="assets_info")
    op.drop_index("ix_assets_info_asset_id", table_name="assets_info")
    op.drop_index("ix_assets_info_owner_id", table_name="assets_info")
    op.drop_table("assets_info")


def downgrade() -> None:
    # This is a complex migration - downgrade would require careful data splitting
    # For safety, we don't support automatic downgrade
    raise NotImplementedError(
        "Downgrade from 0005_merge_to_asset_references is not supported. "
        "Please restore from backup if needed."
    )
