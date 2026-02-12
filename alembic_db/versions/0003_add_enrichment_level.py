"""
Add enrichment_level column to asset_cache_state for phased scanning

Level 0: Stub record (path, size, mtime only)
Level 1: Metadata extracted (safetensors header, mime type)
Level 2: Hash computed (blake3)

Revision ID: 0003_add_enrichment_level
Revises: 0002_add_is_missing
Create Date: 2025-02-10 00:00:00
"""

from alembic import op
import sqlalchemy as sa

revision = "0003_add_enrichment_level"
down_revision = "0002_add_is_missing"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "asset_cache_state",
        sa.Column(
            "enrichment_level",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.create_index(
        "ix_asset_cache_state_enrichment_level",
        "asset_cache_state",
        ["enrichment_level"],
    )
    # Treat existing records as fully enriched (level 1 = metadata done)
    # since they were created with the old scanner that extracted metadata
    op.execute("UPDATE asset_cache_state SET enrichment_level = 1")


def downgrade() -> None:
    op.drop_index("ix_asset_cache_state_enrichment_level", table_name="asset_cache_state")
    op.drop_column("asset_cache_state", "enrichment_level")
