"""
Drop unique constraint on assets_info (asset_id, owner_id, name)

Allow multiple files with the same name to reference the same asset.

Revision ID: 0004_drop_asset_info_unique
Revises: 0003_add_enrichment_level
Create Date: 2025-02-11 00:00:00
"""

from alembic import op
import sqlalchemy as sa

revision = "0004_drop_asset_info_unique"
down_revision = "0003_add_enrichment_level"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("assets_info") as batch_op:
        batch_op.drop_constraint(
            "uq_assets_info_asset_owner_name", type_="unique"
        )


def downgrade() -> None:
    with op.batch_alter_table("assets_info") as batch_op:
        batch_op.create_unique_constraint(
            "uq_assets_info_asset_owner_name",
            ["asset_id", "owner_id", "name"],
        )
