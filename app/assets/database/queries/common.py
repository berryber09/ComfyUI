"""Shared utilities for database query modules."""

from typing import Iterable

import sqlalchemy as sa

from app.assets.database.models import AssetReference

MAX_BIND_PARAMS = 800


def calculate_rows_per_statement(cols: int) -> int:
    """Calculate how many rows can fit in one statement given column count."""
    return max(1, MAX_BIND_PARAMS // max(1, cols))


def iter_chunks(seq, n: int):
    """Yield successive n-sized chunks from seq."""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def iter_row_chunks(rows: list[dict], cols_per_row: int) -> Iterable[list[dict]]:
    """Yield chunks of rows sized to fit within bind param limits."""
    if not rows:
        return
    rows_per_stmt = calculate_rows_per_statement(cols_per_row)
    for i in range(0, len(rows), rows_per_stmt):
        yield rows[i : i + rows_per_stmt]


def build_visible_owner_clause(owner_id: str) -> sa.sql.ClauseElement:
    """Build owner visibility predicate for reads.

    Owner-less rows are visible to everyone.
    """
    owner_id = (owner_id or "").strip()
    if owner_id == "":
        return AssetReference.owner_id == ""
    return AssetReference.owner_id.in_(["", owner_id])
