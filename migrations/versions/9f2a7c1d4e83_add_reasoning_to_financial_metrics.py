"""add reasoning column to financial_metrics

The extractor has always produced a `reasoning` string ("forces the model to
show its work — cheap sanity check"), and the repository's row type has
always declared it, but the table never had the column: `upsert` silently
dropped it, and all three read methods listed it in their SELECT, so every
one of them raised UndefinedColumn. Nothing called them, so the mismatch
was invisible until the read side was needed.

Nullable with a '' default: rows written before this migration have no
reasoning to recover, and an empty string is the honest representation of
that — not a NOT NULL constraint that would reject them.

Revision ID: 9f2a7c1d4e83
Revises: 7c3e9a1f5b2d
Create Date: 2026-09-12

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '9f2a7c1d4e83'
down_revision: Union[str, Sequence[str], None] = '7c3e9a1f5b2d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "financial_metrics",
        sa.Column("reasoning", sa.Text, nullable=True, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("financial_metrics", "reasoning")
