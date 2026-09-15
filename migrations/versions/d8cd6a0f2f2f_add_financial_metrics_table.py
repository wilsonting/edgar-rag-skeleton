"""add financial_metrics table

Revision ID: d8cd6a0f2f2f
Revises: 25b4b18dcb28
Create Date: 2026-07-11 08:42:44.279222

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'd8cd6a0f2f2f'
down_revision: Union[str, Sequence[str], None] = '25b4b18dcb28'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "financial_metrics",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("ticker", sa.String, nullable=False),
        sa.Column("filing_type", sa.String, nullable=False),  # 10-Q, 10-K
        sa.Column("fiscal_period", sa.String, nullable=False), # "Q1 2026"
        sa.Column("filed_date", sa.Date, nullable=False),
        sa.Column("revenue", sa.Numeric, nullable=True),
        sa.Column("gross_margin_pct", sa.Numeric, nullable=True),
        sa.Column("gaap_net_income", sa.Numeric, nullable=True),
        sa.Column("free_cash_flow", sa.Numeric, nullable=True),
        sa.Column("sbc_pct_of_revenue", sa.Numeric, nullable=True),
        sa.Column("net_dollar_retention", sa.Numeric, nullable=True),
        sa.Column("extraction_confidence", sa.String, nullable=False), # stated/computed/not_disclosed
        sa.Column("source_citations", postgresql.JSONB, nullable=False),
        sa.Column("extracted_at", sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint("ticker", "filing_type", "fiscal_period", name="uq_metric_period"),
    )


def downgrade() -> None:
    # The constraint lives on financial_metrics, not on a table called
    # "ticker" — this downgrade could never have run.
    op.drop_constraint(
        constraint_name="uq_metric_period",
        table_name="financial_metrics",
        type_="unique",
    )
    op.drop_table("financial_metrics")
