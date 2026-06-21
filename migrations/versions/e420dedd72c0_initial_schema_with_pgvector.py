"""initial schema with pgvector

Revision ID: e420dedd72c0
Revises: 
Create Date: 2026-06-21 21:17:18.230355

"""
from typing import Sequence, Union
from sqlalchemy.dialects import postgresql

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e420dedd72c0'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # --- Extensions ---
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # --- listed_securities ---
    op.create_table(
        "listed_securities",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("cik", sa.Text, nullable=False),
        sa.Column("ticker", sa.Text, nullable=False),
        sa.Column("exchange", sa.Text, nullable=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_listed_securities_cik_uq", "listed_securities", ["cik"], unique=True
    )
    op.create_index(
        "ix_listed_securities_ticker_uq", "listed_securities", ["ticker"], unique=True
    )

    # --- filings ---
    filing_status_enum = postgresql.ENUM(
        "discovered", "downloaded", "parsed", "chunked", "embedded", "failed",
        name="filing_status",
        create_type=False,   # ← critical: don't auto-create from create_table
    )
    filing_status_enum.create(op.get_bind(), checkfirst=True)
    
    op.create_table(
        "filings",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "security_id",
            sa.BigInteger,
            sa.ForeignKey("listed_securities.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("filing_type", sa.Text, nullable=False),
        sa.Column("filed_date", sa.Date, nullable=False),
        sa.Column("period_of_report", sa.Date, nullable=True),
        sa.Column("accession_number", sa.Text, nullable=False),
        sa.Column(
            "status",
            filing_status_enum,
            nullable=False,
            server_default="discovered",
        ),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_filings_accession_uq", "filings", ["accession_number"], unique=True
    )
    op.create_index("ix_filings_security", "filings", ["security_id"])
    op.create_index("ix_filings_status", "filings", ["status"])

    # --- documents ---
    op.create_table(
        "documents",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "filing_id",
            sa.BigInteger,
            sa.ForeignKey("filings.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("primary_document_name", sa.Text, nullable=False),
        sa.Column("document_type", sa.Text, nullable=False),
        sa.Column("original_url", sa.Text, nullable=False),
        sa.Column("local_path", sa.Text, nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_documents_filing", "documents", ["filing_id"])

    # --- sections ---
    op.create_table(
        "sections",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column(
            "document_id",
            sa.BigInteger,
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "section_path",
            sa.ARRAY(sa.Text),
            nullable=False,
        ),
        sa.Column("order", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_sections_document", "sections", ["document_id"])

    # --- chunks ---
    # We create this with raw SQL because pgvector's `vector` type isn't in
    # vanilla SQLAlchemy. Cleaner than adding the pgvector SQLAlchemy plugin
    # just for one column.
    op.execute(
        """
        CREATE TABLE chunks (
            id              BIGSERIAL PRIMARY KEY,
            section_id      BIGINT NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
            content         TEXT NOT NULL,
            chunk_index     INTEGER NOT NULL,
            token_count     INTEGER NOT NULL,
            embedding       vector(1536),
            -- Denormalized for retrieval-time filtering performance.
            ticker          TEXT NOT NULL,
            filed_date      DATE NOT NULL,
            filing_type     TEXT NOT NULL,
            section_path    TEXT[] NOT NULL,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )

    # --- chunk indexes ---
    op.create_index("ix_chunks_section", "chunks", ["section_id"])
    op.create_index("ix_chunks_ticker", "chunks", ["ticker"])
    op.create_index("ix_chunks_filed_date", "chunks", ["filed_date"])
    op.create_index("ix_chunks_filing_type", "chunks", ["filing_type"])
    # GIN for array containment queries: WHERE 'Risk Factors' = ANY(section_path)
    op.execute(
        "CREATE INDEX ix_chunks_section_path ON chunks USING gin (section_path)"
    )
    # HNSW for cosine similarity. ef_construction/m at defaults; tune later.
    op.execute(
        """
        CREATE INDEX ix_chunks_embedding
        ON chunks USING hnsw (embedding vector_cosine_ops)
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS ix_chunks_embedding")
    op.execute("DROP INDEX IF EXISTS ix_chunks_section_path")
    op.drop_index("ix_chunks_filing_type", table_name="chunks")
    op.drop_index("ix_chunks_filed_date", table_name="chunks")
    op.drop_index("ix_chunks_ticker", table_name="chunks")
    op.drop_index("ix_chunks_section", table_name="chunks")
    op.execute("DROP TABLE IF EXISTS chunks")

    op.drop_index("ix_sections_document", table_name="sections")
    op.drop_table("sections")

    op.drop_index("ix_documents_filing", table_name="documents")
    op.drop_table("documents")

    op.drop_index("ix_filings_status", table_name="filings")
    op.drop_index("ix_filings_security", table_name="filings")
    op.drop_index("ix_filings_accession_uq", table_name="filings")
    op.drop_table("filings")
    
    filing_status_enum = postgresql.ENUM(name="filing_status")
    filing_status_enum.drop(op.get_bind(), checkfirst=True)

    op.drop_index("ix_listed_securities_ticker_uq", table_name="listed_securities")
    op.drop_index("ix_listed_securities_cik_uq", table_name="listed_securities")
    op.drop_table("listed_securities")
