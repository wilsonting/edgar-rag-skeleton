"""unique chunk per (section_id, chunk_index)

Nothing stopped a re-chunk from inserting a second copy of every chunk: a run
that died after IngestionService._chunk committed its insert, but before the
filing was marked CHUNKED, re-chunked on the next run and appended. _chunk
now clears a document's chunks first; this index makes any duplicate fail
loudly instead of silently doubling a filing's weight in retrieval.

Revision ID: 7c3e9a1f5b2d
Revises: d8cd6a0f2f2f
Create Date: 2026-09-11

"""
from typing import Sequence, Union

from alembic import op


revision: str = '7c3e9a1f5b2d'
down_revision: Union[str, Sequence[str], None] = 'd8cd6a0f2f2f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ux_chunks_section_chunk_index",
        "chunks",
        ["section_id", "chunk_index"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ux_chunks_section_chunk_index", table_name="chunks")
