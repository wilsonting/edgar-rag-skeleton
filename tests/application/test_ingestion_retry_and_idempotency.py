"""Ingestion robustness (docs/code_review.md, Medium #7).

- FAILED was terminal: nothing reset it, so one transient error removed a
  filing from the index for good. `retry_failed` now sends it back to
  DISCOVERED through the domain state machine.
- `_chunk` appended: a run that died after inserting chunks but before the
  filing was marked CHUNKED inserted a second copy of every chunk on the
  next run. It now replaces a document's chunks.
"""

from __future__ import annotations

import asyncio
from datetime import date
from types import SimpleNamespace

from app.application.ingestion_service import IngestionService
from app.domain.filing import Filing
from app.domain.listed_security import ListedSecurity
from app.domain.values import FilingStatus

ACCESSION = "0001467373-25-000217"


class _Edgar:
    async def list_filings(self, cik, form_types=None, since=None):
        return [SimpleNamespace(accession_number=ACCESSION)]


class _FilingRepo:
    def __init__(self, status: FilingStatus):
        self.filing = Filing(
            id=7, security_id=1, filing_type="10-K", filed_date=date(2025, 10, 10),
            accession_number=ACCESSION, status=status, error_message="embeddings 429",
        )
        self.marked: list[FilingStatus] = []

    async def get_by_accession(self, accession):
        return self.filing

    async def mark_status(self, filing_id, status, error_message=None):
        self.marked.append(status)
        self.filing = self.filing.model_copy(update={"status": status, "error_message": error_message})


def _service(filing_repo) -> tuple[IngestionService, list]:
    service = IngestionService(
        edgar_client=_Edgar(), ticker_resolver=None, embedding_service=None,
        security_repo=None, filing_repo=filing_repo, document_repo=None,
        section_repo=None, chunk_repo=None,
    )
    advanced: list = []

    async def fake_upsert_security(ticker):
        return ListedSecurity(id=1, cik="0001467373", ticker=ticker, name=ticker)

    async def fake_advance(filing, summary, security):
        advanced.append(filing.status)

    service._upsert_security = fake_upsert_security
    service._advance_filing = fake_advance
    return service, advanced


def test_a_failed_filing_is_skipped_by_default_and_says_how_to_retry(caplog):
    repo = _FilingRepo(FilingStatus.FAILED)
    service, advanced = _service(repo)

    with caplog.at_level("WARNING"):
        asyncio.run(service.ingest_security("ACN", form_types=["10-K"]))

    assert advanced == []
    assert repo.marked == []
    assert "retry_failed" in caplog.text


def test_retry_failed_sends_the_filing_back_through_the_pipeline():
    repo = _FilingRepo(FilingStatus.FAILED)
    service, advanced = _service(repo)

    asyncio.run(service.ingest_security("ACN", form_types=["10-K"], retry_failed=True))

    assert repo.marked == [FilingStatus.DISCOVERED]
    assert advanced == [FilingStatus.DISCOVERED]


def test_retry_failed_leaves_healthy_filings_alone():
    repo = _FilingRepo(FilingStatus.EMBEDDED)
    service, advanced = _service(repo)

    asyncio.run(service.ingest_security("ACN", form_types=["10-K"], retry_failed=True))

    assert repo.marked == []
    assert advanced == [FilingStatus.EMBEDDED]


# ---------------------------------------------------------------------------
# Re-chunking replaces instead of appending
# ---------------------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.calls: list[tuple] = []


def test_rechunking_a_document_clears_its_existing_chunks_first():
    log = _Recorder()
    sections = [
        SimpleNamespace(id=11, section_path=["Part II", "Item 7"], order=0,
                        content="Revenue grew. " * 200),
        SimpleNamespace(id=12, section_path=["Part II", "Item 8"], order=1,
                        content="Net cash provided by operating activities. " * 200),
    ]

    class DocRepo:
        async def list_for_filing(self, filing_id):
            return [SimpleNamespace(id=3)]

    class SectionRepo:
        async def list_for_document(self, doc_id):
            return sections

    class ChunkRepo:
        async def delete_for_sections(self, section_ids):
            log.calls.append(("delete", tuple(section_ids)))
            return 5   # an interrupted run left some behind

        async def bulk_insert(self, chunks):
            log.calls.append(("insert", len(chunks)))
            return chunks

    class FilingRepo:
        async def mark_status(self, filing_id, status, error_message=None):
            log.calls.append(("mark", status))

    service = IngestionService(
        edgar_client=None, ticker_resolver=None, embedding_service=None,
        security_repo=None, filing_repo=FilingRepo(), document_repo=DocRepo(),
        section_repo=SectionRepo(), chunk_repo=ChunkRepo(),
    )
    filing = Filing(
        id=7, security_id=1, filing_type="10-K", filed_date=date(2025, 10, 10),
        accession_number=ACCESSION, status=FilingStatus.PARSED,
    )
    security = ListedSecurity(id=1, cik="0001467373", ticker="ACN", name="ACN")

    asyncio.run(service._chunk(filing, security))

    kinds = [c[0] for c in log.calls]
    assert kinds == ["delete", "insert", "mark"]
    assert log.calls[0] == ("delete", (11, 12))
    assert log.calls[1][1] > 0
    assert log.calls[2] == ("mark", FilingStatus.CHUNKED)
