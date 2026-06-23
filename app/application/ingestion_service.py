import logging
from datetime import date
from pathlib import Path

from app.domain.chunk import Chunk, chunk_from_draft
from app.domain.document import Document
from app.domain.filing import Filing
from app.domain.listed_security import ListedSecurity
from app.domain.section import Section
from app.domain.values import FilingStatus

from app.infrastructure.chunking.section_chunker import chunk_filing
from app.infrastructure.edgar.client import EdgarClient
from app.infrastructure.edgar.models import FilingSummary
from app.infrastructure.edgar.ticker_resolver import TickerResolver
from app.infrastructure.parsing.filing_parser import parse_filing

from app.infrastructure.repositories.chunk_repo import ChunkRepository
from app.infrastructure.repositories.document_repo import DocumentRepository
from app.infrastructure.repositories.filing_repo import FilingRepository
from app.infrastructure.repositories.listed_security_repo import (
    ListedSecurityRepository,
)
from app.infrastructure.repositories.section_repo import SectionRepository

from .embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

class IngestionService:
    """
    Orchestrates the full ingestion pipeline for one ListedSecurity.

    Each phase is idempotent and corresponds to a FilingStatus transition.
    Re-running ingestion picks up from wherever the last run left off.
    """

    def __init__(
        self,
        edgar_client: EdgarClient,
        ticker_resolver: TickerResolver,
        embedding_service: EmbeddingService,
        security_repo: ListedSecurityRepository,
        filing_repo: FilingRepository,
        document_repo: DocumentRepository,
        section_repo: SectionRepository,
        chunk_repo: ChunkRepository,
    ):
        self.edgar = edgar_client
        self.resolver = ticker_resolver
        self.embedder = embedding_service
        self.security_repo = security_repo
        self.filing_repo = filing_repo
        self.document_repo = document_repo
        self.section_repo = section_repo
        self.chunk_repo = chunk_repo

    # ----------------------- Top-level entry point -----------------------
    async def ingest_security(
        self,
        ticker: str,
        form_types: list[str] | None = None,
        limit: int | None = None,
        since: date | None = None,
    ) -> None:
        """
        Pull recent filings for a ticker and run them through the full pipeline.

        Idempotent: known filings are re-checked but not re-downloaded if
        already in their target state.
        """
        form_types = form_types or ["10-K"]

        security = await self._upsert_security(ticker)
        logger.info("Resolved %s -> security id=%s cik=%s",
                    ticker, security.id, security.cik)

        summaries = await self.edgar.list_filings(
            cik=security.cik, form_types=form_types, since=since
        )
        if limit:
            summaries = summaries[:limit]
        logger.info("Found %d filings to consider", len(summaries))

        # Phase 1: Discover (upsert filing rows, no work done yet)
        filings = []
        for summary in summaries:
            filing = await self._upsert_filing(security.id, summary)
            filings.append((filing, summary))

        # Phase 2: Walk each filing forward through states
        for filing, summary in filings:
            try:
                await self._advance_filing(filing, summary, security)
            except Exception as e:
                logger.exception("Filing %s failed", filing.accession_number)
                await self.filing_repo.mark_status(
                    filing.id, FilingStatus.FAILED, str(e)
                )

    # ----------------------- Discovery -----------------------

    async def _upsert_security(self, ticker: str) -> ListedSecurity:
        cik = await self.resolver.resolve(ticker)
        if not cik:
            raise ValueError(f"Unknown ticker: {ticker}")

        existing = await self.security_repo.get_by_cik(cik)
        if existing:
            return existing

        security = ListedSecurity(
            cik=cik,
            ticker=ticker,
            name=ticker,    # name unknown until we parse a filing
        )
        return await self.security_repo.upsert(security)

    async def _upsert_filing(
        self, security_id: int, summary: FilingSummary
    ) -> Filing:
        existing = await self.filing_repo.get_by_accession(summary.accession_number)
        if existing:
            return existing

        filing = Filing(
            security_id=security_id,
            filing_type=summary.form,
            filed_date=summary.filing_date,
            period_of_report=summary.report_date,
            accession_number=summary.accession_number,
            status=FilingStatus.DISCOVERED,
        )
        return await self.filing_repo.upsert(filing)

    # ----------------------- State transitions -----------------------

    async def _advance_filing(
        self,
        filing: Filing,
        summary: FilingSummary,
        security: ListedSecurity,
    ) -> None:
        """Walk one filing through whatever transitions remain."""
        # Re-load to catch up to current state in DB
        current = await self.filing_repo.get_by_accession(filing.accession_number)
        assert current and current.id is not None

        if current.status == FilingStatus.DISCOVERED:
            await self._download(current, summary, security)
            current.status = FilingStatus.DOWNLOADED

        if current.status == FilingStatus.DOWNLOADED:
            await self._parse(current, security)
            current.status = FilingStatus.PARSED

        if current.status == FilingStatus.PARSED:
            await self._chunk(current, security)
            current.status = FilingStatus.CHUNKED

        if current.status == FilingStatus.CHUNKED:
            await self._embed(current)
            current.status = FilingStatus.EMBEDDED

        logger.info(
            "Filing %s: final status %s",
            current.accession_number, current.status.value,
        )

    async def _download(
        self,
        filing: Filing,
        summary: FilingSummary,
        security: ListedSecurity,
    ) -> None:
        """Fetch the primary document, register Document row."""
        # If a document already exists for this filing, skip the fetch
        existing_docs = await self.document_repo.list_for_filing(filing.id)
        if existing_docs:
            await self.filing_repo.mark_status(filing.id, FilingStatus.DOWNLOADED)
            return

        local_path = await self.edgar.download_filing(security.cik, summary)
        original_url = self.edgar.ARCHIVES_URL.format(
            cik=str(int(security.cik)),
            accession=summary.accession_no_dashes,
            document=summary.primary_document,
        )

        document = Document(
            filing_id=filing.id,
            primary_document_name=summary.primary_document,
            document_type=summary.form,
            original_url=original_url,
            local_path=str(local_path),
        )
        await self.document_repo.insert(document)
        await self.filing_repo.mark_status(filing.id, FilingStatus.DOWNLOADED)
        logger.info("Downloaded filing %s", filing.accession_number)


    async def _parse(self, filing: Filing, security: ListedSecurity) -> None:
        """Parse HTML into Section rows."""
        documents = await self.document_repo.list_for_filing(filing.id)
        if not documents:
            raise RuntimeError(
                f"Filing {filing.accession_number} has no Document row to parse"
            )

        # If sections already exist for any doc, skip
        for doc in documents:
            existing = await self.section_repo.list_for_document(doc.id)
            if existing:
                continue

            parsed_sections = parse_filing(Path(doc.local_path))
            section_models = [
                Section(
                    document_id=doc.id,
                    section_path=ps.section_path,
                    order=ps.order,
                    content=ps.content,
                )
                for ps in parsed_sections
            ]
            await self.section_repo.bulk_insert(section_models)

        await self.filing_repo.mark_status(filing.id, FilingStatus.PARSED)
        logger.info("Parsed filing %s", filing.accession_number)

    async def _chunk(self, filing: Filing, security: ListedSecurity) -> None:
        """Produce chunks for every section of the filing."""
        documents = await self.document_repo.list_for_filing(filing.id)
        for doc in documents:
            sections = await self.section_repo.list_for_document(doc.id)
            if not sections:
                continue

            # Convert persisted sections back into parser-shaped objects
            # for the chunker, since chunk_filing operates on ParsedSection
            from app.infrastructure.parsing.models import ParsedSection
            parsed = [
                ParsedSection(
                    section_path=s.section_path,
                    order=s.order,
                    content=s.content,
                )
                for s in sections
            ]
            drafts = chunk_filing(parsed)

            # Map drafts back to their section_id by (section_path, order).
            # Each draft inherits from the section at the index matching
            # its global chunk_index — but actually we need to match by
            # section_path since drafts don't carry order. Easier to lift
            # the chunker output per section.
            chunks_to_save: list[Chunk] = []
            section_by_path = {tuple(s.section_path): s for s in sections}
            for draft in drafts:
                key = tuple(draft.section_path)
                section = section_by_path.get(key)
                if section is None:
                    logger.warning("No section matched chunk path %s", draft.section_path)
                    continue
                chunks_to_save.append(
                    chunk_from_draft(
                        draft=draft,
                        section_id=section.id,
                        ticker=security.ticker,
                        filed_date=filing.filed_date,
                        filing_type=filing.filing_type,
                    )
                )

            await self.chunk_repo.bulk_insert(chunks_to_save)

        await self.filing_repo.mark_status(filing.id, FilingStatus.CHUNKED)
        logger.info("Chunked filing %s", filing.accession_number)

    async def _embed(self, filing: Filing) -> None:
        """Embed chunks that don't yet have vectors."""
        chunks = await self.chunk_repo.list_without_embeddings(filing_id=filing.id)
        if not chunks:
            await self.filing_repo.mark_status(filing.id, FilingStatus.EMBEDDED)
            return

        logger.info("Embedding %d chunks for filing %s",
                    len(chunks), filing.accession_number)
        vectors = await self.embedder.embed_many([c.content for c in chunks])
        updates = [(c.id, v) for c, v in zip(chunks, vectors)]
        await self.chunk_repo.update_embeddings(updates)
        await self.filing_repo.mark_status(filing.id, FilingStatus.EMBEDDED)
        logger.info("Embedded filing %s", filing.accession_number)