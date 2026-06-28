## 2026-06-28 — Session 1: Apple supply chain risk

**Question:** What does Apple say about its dependence on Chinese
manufacturing? Has the language changed between 2022 and 2025?

**Filters used:** tickers=AAPL, filing_types=10-K, section_path_contains=Risk Factors

**Command used:**
```bash
curl -sX POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "question": "What does Apple say about its dependence on Chinese manufacturing? Has the language changed between 2022 and 2025?",
    "section_path_contains": ["Risk Factors"]
  }' | jq
```
**Top 5 chunks retrieved:**
- All 5 from [AAPL 10-K 2025 §Item 1A]
- Scores: 0.55, 0.51, 0.50, 0.49, 0.48
- Zero chunks from 2022 despite explicit ask

**Diagnostic re-run** with `filed_after=2022-01-01, filed_before=2022-12-31`:
- 5 chunks from 2022, top similarity 0.62
- Content directly addresses geographic manufacturing concentration
- Confirms: 2022 chunks does not exist

**LLM answer:** 
```
**Apple's dependence on Chinese manufacturing (2025 10-K)**\n\nApple states that \"A significant majority of the Company's manufacturing is performed in whole or in part by outsourcing partners located primarily in China mainland, India, Japan, South Korea, Taiwan and Vietnam, in addition to sourcing from partners and facilities located in the U.S.\" [AAPL 10-K 2025 §Item 1A]\n\nThe 2025 filing also discusses heightened trade risk specifically tied to these regions, noting that \"Beginning in the second quarter of 2025, new tariffs were announced on imports to the U.S. ('U.S. Tariffs'), including additional tariffs on imports from China, India, Japan, South Korea, Taiwan, Vietnam and the European Union ('EU'), among others.\" It further notes that \"Global supply chains can be highly concentrated, and an escalation of geopolitical tensions or conflict could result in significant disruptions.\" [AAPL 10-K 2025 §Item 1A]\n\n**Comparison to 2022**\n\nThe provided context does not include any excerpts from Apple's 2022 10-K, so I cannot compare the specific language used in 2022 versus 2025. To answer whether the disclosure language has changed between those years, excerpts from the 2022 filing would be needed.
```

**My assessment:**
- Retrieval: ✅ all 5 from Risk Factors, all relevant
- Answer quality: ⚠️ missed a clear quote from 2022 10-K about geographic concentration
- Citations: ✅ all citations check out

**Tagged issue:** 
1. **[Resolved] Phase 1 data integrity bug** — incomplete ingestion
   gave plausible but misleading answers. Fixed by re-running with
   correct --limit. Long-term mitigation: add `corpus-status` CLI

2. **Narrow similarity range across topical chunks** — top-8 spans
   0.519–0.569, only 0.05 of separation. Embedding model can identify
   the topic ("supply chain risk") but can't rank within-topic
   relevance. Strong signal that reranking with a cross-encoder
   would meaningfully improve top-k quality.

3. **Templated disclosures defeat naive retrieval** — risk factor
   paragraphs are nearly identical year-over-year. Top-k returns
   8 versions of the same paragraph rather than surfacing the
   year-over-year deltas. Suggests need for either:
     a) MMR-style diversification (penalize near-duplicate chunks)
     b) Diff-aware retrieval ("what's NEW in 2024 vs 2023")
   This is more interesting than generic hybrid search.

4. **LLM honesty win (revalidated)** — model would have correctly
   answered the 2022 vs 2025 comparison if given chunks from both.
   Behavior was correct in both runs.

