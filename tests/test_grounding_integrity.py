import asyncio
import tempfile
import unittest
from pathlib import Path

from research_harness.citation_validation import coverage, validate_claim_citations
from research_harness.document_ingestion import ingest_document
from research_harness.schemas import Source
from research_harness.store import ArtifactStore


class GroundingIntegrityTest(unittest.TestCase):
    def test_html_ingestion_retains_heading_locators(self) -> None:
        result = ingest_document(
            b"<h1>Methods</h1><p>The experiment uses a fixed seed.</p><h2>Results</h2><p>Coverage improved.</p>",
            "text/html", max_characters=2000,
        )
        self.assertEqual(result["document_type"], "html")
        self.assertIn("Methods", result["evidence_sections"])
        self.assertEqual(result["evidence_locators"]["Methods"][0]["kind"], "html_section")

    def test_search_lead_cannot_satisfy_claim_grounding(self) -> None:
        source = {"id": "lead", "url": "https://example.org/paper", "evidence_kind": "lead", "evidence_sections": {"snippet": "The experiment uses a fixed seed."}}
        checks = validate_claim_citations("The experiment uses a fixed seed. https://example.org/paper", [source])
        self.assertEqual(coverage(checks), 0.0)
        self.assertIn("leads", checks[0].reason)

    def test_verified_document_measures_support_and_returns_locator(self) -> None:
        source = {
            "id": "evidence", "url": "https://example.org/paper", "evidence_kind": "verified_document",
            "evidence_sections": {"page_3": "The experiment uses a fixed seed for every trial."},
            "evidence_locators": {"page_3": [{"kind": "pdf_page", "page": 3}]},
        }
        checks = validate_claim_citations("The experiment uses a fixed seed for every trial. https://example.org/paper", [source])
        self.assertTrue(checks[0].passed)
        self.assertEqual(checks[0].locators[0]["page"], 3)

    def test_tool_source_commit_is_serial_and_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            source = Source("https://example.org/a", "A", "", "", "web_result", "lead", 1.0, 0.5)
            committed = store.commit_tool_sources([source.__dict__, source.__dict__])
            self.assertEqual(len(committed), 2)
            self.assertEqual(len(store.list("sources")), 1)

