import asyncio
import tempfile
import unittest
from pathlib import Path

from research_harness.agent_loop import AgentLoop, AgentRunConfig
from research_harness.llm import ModelToolCall, ModelTurn
from research_harness.schemas import Source
from research_harness.store import ArtifactStore
from research_harness.tools import DocumentAnalysisTool, SVGChartTool, StructuredDataExtractionTool, ToolContext, ToolRegistry


def _source() -> Source:
    return Source("https://example.org/paper", "Paper", "A", "", "fetched_document", "summary", 1.0, .8,
                  evidence_sections={"Results": "Latency: 12 ms. Accuracy: 95%."}, evidence_kind="verified_document",
                  evidence_locators={"Results": [{"kind": "pdf_page", "page": 4}]},
                  structured_tables=[{"name": "table_1", "headers": ["Method", "Latency (ms)", "Accuracy (%)"], "rows": [["A", "12", "95"], ["B", "10", "96"]], "locator": {"kind": "pdf_table", "page": 4}}], id="source_test")


class _AnalysisLLM:
    def complete_json(self, *_args, **_kwargs):
        return {"explicit": {"introduction": [], "research_question": [], "methodology": [], "experimental_setup": [], "results": [{"text": "Latency is reported.", "section": "Results"}], "conclusion": [], "assumptions": [], "limitations": []}, "inferences": [{"text": "Method B may be faster.", "basis_sections": ["Results"], "confidence": "low"}]}


class _TrajectoryDecider:
    def __init__(self): self.turn = 0
    async def decide(self, _messages, _tools):
        self.turn += 1
        if self.turn == 1:
            return ModelTurn("", [ModelToolCall("extract", "extract_structured_data", {"source_id": "source_test", "dataset_id": "trajectory-data"})], "tool_calls", "test", "test")
        if self.turn == 2:
            return ModelTurn("", [ModelToolCall("chart", "generate_svg_chart", {"dataset_id": "trajectory-data", "chart_type": "bar", "x_column": "Method", "y_column": "Latency (ms)"})], "tool_calls", "test", "test")
        return ModelTurn("Done.", [], "stop", "test", "test")


class DocumentDataToolsTest(unittest.TestCase):
    def _context(self):
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        store = ArtifactStore(Path(directory.name) / "run"); source = store.add_source(_source())
        return store, source, ToolContext(workspace=Path(directory.name), store=store, run_id="run-data")

    def test_extracts_dataset_with_table_row_provenance(self):
        store, source, context = self._context()
        result = asyncio.run(StructuredDataExtractionTool().execute({"source_id": source.id, "dataset_id": "results"}, context))
        self.assertEqual(result.status, "ok")
        dataset = store.read_dataset("results")
        self.assertEqual(dataset["records"][0]["Latency (ms)"], "12")
        self.assertEqual(dataset["provenance"][0]["locator"]["page"], 4)
        self.assertTrue(any(edge["relationship"] == "extracted_into" for edge in store.list("provenance_edges")))

    def test_analysis_separates_explicit_findings_and_inferences(self):
        store, source, context = self._context()
        result = asyncio.run(DocumentAnalysisTool(_AnalysisLLM()).execute({"source_id": source.id}, context))
        self.assertEqual(result.status, "ok")
        payload = __import__("json").loads((store.document_analyses_dir / f"analysis-{source.id}.json").read_text())
        self.assertEqual(payload["analysis"]["explicit"]["results"][0]["locators"][0]["page"], 4)
        self.assertIn("inferences", payload["analysis"])
        self.assertEqual(payload["analysis"]["inferences"][0]["locators"]["Results"][0]["page"], 4)

    def test_chart_persists_svg_configuration_and_provenance(self):
        store, source, context = self._context()
        asyncio.run(StructuredDataExtractionTool().execute({"source_id": source.id, "dataset_id": "results"}, context))
        result = asyncio.run(SVGChartTool().execute({"dataset_id": "results", "chart_type": "bar", "x_column": "Method", "y_column": "Latency (ms)", "chart_id": "latency"}, context))
        self.assertEqual(result.status, "ok")
        self.assertIn("<svg", (store.charts_dir / "latency.svg").read_text())
        self.assertTrue((store.charts_dir / "latency.json").is_file())
        self.assertTrue(any(edge["relationship"] == "visualized_as" for edge in store.list("provenance_edges")))

    def test_rejects_malformed_ids_and_incompatible_units(self):
        store, source, context = self._context()
        bad = asyncio.run(StructuredDataExtractionTool().execute({"source_id": source.id, "dataset_id": "../escape"}, context))
        self.assertEqual(bad.status, "error")
        store.write_dataset("mixed", {"id": "mixed", "source_id": source.id, "columns": [{"name": "Method", "unit": None}, {"name": "Time (ms)", "unit": "ms"}, {"name": "Distance (m)", "unit": "m"}], "records": [{"Method": "A", "Time (ms)": "2", "Distance (m)": "3"}]})
        incompatible = asyncio.run(SVGChartTool().execute({"dataset_id": "mixed", "chart_type": "grouped-bar", "x_column": "Method", "y_column": "Time (ms)", "y_columns": ["Time (ms)", "Distance (m)"]}, context))
        self.assertEqual(incompatible.status, "error")

    def test_model_selected_extraction_and_chart_trajectory(self):
        store, source, context = self._context()
        result = asyncio.run(AgentLoop(_TrajectoryDecider(), ToolRegistry([StructuredDataExtractionTool(), SVGChartTool()]), AgentRunConfig(max_iterations=4)).run("Inspect evidence.", context))
        self.assertEqual(result.status, "completed")
        self.assertEqual([call["tool"] for call in result.tool_calls], ["extract_structured_data", "generate_svg_chart"])
        self.assertTrue((store.charts_dir / "chart-trajectory-data.svg").is_file())
