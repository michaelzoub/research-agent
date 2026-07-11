from __future__ import annotations

import json
import asyncio
import subprocess
import tempfile
import time
import unittest
from unittest import mock
from pathlib import Path
from typing import Any

from research_harness.cli import build_parser
from research_harness.llm import ModelToolCall, ModelTurn
from research_harness.agents import SpecialistConsultationTool
from research_harness.orchestrator import HarnessConfig
from research_harness.research_agent import AgentRunConfig, FinalAnswerValidator, ResearchAgent, _join_answer_chunks, _partial_synthesis
from research_harness.search import ArxivSearch, LocalCorpusSearch, WebSearch, _arxiv_identifier, _arxiv_query, _retrieval_query
from research_harness.store import ArtifactStore
from research_harness.tools import DocumentFigureTool, SearchTool, TerminalExecutionTool, ToolContext, ToolRegistry, ToolResult, WebFetchTool
from research_harness.tools.research import _FigureHTMLParser, _image_dimensions, _inspection_urls


class ScriptedDecider:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.observed_messages: list[list[dict[str, Any]]] = []

    def decide(self, messages: list[dict[str, Any]], _tools: list[dict[str, Any]]) -> dict[str, Any]:
        self.observed_messages.append(list(messages))
        return self.responses.pop(0)


class ResearchAgentTest(unittest.TestCase):
    def _agent(self, decider: ScriptedDecider) -> ResearchAgent:
        backend = LocalCorpusSearch(Path("examples/corpus/research_corpus.json"))
        return ResearchAgent(decider, ToolRegistry([SearchTool(backend)]), AgentRunConfig(max_iterations=4))

    def test_agent_can_answer_without_a_tool_call(self) -> None:
        agent = self._agent(ScriptedDecider([{"type": "final", "answer": "No external evidence is needed."}]))

        result = agent.run("Rewrite this sentence.", workspace=Path.cwd())

        self.assertEqual(result.termination_reason, "completed")
        self.assertEqual(result.final_answer, "No external evidence is needed.")
        self.assertEqual(result.tool_calls, [])

    def test_agent_recovers_after_bad_tool_selection_using_observation(self) -> None:
        decider = ScriptedDecider([
            {"type": "tool_call", "tool_name": "not_registered", "arguments": {}},
            {"type": "final", "answer": "I cannot access that capability, so here is the limitation."},
        ])
        result = self._agent(decider).run("Find evidence.", workspace=Path.cwd())

        self.assertEqual(result.termination_reason, "completed")
        self.assertEqual(result.tool_calls[0]["status"], "error")
        self.assertIn("Unknown tool", decider.observed_messages[1][-1]["content"]["error"])

    def test_tool_error_is_persisted_to_failed_paths(self) -> None:
        decider = ScriptedDecider([
            {"type": "tool_call", "tool_name": "not_registered", "arguments": {}},
            {"type": "final", "answer": "I cannot access that capability."},
        ])
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            self._agent(decider).run("Find evidence.", workspace=Path.cwd(), store=store, run_id="run_error")
            failures = store.list("failed_paths")

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["failure_component"], "tool")
        self.assertIn("Unknown tool", failures[0]["reason"])

    def test_selected_search_tool_persists_real_sources_and_trace(self) -> None:
        decider = ScriptedDecider([
            {"type": "tool_call", "tool_name": "local_corpus_search", "arguments": {"query": "multi-agent systems", "limit": 2}},
            {"type": "final", "answer": "Grounded synthesis. https://example.org/single-agent-baseline-limitations"},
        ])
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            result = self._agent(decider).run("Research multi-agent systems.", workspace=Path.cwd(), store=store, run_id="run_agent")

            self.assertEqual(result.termination_reason, "completed")
            self.assertGreater(len(store.list("sources")), 0)
            transcript = json.loads(store.agent_transcript_path.read_text(encoding="utf-8"))
            self.assertEqual(transcript["termination_reason"], "completed")
            self.assertEqual(transcript["tool_calls"][0]["tool"], "local_corpus_search")
            trace = store.list("agent_traces")[0]
            self.assertEqual(trace["tools_used"], ["local_corpus_search"])
            self.assertEqual(trace["tool_calls"][0]["results"], len(store.list("sources")))
            event_rows = [json.loads(line) for line in store.agent_event_log_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["event_type"] for event in event_rows], ["model_turn", "tool_requested", "tool_result", "model_turn", "final_validation"])

    def test_iteration_limit_is_not_reported_as_a_final_answer(self) -> None:
        decider = ScriptedDecider([{"type": "tool_call", "tool_name": "not_registered", "arguments": {}}] * 3)
        result = ResearchAgent(decider, ToolRegistry([]), AgentRunConfig(max_iterations=2)).run("Do work.", workspace=Path.cwd())

        self.assertEqual(result.termination_reason, "budget_exhausted")
        self.assertIn("Incomplete evidence packet", result.final_answer)

    def test_web_fetch_rejects_private_network_targets(self) -> None:
        result = asyncio.run(WebFetchTool().execute({"url": "http://127.0.0.1/private"}, ToolContext(workspace=Path.cwd())))

        self.assertEqual(result.status, "error")
        self.assertIn("Private or loopback", result.error or "")

    def test_terminal_executes_direct_argv_and_preserves_output(self) -> None:
        captured: dict[str, Any] = {}

        def runner(argv: list[str], cwd: Path, timeout_seconds: float):
            captured.update({"argv": argv, "cwd": cwd, "timeout": timeout_seconds})
            return subprocess.CompletedProcess(argv, 0, stdout="real command output", stderr="")

        tool = TerminalExecutionTool(runner=runner)
        with mock.patch("research_harness.tools.terminal.shutil.which", return_value="/usr/bin/curl"), mock.patch("research_harness.tools.terminal._public_url_error", return_value=None):
            result = asyncio.run(tool.execute(
                {"command": "curl", "args": ["https://example.org/paper"], "timeout_seconds": 12},
                ToolContext(workspace=Path.cwd()),
            ))

        self.assertEqual(result.status, "ok")
        self.assertEqual(captured["argv"], ["/usr/bin/curl", "https://example.org/paper"])
        self.assertEqual(captured["cwd"], Path.cwd().resolve())
        self.assertEqual(captured["timeout"], 12)
        self.assertEqual(result.data["stdout"], "real command output")
        self.assertEqual(result.source_metadata[0]["url"], "https://example.org/paper")

    def test_terminal_rejects_private_curl_and_mutating_npm(self) -> None:
        tool = TerminalExecutionTool()
        private_curl = asyncio.run(tool.execute(
            {"command": "curl", "args": ["http://127.0.0.1/private"]}, ToolContext(workspace=Path.cwd())
        ))
        npm_install = asyncio.run(tool.execute(
            {"command": "npm", "args": ["install", "some-package"]}, ToolContext(workspace=Path.cwd())
        ))

        self.assertEqual(private_curl.status, "error")
        self.assertIn("Private or loopback", private_curl.error or "")
        self.assertEqual(npm_install.status, "error")
        self.assertIn("limited", npm_install.error or "")

    def test_registry_rejects_invalid_tool_arguments_before_execution(self) -> None:
        backend = LocalCorpusSearch(Path("examples/corpus/research_corpus.json"))
        result = asyncio.run(ToolRegistry([SearchTool(backend)]).execute(
            "local_corpus_search", {"query": "ok", "limit": 0}, ToolContext(workspace=Path.cwd())
        ))

        self.assertEqual(result.status, "error")
        self.assertIn("below the minimum", result.error or "")

    def test_cli_and_config_have_no_execution_mode(self) -> None:
        parser = build_parser()
        self.assertNotIn("--mode", parser.format_help())
        self.assertFalse(hasattr(HarnessConfig(), "mode"))

    def test_multiple_read_only_model_requested_tools_run_concurrently(self) -> None:
        class DelayedTool:
            is_read_only = True
            input_schema = {"type": "object", "required": [], "properties": {}, "additionalProperties": False}

            def __init__(self, name: str):
                self.name, self.description = name, name

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                await asyncio.sleep(0.08)
                return ToolResult("ok", {"name": self.name})

        class NativeScript:
            def __init__(self) -> None:
                self.turns = [
                    ModelTurn("", [ModelToolCall("a", "first", {}), ModelToolCall("b", "second", {})], "tool_calls", "test", "test"),
                    ModelTurn("Done.", [], "stop", "test", "test"),
                ]

            def decide(self, _messages: Any, _tools: Any) -> ModelTurn:
                return self.turns.pop(0)

        agent = ResearchAgent(NativeScript(), ToolRegistry([DelayedTool("first"), DelayedTool("second")]), AgentRunConfig(max_iterations=3))
        started = time.monotonic()
        result = agent.run("Do two independent reads.", workspace=Path.cwd())
        self.assertLess(time.monotonic() - started, 0.14)
        self.assertEqual(result.status, "completed")
        self.assertEqual([call["id"] for call in result.tool_calls], ["a", "b"])

    def test_unsupported_citation_is_returned_for_revision(self) -> None:
        decider = ScriptedDecider([
            {"type": "tool_call", "tool_name": "local_corpus_search", "arguments": {"query": "multi-agent systems", "limit": 1}},
            {"type": "final", "answer": "Unsupported claim https://invalid.example/not-retrieved"},
            {"type": "final", "answer": "Supported claim https://example.org/single-agent-baseline-limitations"},
        ])
        result = self._agent(decider).run("Find evidence.", workspace=Path.cwd())
        self.assertEqual(result.status, "completed")
        feedback = [event for event in result.events if event.event_type == "final_validation"]
        self.assertEqual(feedback[0].observation["status"], "REVISE")

    def test_length_limited_turn_is_continued_not_accepted_as_final(self) -> None:
        class LengthThenFinal:
            def __init__(self) -> None:
                self.turns = [
                    ModelTurn("First half.", [], "length", "test", "test"),
                    ModelTurn("Second half.", [], "stop", "test", "test"),
                ]

            def decide(self, _messages: Any, _tools: Any) -> ModelTurn:
                return self.turns.pop(0)

        result = ResearchAgent(LengthThenFinal(), ToolRegistry([]), AgentRunConfig(max_iterations=3)).run("Explain something.", workspace=Path.cwd())
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.final_answer, "First half.\n\nSecond half.")

    def test_budget_rejected_calls_receive_matching_tool_messages(self) -> None:
        class TwoCallsThenFinal:
            def __init__(self) -> None:
                self.turns = [
                    ModelTurn("Need two reads.", [ModelToolCall("first", "known", {}), ModelToolCall("second", "known", {})], "tool_calls", "test", "test"),
                    ModelTurn("Grounded answer.", [], "stop", "test", "test"),
                ]
                self.second_messages: list[dict[str, Any]] = []

            def decide(self, messages: Any, _tools: Any) -> ModelTurn:
                if len(self.turns) == 1:
                    self.second_messages = list(messages)
                return self.turns.pop(0)

        class KnownTool:
            name, description, is_read_only = "known", "known", True
            input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                return ToolResult("ok")

        decider = TwoCallsThenFinal()
        result = ResearchAgent(decider, ToolRegistry([KnownTool()]), AgentRunConfig(max_iterations=3, max_tool_calls=1)).run("Use tools.", workspace=Path.cwd())
        responses = [message for message in decider.second_messages if message.get("role") == "tool"]
        self.assertEqual(result.status, "completed")
        self.assertEqual({message["tool_call_id"] for message in responses}, {"first", "second"})
        self.assertEqual(result.tool_calls[1]["status"], "skipped")

    def test_retryable_tool_errors_do_not_consume_evidence_budget(self) -> None:
        class RecoveringTool:
            name, description, is_read_only = "recovering", "recovering", True
            input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

            def __init__(self) -> None:
                self.attempts = 0

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                self.attempts += 1
                if self.attempts == 1:
                    return ToolResult("error", error="temporary search backend failure", retryable=True)
                return ToolResult("ok", {"evidence": "retrieved"})

        class ErrorThenRecover:
            def __init__(self) -> None:
                self.turns = [
                    ModelTurn("Try discovery.", [ModelToolCall("one", "recovering", {})], "tool_calls", "test", "test"),
                    ModelTurn("Try an alternative route.", [ModelToolCall("two", "recovering", {})], "tool_calls", "test", "test"),
                    ModelTurn("Grounded result.", [], "stop", "test", "test"),
                ]

            def decide(self, _messages: Any, _tools: Any) -> ModelTurn:
                return self.turns.pop(0)

        result = ResearchAgent(ErrorThenRecover(), ToolRegistry([RecoveringTool()]), AgentRunConfig(max_iterations=4, max_tool_calls=1)).run("Recover from a temporary source failure.", workspace=Path.cwd())

        self.assertEqual(result.status, "completed")
        self.assertEqual([call["status"] for call in result.tool_calls], ["error", "ok"])

    def test_empty_discovery_does_not_consume_evidence_budget(self) -> None:
        class EmptyThenEvidenceTool:
            name, description, is_read_only = "discovery", "discovery", True
            input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

            def __init__(self) -> None:
                self.attempts = 0

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                self.attempts += 1
                if self.attempts == 1:
                    return ToolResult("ok", {"result_count": 0})
                return ToolResult("ok", {"result_count": 1}, source_metadata=[{"url": "https://example.org/evidence"}])

        class TwoDiscoveryPasses:
            def __init__(self) -> None:
                self.turns = [
                    ModelTurn("Try one source.", [ModelToolCall("one", "discovery", {})], "tool_calls", "test", "test"),
                    ModelTurn("Try a second source.", [ModelToolCall("two", "discovery", {})], "tool_calls", "test", "test"),
                    ModelTurn("Grounded result https://example.org/evidence", [], "stop", "test", "test"),
                ]

            def decide(self, _messages: Any, _tools: Any) -> ModelTurn:
                return self.turns.pop(0)

        result = ResearchAgent(TwoDiscoveryPasses(), ToolRegistry([EmptyThenEvidenceTool()]), AgentRunConfig(max_iterations=4, max_tool_calls=1)).run("Find external evidence.", workspace=Path.cwd())

        self.assertEqual(result.status, "completed")
        self.assertEqual([call["status"] for call in result.tool_calls], ["ok", "ok"])

    def test_html_figure_parser_preserves_caption_and_image_url(self) -> None:
        parser = _FigureHTMLParser("https://example.org/paper")
        parser.feed('<figure class="ltx_figure"><img src="images/plot.png"><figcaption>Figure 2: Agent performance over time.</figcaption></figure>')

        self.assertEqual(parser.figures, [{"image_url": "https://example.org/images/plot.png", "caption": "Figure 2: Agent performance over time."}])

    def test_figure_inspection_registers_direct_figure_assets_for_citation(self) -> None:
        tool = DocumentFigureTool()
        inspected = ToolResult("ok", {
            "source_url": "https://example.org/paper",
            "figures": [{"image_url": "https://example.org/figures/one.png", "caption": "Figure 1: Verified result."}],
        })
        with mock.patch.object(tool, "_inspect", return_value=inspected):
            result = asyncio.run(tool.execute({"url": "https://example.org/paper"}, ToolContext(workspace=Path.cwd())))

        self.assertEqual(result.status, "ok")
        self.assertEqual(
            {source["url"] for source in result.source_metadata},
            {"https://example.org/paper", "https://example.org/figures/one.png"},
        )

    def test_image_dimensions_reads_png_header(self) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + (1200).to_bytes(4, "big") + (800).to_bytes(4, "big")
        self.assertEqual(_image_dimensions(png), (1200, 800))

    def test_arxiv_query_preserves_exact_ids_and_filters_irrelevant_results(self) -> None:
        self.assertEqual(_arxiv_identifier("Concrete Problems in AI Safety 1606.06565"), "1606.06565")
        self.assertNotIn("%", _arxiv_query("Stochastic Parrots Bender Gebru 2021"))
        payload = b'''<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2607.08562v1</id><title>XShooter DESI Lens Program</title><summary>Astronomy lens observations.</summary><published>2026-07-09T00:00:00Z</published><author><name>Author</name></author><category term="astro-ph"/></entry></feed>'''

        response = mock.MagicMock()
        response.read.return_value = payload
        response.__enter__.return_value = response
        with mock.patch("research_harness.search.urllib.request.urlopen", return_value=response) as urlopen:
            results = ArxivSearch().search("Concrete Problems in AI Safety Amodei 2016", limit=5)

        self.assertEqual(results, [])
        self.assertNotIn("%25", urlopen.call_args.args[0].full_url)

    def test_arxiv_retrieval_query_drops_figure_format_noise(self) -> None:
        self.assertEqual(
            _retrieval_query("FunSearch figure chart algorithm discovery performance"),
            "funsearch algorithm discovery",
        )

    def test_figure_inspection_uses_arxiv_pdf_when_html_is_unavailable(self) -> None:
        self.assertEqual(
            _inspection_urls("https://arxiv.org/abs/2304.03442v2"),
            [
                "https://arxiv.org/html/2304.03442v2",
                "https://arxiv.org/pdf/2304.03442v2",
                "https://arxiv.org/abs/2304.03442v2",
            ],
        )

    def test_web_search_reports_duckduckgo_bot_challenge(self) -> None:
        response = mock.MagicMock()
        response.read.return_value = b"<html><div class='anomaly-modal'>Unfortunately, bots use DuckDuckGo too.</div></html>"
        response.__enter__.return_value = response
        with mock.patch("research_harness.search.urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "bot challenge"):
                WebSearch().search("AI safety")

    def test_external_source_objective_cannot_pass_without_sources(self) -> None:
        status, feedback = FinalAnswerValidator().validate("A confident but uncited answer.", "Use external sources to explain AGI limitations.", [])
        self.assertEqual(status, "REVISE")
        self.assertIn("external evidence", feedback)

    def test_generic_source_handoff_is_not_accepted_as_final_research_answer(self) -> None:
        status, feedback = FinalAnswerValidator().validate(
            "Pick one: Option A: send 5 candidate URLs. Option B: give permission to use another discovery source.",
            "Use external sources to find figures.",
            [],
        )
        self.assertEqual(status, "REVISE")
        self.assertIn("generic request", feedback)

    def test_citation_validation_normalizes_http_and_https(self) -> None:
        status, _feedback = FinalAnswerValidator().validate(
            "See https://arxiv.org/abs/1606.06565v2.",
            "Use external sources.",
            [{"url": "http://arxiv.org/abs/1606.06565v2"}],
        )
        self.assertEqual(status, "PASS")

    def test_length_continuation_does_not_break_url(self) -> None:
        self.assertEqual(
            _join_answer_chunks(["Source: https://arxiv.org", "/abs/1606.06565v2" ]),
            "Source: https://arxiv.org/abs/1606.06565v2",
        )

    def test_incomplete_evidence_packet_preserves_retrieved_summary(self) -> None:
        report = _partial_synthesis(
            [{"title": "Primary result", "url": "https://example.org/result", "summary": "The source contains the directly retrieved result."}],
            None,
        )

        self.assertIn("Incomplete evidence packet", report)
        self.assertIn("directly retrieved result", report)
        self.assertIn("https://example.org/result", report)

    def test_controller_can_consult_a_model_chosen_specialist(self) -> None:
        class SpecialistLLM:
            model_label = "specialist-test"

            def complete(self, _system: str, _prompt: str, **_kwargs: Any):
                from research_harness.llm import LLMResponse
                return LLMResponse("Evidence is weak because the baseline is mismatched.", "specialist-test", "test", 12, 8, 0.01)

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            tool = SpecialistConsultationTool(SpecialistLLM())
            result = asyncio.run(tool.execute(
                {"specialty": "evidence critic", "question": "Assess the baseline.", "evidence": ["Model A beat Model B."]},
                ToolContext(workspace=Path.cwd(), store=store, run_id="run_specialist"),
            ))
            self.assertEqual(result.status, "ok")
            self.assertIn("baseline is mismatched", result.data["response"])
            self.assertEqual(store.list("agent_traces")[0]["role"], "specialist_consultation")


if __name__ == "__main__":
    unittest.main()
