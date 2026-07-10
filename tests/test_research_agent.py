from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from research_harness.research_agent import AgentRunConfig, ResearchAgent
from research_harness.search import LocalCorpusSearch
from research_harness.store import ArtifactStore
from research_harness.tools import SearchTool, ToolContext, ToolRegistry, WebFetchTool


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

        self.assertEqual(result.termination_reason, "final")
        self.assertEqual(result.final_answer, "No external evidence is needed.")
        self.assertEqual(result.tool_calls, [])

    def test_agent_recovers_after_bad_tool_selection_using_observation(self) -> None:
        decider = ScriptedDecider([
            {"type": "tool_call", "tool_name": "not_registered", "arguments": {}},
            {"type": "final", "answer": "I cannot access that capability, so here is the limitation."},
        ])
        result = self._agent(decider).run("Find evidence.", workspace=Path.cwd())

        self.assertEqual(result.termination_reason, "final")
        self.assertEqual(result.tool_calls[0]["status"], "error")
        self.assertIn("Unknown tool", decider.observed_messages[1][-1]["content"]["error"])

    def test_selected_search_tool_persists_real_sources_and_trace(self) -> None:
        decider = ScriptedDecider([
            {"type": "tool_call", "tool_name": "local_corpus_search", "arguments": {"query": "multi-agent systems", "limit": 2}},
            {"type": "final", "answer": "Grounded synthesis."},
        ])
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            result = self._agent(decider).run("Research multi-agent systems.", workspace=Path.cwd(), store=store, run_id="run_agent")

            self.assertEqual(result.termination_reason, "final")
            self.assertGreater(len(store.list("sources")), 0)
            transcript = json.loads(store.agent_transcript_path.read_text(encoding="utf-8"))
            self.assertEqual(transcript["termination_reason"], "final")
            self.assertEqual(transcript["tool_calls"][0]["tool"], "local_corpus_search")
            trace = store.list("agent_traces")[0]
            self.assertEqual(trace["tools_used"], ["local_corpus_search"])
            self.assertEqual(trace["tool_calls"][0]["results"], len(store.list("sources")))

    def test_iteration_limit_is_not_reported_as_a_final_answer(self) -> None:
        decider = ScriptedDecider([{"type": "tool_call", "tool_name": "not_registered", "arguments": {}}] * 3)
        result = ResearchAgent(decider, ToolRegistry([]), AgentRunConfig(max_iterations=2)).run("Do work.", workspace=Path.cwd())

        self.assertEqual(result.termination_reason, "iteration_limit")
        self.assertEqual(result.final_answer, "")

    def test_web_fetch_rejects_private_network_targets(self) -> None:
        result = WebFetchTool().execute({"url": "http://127.0.0.1/private"}, ToolContext(workspace=Path.cwd()))

        self.assertEqual(result.status, "error")
        self.assertIn("Private or loopback", result.error or "")

    def test_registry_rejects_invalid_tool_arguments_before_execution(self) -> None:
        backend = LocalCorpusSearch(Path("examples/corpus/research_corpus.json"))
        result = ToolRegistry([SearchTool(backend)]).execute(
            "local_corpus_search", {"query": "ok", "limit": 0}, ToolContext(workspace=Path.cwd())
        )

        self.assertEqual(result.status, "error")
        self.assertIn("below the minimum", result.error or "")


if __name__ == "__main__":
    unittest.main()
