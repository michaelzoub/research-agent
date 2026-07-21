"""Production-path smoke tests for the single model-directed architecture."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_harness.orchestrator import HarnessConfig, Orchestrator, goal_slug


class OrchestratorSmokeTest(unittest.TestCase):
    def test_every_run_uses_one_agent_path_and_writes_actual_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                Path("examples/corpus/research_corpus.json"),
                Path(directory),
                HarnessConfig(retriever="local", llm_provider="local", echo_progress=False, enable_sessions=False),
            )
            run, store = asyncio.run(orchestrator.run("Rewrite this sentence more clearly."))

            self.assertEqual(run.status, "partial")
            state = json.loads(store.run_state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["schema_version"], "model_directed_run_state_v1")
            self.assertNotIn("execution_mode", state)
            self.assertNotIn("plan", state)
            self.assertIn("observed_counts", state)
            self.assertNotIn("events", state)
            self.assertTrue(store.agent_transcript_path.exists())
            self.assertTrue(store.agent_timeline_path.exists())
            self.assertTrue(store.agent_timeline_svg_path.exists())
            self.assertFalse(store.candidate_graph_path.exists())

            # Collections appear only after a real record is written; research
            # runs should not be cluttered with unrelated empty optimizer data.
            self.assertFalse((store.root / "hypotheses.json").exists())
            self.assertFalse((store.root / "variants.json").exists())

    def test_goal_slug_is_stable(self) -> None:
        self.assertEqual(goal_slug("Please research new agent paradigms"), "please-research-new-agent-paradigms")

    def test_configured_grader_retains_the_base_agent_loop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                Path("examples/corpus/research_corpus.json"),
                Path(directory),
                HarnessConfig(
                    retriever="local", llm_provider="local", evaluator_name="prediction_market",
                    echo_progress=False, enable_sessions=False,
                ),
            )
            with patch.dict(os.environ, {"PREDICTION_MARKET_ALLOW_UNSANDBOXED_UPSTREAM": "1"}):
                _run, store = asyncio.run(orchestrator.run("Optimize PM challenge."))

            self.assertTrue(store.agent_transcript_path.exists())
            self.assertFalse(store.optimization_result_path.exists())
            state = json.loads(store.run_state_path.read_text(encoding="utf-8"))
            self.assertIn("evaluate_prediction_market_candidate", state["available_tools"])
            preflight = json.loads(store.grader_preflight_path.read_text(encoding="utf-8"))
            self.assertTrue(preflight["ok"])
            self.assertIn("class Strategy(BaseStrategy)", preflight["baseline_code"])

    def test_unavailable_grader_stops_before_any_model_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"PREDICTION_MARKET_USE_UPSTREAM": "0"}
        ):
            orchestrator = Orchestrator(
                Path("examples/corpus/research_corpus.json"),
                Path(directory),
                HarnessConfig(
                    retriever="local", llm_provider="local", evaluator_name="prediction_market",
                    echo_progress=False, enable_sessions=False,
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "Official grader preflight failed"):
                asyncio.run(orchestrator.run("Optimize PM challenge."))

            run_root = next(path for path in Path(directory).iterdir() if path.is_dir())
            preflight = json.loads((run_root / "grader_preflight.json").read_text(encoding="utf-8"))
            self.assertFalse(preflight["ok"])
            self.assertFalse((run_root / "agent_messages.json").exists())

    def test_auto_retrieval_excludes_the_fixture_corpus(self) -> None:
        orchestrator = Orchestrator(
            Path("examples/corpus/research_corpus.json"),
            Path(tempfile.gettempdir()),
            HarnessConfig(llm_provider="local"),
        )

        self.assertNotIn("local", orchestrator._enabled_retrievers())


if __name__ == "__main__":
    unittest.main()
