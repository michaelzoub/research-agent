"""Production-path smoke tests for the single model-directed architecture."""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

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

            self.assertEqual(run.status, "completed")
            state = json.loads(store.run_state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["schema_version"], "model_directed_run_state_v1")
            self.assertNotIn("execution_mode", state)
            self.assertNotIn("plan", state)
            self.assertIn("events", state)
            self.assertTrue(store.agent_transcript_path.exists())

    def test_goal_slug_is_stable(self) -> None:
        self.assertEqual(goal_slug("Please research new agent paradigms"), "new-agent-paradigms")


if __name__ == "__main__":
    unittest.main()
