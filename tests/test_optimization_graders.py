from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import patch

from optimization_graders import get_optimization_grader, list_optimization_graders, optimization_grader_baselines
from research_harness.sandbox import SandboxExecutionResult
from research_harness.store import ArtifactStore


class OptimizationGraderTest(unittest.TestCase):
    def test_prediction_market_is_discoverable_candidate_grader(self) -> None:
        self.assertIn("prediction_market", list_optimization_graders())
        grader = get_optimization_grader("prediction_market")
        self.assertEqual(grader.identifier, "prediction_market")
        self.assertIn("prediction-market-challenge", grader.upstream_url)
        baselines = optimization_grader_baselines("prediction_market")
        self.assertIn("starter_strategy", baselines)
        self.assertTrue(baselines["starter_strategy"].name.endswith(".py"))
        self.assertIn("class Strategy", baselines["starter_strategy"].read_text(encoding="utf-8"))

    def test_missing_official_scorer_is_zero_and_not_promotable(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"PREDICTION_MARKET_USE_UPSTREAM": "0"}):
            candidate = Path(directory) / "strategy.py"
            candidate.write_text("class Strategy: pass\n", encoding="utf-8")
            result = get_optimization_grader("prediction_market").evaluate(candidate)

        self.assertFalse(result["official_measured"])
        self.assertFalse(result["score_eligible"])
        self.assertEqual(result["mean_edge"], 0.0)
        self.assertEqual(result["candidate_code"], "class Strategy: pass\n")
        self.assertIn("upstream_url", result["upstream"])

    def test_getattr_candidate_is_rejected_without_official_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "strategy.py"
            candidate.write_text("class Strategy:\n    value = getattr(object(), 'x', None)\n", encoding="utf-8")
            result = get_optimization_grader("prediction_market").evaluate(candidate)

        self.assertFalse(result["official_measured"])
        self.assertFalse(result["score_eligible"])
        self.assertEqual(result["score_source"], "candidate_contract_invalid")
        self.assertIn("getattr()", result["error"])

    def test_all_failed_simulations_surface_aggregated_scorer_error(self) -> None:
        payload = {"simulation_results": [
            {"failed": True, "error": "OrderBookError: price_ticks must be an integer"},
            {"failed": True, "error": "OrderBookError: price_ticks must be an integer"},
        ]}
        completed = SandboxExecutionResult(json.dumps(payload), "", 0, command=("docker", "run"))
        grader = get_optimization_grader("prediction_market")
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "strategy.py"
            candidate.write_text("class Strategy: pass\n", encoding="utf-8")
            with patch.object(grader, "find_upstream_path", return_value=Path("challenges/prediction-market-challenge")), patch(
                "optimization_graders.prediction_market.adapter.DockerSandboxRunner.execute_prediction_market",
                return_value=completed,
            ):
                result = grader.evaluate(candidate)

        self.assertIn("2x OrderBookError: price_ticks must be an integer", result["error"])
        self.assertEqual(result["failure_count"], 2)
        self.assertEqual(result["simulations"], 2)

    def test_trial_artifact_retains_audit_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            path = store.write_optimization_trial(
                "candidate/001",
                {
                    "rendered_code": "class Strategy: pass\n",
                    "command": ["docker", "run"],
                    "upstream": {"upstream_url": "https://example.test/upstream", "upstream_revision": "abc"},
                    "score": 0.0,
                    "stdout": "",
                    "stderr": "failed",
                    "failure": "official scorer unavailable",
                },
            )
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["score"], 0.0)
        self.assertEqual(payload["command"], ["docker", "run"])
        self.assertEqual(payload["failure"], "official scorer unavailable")
        self.assertIn("rendered_code", payload)

if __name__ == "__main__":
    unittest.main()
