"""Black-box coverage of the user-facing ``autore`` command."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUTORE = ROOT / "autore"


class AutoreCliE2ETest(unittest.TestCase):
    def _run(self, *args: str, output: Path) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONPYCACHEPREFIX"] = "/private/tmp/research-harness-pycache"
        # Black-box CLI tests validate wiring without depending on the test
        # runner's access to the host Docker socket.
        env["PREDICTION_MARKET_ALLOW_UNSANDBOXED_UPSTREAM"] = "1"
        return subprocess.run(
            [str(AUTORE), *args, "--output", str(output), "--llm-provider", "local", "--llm-model", "local/test", "--quiet", "--no-sessions"],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=30,
        )

    @staticmethod
    def _only_run(output: Path) -> Path:
        runs = list(output.glob("*_run_*"))
        if len(runs) != 1:
            raise AssertionError(f"Expected exactly one run artifact, found {runs!r}")
        return runs[0]

    def test_standard_and_grader_cli_paths_persist_their_real_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            standard = self._run("Explain a concept.", "--retriever", "local", output=root / "standard")
            grader = self._run(
                "Optimize the PM challenge.", "--retriever", "local",
                "--grader", "--grader-loops", "1", output=root / "grader",
            )

            self.assertEqual(standard.returncode, 0, standard.stderr)
            self.assertEqual(grader.returncode, 0, grader.stderr)

            standard_transcript = json.loads((self._only_run(root / "standard") / "agent_messages.json").read_text(encoding="utf-8"))
            grader_root = self._only_run(root / "grader")
            grader_transcript = json.loads((grader_root / "agent_messages.json").read_text(encoding="utf-8"))
            grader_state = json.loads((grader_root / "run_state.json").read_text(encoding="utf-8"))

            # A local provider cannot fabricate a native-tool trajectory. The
            # artifact must say so rather than presenting a false success.
            self.assertEqual(standard_transcript["status"], "partial")
            self.assertIn("live model provider", str(standard_transcript["events"][-1].get("error") or "").lower())
            self.assertIn("prediction-market challenge", grader_transcript["objective"])
            self.assertIn("challenges/prediction-market-challenge/examples/starter_strategy.py", grader_transcript["objective"])
            self.assertIn("class Strategy(BaseStrategy)", grader_transcript["objective"])
            self.assertIn("evaluate_prediction_market_candidate", grader_state["available_tools"])
            self.assertTrue((grader_root / "grader_preflight.json").exists())
            self.assertNotIn("prior_artifact_memory_search", grader_state["available_tools"])
            self.assertNotIn("prior_relevant_memory", grader_state)
            self.assertNotIn("task_mode", grader_state["run"])
            self.assertNotIn("product_agent", grader_state["run"])
            self.assertFalse((grader_root / "task_ingestion_decisions.json").exists())

    def test_a_new_run_never_reads_or_exposes_prior_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "outputs"
            prior = output / "001_run_prior"
            prior.mkdir(parents=True)
            marker = "SHOULD_NOT_ENTER_THE_NEW_RUN"
            (prior / "run_state.json").write_text(
                json.dumps({"goal": "same goal", "secret_marker": marker}), encoding="utf-8"
            )
            (prior / "sources.json").write_text(
                json.dumps([{"title": marker, "url": str(prior)}]), encoding="utf-8"
            )

            completed = self._run("same goal", "--retriever", "local", output=output)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            new_run = output / "002_run_same-goal"
            state = json.loads((new_run / "run_state.json").read_text(encoding="utf-8"))
            transcript = (new_run / "agent_messages.json").read_text(encoding="utf-8")
            self.assertNotIn("prior_relevant_memory", state)
            self.assertNotIn("prior_artifact_memory_search", state["available_tools"])
            self.assertNotIn(marker, transcript)

    def test_cli_argument_errors_are_nonzero_and_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = self._run("anything", "--retriever", "not-a-retriever", output=Path(directory) / "invalid")

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("invalid choice", completed.stderr)

    def test_removed_legacy_flags_and_memory_retriever_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid"
            for args in (
                ("anything", "--retriever", "memory"),
                ("anything", "--evaluator", "prediction_market"),
                ("anything", "--mode", "optimize"),
                ("anything", "--task-mode", "optimize"),
            ):
                completed = self._run(*args, output=output)
                self.assertNotEqual(completed.returncode, 0, args)


if __name__ == "__main__":
    unittest.main()
