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
            grader = self._run("Optimize the PM challenge.", "--retriever", "local", "--grader", "prediction_market", output=root / "grader")

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
            self.assertIn("evaluate_prediction_market_candidate", grader_state["available_tools"])

    def test_cli_argument_errors_are_nonzero_and_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = self._run("anything", "--retriever", "not-a-retriever", output=Path(directory) / "invalid")

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("invalid choice", completed.stderr)


if __name__ == "__main__":
    unittest.main()
