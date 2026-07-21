from __future__ import annotations

import io
import json
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

from research_harness.cli_progress import CLIProgressRenderer
from research_harness.cli import _banner_left_lines
from research_harness.run_visuals import build_parent_trace_projection
from research_harness.store import ArtifactStore
from research_harness.tools.base import ToolContext
from research_harness.tools.graders import CompareCandidateToChampionTool


class _TTY(io.StringIO):
    def isatty(self):
        return True


class ProgressRendererTests(unittest.TestCase):
    def test_banner_uses_large_autore_wordmark_without_stickman(self):
        banner = "\n".join(_banner_left_lines(46))
        self.assertIn("_____ ___  ____  _____", banner)
        self.assertIn("<AUTORE>", banner)
        self.assertNotIn("◉", banner)
        self.assertNotIn("▴", banner)

    def test_nested_worker_active_and_completed_snapshots(self):
        renderer = CLIProgressRenderer(stream=_TTY(), enabled=False)
        renderer.consume({"event_type": "tool_requested", "tool_call_id": "d", "tool_name": "delegate_task", "arguments": {"profile": "researcher"}})
        self.assertIn("Literature worker · researcher", renderer.render())
        renderer.consume({"event_type": "tool_result", "tool_call_id": "d", "tool_name": "delegate_task", "result_status": "ok"})
        self.assertIn("✓ Literature worker", renderer.render())

    def test_failed_and_interrupted_states(self):
        renderer = CLIProgressRenderer(stream=_TTY(), enabled=False)
        renderer.consume({"event_type": "tool_requested", "tool_call_id": "x", "tool_name": "fetch_document"})
        renderer.consume({"event_type": "tool_result", "tool_call_id": "x", "tool_name": "fetch_document", "result_status": "error"})
        self.assertIn("✗ fetch_document", renderer.render())
        renderer.close()  # must be safe even when animation is disabled

    def test_non_tty_disables_animation(self):
        renderer = CLIProgressRenderer(stream=io.StringIO(), enabled=True)
        self.assertFalse(renderer.enabled)

    def test_tty_refreshes_between_events(self):
        stream = _TTY()
        with patch.dict("os.environ", {"NO_COLOR": "", "CI": ""}):
            renderer = CLIProgressRenderer(stream=stream, enabled=True)
            renderer.consume({"event_type": "model_request", "model_call_id": "m"})
            first = stream.getvalue()
            time.sleep(0.2)
            renderer.close()
        self.assertGreater(len(stream.getvalue()), len(first))


class HierarchicalTraceTests(unittest.TestCase):
    def test_nested_failed_worker_relationship_and_duplicate_prevention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ArtifactStore(root, sqlite_path=root / "world.sqlite")
            worker = root / "workers" / "worker_1"
            worker.mkdir(parents=True)
            event = {"sequence": 1, "event_type": "model_turn", "timestamp": "2026-01-01T00:00:01+00:00", "run_id": "worker_1", "result_status": "error"}
            (worker / "agent_events.jsonl").write_text(json.dumps(event) + "\n" + json.dumps(event) + "\n", encoding="utf-8")
            (worker / "worker_result.json").write_text(json.dumps({"worker_run_id": "worker_1", "parent_run_id": "parent", "profile": "researcher", "status": "failed", "runtime_ms": 12, "total_tokens": 3, "cost_usd": .01, "artifacts_path": str(worker), "events_path": str(worker / "agent_events.jsonl")}), encoding="utf-8")
            parent = [{"sequence": 1, "event_type": "tool_result", "timestamp": "2026-01-01T00:00:02+00:00", "tool_name": "delegate_task", "tool_call_id": "d", "observation": {"data": {"worker_run_id": "worker_1"}}}]
            merged = build_parent_trace_projection(store, parent)
            worker_rows = [row for row in merged if row.get("worker_run_id") == "worker_1"]
            self.assertEqual(len(worker_rows), 1)
            self.assertEqual(worker_rows[0]["parent_span_id"], "tool:d")
            self.assertEqual(worker_rows[0]["worker_status"], "failed")


class CandidateComparisonTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_incompatible_and_no_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ArtifactStore(root, sqlite_path=root / "world.sqlite")
            for candidate_id, score, version in (("champ", 1.0, "v1"), ("candidate", 1.2, "v1")):
                store.write_optimization_trial(candidate_id, {"trial_id": candidate_id, "grader_id": "g", "grader_version": version, "official_measured": True, "score_eligible": True, "score": score, "metrics": {"quality": score}})
            context = ToolContext(root, store=store, run_id="run")
            result = await CompareCandidateToChampionTool().execute({"candidate_id": "candidate", "champion_id": "champ"}, context)
            self.assertTrue(result.data["compatible"])
            self.assertAlmostEqual(result.data["overall_delta"], .2)
            self.assertFalse(store.champion_history_path.exists())
            trial = json.loads((store.optimization_trials_dir / "candidate.json").read_text())
            trial["grader_version"] = "v2"
            (store.optimization_trials_dir / "candidate.json").write_text(json.dumps(trial))
            incompatible = await CompareCandidateToChampionTool().execute({"candidate_id": "candidate", "champion_id": "champ"}, context)
            self.assertFalse(incompatible.data["compatible"])
            self.assertEqual(incompatible.data["recommended_status"], "incompatible")

    async def test_missing_candidate_and_no_champion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ArtifactStore(root, sqlite_path=root / "world.sqlite")
            context = ToolContext(root, store=store)
            missing = await CompareCandidateToChampionTool().execute({"candidate_id": "missing"}, context)
            self.assertEqual(missing.status, "error")


if __name__ == "__main__":
    unittest.main()
