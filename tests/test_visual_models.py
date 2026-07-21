from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research_harness.future.evolutionary.loops import EvolutionaryOuterLoop
from research_harness.run_visuals import _timeline_data, _timeline_svg
from research_harness.schemas import EvolutionRound, Variant, VariantEvaluation
from research_harness.store import ArtifactStore
from research_harness.visual_operations import CATEGORY_COLORS, operation_for


class TimelineModelTests(unittest.TestCase):
    def _store(self, root: Path) -> ArtifactStore:
        return ArtifactStore(root, sqlite_path=root / "world.sqlite")

    def test_ordinals_are_per_canonical_operation_and_search_colors_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp))
            events = [
                {"event_type": "tool_requested", "timestamp": "2026-01-01T00:00:00+00:00", "tool_call_id": "w1", "tool_name": "web_search"},
                {"event_type": "tool_requested", "timestamp": "2026-01-01T00:00:00.100000+00:00", "tool_call_id": "o1", "tool_name": "openalex_api_search"},
                {"event_type": "tool_result", "timestamp": "2026-01-01T00:00:01+00:00", "tool_call_id": "w1", "result_status": "ok"},
                {"event_type": "tool_result", "timestamp": "2026-01-01T00:00:01.100000+00:00", "tool_call_id": "o1", "result_status": "ok"},
            ]
            data = _timeline_data(store, events)
            self.assertEqual([span["label"] for span in data["spans"]], ["Web search 1", "OpenAlex search 1"])
            self.assertEqual({span["color"] for span in data["spans"]}, {CATEGORY_COLORS["search"]})
            self.assertEqual(operation_for(tool_name="web_search").color, operation_for(tool_name="openalex_api_search").color)
            self.assertEqual(data["peak_parallel_tools"], 2)

    def test_retry_is_metadata_and_status_style_is_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = self._store(Path(tmp))
            events = [
                {"event_type": "tool_requested", "timestamp": "2026-01-01T00:00:00+00:00", "tool_call_id": "x", "tool_name": "fetch_document"},
                {"event_type": "tool_requested", "timestamp": "2026-01-01T00:00:00.200000+00:00", "tool_call_id": "x", "tool_name": "fetch_document"},
                {"event_type": "tool_result", "timestamp": "2026-01-01T00:00:01+00:00", "tool_call_id": "x", "result_status": "skipped"},
            ]
            data = _timeline_data(store, events)
            self.assertEqual(len(data["spans"]), 1)
            self.assertEqual(data["spans"][0]["label"], "Document fetch 1")
            self.assertEqual(data["spans"][0]["retry_count"], 1)
            self.assertIn("1 retry", data["spans"][0]["detail"])
            first = _timeline_svg(data)
            self.assertEqual(first, _timeline_svg(data))
            self.assertIn('stroke-dasharray="6 4"', first)
            data["spans"][0]["status"] = "failed"
            failed_svg = _timeline_svg(data)
            self.assertIn('stroke="#b91c1c"', failed_svg)
            self.assertIn('fill-opacity="0.42"', failed_svg)


class CandidateGraphTests(unittest.TestCase):
    def test_branching_and_promotion_history_are_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ArtifactStore(root, sqlite_path=root / "world.sqlite")
            base = Variant("run", 1, "proposal", "base", [], {})
            rejected = Variant("run", 2, "proposal", "branch", [base.id], {})
            revisit = Variant("run", 3, "proposal", "revisit", [base.id], {"lineage_type": "reverted_to"})
            for variant in (base, rejected, revisit):
                store.add_variant(variant)
            for variant, score, passed in ((base, 0.8, True), (rejected, 0.2, False), (revisit, 0.9, True)):
                store.add_variant_evaluation(VariantEvaluation("run", variant.id, "optimize", score, {"eligible": passed}, [], "result", passed))
            store.add_evolution_round(EvolutionRound("run", 1, "optimize", [base.id], base.id, 0.8, "continue", 0))
            store.add_evolution_round(EvolutionRound("run", 3, "optimize", [revisit.id], revisit.id, 0.9, "done", 0))
            store.append_champion_history({"sequence": 1, "candidate_id": base.id, "promoted": True})
            store.append_champion_history({"sequence": 2, "candidate_id": revisit.id, "promoted": True})
            loop = EvolutionaryOuterLoop.__new__(EvolutionaryOuterLoop)
            loop.run_id, loop._champion_variant_id, loop._champion_score = "run", revisit.id, 0.9
            loop._write_candidate_graph(store)
            graph = json.loads(store.candidate_graph_path.read_text(encoding="utf-8"))
            revisit_node = next(node for node in graph["nodes"] if node["id"] == revisit.id)
            self.assertEqual(revisit_node["parent_candidate_ids"], [base.id])
            self.assertIn({"from": base.id, "to": revisit.id, "type": "reverted_to"}, graph["edges"])
            self.assertEqual(len(json.loads(store.champion_history_path.read_text(encoding="utf-8"))), 2)
            self.assertTrue(store.champion_tree_path.exists())
            self.assertIn("deprecated", json.loads(store.champion_tree_path.read_text(encoding="utf-8")))

    def test_multiple_parents_do_not_invent_a_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = ArtifactStore(root, sqlite_path=root / "world.sqlite")
            first = Variant("run", 1, "proposal", "a", [], {})
            second = Variant("run", 1, "proposal", "b", [], {})
            child = Variant("run", 2, "proposal", "c", [first.id, second.id], {})
            for variant in (first, second, child):
                store.add_variant(variant)
                store.add_variant_evaluation(VariantEvaluation("run", variant.id, "optimize", 0.5, {}, [], "ok", True))
            loop = EvolutionaryOuterLoop.__new__(EvolutionaryOuterLoop)
            loop.run_id, loop._champion_variant_id, loop._champion_score = "run", child.id, 0.5
            loop._write_candidate_graph(store)
            graph = json.loads(store.candidate_graph_path.read_text(encoding="utf-8"))
            child_edges = [edge for edge in graph["edges"] if edge["to"] == child.id]
            self.assertEqual({edge["type"] for edge in child_edges}, {"derived_from"})


if __name__ == "__main__":
    unittest.main()
