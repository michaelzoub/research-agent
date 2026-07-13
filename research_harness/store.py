from __future__ import annotations

import json
import hashlib
import os
import re
import sqlite3
import sys
import threading
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Optional

from .diagnostics import classify_failure, diagnose_snapshot
from .schemas import (
    AgentTrace,
    Claim,
    Contradiction,
    CostEvent,
    Experiment,
    FailedPath,
    HarnessChange,
    Hypothesis,
    LoopIteration,
    LoopTask,
    LoopContinuationDecision,
    OpenQuestion,
    ProvenanceEdge,
    RunRecord,
    Source,
    Variant,
    VariantEvaluation,
    EvolutionRound,
    now_iso,
    to_dict,
)


ENTITY_FILES = {
    "sources": "sources.json",
    "claims": "claims.json",
    "hypotheses": "hypotheses.json",
    "experiments": "experiments.json",
    "open_questions": "open_questions.json",
    "contradictions": "contradictions.json",
    "failed_paths": "failed_paths.json",
    "harness_changes": "harness_changes.json",
    "runs": "runs.json",
    "agent_traces": "agent_traces.json",
    "provenance_edges": "provenance_edges.json",
    "cost_events": "cost_events.json",
    "harness_diagnoses": "harness_diagnoses.json",
    "loop_tasks": "tasks.json",
    "loop_iterations": "loop_iterations.json",
    "task_ingestion_decisions": "task_ingestion_decisions.json",
    "variants": "variants.json",
    "variant_evaluations": "variant_evaluations.json",
    "evolution_rounds": "evolution_rounds.json",
    "loop_continuation_decisions": "loop_continuation_decisions.json",
}


class ArtifactStore:
    """File-backed shared artifact store for one workspace.

    The store writes canonical JSON collections plus a JSONL trace stream. It is
    intentionally small, deterministic, and easy to inspect.
    """

    def __init__(
        self,
        root: Path,
        echo_progress: bool = False,
        session_store: Optional[Any] = None,
        sqlite_path: Optional[Path] = None,
    ):
        self.root = root
        self._write_lock = threading.RLock()
        self.echo_progress = echo_progress
        self.session_store = session_store
        self.root.mkdir(parents=True, exist_ok=True)
        self.trace_log_path = self.root / "trace.jsonl"
        self.cost_path = self.root / "cost.json"
        self.sqlite_path = sqlite_path or self.root.parent / "world_model.sqlite"
        self.report_path = self.root / "final_report.md"
        self.report_tex_path = self.root / "final_report.tex"
        self.report_pdf_path = self.root / "final_report.pdf"
        self.report_preview_path = self.root / "final_report_preview.png"
        self.run_state_path = self.root / "run_state.json"
        self.grader_preflight_path = self.root / "grader_preflight.json"
        self.optimizer_seed_context_path = self.root / "optimizer_seed_context.json"
        self.optimizer_agent_steps_path = self.root / "optimization_agent_steps.json"
        self.optimizer_agent_summary_path = self.root / "optimizer_agent_summary.md"
        self.role_trajectory_contract_path = self.root / "role_trajectory_contract.md"
        self.optimized_candidate_path = self.root / "optimized_candidate.txt"
        self.optimal_code_path = self.root / "optimal_code.py"
        self.candidates_dir = self.root / "candidates"
        self.champions_dir = self.root / "champions"
        self.champion_tree_path = self.root / "champion_tree.json"
        self.champion_tree_graph_path = self.root / "champion_tree.png"
        self.champion_tree_svg_path = self.root / "champion_tree.svg"
        self.champion_tree_mermaid_path = self.root / "champion_tree.mmd"
        self.current_champion_path = self.root / "current_champion.json"
        self.optimization_result_path = self.root / "optimization_result.json"
        self.optimization_trials_dir = self.root / "optimization_trials"
        self.solution_path = self.root / "solution.py"
        self.run_benchmark_path = self.root / "run_benchmark.html"
        self.decision_dag_path = self.root / "decision_dag.png"
        self.agent_timeline_path = self.root / "agent_timeline.png"
        self.agent_timeline_svg_path = self.root / "agent_timeline.svg"
        self.score_improvement_path = self.root / "score_improvement.png"
        self.score_improvement_svg_path = self.root / "score_improvement.svg"
        self.run_notebook_path = self.root / "run_notebook.ipynb"
        self.harness_diagnosis_path = self.root / "harness_diagnosis.json"
        self.loop_continuation_path = self.root / "loop_continuation_decisions.json"
        self.agent_transcript_path = self.root / "agent_messages.json"
        self.agent_event_log_path = self.root / "agent_events.jsonl"
        self.user_steering_inbox_path = self.root / "user_steering_inbox.jsonl"
        self.user_steering_state_path = self.root / "user_steering_state.json"
        self.progress_path = self.root / "progress.txt"
        self.learnings_path = self.root / "learnings.md"
        self.learning_log_path = self.root / "learnings.jsonl"
        if not self.progress_path.exists():
            self.progress_path.write_text("", encoding="utf-8")
        if not self.agent_event_log_path.exists():
            self.agent_event_log_path.write_text("", encoding="utf-8")
        if not self.learning_log_path.exists():
            self.learning_log_path.write_text("", encoding="utf-8")
        self._migrate_sqlite()

    def append_learning(self, *, title: str, finding: str, evidence: str, status: str, run_id: str) -> Path:
        """Persist a controller-selected learning in readable and replayable forms."""
        payload = {"run_id": run_id, "title": title, "finding": finding, "evidence": evidence, "status": status, "created_at": now_iso()}
        with self._write_lock:
            with self.learnings_path.open("a", encoding="utf-8") as handle:
                handle.write(f"## {title}\n\n{finding}\n\nEvidence: {evidence}\n\nStatus: {status}\n\n")
            with self.learning_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._record_artifact_write(self.learnings_path, "learnings")
        self._record_artifact_write(self.learning_log_path, "learning_log")
        return self.learnings_path

    def add_source(self, source: Source) -> Source:
        with self._write_lock:
            return self._add_source_locked(source)

    def _add_source_locked(self, source: Source) -> Source:
        # Primary dedup: exact URL match.
        existing = self.find_by("sources", "url", source.url)
        if existing:
            return Source(**existing)
        # Secondary dedup: normalized title match catches the same paper returned
        # by different retrievers with different URLs (e.g. arXiv vs OpenAlex).
        normalized = _normalize_title(source.title)
        if normalized:
            for row in self.list("sources"):
                if _normalize_title(str(row.get("title", ""))) == normalized:
                    return Source(**row)
        self._annotate_dedup("sources", source, _canonical_key("sources", to_dict(source)))
        self._append("sources", source)
        return source

    def commit_tool_sources(self, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Commit one completed tool result as an ordered, atomic source batch.

        Network tools may run concurrently, but their completed observations are
        committed in model call order.  The lock prevents a future background
        producer from turning read-modify-write JSON collections into lost
        updates; SQLite mirrors are committed in the same critical section.
        """
        committed: list[dict[str, Any]] = []
        with self._write_lock:
            for raw in sources:
                source = Source(**raw)
                committed.append(to_dict(self._add_source_locked(source)))
        return committed

    def add_claim(self, claim: Claim) -> Claim:
        self._annotate_dedup("claims", claim, _canonical_key("claims", to_dict(claim)))
        self._append("claims", claim)
        for source_id in claim.source_ids:
            self.add_provenance_edge(
                ProvenanceEdge(
                    run_id=claim.run_id,
                    from_type="source",
                    from_id=source_id,
                    to_type="claim",
                    to_id=claim.id,
                    relationship="supports",
                    metadata={"confidence": claim.confidence, "support_level": claim.support_level},
                )
            )
        if claim.duplicate_of:
            self.add_provenance_edge(
                ProvenanceEdge(
                    run_id=claim.run_id,
                    from_type="claim",
                    from_id=claim.id,
                    to_type="claim",
                    to_id=claim.duplicate_of,
                    relationship="duplicate_of",
                )
            )
        return claim

    def add_hypothesis(self, hypothesis: Hypothesis) -> Hypothesis:
        if hypothesis.run_id is None:
            hypothesis.run_id = self.root.name
        self._annotate_dedup("hypotheses", hypothesis, _canonical_key("hypotheses", to_dict(hypothesis)))
        self._append("hypotheses", hypothesis)
        for claim_id in hypothesis.supporting_claim_ids:
            self.add_provenance_edge(
                ProvenanceEdge(
                    run_id=hypothesis.run_id or self.root.name,
                    from_type="claim",
                    from_id=claim_id,
                    to_type="hypothesis",
                    to_id=hypothesis.id,
                    relationship="supports",
                    metadata={"confidence": hypothesis.confidence},
                )
            )
        for claim_id in hypothesis.contradicting_claim_ids:
            self.add_provenance_edge(
                ProvenanceEdge(
                    run_id=hypothesis.run_id or self.root.name,
                    from_type="claim",
                    from_id=claim_id,
                    to_type="hypothesis",
                    to_id=hypothesis.id,
                    relationship="contradicts",
                )
            )
        return hypothesis

    def add_experiment(self, experiment: Experiment) -> Experiment:
        self._append("experiments", experiment)
        return experiment

    def add_open_question(self, question: OpenQuestion) -> OpenQuestion:
        self._append("open_questions", question)
        return question

    def add_contradiction(self, contradiction: Contradiction) -> Contradiction:
        self._append("contradictions", contradiction)
        run_id = self.root.name
        for claim_id in [contradiction.claim_a, contradiction.claim_b]:
            self.add_provenance_edge(
                ProvenanceEdge(
                    run_id=run_id,
                    from_type="claim",
                    from_id=claim_id,
                    to_type="contradiction",
                    to_id=contradiction.id,
                    relationship="contradicts",
                    metadata={"severity": contradiction.severity},
                )
            )
        return contradiction

    def add_failed_path(self, failed_path: FailedPath) -> FailedPath:
        detail = classify_failure(failed_path.reason, component=failed_path.failure_component)
        failed_path.failure_category = detail["category"]
        failed_path.failure_component = detail["component"]
        failed_path.retryable = bool(detail["retryable"])
        failed_path.severity = str(detail["severity"])
        self._append("failed_paths", failed_path)
        return failed_path

    def add_harness_change(self, change: HarnessChange) -> HarnessChange:
        self._append("harness_changes", change)
        return change

    def add_run(self, run: RunRecord) -> RunRecord:
        self._append("runs", run)
        self._upsert_run_observability(run.id, run.harness_config_snapshot, run.prompt_versions, {})
        return run

    def add_loop_task(self, task: LoopTask) -> LoopTask:
        self._append("loop_tasks", task)
        return task

    def update_loop_task(self, task: LoopTask) -> None:
        rows = self.list("loop_tasks")
        for index, row in enumerate(rows):
            if row["id"] == task.id:
                rows[index] = to_dict(task)
                self._write("loop_tasks", rows)
                return
        self.add_loop_task(task)

    def add_loop_iteration(self, iteration: LoopIteration) -> LoopIteration:
        self._append("loop_iterations", iteration)
        return iteration

    def add_variant(self, variant: Variant) -> Variant:
        self._append("variants", variant)
        return variant

    def add_variant_evaluation(self, evaluation: VariantEvaluation) -> VariantEvaluation:
        self._append("variant_evaluations", evaluation)
        return evaluation

    def add_evolution_round(self, round_record: EvolutionRound) -> EvolutionRound:
        self._append("evolution_rounds", round_record)
        return round_record

    def add_loop_continuation_decision(self, decision: LoopContinuationDecision) -> LoopContinuationDecision:
        self._append("loop_continuation_decisions", decision)
        return decision

    def add_provenance_edge(self, edge: ProvenanceEdge) -> ProvenanceEdge:
        for row in self.list("provenance_edges"):
            if (
                row.get("from_type") == edge.from_type
                and row.get("from_id") == edge.from_id
                and row.get("to_type") == edge.to_type
                and row.get("to_id") == edge.to_id
                and row.get("relationship") == edge.relationship
            ):
                return ProvenanceEdge(**row)
        self._append("provenance_edges", edge)
        self._mirror_provenance_edge(to_dict(edge))
        return edge

    def add_cost_event(self, event: CostEvent) -> CostEvent:
        self._append("cost_events", event)
        return event

    def append_user_steering(self, text: str) -> None:
        clean = text.strip()
        if not clean:
            return
        payload = {
            "text": clean,
            "created_at": now_iso(),
        }
        with self.user_steering_inbox_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        if self.session_store is not None:
            self.session_store.append_event("user_steering", payload)

    def ingest_pending_user_steering(self, run_id: Optional[str] = None) -> int:
        if not self.user_steering_inbox_path.exists():
            return 0
        lines = self.user_steering_inbox_path.read_text(encoding="utf-8").splitlines()
        state = _read_json(self.user_steering_state_path, {"consumed": 0})
        consumed = max(0, int(state.get("consumed") or 0))
        ingested = 0
        for index, line in enumerate(lines[consumed:], start=consumed):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                consumed = index + 1
                continue
            text = str(payload.get("text") or "").strip()
            if not text:
                consumed = index + 1
                continue
            source = self.add_source(_source_from_user_steering(text, run_id or self.root.name))
            self.add_claim(
                Claim(
                    text=f"User steering: {_steering_summary(text)}",
                    source_ids=[source.id],
                    confidence=0.9,
                    support_level="user_provided",
                    created_by_agent="user_steering",
                    run_id=run_id or self.root.name,
                )
            )
            ingested += 1
            consumed = index + 1
        self.user_steering_state_path.write_text(json.dumps({"consumed": consumed}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if ingested:
            self.append_progress(f"User steering: ingested {ingested} new note/article(s) into the next round context")
        return ingested

    def append_progress(self, text: str) -> None:
        if self.echo_progress:
            print(_format_progress_for_terminal(text), flush=True)
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(text.rstrip() + "\n")
        if self.session_store is not None:
            self.session_store.append_event("progress", {"text": text.rstrip()})

    def update_run(self, run: RunRecord) -> None:
        rows = self.list("runs")
        for index, row in enumerate(rows):
            if row["id"] == run.id:
                rows[index] = to_dict(run)
                self._write("runs", rows)
                self._mirror_to_sqlite("runs", rows[index])
                self._upsert_run_observability(run.id, run.harness_config_snapshot, run.prompt_versions, _read_json(self.cost_path, {}))
                return
        self.add_run(run)

    def add_trace(self, trace: AgentTrace) -> AgentTrace:
        if trace.status != "completed" or trace.errors:
            detail = classify_failure(" ".join(trace.errors), component=trace.failure_component)
            trace.failure_category = detail["category"]
            trace.failure_component = detail["component"]
            trace.retryable = bool(detail["retryable"])
        elif trace.failure_category == "none":
            trace.failure_component = trace.failure_component if trace.failure_component != "unknown" else _component_from_role(trace.role, trace.agent_name)
        self._append("agent_traces", trace)
        payload = to_dict(trace)
        with self.trace_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        if self.session_store is not None:
            self.session_store.append_event("agent_trace", payload)
        return trace

    def list(self, entity: str) -> list[dict[str, Any]]:
        path = self.root / ENTITY_FILES[entity]
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def snapshot(self) -> dict[str, list[dict[str, Any]]]:
        return {entity: self.list(entity) for entity in ENTITY_FILES}

    def write_report(self, text: str) -> Path:
        self._snapshot_before_write(self.report_path, "before writing final report")
        self.report_path.write_text(text, encoding="utf-8")
        self._record_artifact_write(self.report_path, "report")
        report_id = self.report_path.name
        for entity, from_type in [("claims", "claim"), ("hypotheses", "hypothesis"), ("contradictions", "contradiction")]:
            for row in self.list(entity):
                self.add_provenance_edge(
                    ProvenanceEdge(
                        run_id=str(row.get("run_id") or self.root.name),
                        from_type=from_type,
                        from_id=str(row.get("id")),
                        to_type="report",
                        to_id=report_id,
                        relationship="cited_or_summarized_by",
                    )
                )
        return self.report_path

    def write_report_tex(self, text: str) -> Path:
        self._snapshot_before_write(self.report_tex_path, "before writing final report TeX")
        self.report_tex_path.write_text(text, encoding="utf-8")
        self._record_artifact_write(self.report_tex_path, "report_tex")
        return self.report_tex_path

    def write_report_pdf(self, payload: bytes) -> Path:
        self._snapshot_before_write(self.report_pdf_path, "before writing final report PDF")
        self.report_pdf_path.write_bytes(payload)
        self._record_artifact_write(self.report_pdf_path, "report_pdf")
        return self.report_pdf_path

    def write_report_preview(self, payload: bytes) -> Path:
        self._snapshot_before_write(self.report_preview_path, "before writing final report preview")
        self.report_preview_path.write_bytes(payload)
        self._record_artifact_write(self.report_preview_path, "report_preview")
        return self.report_preview_path

    def write_run_state(self, payload: dict[str, Any]) -> Path:
        self._snapshot_before_write(self.run_state_path, "before writing run state")
        self.run_state_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._record_artifact_write(self.run_state_path, "run_state")
        return self.run_state_path

    def write_grader_preflight(self, payload: dict[str, Any]) -> Path:
        self.grader_preflight_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._record_artifact_write(self.grader_preflight_path, "grader_preflight")
        return self.grader_preflight_path

    def write_optimizer_seed_context(self, payload: dict[str, Any]) -> Path:
        self._snapshot_before_write(self.optimizer_seed_context_path, "before writing optimizer seed context")
        self.optimizer_seed_context_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._record_artifact_write(self.optimizer_seed_context_path, "optimizer_seed_context")
        return self.optimizer_seed_context_path

    def write_agent_transcript(self, payload: dict[str, Any]) -> Path:
        self._snapshot_before_write(self.agent_transcript_path, "before writing agent message transcript")
        self.agent_transcript_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        self._record_artifact_write(self.agent_transcript_path, "agent_messages")
        return self.agent_transcript_path

    def append_agent_event(self, event: dict[str, Any]) -> None:
        """Durably record each observed model turn and tool result as it occurs."""
        with self.agent_event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
        if self.session_store is not None:
            self.session_store.append_event("agent_event", event)

    def write_solution(self, text: str) -> Path:
        self._snapshot_before_write(self.solution_path, "before writing solution code")
        self.solution_path.write_text(text, encoding="utf-8")
        self._record_artifact_write(self.solution_path, "solution")
        return self.solution_path

    def write_optimized_candidate(self, text: str) -> Path:
        self._snapshot_before_write(self.optimized_candidate_path, "before writing optimized candidate")
        self.optimized_candidate_path.write_text(text, encoding="utf-8")
        self._record_artifact_write(self.optimized_candidate_path, "optimized_candidate")
        return self.optimized_candidate_path

    def write_optimal_code(self, text: str) -> Path:
        self._snapshot_before_write(self.optimal_code_path, "before writing optimal code")
        self.optimal_code_path.write_text(text, encoding="utf-8")
        self._record_artifact_write(self.optimal_code_path, "optimal_code")
        return self.optimal_code_path

    def write_optimization_result(self, payload: dict[str, Any]) -> Path:
        self._snapshot_before_write(self.optimization_result_path, "before writing optimization result")
        self.optimization_result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._record_artifact_write(self.optimization_result_path, "optimization_result")
        return self.optimization_result_path

    def write_optimization_trial(self, trial_id: str, payload: dict[str, Any]) -> Path:
        """Persist one immutable candidate-evaluation record for audit and replay."""
        self.optimization_trials_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", trial_id).strip("._") or "trial"
        path = self.optimization_trials_dir / f"{safe_id}.json"
        self._snapshot_before_write(path, "before writing optimization trial")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        self._record_artifact_write(path, "optimization_trial")
        return path

    def write_optimization_trial_code(self, trial_id: str, code: str) -> Path:
        """Write the evaluator-ready Python artifact beside its trial record."""
        self.optimization_trials_dir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", trial_id).strip("._") or "trial"
        path = self.optimization_trials_dir / f"{safe_id}.py"
        self._snapshot_before_write(path, "before writing optimization trial code")
        path.write_text(code, encoding="utf-8")
        self._record_artifact_write(path, "optimization_trial_code")
        return path

    def write_round_champion(self, round_index: int, payload: dict[str, Any]) -> Path:
        self.champions_dir.mkdir(parents=True, exist_ok=True)
        path = self.champions_dir / f"round_{round_index:03d}_champion.json"
        self._snapshot_before_write(path, "before writing round champion")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.current_champion_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._record_artifact_write(path, "round_champion")
        self._record_artifact_write(self.current_champion_path, "current_champion")
        return path

    def write_champion_tree(self, payload: dict[str, Any]) -> Path:
        self._snapshot_before_write(self.champion_tree_path, "before writing champion tree")
        self.champion_tree_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._record_artifact_write(self.champion_tree_path, "champion_tree")
        return self.champion_tree_path

    def write_cost(self, payload: dict[str, Any]) -> Path:
        self._snapshot_before_write(self.cost_path, "before writing cost summary")
        self.cost_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._record_artifact_write(self.cost_path, "cost")
        self._upsert_run_observability(str(payload.get("run_id") or self.root.name), {}, {}, payload)
        return self.cost_path

    def write_harness_diagnosis(self, payload: Optional[dict[str, Any]] = None) -> Path:
        diagnosis = payload or diagnose_snapshot(self.snapshot())
        self._snapshot_before_write(self.harness_diagnosis_path, "before writing harness diagnosis")
        self.harness_diagnosis_path.write_text(json.dumps(diagnosis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._record_artifact_write(self.harness_diagnosis_path, "harness_diagnosis")
        self._append("harness_diagnoses", {"id": f"diagnosis_{self.root.name}", "run_id": self.root.name, **diagnosis})
        return self.harness_diagnosis_path

    def find_by(self, entity: str, key: str, value: Any) -> Optional[dict[str, Any]]:
        return next((row for row in self.list(entity) if row.get(key) == value), None)

    def _append(self, entity: str, value: Any) -> None:
        with self._write_lock:
            rows = self.list(entity)
            row = to_dict(value) if is_dataclass(value) else value
            rows.append(row)
            self._write(entity, rows)
            self._mirror_to_sqlite(entity, row)

    def _write(self, entity: str, rows: list[dict[str, Any]]) -> None:
        path = self.root / ENTITY_FILES[entity]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _snapshot_before_write(self, path: Path, reason: str) -> None:
        if self.session_store is not None and path.exists():
            self.session_store.snapshot_files([path], reason=reason)

    def _record_artifact_write(self, path: Path, kind: str) -> None:
        if self.session_store is not None:
            self.session_store.append_event("artifact_write", {"kind": kind, "path": str(path)})

    def _migrate_sqlite(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        migrations_dir = Path(__file__).resolve().parent / "migrations"
        with sqlite3.connect(self.sqlite_path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            applied = {row[0] for row in connection.execute("SELECT version FROM schema_migrations")}
            for migration in sorted(migrations_dir.glob("*.sql")):
                version = migration.stem
                if version in applied:
                    continue
                connection.executescript(migration.read_text(encoding="utf-8"))
                connection.execute(
                    "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, datetime('now'))",
                    (version,),
                )
            connection.commit()

    def _annotate_dedup(self, entity: str, value: Any, canonical_key: str) -> None:
        if not canonical_key:
            return
        row = to_dict(value) if is_dataclass(value) else dict(value)
        duplicate = self._find_sqlite_duplicate(entity, canonical_key)
        if duplicate is None and entity == "sources":
            duplicate = self._find_source_duplicate_by_title(row)
        if duplicate is None:
            setattr(value, "canonical_id", getattr(value, "id", None))
            return
        duplicate_id = str(duplicate["id"])
        setattr(value, "canonical_id", duplicate_id)
        setattr(value, "duplicate_of", duplicate_id)

    def _find_sqlite_duplicate(self, entity: str, canonical_key: str) -> Optional[dict[str, Any]]:
        with sqlite3.connect(self.sqlite_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT id, run_id
                FROM artifacts
                WHERE entity = ? AND canonical_key = ?
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (entity, canonical_key),
            ).fetchone()
            return dict(row) if row else None

    def _find_source_duplicate_by_title(self, row: dict[str, Any]) -> Optional[dict[str, Any]]:
        normalized = _normalize_title(str(row.get("title") or ""))
        if not normalized:
            return None
        with sqlite3.connect(self.sqlite_path) as connection:
            for artifact_id, run_id, payload_json in connection.execute(
                "SELECT id, run_id, payload_json FROM artifacts WHERE entity = 'sources' ORDER BY created_at ASC"
            ):
                try:
                    payload = json.loads(payload_json)
                except json.JSONDecodeError:
                    continue
                if _normalize_title(str(payload.get("title") or "")) == normalized:
                    return {"id": artifact_id, "run_id": run_id}
        return None

    def _mirror_to_sqlite(self, entity: str, row: dict[str, Any]) -> None:
        if entity == "provenance_edges":
            self._mirror_provenance_edge(row)
            return
        row_id = str(row.get("id") or f"{entity}_{len(self.list(entity))}")
        run_id = str(row.get("run_id") or self.root.name)
        canonical_key = _canonical_key(entity, row)
        created_at = str(row.get("created_at") or row.get("retrieved_at") or row.get("started_at") or row.get("completed_at") or "")
        with sqlite3.connect(self.sqlite_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO artifacts(entity, id, run_id, canonical_key, duplicate_of, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, COALESCE(NULLIF(?, ''), datetime('now')))
                """,
                (
                    entity,
                    row_id,
                    run_id,
                    canonical_key,
                    row.get("duplicate_of"),
                    json.dumps(row, sort_keys=True),
                    created_at,
                ),
            )
            connection.commit()

    def _mirror_provenance_edge(self, row: dict[str, Any]) -> None:
        with sqlite3.connect(self.sqlite_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO provenance_edges(
                  id, run_id, from_type, from_id, to_type, to_id, relationship, metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row.get("id"),
                    row.get("run_id") or self.root.name,
                    row.get("from_type"),
                    row.get("from_id"),
                    row.get("to_type"),
                    row.get("to_id"),
                    row.get("relationship"),
                    json.dumps(row.get("metadata") or {}, sort_keys=True),
                    row.get("created_at") or "",
                ),
            )
            connection.commit()

    def _upsert_run_observability(
        self,
        run_id: str,
        harness_config: dict[str, Any],
        prompt_versions: dict[str, str],
        cost: dict[str, Any],
    ) -> None:
        existing_config = harness_config
        existing_prompts = prompt_versions
        if not existing_config or not existing_prompts:
            run = self.find_by("runs", "id", run_id)
            if run:
                existing_config = existing_config or dict(run.get("harness_config_snapshot") or {})
                existing_prompts = existing_prompts or dict(run.get("prompt_versions") or {})
        with sqlite3.connect(self.sqlite_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO run_observability(
                  run_id, harness_config_json, prompt_versions_json, cost_json, created_at
                )
                VALUES (?, ?, ?, ?, datetime('now'))
                """,
                (
                    run_id,
                    json.dumps(existing_config, sort_keys=True, default=str),
                    json.dumps(existing_prompts, sort_keys=True),
                    json.dumps(cost, sort_keys=True, default=str),
                ),
            )
            connection.commit()


def _normalize_title(title: str) -> str:
    """Return a lowercase, punctuation-stripped title prefix for dedup comparison."""
    normalized = re.sub(r"[^a-z0-9]", "", title.lower())
    return normalized[:80]


def _canonical_key(entity: str, row: dict[str, Any]) -> str:
    if entity == "sources":
        url = str(row.get("url") or "").strip().lower()
        if url:
            return f"url:{url}"
        title = _normalize_title(str(row.get("title") or ""))
        return f"title:{title}" if title else ""
    if entity in {"claims", "hypotheses"}:
        text = re.sub(r"\s+", " ", str(row.get("text") or "").strip().lower())
        text = re.sub(r"[^a-z0-9 ]", "", text)
        return f"text:{text[:240]}" if text else ""
    if entity == "runs":
        return str(row.get("id") or "")
    return str(row.get("id") or "")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _source_from_user_steering(text: str, run_id: str) -> Source:
    url, title, summary = _parse_user_steering_text(text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return Source(
        url=url or f"memory://user-steering/{run_id}/{digest}",
        title=title or _steering_summary(text, limit=80),
        author="user",
        date=now_iso().split("T")[0],
        source_type="user_steering",
        summary=summary,
        relevance_score=1.0,
        credibility_score=0.85,
        evidence_sections={"user_supplied_text": text},
    )


def _parse_user_steering_text(text: str) -> tuple[str, str, str]:
    clean = text.strip()
    if clean.startswith("/article"):
        clean = clean[len("/article"):].strip()
    elif clean.startswith("/steer"):
        clean = clean[len("/steer"):].strip()
    elif clean.startswith("/note"):
        clean = clean[len("/note"):].strip()
    parts = [part.strip() for part in clean.split("|")]
    url_match = re.search(r"https?://\S+", clean)
    url = url_match.group(0).rstrip(".,);]") if url_match else ""
    if len(parts) >= 3:
        first_is_url = parts[0].startswith(("http://", "https://"))
        return parts[0] if first_is_url else url, parts[1] if first_is_url else parts[0], parts[2]
    if len(parts) == 2:
        first_is_url = parts[0].startswith(("http://", "https://"))
        return parts[0] if first_is_url else url, parts[1] if first_is_url else parts[0], parts[1] if first_is_url else parts[1]
    title = _steering_summary(clean, limit=80)
    return url, title, clean


def _steering_summary(text: str, limit: int = 240) -> str:
    clean = re.sub(r"\s+", " ", text.strip())
    return clean if len(clean) <= limit else clean[: limit - 1].rstrip() + "…"


def _component_from_role(role: str, agent_name: str) -> str:
    text = f"{role} {agent_name}".lower()
    if any(term in text for term in ["search", "literature", "retriever", "memory"]):
        return "retrieval"
    if "hypothesis" in text:
        return "hypothesis_generation"
    if "critic" in text:
        return "critic"
    if "synthesis" in text:
        return "synthesis"
    if any(term in text for term in ["optimize", "evaluator", "prediction_market"]):
        return "optimizer"
    if any(term in text for term in ["router", "loop_controller"]):
        return "loop_control"
    if "orchestration" in text:
        return "orchestration"
    if "debug" in text:
        return "harness_debugger"
    return "unknown"


def _format_progress_for_terminal(text: str) -> str:
    if os.environ.get("NO_COLOR") or os.environ.get("RESEARCH_HARNESS_COLOR") == "0":
        return text
    if not sys.stdout.isatty() and os.environ.get("RESEARCH_HARNESS_COLOR") != "1":
        return text
    reset = "\033[0m"
    bold = "\033[1m"
    dim = "\033[2m"
    red_italic = "\033[31;3m"
    cyan_bold = "\033[1;36m"
    green_bold = "\033[1;32m"
    yellow_bold = "\033[1;33m"
    magenta_bold = "\033[1;35m"

    lowered = text.lower()
    if any(term in lowered for term in ["error", "failed", "traceback", "fallback:", "http error", "timeout"]):
        return f"{red_italic}{text}{reset}"
    if text.startswith("# "):
        return f"{bold}{text}{reset}"
    if re.match(r"^(Starting run|Execution mode|Goal|Run state|Run:|Status|Artifacts):", text):
        return f"{cyan_bold}{text}{reset}"
    if re.match(r"^(Task \d+: passed|<promise>COMPLETE</promise>)", text):
        return f"{green_bold}{text}{reset}"
    if re.match(r"^(Optimization-query phase|Optimizer phase|Prediction-market optimizer round|Outer \d+:|Literature grounding|Literature refresh)", text):
        return f"{magenta_bold}{text}{reset}"
    if re.match(r"^(Retriever search|Retriever done|Optimized candidate|Optimal code|Solution|Optimization result|Report|Run benchmark|Decision DAG):", text):
        return f"{yellow_bold}{text}{reset}"
    if text.startswith("  ") or "LLM judge" in text or "thinking" in lowered:
        return f"{dim}{text}{reset}"
    return text
