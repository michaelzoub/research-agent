from __future__ import annotations

import asyncio
import json
import os
import shutil
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from ..orchestrator import HarnessConfig, Orchestrator
from ..schemas import now_iso
from ..store import ArtifactStore
from .graders import aggregate_results, default_graders, _grade_parallel_trial_isolation_from_trials
from .trajectory import outcome_from_store, write_trajectory_graph_artifacts
from .types import EvalRunSummary, EvalSuite, EvalTask, EvalTrial, GraderResult


class EvaluationHarness:
    """Runs eval tasks end-to-end, records transcripts, grades outcomes, and aggregates results."""

    def __init__(
        self,
        *,
        corpus_path: Path = Path("examples/corpus/research_corpus.json"),
        output_root: Path = Path("eval_outputs"),
    ) -> None:
        self.corpus_path = corpus_path
        self.output_root = output_root
        self.grader_registry = default_graders()

    async def run_suite(self, suite: EvalSuite) -> EvalRunSummary:
        started_at = now_iso()
        self._prepare_eval_root()
        trials: list[EvalTrial] = []
        for task in suite.tasks:
            task_trials = task.trials or suite.trials_per_task
            seeds = task.seeds or suite.seeds
            models = task.models or suite.models
            for model in models:
                for seed in seeds:
                    for trial_index in range(1, task_trials + 1):
                        trials.append(await self._run_trial(task, trial_index, model, seed))
        self._apply_cross_trial_graders(suite, trials)
        passed_trials = sum(1 for trial in trials if trial.passed)
        aggregate_score = sum(trial.aggregate_score for trial in trials) / max(len(trials), 1)
        summary = EvalRunSummary(
            suite_id=suite.id,
            suite_name=suite.name,
            trials_per_task=suite.trials_per_task,
            started_at=started_at,
            completed_at=now_iso(),
            task_count=len(suite.tasks),
            trial_count=len(trials),
            passed_trials=passed_trials,
            aggregate_score=round(aggregate_score, 3),
            trials=[asdict(trial) for trial in trials],
            comparisons=self._comparison_summary(trials),
        )
        (self.output_root / f"{suite.id}_summary.json").write_text(
            json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary

    def _apply_cross_trial_graders(self, suite: EvalSuite, trials: list[EvalTrial]) -> None:
        tasks_by_id = {task.id: task for task in suite.tasks}
        for task in suite.tasks:
            if "parallel_trial_isolation" not in task.grader_ids:
                continue
            matching = [trial for trial in trials if trial.task_id == task.id]
            result = _grade_parallel_trial_isolation_from_trials(task, matching)
            for trial in matching:
                trial.grader_results.append(asdict(result))
                aggregate_score, passed = aggregate_results(
                    tasks_by_id[trial.task_id],
                    [GraderResult(**grader) for grader in trial.grader_results],
                )
                trial.aggregate_score = aggregate_score
                trial.passed = passed

    def _comparison_summary(self, trials: list[EvalTrial]) -> dict[str, object]:
        grouped: dict[str, list[EvalTrial]] = {}
        for trial in trials:
            grouped.setdefault(f"{trial.task_id}|{trial.model}", []).append(trial)
        return {
            key: {
                "attempts": len(group), "pass_rate": round(sum(item.passed for item in group) / len(group), 3),
                "mean_score": round(statistics.mean(item.aggregate_score for item in group), 3),
                "score_stdev": round(statistics.pstdev(item.aggregate_score for item in group), 3) if len(group) > 1 else 0.0,
                "seeds": sorted({item.seed for item in group}),
            }
            for key, group in grouped.items()
        }

    async def _run_trial(self, task: EvalTask, trial_index: int, model: str, seed: int) -> EvalTrial:
        trial_root = self._prepare_trial_root(task, trial_index, model, seed)
        # Run artifacts land directly in trial_root (mirroring the outputs/ folder layout).
        trial_output_root = trial_root
        trial_tmp = trial_root / "tmp"
        trial_tmp.mkdir(parents=True, exist_ok=True)
        config = HarnessConfig(
            retriever=task.retriever,
            max_iterations=task.max_iterations,
            evaluator_name=task.evaluator_name,
            echo_progress=False,
            llm_provider=_provider_for_model(model),
            llm_model=_model_name(model),
            llm_seed=seed,
        )
        orchestrator = Orchestrator(self.corpus_path, trial_output_root, config)
        previous_tmpdir = os.environ.get("TMPDIR")
        os.environ["TMPDIR"] = str(trial_tmp)
        run, store = await orchestrator.run(task.prompt)
        if previous_tmpdir is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = previous_tmpdir
        graph_paths = write_trajectory_graph_artifacts(store, trial_root)
        grader_results = [
            self.grader_registry[grader_id].grade(task, store)
            for grader_id in task.grader_ids
            if grader_id != "parallel_trial_isolation"
        ]
        aggregate_score, passed = aggregate_results(task, grader_results)
        return EvalTrial(
            task_id=task.id,
            trial_index=trial_index,
            run_id=run.id,
            transcript_path=str(store.trace_log_path),
            trajectory_graph_path=str(graph_paths["svg"]),
            isolation={
                "trial_root": str(trial_root),
                "output_root": str(trial_output_root),
                "tmpdir": str(trial_tmp),
                "clean_start": True,
                "production_agent_path": "research_harness.orchestrator.Orchestrator",
                "shared_state_policy": "No shared output directories between trials; local corpus is read-only; TMPDIR is per-trial.",
            },
            outcome=outcome_from_store(store),
            grader_results=[asdict(result) for result in grader_results],
            aggregate_score=aggregate_score,
            passed=passed,
            seed=seed,
            model=model,
        )

    def _prepare_eval_root(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _prepare_trial_root(self, task: EvalTask, trial_index: int, model: str, seed: int) -> Path:
        # eval_outputs/<task_id>/trial_001/ — mirrors outputs/ structure per task.
        safe_model = model.replace("/", "_").replace(":", "_")
        trial_root = self.output_root / task.id / f"{safe_model}_seed_{seed:04d}" / f"trial_{trial_index:03d}"
        if trial_root.exists():
            shutil.rmtree(trial_root)
        trial_root.mkdir(parents=True, exist_ok=True)
        return trial_root


def _provider_for_model(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else model


def _model_name(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else "gpt-5.2"
