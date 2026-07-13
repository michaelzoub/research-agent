from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from ..schemas import now_iso
from ..store import ArtifactStore


GraderType = Literal["code", "model", "human"]
AggregationMode = Literal["weighted", "binary", "hybrid"]
TrajectoryMatchMode = Literal["strict", "unordered", "subset", "superset"]


@dataclass
class EvalTask:
    """A single problem/test case with fixed inputs and success criteria."""

    id: str
    name: str
    prompt: str
    task_mode: str
    success_criteria: list[str]
    evaluator_name: Optional[str] = None
    retriever: str = "local"
    max_iterations: int = 1
    grader_ids: list[str] = field(default_factory=list)
    aggregation: AggregationMode = "hybrid"
    threshold: float = 0.8
    metadata: dict[str, Any] = field(default_factory=dict)
    trials: Optional[int] = None
    seeds: Optional[list[int]] = None
    models: Optional[list[str]] = None


@dataclass
class EvalTrial:
    """One attempt at one eval task."""

    task_id: str
    trial_index: int
    run_id: str
    transcript_path: str
    trajectory_graph_path: str
    isolation: dict[str, Any]
    outcome: dict[str, Any]
    grader_results: list[dict[str, Any]]
    aggregate_score: float
    passed: bool
    seed: int = 1
    model: str = "local"
    completed_at: str = field(default_factory=now_iso)


@dataclass
class EvalSuite:
    """A collection of tasks that measure related harness behavior."""

    id: str
    name: str
    description: str
    tasks: list[EvalTask]
    trials_per_task: int = 1
    seeds: list[int] = field(default_factory=lambda: [1])
    models: list[str] = field(default_factory=lambda: ["local"])


@dataclass
class EvalRunSummary:
    suite_id: str
    suite_name: str
    trials_per_task: int
    started_at: str
    completed_at: str
    task_count: int
    trial_count: int
    passed_trials: int
    aggregate_score: float
    trials: list[dict[str, Any]]
    comparisons: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraderResult:
    grader_id: str
    grader_type: GraderType
    method: str
    score: float
    passed: bool
    weight: float
    summary: str
    assertions: list[dict[str, Any]]


@dataclass
class Grader:
    id: str
    grader_type: GraderType
    method: str
    weight: float
    threshold: float
    fn: Callable[[EvalTask, ArtifactStore], GraderResult]

    def grade(self, task: EvalTask, store: ArtifactStore) -> GraderResult:
        return self.fn(task, store)
