from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional
from uuid import uuid4


TaskType = Literal["bounded", "open_ended"]
TaskMode = Literal["optimize", "research", "optimize_query"]
ProductAgent = Literal["research", "optimize", "challenge"]
RunStatus = Literal["running", "completed", "needs_input", "partial", "budget_exhausted", "cancelled", "safety_stopped", "failed"]
WritePolicy = Literal["append_only", "upsert_by_url", "upsert_by_text"]
LoopTaskStatus = Literal["pending", "running", "passed", "failed", "skipped"]
LoopTaskAction = Literal["search", "hypothesize", "critique", "synthesize", "debug_harness"]
VariantKind = Literal["code", "query"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass
class AgentBudget:
    max_steps: int = 4
    max_tokens: int = 4000
    max_tool_calls: int = 8
    max_runtime_seconds: float = 30.0
    write_policy: WritePolicy = "append_only"
    reporting_schema: str = "structured_artifact_v1"
    trace_id: str = field(default_factory=lambda: new_id("trace"))
    cancelled: bool = False


@dataclass
class LoopTask:
    title: str
    action: LoopTaskAction
    priority: int
    params: dict[str, Any]
    acceptance_criteria: list[str]
    status: LoopTaskStatus = "pending"
    passes: bool = False
    attempts: int = 0
    last_error: Optional[str] = None
    result_summary: Optional[str] = None
    created_at: str = field(default_factory=now_iso)
    completed_at: Optional[str] = None
    id: str = field(default_factory=lambda: new_id("task"))


@dataclass
class LoopIteration:
    run_id: str
    iteration: int
    task_id: str
    task_title: str
    agent_name: str
    status: str
    summary: str
    errors: list[str]
    started_at: str = field(default_factory=now_iso)
    completed_at: Optional[str] = None
    id: str = field(default_factory=lambda: new_id("iter"))


@dataclass
class TaskIngestionDecision:
    requested_mode: str
    selected_mode: TaskMode
    reason: str
    evaluator_name: Optional[str] = None
    product_agent: ProductAgent = "research"
    id: str = field(default_factory=lambda: new_id("decision"))


@dataclass
class Variant:
    run_id: str
    outer_iteration: int
    kind: VariantKind
    payload: str
    parent_ids: list[str]
    metadata: dict[str, Any]
    id: str = field(default_factory=lambda: new_id("variant"))


@dataclass
class VariantEvaluation:
    run_id: str
    variant_id: str
    inner_loop: TaskMode
    score: float
    metrics: dict[str, Any]
    judge_scores: list[float]
    summary: str
    passed: bool
    id: str = field(default_factory=lambda: new_id("eval"))


@dataclass
class EvolutionRound:
    run_id: str
    outer_iteration: int
    mode: TaskMode
    variant_ids: list[str]
    best_variant_id: Optional[str]
    best_score: float
    termination_signal: str
    plateau_count: int
    completed_at: str = field(default_factory=now_iso)
    id: str = field(default_factory=lambda: new_id("round"))


@dataclass
class LoopContinuationDecision:
    run_id: str
    loop_name: str
    iteration: int
    mode: TaskMode
    decision: Literal["continue", "exit"]
    reason: str
    termination_signal: str
    best_score: float
    plateau_count: int
    next_action: str
    decided_at: str = field(default_factory=now_iso)
    id: str = field(default_factory=lambda: new_id("continue"))


@dataclass
class Source:
    url: str
    title: str
    author: str
    date: str
    source_type: str
    summary: str
    relevance_score: float
    credibility_score: float
    evidence_sections: dict[str, str] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("src"))
    retrieved_at: str = field(default_factory=now_iso)
    canonical_id: Optional[str] = None
    duplicate_of: Optional[str] = None


@dataclass
class Claim:
    text: str
    source_ids: list[str]
    confidence: float
    support_level: str
    created_by_agent: str
    run_id: str
    contradicted_by: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: new_id("claim"))
    canonical_id: Optional[str] = None
    duplicate_of: Optional[str] = None


@dataclass
class Hypothesis:
    text: str
    supporting_claim_ids: list[str]
    contradicting_claim_ids: list[str]
    confidence: float
    novelty_score: float
    testability_score: float
    next_experiment: str
    id: str = field(default_factory=lambda: new_id("hyp"))
    run_id: Optional[str] = None
    canonical_id: Optional[str] = None
    duplicate_of: Optional[str] = None


@dataclass
class Experiment:
    description: str
    hypothesis_id: Optional[str]
    expected_signal: str
    priority: int
    id: str = field(default_factory=lambda: new_id("exp"))


@dataclass
class OpenQuestion:
    question: str
    priority: int
    reason: str
    created_by_agent: str
    status: str = "open"
    id: str = field(default_factory=lambda: new_id("q"))


@dataclass
class Contradiction:
    claim_a: str
    claim_b: str
    explanation: str
    severity: str
    resolution_status: str = "unresolved"
    id: str = field(default_factory=lambda: new_id("contra"))


@dataclass
class FailedPath:
    description: str
    reason: str
    created_by_agent: str
    run_id: str
    id: str = field(default_factory=lambda: new_id("fail"))
    failure_category: str = "unknown"
    failure_component: str = "unknown"
    retryable: bool = False
    severity: str = "medium"


@dataclass
class HarnessChange:
    change: str
    reason: str
    expected_effect: str
    risk: str
    evaluation: str
    run_id: str
    status: str = "pending"
    created_at: str = field(default_factory=now_iso)
    id: str = field(default_factory=lambda: new_id("change"))
    component: str = "harness"
    diagnosis: str = ""
    risk_score: float = 0.5
    expected_value_score: float = 0.5
    priority_score: float = 0.5
    trace_pattern_delta: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRecord:
    user_goal: str
    task_type: TaskType
    harness_config_id: str
    task_mode: Optional[TaskMode] = None
    product_agent: Optional[ProductAgent] = None
    status: RunStatus = "running"
    total_cost: float = 0.0
    total_tokens: int = 0
    started_at: str = field(default_factory=now_iso)
    completed_at: Optional[str] = None
    id: str = field(default_factory=lambda: new_id("run"))
    session_id: Optional[str] = None
    session_jsonl_path: Optional[str] = None
    session_metadata_path: Optional[str] = None
    prompt_versions: dict[str, str] = field(default_factory=dict)
    harness_config_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTrace:
    run_id: str
    agent_name: str
    role: str
    prompt: str
    model: str
    tools_used: list[str]
    tool_calls: list[dict[str, Any]]
    token_usage: int
    runtime_ms: int
    status: str
    errors: list[str]
    output_summary: str
    id: str = field(default_factory=lambda: new_id("trace"))
    started_at: str = ""  # ISO timestamp when agent execution began (wall clock)
    prompt_version: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    failure_category: str = "none"
    failure_component: str = "unknown"
    retryable: bool = False


@dataclass
class ProvenanceEdge:
    run_id: str
    from_type: str
    from_id: str
    to_type: str
    to_id: str
    relationship: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)
    id: str = field(default_factory=lambda: new_id("edge"))


@dataclass
class CostEvent:
    run_id: str
    component: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    call_type: str = "model"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso)
    id: str = field(default_factory=lambda: new_id("cost"))


@dataclass
class ResearchPlan:
    task_type: TaskType
    goal: str
    strategy: str
    search_angles: list[str]
    hypothesis_angles: list[str]
    stopping_signals: list[str]
    topics: list[str] = field(default_factory=list)
    topic_queries: list[str] = field(default_factory=list)
    planner: str = "deterministic-fallback"


@dataclass
class SourceStrategyItem:
    name: str
    retriever: str
    purpose: str
    queries: list[str]
    limit: int = 6


def to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
