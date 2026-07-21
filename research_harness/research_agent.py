"""Public facade for the model-directed research agent."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

from .agent_loop import (
    AgentDecider,
    AgentLoop,
    AgentRunConfig,
    _OPTIMIZATION_EXPLORATION_GUIDANCE,
    _SYSTEM_INSTRUCTIONS,
    _join_answer_chunks,
    _partial_synthesis,
)
from .agent_state import AgentEvent, AgentRunResult
from .agents import SpecialistConsultationTool
from .citation_validation import coverage as citation_coverage, validate_claim_citations
from .external_services import default_external_service_registry
from .llm import LLMClient, ModelTurn
from .schemas import AgentTrace, Claim, now_iso
from .store import ArtifactStore
from .tools import (
    CodeExecutionTool,
    DocumentAnalysisTool,
    CompareCandidateToChampionTool,
    DocumentFigureTool,
    FileReadTool,
    PredictionMarketEvaluationTool,
    SaveLearningTool,
    SearchTool,
    SVGChartTool,
    StructuredDataExtractionTool,
    TerminalExecutionTool,
    ToolContext,
    ToolRegistry,
    WebFetchTool,
)
from .validation import FinalAnswerValidator, ValidationResult
from .worker_registry import DelegateTaskTool, WorkerBudget, WorkerProfile, WorkerRegistry


class LLMToolDecider:
    """Native provider tool calling. JSON decisions are not the default protocol."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def decide(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> ModelTurn:
        if not self.llm.is_live:
            raise RuntimeError("A live model provider with native tool calling is required. Configure OpenAI, Anthropic, or a compatible local model.")
        return await asyncio.to_thread(self.llm.complete_turn, messages, tools)


class ResearchAgent:
    """The only general-purpose research controller in production execution."""

    def __init__(self, decider: AgentDecider, tools: ToolRegistry, config: Optional[AgentRunConfig] = None):
        self.decider, self.tools, self.config = decider, tools, config or AgentRunConfig()

    @classmethod
    def with_research_tools(
        cls,
        llm: LLMClient,
        backends: Sequence[Any],
        config: Optional[AgentRunConfig] = None,
        evaluator_name: Optional[str] = None,
    ) -> "ResearchAgent":
        evaluator_tools = [PredictionMarketEvaluationTool(), CompareCandidateToChampionTool(), SaveLearningTool()] if evaluator_name == "prediction_market" else []
        external_service_tools = default_external_service_registry().tools()
        tools = [
            *(SearchTool(backend) for backend in backends), WebFetchTool(),
            DocumentFigureTool(), StructuredDataExtractionTool(), DocumentAnalysisTool(llm), SVGChartTool(),
            FileReadTool(), CodeExecutionTool(),
            TerminalExecutionTool(), *external_service_tools, *evaluator_tools,
            SpecialistConsultationTool(llm),
        ]
        base_registry = ToolRegistry(tools)
        safe_worker_tools = tuple(tool.name for tool in tools if tool.name != "save_learning")
        workers = WorkerRegistry([
            WorkerProfile("researcher", "You are a focused research worker. Investigate the assignment and return concise evidence-backed findings.", safe_worker_tools),
            WorkerProfile("critic", "You are a critical review worker. Stress-test the assignment, identify gaps, and return actionable findings.", safe_worker_tools, budget=WorkerBudget(max_iterations=3, max_tokens=3000, max_tool_calls=6, max_runtime_seconds=90.0)),
        ], lambda profile: LLMToolDecider(LLMClient(
            provider=profile.model_provider or llm.provider, model=profile.model_name or llm.model,
            timeout_seconds=min(llm.timeout_seconds, profile.budget.max_runtime_seconds), seed=llm.seed,
        )))
        return cls(LLMToolDecider(llm), ToolRegistry([*tools, DelegateTaskTool(workers, base_registry)]), config)

    async def arun(
        self,
        objective: str,
        *,
        workspace: Path,
        store: Optional[ArtifactStore] = None,
        run_id: str = "",
        readable_roots: Optional[Sequence[Path]] = None,
    ) -> AgentRunResult:
        started, started_at = time.perf_counter(), now_iso()
        result = await AgentLoop(self.decider, self.tools, self.config).run(
            objective,
            ToolContext(
                workspace=workspace,
                readable_roots=readable_roots or [workspace],
                store=store,
                run_id=run_id,
            ),
        )
        if store is not None:
            result = _finalize_measured_optimization(store, result, run_id=run_id)
            store.write_agent_transcript({
                "objective": objective,
                "termination_reason": result.termination_reason,
                "status": result.status,
                "messages": result.messages,
                "tool_calls": result.tool_calls,
                "events": [asdict(event) for event in result.events],
                "sources": result.sources,
            })
            if result.final_answer:
                store.write_report(result.final_answer)
            if result.status == "completed":
                for check in validate_claim_citations(result.final_answer, result.sources):
                    if check.passed:
                        store.add_claim(Claim(
                            text=check.claim,
                            source_ids=check.source_ids,
                            confidence=check.support,
                            support_level="verified_document",
                            created_by_agent="citation_validator",
                            run_id=run_id,
                            citation_support=check.support,
                            citation_coverage=citation_coverage([check]),
                            citation_locators=check.locators,
                        ))
            llm = getattr(self.decider, "llm", None)
            store.add_trace(AgentTrace(
                run_id=run_id,
                agent_name="research_agent",
                role="model_directed_agent",
                prompt=objective,
                model=getattr(llm, "model_label", "agent-decider"),
                tools_used=[call["tool"] for call in result.tool_calls],
                tool_calls=result.tool_calls,
                token_usage=0,
                runtime_ms=int((time.perf_counter() - started) * 1000),
                status="completed" if result.status == "completed" else result.status,
                errors=[] if result.status == "completed" else [result.termination_reason],
                output_summary=result.final_answer[:500],
                started_at=started_at,
                prompt_tokens=getattr(llm, "total_prompt_tokens", 0),
                completion_tokens=getattr(llm, "total_completion_tokens", 0),
                cost_usd=round(getattr(llm, "total_cost", lambda: 0.0)(), 6),
                failure_component="agent_loop",
            ))
        return result

    def run(
        self,
        objective: str,
        *,
        workspace: Path,
        store: Optional[ArtifactStore] = None,
        run_id: str = "",
        readable_roots: Optional[Sequence[Path]] = None,
    ) -> AgentRunResult:
        """Synchronous convenience wrapper for embedding outside an event loop."""
        return asyncio.run(self.arun(
            objective,
            workspace=workspace,
            store=store,
            run_id=run_id,
            readable_roots=readable_roots,
        ))


def _finalize_measured_optimization(store: ArtifactStore, result: AgentRunResult, *, run_id: str) -> AgentRunResult:
    """Materialize the best eligible trial and a complete best-so-far report.

    This is deterministic run finalization, not an additional model turn. It
    therefore still runs when the configured model-iteration budget is spent.
    """
    trials: list[dict[str, Any]] = []
    for path in sorted(store.optimization_trials_dir.glob("*.json")):
        try:
            trial = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if trial.get("official_measured") and trial.get("score_eligible") and str(trial.get("rendered_code") or "").strip():
            trial["_trial_path"] = str(path)
            trials.append(trial)
    if not trials:
        return result

    best = max(trials, key=lambda trial: float(trial.get("score") or 0.0))
    code = str(best["rendered_code"]).rstrip() + "\n"
    candidate_path = store.write_optimized_candidate(code)
    optimal_code_path = store.write_optimal_code(code)
    solution_path = store.write_solution(code)
    metrics = dict(best.get("metrics") or {})
    payload = {
        "run_id": run_id,
        "status": "best_at_iteration_limit" if result.termination_reason == "budget_exhausted" else "completed",
        "termination_reason": result.termination_reason,
        "best_candidate_id": best.get("trial_id"),
        "best_candidate_path": str(candidate_path),
        "optimal_code_path": str(optimal_code_path),
        "solution_path": str(solution_path),
        "trial_path": str(best["_trial_path"]),
        "score": float(best.get("score") or 0.0),
        "metrics": metrics,
        "official_measured": True,
        "score_eligible": True,
        "evaluated_candidates": len(trials),
        "official_result": {
            "measured": True,
            "score_eligible": True,
            "score": float(best.get("score") or 0.0),
            "score_source": metrics.get("score_source"),
            "candidate_path": str(best.get("candidate_path") or ""),
            "trial_path": str(best["_trial_path"]),
            "metrics": metrics,
        },
    }
    store.write_optimization_result(payload)

    prior_learning_titles = {
        str(item.get("title") or "")
        for line in store.learning_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for item in [json.loads(line)]
    }
    if "Best officially measured candidate" not in prior_learning_titles:
        score_rows = sorted(
            ((str(trial.get("trial_id") or "candidate"), float(trial.get("score") or 0.0)) for trial in trials),
            key=lambda row: row[1], reverse=True,
        )
        evidence = ", ".join(f"{candidate_id}={score:g}" for candidate_id, score in score_rows[:8])
        store.append_learning(
            title="Best officially measured candidate",
            finding=f"{best.get('trial_id')} ranked first among {len(trials)} eligible candidate(s) with mean edge {float(best.get('score') or 0.0):g}.",
            evidence=f"Official eligible measurements: {evidence}.",
            status="confirmed",
            run_id=run_id,
        )

    learnings = store.learnings_path.read_text(encoding="utf-8").strip() if store.learnings_path.exists() else "No additional learnings were recorded."
    limit_note = (
        "The configured model-iteration limit was reached. The artifacts below are the complete, officially measured best-so-far result."
        if result.termination_reason == "budget_exhausted"
        else "The run completed with the officially measured best candidate below."
    )
    report = "\n".join([
        "# Optimization result",
        "",
        limit_note,
        "",
        "## Best candidate",
        "",
        f"- Candidate: `{best.get('trial_id')}`",
        f"- Official mean edge: `{float(best.get('score') or 0.0):g}`",
        f"- Eligible candidates evaluated: `{len(trials)}`",
        f"- Optimal code artifact: `{optimal_code_path}`",
        "",
        "```python",
        code.rstrip(),
        "```",
        "",
        "## Learnings",
        "",
        learnings,
    ])
    result.final_answer = report
    return result


__all__ = [
    "AgentEvent", "AgentLoop", "AgentRunConfig", "AgentRunResult",
    "FinalAnswerValidator", "LLMToolDecider", "ResearchAgent", "ValidationResult",
]
