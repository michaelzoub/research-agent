"""Public facade for the model-directed research agent."""
from __future__ import annotations

import asyncio
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
    DocumentFigureTool,
    FileReadTool,
    OptimizationSwarmTool,
    ParameterSweepTool,
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
        evaluator_tools = [
            PredictionMarketEvaluationTool(), OptimizationSwarmTool(llm),
            ParameterSweepTool(), SaveLearningTool(),
        ] if evaluator_name == "prediction_market" else []
        external_service_tools = default_external_service_registry().tools()
        tools = [
            *(SearchTool(backend) for backend in backends), WebFetchTool(),
            DocumentFigureTool(), StructuredDataExtractionTool(), DocumentAnalysisTool(llm), SVGChartTool(),
            FileReadTool(), CodeExecutionTool(),
            TerminalExecutionTool(), *external_service_tools, *evaluator_tools,
            SpecialistConsultationTool(llm),
        ]
        base_registry = ToolRegistry(tools)
        safe_worker_tools = tuple(tool.name for tool in tools if tool.name not in {
            "spawn_optimization_agents", "run_parameter_sweep", "save_learning"
        })
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


__all__ = [
    "AgentEvent", "AgentLoop", "AgentRunConfig", "AgentRunResult",
    "FinalAnswerValidator", "LLMToolDecider", "ResearchAgent", "ValidationResult",
]
