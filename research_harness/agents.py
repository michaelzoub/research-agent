"""Generic agent tracing primitives.

Production research is controlled by :class:`research_harness.research_agent.ResearchAgent`.
This module deliberately contains no predefined roles, search directions, stop
conditions, domain vocabularies, or report-selection heuristics.  It remains as
a small compatibility layer for an embedding application that supplies its own
agent implementation and needs the harness to record runtime, budget, and cost.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Protocol

from .llm import LLMClient
from .schemas import AgentBudget, AgentTrace, RunRecord, now_iso
from .store import ArtifactStore
from .tools.base import ToolContext, ToolResult


class Agent(Protocol):
    """An externally defined agent whose behavior is not prescribed here."""

    name: str
    role: str
    budget: AgentBudget

    async def run(self, run: RunRecord, store: ArtifactStore) -> str: ...


@dataclass
class AgentResult:
    agent_name: str
    summary: str
    errors: list[str]


class BaseAgent:
    """Apply the harness budget and persist the observed result of a custom agent."""

    def __init__(
        self,
        name: str,
        role: str,
        budget: Optional[AgentBudget] = None,
        llm: Optional[LLMClient] = None,
        model: Optional[str] = None,
    ):
        self.name = name
        self.role = role
        self.budget = budget or AgentBudget()
        self.llm = llm or LLMClient()
        self.model = model or self.llm.model_label
        self.tool_calls: list[dict[str, object]] = []
        self.tools_used: list[str] = []

    async def execute(self, run: RunRecord, store: ArtifactStore) -> AgentResult:
        started = time.perf_counter()
        started_at = now_iso()
        errors: list[str] = []
        status = "completed"
        summary = ""
        tokens_before = self.llm.total_prompt_tokens + self.llm.total_completion_tokens
        prompt_tokens_before = self.llm.total_prompt_tokens
        completion_tokens_before = self.llm.total_completion_tokens
        cost_before = self.llm.total_cost()
        try:
            if self.budget.cancelled:
                status = "cancelled"
                summary = "Agent cancelled before execution."
            else:
                summary = await asyncio.wait_for(self.run(run, store), timeout=self.budget.max_runtime_seconds)
        except Exception as exc:  # pragma: no cover - defensive trace path
            status = "failed"
            errors.append(f"{type(exc).__name__}: {exc}")
            summary = "Agent failed; see errors."

        runtime_ms = int((time.perf_counter() - started) * 1000)
        store.add_trace(
            AgentTrace(
                id=self.budget.trace_id,
                run_id=run.id,
                agent_name=self.name,
                role=self.role,
                prompt=run.user_goal,
                model=self.model,
                tools_used=self.tools_used,
                tool_calls=self.tool_calls,
                token_usage=(self.llm.total_prompt_tokens + self.llm.total_completion_tokens) - tokens_before,
                runtime_ms=runtime_ms,
                status=status,
                errors=errors,
                output_summary=summary,
                started_at=started_at,
                prompt_tokens=self.llm.total_prompt_tokens - prompt_tokens_before,
                completion_tokens=self.llm.total_completion_tokens - completion_tokens_before,
                cost_usd=round(max(0.0, self.llm.total_cost() - cost_before), 6),
            )
        )
        return AgentResult(self.name, summary, errors)

    async def run(self, run: RunRecord, store: ArtifactStore) -> str:
        raise NotImplementedError("Provide agent behavior in the embedding application.")


class SpecialistConsultationTool:
    """Model-invoked specialist agent with a role chosen by the controller.

    The harness provides the execution, trace, timeout, and accounting boundary.
    The controller decides whether consultation is useful and supplies the role,
    question, and any evidence to review.  No specialty, query, or stopping
    direction is selected here.
    """

    name = "consult_specialist"
    description = "Consult a separate specialist agent. Choose the specialty and question yourself; include the evidence or constraints it must review. The specialist cannot use tools or decide whether the overall run should stop."
    # LLMClient accounting mutates per-call counters, so consultations are
    # deliberately serialized by ToolRegistry alongside other stateful tools.
    is_read_only = False
    input_schema = {
        "type": "object",
        "required": ["specialty", "question"],
        "properties": {
            "specialty": {"type": "string", "minLength": 2, "maxLength": 120},
            "question": {"type": "string", "minLength": 8, "maxLength": 12000},
            "evidence": {"type": "array", "items": {"type": "string", "maxLength": 4000}, "maxItems": 12},
        },
        "additionalProperties": False,
    }

    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def execute(self, arguments: dict[str, object], context: ToolContext) -> ToolResult:
        specialty = str(arguments["specialty"]).strip()
        question = str(arguments["question"]).strip()
        evidence = [str(item) for item in arguments.get("evidence", []) if str(item).strip()]
        system = (
            f"You are a specialist consultant acting as: {specialty}. "
            "Answer the controller's concrete question using only the evidence and constraints supplied. "
            "Do not invent citations, make new tool calls, choose the overall search strategy, or decide when the run ends."
        )
        prompt = question + ("\n\nEvidence to review:\n- " + "\n- ".join(evidence) if evidence else "")
        started, started_at = time.perf_counter(), now_iso()
        try:
            response = await asyncio.to_thread(self.llm.complete, system, prompt, max_output_tokens=800, temperature=0.2)
        except Exception as exc:
            return ToolResult("error", error=f"Specialist {specialty} failed: {type(exc).__name__}: {exc}", retryable=True)
        runtime_ms = int((time.perf_counter() - started) * 1000)
        if context.store is not None:
            context.store.add_trace(
                AgentTrace(
                    run_id=context.run_id or context.store.root.name,
                    agent_name=f"specialist:{specialty}",
                    role="specialist_consultation",
                    prompt=prompt,
                    model=response.model,
                    tools_used=[],
                    tool_calls=[],
                    token_usage=response.prompt_tokens + response.completion_tokens,
                    runtime_ms=runtime_ms,
                    status="completed",
                    errors=[],
                    output_summary=response.text[:500],
                    started_at=started_at,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    cost_usd=round(response.cost, 6),
                )
            )
        return ToolResult("ok", {"specialty": specialty, "response": response.text, "model": response.model})
