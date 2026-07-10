"""Model-directed research agent and bounded tool-use loop."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

from .llm import LLMClient
from .schemas import AgentTrace, now_iso
from .store import ArtifactStore
from .tools import CodeExecutionTool, FileReadTool, SearchTool, ToolContext, ToolRegistry, WebFetchTool


class AgentDecider(Protocol):
    def decide(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> dict[str, Any]: ...


class LLMToolDecider:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def decide(self, messages: Sequence[dict[str, Any]], tools: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not self.llm.is_live:
            return {
                "type": "final",
                "answer": (
                    "A live language-model provider is required for model-directed tool use. "
                    "Configure OPENAI_API_KEY, ANTHROPIC_API_KEY, MOONSHOT_API_KEY, or a reachable Ollama model."
                ),
            }
        system = (
            "You are a research agent in a probabilistic, evidence-driven loop. Decide each next action yourself. "
            "The human defines tools, environment, state, evaluator, budgets, and safety boundaries; you control which tools to use, what to investigate, what to do next, when evidence is insufficient, and when to stop. "
            "Use a registered tool only when it materially helps; never follow a fixed tool sequence. Learn which search strategies improve results across repeated evaluations by using the evidence and evaluator feedback in the conversation. "
            "When tool results provide sources and they inform the answer, cite their URLs in the final answer. "
            "Return JSON only: either {\"type\":\"tool_call\",\"tool_name\":string,\"arguments\":object}, "
            "{\"type\":\"final\",\"answer\":string}, or {\"type\":\"needs_input\",\"question\":string}."
        )
        user = json.dumps({"messages": list(messages), "available_tools": list(tools)}, sort_keys=True, default=str)
        return self.llm.complete_json(system, user, max_output_tokens=1200, temperature=0.35)


@dataclass(frozen=True)
class AgentRunConfig:
    max_iterations: int = 8
    max_cost_usd: Optional[float] = None
    max_runtime_seconds: float = 120.0


@dataclass
class AgentRunResult:
    final_answer: str
    termination_reason: str
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    sources: list[dict[str, Any]] = field(default_factory=list)


class AgentLoop:
    def __init__(self, decider: AgentDecider, registry: ToolRegistry, config: AgentRunConfig):
        self.decider, self.registry, self.config = decider, registry, config

    def run(self, objective: str, context: ToolContext) -> AgentRunResult:
        messages: list[dict[str, Any]] = [{"role": "user", "content": objective}]
        calls: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        started = time.monotonic()
        cost_before = _decider_cost(self.decider)
        for iteration in range(1, self.config.max_iterations + 1):
            if context.cancelled:
                return AgentRunResult("", "cancelled", messages, calls, sources)
            if time.monotonic() - started > self.config.max_runtime_seconds:
                return AgentRunResult("", "runtime_limit", messages, calls, sources)
            if self.config.max_cost_usd is not None and _decider_cost(self.decider) - cost_before >= self.config.max_cost_usd:
                return AgentRunResult("", "cost_limit", messages, calls, sources)
            try:
                response = self.decider.decide(messages, self.registry.schemas())
            except Exception as exc:
                return AgentRunResult("", "model_error: %s" % exc, messages, calls, sources)
            kind = response.get("type") if isinstance(response, dict) else None
            if kind == "final" and isinstance(response.get("answer"), str):
                return AgentRunResult(response["answer"], "final", messages + [{"role": "assistant", "content": response}], calls, sources)
            if kind == "needs_input" and isinstance(response.get("question"), str):
                return AgentRunResult(response["question"], "needs_user_input", messages + [{"role": "assistant", "content": response}], calls, sources)
            if kind != "tool_call" or not isinstance(response.get("tool_name"), str):
                return AgentRunResult("", "invalid_model_response", messages, calls, sources)
            name = response["tool_name"]
            arguments = response.get("arguments", {})
            result = self.registry.execute(name, arguments, context)
            call = {"iteration": iteration, "tool": name, "arguments": arguments, "status": result.status, "error": result.error, "retryable": result.retryable, "results": len(result.source_metadata)}
            calls.append(call)
            sources.extend(result.source_metadata)
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "tool", "name": name, "content": result.as_message()})
        return AgentRunResult("", "iteration_limit", messages, calls, sources)


class ResearchAgent:
    """High-level agent configuration; integrations live exclusively in tools."""
    def __init__(self, decider: AgentDecider, tools: ToolRegistry, config: Optional[AgentRunConfig] = None):
        self.decider, self.tools, self.config = decider, tools, config or AgentRunConfig()

    @classmethod
    def with_research_tools(cls, llm: LLMClient, backends: Sequence[Any], config: Optional[AgentRunConfig] = None) -> "ResearchAgent":
        tools = [SearchTool(backend) for backend in backends]
        tools.extend([WebFetchTool(), FileReadTool(), CodeExecutionTool()])
        return cls(LLMToolDecider(llm), ToolRegistry(tools), config)

    def run(self, objective: str, *, workspace: Path, store: Optional[ArtifactStore] = None, run_id: str = "") -> AgentRunResult:
        started, started_at = time.perf_counter(), now_iso()
        result = AgentLoop(self.decider, self.tools, self.config).run(objective, ToolContext(workspace=workspace, store=store, run_id=run_id))
        if store is not None:
            store.write_agent_transcript({
                "objective": objective,
                "termination_reason": result.termination_reason,
                "messages": result.messages,
                "tool_calls": result.tool_calls,
                "sources": result.sources,
            })
            if result.final_answer:
                store.write_report(result.final_answer)
            llm = getattr(self.decider, "llm", None)
            store.add_trace(AgentTrace(
                run_id=run_id, agent_name="research_agent", role="tool_using_research_agent", prompt=objective,
                model=getattr(llm, "model_label", "agent-decider"), tools_used=[call["tool"] for call in result.tool_calls],
                tool_calls=result.tool_calls, token_usage=0, runtime_ms=int((time.perf_counter() - started) * 1000),
                status="completed" if result.termination_reason == "final" else "failed", errors=[] if result.termination_reason == "final" else [result.termination_reason],
                output_summary=result.final_answer[:500] or result.termination_reason, started_at=started_at,
                prompt_tokens=getattr(llm, "total_prompt_tokens", 0), completion_tokens=getattr(llm, "total_completion_tokens", 0),
                cost_usd=round(getattr(llm, "total_cost", lambda: 0.0)(), 6), failure_component="agent_loop",
            ))
        return result


def _decider_cost(decider: AgentDecider) -> float:
    llm = getattr(decider, "llm", None)
    return float(llm.total_cost()) if llm is not None and hasattr(llm, "total_cost") else 0.0
