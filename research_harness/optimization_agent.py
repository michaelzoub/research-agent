from __future__ import annotations

import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .llm import LLMClient
from .schemas import AgentTrace, now_iso
from .store import ArtifactStore


@dataclass
class OptimizerToolbox:
    read_champion: Callable[[], dict[str, Any]]
    read_failures: Callable[[], list[dict[str, Any]]]
    read_evaluator_summary: Callable[[], dict[str, Any]]
    fetch_literature: Callable[[str], Awaitable[dict[str, Any]] | dict[str, Any]]
    propose_strategy: Callable[[dict[str, Any]], dict[str, Any]]
    run_eval: Callable[[dict[str, Any]], dict[str, Any]]
    compare_variants: Callable[[], dict[str, Any]]
    stop: Callable[[str], dict[str, Any]]


@dataclass
class OptimizerControllerResult:
    round_index: int
    actions: list[dict[str, Any]]
    reflection: str
    mechanism_change_required: bool
    literature_required: bool
    prompt_context: dict[str, Any] = field(default_factory=dict)


class OptimizationAgent:
    """Model-driven controller for optimization rounds.

    The evaluator loop remains the execution engine, but this controller owns
    round-level observation and action choice. It exposes the optimizer tools as
    an explicit menu, records which tools were actually used, and returns a
    prompt context that code generation must consume.
    """

    TOOL_NAMES = [
        "read_champion",
        "read_failures",
        "read_evaluator_summary",
        "fetch_literature",
        "propose_strategy",
        "run_eval",
        "compare_variants",
        "stop",
    ]
    ROLE_CONTRACT = {
        "role": "optimization_controller",
        "single_goal": "choose the next optimization direction and tool sequence needed to improve evaluator score",
        "does_not_do": [
            "write final candidate code",
            "decide that work is correct",
            "run final validation as its own judge",
            "accumulate unrelated research context",
        ],
        "context_policy": [
            "read only champion, recent failures, evaluator summary, variant comparison, targeted literature, and compact prior controller state",
            "ignore context that does not change the next mechanism, failure reflection, or literature query",
            "handoff implementation details to proposal/worker steps and correctness judgment to independent validators",
        ],
    }

    def __init__(self, *, run_id: str, goal: str, llm: Optional[LLMClient] = None):
        self.run_id = run_id
        self.goal = goal
        self.llm = llm or LLMClient()

    async def plan_round(
        self,
        *,
        store: ArtifactStore,
        round_index: int,
        toolbox: OptimizerToolbox,
        prior_context: Optional[dict[str, Any]] = None,
    ) -> OptimizerControllerResult:
        started = time.perf_counter()
        started_at = now_iso()
        actions: list[dict[str, Any]] = []
        tools_used: list[str] = []
        errors: list[str] = []

        champion = self._call_tool("read_champion", toolbox.read_champion, actions, tools_used)
        failures = self._call_tool("read_failures", toolbox.read_failures, actions, tools_used)
        evaluator_summary = self._call_tool("read_evaluator_summary", toolbox.read_evaluator_summary, actions, tools_used)
        comparison = self._call_tool("compare_variants", toolbox.compare_variants, actions, tools_used)
        state = {
            "role_contract": self.ROLE_CONTRACT,
            "goal": self.goal,
            "round_index": round_index,
            "champion": champion,
            "recent_failures": failures,
            "evaluator_summary": evaluator_summary,
            "variant_comparison": comparison,
            "prior_context": self._scoped_prior_context(prior_context),
        }
        decision = self._decide(state, errors)
        if decision.get("fetch_literature"):
            literature_query = str(decision.get("literature_query") or self.goal)
            literature = await self._call_tool_async("fetch_literature", toolbox.fetch_literature, literature_query, actions, tools_used)
            state["fresh_literature"] = literature
        strategy = self._call_tool("propose_strategy", toolbox.propose_strategy, state, actions, tools_used)
        state["strategy_directive"] = strategy

        result = OptimizerControllerResult(
            round_index=round_index,
            actions=actions,
            reflection=str(decision.get("failure_reflection") or strategy.get("failure_reflection") or ""),
            mechanism_change_required=bool(decision.get("mechanism_change_required", True)),
            literature_required=bool(decision.get("fetch_literature", False)),
            prompt_context={
                "controller": "OptimizationAgent",
                "role_contract": self.ROLE_CONTRACT,
                "tool_menu": self.TOOL_NAMES,
                "actions": actions,
                "failure_reflection": str(decision.get("failure_reflection") or ""),
                "next_mechanism": str(decision.get("next_mechanism") or strategy.get("next_mechanism") or ""),
                "mechanism_change_required": bool(decision.get("mechanism_change_required", True)),
                "literature_required": bool(decision.get("fetch_literature", False)),
                "literature_query": str(decision.get("literature_query") or ""),
                "state": state,
            },
        )
        self._persist(store, result)
        runtime_ms = int((time.perf_counter() - started) * 1000)
        store.add_trace(
            AgentTrace(
                run_id=self.run_id,
                agent_name=f"optimization_agent:round_{round_index}",
                role="optimizer_controller",
                prompt=json.dumps(state, sort_keys=True, default=str)[:12000],
                model=self.llm.model_label,
                tools_used=tools_used,
                tool_calls=actions,
                token_usage=0,
                runtime_ms=runtime_ms,
                status="completed" if not errors else "failed",
                errors=errors,
                output_summary=(
                    f"Controller selected {len(actions)} action(s); "
                    f"mechanism_change_required={result.mechanism_change_required}; "
                    f"literature_required={result.literature_required}."
                ),
                started_at=started_at,
            )
        )
        store.append_progress(
            f"Optimization agent round {round_index}: tools={','.join(tools_used)} "
            f"mechanism='{result.prompt_context.get('next_mechanism', '')}'"
        )
        return result

    def _decide(self, state: dict[str, Any], errors: list[str]) -> dict[str, Any]:
        if self.llm.is_live:
            system = (
                "You are the controller for an optimization agent. Your single goal is to choose the next "
                "optimization direction and tool sequence needed to improve evaluator score. You do not write "
                "final code, decide correctness, or perform validation as your own judge. Choose actions from this tool menu only: "
                f"{', '.join(self.TOOL_NAMES)}. Return JSON with fetch_literature, literature_query, "
                "failure_reflection, next_mechanism, and mechanism_change_required. "
                "Use only context relevant to the next mechanism. If recent mean_edge values are non-positive "
                "or flat, require a foundational mechanism change."
            )
            try:
                decision = self.llm.complete_json(system, json.dumps(state, sort_keys=True, default=str), max_output_tokens=900, temperature=0.2)
                if isinstance(decision, dict):
                    return decision
            except Exception as exc:  # pragma: no cover - defensive live-model fallback
                errors.append(f"{type(exc).__name__}: {exc}")
        return self._fallback_decision(state)

    def _fallback_decision(self, state: dict[str, Any]) -> dict[str, Any]:
        failures = state.get("recent_failures") if isinstance(state.get("recent_failures"), list) else []
        edges = [float(row.get("mean_edge", 0.0)) for row in failures if row.get("mean_edge") is not None]
        flat_or_negative = bool(edges) and max(edges) <= 0.0
        summaries = "; ".join(str(row.get("summary", ""))[:160] for row in failures[:3])
        return {
            "fetch_literature": flat_or_negative or state.get("round_index", 1) > 1,
            "literature_query": f"{self.goal} mechanism evidence after failures {summaries}".strip(),
            "failure_reflection": summaries or "No prior failures yet; establish a concrete mechanism and evaluator-facing baseline.",
            "next_mechanism": (
                "replace the current quoting architecture with a new evidence-grounded mechanism"
                if flat_or_negative
                else "test a concrete evaluator-facing strategy with explicit order placement and risk gates"
            ),
            "mechanism_change_required": flat_or_negative,
        }

    def _scoped_prior_context(self, prior_context: Optional[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(prior_context, dict):
            return {}
        allowed = {
            "failure_reflection",
            "next_mechanism",
            "mechanism_change_required",
            "literature_required",
            "literature_query",
        }
        scoped = {key: prior_context.get(key) for key in allowed if key in prior_context}
        actions = prior_context.get("actions")
        if isinstance(actions, list):
            scoped["previous_tools"] = [
                str(action.get("tool"))
                for action in actions
                if isinstance(action, dict) and action.get("tool")
            ][:12]
        return scoped

    def _call_tool(
        self,
        name: str,
        fn: Callable[..., Any],
        *args: Any,
    ) -> Any:
        actions = args[-2]
        tools_used = args[-1]
        call_args = args[:-2]
        started = time.perf_counter()
        try:
            output = fn(*call_args)
            status = "completed"
        except Exception as exc:  # pragma: no cover - defensive trace path
            output = {"error": f"{type(exc).__name__}: {exc}"}
            status = "failed"
        tools_used.append(name)
        actions.append(
            {
                "tool": name,
                "status": status,
                "runtime_ms": int((time.perf_counter() - started) * 1000),
                "output_preview": _preview(output),
            }
        )
        return output

    async def _call_tool_async(
        self,
        name: str,
        fn: Callable[..., Any],
        *args: Any,
    ) -> Any:
        actions = args[-2]
        tools_used = args[-1]
        call_args = args[:-2]
        started = time.perf_counter()
        try:
            output = fn(*call_args)
            if inspect.isawaitable(output):
                output = await output
            status = "completed"
        except Exception as exc:  # pragma: no cover - defensive trace path
            output = {"error": f"{type(exc).__name__}: {exc}"}
            status = "failed"
        tools_used.append(name)
        actions.append(
            {
                "tool": name,
                "status": status,
                "runtime_ms": int((time.perf_counter() - started) * 1000),
                "output_preview": _preview(output),
            }
        )
        return output

    def _persist(self, store: ArtifactStore, result: OptimizerControllerResult) -> None:
        path = getattr(store, "optimizer_agent_steps_path", store.root / "optimization_agent_steps.json")
        rows: list[dict[str, Any]]
        if path.exists():
            try:
                rows = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                rows = []
        else:
            rows = []
        rows.append(
            {
                "round_index": result.round_index,
                "actions": result.actions,
                "reflection": result.reflection,
                "mechanism_change_required": result.mechanism_change_required,
                "literature_required": result.literature_required,
                "prompt_context": result.prompt_context,
            }
        )
        path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _preview(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)[:700]
