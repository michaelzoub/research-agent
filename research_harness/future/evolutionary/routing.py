from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from challenges.prediction_market import prediction_market_score

from ...llm import LLMClient
from ...schemas import ProductAgent, TaskMode, new_id


EvaluatorFn = Callable[[str], object]


@dataclass
class TaskIngestionDecision:
    """Archived router decision shape; not part of production run state."""

    requested_mode: str
    selected_mode: TaskMode
    reason: str
    evaluator_name: Optional[str] = None
    product_agent: ProductAgent = "research"
    id: str = field(default_factory=lambda: new_id("decision"))


OPTIMIZE_HINTS = {
    "benchmark",
    "kernel",
    "latency",
    "optimize",
    "performance",
    "score",
    "speed",
    "strategy",
    "swe-bench",
    "throughput",
}


class EvaluatorRegistry:
    """Registry for deterministic evaluators.

    Optimize mode is only valid when a deterministic evaluator is available.
    The tiny built-in evaluator is for smoke tests and architecture demos; real
    tasks should register domain evaluators such as a benchmark harness.
    """

    def __init__(self) -> None:
        self._evaluators: dict[str, EvaluatorFn] = {
            "length_score": lambda payload: 1.0 / max(1, len(payload.split())),
            "prediction_market": prediction_market_score,
        }

    def get(self, name: Optional[str]) -> Optional[EvaluatorFn]:
        if not name:
            return None
        return self._evaluators.get(name)

    def register(self, name: str, evaluator: EvaluatorFn) -> None:
        self._evaluators[name] = evaluator


class TaskRouter:
    def __init__(self, evaluator_registry: EvaluatorRegistry, llm: Optional[LLMClient] = None):
        self.evaluator_registry = evaluator_registry
        self.llm = llm

    def decide(self, goal: str, requested_mode: str = "auto", evaluator_name: Optional[str] = None) -> TaskIngestionDecision:
        requested = requested_mode.lower()
        evaluator = self.evaluator_registry.get(evaluator_name)

        # This challenge must be optimized against its official adapter, never
        # the lightweight legacy proxy evaluator in the generic optimize path.
        if evaluator_name == "prediction_market" and evaluator:
            return self._decide_explicit(goal, "optimize", evaluator_name, evaluator)

        # Explicit mode flags bypass LLM routing because the user already decided.
        if requested != "auto":
            return self._decide_explicit(goal, requested, evaluator_name, evaluator)

        if self.llm and self.llm.is_live:
            try:
                return self._llm_decide(goal, evaluator_name, evaluator)
            except Exception:
                pass

        return self._heuristic_decide(goal, requested_mode, evaluator_name, evaluator)

    def _decide_explicit(
        self,
        goal: str,
        requested: str,
        evaluator_name: Optional[str],
        evaluator: Optional[object],
    ) -> TaskIngestionDecision:
        if requested == "research":
            return TaskIngestionDecision(
                requested_mode=requested,
                selected_mode="research",
                reason="Research mode was explicitly requested via --task-mode.",
                product_agent="research",
            )
        if requested == "optimize_query":
            product_agent = product_agent_for("optimize_query", goal, evaluator_name)
            return TaskIngestionDecision(
                requested_mode=requested,
                selected_mode="optimize_query",
                evaluator_name=evaluator_name if evaluator else None,
                product_agent=product_agent,
                reason=(
                    f"{product_agent.title()} agent selected optimization-query loop (explicit --task-mode); "
                    + (f"evaluator '{evaluator_name}' is registered." if evaluator else "no evaluator registered.")
                ),
            )
        if requested == "optimize" and evaluator_name == "prediction_market" and evaluator:
            return TaskIngestionDecision(
                requested_mode=requested,
                selected_mode="optimize_query",
                evaluator_name=evaluator_name,
                product_agent="challenge",
                reason=(
                    "Challenge agent selected optimization-query loop because the prediction_market evaluator "
                    "requires challenge strategy research before scoring."
                ),
            )
        if requested == "optimize" and evaluator:
            product_agent = product_agent_for("optimize", goal, evaluator_name)
            return TaskIngestionDecision(
                requested_mode=requested,
                selected_mode="optimize",
                evaluator_name=evaluator_name,
                product_agent=product_agent,
                reason=f"{product_agent.title()} agent selected optimize loop (explicit --task-mode) with evaluator '{evaluator_name}'.",
            )
        if requested == "optimize" and not evaluator:
            return TaskIngestionDecision(
                requested_mode=requested,
                selected_mode="research",
                evaluator_name=evaluator_name,
                product_agent="research",
                reason="Optimize mode requested explicitly but no deterministic evaluator was registered; falling back to research mode.",
            )
        return self._heuristic_decide(goal, requested, evaluator_name, evaluator)

    def _llm_decide(
        self,
        goal: str,
        evaluator_name: Optional[str],
        evaluator: Optional[object],
    ) -> TaskIngestionDecision:
        assert self.llm is not None
        system = (
            "You are the task router in a research-and-optimization harness. "
            "Classify the user's goal into exactly one of three modes and explain your reasoning.\n\n"
            "Modes:\n"
            "- research: open-ended literature review or knowledge synthesis with no deterministic score.\n"
            "- optimize: direct optimization against a registered deterministic evaluator (no research phase needed).\n"
            "- optimize_query: research-then-optimize; the agent first explores literature to build strategy context, "
            "then runs an optimizer. Use this when the goal mentions researching before optimizing, or when the task "
            "is a challenge that benefits from domain grounding.\n\n"
            "Return JSON only: {\"selected_mode\": str, \"product_agent\": str, \"confidence\": float, \"reason\": str}\n"
            "product_agent must be one of: research, optimize, challenge.\n"
            "Use 'challenge' when the goal references a specific scored competition or external evaluator."
        )
        user = json.dumps(
            {
                "goal": goal,
                "evaluator_registered": evaluator_name if evaluator else None,
                "available_modes": ["research", "optimize", "optimize_query"],
            },
            indent=2,
        )
        payload = self.llm.complete_json(system, user, max_output_tokens=400)
        selected_mode = str(payload.get("selected_mode", "research")).lower()
        product_agent = str(payload.get("product_agent", "research")).lower()
        reason = str(payload.get("reason", "LLM router selected this mode."))
        confidence = float(payload.get("confidence", 0.8))

        if selected_mode not in {"research", "optimize", "optimize_query"}:
            selected_mode = "research"
        if product_agent not in {"research", "optimize", "challenge"}:
            product_agent = "research"

        return TaskIngestionDecision(
            requested_mode="auto",
            selected_mode=selected_mode,  # type: ignore[arg-type]
            evaluator_name=evaluator_name if evaluator else None,
            product_agent=product_agent,  # type: ignore[arg-type]
            reason=f"[LLM router, confidence={confidence:.2f}] {reason}",
        )

    def _heuristic_decide(
        self,
        goal: str,
        requested_mode: str,
        evaluator_name: Optional[str],
        evaluator: Optional[object],
    ) -> TaskIngestionDecision:
        if looks_like_optimization_query(goal):
            product_agent = product_agent_for("optimize_query", goal, evaluator_name)
            return TaskIngestionDecision(
                requested_mode=requested_mode,
                selected_mode="optimize_query",
                evaluator_name=evaluator_name if evaluator else None,
                product_agent=product_agent,
                reason=(
                    f"[heuristic router] Prompt contains research+optimization signals; "
                    f"routed to optimize_query for {product_agent} agent"
                    + (" with registered evaluator." if evaluator else ".")
                ),
            )
        if evaluator:
            product_agent = product_agent_for("optimize", goal, evaluator_name)
            return TaskIngestionDecision(
                requested_mode=requested_mode,
                selected_mode="optimize",
                evaluator_name=evaluator_name,
                product_agent=product_agent,
                reason=f"[heuristic router] Deterministic evaluator '{evaluator_name}' is registered; selected optimize mode.",
            )
        if any(hint in goal.lower() for hint in OPTIMIZE_HINTS):
            return TaskIngestionDecision(
                requested_mode=requested_mode,
                selected_mode="research",
                product_agent="research",
                reason="[heuristic router] Prompt looks optimization-shaped but no evaluator is registered; using research mode.",
            )
        return TaskIngestionDecision(
            requested_mode=requested_mode,
            selected_mode="research",
            product_agent="research",
            reason="[heuristic router] No evaluator registered and no optimization signal found; defaulting to research mode.",
        )


def looks_like_optimization_query(goal: str) -> bool:
    normalized = goal.lower()
    query_terms = {"research", "find", "query", "search", "investigate", "explore", "look for"}
    return any(term in normalized for term in query_terms) and any(term in normalized for term in OPTIMIZE_HINTS)


def product_agent_for(selected_mode: TaskMode, goal: str, evaluator_name: Optional[str]) -> ProductAgent:
    normalized = goal.lower()
    if evaluator_name == "prediction_market" or "challenge" in normalized:
        return "challenge"
    if selected_mode == "research":
        return "research"
    return "optimize"
