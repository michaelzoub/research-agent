from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LoopObjective:
    kind: str = "score"
    target: Optional[float] = None
    no_stop_until_target: bool = False

    @property
    def has_explicit_target(self) -> bool:
        return self.target is not None


def loop_objective_from_goal(goal: str, evaluator_name: Optional[str]) -> LoopObjective:
    normalized = goal.lower()
    no_stop = bool(re.search(r"\bdo\s*not\s*stop\b|\bdon't\s*stop\b|\bdont\s*stop\b|until\s+you", normalized))
    if evaluator_name == "prediction_market":
        target = profit_target_from_goal(goal)
        return LoopObjective(kind="profit_usd", target=target, no_stop_until_target=no_stop and target is not None)
    return LoopObjective(kind="score", target=None, no_stop_until_target=no_stop)


def profit_target_from_goal(goal: str) -> Optional[float]:
    normalized = goal.lower()
    patterns = [
        r"(?:get\s+to|reach|hit|achieve|make|earn)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:\$|usd|dollars?)?\s*(?:profit|edge)?",
        r"\$+\s*([0-9]+(?:\.[0-9]+)?)\s*(?:profit|edge|usd|dollars?)",
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:\$|usd|dollars?)\s*(?:profit|edge)",
    ]
    if not any(term in normalized for term in ["profit", "profitable", "edge", "$", "usd", "dollar"]):
        return None
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return float(match.group(1))
    return None


def objective_metadata(evaluator_name: str) -> dict[str, object]:
    if evaluator_name == "prediction_market":
        return {
            "objective_name": "prediction_market_mean_edge",
            "objective_direction": "maximize",
            "official_result": {
                "measured": True,
                "profit_usd": None,
                "score_source": "upstream_orderbook_pm_challenge_when_available",
                "required_evaluator": "https://github.com/danrobinson/prediction-market-challenge",
                "reason": "Optimization evaluates generated candidates with the upstream orderbook-pm runner when available.",
            },
            "note": (
                "Prediction-market optimization uses upstream mean edge when the orderbook_pm_challenge repo is available. "
                "The normalized score is derived from mean edge for harness aggregation."
            ),
        }
    if evaluator_name == "length_score":
        return {
            "objective_name": "length_score",
            "objective_direction": "maximize",
            "official_result": {"measured": True, "score_source": "local_deterministic_evaluator"},
            "note": "This evaluator maximizes 1 / token_count for smoke-test optimization.",
        }
    return {
        "objective_name": evaluator_name or "deterministic_score",
        "objective_direction": "maximize",
        "official_result": {"measured": True, "score_source": "local_deterministic_evaluator"},
        "note": "Optimization result from the registered deterministic evaluator.",
    }
