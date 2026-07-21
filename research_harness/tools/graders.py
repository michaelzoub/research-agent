"""Model-selectable evaluator tools for the model-directed agent loop."""
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from .base import ToolContext, ToolResult


def get_optimization_grader(identifier: str):
    """Lazy import avoids a package-startup cycle with tool registration."""
    from optimization_graders import get_optimization_grader as resolve_grader

    return resolve_grader(identifier)


class PredictionMarketEvaluationTool:
    """Run a model-authored strategy against the official upstream scorer."""

    name = "evaluate_prediction_market_candidate"
    is_read_only = False
    description = (
        "Evaluate one complete, model-authored Python Strategy(BaseStrategy) candidate against the configured "
        "prediction-market upstream grader. Use when scorer feedback would help decide the next action. "
        "The tool persists the exact candidate and official command/output; unavailable or failed scorer runs are errors, not scores."
    )
    input_schema = {
        "type": "object",
        "required": ["code"],
        "properties": {
            "code": {"type": "string", "minLength": 40, "maxLength": 50_000},
            "rationale": {"type": "string", "maxLength": 2_000},
        },
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        code = str(arguments["code"])
        if "class Strategy" not in code:
            return ToolResult("error", error="Candidate must define class Strategy(BaseStrategy); no grader run was attempted.")
        if context.store is None:
            return ToolResult("error", error="Prediction-market evaluation requires an artifact store.")

        strategy_id = f"model_{hashlib.sha256(code.encode('utf-8')).hexdigest()[:16]}"
        # A strategy fingerprint is not an evaluation identity: the model can
        # legitimately submit the same code more than once (for example when
        # confirming a noisy scorer result).  Reusing the fingerprint as the
        # trial filename silently overwrote the earlier JSON record, making an
        # eight-round CLI run appear to contain only one evaluation.
        round_index = _next_measured_round_index(context.store)
        candidate_id = f"{strategy_id}_round_{round_index:03d}"
        candidate_path = Path(context.store.candidates_dir) / f"{candidate_id}.py"
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_text(code, encoding="utf-8")
        # Keep a browsable Python artifact beside the JSON trial audit record.
        trial_code_path = context.store.write_optimization_trial_code(candidate_id, code)
        grader = get_optimization_grader("prediction_market")
        result = await asyncio.to_thread(grader.evaluate, candidate_path)
        trial = {
            "trial_id": candidate_id,
            "strategy_id": strategy_id,
            "run_id": context.run_id,
            "grader_id": "prediction_market",
            "candidate_path": str(candidate_path),
            "trial_code_path": str(trial_code_path),
            "round_index": round_index,
            "rendered_code": code,
            "rationale": str(arguments.get("rationale") or ""),
            "command": list((result.get("upstream") or {}).get("command") or []),
            "upstream": result.get("upstream") or {},
            "score": float(result.get("mean_edge") or 0.0),
            "score_eligible": bool(result.get("score_eligible", False)),
            "official_measured": bool(result.get("official_measured", False)),
            "stdout": str(result.get("stdout") or ""),
            "stderr": str(result.get("stderr") or ""),
            "failure": result.get("error"),
            "metrics": result,
        }
        trial_path = context.store.write_optimization_trial(candidate_id, trial)
        data = {
            "candidate_id": candidate_id,
            "trial_path": str(trial_path),
            "official_measured": trial["official_measured"],
            "score_eligible": trial["score_eligible"],
            "mean_edge": result.get("mean_edge"),
            "mean_arb_edge": result.get("mean_arb_edge"),
            "mean_retail_edge": result.get("mean_retail_edge"),
            "score_source": result.get("score_source"),
            "failure": result.get("error"),
        }
        if trial["official_measured"] and trial["score_eligible"]:
            data["promotion"] = _promote_model_directed_round(
                context.store,
                candidate_id=candidate_id,
                trial_path=trial_path,
                round_index=round_index,
                score=trial["score"],
                metrics=result,
            )
        if not trial["official_measured"] or not trial["score_eligible"]:
            return ToolResult("error", data, error=str(result.get("error") or "The official scorer did not produce an eligible measurement."), retryable=True)
        return ToolResult("ok", data)


def _promote_model_directed_round(
    store: Any,
    *,
    candidate_id: str,
    trial_path: Path,
    round_index: int,
    score: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Persist the measured winner and return compact comparison context.

    The active harness has one model-directed trajectory, so each eligible
    evaluator call is a candidate round.  This makes the result of every round
    durable and supplies the following model turn with the incumbent and the
    measured history it must improve upon.
    """
    trials: list[dict[str, Any]] = []
    for path in sorted(store.optimization_trials_dir.glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if item.get("official_measured") and item.get("score_eligible"):
            trials.append(item)
    incumbent = max(trials, key=lambda item: float(item.get("score") or 0.0))
    promoted = incumbent.get("trial_id") == candidate_id
    champion = {
        "round_index": round_index,
        "variant_id": candidate_id,
        "score": score,
        "trial_path": str(trial_path),
        "metrics": {
            "mean_edge": metrics.get("mean_edge"),
            "mean_arb_edge": metrics.get("mean_arb_edge"),
            "mean_retail_edge": metrics.get("mean_retail_edge"),
            "score_source": metrics.get("score_source"),
        },
        "global_champion": {
            "variant_id": incumbent.get("trial_id"),
            "score": float(incumbent.get("score") or 0.0),
            "trial_path": str(incumbent.get("trial_path") or ""),
            "promoted_this_round": promoted,
        },
    }
    store.write_round_champion(round_index, champion)
    recent = [
        {
            "candidate_id": item.get("trial_id"),
            "mean_edge": item.get("metrics", {}).get("mean_edge"),
            "mean_arb_edge": item.get("metrics", {}).get("mean_arb_edge"),
            "mean_retail_edge": item.get("metrics", {}).get("mean_retail_edge"),
            "rationale": item.get("rationale"),
        }
        for item in trials[-8:]
    ]
    return {
        "round_index": round_index,
        "promoted_this_round": promoted,
        "global_champion": champion["global_champion"],
        "measured_history": recent,
    }


def _next_measured_round_index(store: Any) -> int:
    rounds = 0
    for path in store.optimization_trials_dir.glob("*.json"):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if item.get("official_measured") and item.get("score_eligible"):
            rounds = max(rounds, int(item.get("round_index") or 0))
    return rounds + 1


def evaluator_context(evaluator_name: str | None) -> str:
    if evaluator_name != "prediction_market":
        return ""
    return (
        "A prediction_market grader is configured. ‘PM challenge’ means the configured prediction-market challenge, "
        "not project/product management. The only scoring authority is the upstream prediction-market evaluator. "
        "The CLI injects the exact adapter-resolved upstream baseline source and absolute path into this objective; do not guess or search for a starter_strategy path. Use it to verify the public interface, but improve it only through observed official scorer feedback. "
        "Selecting this grader is an explicit request to optimize against it, not to give a generic PM overview. "
        "The required public interface is exactly: "
        "from orderbook_pm_challenge.strategy import BaseStrategy; "
        "from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState; "
        "class Strategy(BaseStrategy): def on_step(self, state: StepState): return a sequence of CancelAll(), "
        "PlaceOrder(side=Side.BUY or Side.SELL, price_ticks=1..99, quantity=float), or both. "
        "StepState provides step, steps_remaining, yes_inventory, no_inventory, cash, reserved_cash, free_cash, "
        "competitor_best_bid_ticks, competitor_best_ask_ticks, buy_filled_quantity, sell_filled_quantity, and own_orders. "
        "Do not use getattr(), setattr(), delattr(), vars(), eval(), exec(), or __import__(): the sandbox rejects them. "
        "Develop a candidate and call "
        "evaluate_prediction_market_candidate to obtain real scorer feedback. Treat every disappointing or non-positive score as a "
        "diagnostic observation, not a reason to stop: inspect its component metrics, fetch or search for evidence about the exposed "
        "failure mode, and use that evidence to make the next candidate materially different. When --grader-loops is configured, it is "
        "the user's requested number of official candidate evaluations (subject only to a genuine blocker such as an unavailable scorer); "
        "continue the measured feedback loop until that budget is used instead of declaring a best-so-far result after a few trials. "
        "Do not ask what PM means, invent a fallback score, or follow a fixed strategy sequence."
    )
