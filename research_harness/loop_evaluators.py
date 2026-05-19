from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class EvaluatorResult:
    score: float
    status: str = "completed"
    metrics: dict[str, object] | None = None
    diagnostics: dict[str, object] | None = None
    loss_reason: str = ""
    summary: str = ""


EvaluatorPayload = Union[float, int, dict[str, object], EvaluatorResult]


def normalize_evaluator_result(raw_result: EvaluatorPayload) -> EvaluatorResult:
    if isinstance(raw_result, EvaluatorResult):
        return EvaluatorResult(
            score=max(0.0, min(1.0, float(raw_result.score))),
            status=raw_result.status or "completed",
            metrics=dict(raw_result.metrics or {}),
            diagnostics=dict(raw_result.diagnostics or {}),
            loss_reason=raw_result.loss_reason,
            summary=raw_result.summary,
        )
    if isinstance(raw_result, dict):
        score = float(raw_result.get("score", raw_result.get("deterministic_score", 0.0)) or 0.0)
        diagnostics = raw_result.get("diagnostics") if isinstance(raw_result.get("diagnostics"), dict) else {}
        metrics = raw_result.get("metrics") if isinstance(raw_result.get("metrics"), dict) else {}
        if not metrics:
            metrics = {
                key: value
                for key, value in raw_result.items()
                if key not in {"score", "deterministic_score", "status", "diagnostics", "loss_reason", "summary"}
            }
        return EvaluatorResult(
            score=max(0.0, min(1.0, score)),
            status=str(raw_result.get("status") or "completed"),
            metrics=dict(metrics),
            diagnostics=dict(diagnostics),
            loss_reason=str(raw_result.get("loss_reason") or raw_result.get("failure_reason") or ""),
            summary=str(raw_result.get("summary") or ""),
        )
    score = max(0.0, min(1.0, float(raw_result)))
    return EvaluatorResult(
        score=score,
        status="completed",
        metrics={"scalar_score": score},
        diagnostics={},
        loss_reason="" if score > 0 else "zero_score",
        summary=f"Deterministic scalar evaluator returned {score:.3f}.",
    )


def exception_evaluator_result(exc: Exception) -> EvaluatorResult:
    exc_type = type(exc).__name__
    lower = str(exc).lower()
    if "timeout" in lower or exc_type.lower().find("timeout") >= 0:
        reason = "timeout"
    elif "compile" in lower or "syntax" in lower:
        reason = "compile_error"
    elif "correct" in lower or "assert" in lower:
        reason = "correctness_fail"
    elif "slow" in lower or "baseline" in lower:
        reason = "slower_than_baseline"
    else:
        reason = "runtime_error"
    return EvaluatorResult(
        score=0.0,
        status="failed",
        metrics={},
        diagnostics={"exception_type": exc_type, "exception": str(exc)},
        loss_reason=reason,
        summary=f"{exc_type}: {exc}",
    )


def evaluator_json_response(result: EvaluatorResult) -> dict[str, object]:
    return {
        "status": result.status,
        "score": result.score,
        "metrics": result.metrics or {},
        "diagnostics": result.diagnostics or {},
        "loss_reason": result.loss_reason,
        "summary": result.summary,
    }
