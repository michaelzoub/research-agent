from __future__ import annotations

import hashlib
import re
import time
from typing import Optional

from .schemas import AgentTrace, VariantEvaluation
from .store import ArtifactStore


CONTEXT_STOPWORDS = {
    "and",
    "are",
    "but",
    "candidate",
    "challenge",
    "code",
    "for",
    "from",
    "goal",
    "into",
    "none",
    "optimization",
    "optimize",
    "query",
    "research",
    "round",
    "score",
    "strategy",
    "the",
    "this",
    "variant",
    "with",
}


def record_timing_trace(
    store: ArtifactStore,
    run_id: str,
    *,
    agent_name: str,
    role: str,
    prompt: str,
    model: str,
    started_at: str,
    started: float,
    status: str,
    output_summary: str,
    token_usage: int = 0,
    tools_used: Optional[list[str]] = None,
    tool_calls: Optional[list[dict[str, object]]] = None,
    errors: Optional[list[str]] = None,
) -> None:
    store.add_trace(
        AgentTrace(
            run_id=run_id,
            agent_name=agent_name,
            role=role,
            prompt=prompt,
            model=model,
            tools_used=tools_used or [],
            tool_calls=tool_calls or [],
            token_usage=max(0, token_usage),
            runtime_ms=max(0, int((time.perf_counter() - started) * 1000)),
            status=status,
            errors=errors or [],
            output_summary=output_summary,
            started_at=started_at,
            prompt_version=hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            prompt_tokens=max(0, token_usage),
            failure_component=trace_component(role, agent_name),
        )
    )


def trace_component(role: str, agent_name: str) -> str:
    text = f"{role} {agent_name}".lower()
    if any(term in text for term in ["search", "literature", "retriever", "memory"]):
        return "retrieval"
    if "hypothesis" in text:
        return "hypothesis_generation"
    if "critic" in text:
        return "critic"
    if "synthesis" in text:
        return "synthesis"
    if any(term in text for term in ["optimize", "evaluator", "prediction_market"]):
        return "optimizer"
    if "loop_controller" in text:
        return "loop_control"
    if "orchestration" in text:
        return "orchestration"
    return "unknown"


def score_history(store: Optional[ArtifactStore], *, mode: str, limit: int = 8) -> list[dict[str, object]]:
    if store is None:
        return []
    variants = {str(row.get("id")): row for row in store.list("variants")}
    rows = [row for row in store.list("variant_evaluations") if str(row.get("inner_loop")) == mode]
    rows.sort(key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)
    history: list[dict[str, object]] = []
    for row in rows[:limit]:
        variant = variants.get(str(row.get("variant_id")), {})
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        history.append(
            {
                "variant_id": row.get("variant_id"),
                "score": row.get("score"),
                "mean_edge": metrics.get("mean_edge"),
                "score_source": metrics.get("score_source"),
                "payload": str(variant.get("payload", ""))[:900],
                "summary": str(row.get("summary", ""))[:400],
            }
        )
    return history


def json_evaluator_responses(store: ArtifactStore, *, mode: str) -> list[dict[str, object]]:
    rows = [row for row in store.list("variant_evaluations") if str(row.get("inner_loop")) == mode]
    responses: list[dict[str, object]] = []
    for row in rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        response = metrics.get("json_response") if isinstance(metrics, dict) else None
        if not isinstance(response, dict):
            response = {
                "status": metrics.get("evaluator_status", "completed") if isinstance(metrics, dict) else "completed",
                "score": float(row.get("score", 0.0) or 0.0),
                "metrics": metrics,
                "diagnostics": metrics.get("diagnostics", {}) if isinstance(metrics, dict) else {},
                "loss_reason": metrics.get("loss_reason", "") if isinstance(metrics, dict) else "",
                "summary": row.get("summary", ""),
            }
        responses.append(
            {
                "variant_id": row.get("variant_id"),
                "inner_loop": row.get("inner_loop"),
                **response,
            }
        )
    return responses


def context_terms(text: str, limit: int = 12) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", text.lower().replace("-", " ")):
        if token in CONTEXT_STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def strip_run_artifacts(text: str) -> str:
    text = re.sub(r"\b(?:outputs|eval_outputs)/\S+", " ", text)
    text = re.sub(r"\b(?:variant|round)_[A-Za-z0-9_]+\b", " ", text)
    text = re.sub(r"\b[a-f0-9]{12,}\b", " ", text)
    text = re.sub(r"\b\w+\.py\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def support_level(confidence: float) -> str:
    if confidence >= 0.75:
        return "strong"
    if confidence >= 0.55:
        return "moderate"
    return "weak"


def shorten(text: str, limit: int = 140) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
