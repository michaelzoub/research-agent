from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def classify_failure(message: str, *, component: str = "unknown") -> dict[str, Any]:
    text = message.lower()
    category = "unknown"
    retryable = False
    severity = "medium"
    if any(term in text for term in ["timeout", "timed out", "deadline"]):
        category = "timeout"
        retryable = True
    elif any(term in text for term in ["http error", "urlerror", "connection", "rate limit", "429", "503"]):
        category = "tool_or_network"
        retryable = True
    elif any(term in text for term in ["openai", "llm", "model did not", "invalid json"]):
        category = "model_response"
        retryable = True
    elif any(term in text for term in ["evaluator", "score", "sandbox", "subprocess"]):
        category = "evaluator"
    elif any(term in text for term in ["no sources", "no evidence", "too few claims", "low-yield"]):
        category = "evidence_low_yield"
        severity = "high"
    elif any(term in text for term in ["contradiction", "unsupported", "fabricated"]):
        category = "grounding_or_consistency"
        severity = "high"
    elif any(term in text for term in ["objective incomplete", "target", "budget reached"]):
        category = "objective_not_met"
        severity = "high"
    return {
        "category": category,
        "component": component,
        "retryable": retryable,
        "severity": severity,
    }


def component_from_trace(trace: dict[str, Any]) -> str:
    role = str(trace.get("role") or "").lower()
    name = str(trace.get("agent_name") or "").lower()
    text = f"{role} {name}"
    if any(term in text for term in ["search", "literature", "retriever", "memory"]):
        return "retrieval"
    if "hypothesis" in text:
        return "hypothesis_generation"
    if "critic" in text:
        return "critic"
    if "synthesis" in text:
        return "synthesis"
    if any(term in text for term in ["optimize", "evaluator", "prediction_market", "candidate"]):
        return "optimizer"
    if any(term in text for term in ["router", "loop_controller"]):
        return "loop_control"
    if "orchestration" in text:
        return "orchestration"
    if "debug" in text:
        return "harness_debugger"
    return "unknown"


def diagnose_snapshot(snapshot: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    traces = snapshot.get("agent_traces", [])
    claims = snapshot.get("claims", [])
    sources = snapshot.get("sources", [])
    hypotheses = snapshot.get("hypotheses", [])
    contradictions = snapshot.get("contradictions", [])
    failed_paths = snapshot.get("failed_paths", [])
    evaluations = snapshot.get("variant_evaluations", [])
    rounds = snapshot.get("evolution_rounds", [])
    components: dict[str, dict[str, Any]] = defaultdict(lambda: {"trace_count": 0, "failures": 0, "runtime_ms": 0, "tokens": 0})
    failures = []
    for trace in traces:
        component = component_from_trace(trace)
        bucket = components[component]
        bucket["trace_count"] += 1
        bucket["runtime_ms"] += int(trace.get("runtime_ms") or 0)
        bucket["tokens"] += int(trace.get("token_usage") or 0)
        if trace.get("status") != "completed" or trace.get("errors"):
            bucket["failures"] += 1
            detail = classify_failure(" ".join(str(error) for error in trace.get("errors", [])), component=component)
            failures.append({"trace_id": trace.get("id"), "agent_name": trace.get("agent_name"), **detail})
    for failed in failed_paths:
        detail = classify_failure(str(failed.get("reason", "")), component=str(failed.get("failure_component") or "retrieval"))
        failures.append({"failed_path_id": failed.get("id"), "agent_name": failed.get("created_by_agent"), **detail})
    for component, bucket in components.items():
        count = max(int(bucket["trace_count"]), 1)
        bucket["avg_runtime_ms"] = round(int(bucket["runtime_ms"]) / count, 1)
    score_history = [float(row.get("best_score") or 0.0) for row in rounds]
    diagnosis = {
        "artifact_yield": {
            "sources": len(sources),
            "claims": len(claims),
            "hypotheses": len(hypotheses),
            "contradictions": len(contradictions),
            "failed_paths": len(failed_paths),
        },
        "components": dict(sorted(components.items())),
        "failures": failures,
        "failure_taxonomy": dict(Counter(str(item["category"]) for item in failures)),
        "trace_patterns": _trace_patterns(traces),
        "score_patterns": {
            "history": score_history,
            "plateau_rounds": sum(1 for row in rounds if "plateau" in str(row.get("termination_signal", ""))),
            "best_score": max(score_history, default=0.0),
            "evaluation_count": len(evaluations),
        },
        "localized_components": _localize_components(components, failures, claims, sources, contradictions, evaluations),
    }
    return diagnosis


def score_harness_change(change: dict[str, Any], diagnosis: dict[str, Any]) -> dict[str, float]:
    change_text = " ".join(str(change.get(key, "")) for key in ["change", "reason", "expected_effect", "risk"]).lower()
    localized = diagnosis.get("localized_components") or []
    impact = 0.45
    if localized:
        impact += min(0.35, 0.08 * len(localized))
    if any(term in change_text for term in ["contradiction", "unsupported", "ground", "source"]):
        impact += 0.15
    if any(term in change_text for term in ["plateau", "stopping", "loop"]):
        impact += 0.1
    risk = 0.25
    if any(term in change_text for term in ["more runtime", "cost", "parallel", "extra", "threshold"]):
        risk += 0.25
    if any(term in change_text for term in ["replace", "rewrite", "global"]):
        risk += 0.25
    expected_value = max(0.0, min(1.0, impact))
    risk_score = max(0.0, min(1.0, risk))
    priority = max(0.0, min(1.0, (expected_value * 0.75) + ((1.0 - risk_score) * 0.25)))
    return {
        "risk_score": round(risk_score, 3),
        "expected_value_score": round(expected_value, 3),
        "priority_score": round(priority, 3),
    }


def _trace_patterns(traces: list[dict[str, Any]]) -> dict[str, Any]:
    components = Counter(component_from_trace(trace) for trace in traces)
    failure_categories = Counter()
    role_sequence = []
    for trace in traces:
        role_sequence.append(str(trace.get("role") or trace.get("agent_name") or "unknown"))
        if trace.get("status") != "completed" or trace.get("errors"):
            component = component_from_trace(trace)
            detail = classify_failure(" ".join(str(error) for error in trace.get("errors", [])), component=component)
            failure_categories.update([detail["category"]])
    return {
        "components": dict(components),
        "failure_categories": dict(failure_categories),
        "role_sequence": role_sequence[:40],
    }


def _localize_components(
    components: dict[str, dict[str, Any]],
    failures: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    contradictions: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    localized = []
    failure_components = Counter(str(item.get("component") or "unknown") for item in failures)
    for component, count in failure_components.items():
        localized.append({"component": component, "reason": f"{count} structured failure(s) localized to this component.", "severity": "high"})
    if len(sources) < 2:
        localized.append({"component": "retrieval", "reason": "Low source yield limits evidence coverage.", "severity": "high"})
    if len(claims) < 4:
        localized.append({"component": "claim_extraction", "reason": "Low claim yield limits synthesis and hypothesis generation.", "severity": "high"})
    if contradictions and len(contradictions) >= max(3, len(claims)):
        localized.append({"component": "critic", "reason": "Contradiction volume is high relative to claim count.", "severity": "medium"})
    if evaluations:
        best = max(float(row.get("score") or 0.0) for row in evaluations)
        if best < 0.5:
            localized.append({"component": "optimizer", "reason": f"Best evaluation score is low ({best:.3f}).", "severity": "medium"})
    seen = set()
    unique = []
    for item in localized:
        key = (item["component"], item["reason"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique
