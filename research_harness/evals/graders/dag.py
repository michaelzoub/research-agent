from __future__ import annotations

from typing import Any

from ..types import GraderResult, GraderType
from .common import _result


def dag_grader_result(
    grader: str,
    grader_type: GraderType,
    method: str,
    *,
    nodes: list[dict[str, Any]],
    weight: float,
    threshold: float,
    summary: str,
) -> GraderResult:
    scored = [node for node in nodes if float(node.get("max_score", 0.0) or 0.0) > 0]
    earned = sum(float(node.get("score", 0.0) or 0.0) for node in scored)
    possible = sum(float(node.get("max_score", 0.0) or 0.0) for node in scored)
    score = earned / possible if possible else 0.0
    right_behaviors = [
        behavior
        for node in nodes
        for behavior in (node.get("right_behaviors") if isinstance(node.get("right_behaviors"), list) else [])
    ]
    wrong_behaviors = [
        behavior
        for node in nodes
        for behavior in (node.get("wrong_behaviors") if isinstance(node.get("wrong_behaviors"), list) else [])
    ]
    explanation = _dag_explanation(nodes, right_behaviors, wrong_behaviors)
    return _result(
        grader,
        grader_type,
        method,
        score,
        score >= threshold,
        weight,
        f"{summary} Score {score:.3f}. {explanation}",
        [
            {
                "type": "deep_acyclic_graph",
                "score": round(score, 3),
                "threshold": threshold,
                "right_behaviors": right_behaviors,
                "wrong_behaviors": wrong_behaviors,
                "nodes": nodes,
                "explanation": explanation,
            }
        ],
    )


def dag_node(
    node_id: str,
    criteria: str,
    verdict: str,
    score: float,
    max_score: float = 1.0,
    *,
    right: list[str] | None = None,
    wrong: list[str] | None = None,
    evidence: dict[str, Any] | None = None,
    children: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "criteria": criteria,
        "verdict": verdict,
        "score": round(max(0.0, min(score, max_score)), 3),
        "max_score": max_score,
        "right_behaviors": right or [],
        "wrong_behaviors": wrong or [],
        "evidence": evidence or {},
        "children": children or [],
    }


def verdict_from_score(score: float) -> str:
    if score >= 0.9:
        return "strong"
    if score >= 0.7:
        return "adequate"
    if score >= 0.4:
        return "partial"
    return "missing"


def right_wrong(condition: bool, right: str, wrong: str) -> tuple[list[str], list[str]]:
    return ([right], []) if condition else ([], [wrong])


def _dag_explanation(nodes: list[dict[str, Any]], right_behaviors: list[str], wrong_behaviors: list[str]) -> str:
    node_bits = [
        f"{node.get('id')}: {node.get('verdict')} ({float(node.get('score', 0.0) or 0.0):.2f}/{float(node.get('max_score', 0.0) or 0.0):.2f})"
        for node in nodes
    ]
    right = "; ".join(str(item) for item in right_behaviors[:4]) or "No positive behaviors detected."
    wrong = "; ".join(str(item) for item in wrong_behaviors[:4]) or "No major wrong behaviors detected."
    return f"Path: {' -> '.join(node_bits)}. Right: {right}. Wrong: {wrong}."
