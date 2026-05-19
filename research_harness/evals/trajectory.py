from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..store import ArtifactStore
from .types import TrajectoryMatchMode


def outcome_from_store(store: ArtifactStore) -> dict[str, Any]:
    runs = store.list("runs")
    run = runs[0] if runs else {}
    evaluations = store.list("variant_evaluations")
    best = max(evaluations, key=lambda row: float(row.get("score", 0.0)), default={})
    return {
        "run_id": run.get("id"),
        "status": run.get("status"),
        "task_mode": run.get("task_mode"),
        "product_agent": run.get("product_agent"),
        "best_score": best.get("score"),
        "best_inner_loop": best.get("inner_loop"),
        "report_exists": store.report_path.exists(),
        "solution_exists": store.solution_path.exists(),
        "optimized_candidate_exists": store.optimized_candidate_path.exists(),
        "optimal_code_exists": store.optimal_code_path.exists(),
        "optimization_result_exists": store.optimization_result_path.exists(),
        "seed_context_exists": store.optimizer_seed_context_path.exists(),
    }


def trajectory_graph(store: ArtifactStore) -> dict[str, Any]:
    decisions = store.list("task_ingestion_decisions")
    decision = decisions[0] if decisions else {}
    rounds = store.list("evolution_rounds")
    evaluations = store.list("variant_evaluations")
    traces = store.list("agent_traces")
    return {
        "nodes": [
            {"id": "prompt", "label": "Prompt"},
            {"id": "router", "label": f"Router: {decision.get('selected_mode', 'unknown')}"},
            {"id": "outer", "label": "Outer loop"},
            {"id": "inner", "label": "Inner evaluator"},
            {"id": "select", "label": "Selection"},
            {"id": "agents", "label": f"Role agents: {len(traces)}"},
            {"id": "outcome", "label": "Outcome"},
        ],
        "edges": [
            {"from": "prompt", "to": "router"},
            {"from": "router", "to": "outer"},
            {"from": "outer", "to": "inner"},
            {"from": "inner", "to": "select"},
            {"from": "select", "to": "outer", "kind": "loop"},
            {"from": "select", "to": "agents"},
            {"from": "agents", "to": "outcome"},
        ],
        "rounds": rounds,
        "evaluation_count": len(evaluations),
    }


def write_trajectory_graph_artifacts(store: ArtifactStore, trial_root: Path) -> dict[str, Path]:
    graph = trajectory_graph(store)
    mmd_path = trial_root / "trajectory_graph.mmd"
    svg_path = trial_root / "trajectory_graph.svg"
    json_path = trial_root / "trajectory_graph.json"
    json_path.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["flowchart TD"]
    for node in graph["nodes"]:
        lines.append(f'  {node["id"]}["{_mmd(str(node["label"]))}"]')
    for edge in graph["edges"]:
        connector = "-.->" if edge.get("kind") == "loop" else "-->"
        lines.append(f'  {edge["from"]} {connector} {edge["to"]}')
    for index, round_record in enumerate(graph["rounds"], start=1):
        round_id = f"round_{index}"
        label = (
            f"Round {round_record.get('outer_iteration')}: "
            f"{round_record.get('mode')} best={float(round_record.get('best_score', 0.0)):.3f} "
            f"{round_record.get('termination_signal')}"
        )
        lines.append(f'  {round_id}["{_mmd(label)}"]')
        lines.append(f"  inner --> {round_id} --> select")
    mmd_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    svg_path.write_text(_trajectory_graph_svg(graph), encoding="utf-8")
    return {"mmd": mmd_path, "svg": svg_path, "json": json_path}


def trajectory_optimizer_flow(store: ArtifactStore) -> str:
    rounds = store.list("evolution_rounds")
    claims = store.list("claims")
    sources_by_id = {str(source.get("id")): source for source in store.list("sources")}
    lines = ["flowchart TD", '  start["Optimizer trajectory"]']
    previous_node = "start"
    for index, round_record in enumerate(rounds, start=1):
        iteration = _round_iteration(round_record, index)
        signal = str(round_record.get("termination_signal", "unknown"))
        mode = str(round_record.get("mode", "unknown"))
        score = float(round_record.get("best_score", 0.0) or 0.0)
        round_node = f"round_{iteration}"
        entropy_node = f"entropy_{iteration}"
        label = f"Round {iteration}\\n{mode} best={score:.3f}\\nsignal={signal}"
        lines.append(f'  {round_node}["{_mmd(label)}"]')
        lines.append(f"  {previous_node} --> {round_node}")
        previous_node = round_node
        if signal == "continue":
            entropy = _post_round_entropy_record(round_record, iteration, claims, sources_by_id)
            if entropy["introduced_after_round"]:
                entropy_label = (
                    f"Post-round entropy introduced\\n"
                    f"{entropy['reason']}\\n"
                    f"{entropy['source_title']}"
                )
                edge_label = "fresh literature before next proposal"
            else:
                entropy_label = (
                    "Missing post-round entropy\\n"
                    f"expected {entropy['reason']}\\n"
                    "no later source evidence found"
                )
                edge_label = "entropy missing"
            lines.append(f'  {entropy_node}["{_mmd(entropy_label)}"]')
            lines.append(f"  {round_node} -->|{_mmd(edge_label)}| {entropy_node}")
            previous_node = entropy_node
    return "\n".join(lines) + "\n"


def normalized_trajectory_events(store: ArtifactStore) -> list[dict[str, str]]:
    runs = store.list("runs")
    run = runs[0] if runs else {}
    decisions = store.list("task_ingestion_decisions")
    decision = decisions[0] if decisions else {}
    events = [
        {"type": "router", "name": str(decision.get("selected_mode", run.get("task_mode", "unknown")))},
    ]
    for round_record in store.list("evolution_rounds"):
        mode = str(round_record.get("mode", "unknown"))
        signal = str(round_record.get("termination_signal", "unknown"))
        events.extend(
            [
                {"type": "outer_loop", "name": mode},
                {"type": "inner_loop", "name": mode},
                {"type": "selection", "name": "variant"},
                {"type": "signal", "name": signal},
            ]
        )
    for trace in store.list("agent_traces"):
        role = str(trace.get("role", trace.get("agent_name", "agent")))
        events.append({"type": "role_agent", "name": role})
    status = str(run.get("status", "unknown"))
    events.append({"type": "outcome", "name": status})
    return events


def trajectory_match(actual: list[dict[str, str]], reference: list[dict[str, str]], mode: TrajectoryMatchMode) -> dict[str, Any]:
    if mode == "strict":
        passed = _trajectory_startswith(actual, reference)
    elif mode == "unordered":
        passed = _multiset_contains(actual, reference) and len(actual) >= len(reference)
    elif mode == "subset":
        allowed = {_event_key(item) for item in reference}
        passed = all(_event_key(item) in allowed for item in actual)
    elif mode == "superset":
        passed = _ordered_subsequence(actual, reference)
    else:
        passed = False
    return {
        "mode": mode,
        "passed": passed,
        "actual_count": len(actual),
        "reference_count": len(reference),
        "actual": actual,
        "reference": reference,
    }


def graph_trajectory_match(graph: dict[str, Any], required_edges: list[list[str]]) -> dict[str, Any]:
    actual_edges = {(edge.get("from"), edge.get("to")) for edge in graph.get("edges", [])}
    missing = [edge for edge in required_edges if tuple(edge) not in actual_edges]
    return {
        "passed": not missing,
        "missing_edges": missing,
        "actual_edges": sorted([list(edge) for edge in actual_edges]),
    }


def _round_iteration(round_record: dict[str, Any], fallback: int) -> int:
    try:
        return int(round_record.get("outer_iteration") or fallback)
    except (TypeError, ValueError):
        return fallback


def _post_round_entropy_record(
    round_record: dict[str, Any],
    iteration: int,
    claims: list[dict[str, Any]],
    sources_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    mode = str(round_record.get("mode", ""))
    reasons = [f"optimizer_entropy_after_round_{iteration}"]
    if mode in {"optimize", "optimize_query"}:
        reasons.append(f"prediction_market_entropy_after_round_{iteration}")
    round_completed = _parse_iso(str(round_record.get("completed_at", "")))
    for reason in reasons:
        matching_claims = [
            claim
            for claim in claims
            if claim.get("created_by_agent") == "literature_grounding_policy"
            and f"Literature grounding ({reason})" in str(claim.get("text", ""))
        ]
        for claim in matching_claims:
            for source_id in claim.get("source_ids", []):
                source = sources_by_id.get(str(source_id))
                if not source:
                    continue
                retrieved_at = _parse_iso(str(source.get("retrieved_at", "")))
                if retrieved_at is None or round_completed is None or retrieved_at >= round_completed:
                    return {
                        "introduced_after_round": True,
                        "reason": reason,
                        "source_title": str(source.get("title") or source.get("url") or source_id),
                    }
    return {
        "introduced_after_round": False,
        "reason": reasons[0],
        "source_title": "",
    }


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _event_key(event: dict[str, str]) -> tuple[str, str]:
    return (str(event.get("type", "")), str(event.get("name", "")))


def _trajectory_startswith(actual: list[dict[str, str]], reference: list[dict[str, str]]) -> bool:
    if len(actual) < len(reference):
        return False
    return [_event_key(item) for item in actual[: len(reference)]] == [_event_key(item) for item in reference]


def _ordered_subsequence(actual: list[dict[str, str]], reference: list[dict[str, str]]) -> bool:
    actual_keys = [_event_key(item) for item in actual]
    position = 0
    for ref in reference:
        key = _event_key(ref)
        try:
            position = actual_keys.index(key, position) + 1
        except ValueError:
            return False
    return True


def _multiset_contains(actual: list[dict[str, str]], reference: list[dict[str, str]]) -> bool:
    remaining = [_event_key(item) for item in actual]
    for ref in reference:
        key = _event_key(ref)
        if key not in remaining:
            return False
        remaining.remove(key)
    return True


def _mmd(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ")


def _trajectory_graph_svg(graph: dict[str, Any]) -> str:
    nodes = graph["nodes"]
    rounds = graph["rounds"]
    width = 1180
    height = 220 + max(0, len(rounds) - 1) * 42
    box_w = 132
    x0 = 28
    y = 42
    gap = 34
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='#fbfaf7'/>",
        "<text x='28' y='26' font-size='18' font-family='Verdana' fill='#102a43'>Eval Trajectory Graph</text>",
    ]
    for index, node in enumerate(nodes):
        x = x0 + index * (box_w + gap)
        parts.append(f"<rect x='{x}' y='{y}' width='{box_w}' height='66' rx='8' fill='#ffffff' stroke='#52606d'/>")
        parts.append(f"<text x='{x + 10}' y='{y + 38}' font-size='12' font-family='Verdana' fill='#102a43'>{html.escape(str(node['label']))}</text>")
        if index < len(nodes) - 1:
            ax = x + box_w
            parts.append(f"<path d='M {ax + 3} {y + 33} L {ax + gap - 5} {y + 33}' stroke='#829ab1' stroke-width='2'/>")
            parts.append(f"<path d='M {ax + gap - 13} {y + 27} L {ax + gap - 4} {y + 33} L {ax + gap - 13} {y + 39}' fill='none' stroke='#829ab1' stroke-width='2'/>")
    ry = 148
    for index, round_record in enumerate(rounds, start=1):
        label = (
            f"Round {round_record.get('outer_iteration')} | {round_record.get('mode')} | "
            f"best {float(round_record.get('best_score', 0.0)):.3f} | {round_record.get('termination_signal')}"
        )
        parts.append(f"<rect x='198' y='{ry}' width='760' height='30' rx='6' fill='#eef2ff' stroke='#627d98'/>")
        parts.append(f"<text x='212' y='{ry + 20}' font-size='12' font-family='Verdana' fill='#243b53'>{html.escape(label)}</text>")
        ry += 42
    parts.append("</svg>")
    return "\n".join(parts)
