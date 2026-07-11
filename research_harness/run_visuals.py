"""Small, content-backed visual artifacts for model-directed research runs."""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .run_benchmarks import _PngCanvas, _write_png_from_svg_or_fallback
from .store import ArtifactStore


def write_research_run_visuals(store: ArtifactStore, events: Sequence[Any]) -> dict[str, Path]:
    """Write only the two visuals that describe an actual research trajectory.

    Tool calls are concurrent work performed for a single controller agent; the
    timeline makes that distinction explicit instead of misrepresenting tools as
    additional spawned agents.  The historical ``champion_tree`` name is kept
    for compatibility, while research runs label the graphic as a trajectory.
    """
    rows = [_event_row(event) for event in events]
    timeline = _timeline_data(store, rows)
    timeline_svg = _timeline_svg(timeline)
    store.agent_timeline_svg_path.write_text(timeline_svg, encoding="utf-8")
    _write_png_from_svg_or_fallback(
        store.agent_timeline_path,
        timeline_svg,
        lambda: _timeline_png(timeline),
    )

    trajectory = _trajectory_tree(store, rows)
    store.write_champion_tree(trajectory)
    tree_svg = _trajectory_tree_svg(trajectory)
    store.champion_tree_svg_path.write_text(tree_svg, encoding="utf-8")
    _write_png_from_svg_or_fallback(
        store.champion_tree_graph_path,
        tree_svg,
        lambda: _trajectory_tree_png(trajectory),
    )
    return {
        "agent_timeline": store.agent_timeline_path,
        "agent_timeline_svg": store.agent_timeline_svg_path,
        "champion_tree": store.champion_tree_path,
        "champion_tree_graph": store.champion_tree_graph_path,
        "champion_tree_svg": store.champion_tree_svg_path,
    }


def _event_row(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    return {
        "event_type": getattr(event, "event_type", ""),
        "timestamp": getattr(event, "timestamp", ""),
        "tool_call_id": getattr(event, "tool_call_id", None),
        "tool_name": getattr(event, "tool_name", None),
        "result_status": getattr(event, "result_status", None),
        "model_turn": getattr(event, "model_turn", None),
        "sequence": getattr(event, "sequence", 0),
    }


def _timeline_data(store: ArtifactStore, events: list[dict[str, Any]]) -> dict[str, Any]:
    controller_count = max(1, len({str(trace.get("agent_name")) for trace in store.list("agent_traces") if trace.get("agent_name")}))
    timestamped = [(row, _timestamp(row.get("timestamp"))) for row in events]
    points = [point for _, point in timestamped if point is not None]
    origin = min(points) if points else 0.0
    current_batch = 0
    starts: dict[str, tuple[int, float, str]] = {}
    controller_spans: list[dict[str, Any]] = []
    tool_spans: list[dict[str, Any]] = []
    model_turn_count = 0

    for row, point in timestamped:
        if point is None:
            continue
        kind = str(row.get("event_type") or "")
        if kind == "model_turn":
            current_batch += 1
            model_turn_count += 1
            controller_spans.append({"label": f"Model turn {model_turn_count}", "start": point, "end": point + 0.18, "kind": "controller"})
        elif kind == "tool_requested":
            tool_id = str(row.get("tool_call_id") or f"request_{len(starts)}")
            starts[tool_id] = (current_batch, point, str(row.get("tool_name") or "tool"))
        elif kind == "tool_result":
            tool_id = str(row.get("tool_call_id") or "")
            request = starts.get(tool_id)
            if request is None:
                continue
            batch, start, name = request
            status = str(row.get("result_status") or "unknown")
            tool_spans.append({
                "label": _tool_label(name, len(tool_spans) + 1),
                "start": start,
                "end": max(start + 0.02, point),
                "kind": "tools_error" if status not in {"ok", "skipped"} else "tools",
                "detail": f"{name} · {status} · batch {batch}",
                "parallel_calls": 1,
                "batch": batch,
            })

    spans = controller_spans + tool_spans
    end = max((float(span["end"]) for span in spans), default=origin + 1.0)
    return {
        "controller_count": controller_count,
        "model_turn_count": model_turn_count,
        "tool_call_count": len(tool_spans),
        "peak_parallel_tools": _peak_parallelism(tool_spans),
        "origin": origin,
        "duration": max(end - origin, 1.0),
        "spans": spans,
    }


def _timeline_svg(data: dict[str, Any]) -> str:
    tool_spans = [span for span in data["spans"] if span["kind"] != "controller"]
    width, left, right, axis_y, row_height = 1600, 340, 36, 56, 38
    height = axis_y + (len(tool_spans) + 1) * row_height + 44
    chart_width = width - left - right
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="timeline-title timeline-desc" style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif">',
        '<title id="timeline-title">Agent execution timeline</title>',
        f'<desc id="timeline-desc">{data["controller_count"]} controller agent, {data["model_turn_count"]} model turns, {data["tool_call_count"]} tool calls, with up to {data["peak_parallel_tools"]} tool calls running in parallel.</desc>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    for row in range(len(tool_spans) + 1):
        y = axis_y + row * row_height
        if row % 2:
            out.append(f'<rect x="{left}" y="{y}" width="{chart_width + right}" height="{row_height}" fill="#f7f9fc"/>')
    for tick in _time_ticks(data["duration"]):
        x = left + tick / data["duration"] * chart_width
        out.append(f'<line x1="{x:.1f}" y1="{axis_y - 8}" x2="{x:.1f}" y2="{height - 30}" stroke="#dde4ee" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{axis_y - 18}" text-anchor="middle" font-size="13" fill="#8ca0be">{_time_label(tick)}</text>')
    out.append(f'<line x1="{left}" y1="{axis_y - 8}" x2="{width - right}" y2="{axis_y - 8}" stroke="#dde4ee"/>')
    out.append(f'<text x="{left - 14}" y="{axis_y + 24}" text-anchor="end" font-size="15" font-weight="600" fill="#7c3aed">Main LLM calls</text>')
    for index, span in enumerate(tool_spans, start=1):
        y = axis_y + index * row_height + 24
        color = _span_color(span)
        out.append(f'<text x="{left - 14}" y="{y}" text-anchor="end" font-size="15" font-weight="600" fill="{color}">{html.escape(str(span["label"]))}</text>')
    for span in data["spans"]:
        start = (float(span["start"]) - data["origin"]) / data["duration"]
        end = (float(span["end"]) - data["origin"]) / data["duration"]
        x = left + start * chart_width
        width_px = max(5, (end - start) * chart_width)
        row = 0 if span["kind"] == "controller" else tool_spans.index(span) + 1
        y = axis_y + row * row_height + 7
        color = _span_color(span)
        detail = str(span.get("detail") or span["label"])
        out.append(f'<rect x="{x:.1f}" y="{y}" width="{width_px:.1f}" height="25" rx="5" fill="{color}"><title>{html.escape(detail)}</title></rect>')
        if width_px >= 72:
            label = _truncate(str(span["label"]), max(7, int(width_px // 8)))
            out.append(f'<text x="{x + 9:.1f}" y="{y + 17}" font-size="12" font-weight="600" fill="#ffffff">{html.escape(label)}</text>')
    footer = f"{data['controller_count']} controller agent · {data['tool_call_count']} tool calls · peak {data['peak_parallel_tools']} tools in parallel"
    out.append(f'<text x="24" y="{height - 12}" font-size="12" fill="#64748b">{footer}. Tool rows are concurrent work, not separately spawned agents.</text>')
    out.append('</svg>')
    return "\n".join(out)


def _timeline_png(data: dict[str, Any]) -> bytes:
    tool_spans = [span for span in data["spans"] if span["kind"] != "controller"]
    width, left, right, axis_y, row_height = 1600, 340, 36, 56, 38
    canvas = _PngCanvas(width, axis_y + (len(tool_spans) + 1) * row_height + 44, "#ffffff")
    duration = data["duration"]
    chart_width = width - left - right
    for row in range(len(tool_spans) + 1):
        if row % 2:
            canvas.rect(left, axis_y + row * row_height, chart_width + right, row_height, "#f7f9fc")
    for tick in _time_ticks(duration):
        x = left + int(tick / duration * chart_width)
        canvas.rect(x, axis_y - 8, 1, canvas.height - axis_y - 36, "#dde4ee")
        canvas.text(max(left, x - 12), axis_y - 28, _time_label(tick).upper(), "#8ca0be", 1)
    canvas.text(18, axis_y + 9, "MAIN LLM CALLS", "#7c3aed", 1)
    for index, span in enumerate(tool_spans, start=1):
        canvas.text(18, axis_y + index * row_height + 9, _truncate(str(span["label"]), 38).upper(), _span_color(span), 1)
    for span in data["spans"]:
        start = (float(span["start"]) - data["origin"]) / duration
        end = (float(span["end"]) - data["origin"]) / duration
        x = left + int(start * chart_width)
        bar_width = max(5, int((end - start) * chart_width))
        row = 0 if span["kind"] == "controller" else tool_spans.index(span) + 1
        y = axis_y + row * row_height + 7
        canvas.rect(x, y, bar_width, 25, _span_color(span))
        if bar_width > 72:
            canvas.text(x + 7, y + 8, _truncate(str(span["label"]), max(7, bar_width // 8)).upper(), "#ffffff", 1)
    return canvas.png()


def _tool_label(name: str, index: int) -> str:
    labels = {
        "arxiv_api_search": "Literature search",
        "semantic_scholar_api_search": "Scholar search",
        "openalex_api_search": "OpenAlex search",
        "web_search": "Web search",
        "docs_blogs_search": "Docs search",
        "fetch_document": "Document fetch",
        "inspect_document_figures": "Figure inspection",
    }
    return f"{labels.get(name, name.replace('_', ' ').title())} {index}"


def _peak_parallelism(spans: Sequence[dict[str, Any]]) -> int:
    points: list[tuple[float, int]] = []
    for span in spans:
        points.extend([(float(span["start"]), 1), (float(span["end"]), -1)])
    active = peak = 0
    for _, delta in sorted(points, key=lambda point: (point[0], point[1])):
        active += delta
        peak = max(peak, active)
    return peak


def _span_color(span: dict[str, Any]) -> str:
    if span["kind"] == "controller":
        return "#7c3aed"
    if span["kind"] == "tools_error":
        return "#ef4444"
    return "#2563eb" if "search" in str(span.get("detail") or "") else "#06b6d4"


def _time_ticks(duration: float) -> list[float]:
    if duration <= 60:
        step = 10
    elif duration <= 180:
        step = 30
    elif duration <= 600:
        step = 60
    else:
        step = 300
    ticks = list(range(0, int(duration) + step, step))
    return [float(tick) for tick in ticks]


def _time_label(value: float) -> str:
    minutes, seconds = divmod(int(value), 60)
    return f"{minutes}:{seconds:02d}" if minutes else f"{seconds}s"


def _truncate(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[: max(1, limit - 1)].rstrip() + "…"


def _trajectory_tree(store: ArtifactStore, events: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = [{"id": "research_agent", "label": "Controller agent", "kind": "controller"}]
    edges: list[dict[str, str]] = []
    parent = "research_agent"
    number = 0
    for event in events:
        if event.get("event_type") != "model_turn":
            continue
        number += 1
        node_id = f"turn_{number}"
        nodes.append({"id": node_id, "label": f"Model turn {number}", "kind": "model_turn"})
        edges.append({"from": parent, "to": node_id})
        parent = node_id
    return {"kind": "research_trajectory", "run_id": store.root.name, "nodes": nodes, "edges": edges}


def _trajectory_tree_svg(tree: dict[str, Any]) -> str:
    nodes = list(tree["nodes"])
    width = max(960, 160 + len(nodes) * 140)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="180" viewBox="0 0 {width} 180" style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif">',
        f'<rect width="{width}" height="180" fill="#ffffff"/>',
        '<text x="24" y="32" font-size="18" font-weight="700" fill="#0f172a">Research controller trajectory</text>',
        '<text x="24" y="52" font-size="11" fill="#64748b">This run created one controller agent. Nodes are observed model turns, not optimization candidates.</text>',
    ]
    positions = {str(node["id"]): 90 + index * 140 for index, node in enumerate(nodes)}
    for edge in tree["edges"]:
        x1, x2 = positions[edge["from"]], positions[edge["to"]]
        out.append(f'<path d="M{x1 + 28},106 H{x2 - 28}" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>')
    out.insert(1, '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#64748b"/></marker></defs>')
    for node in nodes:
        x = positions[str(node["id"])]
        fill = "#ede9fe" if node["kind"] == "controller" else "#eff6ff"
        stroke = "#7c3aed" if node["kind"] == "controller" else "#2563eb"
        out.append(f'<rect x="{x - 42}" y="82" width="84" height="48" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        out.append(f'<text x="{x}" y="111" text-anchor="middle" font-size="11" fill="#0f172a">{html.escape(str(node["label"]))}</text>')
    out.append('</svg>')
    return "\n".join(out)


def _trajectory_tree_png(tree: dict[str, Any]) -> bytes:
    canvas = _PngCanvas(1280, 180, "#ffffff")
    canvas.text(24, 18, "RESEARCH CONTROLLER TRAJECTORY", "#0f172a", 2)
    nodes = list(tree["nodes"])
    for index, node in enumerate(nodes[:8]):
        x = 70 + index * 150
        color = "#7c3aed" if node["kind"] == "controller" else "#2563eb"
        canvas.rect(x, 82, 108, 38, "#ede9fe" if node["kind"] == "controller" else "#eff6ff")
        canvas.outline(x, 82, 108, 38, color)
        canvas.text(x + 8, 96, str(node["label"]).upper(), "#0f172a", 1, 16)
        if index:
            canvas.line(x - 42, 101, x, 101, "#64748b")
    return canvas.png()


def _timestamp(value: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None
