"""Small, content-backed visual artifacts for model-directed research runs."""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from .run_benchmarks import _PngCanvas, _write_png_from_svg_or_fallback
from .store import ArtifactStore
from .visual_operations import operation_for


def write_research_run_visuals(store: ArtifactStore, events: Sequence[Any]) -> dict[str, Path]:
    """Write the event-backed timeline for an actual research trajectory.

    Tool calls are concurrent work performed for a single controller agent; the
    timeline makes that distinction explicit instead of misrepresenting tools as
    additional spawned agents. Candidate graphs are optimization artifacts and
    are deliberately not fabricated for ordinary research runs.
    """
    rows = build_parent_trace_projection(store, [_event_row(event) for event in events])
    timeline = _timeline_data(store, rows)
    timeline_svg = _timeline_svg(timeline)
    store.agent_timeline_svg_path.write_text(timeline_svg, encoding="utf-8")
    _write_png_from_svg_or_fallback(
        store.agent_timeline_path,
        timeline_svg,
        lambda: _timeline_png(timeline),
    )

    return {
        "agent_timeline": store.agent_timeline_path,
        "agent_timeline_svg": store.agent_timeline_svg_path,
    }


def build_parent_trace_projection(store: ArtifactStore, parent_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge isolated worker logs into one deterministic, reference-backed trace."""
    rows: list[dict[str, Any]] = []
    delegate_by_worker: dict[str, str] = {}
    for row in parent_events:
        copy = dict(row)
        copy.setdefault("run_id", store.root.name)
        if copy.get("event_type") == "tool_result" and copy.get("tool_name") == "delegate_task":
            observation = copy.get("observation") or {}
            data = observation.get("data") if isinstance(observation, dict) else {}
            if isinstance(data, dict) and data.get("worker_run_id"):
                delegate_by_worker[str(data["worker_run_id"])] = f"tool:{copy.get('tool_call_id')}"
        rows.append(copy)
    for result_path in sorted((store.root / "workers").glob("*/worker_result.json")):
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        worker_id = str(result.get("worker_run_id") or result_path.parent.name)
        parent_span = delegate_by_worker.get(worker_id)
        events_path = Path(str(result.get("events_path") or result_path.parent / "agent_events.jsonl"))
        if not events_path.exists():
            continue
        for line in events_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row["worker_run_id"] = worker_id
            row["parent_run_id"] = result.get("parent_run_id")
            row["profile"] = result.get("profile")
            row["worker_status"] = result.get("status")
            row["worker_runtime_ms"] = result.get("runtime_ms")
            row["worker_tokens"] = result.get("total_tokens")
            row["worker_cost_usd"] = result.get("cost_usd")
            row["artifact_refs"] = [result.get("artifacts_path"), result.get("events_path")]
            row["parent_span_id"] = parent_span
            rows.append(row)
    unique: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("run_id") or row.get("worker_run_id") or "parent"), int(row.get("sequence") or 0), str(row.get("event_type") or ""))
        unique[key] = row
    merged = sorted(unique.values(), key=lambda row: (str(row.get("timestamp") or ""), str(row.get("run_id") or ""), int(row.get("sequence") or 0)))
    store.parent_trace_path.write_text(json.dumps({"schema_version": "hierarchical_trace_v1", "events": merged}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return merged


def _event_row(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        return event
    return {
        "event_type": getattr(event, "event_type", ""),
        "timestamp": getattr(event, "timestamp", ""),
        "started_at": getattr(event, "started_at", None),
        "completed_at": getattr(event, "completed_at", None),
        "runtime_ms": getattr(event, "runtime_ms", None),
        "model_call_id": getattr(event, "model_call_id", None),
        "tool_call_id": getattr(event, "tool_call_id", None),
        "tool_name": getattr(event, "tool_name", None),
        "result_status": getattr(event, "result_status", None),
        "model_turn": getattr(event, "model_turn", None),
        "sequence": getattr(event, "sequence", 0),
        "run_id": getattr(event, "run_id", None),
        "parent_run_id": getattr(event, "parent_run_id", None),
        "worker_run_id": getattr(event, "worker_run_id", None),
        "span_id": getattr(event, "span_id", None),
        "parent_span_id": getattr(event, "parent_span_id", None),
    }


def _timeline_data(store: ArtifactStore, events: list[dict[str, Any]]) -> dict[str, Any]:
    controller_count = max(1, len({str(trace.get("agent_name")) for trace in store.list("agent_traces") if trace.get("agent_name")}))
    timestamped = [(row, _timestamp(row.get("timestamp"))) for row in events]
    points = [
        point
        for row, point in timestamped
        for point in (_timestamp(row.get("started_at")), point, _timestamp(row.get("completed_at")))
        if point is not None
    ]
    origin = min(points) if points else 0.0
    current_batch = 0
    starts: dict[str, list[tuple[int, float, str]]] = {}
    controller_spans: list[dict[str, Any]] = []
    tool_spans: list[dict[str, Any]] = []
    model_turn_count = 0
    ordinals: dict[str, int] = {}

    for row, point in timestamped:
        if point is None:
            continue
        kind = str(row.get("event_type") or "")
        if kind == "model_turn":
            current_batch += 1
            model_turn_count += 1
            started_at = _timestamp(row.get("started_at")) or point
            completed_at = _timestamp(row.get("completed_at")) or point
            operation = operation_for(event_type=kind)
            ordinals[operation.ordinal_scope] = ordinals.get(operation.ordinal_scope, 0) + 1
            controller_spans.append({
                "label": f"{'Worker model call' if row.get('worker_run_id') else operation.label} {ordinals[operation.ordinal_scope]}",
                "start": started_at,
                "end": max(started_at + 0.02, completed_at),
                "kind": "controller", "operation_kind": operation.kind,
                "category": operation.category, "color": operation.color, "status": str(row.get("result_status") or "completed"),
                "detail": f"Model request · {int(row.get('runtime_ms') or 0)} ms",
                "depth": 1 if row.get("worker_run_id") else 0,
                "worker_run_id": row.get("worker_run_id"), "parent_span_id": row.get("parent_span_id"),
                "span_id": row.get("span_id"),
            })
        elif kind == "tool_requested":
            tool_id = str(row.get("tool_call_id") or f"request_{len(starts)}")
            trace_tool_id = f"{row.get('run_id') or row.get('worker_run_id') or 'parent'}:{tool_id}"
            starts.setdefault(trace_tool_id, []).append((current_batch, point, str(row.get("tool_name") or "tool")))
        elif kind == "tool_result":
            tool_id = str(row.get("tool_call_id") or "")
            trace_tool_id = f"{row.get('run_id') or row.get('worker_run_id') or 'parent'}:{tool_id}"
            attempts = starts.get(trace_tool_id) or []
            if not attempts:
                continue
            batch, _, name = attempts[-1]
            start = attempts[0][1]
            status = str(row.get("result_status") or "unknown")
            operation = operation_for(tool_name=name)
            ordinals[operation.ordinal_scope] = ordinals.get(operation.ordinal_scope, 0) + 1
            retry_count = max(0, len(attempts) - 1)
            tool_spans.append({
                "label": f"{operation.label} {ordinals[operation.ordinal_scope]}",
                "start": start,
                "end": max(start + 0.02, point),
                "kind": "tools", "operation_kind": operation.kind,
                "category": operation.category, "color": operation.color, "status": status,
                "detail": f"{operation.label} · {status} · batch {batch}" + (f" · {retry_count} retr{'y' if retry_count == 1 else 'ies'}" if retry_count else ""),
                "retry_count": retry_count, "tool_call_id": tool_id,
                "tooltip_metadata": {key: row.get(key) for key in operation.tooltip_metadata if row.get(key) is not None},
                "parallel_calls": 1,
                "batch": batch,
                "depth": 1 if row.get("worker_run_id") else 0,
                "worker_run_id": row.get("worker_run_id"), "parent_span_id": row.get("parent_span_id"),
                "span_id": row.get("span_id") or f"tool:{tool_id}",
                "tooltip_metadata": {**{key: row.get(key) for key in operation.tooltip_metadata if row.get(key) is not None}, **{key: row.get(key) for key in ("profile", "worker_runtime_ms", "worker_tokens", "worker_cost_usd", "worker_status", "artifact_refs") if row.get(key) is not None}},
            })

    spans = sorted(controller_spans + tool_spans, key=lambda span: (float(span["start"]), int(span.get("depth") or 0), str(span.get("label") or "")))
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
    lane_spans = list(data["spans"])
    width, left, right, axis_y, row_height = 1600, 340, 36, 56, 38
    height = axis_y + len(lane_spans) * row_height + 44
    chart_width = width - left - right
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="timeline-title timeline-desc" style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif">',
        '<title id="timeline-title">Agent execution timeline</title>',
        f'<desc id="timeline-desc">{data["controller_count"]} controller agent, {data["model_turn_count"]} model turns, {data["tool_call_count"]} tool calls, with up to {data["peak_parallel_tools"]} tool calls running in parallel.</desc>',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
    ]
    for row in range(len(lane_spans)):
        y = axis_y + row * row_height
        if row % 2:
            out.append(f'<rect x="{left}" y="{y}" width="{chart_width + right}" height="{row_height}" fill="#f7f9fc"/>')
    for tick in _time_ticks(data["duration"]):
        x = left + tick / data["duration"] * chart_width
        out.append(f'<line x1="{x:.1f}" y1="{axis_y - 8}" x2="{x:.1f}" y2="{height - 30}" stroke="#dde4ee" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{axis_y - 18}" text-anchor="middle" font-size="13" fill="#8ca0be">{_time_label(tick)}</text>')
    out.append(f'<line x1="{left}" y1="{axis_y - 8}" x2="{width - right}" y2="{axis_y - 8}" stroke="#dde4ee"/>')
    for index, span in enumerate(lane_spans):
        y = axis_y + index * row_height + 24
        color = _span_color(span)
        prefix = "↳ " if span.get("depth") else ""
        out.append(f'<text x="{left - 14}" y="{y}" text-anchor="end" font-size="15" font-weight="600" fill="{color}">{html.escape(prefix + str(span["label"]))}</text>')
    for span in data["spans"]:
        start = (float(span["start"]) - data["origin"]) / data["duration"]
        end = (float(span["end"]) - data["origin"]) / data["duration"]
        x = left + start * chart_width
        width_px = max(5, (end - start) * chart_width)
        row = lane_spans.index(span)
        y = axis_y + row * row_height + 7
        color = _span_color(span)
        metadata = span.get("tooltip_metadata") or {}
        detail = str(span.get("detail") or span["label"]) + (" · " + json.dumps(metadata, sort_keys=True) if metadata else "")
        status = str(span.get("status") or "completed")
        opacity = "0.42" if status in {"failed", "error", "cancelled"} else "0.62" if status == "skipped" else "1"
        dash = ' stroke-dasharray="6 4"' if status == "skipped" else ""
        stroke = "#b91c1c" if status in {"failed", "error", "cancelled"} else color
        out.append(f'<rect x="{x:.1f}" y="{y}" width="{width_px:.1f}" height="25" rx="5" fill="{color}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="1.5"{dash}><title>{html.escape(detail)}</title></rect>')
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
        status = str(span.get("status") or "completed")
        if status in {"failed", "error", "cancelled"}:
            canvas.outline(x, y, bar_width, 25, "#b91c1c")
            canvas.rect(x + 2, y + 2, 4, 4, "#b91c1c")
        elif status == "skipped":
            canvas.outline(x, y, bar_width, 25, "#64748b")
            canvas.rect(x + 2, y + 2, 4, 4, "#64748b")
        if bar_width > 72:
            canvas.text(x + 7, y + 8, _truncate(str(span["label"]), max(7, bar_width // 8)).upper(), "#ffffff", 1)
    canvas.text(
        18,
        canvas.height - 14,
        f"{data['controller_count']} CONTROLLER AGENT / {data['tool_call_count']} TOOL CALLS / PEAK {data['peak_parallel_tools']} TOOLS IN PARALLEL",
        "#64748b",
        1,
    )
    return canvas.png()


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
    return str(span.get("color") or "#64748b")


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


def _timestamp(value: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None
