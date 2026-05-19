from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
import struct
import tempfile
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .store import ArtifactStore

# ── Role colour + label maps ──────────────────────────────────────────────────
_ROLE_COLORS: dict[str, str] = {
    "search_literature":         "#3b82f6",
    "hypothesis_generation":     "#8b5cf6",
    "critic_reviewer":           "#f59e0b",
    "synthesis_agent":           "#10b981",
    "harness_debugger":          "#6b7280",
    "task_router":               "#ec4899",
    "plateau_recovery_policy":   "#f97316",
    "literature_grounding_policy": "#06b6d4",
    "research_variant_agent":    "#2563eb",
    "optimize_evaluator":        "#dc2626",
    "llm_thinking":              "#7c3aed",
    "loop_controller":           "#14b8a6",
    "orchestration":             "#64748b",
    "memory":                    "#0f766e",
}
_DEFAULT_ROLE_COLOR = "#94a3b8"

# Harness bookkeeping traces — omitted from agent_timeline.png so the chart shows
# model-backed agents and evaluators (wall clock until output is ready).
_TIMELINE_CHART_EXCLUDED_ROLES: frozenset[str] = frozenset(
    {"orchestration", "loop_controller", "memory"}
)

_ROLE_SHORT: dict[str, str] = {
    "search_literature":         "Search",
    "hypothesis_generation":     "Hyp",
    "critic_reviewer":           "Critic",
    "synthesis_agent":           "Synth",
    "harness_debugger":          "Debug",
    "task_router":               "Router",
    "plateau_recovery_policy":   "Plateau",
    "literature_grounding_policy": "Ground",
    "research_variant_agent":    "Research",
    "optimize_evaluator":        "Eval",
    "llm_thinking":              "LLM",
    "loop_controller":           "Loop",
    "orchestration":             "Orch",
    "memory":                    "Memory",
}

_FONT: dict[str, tuple[str, ...]] = {
    " ": ("000","000","000","000","000","000","000"),
    "0": ("111","101","101","101","101","101","111"), "1": ("010","110","010","010","010","010","111"),
    "2": ("111","001","001","111","100","100","111"), "3": ("111","001","001","111","001","001","111"),
    "4": ("101","101","101","111","001","001","001"), "5": ("111","100","100","111","001","001","111"),
    "6": ("111","100","100","111","101","101","111"), "7": ("111","001","001","010","010","010","010"),
    "8": ("111","101","101","111","101","101","111"), "9": ("111","101","101","111","001","001","111"),
    "A": ("010","101","101","111","101","101","101"), "B": ("110","101","101","110","101","101","110"),
    "C": ("111","100","100","100","100","100","111"), "D": ("110","101","101","101","101","101","110"),
    "E": ("111","100","100","110","100","100","111"), "F": ("111","100","100","110","100","100","100"),
    "G": ("111","100","100","101","101","101","111"), "H": ("101","101","101","111","101","101","101"),
    "I": ("111","010","010","010","010","010","111"), "J": ("001","001","001","001","101","101","111"),
    "K": ("101","101","110","100","110","101","101"), "L": ("100","100","100","100","100","100","111"),
    "M": ("101","111","111","101","101","101","101"), "N": ("101","111","111","111","101","101","101"),
    "O": ("111","101","101","101","101","101","111"), "P": ("111","101","101","111","100","100","100"),
    "Q": ("111","101","101","101","111","001","001"), "R": ("110","101","101","110","101","101","101"),
    "S": ("111","100","100","111","001","001","111"), "T": ("111","010","010","010","010","010","010"),
    "U": ("101","101","101","101","101","101","111"), "V": ("101","101","101","101","101","101","010"),
    "W": ("101","101","101","101","111","111","101"), "X": ("101","101","101","010","101","101","101"),
    "Y": ("101","101","101","010","010","010","010"), "Z": ("111","001","001","010","100","100","111"),
    "-": ("000","000","000","111","000","000","000"), "_": ("000","000","000","000","000","000","111"),
    ".": ("000","000","000","000","000","110","110"), ":": ("000","110","110","000","110","110","000"),
    "/": ("001","001","001","010","100","100","100"), "?": ("111","001","001","010","010","000","010"),
    "(": ("001","010","100","100","100","010","001"), ")": ("100","010","001","001","001","010","100"),
    "+": ("000","010","010","111","010","010","000"), "$": ("010","111","100","111","001","111","010"),
    "%": ("101","001","010","010","010","100","101"), ",": ("000","000","000","000","000","010","100"),
}


class _PngCanvas:
    def __init__(self, width: int, height: int, bg: str = "#ffffff"):
        self.width = width
        self.height = height
        self.pixels = bytearray(_rgb(bg) * width * height)

    def rect(self, x: int, y: int, w: int, h: int, color: str) -> None:
        r, g, b = _rgb_tuple(color)
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + w), min(self.height, y + h)
        for yy in range(y0, y1):
            offset = (yy * self.width + x0) * 3
            self.pixels[offset : offset + (x1 - x0) * 3] = bytes([r, g, b]) * (x1 - x0)

    def outline(self, x: int, y: int, w: int, h: int, color: str) -> None:
        self.rect(x, y, w, 1, color)
        self.rect(x, y + h - 1, w, 1, color)
        self.rect(x, y, 1, h, color)
        self.rect(x + w - 1, y, 1, h, color)

    def line(self, x0: int, y0: int, x1: int, y1: int, color: str) -> None:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.rect(x0, y0, 2, 2, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def circle(self, cx: int, cy: int, r: int, fill: str, stroke: str) -> None:
        for yy in range(cy - r, cy + r + 1):
            for xx in range(cx - r, cx + r + 1):
                dist = (xx - cx) ** 2 + (yy - cy) ** 2
                if dist <= r * r:
                    self.rect(xx, yy, 1, 1, fill)
        border_outer = r * r
        border_inner = max(0, (r - 2) * (r - 2))
        for yy in range(cy - r, cy + r + 1):
            for xx in range(cx - r, cx + r + 1):
                dist = (xx - cx) ** 2 + (yy - cy) ** 2
                if border_inner <= dist <= border_outer:
                    self.rect(xx, yy, 1, 1, stroke)

    def text(self, x: int, y: int, text: str, color: str = "#0f172a", scale: int = 2, max_chars: Optional[int] = None) -> None:
        if max_chars is not None and len(text) > max_chars:
            text = text[: max(0, max_chars - 3)] + "..."
        cx = x
        for char in text:
            glyph = _FONT.get(char) or _FONT.get(char.upper()) or _FONT.get("?")
            if glyph is None:
                cx += 4 * scale
                continue
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        self.rect(cx + gx * scale, y + gy * scale, scale, scale, color)
            cx += 4 * scale

    def png(self) -> bytes:
        rows = []
        stride = self.width * 3
        for y in range(self.height):
            rows.append(b"\x00" + bytes(self.pixels[y * stride : (y + 1) * stride]))
        raw = b"".join(rows)
        return (
            b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
            + _png_chunk(b"IDAT", zlib.compress(raw, 9))
            + _png_chunk(b"IEND", b"")
        )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def _rgb(color: str) -> bytes:
    return bytes(_rgb_tuple(color))


def _rgb_tuple(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)


def _role_color(role: str) -> str:
    return _ROLE_COLORS.get(role, _DEFAULT_ROLE_COLOR)


def _role_short(role: str) -> str:
    return _ROLE_SHORT.get(role, role.replace("_", " ").title()[:8])


def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _fmt_duration(seconds: float) -> str:
    if seconds < 0:
        return "0.0s"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def _fmt_tick(ms: int) -> str:
    s = ms // 1000
    if s < 60:
        return f"{s}s"
    m, rem = divmod(s, 60)
    return f"{m}:{rem:02d}"


def _nice_tick_ms(total_ms: int, target_ticks: int = 6) -> int:
    if total_ms <= 0:
        return 1000
    approx = total_ms / target_ticks
    for step in [500, 1000, 2000, 5000, 10_000, 15_000, 30_000, 60_000, 120_000, 300_000, 600_000]:
        if approx <= step:
            return step
    return 600_000


def _shorten(text: str, max_len: int = 22) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _human_span_label(raw_name: str, role: str, counts: Counter[str]) -> str:
    counts[role or "unknown"] += 1
    index = counts[role or "unknown"]
    if role == "research_variant_agent":
        return f"Query eval {index}"
    if role == "optimize_evaluator":
        return f"Optimizer eval {index}"
    if role == "llm_thinking":
        round_match = re.search(r"round[_-](\d+)", raw_name)
        if "query" in raw_name:
            return f"LLM query proposal r{round_match.group(1)}" if round_match else f"LLM query proposal {index}"
        if "prediction_market" in raw_name:
            return f"LLM PM code proposal r{round_match.group(1)}" if round_match else f"LLM PM code proposal {index}"
        if "code" in raw_name:
            return f"LLM code proposal r{round_match.group(1)}" if round_match else f"LLM code proposal {index}"
        return f"Main LLM call {index}"
    if role == "loop_controller":
        round_match = re.search(r"round[_-](\d+)", raw_name)
        return f"Continue? r{round_match.group(1)}" if round_match else f"Loop decision {index}"
    if role == "orchestration":
        round_match = re.search(r"round[_-](\d+)", raw_name)
        if "persist" in raw_name:
            return f"Persist variants r{round_match.group(1)}" if round_match else f"Persist variants {index}"
        if "propose" in raw_name:
            return f"Propose variants r{round_match.group(1)}" if round_match else f"Propose variants {index}"
        if "rank" in raw_name:
            return f"Rank/select r{round_match.group(1)}" if round_match else f"Rank/select {index}"
        if "seed" in raw_name:
            return "Build seed context"
        return f"Orchestration {index}"
    if role == "memory":
        return "Memory / PRD"
    if role == "search_literature":
        number = _trailing_number(raw_name)
        return f"Literature search {number}" if number else "Literature search"
    if role == "hypothesis_generation":
        number = _trailing_number(raw_name)
        return f"Hypothesis agent {number}" if number else "Hypothesis agent"
    if role == "critic_reviewer":
        return "Critic review"
    if role == "synthesis_agent":
        return "Synthesis"
    if role == "harness_debugger":
        return "Harness debugger"
    if role == "task_router":
        return "Task router"
    if role == "literature_grounding_policy":
        return "Literature grounding"
    if role == "plateau_recovery_policy":
        return "Plateau recovery"
    return raw_name.replace("_", " ").replace(":", " ").title()


def _trailing_number(text: str) -> Optional[str]:
    match = re.search(r"(?:_|-)(\d+)$", text)
    return match.group(1) if match else None


def _timeline_chart_lane(role: str, human_label: str) -> Optional[str]:
    """Map a trace to a Gantt row lane for the agent-focused chart, or None to omit."""
    if role in _TIMELINE_CHART_EXCLUDED_ROLES:
        return None
    if role == "research_variant_agent":
        return human_label
    if role == "llm_thinking":
        return "Main LLM calls"
    if role == "optimize_evaluator":
        return "Optimizer evaluation"
    if role == "search_literature":
        return "Role agent: literature"
    if role == "hypothesis_generation":
        return "Role agent: hypothesis"
    if role == "critic_reviewer":
        return "Role agent: critic"
    if role == "synthesis_agent":
        return "Role agent: synthesis"
    if role == "harness_debugger":
        return "Harness debugger"
    if role == "task_router":
        return "Task router"
    if role == "literature_grounding_policy":
        return "Literature grounding"
    if role == "plateau_recovery_policy":
        return "Plateau recovery"
    return human_label


def _gantt_row_label(span: dict[str, Any]) -> str:
    """Left-axis label: consolidated lane when present, else full span label."""
    return str(span.get("row_label") or span["label"])


def _build_timeline_spans(
    summary: dict[str, Any],
    *,
    for_agent_chart: bool = False,
) -> tuple[list[dict[str, Any]], int, int]:
    """Parse agent trace summaries into Gantt spans.

    Returns (spans, num_rows, total_ms).  Each span dict has:
      label, role, status, offset_ms, runtime_ms, end_ms, token_usage, summary, row.
    Rows are assigned by unique agent_name in order of first appearance so
    parallel agents land on separate rows and the chart reads top-to-bottom.

    When for_agent_chart is True, orchestration / loop / memory traces are dropped
    and several roles share one row (e.g. all LLM calls on \"LLM\") so the PNG
    highlights agents and wall-clock time until model output is ready.
    """
    run = summary.get("run") or {}
    run_start = _parse_iso(str(run.get("started_at", "")))
    traces = summary.get("trace_summaries") or []

    spans: list[dict[str, Any]] = []
    cursor_ms = 0

    for trace in traces:
        runtime_ms = max(int(trace.get("runtime_ms") or 0), 0)
        started_dt = _parse_iso(str(trace.get("started_at") or ""))

        if started_dt and run_start:
            offset_ms = max(0, int((started_dt - run_start).total_seconds() * 1000))
        else:
            # Sequential fallback when no wall-clock start is stored.
            offset_ms = cursor_ms

        end_ms = offset_ms + runtime_ms
        cursor_ms = max(cursor_ms, end_ms)

        raw_label = str(trace.get("agent_name") or "unknown")
        spans.append({
            "label":       raw_label,
            "raw_label":   raw_label,
            "role":        str(trace.get("role") or ""),
            "status":      str(trace.get("status") or ""),
            "model":       str(trace.get("model") or ""),
            "offset_ms":   offset_ms,
            "runtime_ms":  runtime_ms,
            "end_ms":      end_ms,
            "token_usage": int(trace.get("token_usage") or 0),
            "summary":     str(trace.get("summary") or "")[:120],
            "row":         0,
        })

    spans.sort(key=lambda s: (s["offset_ms"], s["label"]))
    label_counts: Counter[str] = Counter()
    for span in spans:
        span["label"] = _human_span_label(str(span["raw_label"]), str(span["role"]), label_counts)

    if for_agent_chart:
        chart_spans: list[dict[str, Any]] = []
        for span in spans:
            lane = _timeline_chart_lane(str(span["role"]), str(span["label"]))
            if lane is None:
                continue
            span["row_label"] = lane
            chart_spans.append(span)
        spans = chart_spans

    # Dedicate one row per unique label (or consolidated row_label for agent chart).
    agent_to_row: dict[str, int] = {}
    for span in spans:
        name = str(span["row_label"]) if for_agent_chart else span["label"]
        if name not in agent_to_row:
            agent_to_row[name] = len(agent_to_row)
        span["row"] = agent_to_row[name]

    num_rows = max(len(agent_to_row), 1)

    # Total wall-clock duration.
    started = _parse_iso(str(run.get("started_at", "")))
    completed = _parse_iso(str(run.get("completed_at", "")))
    if started and completed:
        total_ms = max(int((completed - started).total_seconds() * 1000), 1)
    else:
        total_ms = max((s["end_ms"] for s in spans), default=5000)

    return spans, num_rows, total_ms


def _gantt_svg(spans: list[dict[str, Any]], num_rows: int, total_ms: int) -> str:
    if not spans or total_ms <= 0:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 80" width="100%" '
            'style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;display:block;">'
            '<rect width="960" height="80" fill="#fff"/>'
            '<text x="24" y="44" font-size="12" fill="#94a3b8">No agent timing data available.</text>'
            '</svg>'
        )

    SVG_W     = 960
    LEFT_PAD  = 220   # label column (room for "Optimizer evaluation", etc.)
    RIGHT_PAD = 16
    CHART_W   = SVG_W - LEFT_PAD - RIGHT_PAD
    BAR_H     = 18
    ROW_H     = 26
    AXIS_H    = 34
    MAX_ROWS  = 40
    BOT_PAD   = 38 if num_rows > MAX_ROWS else 28  # caption (+ optional overflow line)

    display_rows  = min(num_rows, MAX_ROWS)
    display_spans = [s for s in spans if s["row"] < display_rows]
    svg_h         = AXIS_H + display_rows * ROW_H + BOT_PAD

    p: list[str] = []
    p.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_W} {svg_h}" '
        f'width="100%" style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;display:block;">'
    )
    p.append(f'<rect width="{SVG_W}" height="{svg_h}" fill="#fff"/>')

    # Alternating row stripes.
    for row in range(display_rows):
        ry = AXIS_H + row * ROW_H
        if row % 2 == 1:
            p.append(f'<rect x="{LEFT_PAD}" y="{ry}" width="{CHART_W + RIGHT_PAD}" height="{ROW_H}" fill="#f8fafc"/>')

    # Time axis ticks.
    tick_ms = _nice_tick_ms(total_ms)
    tick = 0
    while True:
        tx = LEFT_PAD + int(tick / total_ms * CHART_W)
        if tx > SVG_W - RIGHT_PAD + 2:
            break
        label = _fmt_tick(tick)
        p.append(
            f'<line x1="{tx}" y1="{AXIS_H - 4}" x2="{tx}" y2="{AXIS_H + display_rows * ROW_H}" '
            f'stroke="#e2e8f0" stroke-width="1"/>'
        )
        p.append(f'<text x="{tx}" y="{AXIS_H - 8}" text-anchor="middle" font-size="10" fill="#94a3b8">{html.escape(label)}</text>')
        tick += tick_ms
        if tick > total_ms + tick_ms:
            break

    # Axis baseline.
    p.append(f'<line x1="{LEFT_PAD}" y1="{AXIS_H}" x2="{SVG_W - RIGHT_PAD}" y2="{AXIS_H}" stroke="#e2e8f0" stroke-width="1"/>')

    labeled_rows: set[int] = set()

    for span in display_spans:
        offset  = span["offset_ms"]
        runtime = max(span["runtime_ms"], 50)   # min 50 ms so 0-duration spans show
        row     = span["row"]
        color   = _role_color(span["role"])

        bx = LEFT_PAD + int(offset / total_ms * CHART_W)
        bw = max(int(runtime / total_ms * CHART_W), 3)
        bw = min(bw, LEFT_PAD + CHART_W - bx)   # clamp to chart area
        by = AXIS_H + row * ROW_H + (ROW_H - BAR_H) // 2

        opacity = "0.35" if span["status"] == "failed" else "1"

        tok = span["token_usage"]
        tip = "\n".join(filter(None, [
            span["label"],
            f"Trace: {span['raw_label']}" if span.get("raw_label") != span["label"] else None,
            f"Role: {span['role']}",
            f"Status: {span['status']}",
            f"Duration: {_fmt_duration(span['runtime_ms'] / 1000)}",
            f"Tokens: {tok:,}" if tok else None,
            f"Start: +{_fmt_duration(offset / 1000)}",
            f"Model: {span['model']}" if span["model"] else None,
            span["summary"][:90] if span["summary"] else None,
        ]))

        p.append(
            f'<rect x="{bx}" y="{by}" width="{bw}" height="{BAR_H}" rx="4" '
            f'fill="{color}" opacity="{opacity}">'
            f'<title>{html.escape(tip)}</title>'
            f'</rect>'
        )

        # Label inside bar when wide enough.
        if bw > 50:
            chars = max(4, bw // 6)
            short = _shorten(span["label"], chars)
            p.append(
                f'<text x="{bx + 6}" y="{by + BAR_H // 2 + 4}" '
                f'font-size="9" fill="#fff" font-weight="600" style="pointer-events:none;">'
                f'{html.escape(short)}</text>'
            )

        # Row label (left column) — first span per row only.
        if row not in labeled_rows:
            labeled_rows.add(row)
            label_y = AXIS_H + row * ROW_H + ROW_H // 2 + 4
            short = _shorten(_gantt_row_label(span), 30)
            p.append(
                f'<text x="{LEFT_PAD - 10}" y="{label_y}" text-anchor="end" '
                f'font-size="11" fill="{color}" font-weight="500">'
                f'{html.escape(short)}</text>'
            )

    caption_y = AXIS_H + display_rows * ROW_H + 14
    p.append(
        f'<text x="12" y="{caption_y}" font-size="9" fill="#64748b">'
        f'Query eval rows are individual variant evaluations; role-agent rows are spawned agents; Main LLM calls are proposal/judge model latency.</text>'
    )

    if num_rows > MAX_ROWS:
        p.append(
            f'<text x="{SVG_W // 2}" y="{caption_y + 12}" text-anchor="middle" '
            f'font-size="10" fill="#94a3b8">… {num_rows - MAX_ROWS} more rows not shown</text>'
        )

    p.append("</svg>")
    return "\n".join(p)


def _gantt_png(spans: list[dict[str, Any]], num_rows: int, total_ms: int) -> bytes:
    width = 1280
    left_pad = 272
    right_pad = 24
    chart_w = width - left_pad - right_pad
    row_h = 26
    axis_h = 34
    bottom = 38 if num_rows > 40 else 28  # caption (+ optional overflow line)
    display_rows = min(num_rows, 40)
    height = axis_h + max(display_rows, 1) * row_h + bottom
    canvas = _PngCanvas(width, height, "#ffffff")
    if not spans or total_ms <= 0:
        canvas.text(24, 44, "No agent timing data available", "#94a3b8", 2)
        return canvas.png()
    for row in range(display_rows):
        y = axis_h + row * row_h
        if row % 2:
            canvas.rect(left_pad, y, chart_w + right_pad, row_h, "#f8fafc")
    tick_ms = _nice_tick_ms(total_ms)
    tick = 0
    while tick <= total_ms + tick_ms:
        x = left_pad + int(tick / total_ms * chart_w)
        if x > width - right_pad:
            break
        canvas.rect(x, axis_h - 4, 1, height - axis_h - bottom + 4, "#e2e8f0")
        canvas.text(max(left_pad, x - 12), 14, _fmt_tick(tick), "#94a3b8", 1)
        tick += tick_ms
    canvas.rect(left_pad, axis_h, chart_w, 1, "#cbd5e1")
    labeled_rows: set[int] = set()
    for span in [s for s in spans if s["row"] < display_rows]:
        row = int(span["row"])
        x = left_pad + int(span["offset_ms"] / total_ms * chart_w)
        w = max(int(max(span["runtime_ms"], 50) / total_ms * chart_w), 3)
        w = min(w, left_pad + chart_w - x)
        y = axis_h + row * row_h + 5
        color = _role_color(str(span["role"]))
        if span.get("status") == "failed":
            color = "#fca5a5"
        canvas.rect(x, y, w, 16, color)
        if w > 56:
            canvas.text(x + 4, y + 3, _shorten(str(span["label"]), max(6, w // 9)), "#ffffff", 1)
        if row not in labeled_rows:
            labeled_rows.add(row)
            canvas.text(8, axis_h + row * row_h + 8, _shorten(_gantt_row_label(span), 32), color, 1)
    caption_y = axis_h + display_rows * row_h + 12
    canvas.text(12, caption_y, "Query eval rows are variant evaluations; role-agent rows are spawned agents; Main LLM calls are proposal/judge latency.", "#64748b", 1)
    if num_rows > display_rows:
        canvas.text(width // 2 - 100, caption_y + 12, f"... {num_rows - display_rows} more rows", "#94a3b8", 1)
    return canvas.png()


def _event_rows_html(spans: list[dict[str, Any]]) -> str:
    if not spans:
        return (
            '<tr><td colspan="6" style="color:#94a3b8;text-align:center;padding:20px;">'
            'No agent events recorded.</td></tr>'
        )
    rows: list[str] = []
    for span in sorted(spans, key=lambda s: s["offset_ms"]):
        color = _role_color(span["role"])
        badge = (
            f'<span style="display:inline-block;padding:1px 7px;border-radius:4px;'
            f'font-size:10px;font-weight:700;letter-spacing:.04em;color:#fff;'
            f'background:{color};">{html.escape(_role_short(span["role"]))}</span>'
        )
        tok   = f'{span["token_usage"]:,}' if span["token_usage"] else "—"
        dur   = _fmt_duration(span["runtime_ms"] / 1000)
        off   = f'+{_fmt_duration(span["offset_ms"] / 1000)}'
        dim   = ' style="opacity:.45;"' if span["status"] == "failed" else ""
        summ  = html.escape(span["summary"]) if span["summary"] else '<span style="color:#94a3b8">—</span>'
        name  = html.escape(_shorten(span["label"], 32))
        rows.append(
            f'<tr{dim}>'
            f'<td>{badge}</td>'
            f'<td style="font-family:\'SF Mono\',\'Fira Code\',monospace;font-size:12px;">{name}</td>'
            f'<td style="color:#64748b;max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{summ}</td>'
            f'<td style="text-align:right;font-family:monospace;color:#475569;">{html.escape(tok)}</td>'
            f'<td style="text-align:right;font-family:monospace;">{html.escape(dur)}</td>'
            f'<td style="text-align:right;font-family:monospace;color:#94a3b8;">{html.escape(off)}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def _stats_cards_html(summary: dict[str, Any]) -> str:
    counts = summary.get("counts") or {}
    best   = float((summary.get("best_evaluation") or {}).get("score") or 0.0)
    items  = [
        ("Sources",     counts.get("sources", 0)),
        ("Claims",      counts.get("claims", 0)),
        ("Hypotheses",  counts.get("hypotheses", 0)),
        ("Variants",    counts.get("variants", 0)),
        ("Evals",       counts.get("evaluations", 0)),
        ("Rounds",      counts.get("outer_rounds", 0)),
        ("Decisions",   counts.get("continuation_decisions", 0)),
        ("Tasks",       f'{counts.get("passed_tasks", 0)}/{counts.get("tasks", 0)}'),
        ("Best score",  f"{best:.3f}"),
        ("Agents",      counts.get("agent_traces", 0)),
        ("Errors",      counts.get("failed_agents", 0)),
    ]
    return "".join(
        f'<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;'
        f'padding:10px 14px;min-width:90px;">'
        f'<div style="font-size:10px;font-weight:600;letter-spacing:.08em;'
        f'text-transform:uppercase;color:#94a3b8;">{html.escape(lbl)}</div>'
        f'<div style="font-size:22px;font-weight:700;color:#1e293b;margin-top:3px;'
        f'font-family:\'SF Mono\',monospace;">{html.escape(str(val))}</div>'
        f'</div>'
        for lbl, val in items
    )


def _round_rows_html(summary: dict[str, Any]) -> str:
    rounds = summary.get("rounds") or []
    if not rounds:
        return (
            '<tr><td colspan="5" style="color:#94a3b8;text-align:center;padding:16px;">'
            'No evolution rounds recorded.</td></tr>'
        )
    rows: list[str] = []
    for r in rounds:
        score  = float(r.get("best_score") or 0.0)
        bar_w  = int(score * 80)
        bar    = (
            f'<span style="display:inline-block;height:6px;width:{bar_w}px;'
            f'border-radius:3px;background:#3b82f6;vertical-align:middle;margin-right:6px;"></span>'
        )
        signal = str(r.get("termination_signal") or "—")
        sig_color = "#10b981" if "threshold" in signal else "#f59e0b" if "plateau" in signal else "#64748b"
        rows.append(
            f'<tr>'
            f'<td style="color:#64748b;">{r.get("outer_iteration", "—")}</td>'
            f'<td>{html.escape(str(r.get("mode", "—")))}</td>'
            f'<td>{bar}<span style="font-family:monospace;">{score:.3f}</span></td>'
            f'<td><span style="color:{sig_color};font-size:11px;font-weight:600;">{html.escape(signal)}</span></td>'
            f'<td style="color:#94a3b8;">{r.get("plateau_count", 0)}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def _optimizer_trace_rows_html(summary: dict[str, Any]) -> str:
    trace = summary.get("optimizer_trace") or []
    if not trace:
        return (
            '<tr><td colspan="9" style="color:#94a3b8;text-align:center;padding:16px;">'
            'No optimizer trace recorded.</td></tr>'
        )
    rows: list[str] = []
    for item in trace:
        score = float(item.get("best_score") or 0.0)
        edge = item.get("best_mean_edge")
        learned = bool(item.get("uses_prior_parent") or item.get("uses_score_feedback"))
        parameter_only = bool(item.get("all_variants_parameter_nudges"))
        status_color = "#16a34a" if learned and not parameter_only else "#f97316" if parameter_only else "#0284c7"
        status = item.get("learning_status", "unknown")
        rows.append(
            "<tr>"
            f'<td style="color:#64748b;">{html.escape(str(item.get("round", "—")))}</td>'
            f'<td><span style="color:{status_color};font-size:11px;font-weight:700;">{html.escape(str(status))}</span></td>'
            f'<td style="font-family:monospace;">{score:.3f}</td>'
            f'<td style="font-family:monospace;">{html.escape(str(edge if edge is not None else "—"))}<br><span style="color:#94a3b8;">spread {float(item.get("round_score_spread") or 0.0):.3f}</span></td>'
            f'<td>{html.escape(str(item.get("best_score_source") or "—"))}</td>'
            f'<td>{item.get("variant_count", 0)} / {item.get("evaluation_count", 0)}</td>'
            f'<td>{"yes" if item.get("uses_prior_parent") else "no"} / {"yes" if item.get("uses_score_feedback") else "no"}</td>'
            f'<td>{"yes" if parameter_only else "no"}</td>'
            f'<td style="color:#64748b;max-width:380px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{html.escape(str(item.get("best_payload_preview") or ""))}</td>'
            "</tr>"
        )
        for variant in item.get("variants", [])[:6]:
            change = str(variant.get("change_type") or "unknown")
            rows.append(
                '<tr style="background:#fbfdff;">'
                '<td></td>'
                f'<td colspan="2" style="font-family:\'SF Mono\',\'Fira Code\',monospace;font-size:11px;color:#475569;">{html.escape(str(variant.get("variant_id") or ""))}</td>'
                f'<td style="font-family:monospace;">{html.escape(str(variant.get("mean_edge") if variant.get("mean_edge") is not None else "—"))}</td>'
                f'<td>{html.escape(str(variant.get("score_source") or "—"))}</td>'
                f'<td>{html.escape(change)}</td>'
                f'<td>{"yes" if variant.get("uses_prior_parent") else "no"} / {"yes" if variant.get("uses_score_feedback") else "no"}</td>'
                f'<td>{html.escape(str(variant.get("meaningful_entropy_action") or "—"))}</td>'
                f'<td style="color:#64748b;max-width:380px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{html.escape(str(variant.get("change_summary") or variant.get("payload_preview") or ""))}</td>'
                '</tr>'
            )
    return "\n".join(rows)


def write_run_benchmarks(store: ArtifactStore) -> None:
    if not store.harness_diagnosis_path.exists():
        store.write_harness_diagnosis()
    summary = build_run_summary(store)
    (store.root / "run_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dag = decision_dag_mermaid(summary)
    spans, num_rows, total_ms = _build_timeline_spans(summary, for_agent_chart=True)
    dag_svg = decision_dag_svg(summary)
    timeline_svg = _gantt_svg(spans, num_rows, total_ms)
    optimizer_trace = summary.get("optimizer_trace") or []
    optimizer_flow = optimizer_flow_mermaid(summary)
    optimizer_flow_svg_text = optimizer_flow_svg(summary)
    champion_tree = read_json(store.champion_tree_path, {})
    champion_tree_graph = champion_tree_mermaid(champion_tree)
    champion_tree_svg_text = champion_tree_svg(champion_tree)
    (store.root / "decision_dag.mmd").write_text(dag, encoding="utf-8")
    (store.root / "optimizer_trace.json").write_text(json.dumps(optimizer_trace, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (store.root / "optimizer_flow.mmd").write_text(optimizer_flow, encoding="utf-8")
    (store.root / "optimizer_flow.svg").write_text(optimizer_flow_svg_text, encoding="utf-8")
    store.champion_tree_mermaid_path.write_text(champion_tree_graph, encoding="utf-8")
    store.champion_tree_svg_path.write_text(champion_tree_svg_text, encoding="utf-8")
    _write_png_from_svg_or_fallback(store.decision_dag_path, dag_svg, lambda: decision_dag_png(summary))
    _write_png_from_svg_or_fallback(store.agent_timeline_path, timeline_svg, lambda: _gantt_png(spans, num_rows, total_ms))
    _write_png_from_svg_or_fallback(store.root / "optimizer_flow.png", optimizer_flow_svg_text, lambda: optimizer_flow_png(summary))
    _write_png_from_svg_or_fallback(store.champion_tree_graph_path, champion_tree_svg_text, lambda: champion_tree_png(champion_tree))
    (store.root / "run_benchmark.md").write_text(run_benchmark_markdown(summary, dag, optimizer_flow), encoding="utf-8")
    (store.root / "run_benchmark.html").write_text(run_benchmark_html(summary), encoding="utf-8")
    store.run_notebook_path.write_text(json.dumps(run_notebook_export(summary), indent=2) + "\n", encoding="utf-8")


def build_run_summary(store: ArtifactStore) -> dict[str, Any]:
    runs = store.list("runs")
    run = runs[0] if runs else {}
    traces = store.list("agent_traces")
    tasks = store.list("loop_tasks")
    decisions = store.list("task_ingestion_decisions")
    continuation_decisions = store.list("loop_continuation_decisions")
    variants = store.list("variants")
    evaluations = store.list("variant_evaluations")
    rounds = store.list("evolution_rounds")
    prd = read_json(store.prd_path, {})
    optimizer_seed_context = read_json(store.optimizer_seed_context_path, {})
    optimization_result = read_json(store.optimization_result_path, {})
    optimized_candidate_exists = store.optimized_candidate_path.exists()
    optimal_code_exists = store.optimal_code_path.exists()
    solution_exists = store.solution_path.exists()
    sources = store.list("sources")
    claims = store.list("claims")
    hypotheses = store.list("hypotheses")
    contradictions = store.list("contradictions")
    provenance_edges = store.list("provenance_edges")
    cost_events = store.list("cost_events")
    harness_diagnosis = read_json(store.harness_diagnosis_path, {})
    cost = read_json(store.cost_path, {})
    models = Counter(str(trace.get("model", "unknown")) for trace in traces)
    best_eval = max(evaluations, key=lambda row: float(row.get("score", 0.0)), default={})
    optimizer_trace = build_optimizer_trace(rounds, variants, evaluations)
    return {
        "run": run,
        "counts": {
            "tasks": len(tasks),
            "passed_tasks": sum(1 for task in tasks if task.get("passes")),
            "outer_rounds": len(rounds),
            "continuation_decisions": len(continuation_decisions),
            "variants": len(variants),
            "evaluations": len(evaluations),
            "sources": len(sources),
            "claims": len(claims),
            "hypotheses": len(hypotheses),
            "contradictions": len(contradictions),
            "provenance_edges": len(provenance_edges),
            "cost_events": len(cost_events),
            "agent_traces": len(traces),
            "failed_agents": sum(1 for trace in traces if trace.get("status") != "completed"),
        },
        "task_ingestion": decisions[0] if decisions else None,
        "prd": prd,
        "optimizer_seed_context": optimizer_seed_context,
        "optimization_result": optimization_result,
        "harness_diagnosis": harness_diagnosis,
        "cost": cost,
        "optimized_candidate": str(store.optimized_candidate_path) if optimized_candidate_exists else None,
        "optimal_code": str(store.optimal_code_path) if optimal_code_exists else None,
        "solution": str(store.solution_path) if solution_exists else None,
        "models": dict(models),
        "tasks": tasks,
        "rounds": rounds,
        "continuation_decisions": continuation_decisions,
        "variants": variants,
        "evaluations": evaluations,
        "best_evaluation": best_eval,
        "optimizer_trace": optimizer_trace,
        "trace_summaries": [
            {
                "agent_name":  trace.get("agent_name"),
                "role":        trace.get("role"),
                "model":       trace.get("model"),
                "status":      trace.get("status"),
                "runtime_ms":  trace.get("runtime_ms"),
                "token_usage": trace.get("token_usage"),
                "started_at":  trace.get("started_at", ""),
                "summary":     trace.get("output_summary"),
            }
            for trace in traces
        ],
    }


def build_optimizer_trace(rounds: list[dict[str, Any]], variants: list[dict[str, Any]], evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variant_by_id = {str(row.get("id")): row for row in variants}
    evals_by_variant: dict[str, list[dict[str, Any]]] = {}
    for evaluation in evaluations:
        evals_by_variant.setdefault(str(evaluation.get("variant_id")), []).append(evaluation)
    trace: list[dict[str, Any]] = []
    previous_parent_ids: set[str] = set()
    previous_best_payload = ""
    previous_best_score: Optional[float] = None
    optimizer_rounds = [row for row in rounds if row.get("mode") in {"optimize", "optimize_query"}]
    for round_record in optimizer_rounds:
        variant_ids = [str(variant_id) for variant_id in round_record.get("variant_ids", [])]
        variant_rows = [variant_by_id[variant_id] for variant_id in variant_ids if variant_id in variant_by_id]
        best_variant_id = str(round_record.get("best_variant_id") or "")
        round_evals = [
            evaluation
            for variant_id in variant_ids
            for evaluation in evals_by_variant.get(variant_id, [])
        ]
        round_evals.sort(key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)
        round_scores = [float(evaluation.get("score", 0.0) or 0.0) for evaluation in round_evals]
        score_spread = round(max(round_scores) - min(round_scores), 6) if round_scores else 0.0
        score_mean = sum(round_scores) / len(round_scores) if round_scores else 0.0
        score_stddev = round((sum((score - score_mean) ** 2 for score in round_scores) / len(round_scores)) ** 0.5, 6) if round_scores else 0.0
        best_eval = next((evaluation for evaluation in round_evals if str(evaluation.get("variant_id")) == best_variant_id), round_evals[0] if round_evals else {})
        best_variant = variant_by_id.get(best_variant_id, {})
        variant_details = [
            _optimizer_variant_detail(variant, evals_by_variant.get(str(variant.get("id")), []), previous_parent_ids, previous_best_payload)
            for variant in variant_rows
        ]
        parent_links = sum(len(detail["parent_ids"]) for detail in variant_details)
        prior_parent_reuse = any(detail["uses_prior_parent"] for detail in variant_details)
        score_feedback_reuse = any(detail["uses_score_feedback"] for detail in variant_details)
        meaningful_entropy = any(detail["change_type"] == "meaningful_entropy" for detail in variant_details)
        entropy_label_only = any(detail["change_type"] == "entropy_label_on_numeric_mutation" for detail in variant_details)
        parameter_like_types = {"parameter_nudge", "parameter_exploration", "context_derived_numeric_mutation", "entropy_label_on_numeric_mutation"}
        parameter_only = bool(variant_details) and all(detail["change_type"] in parameter_like_types for detail in variant_details)
        learn_status = "learned_from_prior_score_or_parent" if prior_parent_reuse or score_feedback_reuse else "fresh_or_unlinked_batch"
        if parameter_only and not meaningful_entropy:
            learn_status = f"{learn_status}_but_parameter_only"
        if entropy_label_only:
            learn_status = f"{learn_status}_entropy_label_only"
        score = float(round_record.get("best_score", 0.0) or 0.0)
        score_delta = None if previous_best_score is None else round(score - previous_best_score, 6)
        metrics = best_eval.get("metrics") if isinstance(best_eval.get("metrics"), dict) else {}
        trace.append(
            {
                "round": round_record.get("outer_iteration"),
                "mode": round_record.get("mode"),
                "termination_signal": round_record.get("termination_signal"),
                "plateau_count": round_record.get("plateau_count", 0),
                "variant_count": len(variant_details),
                "evaluation_count": len(round_evals),
                "best_variant_id": best_variant_id or None,
                "best_score": score,
                "best_score_delta_vs_previous_round": score_delta,
                "round_score_spread": score_spread,
                "round_score_stddev": score_stddev,
                "best_mean_edge": metrics.get("mean_edge"),
                "best_score_source": metrics.get("score_source"),
                "best_summary": best_eval.get("summary", ""),
                "best_payload_preview": _shorten(str(best_variant.get("payload", "")), 420),
                "parent_links": parent_links,
                "uses_prior_parent": prior_parent_reuse,
                "uses_score_feedback": score_feedback_reuse,
                "meaningful_entropy": meaningful_entropy,
                "entropy_label_only": entropy_label_only,
                "all_variants_parameter_nudges": parameter_only,
                "learning_status": learn_status,
                "round_change_summary": _optimizer_round_change_summary(variant_details),
                "variants": variant_details,
            }
        )
        previous_parent_ids = {
            str(evaluation.get("variant_id"))
            for evaluation in round_evals[:2]
            if evaluation.get("variant_id")
        }
        previous_best_payload = str(best_variant.get("payload", ""))
        previous_best_score = score
    return trace


def _optimizer_variant_detail(
    variant: dict[str, Any],
    evaluations: list[dict[str, Any]],
    previous_parent_ids: set[str],
    previous_best_payload: str,
) -> dict[str, Any]:
    payload = str(variant.get("payload", ""))
    metadata = variant.get("metadata") if isinstance(variant.get("metadata"), dict) else {}
    parent_ids = [str(parent_id) for parent_id in variant.get("parent_ids", [])]
    best_eval = max(evaluations, key=lambda row: float(row.get("score", 0.0) or 0.0), default={})
    metrics = best_eval.get("metrics") if isinstance(best_eval.get("metrics"), dict) else {}
    entropy = metadata.get("meaningful_entropy_intent") if isinstance(metadata.get("meaningful_entropy_intent"), dict) else {}
    uses_prior_parent = bool(previous_parent_ids.intersection(parent_ids))
    uses_score_feedback = bool(re.search(r"prior_best|score_memory|score_feedback|mean_edge|best_edge", payload, re.I))
    if previous_best_payload:
        uses_score_feedback = uses_score_feedback or _payload_mentions_parent(payload, previous_best_payload)
    return {
        "variant_id": variant.get("id"),
        "kind": variant.get("kind"),
        "score": best_eval.get("score"),
        "mean_edge": metrics.get("mean_edge"),
        "score_source": metrics.get("score_source"),
        "passed": best_eval.get("passed"),
        "summary": best_eval.get("summary", ""),
        "parent_ids": parent_ids,
        "uses_prior_parent": uses_prior_parent,
        "uses_score_feedback": uses_score_feedback,
        "change_type": _optimizer_change_type(payload, metadata, bool(entropy)),
        "meaningful_entropy_action": entropy.get("action") if entropy else None,
        "proposal_source": metadata.get("proposal_source"),
        "recovery": metadata.get("recovery"),
        "rendered_code_hash": metadata.get("rendered_code_hash"),
        "change_summary": _optimizer_variant_change_summary(metadata, payload),
        "payload_preview": _shorten(payload, 360),
    }


def _optimizer_variant_change_summary(metadata: dict[str, Any], payload: str) -> str:
    parts: list[str] = []
    family = metadata.get("strategy_family")
    mechanism = metadata.get("mechanism_hypothesis")
    role = metadata.get("entropy_role")
    recovery = metadata.get("recovery")
    source = metadata.get("proposal_source")
    if family:
        parts.append(f"family={family}")
    if mechanism:
        parts.append(f"mechanism={_shorten(str(mechanism), 90)}")
    if role:
        parts.append(f"role={role}")
    if recovery:
        parts.append(f"recovery={recovery}")
    if source:
        parts.append(f"source={source}")
    if not parts:
        parts.append(_shorten(payload, 120))
    return "; ".join(parts)


def _optimizer_round_change_summary(variant_details: list[dict[str, Any]], limit: int = 3) -> list[str]:
    summaries: list[str] = []
    seen: set[str] = set()
    for detail in variant_details:
        summary = str(detail.get("change_summary") or detail.get("change_type") or "").strip()
        if not summary or summary in seen:
            continue
        seen.add(summary)
        summaries.append(summary)
        if len(summaries) >= limit:
            break
    return summaries


def _optimizer_change_type(payload: str, metadata: dict[str, Any], has_entropy: bool) -> str:
    lowered = payload.lower()
    recovery = str(metadata.get("recovery") or "")
    if has_entropy and (recovery == "context_derived_numeric_mutation" or "contextual_parent_mutation" in lowered):
        return "entropy_label_on_numeric_mutation"
    if has_entropy:
        return "meaningful_entropy"
    if recovery:
        return recovery
    if "class strategy" in lowered:
        return "executable_strategy_code"
    if "contextual_parent_mutation" in lowered or "contextual_score_memory" in lowered:
        return "parameter_nudge"
    if "contextual_score_explore" in lowered:
        return "parameter_exploration"
    if "fresh_literature=" in lowered or "query_seed=" in lowered or "literature_inspiration=" in lowered:
        return "fresh_context_or_literature"
    if "new strategy" in lowered or "mechanism" in lowered or "alternative evaluator" in lowered:
        return "strategy_shift"
    return "unknown"


def _payload_mentions_parent(payload: str, parent_payload: str) -> bool:
    parent_terms = [term for term in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{5,}", parent_payload.lower()) if term not in {"contextual", "strategy", "parent"}]
    if not parent_terms:
        return False
    payload_lower = payload.lower()
    return sum(1 for term in set(parent_terms[:24]) if term in payload_lower) >= 3


def run_notebook_export(summary: dict[str, Any]) -> dict[str, Any]:
    run = summary.get("run") or {}
    counts = summary.get("counts") or {}
    diagnosis = summary.get("harness_diagnosis") or {}
    cost = summary.get("cost") or {}
    cells = [
        _markdown_cell(
            "# Research Harness Run\n\n"
            f"- Run: `{run.get('id', 'unknown')}`\n"
            f"- Status: `{run.get('status', 'unknown')}`\n"
            f"- Goal: {run.get('user_goal', '')}\n"
        ),
        _markdown_cell(
            "## Artifact Counts\n\n"
            + "\n".join(f"- {key}: {value}" for key, value in sorted(counts.items()))
        ),
        _markdown_cell(
            "## Observability\n\n"
            f"- Total cost: `${float(cost.get('cost_usd') or 0.0):.4f}`\n"
            f"- Total tokens: `{cost.get('total_tokens', run.get('total_tokens', 0))}`\n"
            f"- Model calls: `{cost.get('model_call_count', 0)}`\n"
        ),
        _code_cell("harness_diagnosis = " + json.dumps(diagnosis, indent=2, sort_keys=True)),
        _code_cell("optimizer_trace = " + json.dumps(summary.get("optimizer_trace", []), indent=2, sort_keys=True)),
        _code_cell("trace_summaries = " + json.dumps(summary.get("trace_summaries", []), indent=2, sort_keys=True)),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _markdown_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _code_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def decision_dag_mermaid(summary: dict[str, Any]) -> str:
    run = summary.get("run", {})
    decision = summary.get("task_ingestion") or {}
    lines = [
        "flowchart TD",
        f'  prompt["Prompt: {_mermaid(str(run.get("user_goal", "")))}"]',
        f'  route["Task router\\nProduct: {decision.get("product_agent", run.get("product_agent", "unknown"))}\\nMode: {decision.get("selected_mode", run.get("task_mode", "unknown"))}"]',
        '  memory["LeadResearcher memory\\nPRD + plan + objective + seed context"]',
        '  propose["Propose specialized subagent tasks / variants"]',
        '  subagents["Parallel subagents\\nresearch / optimizer evaluators"]',
        '  rank["Evaluate, rank, select parents"]',
        '  continue{"More research or optimization needed?"}',
        '  recover["Refine strategy / spawn next subagents"]',
        '  synth["Critic + synthesis"]',
        '  cite["Citation / grounding pass"]',
        '  persist["Persist report, traces, PNGs, costs"]',
        "  prompt --> route --> memory --> propose --> subagents --> rank --> continue",
        "  continue -- continue --> recover --> propose",
        "  continue -- exit --> synth --> cite --> persist",
    ]
    for index, round_record in enumerate(summary.get("rounds", []), start=1):
        node = f"round{index}"
        label = (
            f"Round {round_record.get('outer_iteration')}: "
            f"best={float(round_record.get('best_score', 0.0)):.3f}, "
            f"{round_record.get('termination_signal', 'continue')}"
        )
        lines.append(f'  {node}["{_mermaid(label)}"]')
        lines.append(f"  rank --> {node} --> continue")
    for index, item in enumerate(summary.get("continuation_decisions", []), start=1):
        node = f"decision{index}"
        label = f"Decision {item.get('iteration')}: {item.get('decision')} ({item.get('termination_signal')})"
        lines.append(f'  {node}["{_mermaid(label)}"]')
        lines.append(f"  continue --> {node}")
    return "\n".join(lines) + "\n"


def optimizer_flow_mermaid(summary: dict[str, Any]) -> str:
    trace = summary.get("optimizer_trace") or []
    lines = ["flowchart LR"]
    if not trace:
        return "flowchart LR\n  none[\"No optimizer rounds recorded\"]\n"
    lines.append('  seed["Seed context / prior findings"]')
    previous = "seed"
    for index, round_trace in enumerate(trace, start=1):
        round_node = f"round{index}"
        best_node = f"best{index}"
        feedback_node = f"feedback{index}"
        change = "parameter-only" if round_trace.get("all_variants_parameter_nudges") else "mixed/structural"
        learned = "uses prior" if round_trace.get("uses_prior_parent") or round_trace.get("uses_score_feedback") else "unlinked"
        label = (
            f"Round {round_trace.get('round')}\\n"
            f"{round_trace.get('variant_count', 0)} variants, {learned}\\n"
            f"{change}\\n"
            f"spread {float(round_trace.get('round_score_spread') or 0.0):.3f}, stddev {float(round_trace.get('round_score_stddev') or 0.0):.3f}"
        )
        best_label = (
            f"Best {float(round_trace.get('best_score') or 0.0):.3f}\\n"
            f"edge {round_trace.get('best_mean_edge', 'n/a')}\\n"
            f"{round_trace.get('termination_signal', 'continue')}"
        )
        feedback_label = (
            "Parent + score feedback"
            if round_trace.get("uses_prior_parent") or round_trace.get("uses_score_feedback")
            else "No parent/score link detected"
        )
        change_summaries = round_trace.get("round_change_summary") if isinstance(round_trace.get("round_change_summary"), list) else []
        if change_summaries:
            feedback_label = f"{feedback_label}\\n" + "\\n".join(str(item)[:90] for item in change_summaries[:2])
        lines.append(f'  {round_node}["{_mermaid(label)}"]')
        lines.append(f'  {best_node}["{_mermaid(best_label)}"]')
        lines.append(f'  {feedback_node}["{_mermaid(feedback_label)}"]')
        lines.append(f"  {previous} --> {round_node} --> {best_node} --> {feedback_node}")
        previous = feedback_node
    return "\n".join(lines) + "\n"


def optimizer_flow_svg(summary: dict[str, Any]) -> str:
    trace = summary.get("optimizer_trace") or []
    if not trace:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="120" viewBox="0 0 960 120" '
            'style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;">'
            '<rect width="100%" height="100%" fill="#fff"/><text x="24" y="64" font-size="14" fill="#94a3b8">'
            'No optimizer rounds recorded.</text></svg>'
        )
    card_w = 250
    card_h = 210
    gap_x = 26
    gap_y = 34
    left = 28
    cards_per_row = 4
    row_count = (len(trace) + cards_per_row - 1) // cards_per_row
    width = max(960, left * 2 + cards_per_row * card_w + (cards_per_row - 1) * gap_x)
    height = 86 + row_count * card_h + max(0, row_count - 1) * gap_y + 28
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;">',
        '<defs><marker id="flowArrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 Z" fill="#64748b"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="32" font-size="18" font-weight="700" fill="#0f172a">Optimizer learning flow</text>',
        '<text x="24" y="52" font-size="11" fill="#64748b">Shows whether each round used prior parents/scores and whether changes were structural or mostly parameter nudges.</text>',
    ]
    for index, round_trace in enumerate(trace):
        row = index // cards_per_row
        col = index % cards_per_row
        x = left + col * (card_w + gap_x)
        y = 78 + row * (card_h + gap_y)
        score = float(round_trace.get("best_score") or 0.0)
        edge = round_trace.get("best_mean_edge")
        learned = bool(round_trace.get("uses_prior_parent") or round_trace.get("uses_score_feedback"))
        parameter_only = bool(round_trace.get("all_variants_parameter_nudges"))
        entropy = bool(round_trace.get("meaningful_entropy"))
        entropy_label_only = bool(round_trace.get("entropy_label_only"))
        fill = "#dcfce7" if learned and not parameter_only else "#ffedd5" if parameter_only else "#e0f2fe"
        border = "#16a34a" if learned and not parameter_only else "#f97316" if parameter_only else "#0284c7"
        status = "learned from prior" if learned else "no prior link detected"
        change = "meaningful entropy" if entropy else "entropy label only" if entropy_label_only else "parameter nudge" if parameter_only else "mixed / structural"
        parts.append(f'<rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="10" fill="{fill}" stroke="{border}" stroke-width="1.5"/>')
        parts.append(f'<text x="{x + 12}" y="{y + 24}" font-size="13" font-weight="700" fill="#0f172a">Round {html.escape(str(round_trace.get("round")))}</text>')
        parts.append(f'<text x="{x + 12}" y="{y + 47}" font-size="11" fill="#334155">best score: {score:.3f}</text>')
        parts.append(f'<text x="{x + 12}" y="{y + 66}" font-size="11" fill="#334155">mean edge: {html.escape(str(edge if edge is not None else "n/a"))}</text>')
        parts.append(f'<text x="{x + 12}" y="{y + 85}" font-size="11" fill="#334155">variants: {round_trace.get("variant_count", 0)} / evals: {round_trace.get("evaluation_count", 0)}</text>')
        parts.append(f'<text x="{x + 12}" y="{y + 104}" font-size="11" fill="#334155">score spread: {float(round_trace.get("round_score_spread") or 0.0):.3f}</text>')
        parts.append(f'<text x="{x + 12}" y="{y + 123}" font-size="11" fill="#334155">score stddev: {float(round_trace.get("round_score_stddev") or 0.0):.3f}</text>')
        parts.append(f'<text x="{x + 12}" y="{y + 142}" font-size="11" fill="#334155">feedback: {html.escape(status)}</text>')
        parts.append(f'<text x="{x + 12}" y="{y + 161}" font-size="11" fill="#334155">change: {html.escape(change)}</text>')
        for change_index, change_item in enumerate((round_trace.get("round_change_summary") if isinstance(round_trace.get("round_change_summary"), list) else [])[:2]):
            parts.append(f'<text x="{x + 12}" y="{y + 180 + change_index * 15}" font-size="10" fill="#475569">{html.escape(_shorten(str(change_item), 42))}</text>')
        parts.append(f'<text x="{x + 12}" y="{y + 203}" font-size="10" fill="#64748b">{html.escape(str(round_trace.get("termination_signal", ""))[:34])}</text>')
        if index < len(trace) - 1:
            next_row = (index + 1) // cards_per_row
            next_col = (index + 1) % cards_per_row
            if next_row == row:
                x1 = x + card_w
                x2 = x + card_w + gap_x
                y_mid = y + card_h // 2
                parts.append(f'<path d="M{x1},{y_mid} L{x2},{y_mid}" fill="none" stroke="#64748b" stroke-width="1.6" marker-end="url(#flowArrow)"/>')
                delta_x = x1 + 5
                delta_y = y_mid - 11
            else:
                x1 = x + card_w // 2
                y1 = y + card_h
                x2 = left + next_col * (card_w + gap_x) + card_w // 2
                y2 = 78 + next_row * (card_h + gap_y)
                mid_y = y1 + gap_y // 2
                parts.append(f'<path d="M{x1},{y1} L{x1},{mid_y} L{x2},{mid_y} L{x2},{y2}" fill="none" stroke="#64748b" stroke-width="1.6" marker-end="url(#flowArrow)"/>')
                delta_x = min(x1, x2) + 8
                delta_y = mid_y - 4
            delta = trace[index + 1].get("best_score_delta_vs_previous_round")
            if delta is not None:
                parts.append(f'<text x="{delta_x}" y="{delta_y}" font-size="10" fill="#64748b">score delta {float(delta):+.3f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def optimizer_flow_png(summary: dict[str, Any]) -> bytes:
    trace = summary.get("optimizer_trace") or []
    cards_per_row = 4
    card_w = 250
    card_h = 170
    gap_x = 24
    gap_y = 28
    row_count = max(1, (len(trace) + cards_per_row - 1) // cards_per_row)
    width = 40 + cards_per_row * card_w + (cards_per_row - 1) * gap_x
    height = 76 + row_count * card_h + (row_count - 1) * gap_y + 24
    canvas = _PngCanvas(width, height, "#ffffff")
    canvas.text(24, 18, "Optimizer learning flow", "#0f172a", 2)
    if not trace:
        canvas.text(24, 70, "No optimizer rounds recorded", "#94a3b8", 2)
        return canvas.png()
    for index, round_trace in enumerate(trace):
        row = index // cards_per_row
        col = index % cards_per_row
        x = 24 + col * (card_w + gap_x)
        y = 72 + row * (card_h + gap_y)
        parameter_only = bool(round_trace.get("all_variants_parameter_nudges"))
        learned = bool(round_trace.get("uses_prior_parent") or round_trace.get("uses_score_feedback"))
        entropy_label_only = bool(round_trace.get("entropy_label_only"))
        fill = "#dcfce7" if learned and not parameter_only else "#ffedd5" if parameter_only else "#e0f2fe"
        border = "#16a34a" if learned and not parameter_only else "#f97316" if parameter_only else "#0284c7"
        canvas.rect(x, y, card_w - 12, card_h, fill)
        canvas.outline(x, y, card_w - 12, card_h, border)
        canvas.text(x + 10, y + 12, f"Round {round_trace.get('round')}", "#0f172a", 1)
        canvas.text(x + 10, y + 34, f"score {float(round_trace.get('best_score') or 0.0):.3f}", "#334155", 1)
        canvas.text(x + 10, y + 54, f"edge {round_trace.get('best_mean_edge', 'n/a')}", "#334155", 1)
        canvas.text(x + 10, y + 74, f"spread {float(round_trace.get('round_score_spread') or 0.0):.3f}", "#334155", 1)
        canvas.text(x + 10, y + 94, "prior yes" if learned else "prior no", "#334155", 1)
        canvas.text(x + 10, y + 114, "entropy-label" if entropy_label_only else "param-only" if parameter_only else "struct/mixed", "#334155", 1)
        summaries = round_trace.get("round_change_summary") if isinstance(round_trace.get("round_change_summary"), list) else []
        if summaries:
            canvas.text(x + 10, y + 134, _shorten(str(summaries[0]), 32), "#475569", 1)
    return canvas.png()


def champion_tree_mermaid(tree: dict[str, Any]) -> str:
    nodes = tree.get("nodes") if isinstance(tree.get("nodes"), list) else []
    edges = tree.get("edges") if isinstance(tree.get("edges"), list) else []
    if not nodes:
        return 'flowchart TD\n  none["No champion tree recorded"]\n'
    lines = ["flowchart TD"]
    for index, node in enumerate(nodes):
        node_id = _mermaid_node_id(str(node.get("id") or f"node_{index}"))
        label = (
            f"{str(node.get('id') or '')[:10]}\\n"
            f"score {float(node.get('score') or 0.0):.3f}\\n"
            f"{node.get('highlight', 'candidate')}"
        )
        lines.append(f'  {node_id}["{_mermaid(label)}"]')
        if node.get("is_global_champion"):
            lines.append(f"  class {node_id} champion")
        elif node.get("is_round_winner"):
            lines.append(f"  class {node_id} winner")
    node_ids = {_mermaid_node_id(str(node.get("id") or "")) for node in nodes}
    for edge in edges:
        from_id = _mermaid_node_id(str(edge.get("from") or ""))
        to_id = _mermaid_node_id(str(edge.get("to") or ""))
        if from_id in node_ids and to_id in node_ids:
            lines.append(f"  {from_id} --> {to_id}")
    lines.append("  classDef champion fill:#dcfce7,stroke:#16a34a,stroke-width:3px,color:#0f172a;")
    lines.append("  classDef winner fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0f172a;")
    return "\n".join(lines) + "\n"


def champion_tree_svg(tree: dict[str, Any]) -> str:
    nodes = tree.get("nodes") if isinstance(tree.get("nodes"), list) else []
    if not nodes:
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="120" viewBox="0 0 960 120" '
            'style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;">'
            '<rect width="100%" height="100%" fill="#fff"/><text x="24" y="64" font-size="14" fill="#94a3b8">'
            'No champion tree recorded.</text></svg>'
        )
    layout = _champion_tree_layout(tree, max_nodes=80)
    shown = layout["nodes"]
    positions = layout["positions"]
    shown_ids = {str(node.get("id")) for node in shown}
    edges = [edge for edge in layout["edges"] if str(edge.get("from")) in shown_ids and str(edge.get("to")) in shown_ids]
    width = int(layout["width"])
    height = int(layout["height"])
    radius = int(layout["radius"])
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;">',
        '<defs><marker id="treeArrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 Z" fill="#334155"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="32" font-size="18" font-weight="700" fill="#0f172a">Champion tree</text>',
        '<text x="24" y="52" font-size="11" fill="#64748b">Actual parent-to-child lineage. Green is the current global champion; blue rings are round winners.</text>',
    ]
    for edge in edges:
        x1, y1 = positions[str(edge.get("from"))]
        x2, y2 = positions[str(edge.get("to"))]
        parts.append(
            f'<path d="M{x1},{y1 + radius} C{x1},{(y1 + y2) // 2} {x2},{(y1 + y2) // 2} {x2},{y2 - radius}" '
            'fill="none" stroke="#334155" stroke-width="1.8" marker-end="url(#treeArrow)"/>'
        )
    for node in shown:
        x, y = positions[str(node.get("id"))]
        highlight = str(node.get("highlight") or "candidate")
        fill = "#dcfce7" if highlight == "global_champion" else "#ffffff"
        border = "#dc2626" if highlight == "global_champion" else "#0284c7" if highlight == "round_winner" else "#0f172a"
        stroke_width = 3 if highlight == "global_champion" else 2 if highlight == "round_winner" else 1.5
        score = float(node.get("score") or 0.0)
        label = f"{score:.2f}" if score else str(node.get("id") or "")[-2:]
        parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" stroke="{border}" stroke-width="{stroke_width}"/>')
        parts.append(f'<text x="{x}" y="{y + 4}" text-anchor="middle" font-size="12" font-weight="700" fill="#0f172a">{html.escape(label)}</text>')
        parts.append(f'<title>{html.escape(str(node.get("id") or ""))} · round {html.escape(str(node.get("outer_iteration") or "n/a"))} · score {score:.3f} · {html.escape(highlight)}</title>')
    if layout["hidden_count"]:
        parts.append(f'<text x="24" y="{height - 14}" font-size="11" fill="#64748b">+ {layout["hidden_count"]} additional nodes in champion_tree.json</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def champion_tree_png(tree: dict[str, Any]) -> bytes:
    nodes = tree.get("nodes") if isinstance(tree.get("nodes"), list) else []
    layout = _champion_tree_layout(tree, max_nodes=40)
    shown = layout["nodes"]
    positions = layout["positions"]
    shown_ids = {str(node.get("id")) for node in shown}
    edges = [edge for edge in layout["edges"] if str(edge.get("from")) in shown_ids and str(edge.get("to")) in shown_ids]
    width = int(layout["width"])
    height = int(layout["height"])
    radius = int(layout["radius"])
    canvas = _PngCanvas(width, height, "#ffffff")
    canvas.text(24, 18, "Champion tree", "#0f172a", 2)
    if not shown:
        canvas.text(24, 62, "No champion tree recorded", "#94a3b8", 2)
        return canvas.png()
    for edge in edges:
        x1, y1 = positions[str(edge.get("from"))]
        x2, y2 = positions[str(edge.get("to"))]
        canvas.line(int(x1), int(y1 + radius), int(x2), int(y2 - radius), "#334155")
    for node in shown:
        x, y = positions[str(node.get("id"))]
        highlight = str(node.get("highlight") or "candidate")
        fill = "#dcfce7" if highlight == "global_champion" else "#ffffff"
        border = "#dc2626" if highlight == "global_champion" else "#0284c7" if highlight == "round_winner" else "#0f172a"
        canvas.circle(int(x), int(y), radius, fill, border)
        canvas.text(int(x - radius + 8), int(y - 5), f"{float(node.get('score') or 0.0):.1f}", "#0f172a", 1)
    return canvas.png()


def _champion_tree_layout(tree: dict[str, Any], *, max_nodes: int) -> dict[str, Any]:
    all_nodes = tree.get("nodes") if isinstance(tree.get("nodes"), list) else []
    all_edges = tree.get("edges") if isinstance(tree.get("edges"), list) else []
    ordered = sorted(all_nodes, key=lambda node: (int(node.get("outer_iteration") or 0), str(node.get("id") or "")))
    champion_id = str(tree.get("global_champion_variant_id") or "")
    champion_ancestors = _champion_ancestor_ids(champion_id, all_edges)
    priority = [
        node for node in ordered
        if str(node.get("id")) == champion_id or str(node.get("id")) in champion_ancestors or node.get("is_round_winner")
    ]
    rest = [node for node in ordered if node not in priority]
    shown = (priority + rest)[:max_nodes]
    shown_ids = {str(node.get("id")) for node in shown}
    edges = [
        edge for edge in all_edges
        if str(edge.get("from")) in shown_ids and str(edge.get("to")) in shown_ids
    ]
    parents_by_child: dict[str, list[str]] = {}
    children_by_parent: dict[str, list[str]] = {}
    for edge in edges:
        parent = str(edge.get("from"))
        child = str(edge.get("to"))
        parents_by_child.setdefault(child, []).append(parent)
        children_by_parent.setdefault(parent, []).append(child)
    depth: dict[str, int] = {}
    roots = [str(node.get("id")) for node in shown if str(node.get("id")) not in parents_by_child]
    queue = list(roots)
    for root in roots:
        depth[root] = 0
    while queue:
        current = queue.pop(0)
        for child in children_by_parent.get(current, []):
            next_depth = depth[current] + 1
            if child not in depth or next_depth > depth[child]:
                depth[child] = next_depth
                queue.append(child)
    for node in shown:
        node_id = str(node.get("id"))
        depth.setdefault(node_id, max(0, int(node.get("outer_iteration") or 1) - 1))
    levels: dict[int, list[dict[str, Any]]] = {}
    for node in shown:
        levels.setdefault(depth[str(node.get("id"))], []).append(node)
    for level_nodes in levels.values():
        level_nodes.sort(key=lambda node: (0 if str(node.get("id")) == champion_id else 1, str(node.get("id") or "")))
    radius = 24
    x_gap = 92
    y_gap = 112
    top = 92
    left = 48
    max_width_count = max((len(level_nodes) for level_nodes in levels.values()), default=1)
    width = max(960, left * 2 + max_width_count * x_gap)
    height = max(180, top + (max(levels.keys(), default=0) + 1) * y_gap + 40)
    positions: dict[str, tuple[int, int]] = {}
    for level, level_nodes in levels.items():
        row_width = (len(level_nodes) - 1) * x_gap
        start_x = max(left + radius, (width - row_width) // 2)
        y = top + level * y_gap
        for index, node in enumerate(level_nodes):
            positions[str(node.get("id"))] = (start_x + index * x_gap, y)
    return {
        "nodes": shown,
        "edges": edges,
        "positions": positions,
        "width": width,
        "height": height,
        "radius": radius,
        "hidden_count": max(0, len(ordered) - len(shown)),
    }


def _champion_ancestor_ids(champion_id: str, edges: list[Any]) -> set[str]:
    parents_by_child: dict[str, list[str]] = {}
    for edge in edges:
        if isinstance(edge, dict):
            parents_by_child.setdefault(str(edge.get("to")), []).append(str(edge.get("from")))
    ancestors: set[str] = set()
    stack = list(parents_by_child.get(champion_id, []))
    while stack:
        parent = stack.pop()
        if parent in ancestors:
            continue
        ancestors.add(parent)
        stack.extend(parents_by_child.get(parent, []))
    return ancestors


def _mermaid_node_id(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not clean or clean[0].isdigit():
        clean = f"n_{clean}"
    return clean[:48]


def _direction_from_summary(summary: str) -> str:
    try:
        payload = json.loads(summary)
    except Exception:
        return summary[:48] if summary else "candidate"
    if isinstance(payload, dict):
        return str(payload.get("loss_reason") or payload.get("status") or payload.get("summary") or "candidate")
    return "candidate"


def decision_dag_svg(summary: dict[str, Any]) -> str:
    run = summary.get("run") or {}
    decision = summary.get("task_ingestion") or {}
    counts = summary.get("counts") or {}
    continuations = summary.get("continuation_decisions") or []
    best = float((summary.get("best_evaluation") or {}).get("score") or 0.0)
    width = 960
    left_x, right_x = 42, 510
    card_w, card_h = 390, 58
    cards: list[dict[str, Any]] = []

    def add(key: str, x: int, y: int, title: str, body: str, fill: str) -> None:
        cards.append({"key": key, "x": x, "y": y, "w": card_w, "h": card_h, "title": title, "body": body, "fill": fill})

    y = 82
    add("prompt", left_x, y, "1 User Prompt", str(run.get("user_goal", ""))[:64], "#dbeafe")
    add("route", right_x, y, "2 Task Router", f"{decision.get('product_agent', run.get('product_agent', 'unknown'))} / {decision.get('selected_mode', run.get('task_mode', 'unknown'))}", "#fce7f3")
    y += 88
    add("memory", left_x, y, "3 Lead Research Context", "PRD, plan, objective, seed context", "#ccfbf1")
    add("propose", right_x, y, "4 Propose Variants", f"{counts.get('variants', 0)} query/code variants across {counts.get('outer_rounds', 0)} rounds", "#e0f2fe")
    y += 88
    add("fanout", left_x, y, "5 Parallel Evaluation Batch", "variant evaluations run as independent async tasks", "#dbeafe")
    add("rank", right_x, y, "6 Evaluate + Rank", f"{counts.get('evaluations', 0)} evals, best score {best:.3f}", "#fee2e2")
    y += 88
    add("continue", left_x, y, "7 Continue Decision", f"{counts.get('continuation_decisions', 0)} explicit loop decisions", "#ccfbf1")
    previous_key = "continue"
    for idx, cont in enumerate(continuations[:8], start=1):
        y += 74
        color = "#dcfce7" if cont.get("decision") == "continue" else "#ffedd5"
        key = f"decision_{idx}"
        add(key, left_x, y, f"Decision r{cont.get('iteration', idx)}: {cont.get('decision', '?')}", str(cont.get("reason", ""))[:64], color)
        previous_key = key
    y += 88
    add("synthesis", left_x, y, "8 Critic + Synthesis", "review claims, contradictions, report", "#fef3c7")
    add("grounding", right_x, y, "9 Citation / Grounding", "claim-source links and source-backed output", "#ede9fe")
    y += 88
    add("persist", right_x, y, "10 Persist Artifacts", "report, traces, timeline, decisions, costs", "#e2e8f0")
    by_key = {card["key"]: card for card in cards}
    height = max(card["y"] + card["h"] for card in cards) + 46
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'style="font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;">',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L8,4 L0,8 Z" fill="#64748b"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<text x="28" y="45" font-size="24" font-weight="700" fill="#0f172a">Comprehensive decision DAG</text>',
    ]
    for card in cards:
        x, y, w, h = card["x"], card["y"], card["w"], card["h"]
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{card["fill"]}" stroke="#64748b"/>')
        parts.append(f'<text x="{x + 12}" y="{y + 23}" font-size="13" font-weight="700" fill="#0f172a">{html.escape(card["title"])}</text>')
        parts.append(f'<text x="{x + 12}" y="{y + 44}" font-size="11" fill="#475569">{html.escape(card["body"])}</text>')

    for start, end in [
        ("prompt", "route"),
        ("route", "memory"),
        ("memory", "propose"),
        ("propose", "fanout"),
        ("fanout", "rank"),
        ("rank", "continue"),
    ]:
        parts.append(_svg_card_arrow(by_key[start], by_key[end]))
    if continuations:
        parts.append(_svg_card_arrow(by_key["continue"], by_key["decision_1"]))
        for idx in range(1, min(len(continuations), 8)):
            parts.append(_svg_card_arrow(by_key[f"decision_{idx}"], by_key[f"decision_{idx + 1}"]))
        parts.append(_svg_card_arrow(by_key[previous_key], by_key["synthesis"]))
    else:
        parts.append(_svg_card_arrow(by_key["continue"], by_key["synthesis"]))
    parts.append(_svg_card_arrow(by_key["synthesis"], by_key["grounding"]))
    parts.append(_svg_card_arrow(by_key["grounding"], by_key["persist"]))
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_card_arrow(start: dict[str, Any], end: dict[str, Any]) -> str:
    sx, sy, sw, sh = int(start["x"]), int(start["y"]), int(start["w"]), int(start["h"])
    ex, ey, ew, eh = int(end["x"]), int(end["y"]), int(end["w"]), int(end["h"])
    if ey > sy + sh:
        return _svg_arrow(sx + sw // 2, sy + sh, ex + ew // 2, ey)
    if ey + eh < sy:
        return _svg_arrow(sx + sw // 2, sy, ex + ew // 2, ey + eh)
    if ex > sx:
        return _svg_arrow(sx + sw, sy + sh // 2, ex, ey + eh // 2)
    return _svg_arrow(sx, sy + sh // 2, ex + ew, ey + eh // 2)


def _svg_arrow(x1: int, y1: int, x2: int, y2: int) -> str:
    if abs(y1 - y2) <= 2 or abs(x1 - x2) <= 2:
        path = f"M{x1},{y1} L{x2},{y2}"
    else:
        mid = (x1 + x2) // 2
        path = f"M{x1},{y1} L{mid},{y1} L{mid},{y2} L{x2},{y2}"
    return f'<path d="{path}" fill="none" stroke="#64748b" stroke-width="1.6" marker-end="url(#arrow)"/>'


def decision_dag_png(summary: dict[str, Any]) -> bytes:
    run = summary.get("run") or {}
    decision = summary.get("task_ingestion") or {}
    counts = summary.get("counts") or {}
    continuations = summary.get("continuation_decisions") or []
    best = float((summary.get("best_evaluation") or {}).get("score") or 0.0)
    width = 960
    cards: list[tuple[int, int, int, int, str, str, str]] = []
    x1, x2 = 42, 510
    w, h = 390, 58
    y = 82
    cards.append((x1, y, w, h, "1 User prompt", str(run.get("user_goal", ""))[:64], "#dbeafe"))
    cards.append((x2, y, w, h, "2 Task router", f"{decision.get('product_agent', run.get('product_agent', 'unknown'))} / {decision.get('selected_mode', run.get('task_mode', 'unknown'))}", "#fce7f3"))
    y += 88
    cards.append((x1, y, w, h, "3 Lead research context", "PRD, source strategy, objective, context", "#ccfbf1"))
    cards.append((x2, y, w, h, "4 Propose variants", f"{counts.get('variants', 0)} query/code variants across {counts.get('outer_rounds', 0)} rounds", "#e0f2fe"))
    y += 88
    cards.append((x1, y, w, h, "5 Parallel evaluation batch", "variant evaluations run as independent tasks", "#dbeafe"))
    cards.append((x2, y, w, h, "6 Evaluate + rank", f"{counts.get('evaluations', 0)} evals, best score {best:.3f}", "#fee2e2"))
    y += 88
    cards.append((x1, y, w, h, "7 Continue decision", f"{counts.get('continuation_decisions', 0)} explicit loop decisions", "#ccfbf1"))
    for idx, cont in enumerate(continuations[:8], start=1):
        y += 74
        color = "#dcfce7" if cont.get("decision") == "continue" else "#ffedd5"
        cards.append((x1, y, w, h, f"Decision r{cont.get('iteration', idx)}: {cont.get('decision', '?')}", str(cont.get("reason", ""))[:64], color))
    y += 88
    cards.append((x1, y, w, h, "8 Critic + synthesis", "review claims, contradictions, write report", "#fef3c7"))
    cards.append((x2, y, w, h, "9 Citation / grounding", "claim-source links and source-backed output", "#ede9fe"))
    y += 88
    cards.append((x2, y, w, h, "10 Persist artifacts", "report, traces, timeline, decisions, costs", "#e2e8f0"))
    height = max(card_y + card_h for _, card_y, _, card_h, _, _, _ in cards) + 42
    canvas = _PngCanvas(width, height, "#f8fafc")
    canvas.text(28, 24, "Comprehensive decision DAG", "#0f172a", 2)
    for card in cards:
        _draw_card(canvas, *card)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]:
        _draw_card_arrow(canvas, cards[a], cards[b])
    if continuations:
        _draw_card_arrow(canvas, cards[6], cards[7])
        first_decision_index = 7
        last_decision = first_decision_index + min(len(continuations), 8) - 1
        for idx in range(first_decision_index, last_decision):
            _draw_card_arrow(canvas, cards[idx], cards[idx + 1])
        _draw_card_arrow(canvas, cards[last_decision], cards[-3])
    else:
        _draw_card_arrow(canvas, cards[6], cards[-3])
    _draw_card_arrow(canvas, cards[-3], cards[-2])
    _draw_card_arrow(canvas, cards[-2], cards[-1])
    return canvas.png()


def _draw_card(canvas: _PngCanvas, x: int, y: int, w: int, h: int, title: str, body: str, fill: str) -> None:
    canvas.rect(x, y, w, h, fill)
    canvas.outline(x, y, w, h, "#64748b")
    canvas.text(x + 10, y + 9, title, "#0f172a", 2, max_chars=32)
    canvas.text(x + 10, y + 34, body, "#475569", 1, max_chars=70)


def _draw_card_arrow(
    canvas: _PngCanvas,
    start: tuple[int, int, int, int, str, str, str],
    end: tuple[int, int, int, int, str, str, str],
) -> None:
    sx, sy, sw, sh = start[:4]
    ex, ey, ew, eh = end[:4]
    if ey > sy + sh:
        _draw_arrow(canvas, sx + sw // 2, sy + sh, ex + ew // 2, ey)
    elif ey + eh < sy:
        _draw_arrow(canvas, sx + sw // 2, sy, ex + ew // 2, ey + eh)
    elif ex > sx:
        _draw_arrow(canvas, sx + sw, sy + sh // 2, ex, ey + eh // 2)
    else:
        _draw_arrow(canvas, sx, sy + sh // 2, ex + ew, ey + eh // 2)


def _draw_arrow(canvas: _PngCanvas, x1: int, y1: int, x2: int, y2: int) -> None:
    if x1 == x2:
        y0, yh = sorted((y1, y2))
        canvas.rect(x1, y0, 2, max(1, yh - y0), "#64748b")
    else:
        x0, xh = sorted((x1, x2))
        canvas.rect(x0, y1, max(1, xh - x0), 2, "#64748b")
        if y1 != y2:
            canvas.rect(x2, min(y1, y2), 2, abs(y2 - y1), "#64748b")
    canvas.rect(x2 - 6, y2 - 4, 7, 2, "#64748b")
    canvas.rect(x2 - 6, y2 + 2, 7, 2, "#64748b")


def _write_png_from_svg_or_fallback(path: Path, svg: str, fallback: Any) -> None:
    if _write_png_from_svg(path, svg):
        return
    path.write_bytes(fallback())


def _write_png_from_svg(path: Path, svg: str) -> bool:
    converters = [
        ("rsvg-convert", _convert_with_rsvg),
        ("magick", _convert_with_magick),
        ("convert", _convert_with_convert),
        ("qlmanage", _convert_with_qlmanage),
    ]
    with tempfile.TemporaryDirectory(prefix="research_harness_svg_") as directory:
        tmp_dir = Path(directory)
        svg_path = tmp_dir / "source.svg"
        svg_path.write_text(svg, encoding="utf-8")
        for command, converter in converters:
            if not shutil.which(command):
                continue
            try:
                if converter(svg_path, path, tmp_dir):
                    return True
            except Exception:
                continue
    return False


def _convert_with_rsvg(svg_path: Path, output_path: Path, _tmp_dir: Path) -> bool:
    completed = subprocess.run(["rsvg-convert", str(svg_path), "-o", str(output_path)], text=True, capture_output=True, check=False)
    return completed.returncode == 0 and output_path.exists()


def _convert_with_magick(svg_path: Path, output_path: Path, _tmp_dir: Path) -> bool:
    completed = subprocess.run(["magick", str(svg_path), str(output_path)], text=True, capture_output=True, check=False)
    return completed.returncode == 0 and output_path.exists()


def _convert_with_convert(svg_path: Path, output_path: Path, _tmp_dir: Path) -> bool:
    completed = subprocess.run(["convert", str(svg_path), str(output_path)], text=True, capture_output=True, check=False)
    return completed.returncode == 0 and output_path.exists()


def _convert_with_qlmanage(svg_path: Path, output_path: Path, tmp_dir: Path) -> bool:
    completed = subprocess.run(
        ["qlmanage", "-t", "-s", "1600", "-o", str(tmp_dir), str(svg_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    generated = tmp_dir / f"{svg_path.name}.png"
    if completed.returncode == 0 and generated.exists():
        output_path.write_bytes(generated.read_bytes())
        return True
    return False


def run_benchmark_markdown(summary: dict[str, Any], dag: str, optimizer_flow: str) -> str:
    counts = summary.get("counts", {})
    decision = summary.get("task_ingestion") or {}
    lines = [
        "# Run Benchmark",
        "",
        f"- Run ID: `{(summary.get('run') or {}).get('id', 'unknown')}`",
        f"- Product agent: `{decision.get('product_agent', (summary.get('run') or {}).get('product_agent', 'unknown'))}`",
        f"- Mode: `{decision.get('selected_mode', (summary.get('run') or {}).get('task_mode', 'unknown'))}`",
        f"- Tasks passed: {counts.get('passed_tasks', 0)} / {counts.get('tasks', 0)}",
        f"- Outer rounds: {counts.get('outer_rounds', 0)}",
        f"- Variants evaluated: {counts.get('evaluations', 0)}",
        f"- Best score: {float((summary.get('best_evaluation') or {}).get('score', 0.0)):.3f}",
        "",
        "## Decision DAG",
        "",
        "```mermaid",
        dag.strip(),
        "```",
        "",
        "## Optimizer Flow",
        "",
        "```mermaid",
        optimizer_flow.strip(),
        "```",
        "",
        "## Round Summary",
    ]
    for round_record in summary.get("rounds", []):
        lines.append(
            f"- Round {round_record.get('outer_iteration')}: best `{round_record.get('best_variant_id')}` "
            f"score {float(round_record.get('best_score', 0.0)):.3f}; signal `{round_record.get('termination_signal')}`."
        )
    lines.extend(["", "## Optimizer Trace"])
    for item in summary.get("optimizer_trace", []):
        lines.append(
            f"- Round {item.get('round')}: `{item.get('learning_status')}`; "
            f"best `{item.get('best_variant_id')}` score {float(item.get('best_score', 0.0)):.3f}; "
            f"mean_edge `{item.get('best_mean_edge')}`; parameter_only `{item.get('all_variants_parameter_nudges')}`."
        )
    return "\n".join(lines) + "\n"


def run_benchmark_html(summary: dict[str, Any]) -> str:
    run      = summary.get("run") or {}
    decision = summary.get("task_ingestion") or {}

    spans, num_rows, total_ms = _build_timeline_spans(summary, for_agent_chart=False)

    run_id      = str(run.get("id", "unknown"))
    goal        = str(run.get("user_goal", ""))
    status      = str(run.get("status", "running"))
    mode        = str(decision.get("selected_mode", run.get("task_mode", "—")))
    product     = str(decision.get("product_agent", run.get("product_agent", "—")))
    total_tok   = int(run.get("total_tokens") or 0)
    total_cost  = float(run.get("total_cost") or 0.0)
    dur_s       = total_ms / 1000

    status_color = {"completed": "#10b981", "failed": "#ef4444", "running": "#3b82f6"}.get(status, "#94a3b8")

    evt_rows  = _event_rows_html(spans)
    stats     = _stats_cards_html(summary)
    rnd_rows  = _round_rows_html(summary)
    opt_rows  = _optimizer_trace_rows_html(summary)

    # Compact colour legend.
    legend = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px;'
        f'font-size:11px;color:#475569;">'
        f'<span style="width:10px;height:10px;border-radius:2px;background:{color};display:inline-block;"></span>'
        f'{html.escape(role.replace("_", " ").title())}</span>'
        for role, color in _ROLE_COLORS.items()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(run_id)} — Research Harness</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 24px 28px 48px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      font-size: 13px; color: #1e293b; background: #f1f5f9; line-height: 1.5;
    }}
    h2 {{
      font-size: 10px; font-weight: 700; letter-spacing: .1em;
      text-transform: uppercase; color: #94a3b8; margin: 20px 0 8px;
    }}
    .card {{
      background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
      padding: 18px 22px; margin-bottom: 14px;
    }}
    .header-top {{ display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }}
    .run-id {{ font-family: "SF Mono","Fira Code",monospace; font-size: 13px; font-weight: 600; color: #334155; }}
    .badge {{
      display: inline-flex; align-items: center; gap: 4px;
      padding: 2px 9px; border-radius: 999px; font-size: 11px;
      font-weight: 700; color: #fff; background: {status_color};
    }}
    .goal {{ font-size: 15px; color: #0f172a; margin: 6px 0 10px; font-weight: 500; }}
    .meta {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 12px; color: #64748b; }}
    .gantt-card {{
      background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
      padding: 14px 16px 10px; margin-bottom: 14px; overflow-x: auto;
    }}
    .legend {{ margin-bottom: 10px; display: flex; flex-wrap: wrap; gap: 4px; }}
    .events-card {{
      background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
      overflow: hidden; margin-bottom: 14px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th {{
      padding: 8px 12px; text-align: left; font-weight: 600; color: #64748b;
      border-bottom: 1px solid #f1f5f9; background: #f8fafc;
      font-size: 10px; letter-spacing: .07em; text-transform: uppercase;
    }}
    td {{ padding: 7px 12px; border-bottom: 1px solid #f8fafc; vertical-align: middle; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #f8fafc; }}
    .stats {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }}
    .rounds-card {{
      background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; overflow: hidden;
    }}
  </style>
</head>
<body>

  <!-- ── Header ───────────────────────────────────────── -->
  <div class="card">
    <div class="header-top">
      <span class="run-id">{html.escape(run_id)}</span>
      <span class="badge">● {html.escape(status)}</span>
    </div>
    <div class="goal">{html.escape(goal[:140])}</div>
    <div class="meta">
      <span>⏱ {html.escape(_fmt_duration(dur_s))}</span>
      <span>⬡ {total_tok:,} tokens</span>
      <span>${total_cost:.4f}</span>
      <span>mode: <b>{html.escape(mode)}</b></span>
      <span>agent: <b>{html.escape(product)}</b></span>
    </div>
  </div>

  <!-- ── Gantt timeline ───────────────────────────────── -->
  <h2>Agent Timeline</h2>
  <div class="gantt-card">
    <div class="legend">{legend}</div>
    <img src="agent_timeline.png" alt="Agent timeline" style="width:100%;display:block;">
  </div>

  <h2>Decision DAG</h2>
  <div class="gantt-card">
    <img src="decision_dag.png" alt="Decision DAG" style="width:100%;display:block;">
  </div>

  <h2>Optimizer Flow</h2>
  <div class="gantt-card">
    <img src="optimizer_flow.png" alt="Optimizer learning flow" style="width:100%;display:block;">
  </div>

  <h2>Optimizer Trace</h2>
  <div class="events-card">
    <table>
      <thead>
        <tr>
          <th>Round</th>
          <th>Learning status</th>
          <th>Best score</th>
          <th>Mean edge</th>
          <th>Score source</th>
          <th>Variants / evals</th>
          <th>Parent / score reuse</th>
          <th>Param only?</th>
          <th>Payload / variant preview</th>
        </tr>
      </thead>
      <tbody>{opt_rows}</tbody>
    </table>
  </div>

  <!-- ── Event log ────────────────────────────────────── -->
  <h2>Agent Events</h2>
  <div class="events-card">
    <table>
      <thead>
        <tr>
          <th>Role</th>
          <th>Agent</th>
          <th>Summary</th>
          <th style="text-align:right;">Tokens</th>
          <th style="text-align:right;">Duration</th>
          <th style="text-align:right;">Offset</th>
        </tr>
      </thead>
      <tbody>{evt_rows}</tbody>
    </table>
  </div>

  <!-- ── Stats cards ──────────────────────────────────── -->
  <h2>Run Stats</h2>
  <div class="stats">{stats}</div>

  <!-- ── Evolution rounds ─────────────────────────────── -->
  <h2>Evolution Rounds</h2>
  <div class="rounds-card">
    <table>
      <thead>
        <tr>
          <th>#</th><th>Mode</th><th>Best score</th>
          <th>Signal</th><th>Plateau</th>
        </tr>
      </thead>
      <tbody>{rnd_rows}</tbody>
    </table>
  </div>

</body>
</html>"""


def _mermaid(text: str) -> str:
    return text.replace('"', "'").replace("\n", " ")[:110]
