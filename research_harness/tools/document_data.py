"""Artifact-backed document extraction, analysis, and deterministic SVG charts."""
from __future__ import annotations

import asyncio
import html
import json
import math
import re
from typing import Any, Sequence

from ..schemas import AgentTrace, ProvenanceEdge, now_iso
from .base import ToolContext, ToolResult


def _source(context: ToolContext, source_id: str) -> dict[str, Any] | None:
    if context.store is None:
        return None
    source = context.store.find_by("sources", "id", source_id)
    return source if source and source.get("evidence_kind") == "verified_document" else None


def _safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", value):
        raise ValueError("Artifact IDs may contain only letters, digits, dots, underscores, and hyphens.")
    return value


def _column_unit(name: str) -> str | None:
    match = re.search(r"(?:\[([^\]]+)\]|\(([^()]+)\))\s*$", name)
    return (match.group(1) or match.group(2)).strip() if match else None


def _number(value: Any) -> float | None:
    cleaned = str(value).strip().replace(",", "")
    # Do not turn a date, CI, or free-form value into a fabricated measurement.
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?%?", cleaned):
        return None
    return float(cleaned[:-1]) / 100.0 if cleaned.endswith("%") else float(cleaned)


class StructuredDataExtractionTool:
    name = "extract_structured_data"
    is_read_only = False
    description = "Extract tables, numeric results, measurements, time-series-like rows, and key-value facts from an already fetched verified document. Persists the complete normalized dataset and returns only a preview plus provenance."
    input_schema = {"type": "object", "required": ["source_id"], "properties": {
        "source_id": {"type": "string", "minLength": 3, "maxLength": 160},
        "dataset_id": {"type": "string", "minLength": 1, "maxLength": 120},
        "kind": {"type": "string", "enum": ["all", "tables", "numeric", "key_values"]},
        "max_rows": {"type": "integer", "minimum": 1, "maximum": 10000},
    }, "additionalProperties": False}

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.store is None:
            return ToolResult("error", error="Structured extraction requires an artifact store.")
        source_id, kind = str(arguments["source_id"]), str(arguments.get("kind", "all"))
        source = _source(context, source_id)
        if source is None:
            return ToolResult("error", error="source_id must identify a previously fetched verified document.")
        try: dataset_id = _safe_id(str(arguments.get("dataset_id") or f"dataset-{source_id}"))
        except ValueError as exc: return ToolResult("error", error=str(exc))
        max_rows = int(arguments.get("max_rows", 5000))
        tables = list(source.get("structured_tables") or []) if kind in {"all", "tables"} else []
        records: list[dict[str, Any]] = []
        provenance: list[dict[str, Any]] = []
        for table in tables:
            headers = [str(value).strip() or f"column_{index + 1}" for index, value in enumerate(table.get("headers") or [])]
            for row_index, row in enumerate(table.get("rows") or [], start=1):
                record = {headers[index]: str(value) for index, value in enumerate(row) if index < len(headers)}
                if record:
                    records.append(record)
                    provenance.append({"source_id": source_id, "table": table.get("name"), "row": row_index, "locator": table.get("locator")})
        if kind in {"all", "numeric", "key_values"}:
            for section, text in dict(source.get("evidence_sections") or {}).items():
                locator = (source.get("evidence_locators") or {}).get(section, [])
                for line in re.split(r"(?<=[.;])\s+|\n", str(text)):
                    match = re.match(r"\s*([^:;]{2,100})\s*[:=]\s*([-+]?\d[\d,.]*(?:%|\s*[A-Za-zµμ/²^]+)?)\s*$", line)
                    if not match:
                        continue
                    label, raw = match.group(1).strip(), match.group(2).strip()
                    value_match = re.match(r"([-+]?\d[\d,.]*%?)(?:\s*(.*))?", raw)
                    if value_match is None: continue
                    record = {"key": label, "value": value_match.group(1), "unit": (value_match.group(2) or "").strip()}
                    records.append(record)
                    provenance.append({"source_id": source_id, "section": section, "locator": locator})
        records, provenance = records[:max_rows], provenance[:max_rows]
        if not records:
            return ToolResult("error", error="No supported structured values were found in the verified document.")
        columns = sorted({key for record in records for key in record})
        payload = {"id": dataset_id, "source_id": source_id, "source_url": source.get("url"), "kind": kind, "columns": [{"name": name, "unit": _column_unit(name)} for name in columns], "records": records, "provenance": provenance, "truncated": len(records) >= max_rows, "created_at": now_iso()}
        try:
            path = context.store.write_dataset(dataset_id, payload)
        except ValueError as exc:
            return ToolResult("error", error=str(exc))
        context.store.add_provenance_edge(ProvenanceEdge(run_id=context.run_id or context.store.root.name, from_type="source", from_id=source_id, to_type="dataset", to_id=dataset_id, relationship="extracted_into", metadata={"records": len(records), "path": str(path.name)}))
        return ToolResult("ok", {"dataset_id": dataset_id, "record_count": len(records), "columns": payload["columns"], "preview": records[:10], "provenance_preview": provenance[:10], "artifact": str(path.relative_to(context.store.root)), "truncated": payload["truncated"]})


class DocumentAnalysisTool:
    name = "analyze_research_document"
    is_read_only = False
    description = "Use the configured LLM to analyze a previously fetched verified paper or technical document. Returns clearly separated source-stated findings and model inferences with source section/page locators; full analysis is persisted as an artifact."
    input_schema = {"type": "object", "required": ["source_id"], "properties": {"source_id": {"type": "string", "minLength": 3, "maxLength": 160}, "focus": {"type": "string", "minLength": 2, "maxLength": 1000}}, "additionalProperties": False}

    def __init__(self, llm: Any): self.llm = llm

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.store is None: return ToolResult("error", error="Document analysis requires an artifact store.")
        source_id = str(arguments["source_id"]); source = _source(context, source_id)
        if source is None: return ToolResult("error", error="source_id must identify a previously fetched verified document.")
        sections = dict(source.get("evidence_sections") or {})
        evidence = "\n\n".join(f"[{name}]\n{text}" for name, text in sections.items())[:18000]
        system = "Return only JSON. Analyze only supplied evidence. Schema: {explicit:{introduction:[{text,section}],research_question:[{text,section}],methodology:[{text,section}],experimental_setup:[{text,section}],results:[{text,section}],conclusion:[{text,section}],assumptions:[{text,section}],limitations:[{text,section}]},inferences:[{text,basis_sections,confidence}]}. Explicit entries must be directly stated; inferences must be labeled interpretations. Use exact supplied section names. Empty arrays are allowed."
        prompt = f"Focus: {arguments.get('focus', 'general technical understanding')}\n\nDocument evidence:\n{evidence}"
        try:
            analysis = await asyncio.to_thread(self.llm.complete_json, system, prompt, max_output_tokens=1400, temperature=0.0)
        except Exception as exc:
            return ToolResult("error", error=f"Document analysis model call failed: {type(exc).__name__}: {exc}", retryable=True)
        if not isinstance(analysis, dict) or not isinstance(analysis.get("explicit"), dict) or not isinstance(analysis.get("inferences"), list):
            return ToolResult("error", error="Document analysis model returned an invalid grounded-analysis schema.")
        locators = source.get("evidence_locators") or {}
        required_sections = ("introduction", "research_question", "methodology", "experimental_setup", "results", "conclusion", "assumptions", "limitations")
        if any(not isinstance(analysis["explicit"].get(name), list) for name in required_sections):
            return ToolResult("error", error="Document analysis must provide every explicit-analysis field as an array.")
        for entries in analysis["explicit"].values():
            if not isinstance(entries, list): return ToolResult("error", error="Document analysis explicit fields must be arrays.")
            for entry in entries:
                if not isinstance(entry, dict) or str(entry.get("section")) not in sections:
                    return ToolResult("error", error="Document analysis cited an unknown source section.")
                entry["locators"] = locators.get(str(entry["section"]), [])
        for inference in analysis["inferences"]:
            basis = inference.get("basis_sections") if isinstance(inference, dict) else None
            if not isinstance(basis, list) or any(str(section) not in sections for section in basis):
                return ToolResult("error", error="Document analysis inference cited an unknown source section.")
            inference["locators"] = {str(section): locators.get(str(section), []) for section in basis}
        try: analysis_id = _safe_id(f"analysis-{source_id}")
        except ValueError as exc: return ToolResult("error", error=str(exc))
        payload = {"id": analysis_id, "source_id": source_id, "source_url": source.get("url"), "focus": arguments.get("focus", ""), "analysis": analysis, "created_at": now_iso()}
        path = context.store.write_document_analysis(analysis_id, payload)
        context.store.add_provenance_edge(ProvenanceEdge(run_id=context.run_id or context.store.root.name, from_type="source", from_id=source_id, to_type="document_analysis", to_id=analysis_id, relationship="analyzed_into"))
        explicit = analysis["explicit"]
        preview = {name: entries[:2] for name, entries in explicit.items() if isinstance(entries, list) and entries}
        return ToolResult("ok", {"analysis_id": analysis_id, "source_id": source_id, "explicit_preview": preview, "inferences_preview": analysis["inferences"][:3], "artifact": str(path.relative_to(context.store.root))})


class SVGChartTool:
    name = "generate_svg_chart"
    is_read_only = False
    description = "Generate a deterministic SVG bar, line, scatter, grouped-bar, stacked-bar, or pie chart from a persisted extracted dataset. Validates numeric values and compatible units, and saves the SVG plus reproducible configuration and provenance."
    input_schema = {"type": "object", "required": ["dataset_id", "chart_type", "y_column"], "properties": {"dataset_id": {"type": "string", "minLength": 1, "maxLength": 120}, "chart_type": {"type": "string", "enum": ["bar", "line", "scatter", "grouped-bar", "stacked-bar", "pie"]}, "x_column": {"type": "string", "minLength": 1, "maxLength": 200}, "y_column": {"type": "string", "minLength": 1, "maxLength": 200}, "y_columns": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 200}, "minItems": 1, "maxItems": 8}, "title": {"type": "string", "maxLength": 300}, "chart_id": {"type": "string", "minLength": 1, "maxLength": 120}}, "additionalProperties": False}

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.store is None: return ToolResult("error", error="SVG chart generation requires an artifact store.")
        dataset_id = str(arguments["dataset_id"])
        try: dataset = context.store.read_dataset(dataset_id)
        except ValueError as exc: return ToolResult("error", error=str(exc))
        if not dataset: return ToolResult("error", error="Dataset artifact was not found.")
        chart_type, y_columns = str(arguments["chart_type"]), list(arguments.get("y_columns") or [arguments["y_column"]])
        if chart_type not in {"grouped-bar", "stacked-bar"} and len(y_columns) != 1: return ToolResult("error", error="Multiple y_columns are supported only for grouped-bar and stacked-bar charts.")
        columns = {item.get("name"): item.get("unit") for item in dataset.get("columns") or []}
        x_column = str(arguments.get("x_column") or "key")
        if x_column not in columns or any(column not in columns for column in y_columns): return ToolResult("error", error="Selected chart columns are not present in the dataset.")
        units = {str(columns[column] or "").strip().lower() for column in y_columns}
        if len(units) > 1: return ToolResult("error", error="Selected value columns have incompatible units; choose columns with the same unit.")
        rows = []
        for index, record in enumerate(dataset.get("records") or []):
            values = [_number(record.get(column)) for column in y_columns]
            if any(value is None or not math.isfinite(value) for value in values): continue
            rows.append((str(record.get(x_column, index + 1)), [float(value) for value in values]))
        if not rows: return ToolResult("error", error="The selected y column contains no finite numeric values.")
        if chart_type == "scatter" and any(_number(label) is None for label, _values in rows): return ToolResult("error", error="Scatter charts require a numeric x_column.")
        try: chart_id = _safe_id(str(arguments.get("chart_id") or f"chart-{dataset_id}"))
        except ValueError as exc: return ToolResult("error", error=str(exc))
        config = {"id": chart_id, "dataset_id": dataset_id, "source_id": dataset.get("source_id"), "chart_type": chart_type, "x_column": x_column, "y_columns": y_columns, "unit": next(iter(units)), "title": str(arguments.get("title") or chart_id), "rows_used": len(rows), "created_at": now_iso()}
        svg = _svg(chart_type, rows, config)
        try: svg_path, config_path = context.store.write_chart(chart_id, svg, config)
        except ValueError as exc: return ToolResult("error", error=str(exc))
        context.store.add_provenance_edge(ProvenanceEdge(run_id=context.run_id or context.store.root.name, from_type="dataset", from_id=dataset_id, to_type="svg_chart", to_id=chart_id, relationship="visualized_as", metadata={"chart_type": chart_type, "unit": config["unit"]}))
        return ToolResult("ok", {"chart_id": chart_id, "dataset_id": dataset_id, "chart_type": chart_type, "rows_used": len(rows), "unit": config["unit"], "svg_artifact": str(svg_path.relative_to(context.store.root)), "config_artifact": str(config_path.relative_to(context.store.root))})


def _svg(chart_type: str, rows: Sequence[tuple[str, list[float]]], config: dict[str, Any]) -> str:
    width, height, left, top, bottom = 800, 480, 64, 54, 72
    plot_w, plot_h = width - left - 28, height - top - bottom
    maximum = max(sum(values) if chart_type == "stacked-bar" else max(values) for _label, values in rows) or 1.0
    esc = html.escape
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img"><title>{esc(config["title"])}</title><rect width="100%" height="100%" fill="white"/><text x="{left}" y="28" font-family="sans-serif" font-size="18">{esc(config["title"])}</text><path d="M {left} {top} V {top + plot_h} H {left + plot_w}" stroke="#333" fill="none"/>']
    palette = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]
    if chart_type == "pie":
        total = sum(values[0] for _label, values in rows)
        if total <= 0: raise ValueError("Pie charts require positive total values.")
        angle, cx, cy, radius = -math.pi / 2, 390, 255, 150
        for index, (label, values) in enumerate(rows):
            next_angle = angle + 2 * math.pi * values[0] / total
            x1, y1, x2, y2 = cx + radius * math.cos(angle), cy + radius * math.sin(angle), cx + radius * math.cos(next_angle), cy + radius * math.sin(next_angle)
            large = int(next_angle - angle > math.pi)
            parts.append(f'<path d="M {cx} {cy} L {x1:.2f} {y1:.2f} A {radius} {radius} 0 {large} 1 {x2:.2f} {y2:.2f} Z" fill="{palette[index % len(palette)]}"/><text x="{left}" y="{top + index * 18}" font-family="sans-serif" font-size="12">{esc(label)}: {values[0]:g}</text>'); angle = next_angle
    elif chart_type in {"line", "scatter"}:
        points = []
        xs = [_number(label) if chart_type == "scatter" else float(i) for i, (label, _values) in enumerate(rows)]
        xmin, xmax = min(xs), max(xs); span = xmax - xmin or 1.0
        for index, ((_label, values), x) in enumerate(zip(rows, xs)):
            px, py = left + (x - xmin) / span * plot_w, top + plot_h - values[0] / maximum * plot_h
            points.append((px, py)); parts.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="4" fill="{palette[0]}"/>')
        if chart_type == "line": parts.append('<polyline points="' + " ".join(f"{x:.2f},{y:.2f}" for x, y in points) + '" fill="none" stroke="#2563eb" stroke-width="2"/>')
    else:
        slot = plot_w / len(rows)
        for index, (label, values) in enumerate(rows):
            base_x = left + index * slot + slot * .12
            if chart_type == "stacked-bar":
                cumulative = 0.0
                for series, value in enumerate(values):
                    h = value / maximum * plot_h; y = top + plot_h - (cumulative + value) / maximum * plot_h
                    parts.append(f'<rect x="{base_x:.2f}" y="{y:.2f}" width="{slot*.76:.2f}" height="{h:.2f}" fill="{palette[series % len(palette)]}"/>'); cumulative += value
            else:
                bar_w = slot * .76 / len(values)
                for series, value in enumerate(values):
                    h = value / maximum * plot_h; x, y = base_x + series * bar_w, top + plot_h - h
                    parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" fill="{palette[series % len(palette)]}"/>')
            parts.append(f'<text x="{base_x:.2f}" y="{top + plot_h + 18}" font-family="sans-serif" font-size="11">{esc(label[:18])}</text>')
    return "".join(parts) + "</svg>"
