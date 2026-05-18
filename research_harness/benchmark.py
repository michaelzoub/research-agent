from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .run_benchmarks import write_run_benchmarks
from .store import ArtifactStore


ARTIFACT_FILES = {
    "sources": "sources.json",
    "claims": "claims.json",
    "hypotheses": "hypotheses.json",
    "experiments": "experiments.json",
    "open_questions": "open_questions.json",
    "contradictions": "contradictions.json",
    "failed_paths": "failed_paths.json",
    "harness_changes": "harness_changes.json",
    "agent_traces": "agent_traces.json",
}


COUNT_KEYS = [
    "sources",
    "claims",
    "hypotheses",
    "experiments",
    "open_questions",
    "contradictions",
    "failed_paths",
    "harness_changes",
]


COUNT_ATTRS = {
    "sources": "source_count",
    "claims": "claim_count",
    "hypotheses": "hypothesis_count",
    "experiments": "experiment_count",
    "open_questions": "open_question_count",
    "contradictions": "contradiction_count",
    "failed_paths": "failed_path_count",
    "harness_changes": "harness_change_count",
}


@dataclass
class RunMetrics:
    run_id: str
    goal: str
    status: str
    task_type: str
    harness_config_id: str
    started_at: str
    completed_at: str
    duration_seconds: float
    agent_runtime_ms: int
    total_tokens: int
    total_cost: float
    source_count: int
    claim_count: int
    hypothesis_count: int
    experiment_count: int
    open_question_count: int
    contradiction_count: int
    failed_path_count: int
    harness_change_count: int
    agent_trace_count: int
    failed_agent_count: int
    error_count: int
    error_types: dict[str, int]
    source_types: dict[str, int]
    avg_claim_confidence: float
    avg_source_relevance: float
    avg_source_credibility: float


def main() -> None:
    args = build_parser().parse_args()
    if args.run_dir:
        store = ArtifactStore(args.run_dir)
        write_run_benchmarks(store)
        trace = json.loads((store.root / "optimizer_trace.json").read_text(encoding="utf-8")) if (store.root / "optimizer_trace.json").exists() else []
        print(f"Run benchmark: {store.run_benchmark_path}")
        print(f"Optimizer trace: {store.root / 'optimizer_trace.json'}")
        print(f"Optimizer flow: {store.root / 'optimizer_flow.png'}")
        print(f"Optimizer flow SVG: {store.root / 'optimizer_flow.svg'}")
        print(f"Champion tree graph: {store.champion_tree_graph_path}")
        print(f"Champion tree SVG: {store.champion_tree_svg_path}")
        if trace:
            best = max(trace, key=lambda row: float(row.get("best_mean_edge") or row.get("best_score") or 0.0))
            parameter_like = sum(1 for row in trace if row.get("all_variants_parameter_nudges"))
            print(
                "Optimizer summary: "
                f"{len(trace)} round(s), best mean_edge={best.get('best_mean_edge')}, "
                f"best score={float(best.get('best_score') or 0.0):.3f}, "
                f"parameter-like rounds={parameter_like}/{len(trace)}"
            )
        else:
            print("Optimizer summary: no optimizer rounds found in this run directory.")
        return
    output_dir = args.benchmark_output or Path("benchmarks") / timestamp_slug()
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = collect_runs(args.outputs)
    write_outputs(runs, output_dir)
    print(f"Runs benchmarked: {len(runs)}")
    print(f"Benchmark report: {output_dir / 'index.html'}")
    print(f"Summary JSON: {output_dir / 'summary.json'}")
    print(f"Summary CSV: {output_dir / 'runs.csv'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark research-harness runs from an outputs directory.")
    parser.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs"),
        help="Directory containing run_* output folders.",
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=None,
        help="Directory where benchmark charts and summaries should be written.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Regenerate the per-run benchmark, optimizer trace, and optimizer flow diagram for one outputs/<run> directory.",
    )
    return parser


def collect_runs(outputs_dir: Path) -> list[RunMetrics]:
    run_dirs = sorted(path for path in outputs_dir.iterdir() if path.is_dir() and is_run_dir(path.name))
    return [collect_run(path) for path in run_dirs]


def collect_run(run_dir: Path) -> RunMetrics:
    artifacts = {name: read_json(run_dir / filename, []) for name, filename in ARTIFACT_FILES.items()}
    run_record = first(read_json(run_dir / "runs.json", []), {})
    traces = artifacts["agent_traces"]
    claims = artifacts["claims"]
    sources = artifacts["sources"]
    errors = [error for trace in traces for error in trace.get("errors", [])]
    source_types = Counter(str(source.get("source_type", "unknown")) for source in sources)
    error_types = Counter(error_type(error) for error in errors)
    return RunMetrics(
        run_id=str(run_record.get("id") or run_dir.name),
        goal=str(run_record.get("user_goal", "")),
        status=str(run_record.get("status", "unknown")),
        task_type=str(run_record.get("task_type", "unknown")),
        harness_config_id=str(run_record.get("harness_config_id", "unknown")),
        started_at=str(run_record.get("started_at", "")),
        completed_at=str(run_record.get("completed_at", "")),
        duration_seconds=duration_seconds(run_record.get("started_at"), run_record.get("completed_at")),
        agent_runtime_ms=sum(int(trace.get("runtime_ms", 0) or 0) for trace in traces),
        total_tokens=int(run_record.get("total_tokens") or sum(int(trace.get("token_usage", 0) or 0) for trace in traces)),
        total_cost=float(run_record.get("total_cost") or 0.0),
        source_count=len(sources),
        claim_count=len(claims),
        hypothesis_count=len(artifacts["hypotheses"]),
        experiment_count=len(artifacts["experiments"]),
        open_question_count=len(artifacts["open_questions"]),
        contradiction_count=len(artifacts["contradictions"]),
        failed_path_count=len(artifacts["failed_paths"]),
        harness_change_count=len(artifacts["harness_changes"]),
        agent_trace_count=len(traces),
        failed_agent_count=sum(1 for trace in traces if trace.get("status") != "completed"),
        error_count=len(errors),
        error_types=dict(error_types),
        source_types=dict(source_types),
        avg_claim_confidence=average(float(claim.get("confidence", 0) or 0) for claim in claims),
        avg_source_relevance=average(float(source.get("relevance_score", 0) or 0) for source in sources),
        avg_source_credibility=average(float(source.get("credibility_score", 0) or 0) for source in sources),
    )


def write_outputs(runs: list[RunMetrics], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = [run.__dict__ for run in runs]
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(runs, output_dir / "runs.csv")
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    chart_specs = [
        ("duration_seconds.svg", "Run Duration (seconds)", [(run.run_id, run.duration_seconds) for run in runs]),
        ("agent_runtime_ms.svg", "Total Agent Runtime (ms)", [(run.run_id, run.agent_runtime_ms) for run in runs]),
        ("token_usage.svg", "Token Usage", [(run.run_id, run.total_tokens) for run in runs]),
        ("errors.svg", "Errors And Failed Agents", [(run.run_id, run.error_count) for run in runs], [(run.run_id, run.failed_agent_count) for run in runs]),
        (
            "harness_changes.svg",
            "Harness Changes Proposed",
            [(run.run_id, run.harness_change_count) for run in runs],
        ),
        (
            "quality_signals.svg",
            "Average Confidence / Relevance / Credibility",
            [(run.run_id, run.avg_claim_confidence) for run in runs],
            [(run.run_id, run.avg_source_relevance) for run in runs],
            [(run.run_id, run.avg_source_credibility) for run in runs],
        ),
    ]
    for filename, title, *series in chart_specs:
        (charts_dir / filename).write_text(grouped_bar_svg(title, series), encoding="utf-8")
    (charts_dir / "artifact_counts.svg").write_text(artifact_counts_svg(runs), encoding="utf-8")
    (charts_dir / "error_types.svg").write_text(counter_svg("Error Types", aggregate_counters(run.error_types for run in runs)), encoding="utf-8")
    (charts_dir / "source_types.svg").write_text(counter_svg("Source Types", aggregate_counters(run.source_types for run in runs)), encoding="utf-8")
    (output_dir / "index.html").write_text(html_report(runs), encoding="utf-8")


def write_csv(runs: list[RunMetrics], path: Path) -> None:
    fieldnames = [
        "run_id",
        "status",
        "task_type",
        "harness_config_id",
        "duration_seconds",
        "agent_runtime_ms",
        "total_tokens",
        "source_count",
        "claim_count",
        "hypothesis_count",
        "experiment_count",
        "open_question_count",
        "contradiction_count",
        "failed_path_count",
        "harness_change_count",
        "agent_trace_count",
        "failed_agent_count",
        "error_count",
        "avg_claim_confidence",
        "avg_source_relevance",
        "avg_source_credibility",
        "goal",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow({field: getattr(run, field) for field in fieldnames})


def html_report(runs: list[RunMetrics]) -> str:
    totals = {
        "runs": len(runs),
        "sources": sum(run.source_count for run in runs),
        "claims": sum(run.claim_count for run in runs),
        "hypotheses": sum(run.hypothesis_count for run in runs),
        "contradictions": sum(run.contradiction_count for run in runs),
        "errors": sum(run.error_count for run in runs),
        "harness_changes": sum(run.harness_change_count for run in runs),
        "tokens": sum(run.total_tokens for run in runs),
    }
    cards = "\n".join(f"<div class='card'><b>{html.escape(name)}</b><span>{value}</span></div>" for name, value in totals.items())
    chart_names = [
        "artifact_counts.svg",
        "duration_seconds.svg",
        "agent_runtime_ms.svg",
        "token_usage.svg",
        "errors.svg",
        "error_types.svg",
        "source_types.svg",
        "harness_changes.svg",
        "quality_signals.svg",
    ]
    charts = "\n".join(
        f"<section><img src='charts/{name}' alt='{html.escape(name)}'></section>" for name in chart_names
    )
    rows = "\n".join(run_row(run) for run in runs)
    generated = datetime.now(timezone.utc).isoformat()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>research-harness benchmark</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #182026; background: #f7f8f8; }}
    h1 {{ margin-bottom: 4px; }}
    .subtle {{ color: #5f6b73; margin-top: 0; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 24px 0; }}
    .card {{ background: white; border: 1px solid #d9dddf; border-radius: 8px; padding: 12px; }}
    .card span {{ display: block; font-size: 24px; margin-top: 6px; }}
    section {{ background: white; border: 1px solid #d9dddf; border-radius: 8px; padding: 16px; margin: 16px 0; overflow-x: auto; }}
    img {{ max-width: 100%; height: auto; }}
    table {{ border-collapse: collapse; width: 100%; background: white; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d9dddf; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #edf1f2; }}
    code {{ white-space: nowrap; }}
  </style>
</head>
<body>
  <h1>research-harness benchmark</h1>
  <p class="subtle">Generated {html.escape(generated)}</p>
  <div class="cards">{cards}</div>
  {charts}
  <section>
    <h2>Runs</h2>
    <table>
      <thead>
        <tr>
          <th>Run</th><th>Status</th><th>Config</th><th>Duration</th><th>Tokens</th><th>Artifacts</th><th>Errors</th><th>Goal</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </section>
</body>
</html>
"""


def run_row(run: RunMetrics) -> str:
    artifacts = (
        f"S:{run.source_count} C:{run.claim_count} H:{run.hypothesis_count} "
        f"Q:{run.open_question_count} X:{run.contradiction_count} HC:{run.harness_change_count}"
    )
    return (
        "<tr>"
        f"<td><code>{html.escape(run.run_id)}</code></td>"
        f"<td>{html.escape(run.status)}</td>"
        f"<td>{html.escape(run.harness_config_id)}</td>"
        f"<td>{run.duration_seconds:.2f}s</td>"
        f"<td>{run.total_tokens}</td>"
        f"<td>{html.escape(artifacts)}</td>"
        f"<td>{run.error_count} errors / {run.failed_agent_count} failed agents</td>"
        f"<td>{html.escape(run.goal)}</td>"
        "</tr>"
    )


def artifact_counts_svg(runs: list[RunMetrics]) -> str:
    series = []
    for key in COUNT_KEYS:
        attr = COUNT_ATTRS[key]
        series.append([(run.run_id, getattr(run, attr)) for run in runs])
    return grouped_bar_svg("Artifact Counts", series, labels=COUNT_KEYS)


def grouped_bar_svg(
    title: str,
    series: list[list[tuple[str, float]]],
    *extra_series: list[tuple[str, float]],
    labels: Optional[list[str]] = None,
) -> str:
    all_series = series + list(extra_series)
    if not all_series:
        all_series = [[]]
    run_ids = [run_id for run_id, _ in all_series[0]]
    labels = labels if labels is not None else ([] if len(all_series) == 1 else [f"value {index + 1}" for index in range(len(all_series))])
    width = max(720, 120 + len(run_ids) * 90)
    height = 420
    left = 70
    top = 54
    bottom = 92
    chart_h = height - top - bottom
    chart_w = width - left - 36
    max_value = max([value for values in all_series for _, value in values] + [1])
    group_w = chart_w / max(len(run_ids), 1)
    bar_w = max(4, group_w / (len(all_series) + 1))
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#d97706", "#0891b2", "#4f46e5", "#be123c"]
    parts = svg_header(width, height, title)
    parts.append(axis(left, top, chart_w, chart_h, max_value))
    for series_index, values in enumerate(all_series):
        value_by_run = dict(values)
        for run_index, run_id in enumerate(run_ids):
            value = value_by_run.get(run_id, 0)
            bar_h = 0 if max_value == 0 else (value / max_value) * chart_h
            x = left + run_index * group_w + 8 + series_index * bar_w
            y = top + chart_h - bar_h
            parts.append(
                f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w - 2:.1f}' height='{bar_h:.1f}' fill='{colors[series_index % len(colors)]}' />"
            )
    for index, run_id in enumerate(run_ids):
        x = left + index * group_w + group_w / 2
        parts.append(f"<text x='{x:.1f}' y='{height - 48}' font-size='10' text-anchor='end' transform='rotate(-35 {x:.1f},{height - 48})'>{html.escape(short_run(run_id))}</text>")
    if labels:
        parts.append(legend(labels, colors, left, height - 24))
    parts.append("</svg>")
    return "\n".join(parts)


def counter_svg(title: str, counter: dict[str, int]) -> str:
    values = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return grouped_bar_svg(title, [values], labels=["count"])


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white' />",
        f"<text x='24' y='30' font-size='18' font-weight='700' fill='#182026'>{html.escape(title)}</text>",
    ]


def axis(left: int, top: int, width: int, height: int, max_value: float) -> str:
    y0 = top + height
    ticks = []
    for index in range(5):
        value = max_value * index / 4
        y = y0 - height * index / 4
        ticks.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left + width}' y2='{y:.1f}' stroke='#edf1f2' />")
        ticks.append(f"<text x='{left - 8}' y='{y + 4:.1f}' font-size='10' text-anchor='end' fill='#5f6b73'>{value:.1f}</text>")
    ticks.append(f"<line x1='{left}' y1='{top}' x2='{left}' y2='{y0}' stroke='#9aa4aa' />")
    ticks.append(f"<line x1='{left}' y1='{y0}' x2='{left + width}' y2='{y0}' stroke='#9aa4aa' />")
    return "\n".join(ticks)


def legend(labels: list[str], colors: list[str], x: int, y: int) -> str:
    parts = []
    cursor = x
    for index, label in enumerate(labels):
        parts.append(f"<rect x='{cursor}' y='{y - 10}' width='10' height='10' fill='{colors[index % len(colors)]}' />")
        parts.append(f"<text x='{cursor + 14}' y='{y}' font-size='11' fill='#334149'>{html.escape(label)}</text>")
        cursor += 14 + len(label) * 7 + 18
    return "\n".join(parts)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def first(values: list[Any], default: Any) -> Any:
    return values[0] if values else default


def average(values: Iterable[float]) -> float:
    collected = list(values)
    if not collected:
        return 0.0
    return round(sum(collected) / len(collected), 3)


def duration_seconds(started_at: Any, completed_at: Any) -> float:
    started = parse_datetime(started_at)
    completed = parse_datetime(completed_at)
    if not started or not completed:
        return 0.0
    return round(max(0.0, (completed - started).total_seconds()), 3)


def parse_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def error_type(error: str) -> str:
    if ":" in error:
        return error.split(":", 1)[0].strip()
    return error.strip() or "unknown"


def aggregate_counters(counters: Iterable[dict[str, int]]) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for counter in counters:
        merged.update(counter)
    return dict(merged)


def short_run(run_id: str) -> str:
    label = run_id
    if "_run_" in label and label.split("_run_", 1)[0].isdigit():
        label = label.split("_run_", 1)[1]
    elif label.startswith("run_"):
        label = label.replace("run_", "", 1)
    if len(label) <= 30:
        return label
    return f"{label[:27]}..."


def is_run_dir(name: str) -> bool:
    return name.startswith("run_") or bool(re.match(r"^\d+_run_", name))


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    main()
