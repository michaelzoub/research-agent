"""Model-selected swarm, sweep, and learning tools for configured graders."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

from ..llm import LLMClient
from ..schemas import AgentTrace, now_iso
from .base import ToolContext, ToolResult
from .graders import get_optimization_grader


def _safe_workspace_path(value: str, context: ToolContext) -> Path | None:
    path = (Path(context.workspace) / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    roots = [Path(root).resolve() for root in context.readable_roots]
    return path if path.is_file() and any(path.is_relative_to(root) for root in roots) else None


def _code_from_response(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


class SaveLearningTool:
    name = "save_learning"
    is_read_only = False
    description = "Save a confirmed breakthrough, dead end, or robust parameter finding for later optimization agents. Include the actual evaluation evidence; do not record speculation as confirmed."
    input_schema = {"type": "object", "required": ["title", "finding", "evidence", "status"], "properties": {
        "title": {"type": "string", "minLength": 3, "maxLength": 160}, "finding": {"type": "string", "minLength": 12, "maxLength": 4000},
        "evidence": {"type": "string", "minLength": 6, "maxLength": 4000}, "status": {"type": "string", "enum": ["confirmed", "dead_end", "hypothesis"]},
    }, "additionalProperties": False}

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.store is None:
            return ToolResult("error", error="Saving a learning requires an artifact store.")
        path = context.store.append_learning(run_id=context.run_id, **{key: str(arguments[key]) for key in ("title", "finding", "evidence", "status")})
        return ToolResult("ok", {"learning_path": str(path), "status": arguments["status"]})


class OptimizationSwarmTool:
    name = "spawn_optimization_agents"
    is_read_only = False
    description = "Launch independent optimization workers for the configured prediction-market grader. Workers can either test a hypothesis from a base strategy or start from scratch to escape a local optimum; every worker produces a complete candidate and receives an official score."
    input_schema = {"type": "object", "required": ["agents"], "properties": {
        "base_strategy_path": {"type": "string", "minLength": 1, "maxLength": 1000},
        "agents": {"type": "array", "minItems": 1, "maxItems": 8, "items": {"type": "object", "required": ["hypothesis", "evaluation_protocol", "target_to_beat"], "properties": {
            "hypothesis": {"type": "string", "minLength": 12, "maxLength": 2000}, "evaluation_protocol": {"type": "string", "minLength": 6, "maxLength": 800},
            "target_to_beat": {"type": "number"}, "base_strategy_notes": {"type": "string", "maxLength": 1000}, "strategy_mode": {"type": "string", "enum": ["base_variant", "from_scratch"]},
        }, "additionalProperties": False}},
    }, "additionalProperties": False}

    def __init__(self, llm: LLMClient): self.llm = llm

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.store is None: return ToolResult("error", error="Swarm execution requires an artifact store.")
        base_path = _safe_workspace_path(str(arguments["base_strategy_path"]), context) if arguments.get("base_strategy_path") else None
        if arguments.get("base_strategy_path") and base_path is None: return ToolResult("error", error="base_strategy_path must be a readable file under an approved workspace root.")
        base_code = base_path.read_text(encoding="utf-8") if base_path else ""
        grader = get_optimization_grader("prediction_market")
        workers: list[dict[str, Any]] = []
        for index, spec in enumerate(arguments["agents"], 1):
            started, started_at = time.perf_counter(), now_iso()
            mode = str(spec.get("strategy_mode") or "base_variant")
            if mode == "base_variant" and not base_code:
                workers.append({"worker": index, "hypothesis": spec["hypothesis"], "failure": "base_variant workers require base_strategy_path", "official_measured": False})
                continue
            system = "You are an independent optimization worker. Return only complete Python source defining Strategy(BaseStrategy). " + ("Ignore all incumbent code and design from first principles to test your assigned hypothesis." if mode == "from_scratch" else "Modify the supplied base strategy only to test your assigned hypothesis.") + " Do not claim a score or write prose."
            prompt = ("Base strategy:\n```python\n%s\n```\n\n" % base_code if mode == "base_variant" else "") + "Strategy mode: %s\nHypothesis: %s\nEvaluation protocol: %s\nTarget to beat: %s\nNotes: %s" % (mode, spec["hypothesis"], spec["evaluation_protocol"], spec["target_to_beat"], spec.get("base_strategy_notes", ""))
            try:
                response = await asyncio.to_thread(self.llm.complete, system, prompt, max_output_tokens=5000, temperature=0.5)
                code = _code_from_response(response.text)
                candidate_id = "swarm_%02d_%s" % (index, hashlib.sha256(code.encode()).hexdigest()[:12])
                path = Path(context.store.candidates_dir) / f"{candidate_id}.py"; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(code, encoding="utf-8")
                result = await asyncio.to_thread(grader.evaluate, path)
                workers.append({"worker": index, "hypothesis": spec["hypothesis"], "candidate_path": str(path), "candidate_id": candidate_id, "official_measured": bool(result.get("official_measured")), "score_eligible": bool(result.get("score_eligible")), "mean_edge": result.get("mean_edge"), "failure": result.get("error"), "evaluation_protocol": spec["evaluation_protocol"], "target_to_beat": spec["target_to_beat"]})
                context.store.add_trace(AgentTrace(run_id=context.run_id, agent_name=f"optimization_worker:{index}", role="grader_swarm_worker", prompt=prompt, model=response.model, tools_used=["evaluate_prediction_market_candidate"], tool_calls=[], token_usage=response.prompt_tokens + response.completion_tokens, runtime_ms=int((time.perf_counter()-started)*1000), status="completed", errors=[] if result.get("official_measured") else [str(result.get("error") or "unmeasured")], output_summary=code[:500], started_at=started_at, prompt_tokens=response.prompt_tokens, completion_tokens=response.completion_tokens, cost_usd=round(response.cost, 6)))
            except Exception as exc:
                workers.append({"worker": index, "hypothesis": spec["hypothesis"], "failure": f"{type(exc).__name__}: {exc}", "official_measured": False})
        audit = context.store.root / "swarm_results.json"; audit.write_text(json.dumps(workers, indent=2, sort_keys=True), encoding="utf-8")
        measured = [item for item in workers if item.get("score_eligible")]
        return ToolResult("ok" if measured else "error", {"workers": workers, "result_path": str(audit), "best": max(measured, key=lambda item: float(item.get("mean_edge") or 0)) if measured else None}, error=None if measured else "No swarm worker produced an eligible official measurement.")


class ParameterSweepTool:
    name = "run_parameter_sweep"
    is_read_only = False
    description = "Create isolated exact-replacement variants from a base strategy, evaluate every variant with the official grader's fixed multi-seed protocol, rank mean edge, and write the measured winner as a new strategy file."
    input_schema = {"type": "object", "required": ["base_strategy_path", "old_value", "values", "seed_start", "simulations"], "properties": {
        "base_strategy_path": {"type": "string", "minLength": 1, "maxLength": 1000}, "old_value": {"type": "string", "minLength": 1, "maxLength": 500},
        "values": {"type": "array", "minItems": 2, "maxItems": 48, "items": {"type": "string", "minLength": 1, "maxLength": 500}}, "seed_start": {"type": "integer", "minimum": 0}, "simulations": {"type": "integer", "minimum": 2, "maximum": 256},
    }, "additionalProperties": False}

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.store is None: return ToolResult("error", error="Parameter sweeps require an artifact store.")
        base_path = _safe_workspace_path(str(arguments["base_strategy_path"]), context)
        if base_path is None: return ToolResult("error", error="base_strategy_path must be a readable file under an approved workspace root.")
        base = base_path.read_text(encoding="utf-8"); old = str(arguments["old_value"])
        if old not in base: return ToolResult("error", error="old_value does not occur in the base strategy; no variants were created.")
        grader, rows = get_optimization_grader("prediction_market"), []
        sweep_dir = context.store.root / "sweeps"; sweep_dir.mkdir(parents=True, exist_ok=True)
        for index, value in enumerate(arguments["values"], 1):
            code = base.replace(old, str(value)); path = sweep_dir / f"variant_{index:03d}.py"; path.write_text(code, encoding="utf-8")
            result = await asyncio.to_thread(grader.evaluate, path, simulations=str(arguments["simulations"]), seed_start=str(arguments["seed_start"]))
            rows.append({"value": value, "candidate_path": str(path), "mean_edge": result.get("mean_edge"), "official_measured": bool(result.get("official_measured")), "score_eligible": bool(result.get("score_eligible")), "failure": result.get("error")})
        eligible = [row for row in rows if row["score_eligible"]]
        if not eligible: return ToolResult("error", {"variants": rows}, error="No sweep variant produced an eligible official measurement.")
        winner = max(eligible, key=lambda row: float(row["mean_edge"] or 0)); winner_path = sweep_dir / "winner.py"; winner_path.write_text(Path(winner["candidate_path"]).read_text(encoding="utf-8"), encoding="utf-8")
        report = {"base_strategy_path": str(base_path), "replacement": old, "seed_start": arguments["seed_start"], "simulations": arguments["simulations"], "variants": rows, "winner": {**winner, "winner_path": str(winner_path)}}
        report_path = sweep_dir / "results.json"; report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return ToolResult("ok", {"result_path": str(report_path), "winner": report["winner"], "variant_count": len(rows)})
