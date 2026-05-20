from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ...store import ArtifactStore
from ..types import EvalTask, GraderResult
from .common import _result


def _grade_prediction_market_solution(task: EvalTask, store: ArtifactStore) -> GraderResult:
    text = store.solution_path.read_text(encoding="utf-8") if store.solution_path.exists() else ""
    checks = [
        ("solution_exists", store.solution_path.exists()),
        ("optimized_candidate_exists", store.optimized_candidate_path.exists()),
        ("optimal_code_exists", store.optimal_code_path.exists()),
        ("optimization_result_exists", store.optimization_result_path.exists()),
        ("defines_strategy", "class Strategy" in text),
        ("has_on_step", "def on_step" in text),
        ("uses_upstream_api", "orderbook_pm_challenge" in text),
        ("uses_cancel_or_orders", "CancelAll" in text and "PlaceOrder" in text),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result("prediction_market_solution", "code", "static solution checks", score, passed, 1.0, "Generated solution.py checked against upstream API shape.", [{"check": name, "passed": passed} for name, passed in checks])


def _grade_prediction_market_proxy_score(task: EvalTask, store: ArtifactStore) -> GraderResult:
    optimize_evals = [row for row in store.list("variant_evaluations") if row.get("inner_loop") == "optimize"]
    best = max((float(row.get("score", 0.0)) for row in optimize_evals), default=0.0)
    # Threshold 0.45: normalized score maps 0-edge strategies to ~0.5; allow a
    # small margin so near-zero-edge strategies (acceptable baseline) still pass.
    passed = best >= 0.45
    return _result(
        "prediction_market_proxy_score",
        "code",
        "local proxy outcome verification",
        best,
        passed,
        1.0,
        f"Best local proxy score={best:.3f}. This is not the official upstream profit score.",
        [{"best_proxy_score": best, "official_score_required": True, "passed": passed}],
    )


def _grade_prediction_market_official_status(task: EvalTask, store: ArtifactStore) -> GraderResult:
    result = json.loads(store.optimization_result_path.read_text(encoding="utf-8")) if store.optimization_result_path.exists() else {}
    official = result.get("official_result", {}) if isinstance(result, dict) else {}
    source = official.get("score_source")
    measured = official.get("measured")
    candidate_path = official.get("candidate_path")
    # Accept either truthful outcome: if the upstream runner was used the result
    # must say so; if the fallback ran it must say it wasn't officially measured.
    if source == "upstream_orderbook_pm_challenge":
        expected_measured = measured is True
    else:
        expected_measured = measured is False
    checks = [
        ("optimization_result_exists", store.optimization_result_path.exists()),
        ("official_result_present", isinstance(official, dict) and bool(official)),
        ("measured_status_truthful", bool(expected_measured)),
        (
            "score_source_recorded",
            source in {
                "upstream_orderbook_pm_challenge",
                "upstream_repo_missing",
                "official_sandbox_failed",
                "official_scorer_json_error",
                "official_scorer_no_successes",
            },
        ),
        ("candidate_path_recorded", bool(candidate_path)),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "prediction_market_official_status",
        "code",
        "official-status verification",
        score,
        passed,
        1.0,
        f"official_result measured={measured}, score_source={source}, candidate_path={candidate_path}.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_prediction_market_artifact_containment(task: EvalTask, store: ArtifactStore) -> GraderResult:
    result = json.loads(store.optimization_result_path.read_text(encoding="utf-8")) if store.optimization_result_path.exists() else {}
    official = result.get("official_result", {}) if isinstance(result, dict) else {}
    candidate_path = Path(str(official.get("candidate_path", ""))) if official.get("candidate_path") else None
    candidate_files = list(store.candidates_dir.glob("*.py")) if store.candidates_dir.exists() else []
    repo_root = Path.cwd()
    leaked_files = [
        path
        for pattern in ("pm_strategy*.py", "tmp_pm*.py", "*prediction_market_strategy*.py")
        for path in repo_root.glob(pattern)
        if path.is_file()
    ]
    checks = [
        ("candidates_dir_exists", store.candidates_dir.exists()),
        ("candidate_files_exist", bool(candidate_files)),
        (
            "winner_inside_candidates_dir",
            bool(candidate_path) and candidate_path.exists() and store.candidates_dir.resolve() in candidate_path.resolve().parents,
        ),
        ("no_repo_root_strategy_leaks", not leaked_files),
        ("optimal_code_exists", store.optimal_code_path.exists()),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "prediction_market_artifact_containment",
        "code",
        "artifact containment",
        score,
        passed,
        1.0,
        f"{len(candidate_files)} candidate file(s); leaked repo-root strategy files={len(leaked_files)}.",
        [
            {"check": name, "passed": passed}
            for name, passed in checks
        ]
        + [{"check": "leaked_file", "path": str(path), "passed": False} for path in leaked_files],
    )


def _repo_root_strategy_leaks(repo_root: Path) -> list[Path]:
    return [
        path
        for pattern in ("pm_strategy*.py", "tmp_pm*.py", "*prediction_market_strategy*.py")
        for path in repo_root.glob(pattern)
        if path.is_file()
    ]


def _grade_prediction_market_candidate_files_only_in_outputs(task: EvalTask, store: ArtifactStore) -> GraderResult:
    result = json.loads(store.optimization_result_path.read_text(encoding="utf-8")) if store.optimization_result_path.exists() else {}
    official = result.get("official_result", {}) if isinstance(result, dict) else {}
    candidate_path = Path(str(official.get("candidate_path", ""))) if official.get("candidate_path") else None
    candidate_files = list(store.candidates_dir.glob("*.py")) if store.candidates_dir.exists() else []
    candidate_parent = store.candidates_dir.resolve()
    checks = [
        ("candidates_dir_exists", store.candidates_dir.exists()),
        ("candidate_files_exist", bool(candidate_files)),
        ("all_candidate_files_under_candidates_dir", all(candidate_parent in path.resolve().parents for path in candidate_files)),
        (
            "winner_candidate_under_candidates_dir",
            bool(candidate_path) and candidate_path.exists() and candidate_parent in candidate_path.resolve().parents,
        ),
        ("optimal_code_promoted", store.optimal_code_path.exists()),
        ("solution_promoted", store.solution_path.exists()),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "prediction_market_candidate_files_only_in_outputs",
        "code",
        "artifact containment",
        score,
        passed,
        1.0,
        f"{len(candidate_files)} candidate Python file(s) under {store.candidates_dir}.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_prediction_market_agentic_optimizer(task: EvalTask, store: ArtifactStore) -> GraderResult:
    steps = _read_json_list(store.root / "optimization_agent_steps.json")
    variants = [row for row in store.list("variants") if row.get("kind") == "code"]
    evaluations = [row for row in store.list("variant_evaluations") if row.get("inner_loop") == "optimize"]
    traces = store.list("agent_traces")
    rounds: dict[int, list[dict[str, object]]] = {}
    for variant in variants:
        try:
            rounds.setdefault(int(variant.get("outer_iteration", 0)), []).append(variant)
        except (TypeError, ValueError):
            continue
    signatures_by_round = {
        round_index: {_pm_strategy_signature(str(variant.get("payload", ""))) for variant in rows}
        for round_index, rows in rounds.items()
        if round_index > 0
    }
    negative_rounds = _negative_prediction_market_rounds(evaluations, variants)
    adjacent_repeated = [
        round_index
        for round_index in sorted(signatures_by_round)
        if round_index + 1 in signatures_by_round and signatures_by_round[round_index] == signatures_by_round[round_index + 1]
    ]
    controller_tools = {
        str(action.get("tool"))
        for step in steps
        for action in (step.get("actions") if isinstance(step.get("actions"), list) else [])
        if isinstance(action, dict)
    }
    reflections = [str(step.get("reflection", "")) for step in steps if isinstance(step, dict)]
    proposal_prompts = [
        str(trace.get("prompt", ""))
        for trace in traces
        if str(trace.get("agent_name", "")).startswith("llm_propose_prediction_market_code")
    ]
    prompt_blob = "\n".join(proposal_prompts)
    changed_after_negative = True
    for round_index in negative_rounds:
        if round_index + 1 not in signatures_by_round:
            continue
        previous = signatures_by_round.get(round_index, set())
        current = signatures_by_round.get(round_index + 1, set())
        if current and current.issubset(previous):
            changed_after_negative = False
            break
    checks = [
        ("controller_steps_exist", bool(steps)),
        ("tool_menu_used", {"read_champion", "read_failures", "read_evaluator_summary", "propose_strategy"}.issubset(controller_tools)),
        ("failure_reflection_recorded", not negative_rounds or any(_looks_like_failure_reflection(text) for text in reflections)),
        ("literature_tool_used_when_required", _literature_requirement_satisfied(steps)),
        ("generation_prompt_has_controller_context", not proposal_prompts or "optimizer_agent_context" in prompt_blob),
        ("generation_prompt_has_literature_context", not proposal_prompts or "literature_refresh_notes" in prompt_blob),
        ("consecutive_rounds_not_same_signature", not adjacent_repeated),
        ("changed_mechanism_after_negative_edge", changed_after_negative),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "prediction_market_agentic_optimizer",
        "code",
        "agentic optimizer trajectory verification",
        score,
        passed,
        1.0,
        (
            f"controller_steps={len(steps)}, tools={sorted(controller_tools)}, "
            f"negative_rounds={sorted(negative_rounds)}, repeated_adjacent_rounds={adjacent_repeated}."
        ),
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_no_repo_root_strategy_files(task: EvalTask, store: ArtifactStore) -> GraderResult:
    leaked_files = _repo_root_strategy_leaks(Path.cwd())
    passed = not leaked_files
    return _result(
        "no_repo_root_strategy_files",
        "code",
        "artifact containment",
        1.0 if passed else 0.0,
        passed,
        1.0,
        f"Repository root generated strategy leak count={len(leaked_files)}.",
        [{"check": "no_repo_root_strategy_files", "passed": passed}]
        + [{"check": "leaked_file", "path": str(path), "passed": False} for path in leaked_files],
    )


def _read_json_list(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _pm_strategy_signature(payload: str) -> str:
    text = payload.lower()
    compact = re.sub(r"\s+", "", text)
    features = [
        "cancel_only" if "return[cancelall()]" in compact else "",
        "place_order" if "placeorder" in text else "",
        "inventory" if "inventory" in text or "yes_inventory" in text or "no_inventory" in text else "",
        "cash" if "free_cash" in text or "cash" in text else "",
        "competitor" if "competitor_best" in text else "",
        "flow" if "filled_quantity" in text or "flow" in text else "",
        "controller" if "optimizer_agent_next_mechanism" in text else "",
    ]
    mechanisms = " ".join(re.findall(r"pm_strategy=[^\s]+|strategy_family=[^\s]+|optimizer_agent_next_mechanism='[^']+'", payload)[:4])
    basis = "|".join([feature for feature in features if feature] + [mechanisms[:200]])
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:12]


def _negative_prediction_market_rounds(
    evaluations: list[dict[str, object]],
    variants: list[dict[str, object]],
) -> set[int]:
    rounds_by_variant = {str(row.get("id")): int(row.get("outer_iteration", 0) or 0) for row in variants}
    negative_rounds: set[int] = set()
    for row in evaluations:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        edge = metrics.get("mean_edge")
        if isinstance(edge, (int, float)) and float(edge) < 0:
            round_index = rounds_by_variant.get(str(row.get("variant_id")), 0)
            if round_index:
                negative_rounds.add(round_index)
    return negative_rounds


def _looks_like_failure_reflection(text: str) -> bool:
    normalized = text.lower()
    return len(normalized.strip()) >= 20 and any(term in normalized for term in ["mean_edge", "failure", "negative", "flat", "score"])


def _literature_requirement_satisfied(steps: list[dict[str, object]]) -> bool:
    required_steps = [step for step in steps if bool(step.get("literature_required"))]
    if not required_steps:
        return True
    for step in required_steps:
        actions = step.get("actions") if isinstance(step.get("actions"), list) else []
        if not any(isinstance(action, dict) and action.get("tool") == "fetch_literature" and action.get("status") == "completed" for action in actions):
            return False
    return True
