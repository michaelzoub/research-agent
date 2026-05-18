from __future__ import annotations

import json
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
