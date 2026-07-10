from __future__ import annotations

import json
import re
from pathlib import Path

from ...store import ArtifactStore
from ..trajectory import outcome_from_store
from ..types import EvalTask, GraderResult
from .common import _result


def _grade_outcome_completed(task: EvalTask, store: ArtifactStore) -> GraderResult:
    status = outcome_from_store(store).get("status")
    passed = status == "completed"
    return _result("outcome_completed", "code", "outcome verification", 1.0 if passed else 0.0, passed, 1.0, f"Run status is {status}.", [{"status": status, "expected": "completed", "passed": passed}])


def _grade_run_actions_executed(task: EvalTask, store: ArtifactStore) -> GraderResult:
    run_state = json.loads(store.run_state_path.read_text(encoding="utf-8")) if store.run_state_path.exists() else {}
    observed_actions = run_state.get("observed_actions", []) if isinstance(run_state, dict) else []
    loop_tasks = {row.get("id"): row for row in store.list("loop_tasks")}
    iterations_by_task = {}
    for iteration in store.list("loop_iterations"):
        iterations_by_task.setdefault(iteration.get("task_id"), []).append(iteration)
    task_checks = []
    for item in observed_actions:
        source_task_id = item.get("source_task_id")
        loop_task = loop_tasks.get(source_task_id)
        iterations = iterations_by_task.get(source_task_id, [])
        status_matches = bool(loop_task) and item.get("status") == loop_task.get("status")
        task_checks.append(
            {
                "action_id": item.get("id"),
                "source_task_id": source_task_id,
                "has_loop_task": bool(loop_task),
                "has_iteration": bool(iterations),
                "status_matches": status_matches,
                "passed": bool(loop_task) and bool(iterations) and status_matches,
            }
        )
    checks = [
        ("run_state_exists", store.run_state_path.exists()),
        ("run_state_has_actions", bool(observed_actions)),
        ("all_actions_have_loop_tasks", all(check["has_loop_task"] for check in task_checks)),
        ("all_actions_have_iterations", all(check["has_iteration"] for check in task_checks)),
        ("all_action_statuses_match", all(check["status_matches"] for check in task_checks)),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "run_actions_executed",
        "code",
        "run action execution verification",
        score,
        passed,
        1.0,
        f"Verified {sum(1 for check in task_checks if check['passed'])}/{len(task_checks)} observed action(s) against loop tasks and iterations.",
        [{"check": name, "passed": passed} for name, passed in checks] + task_checks,
    )


def _grade_run_actions_executed_deterministic(task: EvalTask, store: ArtifactStore) -> GraderResult:
    run_state = json.loads(store.run_state_path.read_text(encoding="utf-8")) if store.run_state_path.exists() else {}
    observed_actions = run_state.get("observed_actions", []) if isinstance(run_state, dict) else []
    loop_tasks = {row.get("id"): row for row in store.list("loop_tasks")}
    iterations_by_task: dict[str, list[dict[str, Any]]] = {}
    for iteration in store.list("loop_iterations"):
        iterations_by_task.setdefault(str(iteration.get("task_id")), []).append(iteration)

    task_assertions: list[dict[str, Any]] = []
    for item in observed_actions:
        source_task_id = str(item.get("source_task_id") or "")
        loop_task = loop_tasks.get(source_task_id)
        iterations = iterations_by_task.get(source_task_id, [])
        terminal_status = str(item.get("status")) in {"passed", "failed", "skipped"}
        iteration_terminal = any(str(iteration.get("status")) in {"passed", "failed", "skipped", "completed"} for iteration in iterations)
        attempted = bool(loop_task) and int(loop_task.get("attempts") or 0) > 0
        evidence = bool(iterations) and bool(loop_task) and bool(loop_task.get("result_summary") or loop_task.get("last_error"))
        passed = bool(loop_task) and terminal_status and attempted and iteration_terminal and evidence
        task_assertions.append(
            {
                "action_id": item.get("id"),
                "source_task_id": source_task_id,
                "title": item.get("title"),
                "has_loop_task": bool(loop_task),
                "terminal_action_status": terminal_status,
                "attempted": attempted,
                "has_terminal_iteration": iteration_terminal,
                "has_result_or_error_evidence": evidence,
                "passed": passed,
            }
        )

    global_checks = [
        ("run_state_exists", store.run_state_path.exists()),
        ("run_state_has_actions", bool(observed_actions)),
        ("no_pending_actions", bool(observed_actions) and all(str(item.get("status")) in {"passed", "failed", "skipped"} for item in observed_actions)),
        ("every_task_attempted_with_evidence", bool(task_assertions) and all(item["passed"] for item in task_assertions)),
    ]
    score = sum(1 for _, passed in global_checks if passed) / len(global_checks)
    passed = score == 1.0
    return _result(
        "run_actions_executed_deterministic",
        "code",
        "deterministic action execution verification",
        score,
        passed,
        1.0,
        f"Deterministically verified {sum(1 for item in task_assertions if item['passed'])}/{len(task_assertions)} action(s) reached terminal execution with evidence.",
        [{"check": name, "passed": passed} for name, passed in global_checks] + task_assertions,
    )


def _grade_mode_selected(task: EvalTask, store: ArtifactStore) -> GraderResult:
    decisions = store.list("task_ingestion_decisions")
    selected = decisions[0].get("selected_mode") if decisions else None
    passed = selected == task.task_mode
    return _result("mode_selected", "code", "tool/output verification", 1.0 if passed else 0.0, passed, 1.0, f"Selected mode {selected}; expected {task.task_mode}.", [{"selected_mode": selected, "expected": task.task_mode, "passed": passed}])


def _grade_report_artifact(task: EvalTask, store: ArtifactStore) -> GraderResult:
    md_ok = store.report_path.exists() and len(store.report_path.read_text(encoding="utf-8").strip()) > 80
    pdf_path = getattr(store, "report_pdf_path", store.root / "final_report.pdf")
    tex_path = getattr(store, "report_tex_path", store.root / "final_report.tex")
    preview_path = getattr(store, "report_preview_path", store.root / "final_report_preview.png")
    pdf_ok = pdf_path.exists() and pdf_path.stat().st_size > 100
    tex_ok = tex_path.exists() and tex_path.stat().st_size > 100
    preview_ok = preview_path.exists() and preview_path.stat().st_size > 100
    passed = md_ok and pdf_ok and tex_ok and preview_ok
    score = sum([md_ok, pdf_ok, tex_ok, preview_ok]) / 4
    return _result(
        "artifact_report",
        "code",
        "artifact existence",
        score,
        passed,
        0.75,
        "Final Markdown, TeX, and PDF report artifacts checked.",
        [
            {"path": str(store.report_path), "artifact": "markdown", "passed": md_ok},
            {"path": str(tex_path), "artifact": "latex", "passed": tex_ok},
            {"path": str(pdf_path), "artifact": "pdf", "passed": pdf_ok},
            {"path": str(preview_path), "artifact": "preview_png", "passed": preview_ok},
        ],
    )


def _grade_transcript_progress(task: EvalTask, store: ArtifactStore) -> GraderResult:
    progress = store.progress_path.read_text(encoding="utf-8") if store.progress_path.exists() else ""
    traces = store.list("agent_traces")
    has_complete = "<promise>COMPLETE</promise>" in progress
    has_incomplete_stop = "Stopped with" in progress and "incomplete loop tasks" in progress
    has_steps = "Task 1:" in progress and len(progress.splitlines()) >= 5
    passed = (has_complete or has_incomplete_stop) and has_steps
    return _result(
        "transcript_progress",
        "code",
        "transcript analysis",
        1.0 if passed else 0.0,
        passed,
        0.75,
        f"Progress lines={len(progress.splitlines())}; traces={len(traces)}.",
        [
            {"check": "complete_or_incomplete_stop_marker", "passed": has_complete or has_incomplete_stop},
            {"check": "step_visibility", "passed": has_steps},
        ],
    )


def _grade_human_placeholder(task: EvalTask, store: ArtifactStore) -> GraderResult:
    return _result(
        "human_spot_check_placeholder",
        "human",
        "spot-check sampling",
        0.0,
        False,
        0.0,
        "Human grader slot recorded but not required in automated local suite.",
        [{"requires_human": True, "passed": False}],
    )


def _grade_isolation_clean_trial(task: EvalTask, store: ArtifactStore) -> GraderResult:
    trial_root = store.root.parent
    run_dirs = [path for path in trial_root.iterdir() if path.is_dir() and _is_run_dir(path.name)]
    has_trial_tmp = (trial_root / "tmp").exists()
    passed = len(run_dirs) == 1 and has_trial_tmp and str(store.root).startswith(str(trial_root))
    return _result(
        "isolation_clean_trial",
        "code",
        "environment isolation check",
        1.0 if passed else 0.0,
        passed,
        1.0,
        f"Trial root {trial_root} contains {len(run_dirs)} run artifact directory/directories.",
        [
            {"check": "single_run_directory", "actual": len(run_dirs), "expected": 1, "passed": len(run_dirs) == 1},
            {"check": "per_trial_tmpdir", "path": str(trial_root / "tmp"), "passed": has_trial_tmp},
            {"check": "store_under_trial_root", "path": str(store.root), "passed": str(store.root).startswith(str(trial_root))},
        ],
    )


def _is_run_dir(name: str) -> bool:
    return name.startswith("run_") or bool(re.match(r"^\d+_run_", name))
