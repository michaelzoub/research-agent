from __future__ import annotations

import json

from ...store import ArtifactStore
from ..types import EvalTask, GraderResult
from .common import _result


def _grade_optimize_score(task: EvalTask, store: ArtifactStore) -> GraderResult:
    optimize_evals = [row for row in store.list("variant_evaluations") if row.get("inner_loop") == "optimize"]
    best = max((float(row.get("score", 0.0)) for row in optimize_evals), default=0.0)
    passed = best > 0.0
    return _result("optimize_score", "code", "outcome verification", best, passed, 1.0, f"Best optimize score={best:.3f}.", [{"best_score": best, "passed": passed}])


def _grade_optimization_code_artifact(task: EvalTask, store: ArtifactStore) -> GraderResult:
    text = store.optimal_code_path.read_text(encoding="utf-8") if store.optimal_code_path.exists() else ""
    result = json.loads(store.optimization_result_path.read_text(encoding="utf-8")) if store.optimization_result_path.exists() else {}
    checks = [
        ("optimal_code_exists", store.optimal_code_path.exists()),
        ("optimal_code_nonempty", len(text.strip()) > 40),
        ("optimization_result_exists", store.optimization_result_path.exists()),
        ("optimization_result_points_to_optimal_code", result.get("optimal_code_path") == str(store.optimal_code_path)),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "optimization_code_artifact",
        "code",
        "artifact contract",
        score,
        passed,
        1.0,
        "Optimization run emitted the universal optimal_code.py artifact.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_seed_context(task: EvalTask, store: ArtifactStore) -> GraderResult:
    exists = store.optimizer_seed_context_path.exists()
    payload = json.loads(store.optimizer_seed_context_path.read_text(encoding="utf-8")) if exists else {}
    top = payload.get("top_query_findings", [])
    passed = exists and bool(top)
    return _result("seed_context", "code", "artifact existence", 1.0 if passed else 0.0, passed, 1.0, "Optimizer seed context checked.", [{"exists": exists, "top_query_findings": len(top) if isinstance(top, list) else 0, "passed": passed}])


def _grade_optimize_query_phases(task: EvalTask, store: ArtifactStore) -> GraderResult:
    loops = {row.get("inner_loop") for row in store.list("variant_evaluations")}
    passed = {"optimize_query", "optimize"}.issubset(loops)
    return _result("optimize_query_phases", "code", "trace/phase verification", 1.0 if passed else 0.0, passed, 1.0, f"Inner loops seen: {sorted(str(loop) for loop in loops)}.", [{"loops": sorted(str(loop) for loop in loops), "passed": passed}])


def _grade_optimizer_skipped_without_evaluator(task: EvalTask, store: ArtifactStore) -> GraderResult:
    seed_context = json.loads(store.optimizer_seed_context_path.read_text(encoding="utf-8")) if store.optimizer_seed_context_path.exists() else {}
    loops = {row.get("inner_loop") for row in store.list("variant_evaluations")}
    progress = store.progress_path.read_text(encoding="utf-8") if store.progress_path.exists() else ""
    checks = [
        ("seed_context_exists", store.optimizer_seed_context_path.exists()),
        ("seed_context_has_no_evaluator", seed_context.get("has_evaluator") is False),
        ("no_optimize_inner_loop", "optimize" not in loops),
        ("skip_recorded_in_progress", "Optimizer phase skipped" in progress),
        ("no_optimization_result_fabricated", not store.optimization_result_path.exists()),
        ("no_optimal_code_fabricated", not store.optimal_code_path.exists()),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "optimizer_skipped_without_evaluator",
        "code",
        "negative-path outcome verification",
        score,
        passed,
        1.0,
        f"Inner loops seen: {sorted(str(loop) for loop in loops)}; has_evaluator={seed_context.get('has_evaluator')}.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )


def _grade_research_search_budget(task: EvalTask, store: ArtifactStore) -> GraderResult:
    max_sources = int(task.metadata.get("max_sources", 8))
    max_claims = int(task.metadata.get("max_claims", 24))
    max_query_evaluations = int(task.metadata.get("max_query_evaluations", 4))
    max_evolution_rounds = int(task.metadata.get("max_evolution_rounds", 1))
    sources = store.list("sources")
    claims = store.list("claims")
    query_evaluations = [
        row for row in store.list("variant_evaluations")
        if row.get("inner_loop") in {"research", "optimize_query"}
    ]
    rounds = store.list("evolution_rounds")
    checks = [
        ("sources_under_budget", len(sources) <= max_sources),
        ("claims_under_budget", len(claims) <= max_claims),
        ("query_evaluations_under_budget", len(query_evaluations) <= max_query_evaluations),
        ("evolution_rounds_under_budget", len(rounds) <= max_evolution_rounds),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "research_search_budget",
        "code",
        "search budget verification",
        score,
        passed,
        1.0,
        (
            f"sources={len(sources)}/{max_sources}, claims={len(claims)}/{max_claims}, "
            f"query_evaluations={len(query_evaluations)}/{max_query_evaluations}, rounds={len(rounds)}/{max_evolution_rounds}."
        ),
        [
            {"check": "sources_under_budget", "actual": len(sources), "max": max_sources, "passed": len(sources) <= max_sources},
            {"check": "claims_under_budget", "actual": len(claims), "max": max_claims, "passed": len(claims) <= max_claims},
            {
                "check": "query_evaluations_under_budget",
                "actual": len(query_evaluations),
                "max": max_query_evaluations,
                "passed": len(query_evaluations) <= max_query_evaluations,
            },
            {
                "check": "evolution_rounds_under_budget",
                "actual": len(rounds),
                "max": max_evolution_rounds,
                "passed": len(rounds) <= max_evolution_rounds,
            },
        ],
    )


def _grade_plateau_entropy_exploration(task: EvalTask, store: ArtifactStore) -> GraderResult:
    rounds = store.list("evolution_rounds")
    plateau_rounds = [
        row for row in rounds
        if row.get("termination_signal") in {"score_plateau", "coverage_plateau"}
    ]
    if not plateau_rounds:
        return _result(
            "plateau_entropy_exploration",
            "code",
            "anti-reward-hack plateau exploration",
            1.0,
            True,
            1.0,
            "No plateau signal was observed, so entropy recovery was not required.",
            [{"check": "plateau_observed", "passed": True, "required": False}],
        )

    sources = store.list("sources")
    claims = store.list("claims")
    variants = store.list("variants")
    progress = store.progress_path.read_text(encoding="utf-8") if store.progress_path.exists() else ""
    recovery_sources = [
        source for source in sources
        if str(source.get("url", "")).startswith("memory://plateau-recovery/")
        and str(source.get("summary", "")).lower().find("expected to improve generalization") >= 0
    ]
    recovery_claims = [
        claim for claim in claims
        if claim.get("created_by_agent") == "plateau_recovery_policy"
        and "expected to improve generalization" in str(claim.get("text", "")).lower()
    ]
    intents = []
    for variant in variants:
        metadata = variant.get("metadata") if isinstance(variant.get("metadata"), dict) else {}
        intent = metadata.get("meaningful_entropy_intent") if isinstance(metadata, dict) else None
        if isinstance(intent, dict):
            intents.append(intent)
    meaningful_actions = {"uncertainty_axis", "literature_mechanism", "alternative_evaluator", "fresh_search_context"}
    actions = {str(intent.get("action", "")) for intent in intents}
    has_expected_generalization = all(
        str(intent.get("expected_generalization", "")).strip()
        and str(intent.get("exploration_path", "")).strip()
        for intent in intents
    )
    hyperparameter_only = bool(actions) and actions <= {"boost_temperature", "random_mutation", "context_derived_numeric_mutation"}
    checks = [
        ("plateau_signal_observed", bool(plateau_rounds)),
        ("progress_records_plateau_entropy", "Plateau entropy round" in progress),
        ("recovery_source_records_generalization_reason", bool(recovery_sources)),
        ("recovery_claim_records_generalization_reason", bool(recovery_claims)),
        ("variants_carry_entropy_intent", bool(intents)),
        ("intent_names_meaningful_path", bool(actions & meaningful_actions)),
        ("intent_records_expected_generalization", bool(intents) and has_expected_generalization),
        ("not_hyperparameter_only", not hyperparameter_only),
    ]
    score = sum(1 for _, passed in checks if passed) / len(checks)
    passed = score == 1.0
    return _result(
        "plateau_entropy_exploration",
        "code",
        "anti-reward-hack plateau exploration",
        score,
        passed,
        1.0,
        f"plateau_rounds={len(plateau_rounds)}; entropy_actions={sorted(actions)}; recovery_sources={len(recovery_sources)}.",
        [{"check": name, "passed": passed} for name, passed in checks],
    )
