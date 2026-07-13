from __future__ import annotations

from .types import EvalSuite, EvalTask


SUITE_CHOICES = ("core", "edge", "preflight", "all")


def default_eval_suite() -> EvalSuite:
    return EvalSuite(
        id="core",
        name="Single-Loop Research Evaluation Suite",
        description="Production ResearchAgent-loop evals. Every factual result must be grounded in fetched document evidence; results are compared across the configured model/seed matrix.",
        tasks=[
            EvalTask(
                id="single_loop_document_grounding",
                name="ResearchAgent turns leads into document-grounded claims",
                prompt="Research how multi-agent systems improve automated literature review quality. Search for sources, fetch the primary documents you rely on, and cite each factual claim inline.",
                task_mode="research",
                max_iterations=6,
                success_criteria=[
                    "The production ResearchAgent loop completes",
                    "Discovery leads are not used as final grounding",
                    "Each substantive claim has a fetched-document citation with measurable support",
                    "Claim artifacts retain page- or section-level locators",
                ],
                grader_ids=[
                    "outcome_completed",
                    "run_actions_executed",
                    "research_groundedness",
                    "report_no_fabricated_sources",
                    "prompt_output_relevance",
                    "artifact_report",
                    "transcript_progress",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
            ),
            EvalTask(
                id="single_loop_claim_coverage",
                name="ResearchAgent cannot pass with snippets or URL-only grounding",
                prompt="Research transformer architecture efficiency. Fetch the primary documents before answering; every factual claim needs an inline citation to the fetched document.",
                task_mode="research",
                max_iterations=6,
                success_criteria=[
                    "A bare search URL does not pass",
                    "The final report has full claim citation coverage",
                    "The final answer is relevant to the original question",
                ],
                grader_ids=[
                    "outcome_completed",
                    "run_actions_executed",
                    "research_groundedness",
                    "report_no_fabricated_sources",
                    "prompt_output_relevance",
                    "transcript_progress",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
            ),
            EvalTask(
                id="single_loop_matrix_isolation",
                name="Model and seed cells remain isolated",
                prompt="Research a small technical question and return one document-grounded finding.",
                task_mode="research",
                max_iterations=4,
                trials=1,
                success_criteria=[
                    "Each model/seed attempt uses the production loop",
                    "No output or temporary directory is shared between cells",
                    "Scores can be compared by model and seed",
                ],
                grader_ids=[
                    "outcome_completed",
                    "run_actions_executed",
                    "transcript_progress",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
            ),
        ],
    )


def edge_eval_suite() -> EvalSuite:
    return EvalSuite(
        id="edge",
        name="Edge-Case Harness Evaluation Suite",
        description=(
            "Regression evals for ambiguous or failure-prone agent-harness behavior: missing evaluators, "
            "prediction-market scorer status, trajectory shape, artifact containment, and trial isolation."
        ),
        tasks=[
            EvalTask(
                id="optimize_query_missing_evaluator_skips_optimizer",
                name="Optimize-query without evaluator records skip instead of fabricating optimization",
                prompt="Research optimization strategies for a tiny benchmark, but do not assume an evaluator exists",
                task_mode="optimize_query",
                success_criteria=[
                    "Run stays in optimize_query mode",
                    "Seed context records that no evaluator is available",
                    "Optimizer phase is skipped cleanly",
                    "No optimization artifacts are fabricated",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "seed_context",
                    "optimizer_skipped_without_evaluator",
                    "trajectory_modes",
                    "transcript_progress",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
                metadata={"required_modes": ["optimize_query"], "forbidden_modes": ["optimize"]},
            ),
            EvalTask(
                id="prediction_market_outputs_are_contained",
                name="Prediction-market generated strategies stay under run outputs",
                prompt=(
                    "Get to $10 profit in the prediction market challenge. Keep every generated strategy as a run artifact, "
                    "never as a source file."
                ),
                task_mode="optimize_query",
                evaluator_name="prediction_market",
                success_criteria=[
                    "Generated candidate strategies are inside outputs/<run>/candidates",
                    "No temporary strategy files are written into the repository source tree",
                    "The selected candidate is promoted to optimal_code.py",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "optimization_code_artifact",
                    "prediction_market_solution",
                    "prediction_market_artifact_containment",
                    "trajectory_modes",
                    "transcript_progress",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
                metadata={"required_modes": ["optimize_query", "optimize"], "candidate_glob": "candidates/*.py"},
            ),
            EvalTask(
                id="prediction_market_unmeasured_official_status",
                name="Prediction-market benchmark requires official scorer measurement",
                prompt="Evaluate a prediction-market challenge strategy without requiring the upstream scorer",
                task_mode="optimize_query",
                evaluator_name="prediction_market",
                success_criteria=[
                    "Fallback scoring is not accepted for benchmark success",
                    "optimization_result.json records official upstream measurement",
                    "score_source is upstream_orderbook_pm_challenge",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "prediction_market_official_status",
                    "optimization_code_artifact",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
            ),
            EvalTask(
                id="challenge_prediction_market_official_unavailable_records_unmeasured",
                name="Prediction-market unavailable official grader fails preflight",
                prompt="Run the prediction-market challenge locally when the upstream official scorer is not required",
                task_mode="optimize_query",
                evaluator_name="prediction_market",
                success_criteria=[
                    "The optimizer does not generate candidates without an official scorer",
                    "The run records a fatal official-evaluator preflight failure",
                    "No proxy score is promoted as benchmark success",
                ],
                grader_ids=[
                    "mode_selected",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
            ),
            EvalTask(
                id="challenge_prediction_market_candidate_files_only_in_outputs",
                name="Prediction-market candidate files are only output artifacts",
                prompt="Generate prediction-market challenge candidates, keeping every candidate inside the run output directory",
                task_mode="optimize_query",
                evaluator_name="prediction_market",
                success_criteria=[
                    "Candidate Python files are written to outputs/<run>/candidates",
                    "The winning candidate path points into that candidates directory",
                    "No generated candidate strategy is written into repository source locations",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "optimization_code_artifact",
                    "prediction_market_candidate_files_only_in_outputs",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
            ),
            EvalTask(
                id="parallel_trials_do_not_share_tmp_or_outputs",
                name="Multiple trials use distinct temp and output roots",
                prompt="Research a small deterministic fact about agent evaluation harnesses",
                task_mode="research",
                max_iterations=1,
                trials=2,
                success_criteria=[
                    "Each trial has a unique trial root",
                    "Each trial has a unique output root",
                    "Each trial has a unique TMPDIR",
                    "Each trial has exactly one run artifact directory",
                ],
                grader_ids=[
                    "outcome_completed",
                    "transcript_progress",
                    "isolation_clean_trial",
                    "parallel_trial_isolation",
                ],
                aggregation="binary",
            ),
            EvalTask(
                id="challenge_prediction_market_no_repo_root_strategy_files",
                name="Prediction-market run does not leak strategy files into repo root",
                prompt="Optimize the prediction-market challenge without creating temporary strategy files in the repository root",
                task_mode="optimize_query",
                evaluator_name="prediction_market",
                success_criteria=[
                    "No pm_strategy*.py files exist in the repository root",
                    "No tmp_pm*.py files exist in the repository root",
                    "Generated strategies remain run artifacts",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "optimization_code_artifact",
                    "no_repo_root_strategy_files",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
            ),
            EvalTask(
                id="research_should_not_oversearch",
                name="Simple research task stays inside a bounded search budget",
                prompt="Research who founded Apple and answer with a concise, sourced summary",
                task_mode="research",
                retriever="local",
                max_iterations=1,
                success_criteria=[
                    "The run completes with a report",
                    "The harness does not fan out unnecessary extra research rounds",
                    "Source, claim, and variant counts stay under the task budget",
                ],
                grader_ids=[
                    "outcome_completed",
                    "artifact_report",
                    "research_search_budget",
                    "transcript_progress",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
                metadata={
                    "max_sources": 8,
                    "max_claims": 24,
                    "max_query_evaluations": 4,
                    "max_evolution_rounds": 1,
                },
            ),
            EvalTask(
                id="nested_loop_multiple_iterations_no_regression",
                name="Nested optimization loop runs multiple rounds without score collapse",
                prompt="Optimize a tiny scoring function across multiple loop rounds and preserve the best candidate",
                task_mode="optimize",
                evaluator_name="length_score",
                max_iterations=4,
                success_criteria=[
                    "The optimizer runs multiple outer loop rounds",
                    "Round scores do not collapse after iteration",
                    "The selected output artifact is still emitted",
                    "A trajectory graph is written for inspection",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "multi_iteration_loop",
                    "loop_no_score_regression",
                    "optimization_code_artifact",
                    "trajectory_graph_artifact",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
                metadata={"min_rounds": 3, "max_score_drop": 0.2},
            ),
            EvalTask(
                id="trajectory_match_modes_are_enforced",
                name="Trajectory evaluators enforce strict, unordered, subset, and superset modes",
                prompt="Optimize a tiny scoring function across multiple rounds so the harness records a trajectory",
                task_mode="optimize",
                evaluator_name="length_score",
                max_iterations=4,
                success_criteria=[
                    "Normalized trajectory events are extracted from artifacts",
                    "Strict matching validates the expected canonical loop prefix",
                    "Unordered, subset, and superset matching all run against the same trajectory",
                    "Graph trajectory edges match the expected harness flow",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "trajectory_match_modes",
                    "graph_trajectory_match",
                    "trajectory_graph_artifact",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
                metadata={
                    "reference_trajectory": [
                        {"type": "router", "name": "optimize"},
                        {"type": "outer_loop", "name": "optimize"},
                        {"type": "inner_loop", "name": "optimize"},
                        {"type": "selection", "name": "variant"},
                        {"type": "outcome", "name": "completed"},
                    ],
                    "required_graph_edges": [
                        ["prompt", "router"],
                        ["router", "outer"],
                        ["outer", "inner"],
                        ["inner", "select"],
                        ["select", "agents"],
                        ["agents", "outcome"],
                    ],
                },
            ),
            EvalTask(
                id="stuck_loop_triggers_literature_search",
                name="Plateaued optimization triggers literature refresh",
                prompt=(
                    "Optimize a tiny scoring function. If the loop gets stuck or plateaus, check existing literature "
                    "before continuing to tweak variants."
                ),
                task_mode="optimize",
                evaluator_name="length_score",
                max_iterations=4,
                success_criteria=[
                    "The loop reaches a plateau or stuck signal",
                    "The harness records a literature refresh trigger",
                    "A literature-refresh source and claim are created",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "multi_iteration_loop",
                    "literature_refresh_on_stuck",
                    "trajectory_graph_artifact",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
                metadata={"min_rounds": 3},
            ),
            EvalTask(
                id="optimize_runs_start_with_literature_grounding",
                name="Optimize and challenge-style runs search literature before producing outputs",
                prompt=(
                    "Optimize a tiny scoring function. Use existing literature and benchmark failure modes before "
                    "deciding which variants to try."
                ),
                task_mode="optimize",
                evaluator_name="length_score",
                max_iterations=2,
                success_criteria=[
                    "The optimize harness records an initial literature-grounding step",
                    "Retrieved grounding sources and claims are stored",
                    "The optimize output artifact is still emitted",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "literature_grounding_present",
                    "optimization_code_artifact",
                    "trajectory_graph_artifact",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
            ),
        ],
    )


def all_eval_suite() -> EvalSuite:
    core = default_eval_suite()
    edge = edge_eval_suite()
    preflight = preflight_eval_suite()
    return EvalSuite(
        id="all",
        name="All Harness Evaluation Suites",
        description=f"{core.description} {edge.description} {preflight.description}",
        tasks=core.tasks + edge.tasks + preflight.tasks,
        trials_per_task=core.trials_per_task,
    )


def preflight_eval_suite() -> EvalSuite:
    return EvalSuite(
        id="preflight",
        name="Preflight Regression Evaluation Suite",
        description=(
            "Fast sentinel evals for behavior that should not regress before autore runs: "
            "tool/source diversity, deterministic routing, trajectory shape, and artifact contracts."
        ),
        tasks=[
            EvalTask(
                id="research_uses_at_least_four_source_families",
                name="Research calls at least four source/tool families",
                prompt="Research recent advances in transformer architecture efficiency for large language models",
                task_mode="research",
                retriever="auto",
                max_iterations=2,
                success_criteria=[
                    "Run completes",
                    "Research calls at least four distinct source/tool families",
                    "Sources and claims are persisted",
                    "Report cites retained sources without fabricated URLs",
                ],
                grader_ids=[
                    "outcome_completed",
                    "research_source_diversity",
                    "research_groundedness",
                    "report_no_fabricated_sources",
                    "prompt_output_relevance",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
                metadata={"min_distinct_source_families": 4},
            ),
            EvalTask(
                id="optimize_direct_preflight",
                name="Direct optimization still uses deterministic evaluator",
                prompt="Optimize a tiny scoring function",
                task_mode="optimize",
                evaluator_name="length_score",
                max_iterations=2,
                success_criteria=[
                    "Run routes to optimize",
                    "Optimizer scores variants with the deterministic evaluator",
                    "An optimization artifact is emitted",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "optimize_score",
                    "optimization_code_artifact",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
            ),
            EvalTask(
                id="trajectory_match_modes_preflight",
                name="Trajectory evaluators still enforce match modes",
                prompt="Optimize a tiny scoring function across multiple rounds so the harness records a trajectory",
                task_mode="optimize",
                evaluator_name="length_score",
                max_iterations=3,
                success_criteria=[
                    "Normalized trajectory events are extracted from artifacts",
                    "Strict, unordered, subset, and superset checks run against the trajectory",
                    "Graph trajectory edges match the expected harness flow",
                ],
                grader_ids=[
                    "outcome_completed",
                    "mode_selected",
                    "trajectory_match_modes",
                    "graph_trajectory_match",
                    "trajectory_graph_artifact",
                    "isolation_clean_trial",
                ],
                aggregation="binary",
                metadata={
                    "reference_trajectory": [
                        {"type": "router", "name": "optimize"},
                        {"type": "outer_loop", "name": "optimize"},
                        {"type": "inner_loop", "name": "optimize"},
                        {"type": "selection", "name": "variant"},
                        {"type": "outcome", "name": "completed"},
                    ],
                    "required_graph_edges": [
                        ["prompt", "router"],
                        ["router", "outer"],
                        ["outer", "inner"],
                        ["inner", "select"],
                        ["select", "agents"],
                        ["agents", "outcome"],
                    ],
                },
            ),
        ],
    )


def eval_suite_by_id(suite_id: str) -> EvalSuite:
    if suite_id == "core":
        return default_eval_suite()
    if suite_id == "edge":
        return edge_eval_suite()
    if suite_id == "preflight":
        return preflight_eval_suite()
    if suite_id == "all":
        return all_eval_suite()
    raise ValueError(f"Unknown eval suite: {suite_id}")


def select_eval_tasks(suite: EvalSuite, eval_ids: list[str]) -> EvalSuite:
    if not eval_ids:
        return suite
    requested = []
    for raw in eval_ids:
        requested.extend(part.strip() for part in raw.split(",") if part.strip())
    selected = [task for task in suite.tasks if task.id in requested]
    found = {task.id for task in selected}
    missing = [eval_id for eval_id in requested if eval_id not in found]
    if missing:
        available = ", ".join(task.id for task in suite.tasks)
        raise ValueError(f"Unknown eval id(s): {', '.join(missing)}. Available in {suite.id}: {available}")
    return EvalSuite(
        id=suite.id if not requested else f"{suite.id}_selected",
        name=suite.name if not requested else f"{suite.name} (selected)",
        description=suite.description,
        tasks=selected,
        trials_per_task=suite.trials_per_task,
    )
