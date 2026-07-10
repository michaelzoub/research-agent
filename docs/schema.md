# Artifact Schema

The harness keeps append-friendly JSON files for inspectability and mirrors the
same records into `world_model.sqlite` in the output root. SQLite migrations live
under `research_harness/migrations/` and maintain cross-run dedupe, provenance,
and observability indexes behind `ArtifactStore`.

## sources

- `id`
- `url`
- `title`
- `author`
- `date`
- `source_type`
- `retrieved_at`
- `summary`
- `relevance_score`
- `credibility_score`
- `canonical_id`
- `duplicate_of`

## claims

- `id`
- `text`
- `source_ids`
- `confidence`
- `support_level`
- `contradicted_by`
- `created_by_agent`
- `run_id`
- `canonical_id`
- `duplicate_of`

## hypotheses

- `id`
- `text`
- `supporting_claim_ids`
- `contradicting_claim_ids`
- `confidence`
- `novelty_score`
- `testability_score`
- `next_experiment`
- `run_id`
- `canonical_id`
- `duplicate_of`

## experiments

- `id`
- `description`
- `hypothesis_id`
- `expected_signal`
- `priority`

## open_questions

- `id`
- `question`
- `priority`
- `reason`
- `created_by_agent`
- `status`

## contradictions

- `id`
- `claim_a`
- `claim_b`
- `explanation`
- `severity`
- `resolution_status`

## failed_paths

- `id`
- `description`
- `reason`
- `created_by_agent`
- `run_id`
- `failure_category`
- `failure_component`
- `retryable`
- `severity`

## harness_changes

- `id`
- `change`
- `reason`
- `expected_effect`
- `risk`
- `evaluation`
- `status`
- `created_at`
- `run_id`
- `component`
- `diagnosis`
- `risk_score`
- `expected_value_score`
- `priority_score`
- `trace_pattern_delta`

## runs

`task_mode` and `product_agent` remain nullable compatibility fields for
historical deterministic utility artifacts. A production model-directed run does
not select either field; its authoritative trajectory is `agent_events.jsonl`
and `run_state.json`.

- `id`
- `user_goal`
- `task_type`
- `task_mode`
- `product_agent`
- `started_at`
- `completed_at`
- `status`
- `total_cost`
- `total_tokens`
- `harness_config_id`
- `prompt_versions`
- `harness_config_snapshot`

## provenance_edges

- `id`
- `run_id`
- `from_type`
- `from_id`
- `to_type`
- `to_id`
- `relationship`
- `metadata`
- `created_at`

## cost_events

- `id`
- `run_id`
- `component`
- `provider`
- `model`
- `prompt_tokens`
- `completion_tokens`
- `cost_usd`
- `call_type`
- `metadata`
- `created_at`

## harness_diagnoses

- `id`
- `run_id`
- `artifact_yield`
- `components`
- `failure_taxonomy`
- `localized_components`
- `prior_run_comparison`
- `score_patterns`
- `trace_patterns`

## task_ingestion_decisions

- `id`
- `requested_mode`
- `selected_mode`
- `product_agent`
- `evaluator_name`
- `reason`

## Optimization artifacts

Optimization and challenge runs that execute an evaluator write:

- `optimized_candidate.txt`: exact selected best candidate payload
- `optimal_code.py`: universal code artifact for the selected best candidate
- `optimization_result.json`: score, evaluator metadata, candidate path, and `optimal_code_path`
- `solution.py`: optional challenge-specific mirror for upstream runners

## agent_traces

- `id`
- `run_id`
- `agent_name`
- `role`
- `prompt`
- `model`
- `tools_used`
- `tool_calls`
- `token_usage`
- `runtime_ms`
- `status`
- `errors`
- `output_summary`
- `prompt_version`
- `prompt_tokens`
- `completion_tokens`
- `cost_usd`
- `failure_category`
- `failure_component`
- `retryable`

## SQLite world model

`world_model.sqlite` is stored next to run directories and contains:

- `schema_migrations`: applied migration versions.
- `artifacts`: JSON payload mirror keyed by `(entity, id)` with `run_id`, `canonical_key`, and `duplicate_of`.
- `provenance_edges`: queryable provenance graph.
- `run_observability`: per-run prompt versions, harness config snapshots, and cost JSON.
