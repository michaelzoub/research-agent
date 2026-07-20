# Future Plans

The production runtime is now one model-directed loop. These plans extend current capabilities without reintroducing a second execution architecture or a mandatory research sequence.

## Completed foundations

- [x] One production `ResearchAgent` path with provider-native tool calls.
- [x] Async tool registry with concurrent read-only execution.
- [x] Append-only model/tool event log and durable failure records.
- [x] Final-answer validation, partial-result termination, and output-limit continuation.
- [x] Workspace/SSRF/sandbox boundaries and compact source context.
- [x] Retrieval safeguards for DuckDuckGo blocking and arXiv relevance/exact IDs.
- [x] Grader context compaction with exact strategy/score retention and append-only audit history.
- [x] Canonical fetched-document cache with retained evidence extracts and locators.
- [x] Optional verified-document structured extraction, grounded analysis, and deterministic SVG dataset visualization with persisted provenance.

## Next: document and source quality

- [ ] Persist fetched documents as hashed sectioned records, rather than only returning a bounded body.
- [ ] Add `inspect_document_metadata`, `read_document_section`, and `search_within_document` tools.
- [ ] Add source-quality and diversity checks to final validation without prescribing which sources the model must use.
- [ ] Add tested fallbacks for search providers that require keys or block automated requests.
- [ ] Record renderer provenance when curl.md is used and make the renderer selectable by policy.

## Next: budgets and observability

- [ ] Expose CLI cost/runtime/tool-call caps and show the remaining budget in compact tool observations.
- [ ] Add a run viewer that reads `agent_events.jsonl`, `failed_paths.json`, and cost events directly.
- [ ] Add redaction rules for sensitive tool arguments and observations before persistence.
- [ ] Add retry policy metadata per tool and verify retryable failures can be recovered by the model.

## Next: controlled experiments

- [ ] Expose `inspect_evaluator`, `propose_experiment`, `submit_experiment`, `inspect_experiment_result`, and `compare_candidates` as tools backed by `ExperimentSystem`.
- [ ] Keep evaluation, worker leasing, reproducibility records, and compare-and-swap promotion deterministic and outside model control.
- [ ] Add experiment-tool integration tests proving a model cannot self-promote a candidate.

## Next: cross-run learning

- [ ] Persist compact trajectory summaries: task characteristics, tools, failures, costs, source quality, answer validation, and experiment outcomes.
- [ ] Add a `search_prior_trajectories` tool so the model may consult relevant history without being forced into past sequences.
- [ ] Add explicit user-feedback capture and link it to prior run summaries.

## Long-running research

- [ ] Add resumable runs with explicit user checkpoints and cancellation semantics.
- [ ] Generalize the grader working-state projector to non-grader research and experiment trajectories.
- [ ] Optionally add model-assisted compression for long literature prose only; keep code, scores, identifiers, and locators deterministic.
- [ ] Add independent replay tests for native tool-call transcripts and deterministic experiment artifacts.
