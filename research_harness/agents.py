from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import subprocess
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from .llm import LLMClient
from .diagnostics import diagnose_snapshot, score_harness_change
from .schemas import (
    AgentBudget,
    AgentTrace,
    Claim,
    Contradiction,
    Experiment,
    FailedPath,
    HarnessChange,
    Hypothesis,
    OpenQuestion,
    RunRecord,
    now_iso,
    to_dict,
)
from .search import SearchBackend
from .store import ArtifactStore


class Agent(Protocol):
    name: str
    role: str
    prompt_template: str
    budget: AgentBudget

    async def run(self, run: RunRecord, store: ArtifactStore) -> str: ...


@dataclass
class AgentResult:
    agent_name: str
    summary: str
    errors: list[str]


class BaseAgent:
    def __init__(
        self,
        name: str,
        role: str,
        prompt_template: str,
        budget: Optional[AgentBudget] = None,
        llm: Optional[LLMClient] = None,
        model: Optional[str] = None,
    ):
        self.name = name
        self.role = role
        self.prompt_template = prompt_template
        self.budget = budget or AgentBudget()
        self.llm = llm or LLMClient()
        self.model = model or self.llm.model_label
        self.tool_calls: list[dict[str, object]] = []
        self.tools_used: list[str] = []

    async def execute(self, run: RunRecord, store: ArtifactStore) -> AgentResult:
        started = time.perf_counter()
        started_at = now_iso()
        errors: list[str] = []
        status = "completed"
        summary = ""
        store.append_progress(f"Agent start: {self.name} ({self.role}) using {self.llm.model_label}")
        # Snapshot LLM token counters before this agent runs so we can compute
        # the delta for this specific agent call.
        tokens_before = self.llm.total_prompt_tokens + self.llm.total_completion_tokens
        prompt_tokens_before = self.llm.total_prompt_tokens
        completion_tokens_before = self.llm.total_completion_tokens
        cost_before = self.llm.total_cost()
        try:
            if self.budget.cancelled:
                status = "cancelled"
                summary = "Agent cancelled before execution."
            else:
                summary = await asyncio.wait_for(
                    self.run(run, store),
                    timeout=self.budget.max_runtime_seconds,
                )
        except Exception as exc:  # pragma: no cover - defensive trace path
            status = "failed"
            errors.append(f"{type(exc).__name__}: {exc}")
            summary = "Agent failed; see errors."
        runtime_ms = int((time.perf_counter() - started) * 1000)
        tokens_after = self.llm.total_prompt_tokens + self.llm.total_completion_tokens
        agent_tokens = tokens_after - tokens_before
        agent_prompt_tokens = self.llm.total_prompt_tokens - prompt_tokens_before
        agent_completion_tokens = self.llm.total_completion_tokens - completion_tokens_before
        agent_cost = max(0.0, self.llm.total_cost() - cost_before)
        rendered_prompt = self.prompt_template.replace("{goal}", run.user_goal)
        store.append_progress(f"Agent {status}: {self.name} in {runtime_ms}ms tokens={agent_tokens} - {summary}")
        store.add_trace(
            AgentTrace(
                id=self.budget.trace_id,
                run_id=run.id,
                agent_name=self.name,
                role=self.role,
                prompt=rendered_prompt,
                model=self.model,
                tools_used=self.tools_used,
                tool_calls=self.tool_calls,
                token_usage=agent_tokens,
                runtime_ms=runtime_ms,
                status=status,
                errors=errors,
                output_summary=summary,
                started_at=started_at,
                prompt_version=_text_sha256(rendered_prompt),
                prompt_tokens=agent_prompt_tokens,
                completion_tokens=agent_completion_tokens,
                cost_usd=round(agent_cost, 6),
            )
        )
        return AgentResult(self.name, summary, errors)

    async def run(self, run: RunRecord, store: ArtifactStore) -> str:
        raise NotImplementedError


class LiteratureAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        query_angle: str,
        corpus: SearchBackend,
        prompt_template: str,
        budget: Optional[AgentBudget] = None,
        search_queries: Optional[list[str]] = None,
        llm: Optional[LLMClient] = None,
    ):
        super().__init__(name, "search_literature", prompt_template, budget, llm=llm)
        self.query_angle = query_angle
        self.corpus = corpus
        self.search_queries = search_queries or []

    async def run(self, run: RunRecord, store: ArtifactStore) -> str:
        self.tools_used.append(self.corpus.tool_name)
        queries = self.search_queries or [f"{run.user_goal} {self.query_angle}"]
        seen_urls: set[str] = set()
        all_results = []
        for query in queries:
            try:
                results = self.corpus.search(query, limit=self.budget.max_tool_calls)
            except Exception as exc:
                store.add_failed_path(
                    FailedPath(
                        description=f"Retriever failed for angle: {self.query_angle}",
                        reason=f"{self.corpus.tool_name} error: {type(exc).__name__}: {exc}",
                        created_by_agent=self.name,
                        run_id=run.id,
                    )
                )
                raise
            self.tool_calls.append({"tool": self.corpus.tool_name, "query": query, "results": len(results)})
            for document, relevance in results:
                if document.url in seen_urls:
                    continue
                seen_urls.add(document.url)
                all_results.append((document, relevance))
        all_results = _llm_filter_sources(self.llm, run.user_goal, self.query_angle, all_results)
        claim_count = 0
        for document, relevance in all_results:
            source = store.add_source(self.corpus.to_source(document, relevance))
            for text in document.claims[: self.budget.max_steps]:
                confidence = min(0.74, round((source.credibility_score * 0.7) + (relevance * 0.3), 2))
                store.add_claim(
                    Claim(
                        text=text,
                        source_ids=[source.id],
                        confidence=confidence,
                        support_level=_support_level(confidence),
                        created_by_agent=self.name,
                        run_id=run.id,
                    )
                )
                claim_count += 1
        if not all_results:
            store.add_failed_path(
                FailedPath(
                    description=f"No sources found for angle: {self.query_angle}",
                    reason=f"{self.corpus.tool_name} returned no matching documents.",
                    created_by_agent=self.name,
                    run_id=run.id,
                )
            )
        return (
            f"Retrieved {len(all_results)} sources from {len(queries)} queries and extracted "
            f"{claim_count} claims for angle '{self.query_angle}'."
        )


class HypothesisAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        hypothesis_angle: str,
        prompt_template: str,
        budget: Optional[AgentBudget] = None,
        llm: Optional[LLMClient] = None,
    ):
        super().__init__(name, "hypothesis_generation", prompt_template, budget, llm=llm)
        self.hypothesis_angle = hypothesis_angle

    async def run(self, run: RunRecord, store: ArtifactStore) -> str:
        claims = _dedupe_by_text(store.list("claims"))
        relevant = [
            claim
            for claim in claims
            if any(term in claim["text"].lower() for term in self.hypothesis_angle.lower().split())
        ] or claims[: self.budget.max_steps]
        created = 0
        for index, claim in enumerate(relevant[: self.budget.max_steps]):
            hypothesis = store.add_hypothesis(
                Hypothesis(
                    text=f"{self.hypothesis_angle.title()} path: {claim['text']}",
                    supporting_claim_ids=[claim["id"]],
                    contradicting_claim_ids=claim.get("contradicted_by", []),
                    confidence=min(0.68, max(0.2, round(float(claim["confidence"]) - 0.1, 2))),
                    novelty_score=round(0.55 + (index * 0.08), 2),
                    testability_score=0.72,
                    next_experiment=f"Search for direct evaluations of: {self.hypothesis_angle}.",
                )
            )
            store.add_experiment(
                Experiment(
                    description=hypothesis.next_experiment,
                    hypothesis_id=hypothesis.id,
                    expected_signal="Evidence that increases or decreases claim confidence.",
                    priority=max(1, 3 - index),
                )
            )
            created += 1
        if not claims:
            store.add_open_question(
                OpenQuestion(
                    question=f"What evidence exists for {self.hypothesis_angle}?",
                    priority=1,
                    reason="No claims were available before hypothesis generation.",
                    created_by_agent=self.name,
                )
            )
        return f"Generated {created} hypotheses for angle '{self.hypothesis_angle}'."


class CriticAgent(BaseAgent):
    async def run(self, run: RunRecord, store: ArtifactStore) -> str:
        claims = _dedupe_by_text(store.list("claims"))
        hypotheses = store.list("hypotheses")
        contradictions = 0
        max_contradictions = 50
        for left in claims:
            for right in claims:
                if left["id"] >= right["id"]:
                    continue
                if _looks_contradictory(left["text"], right["text"]):
                    if contradictions >= max_contradictions:
                        break
                    store.add_contradiction(
                        Contradiction(
                            claim_a=left["id"],
                            claim_b=right["id"],
                            explanation="Claims appear directionally opposed and need source-level resolution.",
                            severity="medium",
                        )
                    )
                    contradictions += 1
            if contradictions >= max_contradictions:
                break
        if contradictions >= max_contradictions:
            store.add_open_question(
                OpenQuestion(
                    question="Which contradiction clusters matter most for the final conclusion?",
                    priority=1,
                    reason="The critic hit the contradiction cap and needs clustering/reranking before deeper review.",
                    created_by_agent=self.name,
                )
            )
        low_support = [claim for claim in claims if claim["confidence"] < 0.55]
        for claim in low_support[:3]:
            store.add_open_question(
                OpenQuestion(
                    question=f"What stronger evidence supports or refutes this claim: {claim['text']}",
                    priority=2,
                    reason="Claim confidence is below the recommended synthesis threshold.",
                    created_by_agent=self.name,
                )
            )
        if not hypotheses:
            store.add_open_question(
                OpenQuestion(
                    question="Which evidence-backed directions should be explored next?",
                    priority=1,
                    reason="No hypotheses were generated from the current evidence set.",
                    created_by_agent=self.name,
                )
            )
        return f"Reviewed {len(claims)} claims and {len(hypotheses)} hypotheses; found {contradictions} contradictions."


class SynthesisAgent(BaseAgent):
    async def run(self, run: RunRecord, store: ArtifactStore) -> str:
        sources = _dedupe_by_id(store.list("sources"))
        claims = _dedupe_by_text(store.list("claims"))
        hypotheses = sorted(_dedupe_by_id(store.list("hypotheses")), key=lambda row: row["confidence"], reverse=True)
        contradictions = store.list("contradictions")
        questions = sorted(store.list("open_questions"), key=lambda row: row["priority"])
        seed_context = _read_optional_json(store.optimizer_seed_context_path)
        sources, claims, hypotheses, contradictions, questions = _filter_report_evidence(
            run,
            sources,
            claims,
            hypotheses,
            contradictions,
            questions,
            seed_context,
        )
        report = _build_report_with_llm(self.llm, run, sources, claims, hypotheses, contradictions, questions, seed_context)
        store.write_report(report)
        tex = _build_latex_report(run, sources, claims, hypotheses, contradictions, questions)
        tex_path = store.write_report_tex(tex)
        pdf_created = _render_report_pdf(store, tex_path)
        preview = _render_report_preview_png(store) or _report_preview_png(run, sources, claims, hypotheses, contradictions)
        store.write_report_preview(preview)
        pdf_note = f"; PDF: {store.report_pdf_path.name}" if pdf_created else ""
        return f"Synthesized final report with {len(sources)} sources, {len(claims)} claims, and {len(hypotheses)} hypotheses{pdf_note}."


class HarnessDebuggerAgent(BaseAgent):
    async def run(self, run: RunRecord, store: ArtifactStore) -> str:
        claims = store.list("claims")
        contradictions = store.list("contradictions")
        traces = store.list("agent_traces")
        sources = store.list("sources")
        variants = store.list("variants")
        evaluations = store.list("variant_evaluations")
        rounds = store.list("evolution_rounds")

        diagnosis = diagnose_snapshot(store.snapshot(), run_root=store.root)
        localized = diagnosis.get("localized_components") or []
        primary_component = str(localized[0].get("component")) if localized else "loop_control"

        product_agent = run.product_agent or "research"
        task_mode = run.task_mode or "research"

        if product_agent == "challenge" or task_mode == "optimize_query":
            change = _debug_change_challenge(run, claims, evaluations, rounds, traces, primary_component, diagnosis)
        elif product_agent == "optimize" or task_mode == "optimize":
            change = _debug_change_optimize(run, variants, evaluations, rounds, traces, primary_component, diagnosis)
        else:
            change = _debug_change_research(run, claims, contradictions, sources, traces, primary_component, diagnosis)

        scores = score_harness_change(to_dict(change), diagnosis)
        change.risk_score = scores["risk_score"]
        change.expected_value_score = scores["expected_value_score"]
        change.priority_score = scores["priority_score"]
        change.trace_pattern_delta = diagnosis.get("prior_run_comparison") or {}
        store.add_harness_change(change)
        store.write_harness_diagnosis(diagnosis)
        failures = [trace for trace in traces if trace["status"] != "completed"]
        return (
            f"Proposed 1 constrained harness change for {change.component}; "
            f"priority={change.priority_score:.3f}; observed {len(failures)} failed traces."
        )


def _llm_filter_sources(
    llm: LLMClient,
    goal: str,
    query_angle: str,
    candidates: list[tuple],
) -> list[tuple]:
    """Ask the LLM to keep only sources that directly address the goal and query angle.

    Falls back to returning all candidates unchanged when the LLM is offline or errors.
    """
    if not candidates or not llm.is_live:
        return candidates
    system = (
        "You are a research librarian selecting sources for a literature review. "
        "From the candidate list keep sources that directly address the goal and also sources that could challenge, complicate, or redirect the query angle. "
        "Return JSON only: {\"selected_urls\": [str]}. "
        "Prefer primary sources, empirical evidence, domain-specific findings, and well-grounded counterevidence. "
        "Exclude generic, off-topic, or low-quality sources."
    )
    user = json.dumps(
        {
            "goal": goal,
            "query_angle": query_angle,
            "candidates": [
                {"url": doc.url, "title": doc.title, "summary": doc.summary[:300]}
                for doc, _ in candidates
            ],
        },
        indent=2,
        sort_keys=True,
    )
    try:
        payload = llm.complete_json(system, user, max_output_tokens=400, temperature=0.45)
        selected_urls = {str(u) for u in payload.get("selected_urls", []) if u}
        if not selected_urls:
            return candidates
        filtered = [(doc, score) for doc, score in candidates if doc.url in selected_urls]
        return filtered if filtered else candidates
    except Exception:
        return candidates


def _debug_change_research(
    run: RunRecord,
    claims: list[dict],
    contradictions: list[dict],
    sources: list[dict],
    traces: list[dict],
    primary_component: str,
    diagnosis: dict,
) -> HarnessChange:
    base = {"run_id": run.id, "diagnosis": _diagnosis_summary(diagnosis)}
    if contradictions:
        return HarnessChange(
            change="Add an inline critic step after each literature batch to resolve contradictions early",
            reason="The run produced directionally opposed claims that were only reviewed at the end.",
            expected_effect="Fewer unresolved contradictions reaching synthesis; better follow-up search targeting",
            risk="More runtime and repeated critic work during exploratory runs",
            evaluation="Compare unresolved contradiction count and synthesis revision rate before and after",
            component="critic",
            **base,
        )
    low_confidence = [c for c in claims if float(c.get("confidence", 1.0)) < 0.55]
    if claims and len(low_confidence) > len(claims) * 0.5:
        return HarnessChange(
            change="Route retrieval toward higher-credibility academic sources when claim confidence is low",
            reason="Over half the extracted claims are below the synthesis confidence threshold, suggesting low-quality sources.",
            expected_effect="Higher average claim confidence and fewer low-support hypotheses",
            risk="Narrowing the source mix may miss relevant practitioner or grey-literature signals",
            evaluation="Compare mean claim confidence and high-support claim fraction before and after",
            component="retrieval",
            **base,
        )
    if len(claims) < 4:
        return HarnessChange(
            change="Increase search-agent query diversity by assigning orthogonal retrievers per query angle",
            reason="The run produced too few claims to support robust synthesis.",
            expected_effect="Higher recall and more diverse evidence from complementary sources",
            risk="More noisy sources and higher critic review burden",
            evaluation="Compare high-confidence claim count and critic objection rate",
            component="retrieval",
            **base,
        )
    if len(sources) > 14:
        return HarnessChange(
            change="Add a URL-level deduplication pass before claim extraction to reduce retrieval overlap",
            reason=f"The run retrieved {len(sources)} sources; deduplication would improve claim precision without reducing coverage.",
            expected_effect="Fewer duplicate claims and faster synthesis",
            risk="Aggressive deduplication may discard sources that differ only in framing or date",
            evaluation="Compare post-dedup unique claim count against current unique claim count",
            component="retrieval",
            **base,
        )
    failed = [t for t in traces if t.get("status") != "completed"]
    if failed:
        return HarnessChange(
            change="Add a retry-with-local-fallback policy for agents that fail on the first attempt",
            reason=f"{len(failed)} agent trace(s) did not complete; retrying with local-corpus fallback would recover partial runs.",
            expected_effect="Higher per-run completion rate and fewer missing artifacts",
            risk="Retries increase runtime and may mask transient retriever failures that need root-cause investigation",
            evaluation="Compare run completion rate and missing-artifact count with and without the retry policy",
            component=primary_component,
            **base,
        )
    return HarnessChange(
        change="Parallelize hypothesis generation across independent research angles",
        reason="Research produced adequate evidence but hypothesis generation is sequential; parallelizing would surface more diverse directions in the same time budget.",
        expected_effect="More hypothesis angles explored per run with no increase in total runtime",
        risk="Parallel hypothesis agents may produce redundant hypotheses when claims overlap",
        evaluation="Compare unique hypothesis angles per run and time-to-synthesis",
        component="hypothesis_generation",
        **base,
    )


def _debug_change_optimize(
    run: RunRecord,
    variants: list[dict],
    evaluations: list[dict],
    rounds: list[dict],
    traces: list[dict],
    primary_component: str,
    diagnosis: dict,
) -> HarnessChange:
    base = {"run_id": run.id, "diagnosis": _diagnosis_summary(diagnosis)}
    if not evaluations:
        return HarnessChange(
            change="Add a fast pre-screening step to filter obviously bad variants before full evaluation",
            reason="No variant evaluations were recorded, suggesting the evaluator may have been unavailable or all variants were trivially invalid.",
            expected_effect="Earlier rejection of poor candidates and lower wasted evaluator budget",
            risk="Pre-screening may incorrectly reject viable candidates if the filter is too aggressive",
            evaluation="Compare evaluator call count and proportion of low-scored variants reaching the full scorer",
            component="optimizer",
            **base,
        )
    scores = [float(e.get("score", 0.0)) for e in evaluations]
    best_score = max(scores, default=0.0)
    if best_score < 0.25:
        return HarnessChange(
            change="Seed the optimizer with literature-grounded initial variants instead of generic fallback seeds",
            reason=f"Best optimization score is only {best_score:.3f}; evidence-based seeds would accelerate convergence.",
            expected_effect="Higher first-round scores and faster convergence toward the objective",
            risk="Literature-derived seeds may be domain-narrow; a mixed seed set hedges that risk",
            evaluation="Compare round-1 best score and rounds-to-target with seeded vs. generic initialization",
            component="optimizer",
            **base,
        )
    score_spread = max(scores) - min(scores) if len(scores) > 1 else 0.0
    if score_spread < 0.05 and len(evaluations) >= 4:
        return HarnessChange(
            change="Apply random-mutation plateau recovery earlier to escape premature population convergence",
            reason=f"Score variance across {len(evaluations)} evaluations is only {score_spread:.3f}; the population has converged before reaching the objective.",
            expected_effect="More diverse variants in subsequent rounds and a higher chance of escaping local optima",
            risk="Random mutation may temporarily reduce best score before discovering better regions",
            evaluation="Compare score variance and plateau count before and after earlier recovery triggering",
            component="optimizer",
            **base,
        )
    plateau_rounds = [r for r in rounds if str(r.get("termination_signal", "")).endswith("_plateau")]
    if plateau_rounds:
        return HarnessChange(
            change="Rotate the primary retriever between plateau recovery rounds to inject novel strategy signals",
            reason=f"The optimization loop plateaued {len(plateau_rounds)} time(s); retriever rotation would expose different strategy evidence.",
            expected_effect="More diverse post-plateau variant proposals and lower repeat-plateau rate",
            risk="Retriever rotation adds per-round retrieval latency and may surface irrelevant signals",
            evaluation="Compare post-plateau score improvement rate with and without retriever rotation",
            component="loop_control",
            **base,
        )
    return HarnessChange(
        change="Add per-round score percentile tracking to detect silent regressions in the variant population",
        reason="Optimization is progressing but the harness only tracks the best variant per round; percentile tracking would catch gradual regressions.",
        expected_effect="Earlier detection of population-wide degradation and more informative stopping signals",
        risk="Percentile tracking adds bookkeeping overhead with marginal benefit when the population is small",
        evaluation="Compare stop-round decision accuracy with and without percentile-based stopping signals",
        component="optimizer",
        **base,
    )


def _debug_change_challenge(
    run: RunRecord,
    claims: list[dict],
    evaluations: list[dict],
    rounds: list[dict],
    traces: list[dict],
    primary_component: str,
    diagnosis: dict,
) -> HarnessChange:
    base = {"run_id": run.id, "diagnosis": _diagnosis_summary(diagnosis)}
    pm_terms = {"prediction", "market", "market-making", "orderbook", "spread", "inventory", "arbitrage", "edge", "retail"}
    pm_claims = [c for c in claims if any(term in str(c.get("text", "")).lower() for term in pm_terms)]
    if claims and len(pm_claims) < len(claims) * 0.3:
        return HarnessChange(
            change="Redirect retrieval to prompt-derived prediction-market sources and evaluator failure evidence",
            reason="Fewer than 30% of retrieved claims are relevant to prediction-market strategy; off-topic sources are diluting the optimizer seed context.",
            expected_effect="Higher PM-signal claim density and more targeted optimizer seed context",
            risk="Domain-narrow retrieval may miss useful adjacent signals from options or AMM literature",
            evaluation="Compare PM-relevant claim fraction and optimizer seed quality before and after the redirect",
            component="retrieval",
            **base,
        )
    opt_evals = [e for e in evaluations if e.get("inner_loop") == "optimize"]
    if opt_evals:
        edges = [float(e.get("metrics", {}).get("mean_edge", 0.0)) for e in opt_evals]
        best_edge = max(edges, default=0.0)
        if best_edge <= 0.0:
            return HarnessChange(
                change="Derive a broader candidate search from failed prediction-market evaluator traces",
                reason=f"Best mean edge across {len(opt_evals)} evaluated strategies is ≤ 0; generated strategies are underperforming the evaluator.",
                expected_effect="Positive edge by quoting outside the competitor ladder where only retail flow fills",
                risk="Over-broad exploration can reduce fill frequency or miss the evaluator-specific failure mode",
                evaluation="Compare mean edge distribution and evaluator failure patterns before and after context-derived exploration",
                component="optimizer",
                **base,
            )
        plateau_rounds = [r for r in rounds if str(r.get("termination_signal", "")).endswith("_plateau")]
        if best_edge > 0.0 and plateau_rounds:
            return HarnessChange(
                change="Introduce inventory-skew controls into plateau-recovery mutations to prevent accumulation risk",
                reason=f"Edge is positive ({best_edge:.3f}) but the loop plateaued {len(plateau_rounds)} time(s); inventory accumulation is likely capping further improvement.",
                expected_effect="Higher and more stable edge by preventing over-inventory adverse fills",
                risk="Aggressive skew dampens fill rate; may reduce gross edge even as net edge improves",
                evaluation="Compare net edge, inventory variance, and adverse-fill fraction before and after skew controls",
                component="optimizer",
                **base,
            )
    failed = [t for t in traces if t.get("status") != "completed"]
    if failed:
        return HarnessChange(
            change="Validate strategy file syntax before submitting to the upstream evaluator",
            reason=f"{len(failed)} trace(s) failed; syntax or import errors in generated strategies waste evaluation budget.",
            expected_effect="Zero wasted evaluation slots due to un-runnable strategy files",
            risk="Syntax validation alone does not catch runtime strategy errors; semantic pre-screening is still needed",
            evaluation="Compare evaluator error rate before and after pre-submission syntax checking",
            component="optimizer",
            **base,
        )
    return HarnessChange(
        change="Run each top-ranked strategy across multiple random seeds before selecting the final candidate",
        reason="Challenge evaluation uses a fixed seed set; a multi-seed pre-screen identifies high-variance strategies before the official submission.",
        expected_effect="Lower variance in the final candidate's official score and higher probability of meeting the profit target",
        risk="Multi-seed evaluation multiplies evaluator cost; budget the number of seeds relative to the iteration allowance",
        evaluation="Compare score variance on the official evaluator for single-seed vs. multi-seed selected candidates",
        component="optimizer",
        **base,
    )


def _support_level(confidence: float) -> str:
    if confidence >= 0.75:
        return "strong"
    if confidence >= 0.55:
        return "moderate"
    return "weak"


def _looks_contradictory(left: str, right: str) -> bool:
    pairs = [
        ("increases", "decreases"),
        ("improves", "worsens"),
        ("supports", "undermines"),
        ("high", "low"),
        ("more", "less"),
    ]
    left_text = left.lower()
    right_text = right.lower()
    return any(a in left_text and b in right_text or b in left_text and a in right_text for a, b in pairs)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _diagnosis_summary(diagnosis: dict[str, object]) -> str:
    localized = diagnosis.get("localized_components") or []
    if isinstance(localized, list) and localized:
        first = localized[0]
        if isinstance(first, dict):
            return f"{first.get('component', 'unknown')}: {first.get('reason', '')}"
    taxonomy = diagnosis.get("failure_taxonomy") or {}
    if isinstance(taxonomy, dict) and taxonomy:
        return "Failure taxonomy: " + ", ".join(f"{key}={value}" for key, value in sorted(taxonomy.items()))
    yield_info = diagnosis.get("artifact_yield") or {}
    if isinstance(yield_info, dict):
        return (
            f"Artifact yield sources={yield_info.get('sources', 0)}, "
            f"claims={yield_info.get('claims', 0)}, hypotheses={yield_info.get('hypotheses', 0)}."
        )
    return "No component-level failure localized."


def _dedupe_by_id(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    unique = []
    for row in rows:
        row_id = str(row["id"])
        if row_id not in seen:
            seen.add(row_id)
            unique.append(row)
    return unique


def _dedupe_by_text(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen: set[str] = set()
    unique = []
    for row in rows:
        key = str(row.get("text", row["id"])).strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def _citation(source: dict[str, object]) -> str:
    suffix = " local corpus fixture, not external literature" if _is_placeholder_source(source) else ""
    return f"{source['title']} ({source['date']}{suffix})"


PREDICTION_MARKET_REPORT_TERMS = {
    "prediction",
    "market",
    "markets",
    "orderbook",
    "order",
    "book",
    "maker",
    "making",
    "market-making",
    "liquidity",
    "retail",
    "arbitrage",
    "arbitrageur",
    "adverse",
    "selection",
    "quote",
    "quotes",
    "stale",
    "inventory",
    "spread",
    "lmsr",
    "amm",
    "scoring",
    "strategy",
    "strategies",
}


def _filter_report_evidence(
    run: RunRecord,
    sources: list[dict[str, object]],
    claims: list[dict[str, object]],
    hypotheses: list[dict[str, object]],
    contradictions: list[dict[str, object]],
    questions: list[dict[str, object]],
    seed_context: dict[str, object],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    if not _is_prediction_market_report(run, seed_context):
        return _filter_general_report_evidence(run, sources, claims, hypotheses, contradictions, questions)

    relevant_sources = [source for source in sources if _prediction_market_relevance_score(source) >= 2]
    relevant_source_ids = {str(source.get("id", "")) for source in relevant_sources}
    filtered_claims = []
    for claim in claims:
        source_ids = [str(source_id) for source_id in claim.get("source_ids", []) if str(source_id) in relevant_source_ids]
        if source_ids:
            filtered_claims.append({**claim, "source_ids": source_ids})
    claim_ids = {str(claim.get("id", "")) for claim in filtered_claims}
    filtered_hypotheses = [
        hypothesis
        for hypothesis in hypotheses
        if _prediction_market_relevance_score(hypothesis) >= 2
        or any(str(claim_id) in claim_ids for claim_id in hypothesis.get("supporting_claim_ids", []))
    ]
    filtered_contradictions = [
        contradiction
        for contradiction in contradictions
        if str(contradiction.get("claim_a", "")) in claim_ids and str(contradiction.get("claim_b", "")) in claim_ids
    ]
    filtered_questions = [question for question in questions if _prediction_market_relevance_score(question) >= 2]
    if relevant_sources:
        relevant_sources = sorted(
            relevant_sources,
            key=lambda source: (
                _prediction_market_relevance_score(source),
                float(source.get("relevance_score") or 0.0),
                float(source.get("credibility_score") or 0.0),
            ),
            reverse=True,
        )[:16]
    return relevant_sources, filtered_claims, filtered_hypotheses, filtered_contradictions, filtered_questions


def _filter_general_report_evidence(
    run: RunRecord,
    sources: list[dict[str, object]],
    claims: list[dict[str, object]],
    hypotheses: list[dict[str, object]],
    contradictions: list[dict[str, object]],
    questions: list[dict[str, object]],
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    """Keep final reports focused on the user's research topic.

    Retrieval can intentionally cast a wide net, but synthesis should not cite
    placeholder/demo sources or challenge-local artifacts when a pure research
    run already has real literature to use.
    """
    retained_sources = list(sources)
    non_challenge = [source for source in retained_sources if not _is_prediction_market_artifact(source)]
    if non_challenge:
        retained_sources = non_challenge
    non_placeholder = [source for source in retained_sources if not _is_placeholder_source(source)]
    if non_placeholder:
        retained_sources = non_placeholder

    topic_terms = _report_topic_terms(run.user_goal)
    if topic_terms and len(retained_sources) > 6:
        topic_sources = [
            source
            for source in retained_sources
            if _topic_relevance_score(source, topic_terms) >= 2
            and _topic_anchor_score(source, topic_terms) >= 1
        ]
        if len(topic_sources) >= 3:
            retained_sources = topic_sources

    retained_source_ids = {source.get("id") for source in retained_sources}
    retained_source_id_text = {str(source_id) for source_id in retained_source_ids if source_id is not None}
    filtered_claims = []
    for claim in claims:
        source_ids = [
            source_id
            for source_id in claim.get("source_ids", [])
            if source_id in retained_source_ids or str(source_id) in retained_source_id_text
        ]
        if source_ids:
            filtered_claims.append({**claim, "source_ids": source_ids})
        elif topic_terms and _topic_relevance_score(claim, topic_terms) >= 2 and not _is_placeholder_source(claim):
            filtered_claims.append({**claim, "source_ids": []})
    if not filtered_claims and claims:
        filtered_claims = [
            claim
            for claim in claims
            if not _is_prediction_market_artifact(claim) and not _is_placeholder_source(claim)
        ][:12]

    retained_claim_ids = {claim.get("id") for claim in filtered_claims}
    retained_claim_id_text = {str(claim_id) for claim_id in retained_claim_ids if claim_id is not None}
    filtered_hypotheses = [
        hypothesis
        for hypothesis in hypotheses
        if any(
            claim_id in retained_claim_ids or str(claim_id) in retained_claim_id_text
            for claim_id in hypothesis.get("supporting_claim_ids", [])
        )
        or (topic_terms and _topic_relevance_score(hypothesis, topic_terms) >= 2)
    ]
    filtered_contradictions = [
        contradiction
        for contradiction in contradictions
        if str(contradiction.get("claim_a", "")) in retained_claim_id_text
        and str(contradiction.get("claim_b", "")) in retained_claim_id_text
    ]
    filtered_questions = [
        question
        for question in questions
        if not _is_prediction_market_artifact(question)
        and not _is_placeholder_source(question)
        and (not topic_terms or _topic_relevance_score(question, topic_terms) > 0)
    ]
    return retained_sources, filtered_claims, filtered_hypotheses, filtered_contradictions, filtered_questions


def _is_prediction_market_report(run: RunRecord, seed_context: dict[str, object]) -> bool:
    if seed_context.get("evaluator_name") == "prediction_market":
        return True
    if run.harness_config_snapshot.get("evaluator_name") == "prediction_market":
        return True
    normalized = run.user_goal.lower().replace("-", " ")
    return "prediction market" in normalized or ("prediction" in normalized and "market" in normalized)


def _prediction_market_relevance_score(row: dict[str, object]) -> int:
    text = " ".join(str(value) for key, value in row.items() if key not in {"id", "retrieved_at"}).lower()
    score = sum(1 for term in PREDICTION_MARKET_REPORT_TERMS if term in text)
    if "prediction market" in text or "prediction-market" in text:
        score += 3
    if "danrobinson/prediction-market-challenge" in text or "orderbook prediction market challenge" in text:
        score += 4
    return score


def _is_placeholder_source(row: dict[str, object]) -> bool:
    text = " ".join(str(value) for value in row.values()).lower()
    return any(domain in text for domain in ["example.org", "example.com", "example.net", "example.edu", "example.invalid"])


def _is_prediction_market_artifact(row: dict[str, object]) -> bool:
    text = " ".join(str(value) for value in row.values()).lower().replace("-", "_")
    return (
        "challenges/prediction_market" in text
        or "prediction_market/evaluator.py" in text
        or "prediction_market/spec.md" in text
        or "danrobinson/prediction_market_challenge" in text
        or "orderbook prediction market challenge" in text
    )


REPORT_STOPWORDS = {
    "about",
    "across",
    "after",
    "agent",
    "agents",
    "also",
    "and",
    "are",
    "because",
    "been",
    "being",
    "build",
    "building",
    "can",
    "chatbots",
    "companies",
    "company",
    "cross",
    "current",
    "data",
    "dead",
    "does",
    "done",
    "drives",
    "evidence",
    "everywhere",
    "find",
    "for",
    "from",
    "gaps",
    "give",
    "has",
    "have",
    "historical",
    "how",
    "identify",
    "into",
    "kinda",
    "literature",
    "lot",
    "more",
    "much",
    "pattern",
    "platform",
    "platforms",
    "pure",
    "research",
    "show",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "thesis",
    "third",
    "which",
    "white",
    "will",
    "with",
    "work",
}


def _report_topic_terms(goal: str) -> set[str]:
    normalized = goal.lower().replace("-", " ")
    tokens = {token for token in re.findall(r"[a-z][a-z0-9]{2,}", normalized) if token not in REPORT_STOPWORDS}
    if "enterprise" in normalized:
        tokens.update({"enterprise", "saas", "adoption", "internalization", "outsourcing", "proprietary"})
    if "harness" in normalized:
        tokens.update({"harness", "harnesses", "orchestration", "scaffold", "evaluation", "evals"})
    if "multi-agent" in normalized or "inter-agent" in normalized:
        tokens.update({"multiagent", "multi", "inter", "coordination", "communication", "collaboration"})
    if "self-modification" in normalized or "self-improvement" in normalized or "self-correct" in normalized:
        tokens.update({"self", "modification", "improvement", "reflection", "reflexion", "repair", "adaptation"})
    if "trading" in normalized:
        tokens.update({"trading", "market", "strategy", "strategies", "portfolio", "financial"})
    if "evolutionary" in normalized:
        tokens.update({"evolutionary", "evolution", "genetic", "population", "mutation", "selection", "computation"})
    if "white collar" in normalized:
        tokens.update({"occupation", "occupations", "employment", "labor", "productivity", "workplace"})
    if "ai" in normalized or "llm" in normalized or "large language" in normalized:
        tokens.update({"ai", "llm", "llms", "artificial", "intelligence", "language", "model"})
    if "agent" in normalized:
        tokens.update({"agentic", "autonomous", "tool", "tools", "planning", "reasoning"})
    return tokens


def _topic_relevance_score(row: dict[str, object], topic_terms: set[str]) -> int:
    text = " ".join(_stringify_report_value(value) for value in row.values()).lower().replace("-", " ")
    tokens = set(re.findall(r"[a-z][a-z0-9]{2,}", text))
    return len(tokens & topic_terms)


REPORT_ANCHOR_TERMS = {
    "adoption",
    "agentic",
    "agents",
    "autonomous",
    "collaboration",
    "computation",
    "coordination",
    "enterprise",
    "evals",
    "evaluation",
    "evolution",
    "evolutionary",
    "genetic",
    "harness",
    "harnesses",
    "improvement",
    "internalization",
    "llm",
    "llms",
    "market",
    "modification",
    "multiagent",
    "outsourcing",
    "proprietary",
    "reflexion",
    "reflection",
    "repair",
    "saas",
    "self",
    "strategy",
    "trading",
}


def _topic_anchor_score(row: dict[str, object], topic_terms: set[str]) -> int:
    anchor_terms = topic_terms & REPORT_ANCHOR_TERMS
    if not anchor_terms:
        return 1
    text = " ".join(_stringify_report_value(value) for value in row.values()).lower().replace("-", " ")
    tokens = set(re.findall(r"[a-z][a-z0-9]{2,}", text))
    return len(tokens & anchor_terms)


def _stringify_report_value(value: object) -> str:
    if isinstance(value, dict):
        return " ".join(_stringify_report_value(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_stringify_report_value(item) for item in value)
    return str(value)


def _build_report(
    run: RunRecord,
    sources: list[dict[str, object]],
    claims: list[dict[str, object]],
    hypotheses: list[dict[str, object]],
    contradictions: list[dict[str, object]],
    questions: list[dict[str, object]],
) -> str:
    source_lookup = {source["id"]: source for source in sources}
    claims_by_id = {claim["id"]: claim for claim in claims}
    external_sources = _external_report_sources(sources)
    fixture_sources = [source for source in sources if _is_placeholder_source(source)]
    source_count_note = (
        f"{len(external_sources)} externally verifiable; {len(fixture_sources)} local fixture(s)"
        if fixture_sources else str(len(sources))
    )
    lines = [
        f"# Research Report: {run.user_goal}",
        "",
        f"- Run ID: `{run.id}`",
        f"- Task type: `{run.task_type}`",
        f"- Completed: {now_iso()}",
        f"- Sources reviewed: {source_count_note}",
        f"- Claims extracted: {len(claims)}",
        f"- Hypotheses ranked: {len(hypotheses)}",
        "",
        "## Executive Synthesis",
        _executive_summary(claims, hypotheses, contradictions),
        "",
        "## Key Takeaways",
    ]
    for takeaway in _key_takeaways(run, sources, claims, hypotheses, contradictions):
        lines.append(f"- {takeaway}")
    lines.extend([
        "",
        "## Key Claims",
    ])
    for claim in sorted(claims, key=lambda row: row["confidence"], reverse=True):
        citations = ", ".join(_citation(source_lookup[source_id]) for source_id in claim["source_ids"] if source_id in source_lookup)
        citation_text = citations or "No retained source citation"
        lines.append(f"- {claim['text']} Confidence: {claim['confidence']} ({claim['support_level']}). Sources: {citation_text}")
    lines.extend(["", "## Evidence Basis"])
    for source in external_sources:
        sections = source.get("evidence_sections") if isinstance(source.get("evidence_sections"), dict) else {}
        available = [name for name in ["abstract", "introduction", "conclusion"] if sections.get(name)]
        if available:
            preview = str(sections[available[0]])[:360]
            lines.append(f"- {_citation(source)} supplied {', '.join(available)} evidence. {available[0].title()}: {preview}")
        else:
            lines.append(f"- {_citation(source)} supplied summary-only evidence; claims from this source should remain low-confidence.")
    if fixture_sources:
        lines.append(
            f"- {len(fixture_sources)} local corpus fixture(s) contributed deterministic demo evidence. "
            "They are not external literature and are listed separately below without source links."
        )
    lines.extend(["", "## Hypothesis Evidence Matrix"])
    for hypothesis in hypotheses:
        supporting = [
            claims_by_id.get(claim_id)
            for claim_id in hypothesis.get("supporting_claim_ids", [])
            if claim_id in claims_by_id
        ]
        contradicting = [
            claims_by_id.get(claim_id)
            for claim_id in hypothesis.get("contradicting_claim_ids", [])
            if claim_id in claims_by_id
        ]
        proof = str(supporting[0]["text"]) if supporting else "No direct supporting claim retained."
        counter = str(contradicting[0]["text"]) if contradicting else "No direct counterpoint retained."
        lines.append(
            f"- {hypothesis['text']} Confidence: {hypothesis['confidence']}; "
            f"novelty: {hypothesis['novelty_score']}; testability: {hypothesis['testability_score']}. "
            f"Proof: {proof} Counterpoint: {counter} Next: {hypothesis['next_experiment']}"
        )
    lines.extend(["", "## Contradictions And Caveats"])
    if contradictions:
        for contradiction in contradictions:
            lines.append(
                f"- {contradiction['claim_a']} vs {contradiction['claim_b']}: "
                f"{contradiction['explanation']} Severity: {contradiction['severity']}."
            )
    else:
        lines.append("- No major contradictions were detected in this pass.")
    lines.extend(["", "## Open Questions"])
    if questions:
        for question in questions:
            lines.append(f"- P{question['priority']}: {question['question']} Reason: {question['reason']}")
    else:
        lines.append("- No open questions were created.")
    lines.extend(["", "## Sources"])
    if external_sources:
        for source in external_sources:
            lines.append(f"- [{source['title']}]({source['url']}) by {source['author']} ({source['date']})")
    else:
        lines.append("- No externally verifiable sources were retained for this report.")
    if fixture_sources:
        lines.extend(["", "## Local Corpus Fixtures"])
        lines.append("- The following records are bundled offline fixtures for deterministic demos/tests, not external sources:")
        for source in fixture_sources:
            lines.append(f"- {source['title']} by {source['author']} ({source['date']}); source_type={source.get('source_type', 'unknown')}")
    return "\n".join(lines) + "\n"


def _build_report_with_llm(
    llm: LLMClient,
    run: RunRecord,
    sources: list[dict[str, object]],
    claims: list[dict[str, object]],
    hypotheses: list[dict[str, object]],
    contradictions: list[dict[str, object]],
    questions: list[dict[str, object]],
    seed_context: dict[str, object],
) -> str:
    deterministic_report = _build_report(run, sources, claims, hypotheses, contradictions, questions)
    seed_section = _optimizer_seed_section(seed_context)
    if not llm.is_live:
        return deterministic_report + seed_section
    payload = {
        "goal": run.user_goal,
        "run_id": run.id,
        "task_type": run.task_type,
        "task_mode": run.task_mode,
        "product_agent": run.product_agent,
        "sources": sources[:12],
        "claims": sorted(claims, key=lambda row: row["confidence"], reverse=True)[:24],
        "hypotheses": hypotheses[:12],
        "contradictions": contradictions[:12],
        "open_questions": questions[:12],
        "optimizer_seed_context": seed_context,
    }
    system = (
        "You are the synthesis agent in a research harness. Write a concise, evidence-grounded "
        "Markdown report. Do not invent citations. Use only source IDs and URLs present in the payload. "
        "Near the start, before reviewing literature, include a 'Key Takeaways' section that gives your own "
        "synthesis opinion in 3-5 bullets, clearly distinguishing evidence-backed judgment from speculation. "
        "Call out uncertainty and contradictions. Treat source.evidence_sections.abstract, "
        "source.evidence_sections.introduction, and source.evidence_sections.conclusion as the only paper-context "
        "text you actually read; do not imply access to methods or results details unless those sections say so. "
        "For each major hypothesis, include a proof or supporting claim and a counterpoint, limitation, or explicit "
        "statement that no counterpoint was retained."
    )
    user = "Create the final report from this JSON payload:\n" + json.dumps(payload, indent=2, sort_keys=True)
    try:
        response = llm.complete(system, user, max_output_tokens=1800)
    except Exception as exc:
        fallback_note = f"\n\n## LLM Synthesis Fallback\n- Live synthesis failed: {type(exc).__name__}: {exc}\n"
        return deterministic_report + fallback_note
    if not response.text.strip():
        return deterministic_report + "\n\n## LLM Synthesis Fallback\n- Live synthesis returned no text.\n"
    fabricated = _fabricated_source_urls(response.text.strip(), sources)
    if fabricated:
        fallback_note = (
            f"\n\n## LLM Synthesis Fallback\n"
            f"- Live synthesis contained {len(fabricated)} fabricated source URL(s) not present in sources.json; "
            f"reverted to deterministic report. Fabricated: {fabricated[:5]}\n"
        )
        return deterministic_report + fallback_note + seed_section
    return response.text.strip() + "\n" + seed_section


def _read_optional_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _optimizer_seed_section(seed_context: dict[str, object]) -> str:
    if not seed_context:
        return ""
    lines = [
        "",
        "## Optimizer Seed Context",
        f"- Has evaluator: {seed_context.get('has_evaluator')}",
        f"- Summary: {seed_context.get('summary', '')}",
    ]
    for item in seed_context.get("top_query_findings", []) if isinstance(seed_context.get("top_query_findings"), list) else []:
        if isinstance(item, dict):
            lines.append(f"- Query seed {item.get('variant_id')}: score {item.get('score')}; {item.get('query')}")
    return "\n".join(lines) + "\n"


def _fabricated_source_urls(report: str, sources: list[dict[str, object]]) -> list[str]:
    """Return URLs cited in the report that are not in the known sources list."""
    known_urls = {str(s.get("url", "")) for s in sources}
    fabricated = []
    report_urls = re.findall(r"\]\((https?://[^)]+)\)", report)
    report_urls.extend(re.findall(r"\\url\{([^}]+)\}", report))
    for url in report_urls:
        if any(domain in url for domain in ["example.org", "example.com", "example.net", "example.edu", "example.invalid"]):
            fabricated.append(url)
        elif url not in known_urls:
            fabricated.append(url)
    return fabricated


def _build_latex_report(
    run: RunRecord,
    sources: list[dict[str, object]],
    claims: list[dict[str, object]],
    hypotheses: list[dict[str, object]],
    contradictions: list[dict[str, object]],
    questions: list[dict[str, object]],
) -> str:
    source_lookup = {source["id"]: source for source in sources}
    claims_by_id = {claim["id"]: claim for claim in claims}
    external_sources = _external_report_sources(sources)
    fixture_sources = [source for source in sources if _is_placeholder_source(source)]
    title = _latex_escape(run.user_goal)
    lines = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{hyperref}",
        r"\usepackage{booktabs}",
        r"\usepackage{enumitem}",
        r"\usepackage{microtype}",
        r"\title{" + title + r"}",
        r"\author{Research Harness}",
        r"\date{" + _latex_escape(now_iso()[:10]) + r"}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        _latex_escape(_executive_summary(claims, hypotheses, contradictions)),
        r"\end{abstract}",
        r"\section{Key Takeaways}",
        r"\begin{itemize}[leftmargin=*]",
    ]
    for takeaway in _key_takeaways(run, sources, claims, hypotheses, contradictions):
        lines.append(r"\item " + _latex_escape(takeaway))
    lines.extend([
        r"\end{itemize}",
        r"\section{Evidence Base}",
        f"Reviewed {_latex_escape(str(len(external_sources)))} externally verifiable sources"
        + (f" and {_latex_escape(str(len(fixture_sources)))} local corpus fixtures" if fixture_sources else "")
        + f"; extracted {_latex_escape(str(len(claims)))} grounded claims.",
        r"\section{Key Claims}",
        r"\begin{enumerate}[leftmargin=*]",
    ])
    for claim in sorted(claims, key=lambda row: row["confidence"], reverse=True)[:18]:
        citations = ", ".join(_plain_citation(source_lookup[source_id]) for source_id in claim.get("source_ids", []) if source_id in source_lookup)
        lines.append(
            r"\item "
            + _latex_escape(str(claim.get("text", "")))
            + f" \\textit{{Confidence: {_latex_escape(str(claim.get('confidence', '')))}; sources: {_latex_escape(citations or 'none retained')}}}."
        )
    lines.extend([r"\end{enumerate}", r"\section{Paper Context Read By The Agent}", r"\begin{itemize}[leftmargin=*]"])
    for source in external_sources[:12]:
        sections = source.get("evidence_sections") if isinstance(source.get("evidence_sections"), dict) else {}
        available = [name for name in ["abstract", "introduction", "conclusion"] if sections.get(name)]
        preview = str(sections.get(available[0], ""))[:500] if available else str(source.get("summary", ""))[:300]
        lines.append(
            r"\item \textbf{"
            + _latex_escape(str(source.get("title", ""))[:180])
            + r"}: "
            + _latex_escape(", ".join(available) if available else "summary-only")
            + ". "
            + _latex_escape(preview)
        )
    if fixture_sources:
        lines.append(
            r"\item "
            + _latex_escape(
                f"{len(fixture_sources)} local corpus fixture(s) contributed deterministic demo evidence; these are not external literature."
            )
        )
    lines.extend([r"\end{itemize}", r"\section{Hypothesis Evidence Matrix}", r"\begin{itemize}[leftmargin=*]"])
    for hypothesis in hypotheses[:12]:
        supporting = [
            claims_by_id[claim_id]
            for claim_id in hypothesis.get("supporting_claim_ids", [])
            if claim_id in claims_by_id
        ]
        contradicting = [
            claims_by_id[claim_id]
            for claim_id in hypothesis.get("contradicting_claim_ids", [])
            if claim_id in claims_by_id
        ]
        proof = str(supporting[0].get("text", "")) if supporting else "No direct supporting claim retained."
        counter = str(contradicting[0].get("text", "")) if contradicting else "No direct counterpoint retained; treat as unresolved."
        lines.append(
            r"\item \textbf{"
            + _latex_escape(str(hypothesis.get("text", ""))[:220])
            + r"}\\ Proof: "
            + _latex_escape(proof[:420])
            + r"\\ Counterpoint: "
            + _latex_escape(counter[:420])
        )
    lines.extend([r"\end{itemize}", r"\section{Limitations And Open Questions}"])
    if contradictions:
        lines.append(r"\begin{itemize}[leftmargin=*]")
        for contradiction in _dedupe_by_field(contradictions, "explanation")[:10]:
            lines.append(r"\item " + _latex_escape(str(contradiction.get("explanation", ""))))
        lines.append(r"\end{itemize}")
    if questions:
        lines.append(r"\begin{itemize}[leftmargin=*]")
        for question in questions[:10]:
            lines.append(r"\item " + _latex_escape(str(question.get("question", ""))))
        lines.append(r"\end{itemize}")
    lines.extend([r"\section{References}", r"\begin{enumerate}[leftmargin=*]"])
    for source in external_sources:
        lines.append(r"\item " + _latex_escape(_plain_citation(source)) + r" \url{" + _latex_url(str(source.get("url", ""))) + r"}")
    if fixture_sources:
        lines.append(r"\item " + _latex_escape("Local corpus fixtures were omitted from References because they are not external sources."))
    lines.extend([r"\end{enumerate}", r"\end{document}", ""])
    return "\n".join(lines)


def _dedupe_by_field(rows: list[dict[str, object]], field: str) -> list[dict[str, object]]:
    seen = set()
    deduped = []
    for row in rows:
        value = str(row.get(field, "")).strip().lower()
        if value in seen:
            continue
        seen.add(value)
        deduped.append(row)
    return deduped


def _render_report_pdf(store: ArtifactStore, tex_path: Path) -> bool:
    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        try:
            subprocess.run(
                [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                cwd=store.root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                check=True,
            )
            generated = tex_path.with_suffix(".pdf")
            if generated.exists():
                store.write_report_pdf(generated.read_bytes())
                return True
        except Exception:
            pass
    store.write_report_pdf(_minimal_pdf_bytes(_plain_text_from_tex(tex_path.read_text(encoding="utf-8"))))
    return True


def _render_report_preview_png(store: ArtifactStore) -> Optional[bytes]:
    """Rasterize the first PDF page so the IDE preview matches the LaTeX report."""
    pdf_path = store.report_pdf_path
    if not pdf_path.exists():
        return None
    renderers = [
        _render_pdf_preview_with_pdftoppm,
        _render_pdf_preview_with_magick,
        _render_pdf_preview_with_qlmanage,
    ]
    for renderer in renderers:
        try:
            preview = renderer(pdf_path)
        except Exception:
            preview = None
        if preview:
            return preview
    return None


def _render_pdf_preview_with_pdftoppm(pdf_path: Path) -> Optional[bytes]:
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        return None
    prefix = pdf_path.with_name(pdf_path.stem + "_preview")
    output = prefix.with_name(prefix.name + "-1.png")
    completed = subprocess.run(
        [pdftoppm, "-png", "-f", "1", "-singlefile", "-r", "160", str(pdf_path), str(prefix)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    if completed.returncode == 0 and output.exists():
        return output.read_bytes()
    return None


def _render_pdf_preview_with_magick(pdf_path: Path) -> Optional[bytes]:
    magick = shutil.which("magick") or shutil.which("convert")
    if not magick:
        return None
    output = pdf_path.with_name(pdf_path.stem + "_preview_render.png")
    source = f"{pdf_path}[0]"
    completed = subprocess.run(
        [magick, "-density", "160", source, "-background", "white", "-alpha", "remove", str(output)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    if completed.returncode == 0 and output.exists():
        return output.read_bytes()
    return None


def _render_pdf_preview_with_qlmanage(pdf_path: Path) -> Optional[bytes]:
    qlmanage = shutil.which("qlmanage")
    if not qlmanage:
        return None
    out_dir = pdf_path.parent / ".preview"
    out_dir.mkdir(exist_ok=True)
    completed = subprocess.run(
        [qlmanage, "-t", "-s", "1400", "-o", str(out_dir), str(pdf_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )
    candidate_names = [
        pdf_path.name + ".png",
        pdf_path.stem + ".png",
        pdf_path.with_suffix(".png").name,
    ]
    for name in candidate_names:
        candidate = out_dir / name
        if completed.returncode == 0 and candidate.exists():
            return candidate.read_bytes()
    candidates = sorted(out_dir.glob("*.png"), key=lambda path: path.stat().st_mtime, reverse=True)
    if completed.returncode == 0 and candidates:
        return candidates[0].read_bytes()
    return None


def _report_preview_png(
    run: RunRecord,
    sources: list[dict[str, object]],
    claims: list[dict[str, object]],
    hypotheses: list[dict[str, object]],
    contradictions: list[dict[str, object]],
) -> bytes:
    from .run_benchmarks import _PngCanvas

    canvas = _PngCanvas(1000, 1300, "#ffffff")
    canvas.rect(0, 0, 1000, 1300, "#f8fafc")
    canvas.rect(56, 48, 888, 1180, "#ffffff")
    canvas.outline(56, 48, 888, 1180, "#cbd5e1")
    canvas.text(88, 86, "Research Report Preview", "#0f172a", 4, max_chars=34)
    y = 144
    for line in _wrap_preview_text(str(run.user_goal), 54)[:3]:
        canvas.text(88, y, line, "#334155", 2, max_chars=58)
        y += 26
    y += 18
    canvas.text(88, y, "Key Takeaways", "#0f172a", 3, max_chars=32)
    y += 42
    for takeaway in _key_takeaways(run, sources, claims, hypotheses, contradictions):
        for index, line in enumerate(_wrap_preview_text(takeaway, 78)[:4]):
            prefix = "- " if index == 0 else "  "
            canvas.text(88, y, prefix + line, "#334155", 2, max_chars=82)
            y += 24
        y += 12
        if y > 650:
            break
    y += 14
    canvas.text(88, y, "Evidence Snapshot", "#0f172a", 3, max_chars=32)
    y += 42
    snapshot = [
        f"Sources reviewed: {len(sources)}",
        f"Grounded claims: {len(claims)}",
        f"Hypotheses ranked: {len(hypotheses)}",
        f"Contradictions/open caveats: {len(contradictions)}",
    ]
    for item in snapshot:
        canvas.text(110, y, item, "#475569", 2, max_chars=70)
        y += 30
    y += 12
    canvas.text(88, y, "Open final_report.pdf for the full paper-style report.", "#64748b", 2, max_chars=80)
    return canvas.png()


def _plain_citation(source: dict[str, object]) -> str:
    suffix = " local corpus fixture, not external literature" if _is_placeholder_source(source) else ""
    return f"{source.get('title', '')} ({source.get('date', '')}{suffix})"


def _external_report_sources(sources: list[dict[str, object]]) -> list[dict[str, object]]:
    return [source for source in sources if not _is_placeholder_source(source)]


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _latex_url(url: str) -> str:
    return url.replace("\\", "%5C").replace("}", "%7D").replace("{", "%7B")


def _plain_text_from_tex(tex: str) -> str:
    text = re.sub(r"\\url\{([^}]*)\}", r"\1", tex)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{([^{}]*)\})?", r"\1", text)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text[:7000]


def _minimal_pdf_bytes(text: str) -> bytes:
    lines = [line[:90] for line in re.findall(r".{1,90}(?:\s+|$)", text)[:42]]
    commands = ["BT", "/F1 10 Tf", "50 760 Td", "14 TL"]
    for index, line in enumerate(lines):
        if index:
            commands.append("T*")
        commands.append(f"({_pdf_escape(line.strip())}) Tj")
    commands.append("ET")
    stream = "\n".join(commands).encode("latin-1", errors="replace")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")
    xref = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    return bytes(pdf)


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_preview_text(text: str, width: int) -> list[str]:
    words = re.sub(r"\s+", " ", text).strip().split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width and current:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines or [""]


def _key_takeaways(
    run: RunRecord,
    sources: list[dict[str, object]],
    claims: list[dict[str, object]],
    hypotheses: list[dict[str, object]],
    contradictions: list[dict[str, object]],
) -> list[str]:
    if not claims:
        return ["The retained evidence is too thin to support a strong synthesis yet."]
    avg_confidence = sum(float(claim.get("confidence", 0.0)) for claim in claims) / max(len(claims), 1)
    strong_claims = [claim for claim in claims if float(claim.get("confidence", 0.0)) >= 0.65]
    topic = str(run.user_goal).rstrip(".")
    strongest = strong_claims[0] if strong_claims else max(claims, key=lambda claim: float(claim.get("confidence", 0.0)))
    strongest_text = str(strongest.get("text", "")).strip()
    source_ids = [str(source_id) for source_id in strongest.get("source_ids", [])]
    source_lookup = {str(source.get("id", "")): source for source in sources}
    cited_sources = [source_lookup[source_id] for source_id in source_ids if source_id in source_lookup]
    source_note = ""
    if cited_sources:
        source_note = f" Best cited source: {_plain_citation(cited_sources[0])}."
    takeaways = [
        (
            f"For '{topic}', this run retained {len(claims)} grounded claim(s) from {len(sources)} source(s); "
            f"average claim confidence is {avg_confidence:.2f}."
        ),
        (
            f"The strongest retained signal is: {strongest_text[:260]}.{source_note}"
        ),
    ]
    if hypotheses:
        takeaways.append(f"The leading evidence-backed direction is: {str(hypotheses[0].get('text', ''))[:260]}.")
    if contradictions:
        takeaways.append(
            f"Uncertainty remains: {len(contradictions)} contradiction or caveat record(s) need source-level resolution before treating the conclusion as settled."
        )
    else:
        takeaways.append(
            "No major contradiction was retained, but absence of contradiction in this run is not proof that the claim is settled."
        )
    if sources:
        source_types = sorted({str(source.get("source_type", "source")) for source in sources if source.get("source_type")})
        takeaways.append(
            f"Before the literature detail, the practical bottom line is to anchor next steps in the retained {', '.join(source_types[:4]) or 'source'} evidence rather than importing an outside storyline."
        )
    return takeaways[:5]


def _executive_summary(
    claims: list[dict[str, object]],
    hypotheses: list[dict[str, object]],
    contradictions: list[dict[str, object]],
) -> str:
    if not claims:
        return "The run did not find enough evidence to produce a reliable synthesis."
    avg_confidence = sum(float(claim["confidence"]) for claim in claims) / len(claims)
    best = hypotheses[0]["text"] if hypotheses else "No ranked hypothesis emerged."
    caveat = "Contradictions require follow-up before acting." if contradictions else "No major contradiction was detected."
    return f"Evidence quality is {avg_confidence:.2f} on average. Leading direction: {best} {caveat}"
