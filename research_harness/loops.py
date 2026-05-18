from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable, Optional, Protocol, Union

from challenges.prediction_market import prediction_market_score

from .llm import LLMClient
from .schemas import (
    AgentTrace,
    Claim,
    EvolutionRound,
    FailedPath,
    LoopContinuationDecision,
    ProductAgent,
    Source,
    SourceStrategyItem,
    TaskIngestionDecision,
    TaskMode,
    Variant,
    VariantEvaluation,
    now_iso,
)
from .search import SearchBackend
from .sandbox import DockerSandboxRunner
from .store import ArtifactStore


@dataclass(frozen=True)
class EvaluatorResult:
    score: float
    status: str = "completed"
    metrics: Optional[dict[str, object]] = None
    diagnostics: Optional[dict[str, object]] = None
    loss_reason: str = ""
    summary: str = ""


EvaluatorPayload = Union[float, int, dict[str, object], EvaluatorResult]
EvaluatorFn = Callable[[str], EvaluatorPayload]
SearchFactory = Callable[[str], SearchBackend]


def _find_pm_upstream_path() -> Optional[Path]:
    """Auto-detect the prediction-market-challenge repo at common locations.

    Set PREDICTION_MARKET_CHALLENGE_PATH to override. Set
    PREDICTION_MARKET_USE_UPSTREAM=0 to force the local fallback regardless.
    """
    if os.environ.get("PREDICTION_MARKET_USE_UPSTREAM") == "0":
        return None
    project_root = Path(__file__).resolve().parents[1]
    candidates: list[Optional[str]] = [
        os.environ.get("PREDICTION_MARKET_CHALLENGE_PATH"),
        str(project_root / "challenges" / "prediction-market-challenge"),
        str(project_root / "challenges" / "prediction_market_challenge"),
        str(project_root / "vendor" / "prediction-market-challenge"),
        "/private/tmp/prediction-market-challenge-src",
        str(Path.home() / "prediction-market-challenge"),
        str(Path.home() / "src" / "prediction-market-challenge"),
        str(Path.cwd() / "prediction-market-challenge"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        p = Path(candidate)
        if _is_pm_upstream_repo(p):
            return p
    return None


def _is_pm_upstream_repo(path: Path) -> bool:
    """Return True only for danrobinson/prediction-market-challenge layout."""
    pyproject = path / "pyproject.toml"
    package_dir = path / "orderbook_pm_challenge"
    if not path.is_dir() or not pyproject.exists() or not package_dir.is_dir():
        return False
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return False
    return 'orderbook-pm = "orderbook_pm_challenge.cli:main"' in text


def _normalize_evaluator_result(raw_result: EvaluatorPayload) -> EvaluatorResult:
    if isinstance(raw_result, EvaluatorResult):
        return EvaluatorResult(
            score=max(0.0, min(1.0, float(raw_result.score))),
            status=raw_result.status or "completed",
            metrics=dict(raw_result.metrics or {}),
            diagnostics=dict(raw_result.diagnostics or {}),
            loss_reason=raw_result.loss_reason,
            summary=raw_result.summary,
        )
    if isinstance(raw_result, dict):
        score = float(raw_result.get("score", raw_result.get("deterministic_score", 0.0)) or 0.0)
        diagnostics = raw_result.get("diagnostics") if isinstance(raw_result.get("diagnostics"), dict) else {}
        metrics = raw_result.get("metrics") if isinstance(raw_result.get("metrics"), dict) else {}
        if not metrics:
            metrics = {
                key: value
                for key, value in raw_result.items()
                if key not in {"score", "deterministic_score", "status", "diagnostics", "loss_reason", "summary"}
            }
        return EvaluatorResult(
            score=max(0.0, min(1.0, score)),
            status=str(raw_result.get("status") or "completed"),
            metrics=dict(metrics),
            diagnostics=dict(diagnostics),
            loss_reason=str(raw_result.get("loss_reason") or raw_result.get("failure_reason") or ""),
            summary=str(raw_result.get("summary") or ""),
        )
    score = max(0.0, min(1.0, float(raw_result)))
    return EvaluatorResult(
        score=score,
        status="completed",
        metrics={"scalar_score": score},
        diagnostics={},
        loss_reason="" if score > 0 else "zero_score",
        summary=f"Deterministic scalar evaluator returned {score:.3f}.",
    )


def _exception_evaluator_result(exc: Exception) -> EvaluatorResult:
    exc_type = type(exc).__name__
    lower = str(exc).lower()
    if "timeout" in lower or exc_type.lower().find("timeout") >= 0:
        reason = "timeout"
    elif "compile" in lower or "syntax" in lower:
        reason = "compile_error"
    elif "correct" in lower or "assert" in lower:
        reason = "correctness_fail"
    elif "slow" in lower or "baseline" in lower:
        reason = "slower_than_baseline"
    else:
        reason = "runtime_error"
    return EvaluatorResult(
        score=0.0,
        status="failed",
        metrics={},
        diagnostics={"exception_type": exc_type, "exception": str(exc)},
        loss_reason=reason,
        summary=f"{exc_type}: {exc}",
    )


def _evaluator_json_response(result: EvaluatorResult) -> dict[str, object]:
    return {
        "status": result.status,
        "score": result.score,
        "metrics": result.metrics or {},
        "diagnostics": result.diagnostics or {},
        "loss_reason": result.loss_reason,
        "summary": result.summary,
    }


OPTIMIZE_HINTS = {
    "benchmark",
    "kernel",
    "latency",
    "optimize",
    "performance",
    "score",
    "speed",
    "strategy",
    "swe-bench",
    "throughput",
}


class EvaluatorRegistry:
    """Registry for deterministic evaluators.

    Optimize mode is only valid when a deterministic evaluator is available.
    The tiny built-in evaluator is for smoke tests and architecture demos; real
    tasks should register domain evaluators such as a benchmark harness.
    """

    def __init__(self) -> None:
        self._evaluators: dict[str, EvaluatorFn] = {
            "length_score": lambda payload: 1.0 / max(1, len(payload.split())),
            "prediction_market": prediction_market_score,
        }

    def get(self, name: Optional[str]) -> Optional[EvaluatorFn]:
        if not name:
            return None
        return self._evaluators.get(name)

    def register(self, name: str, evaluator: EvaluatorFn) -> None:
        self._evaluators[name] = evaluator


class TaskRouter:
    def __init__(self, evaluator_registry: EvaluatorRegistry, llm: Optional[LLMClient] = None):
        self.evaluator_registry = evaluator_registry
        self.llm = llm

    def decide(self, goal: str, requested_mode: str = "auto", evaluator_name: Optional[str] = None) -> TaskIngestionDecision:
        requested = requested_mode.lower()
        evaluator = self.evaluator_registry.get(evaluator_name)

        # Explicit mode flags bypass LLM routing — the user already decided.
        if requested != "auto":
            return self._decide_explicit(goal, requested, evaluator_name, evaluator)

        # Try LLM-based classification for auto mode.
        if self.llm and self.llm.is_live:
            try:
                return self._llm_decide(goal, evaluator_name, evaluator)
            except Exception:
                pass

        # Heuristic fallback when LLM is unavailable.
        return self._heuristic_decide(goal, requested_mode, evaluator_name, evaluator)

    def _decide_explicit(
        self,
        goal: str,
        requested: str,
        evaluator_name: Optional[str],
        evaluator: Optional[object],
    ) -> TaskIngestionDecision:
        if requested == "research":
            return TaskIngestionDecision(
                requested_mode=requested,
                selected_mode="research",
                reason="Research mode was explicitly requested via --task-mode.",
                product_agent="research",
            )
        if requested == "optimize_query":
            product_agent = _product_agent_for("optimize_query", goal, evaluator_name)
            return TaskIngestionDecision(
                requested_mode=requested,
                selected_mode="optimize_query",
                evaluator_name=evaluator_name if evaluator else None,
                product_agent=product_agent,
                reason=(
                    f"{product_agent.title()} agent selected optimization-query loop (explicit --task-mode); "
                    + (f"evaluator '{evaluator_name}' is registered." if evaluator else "no evaluator registered.")
                ),
            )
        if requested == "optimize" and evaluator_name == "prediction_market" and evaluator:
            return TaskIngestionDecision(
                requested_mode=requested,
                selected_mode="optimize_query",
                evaluator_name=evaluator_name,
                product_agent="challenge",
                reason=(
                    "Challenge agent selected optimization-query loop because the prediction_market evaluator "
                    "requires challenge strategy research before scoring."
                ),
            )
        if requested == "optimize" and evaluator:
            product_agent = _product_agent_for("optimize", goal, evaluator_name)
            return TaskIngestionDecision(
                requested_mode=requested,
                selected_mode="optimize",
                evaluator_name=evaluator_name,
                product_agent=product_agent,
                reason=f"{product_agent.title()} agent selected optimize loop (explicit --task-mode) with evaluator '{evaluator_name}'.",
            )
        if requested == "optimize" and not evaluator:
            return TaskIngestionDecision(
                requested_mode=requested,
                selected_mode="research",
                evaluator_name=evaluator_name,
                product_agent="research",
                reason="Optimize mode requested explicitly but no deterministic evaluator was registered; falling back to research mode.",
            )
        return self._heuristic_decide(goal, requested, evaluator_name, evaluator)

    def _llm_decide(
        self,
        goal: str,
        evaluator_name: Optional[str],
        evaluator: Optional[object],
    ) -> TaskIngestionDecision:
        """Classify the task mode using an LLM call, with full reasoning logged."""
        assert self.llm is not None
        system = (
            "You are the task router in a research-and-optimization harness. "
            "Classify the user's goal into exactly one of three modes and explain your reasoning.\n\n"
            "Modes:\n"
            "- research: open-ended literature review or knowledge synthesis with no deterministic score.\n"
            "- optimize: direct optimization against a registered deterministic evaluator (no research phase needed).\n"
            "- optimize_query: research-then-optimize; the agent first explores literature to build strategy context, "
            "then runs an optimizer. Use this when the goal mentions researching before optimizing, or when the task "
            "is a challenge that benefits from domain grounding.\n\n"
            "Return JSON only: {\"selected_mode\": str, \"product_agent\": str, \"confidence\": float, \"reason\": str}\n"
            "product_agent must be one of: research, optimize, challenge.\n"
            "Use 'challenge' when the goal references a specific scored competition or external evaluator."
        )
        user = json.dumps(
            {
                "goal": goal,
                "evaluator_registered": evaluator_name if evaluator else None,
                "available_modes": ["research", "optimize", "optimize_query"],
            },
            indent=2,
        )
        payload = self.llm.complete_json(system, user, max_output_tokens=400)
        selected_mode = str(payload.get("selected_mode", "research")).lower()
        product_agent = str(payload.get("product_agent", "research")).lower()
        reason = str(payload.get("reason", "LLM router selected this mode."))
        confidence = float(payload.get("confidence", 0.8))

        if selected_mode not in {"research", "optimize", "optimize_query"}:
            selected_mode = "research"
        if product_agent not in {"research", "optimize", "challenge"}:
            product_agent = "research"

        return TaskIngestionDecision(
            requested_mode="auto",
            selected_mode=selected_mode,  # type: ignore[arg-type]
            evaluator_name=evaluator_name if evaluator else None,
            product_agent=product_agent,  # type: ignore[arg-type]
            reason=f"[LLM router, confidence={confidence:.2f}] {reason}",
        )

    def _heuristic_decide(
        self,
        goal: str,
        requested_mode: str,
        evaluator_name: Optional[str],
        evaluator: Optional[object],
    ) -> TaskIngestionDecision:
        """Keyword/pattern-based fallback routing when no LLM is available."""
        if _looks_like_optimization_query(goal):
            product_agent = _product_agent_for("optimize_query", goal, evaluator_name)
            return TaskIngestionDecision(
                requested_mode=requested_mode,
                selected_mode="optimize_query",
                evaluator_name=evaluator_name if evaluator else None,
                product_agent=product_agent,
                reason=(
                    f"[heuristic router] Prompt contains research+optimization signals; "
                    f"routed to optimize_query for {product_agent} agent"
                    + (" with registered evaluator." if evaluator else ".")
                ),
            )
        if evaluator:
            product_agent = _product_agent_for("optimize", goal, evaluator_name)
            return TaskIngestionDecision(
                requested_mode=requested_mode,
                selected_mode="optimize",
                evaluator_name=evaluator_name,
                product_agent=product_agent,
                reason=f"[heuristic router] Deterministic evaluator '{evaluator_name}' is registered; selected optimize mode.",
            )
        if any(hint in goal.lower() for hint in OPTIMIZE_HINTS):
            return TaskIngestionDecision(
                requested_mode=requested_mode,
                selected_mode="research",
                product_agent="research",
                reason="[heuristic router] Prompt looks optimization-shaped but no evaluator is registered; using research mode.",
            )
        return TaskIngestionDecision(
            requested_mode=requested_mode,
            selected_mode="research",
            product_agent="research",
            reason="[heuristic router] No evaluator registered and no optimization signal found; defaulting to research mode.",
        )


@dataclass
class InnerLoopResult:
    ranked_evaluations: list[VariantEvaluation]
    termination_signal: str


class InnerLoop(Protocol):
    mode: TaskMode

    async def evaluate(self, variants: list[Variant], store: ArtifactStore) -> InnerLoopResult: ...


class OptimizeLoop:
    mode: TaskMode = "optimize"

    def __init__(self, run_id: str, evaluator: EvaluatorFn, pass_threshold: float = 0.8, parallel_evaluator_cap: int = 8):
        self.run_id = run_id
        self.evaluator = evaluator
        self.pass_threshold = pass_threshold
        self.parallel_evaluator_cap = max(1, parallel_evaluator_cap)

    async def evaluate(self, variants: list[Variant], store: ArtifactStore) -> InnerLoopResult:
        semaphore = asyncio.Semaphore(self.parallel_evaluator_cap)
        evaluations = await asyncio.gather(*(self._evaluate_variant_capped(variant, store, semaphore) for variant in variants))
        for evaluation in evaluations:
            store.add_variant_evaluation(evaluation)
        ranked = sorted(evaluations, key=lambda item: item.score, reverse=True)
        signal = "score_threshold" if ranked and ranked[0].score >= self.pass_threshold else "continue"
        return InnerLoopResult(ranked_evaluations=ranked, termination_signal=signal)

    async def _evaluate_variant_capped(
        self,
        variant: Variant,
        store: ArtifactStore,
        semaphore: asyncio.Semaphore,
    ) -> VariantEvaluation:
        async with semaphore:
            return await self._evaluate_variant(variant, store)

    async def _evaluate_variant(self, variant: Variant, store: ArtifactStore) -> VariantEvaluation:
        started = time.perf_counter()
        started_at = now_iso()
        try:
            raw_result = self.evaluator(variant.payload)
            result = _normalize_evaluator_result(raw_result)
            score = result.score
            metrics = {
                "deterministic_score": score,
                "evaluator_status": result.status,
                "loss_reason": result.loss_reason,
                "diagnostics": result.diagnostics or {},
                "raw_metrics": result.metrics or {},
                "direction": _variant_direction_metadata(variant),
                "json_response": _evaluator_json_response(result),
            }
            summary_payload = {
                "status": result.status,
                "score": score,
                "passed": score >= self.pass_threshold,
                "loss_reason": result.loss_reason,
                "diagnostics": result.diagnostics or {},
                "summary": result.summary,
            }
            evaluation = VariantEvaluation(
                run_id=self.run_id,
                variant_id=variant.id,
                inner_loop="optimize",
                score=score,
                metrics=metrics,
                judge_scores=[score],
                summary=json.dumps(summary_payload, sort_keys=True),
                passed=score >= self.pass_threshold,
            )
            _record_timing_trace(
                store,
                self.run_id,
                agent_name=f"optimize_eval:{variant.id}",
                role="optimize_evaluator",
                prompt=variant.payload,
                model="deterministic-evaluator",
                started_at=started_at,
                started=started,
                status="completed",
                output_summary=evaluation.summary,
            )
            return evaluation
        except Exception as exc:
            failure = _exception_evaluator_result(exc)
            evaluation = VariantEvaluation(
                run_id=self.run_id,
                variant_id=variant.id,
                inner_loop="optimize",
                score=0.0,
                metrics={
                    "deterministic_score": 0.0,
                    "evaluator_status": failure.status,
                    "loss_reason": failure.loss_reason,
                    "diagnostics": failure.diagnostics or {},
                    "raw_metrics": failure.metrics or {},
                    "direction": _variant_direction_metadata(variant),
                    "json_response": _evaluator_json_response(failure),
                },
                judge_scores=[0.0],
                summary=json.dumps(_evaluator_json_response(failure), sort_keys=True),
                passed=False,
            )
            _record_timing_trace(
                store,
                self.run_id,
                agent_name=f"optimize_eval:{variant.id}",
                role="optimize_evaluator",
                prompt=variant.payload,
                model="deterministic-evaluator",
                started_at=started_at,
                started=started,
                status="failed",
                output_summary="Deterministic evaluator failed.",
                errors=[f"{type(exc).__name__}: {exc}"],
            )
            return evaluation


class ResearchLoop:
    mode: TaskMode = "research"

    def __init__(self, run_id: str, search_factory: SearchFactory, llm: Optional[LLMClient] = None, pass_threshold: float = 0.78):
        self.run_id = run_id
        self.search_factory = search_factory
        self.llm = llm or LLMClient()
        self.pass_threshold = pass_threshold

    async def evaluate(self, variants: list[Variant], store: ArtifactStore) -> InnerLoopResult:
        evaluations = await asyncio.gather(*(self._evaluate_variant(variant, store) for variant in variants))
        for evaluation in evaluations:
            store.add_variant_evaluation(evaluation)
        ranked = sorted(evaluations, key=lambda item: item.score, reverse=True)
        signal = "claim_corroboration_threshold" if ranked and ranked[0].score >= self.pass_threshold else "continue"
        return InnerLoopResult(ranked_evaluations=ranked, termination_signal=signal)

    async def _evaluate_variant(self, variant: Variant, store: ArtifactStore) -> VariantEvaluation:
        started = time.perf_counter()
        started_at = now_iso()
        tokens_before = self.llm.total_prompt_tokens + self.llm.total_completion_tokens
        retriever_name = str(variant.metadata.get("retriever", "local"))
        limit = int(variant.metadata.get("limit", 6))
        try:
            backend, backend_results, retrieval_notes = await self._search_with_fallback(retriever_name, variant, limit, store)
            sources = []
            claim_count = 0
            for document, relevance in backend_results:
                source = store.add_source(backend.to_source(document, relevance))
                sources.append(source)
                for claim_text in document.claims[:3]:
                    confidence = round((source.credibility_score * 0.7) + (relevance * 0.3), 2)
                    store.add_claim(
                        Claim(
                            text=claim_text,
                            source_ids=[source.id],
                            confidence=confidence,
                            support_level=_support_level(confidence),
                            created_by_agent=f"research_loop:{variant.id}",
                            run_id=self.run_id,
                        )
                    )
                    claim_count += 1
            metrics = self._research_metrics(sources, claim_count)
            metrics["fallback_used"] = 1.0 if retrieval_notes else 0.0
            judge_scores = [
                metrics["factual_accuracy"],
                metrics["citation_accuracy"],
                metrics["completeness"],
                metrics["source_quality"],
                metrics["tool_efficiency"],
                _stable_judge_score(variant.payload, metrics),
            ]
            llm_score, llm_summary = self._llm_judge_score(variant, metrics, len(sources), claim_count)
            if llm_score is not None:
                judge_scores.append(llm_score)
            score = round(median(judge_scores), 3)
            evaluation = VariantEvaluation(
                run_id=self.run_id,
                variant_id=variant.id,
                inner_loop="research",
                score=score,
                metrics=metrics,
                judge_scores=judge_scores,
                summary=(
                    f"Retrieved {len(sources)} sources and {claim_count} claims; "
                    f"rubric factual={metrics['factual_accuracy']:.3f}, citation={metrics['citation_accuracy']:.3f}, "
                    f"complete={metrics['completeness']:.3f}, source_quality={metrics['source_quality']:.3f}, "
                    f"tool_efficiency={metrics['tool_efficiency']:.3f}; {llm_summary}"
                    f"median judge score {score:.3f}."
                    + (f" Retrieval notes: {'; '.join(retrieval_notes)}" if retrieval_notes else "")
                ),
                passed=score >= self.pass_threshold,
            )
            tokens_after = self.llm.total_prompt_tokens + self.llm.total_completion_tokens
            _record_timing_trace(
                store,
                self.run_id,
                agent_name=f"research_eval:{variant.id}",
                role="research_variant_agent",
                prompt=variant.payload,
                model=self.llm.model_label,
                started_at=started_at,
                started=started,
                status="completed",
                output_summary=evaluation.summary,
                token_usage=tokens_after - tokens_before,
                tools_used=[backend.tool_name],
                tool_calls=[
                    {
                        "tool": backend.tool_name,
                        "requested_tool": retriever_name,
                        "query": variant.payload,
                        "results": len(sources),
                        "fallback_used": bool(retrieval_notes),
                    }
                ],
            )
            return evaluation
        except Exception as exc:
            tokens_after = self.llm.total_prompt_tokens + self.llm.total_completion_tokens
            _record_timing_trace(
                store,
                self.run_id,
                agent_name=f"research_eval:{variant.id}",
                role="research_variant_agent",
                prompt=variant.payload,
                model=self.llm.model_label,
                started_at=started_at,
                started=started,
                status="failed",
                output_summary="Research variant evaluation failed.",
                token_usage=tokens_after - tokens_before,
                tools_used=[retriever_name],
                errors=[f"{type(exc).__name__}: {exc}"],
            )
            raise

    async def _search_with_fallback(
        self,
        retriever_name: str,
        variant: Variant,
        limit: int,
        store: ArtifactStore,
    ) -> tuple[SearchBackend, list[tuple[object, float]], list[str]]:
        backend = self.search_factory(retriever_name)
        notes: list[str] = []
        store.append_progress(f"Retriever search: {retriever_name} for {variant.id} (limit={limit})")
        try:
            results = await _search_backend_with_retry(backend, variant.payload, limit)
            store.append_progress(f"Retriever done: {retriever_name} for {variant.id} returned {len(results)} result(s)")
            if results or retriever_name == "local":
                return backend, results, notes
            notes.append(f"{retriever_name} returned no results")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            store.add_failed_path(
                FailedPath(
                    description=f"Retriever '{retriever_name}' failed for variant {variant.id}",
                    reason=message,
                    created_by_agent=f"research_loop:{variant.id}",
                    run_id=self.run_id,
                )
            )
            store.append_progress(f"Retriever fallback: {retriever_name} failed for {variant.id}: {message}")
            notes.append(f"{retriever_name} failed ({type(exc).__name__})")
            if retriever_name == "local":
                return backend, [], notes
        fallback_names = _retriever_fallbacks(retriever_name)
        last_backend = backend
        for fallback_name in fallback_names:
            fallback_backend = self.search_factory(fallback_name)
            last_backend = fallback_backend
            store.append_progress(f"Retriever search: {fallback_name} fallback for {variant.id} (limit={limit})")
            try:
                results = await _search_backend_with_retry(fallback_backend, variant.payload, limit)
            except Exception as fallback_exc:
                fallback_message = f"{type(fallback_exc).__name__}: {fallback_exc}"
                store.add_failed_path(
                    FailedPath(
                        description=f"Fallback retriever '{fallback_name}' failed for variant {variant.id}",
                        reason=fallback_message,
                        created_by_agent=f"research_loop:{variant.id}",
                        run_id=self.run_id,
                    )
                )
                store.append_progress(f"Retriever fallback: {fallback_name} failed for {variant.id}: {fallback_message}")
                notes.append(f"{fallback_name} fallback failed ({type(fallback_exc).__name__})")
                continue
            notes.append(f"{fallback_name} fallback used")
            store.append_progress(f"Retriever done: {fallback_name} fallback for {variant.id} returned {len(results)} result(s)")
            if results or fallback_name == "local":
                return fallback_backend, results, notes
        return last_backend, [], notes

    def _llm_judge_score(
        self, variant: Variant, metrics: dict[str, float], source_count: int, claim_count: int
    ) -> tuple[Optional[float], str]:
        if not self.llm.is_live:
            return None, ""
        system = (
            "You are a research-loop judge. Return JSON only with keys score and rationale. "
            "Score from 0 to 1 based on evidence coverage, corroboration, relevance, and likely utility "
            "of the query variant for the user's research goal."
        )
        user = json.dumps(
            {
                "query_variant": variant.payload,
                "metadata": variant.metadata,
                "source_count": source_count,
                "claim_count": claim_count,
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
        try:
            payload = self.llm.complete_json(system, user, max_output_tokens=350)
            score = max(0.0, min(1.0, float(payload.get("score", 0.0))))
            rationale = str(payload.get("rationale", "LLM judge returned a score."))
            return round(score, 3), f"LLM judge: {rationale} "
        except Exception as exc:
            return None, f"LLM judge unavailable ({type(exc).__name__}). "

    def _research_metrics(self, sources: list[object], claim_count: int) -> dict[str, float]:
        if not sources:
            return {
                "coverage": 0.0,
                "corroboration": 0.0,
                "credibility": 0.0,
                "factual_accuracy": 0.0,
                "citation_accuracy": 0.0,
                "completeness": 0.0,
                "source_quality": 0.0,
                "tool_efficiency": 0.0,
            }
        credibility = sum(float(source.credibility_score) for source in sources) / len(sources)
        coverage = round(min(1.0, len(sources) / 5), 3)
        corroboration = round(min(1.0, claim_count / 10), 3)
        credibility = round(credibility, 3)
        citation_accuracy = 1.0 if claim_count > 0 else 0.0
        tool_efficiency = round(max(0.0, min(1.0, 1.15 - (max(0, len(sources) - 8) * 0.08))), 3)
        return {
            "coverage": coverage,
            "corroboration": corroboration,
            "credibility": credibility,
            "factual_accuracy": round((credibility * 0.65) + (corroboration * 0.35), 3),
            "citation_accuracy": citation_accuracy,
            "completeness": coverage,
            "source_quality": credibility,
            "tool_efficiency": tool_efficiency,
        }


class OptimizationQueryLoop:
    """Inner loop for the optimize_query task mode.

    Composes a ResearchLoop for retrieval and base scoring, then augments each
    result with optimization-specific metrics (novelty, implementability,
    evaluator relevance) before storing and ranking.  It does NOT inherit from
    ResearchLoop: the two loops have different purposes and the research phase
    here is a means to an end, not the final product.
    """

    mode: TaskMode = "optimize_query"

    def __init__(self, run_id: str, search_factory: SearchFactory, llm: Optional[LLMClient] = None, pass_threshold: float = 0.78):
        self.run_id = run_id
        self.llm = llm or LLMClient()
        self.pass_threshold = pass_threshold
        self._research_loop = ResearchLoop(run_id, search_factory, llm, pass_threshold)

    async def evaluate(self, variants: list[Variant], store: ArtifactStore) -> InnerLoopResult:
        # Run retrieval + base research scoring in parallel.  We call
        # _evaluate_variant directly to get the research result without storing
        # a research-typed VariantEvaluation — we'll store the augmented
        # optimize_query-typed one ourselves below.
        research_evals = await asyncio.gather(
            *(self._research_loop._evaluate_variant(variant, store) for variant in variants)
        )
        augmented: list[VariantEvaluation] = []
        for research_eval, variant in zip(research_evals, variants):
            oq_eval = self._augment(research_eval, variant)
            store.add_variant_evaluation(oq_eval)
            augmented.append(oq_eval)
        ranked = sorted(augmented, key=lambda e: e.score, reverse=True)
        signal = "claim_corroboration_threshold" if ranked and ranked[0].score >= self.pass_threshold else "continue"
        return InnerLoopResult(ranked_evaluations=ranked, termination_signal=signal)

    def _augment(self, research_eval: VariantEvaluation, variant: Variant) -> VariantEvaluation:
        metrics = dict(research_eval.metrics)
        metrics["evidence_coverage"] = metrics.get("coverage", 0.0)
        metrics["novelty"] = _novelty_score(variant.payload)
        metrics["implementability"] = _implementability_score(variant.payload)
        metrics["evaluator_relevance"] = _evaluator_relevance_score(variant.payload, str(variant.metadata.get("evaluator_name", "")))
        judge_scores = list(research_eval.judge_scores) + [
            metrics["novelty"],
            metrics["implementability"],
            metrics["evaluator_relevance"],
        ]
        llm_score, llm_summary = self._llm_judge_score(variant, metrics)
        if llm_score is not None:
            judge_scores.append(llm_score)
        score = round(median(judge_scores), 3)
        return VariantEvaluation(
            run_id=research_eval.run_id,
            variant_id=research_eval.variant_id,
            inner_loop="optimize_query",
            score=score,
            metrics=metrics,
            judge_scores=judge_scores,
            summary=(
                research_eval.summary
                + f" novelty={metrics['novelty']:.3f}; "
                + f"implementability={metrics['implementability']:.3f}; "
                + f"evaluator_relevance={metrics['evaluator_relevance']:.3f}. "
                + llm_summary
            ),
            passed=score >= self.pass_threshold,
        )

    def _llm_judge_score(self, variant: Variant, metrics: dict[str, float]) -> tuple[Optional[float], str]:
        if not self.llm.is_live:
            return None, ""
        system = (
            "You are judging whether a query result will help solve an optimization challenge. "
            "Return JSON only with keys score and rationale. Score 0 to 1 for actionable strategy value, "
            "implementability, and relevance to the evaluator."
        )
        user = json.dumps(
            {
                "query_variant": variant.payload,
                "metadata": variant.metadata,
                "metrics": metrics,
            },
            indent=2,
            sort_keys=True,
        )
        try:
            payload = self.llm.complete_json(system, user, max_output_tokens=350)
            score = max(0.0, min(1.0, float(payload.get("score", 0.0))))
            return round(score, 3), f"Optimization-query LLM judge: {payload.get('rationale', '')}"
        except Exception as exc:
            return None, f"Optimization-query LLM judge unavailable ({type(exc).__name__})."


class PlateauDetector:
    _RECOVERY_ACTIONS = ("new_strategy_family", "literature_mechanism", "alternative_evaluator", "fresh_search_context")

    def __init__(self, mode: TaskMode, patience: Optional[int] = None):
        self.mode = mode
        self.best_score = 0.0
        self.plateau_count = 0
        self.epsilon = 0.005 if mode == "optimize" else 0.03
        self.patience = patience if patience is not None else (2 if mode == "optimize" else 3)
        self._recovery_cycle = 0

    def update(self, score: float) -> str:
        if score > self.best_score + self.epsilon:
            self.best_score = score
            self.plateau_count = 0
            return "improved"
        self.plateau_count += 1
        if self.plateau_count >= self.patience:
            return "coverage_plateau" if self.mode == "research" else "score_plateau"
        return "continue"

    def next_recovery(self, seed_text: str) -> str:
        """Return a context-derived recovery action and advance the audit counter."""
        digest = hashlib.sha256(f"{seed_text}|{self.mode}|{self.plateau_count}|{self._recovery_cycle}".encode("utf-8")).hexdigest()
        action = self._RECOVERY_ACTIONS[int(digest[:8], 16) % len(self._RECOVERY_ACTIONS)]
        self._recovery_cycle += 1
        return action


@dataclass(frozen=True)
class LoopObjective:
    kind: str = "score"
    target: Optional[float] = None
    no_stop_until_target: bool = False

    @property
    def has_explicit_target(self) -> bool:
        return self.target is not None


@dataclass(frozen=True)
class DirectionSpec:
    slot: int
    strategy_family: str
    mechanism_hypothesis: str
    entropy_role: str
    parent_policy: str
    eval_protocol: str
    regime_focus: str
    convergence_lane: str
    ablation_target: str = ""


class EvolutionaryOuterLoop:
    def __init__(
        self,
        run_id: str,
        goal: str,
        task_mode: TaskMode,
        source_strategy: list[SourceStrategyItem],
        search_factory: SearchFactory,
        evaluator: Optional[EvaluatorFn] = None,
        evaluator_name: Optional[str] = None,
        llm: Optional[LLMClient] = None,
        max_outer_iterations: int = 4,
        population_size: int = 4,
        parent_count: int = 2,
        parallel_evaluator_cap: int = 8,
        optimize_plateau_patience: int = 2,
        continue_on_optimize_plateau: bool = False,
        force_direction_entropy: bool = True,
        novelty_fraction: float = 0.25,
        prior_run_memory: Optional[dict[str, object]] = None,
    ):
        self.run_id = run_id
        self.goal = goal
        self.task_mode = task_mode
        self.source_strategy = source_strategy
        self.search_factory = search_factory
        self.evaluator = evaluator
        self.evaluator_name = evaluator_name or ""
        self.llm = llm or LLMClient()
        self.max_outer_iterations = max_outer_iterations
        self.population_size = population_size
        self.parent_count = max(1, parent_count)
        self.parallel_evaluator_cap = max(1, parallel_evaluator_cap)
        self.optimize_plateau_patience = max(1, optimize_plateau_patience)
        self.continue_on_optimize_plateau = continue_on_optimize_plateau
        self.force_direction_entropy = force_direction_entropy
        self.novelty_fraction = max(0.0, min(0.8, novelty_fraction))
        self.prior_run_memory = prior_run_memory or {}
        self.objective = _loop_objective_from_goal(goal, evaluator_name)
        self._champion_variant_id: Optional[str] = None
        self._champion_payload: str = ""
        self._champion_score: float = 0.0
        # One-shot recovery flags set by _apply_plateau_recovery() and
        # consumed at the start of the next _propose_*_variants() call.
        self._recovery_forced_retriever: Optional[str] = None
        self._recovery_temperature: float = 0.7
        self._recovery_inject_mutation: bool = False
        self._recovery_entropy_intent: Optional[dict[str, object]] = None
        self._recovery_retriever_index: int = 0

    async def run(self, store: ArtifactStore) -> None:
        store.ingest_pending_user_steering(self.run_id)
        if self.task_mode in {"optimize", "optimize_query"}:
            await self._record_literature_grounding(store, "initial")
        if self.task_mode == "optimize_query":
            await self._run_optimize_query(store)
            return
        inner_loop = self._inner_loop()
        plateau = PlateauDetector(self.task_mode, patience=self.optimize_plateau_patience if self.task_mode == "optimize" else None)
        parents: list[Variant] = []
        last_result: Optional[InnerLoopResult] = None
        last_variants: list[Variant] = []
        best_eval: Optional[VariantEvaluation] = None
        best_variants: list[Variant] = []
        for outer_iteration in range(1, self.max_outer_iterations + 1):
            store.ingest_pending_user_steering(self.run_id)
            propose_started = time.perf_counter()
            propose_started_at = now_iso()
            variants = self._propose_variants(outer_iteration, parents, store)
            _record_timing_trace(
                store,
                self.run_id,
                agent_name=f"orchestration:propose_{self.task_mode}_round_{outer_iteration}",
                role="orchestration",
                prompt=f"Propose {self.task_mode} variants for round {outer_iteration}",
                model="deterministic-orchestrator",
                started_at=propose_started_at,
                started=propose_started,
                status="completed",
                output_summary=f"Proposed {len(variants)} {self.task_mode} variant(s).",
            )
            last_variants = variants
            persist_started = time.perf_counter()
            persist_started_at = now_iso()
            for variant in variants:
                store.add_variant(variant)
            store.append_progress(f"Outer {outer_iteration}: proposed {len(variants)} {self.task_mode} variants")
            for variant in variants:
                store.append_progress(f"  Variant {variant.id}: {_shorten(variant.payload)}")
            _record_timing_trace(
                store,
                self.run_id,
                agent_name=f"orchestration:persist_{self.task_mode}_round_{outer_iteration}",
                role="orchestration",
                prompt=f"Persist {self.task_mode} variants for round {outer_iteration}",
                model="deterministic-orchestrator",
                started_at=persist_started_at,
                started=persist_started,
                status="completed",
                output_summary=f"Persisted {len(variants)} variant(s) and progress entries.",
            )
            result = await inner_loop.evaluate(variants, store)
            rank_started = time.perf_counter()
            rank_started_at = now_iso()
            last_result = result
            round_best = result.ranked_evaluations[0] if result.ranked_evaluations else None
            if round_best and (best_eval is None or round_best.score > best_eval.score):
                best_eval = round_best
                best_variants = variants
            plateau_signal = plateau.update(round_best.score if round_best else 0.0)
            termination_signal = result.termination_signal
            if termination_signal == "continue":
                termination_signal = plateau_signal
            store.add_evolution_round(
                EvolutionRound(
                    run_id=self.run_id,
                    outer_iteration=outer_iteration,
                    mode=self.task_mode,
                    variant_ids=[variant.id for variant in variants],
                    best_variant_id=round_best.variant_id if round_best else None,
                    best_score=round_best.score if round_best else 0.0,
                    termination_signal=termination_signal,
                    plateau_count=plateau.plateau_count,
                )
            )
            store.append_progress(
                f"Outer {outer_iteration}: mode={self.task_mode} best_score="
                f"{round_best.score if round_best else 0.0:.3f} signal={termination_signal}"
            )
            for evaluation in result.ranked_evaluations[:3]:
                store.append_progress(
                    f"  Score {evaluation.score:.3f} for {evaluation.variant_id}: {_shorten(evaluation.summary)}"
                )
            self._promote_round_champion(store, outer_iteration, variants, result.ranked_evaluations, loop_name=self.task_mode)
            winner_ids = {evaluation.variant_id for evaluation in result.ranked_evaluations[: self.parent_count]}
            parents = [variant for variant in variants if variant.id in winner_ids]
            _record_timing_trace(
                store,
                self.run_id,
                agent_name=f"orchestration:rank_select_{self.task_mode}_round_{outer_iteration}",
                role="orchestration",
                prompt=f"Rank and select parents for round {outer_iteration}",
                model="deterministic-orchestrator",
                started_at=rank_started_at,
                started=rank_started,
                status="completed",
                output_summary=f"Selected {len(parents)} parent(s); signal={termination_signal}.",
            )
            if termination_signal in {"score_plateau", "coverage_plateau"}:
                self._apply_plateau_recovery(plateau, store, outer_iteration, termination_signal)
            should_stop = self._should_stop_outer_loop(termination_signal, round_best, outer_iteration)
            if not should_stop and outer_iteration >= self.max_outer_iterations:
                should_stop = True
            self._record_continuation_decision(
                store,
                loop_name="lead_researcher_outer_loop",
                iteration=outer_iteration,
                mode=self.task_mode,
                should_continue=not should_stop,
                termination_signal=termination_signal,
                best_score=round_best.score if round_best else 0.0,
                plateau_count=plateau.plateau_count,
                reason=_continuation_reason(termination_signal, round_best, plateau.plateau_count, outer_iteration, self.max_outer_iterations),
            )
            if should_stop:
                break
        if self.task_mode == "optimize" and last_result:
            self._write_optimization_outputs(store, best_variants or last_variants, best_eval)

    def _inner_loop(self) -> InnerLoop:
        if self.task_mode == "optimize":
            if self.evaluator is None:
                raise ValueError("OptimizeLoop requires a deterministic evaluator.")
            return OptimizeLoop(self.run_id, self.evaluator, parallel_evaluator_cap=self.parallel_evaluator_cap)
        if self.task_mode == "optimize_query":
            return OptimizationQueryLoop(self.run_id, self.search_factory, self.llm)
        return ResearchLoop(self.run_id, self.search_factory, self.llm)

    def _propose_variants(
        self,
        outer_iteration: int,
        parents: list[Variant],
        store: Optional[ArtifactStore] = None,
    ) -> list[Variant]:
        if self.task_mode == "optimize":
            return self._propose_code_variants(outer_iteration, parents, store)
        return self._propose_query_variants(outer_iteration, parents, store)

    async def _run_optimize_query(self, store: ArtifactStore) -> None:
        query_loop = OptimizationQueryLoop(self.run_id, self.search_factory, self.llm)
        plateau = PlateauDetector("research")
        parents: list[Variant] = []
        last_result: Optional[InnerLoopResult] = None
        for outer_iteration in range(1, self.max_outer_iterations + 1):
            store.ingest_pending_user_steering(self.run_id)
            propose_started = time.perf_counter()
            propose_started_at = now_iso()
            variants = self._propose_query_variants(outer_iteration, parents, store)
            _record_timing_trace(
                store,
                self.run_id,
                agent_name=f"orchestration:propose_query_round_{outer_iteration}",
                role="orchestration",
                prompt=f"Propose optimize-query variants for round {outer_iteration}",
                model="deterministic-orchestrator",
                started_at=propose_started_at,
                started=propose_started,
                status="completed",
                output_summary=f"Proposed {len(variants)} query variant(s).",
            )
            persist_started = time.perf_counter()
            persist_started_at = now_iso()
            for variant in variants:
                variant.metadata.setdefault("challenge_goal", self.goal)
                variant.metadata.setdefault("evaluator_name", self.evaluator_name)
                variant.metadata.setdefault("query_intent", "optimization challenge strategy discovery")
                store.add_variant(variant)
            store.append_progress(f"Optimization-query phase {outer_iteration}: proposed {len(variants)} query variants")
            for variant in variants:
                store.append_progress(f"  Query {variant.id}: {_shorten(variant.payload)}")
            _record_timing_trace(
                store,
                self.run_id,
                agent_name=f"orchestration:persist_query_round_{outer_iteration}",
                role="orchestration",
                prompt=f"Persist optimize-query variants for round {outer_iteration}",
                model="deterministic-orchestrator",
                started_at=persist_started_at,
                started=persist_started,
                status="completed",
                output_summary=f"Persisted {len(variants)} query variant(s).",
            )
            result = await query_loop.evaluate(variants, store)
            rank_started = time.perf_counter()
            rank_started_at = now_iso()
            last_result = result
            best_eval = result.ranked_evaluations[0] if result.ranked_evaluations else None
            plateau_signal = plateau.update(best_eval.score if best_eval else 0.0)
            termination_signal = result.termination_signal
            if termination_signal == "continue":
                termination_signal = plateau_signal
            store.add_evolution_round(
                EvolutionRound(
                    run_id=self.run_id,
                    outer_iteration=outer_iteration,
                    mode="optimize_query",
                    variant_ids=[variant.id for variant in variants],
                    best_variant_id=best_eval.variant_id if best_eval else None,
                    best_score=best_eval.score if best_eval else 0.0,
                    termination_signal=termination_signal,
                    plateau_count=plateau.plateau_count,
                )
            )
            store.append_progress(
                f"Optimization-query phase {outer_iteration}: best_score={best_eval.score if best_eval else 0.0:.3f} signal={termination_signal}"
            )
            for evaluation in result.ranked_evaluations[:3]:
                store.append_progress(f"  Query score {evaluation.score:.3f} for {evaluation.variant_id}: {_shorten(evaluation.summary)}")
            winner_ids = {evaluation.variant_id for evaluation in result.ranked_evaluations[: self.parent_count]}
            parents = [variant for variant in variants if variant.id in winner_ids]
            _record_timing_trace(
                store,
                self.run_id,
                agent_name=f"orchestration:rank_select_query_round_{outer_iteration}",
                role="orchestration",
                prompt=f"Rank optimize-query findings for round {outer_iteration}",
                model="deterministic-orchestrator",
                started_at=rank_started_at,
                started=rank_started,
                status="completed",
                output_summary=f"Selected {len(parents)} query parent(s); signal={termination_signal}.",
            )
            if termination_signal in {"coverage_plateau", "claim_corroboration_threshold"}:
                if not self._should_stop_query_loop(termination_signal, outer_iteration):
                    self._apply_plateau_recovery(plateau, store, outer_iteration, termination_signal)
            should_stop = self._should_stop_query_loop(termination_signal, outer_iteration)
            if not should_stop and outer_iteration >= self.max_outer_iterations:
                should_stop = True
            self._record_continuation_decision(
                store,
                loop_name="lead_researcher_query_loop",
                iteration=outer_iteration,
                mode="optimize_query",
                should_continue=not should_stop,
                termination_signal=termination_signal,
                best_score=best_eval.score if best_eval else 0.0,
                plateau_count=plateau.plateau_count,
                reason=_continuation_reason(termination_signal, best_eval, plateau.plateau_count, outer_iteration, self.max_outer_iterations),
            )
            if should_stop:
                break

        seed_started = time.perf_counter()
        seed_started_at = now_iso()
        store.ingest_pending_user_steering(self.run_id)
        seed_context = self._build_optimizer_seed_context(store, last_result)
        store.write_optimizer_seed_context(seed_context)
        store.append_progress(f"Optimizer seed context: {store.optimizer_seed_context_path}")
        _record_timing_trace(
            store,
            self.run_id,
            agent_name="orchestration:build_seed_context",
            role="orchestration",
            prompt="Build optimizer seed context from top query findings",
            model="deterministic-orchestrator",
            started_at=seed_started_at,
            started=seed_started,
            status="completed",
            output_summary=f"Built seed context with {len(seed_context.get('top_query_findings', [])) if isinstance(seed_context.get('top_query_findings'), list) else 0} finding(s).",
        )
        if self.evaluator is None:
            store.append_progress("Optimizer phase skipped: no evaluator was registered for optimize_query mode.")
            return

        store.append_progress("Optimizer phase: starting code/strategy variants from query seed context")
        seed_parents = self._seed_context_variants(seed_context)
        if self.evaluator_name == "prediction_market":
            await self._run_prediction_market_optimizer(store, seed_parents, seed_context)
            return
        optimize_loop = OptimizeLoop(self.run_id, self.evaluator, parallel_evaluator_cap=self.parallel_evaluator_cap)
        await self._run_generic_optimizer_rounds(store, optimize_loop, seed_parents, seed_context)

    async def _run_generic_optimizer_rounds(
        self,
        store: ArtifactStore,
        optimize_loop: OptimizeLoop,
        parents: list[Variant],
        seed_context: dict[str, object],
    ) -> None:
        plateau = PlateauDetector("optimize", patience=self.optimize_plateau_patience)
        best_eval: Optional[VariantEvaluation] = None
        best_round_variants: list[Variant] = []
        for round_index in range(1, self.max_outer_iterations + 1):
            store.ingest_pending_user_steering(self.run_id)
            propose_started = time.perf_counter()
            propose_started_at = now_iso()
            code_variants = self._propose_code_variants(round_index, parents, store)
            _record_timing_trace(
                store,
                self.run_id,
                agent_name=f"orchestration:propose_code_round_{round_index}",
                role="orchestration",
                prompt=f"Propose optimizer code variants for round {round_index}",
                model="deterministic-orchestrator",
                started_at=propose_started_at,
                started=propose_started,
                status="completed",
                output_summary=f"Proposed {len(code_variants)} code variant(s).",
            )
            persist_started = time.perf_counter()
            persist_started_at = now_iso()
            for variant in code_variants:
                variant.metadata["optimizer_seed_context_path"] = str(store.optimizer_seed_context_path)
                variant.metadata["query_seed_summary"] = seed_context.get("summary", "")
                store.add_variant(variant)
            _record_timing_trace(
                store,
                self.run_id,
                agent_name=f"orchestration:persist_code_round_{round_index}",
                role="orchestration",
                prompt=f"Persist optimizer code variants for round {round_index}",
                model="deterministic-orchestrator",
                started_at=persist_started_at,
                started=persist_started,
                status="completed",
                output_summary=f"Persisted {len(code_variants)} code variant(s).",
            )
            result = await optimize_loop.evaluate(code_variants, store)
            rank_started = time.perf_counter()
            rank_started_at = now_iso()
            round_best = result.ranked_evaluations[0] if result.ranked_evaluations else None
            if round_best and (best_eval is None or round_best.score > best_eval.score):
                best_eval = round_best
                best_round_variants = code_variants
            plateau_signal = plateau.update(round_best.score if round_best else 0.0)
            termination_signal = result.termination_signal if result.termination_signal != "continue" else plateau_signal
            store.add_evolution_round(
                EvolutionRound(
                    run_id=self.run_id,
                    outer_iteration=(len(store.list("evolution_rounds")) + 1),
                    mode="optimize",
                    variant_ids=[variant.id for variant in code_variants],
                    best_variant_id=round_best.variant_id if round_best else None,
                    best_score=round_best.score if round_best else 0.0,
                    termination_signal=termination_signal,
                    plateau_count=plateau.plateau_count,
                )
            )
            store.append_progress(
                f"Optimizer phase round {round_index}: best_score={round_best.score if round_best else 0.0:.3f} signal={termination_signal}"
            )
            for evaluation in result.ranked_evaluations[:3]:
                store.append_progress(f"  Optimize score {evaluation.score:.3f} for {evaluation.variant_id}: {_shorten(evaluation.summary)}")
            self._promote_round_champion(store, round_index, code_variants, result.ranked_evaluations, loop_name="optimizer_loop")
            parents = [
                variant
                for variant in code_variants
                if variant.id in {evaluation.variant_id for evaluation in result.ranked_evaluations[: self.parent_count]}
            ]
            _record_timing_trace(
                store,
                self.run_id,
                agent_name=f"orchestration:rank_select_code_round_{round_index}",
                role="orchestration",
                prompt=f"Rank optimizer code variants for round {round_index}",
                model="deterministic-orchestrator",
                started_at=rank_started_at,
                started=rank_started,
                status="completed",
                output_summary=f"Selected {len(parents)} code parent(s); signal={termination_signal}.",
            )
            if termination_signal == "continue" and plateau.plateau_count == 1 and round_index > 1:
                await self._record_literature_grounding(store, f"optimizer_first_stall_round_{round_index}")
            if termination_signal == "score_plateau":
                await self._record_literature_grounding(store, f"optimizer_plateau_round_{round_index}")
                self._apply_plateau_recovery(plateau, store, round_index, termination_signal)
            should_stop = self._should_stop_optimizer_loop(termination_signal, round_best, round_index)
            if not should_stop and round_index >= self.max_outer_iterations:
                should_stop = True
            self._record_continuation_decision(
                store,
                loop_name="optimizer_loop",
                iteration=round_index,
                mode="optimize",
                should_continue=not should_stop,
                termination_signal=termination_signal,
                best_score=round_best.score if round_best else 0.0,
                plateau_count=plateau.plateau_count,
                reason=_continuation_reason(termination_signal, round_best, plateau.plateau_count, round_index, self.max_outer_iterations),
            )
            if should_stop:
                break
        self._write_optimization_outputs(store, best_round_variants, best_eval)

    async def _run_prediction_market_optimizer(
        self,
        store: ArtifactStore,
        parents: list[Variant],
        seed_context: dict[str, object],
    ) -> None:
        plateau = PlateauDetector("optimize", patience=self.optimize_plateau_patience)
        best_eval: Optional[VariantEvaluation] = None
        best_round_variants: list[Variant] = []
        for round_index in range(1, self.max_outer_iterations + 1):
            store.ingest_pending_user_steering(self.run_id)
            propose_started = time.perf_counter()
            propose_started_at = now_iso()
            code_variants = self._propose_prediction_market_variants(round_index, parents, store)
            _record_timing_trace(
                store,
                self.run_id,
                agent_name=f"orchestration:propose_prediction_market_round_{round_index}",
                role="orchestration",
                prompt=f"Propose prediction-market strategy variants for round {round_index}",
                model="deterministic-orchestrator",
                started_at=propose_started_at,
                started=propose_started,
                status="completed",
                output_summary=f"Proposed {len(code_variants)} prediction-market variant(s).",
            )
            persist_started = time.perf_counter()
            persist_started_at = now_iso()
            for variant in code_variants:
                variant.metadata["optimizer_seed_context_path"] = str(store.optimizer_seed_context_path)
                variant.metadata["query_seed_summary"] = seed_context.get("summary", "")
                store.add_variant(variant)
            _record_timing_trace(
                store,
                self.run_id,
                agent_name=f"orchestration:persist_prediction_market_round_{round_index}",
                role="orchestration",
                prompt=f"Persist prediction-market strategy variants for round {round_index}",
                model="deterministic-orchestrator",
                started_at=persist_started_at,
                started=persist_started,
                status="completed",
                output_summary=f"Persisted {len(code_variants)} prediction-market variant(s).",
            )
            semaphore = asyncio.Semaphore(self.parallel_evaluator_cap)
            evaluations = await asyncio.gather(
                *(self._evaluate_prediction_market_variant_capped(variant, store, round_index, semaphore) for variant in code_variants)
            )
            rank_started = time.perf_counter()
            rank_started_at = now_iso()
            for evaluation in evaluations:
                store.add_variant_evaluation(evaluation)
            eligible_evaluations = [evaluation for evaluation in evaluations if evaluation.metrics.get("score_eligible", True)]
            ranked = sorted(eligible_evaluations, key=lambda item: item.score, reverse=True)
            round_best = ranked[0] if ranked else None
            observed_ranked = sorted(evaluations, key=lambda item: item.score, reverse=True)
            observed_best = round_best or (observed_ranked[0] if observed_ranked else None)
            if round_best and (best_eval is None or round_best.score > best_eval.score):
                best_eval = round_best
                best_round_variants = code_variants
            elif best_eval is None and observed_best is not None:
                best_eval = observed_best
                best_round_variants = code_variants
            plateau_signal = plateau.update(round_best.score if round_best else 0.0)
            objective_met = self._prediction_market_objective_met(round_best)
            if self.objective.has_explicit_target:
                termination_signal = "profit_target" if objective_met else plateau_signal
            else:
                termination_signal = "score_threshold" if round_best and round_best.score >= 0.8 else plateau_signal
            store.add_evolution_round(
                EvolutionRound(
                    run_id=self.run_id,
                    outer_iteration=(len(store.list("evolution_rounds")) + 1),
                    mode="optimize",
                    variant_ids=[variant.id for variant in code_variants],
                    best_variant_id=observed_best.variant_id if observed_best else None,
                    best_score=round_best.score if round_best else 0.0,
                    termination_signal=termination_signal,
                    plateau_count=plateau.plateau_count,
                )
            )
            store.append_progress(
                f"Prediction-market optimizer round {round_index}: best_edge={_pm_edge_from_eval(observed_best):.3f} "
                f"score={round_best.score if round_best else 0.0:.3f} signal={termination_signal}"
            )
            for evaluation in observed_ranked[:3]:
                source = evaluation.metrics.get("score_source", "unknown")
                eligible = "eligible" if evaluation.metrics.get("score_eligible", True) else "unmeasured"
                store.append_progress(f"  Prediction-market score {evaluation.score:.3f} ({eligible}) via {source} for {evaluation.variant_id}: {_shorten(evaluation.summary)}")
            self._promote_round_champion(store, round_index, code_variants, observed_ranked, loop_name="challenge_optimizer_loop")
            parents = [variant for variant in code_variants if variant.id in {evaluation.variant_id for evaluation in ranked[: self.parent_count]}]
            if not parents:
                parents = code_variants[: self.parent_count]
            _record_timing_trace(
                store,
                self.run_id,
                agent_name=f"orchestration:rank_select_prediction_market_round_{round_index}",
                role="orchestration",
                prompt=f"Rank prediction-market variants for round {round_index}",
                model="deterministic-orchestrator",
                started_at=rank_started_at,
                started=rank_started,
                status="completed",
                output_summary=f"Selected {len(parents)} strategy parent(s); signal={termination_signal}.",
            )
            if (
                termination_signal == "continue"
                and plateau.plateau_count == 1
                and round_index > 1
                and round_best is not None
                and round_best.metrics.get("score_eligible", True)
            ):
                await self._record_literature_grounding(store, f"prediction_market_first_stall_round_{round_index}")
            if termination_signal == "score_plateau":
                await self._record_literature_grounding(store, f"prediction_market_plateau_round_{round_index}")
                self._apply_plateau_recovery(plateau, store, round_index, termination_signal)
            should_stop = self._should_stop_optimizer_loop(termination_signal, round_best, round_index)
            if not should_stop and round_index >= self.max_outer_iterations:
                should_stop = True
            self._record_continuation_decision(
                store,
                loop_name="challenge_optimizer_loop",
                iteration=round_index,
                mode="optimize",
                should_continue=not should_stop,
                termination_signal=termination_signal,
                best_score=round_best.score if round_best else 0.0,
                plateau_count=plateau.plateau_count,
                reason=_continuation_reason(termination_signal, round_best, plateau.plateau_count, round_index, self.max_outer_iterations),
            )
            if should_stop:
                break
        self._write_optimization_outputs(store, best_round_variants, best_eval)

    def _record_continuation_decision(
        self,
        store: ArtifactStore,
        *,
        loop_name: str,
        iteration: int,
        mode: TaskMode,
        should_continue: bool,
        termination_signal: str,
        best_score: float,
        plateau_count: int,
        reason: str,
    ) -> None:
        decision = "continue" if should_continue else "exit"
        next_action = "spawn/refine subagents for another loop" if should_continue else "exit loop and synthesize/persist outputs"
        started = time.perf_counter()
        started_at = now_iso()
        store.add_loop_continuation_decision(
            LoopContinuationDecision(
                run_id=self.run_id,
                loop_name=loop_name,
                iteration=iteration,
                mode=mode,
                decision=decision,
                reason=reason,
                termination_signal=termination_signal,
                best_score=round(best_score, 3),
                plateau_count=plateau_count,
                next_action=next_action,
            )
        )
        _record_timing_trace(
            store,
            self.run_id,
            agent_name=f"loop_controller:{loop_name}:round_{iteration}",
            role="loop_controller",
            prompt=f"{loop_name} round {iteration} signal={termination_signal}",
            model="deterministic-loop-controller",
            started_at=started_at,
            started=started,
            status="completed",
            output_summary=f"Decision: {decision}. {reason}",
        )
        store.append_progress(f"Loop decision {loop_name} round {iteration}: {decision} - {reason}")

    def _direction_specs(
        self,
        outer_iteration: int,
        parents: list[Variant],
        store: Optional[ArtifactStore],
    ) -> list[DirectionSpec]:
        if not self.force_direction_entropy:
            return []
        return _forced_direction_specs(
            goal=self.goal,
            task_mode=self.task_mode,
            population_size=self.population_size,
            outer_iteration=outer_iteration,
            novelty_fraction=self.novelty_fraction,
            champion_payload=self._champion_payload,
            parents=parents,
            store=store,
        )

    def _apply_direction_specs(self, variants: list[Variant], directions: list[DirectionSpec], *, kind: str) -> None:
        if not directions:
            return
        for index, variant in enumerate(variants):
            direction = directions[index % len(directions)]
            variant.metadata.update(_direction_metadata(direction, kind=kind))
            if direction.parent_policy in {"ignore_parents", "ignore_champion"}:
                variant.parent_ids = []
            if direction.entropy_role == "ablation":
                variant.metadata["ablation_round"] = True
            if direction.entropy_role == "multi_model_convergence":
                variant.metadata["convergence_check"] = {
                    "lane": direction.convergence_lane,
                    "compare_mechanism": direction.mechanism_hypothesis,
                }

    def _promote_round_champion(
        self,
        store: ArtifactStore,
        round_index: int,
        variants: list[Variant],
        evaluations: list[VariantEvaluation],
        *,
        loop_name: str,
    ) -> None:
        if not evaluations:
            return
        variant_lookup = {variant.id: variant for variant in variants}
        round_winner = evaluations[0]
        round_variant = variant_lookup.get(round_winner.variant_id)
        if round_variant is None:
            return
        promoted_global = round_winner.score >= self._champion_score
        if promoted_global:
            self._champion_variant_id = round_variant.id
            self._champion_payload = round_variant.payload
            self._champion_score = round_winner.score
        champion_payload = {
            "run_id": self.run_id,
            "round_index": round_index,
            "loop_name": loop_name,
            "round_winner": {
                "variant_id": round_variant.id,
                "score": round_winner.score,
                "parent_ids": round_variant.parent_ids,
                "payload": round_variant.payload,
                "metrics": round_winner.metrics,
                "summary": round_winner.summary,
            },
            "global_champion": {
                "variant_id": self._champion_variant_id,
                "score": self._champion_score,
                "payload": self._champion_payload,
                "promoted_this_round": promoted_global,
            },
        }
        path = store.write_round_champion(round_index, champion_payload)
        self._write_champion_tree(store)
        store.append_progress(
            f"Champion promotion round {round_index}: winner={round_variant.id} "
            f"score={round_winner.score:.3f}; global={self._champion_variant_id}; artifact={path}"
        )

    def _write_champion_tree(self, store: ArtifactStore) -> None:
        evaluations = {str(row.get("variant_id")): row for row in store.list("variant_evaluations")}
        rounds = store.list("evolution_rounds")
        round_winners = {str(row.get("best_variant_id")) for row in rounds if row.get("best_variant_id")}
        nodes = []
        edges = []
        for row in store.list("variants"):
            variant_id = str(row.get("id"))
            evaluation = evaluations.get(variant_id, {})
            parent_ids = [str(parent_id) for parent_id in row.get("parent_ids", [])]
            nodes.append(
                {
                    "id": variant_id,
                    "outer_iteration": row.get("outer_iteration"),
                    "kind": row.get("kind"),
                    "parent_ids": parent_ids,
                    "score": float(evaluation.get("score", 0.0) or 0.0),
                    "is_round_winner": variant_id in round_winners,
                    "is_global_champion": variant_id == self._champion_variant_id,
                    "highlight": "global_champion" if variant_id == self._champion_variant_id else ("round_winner" if variant_id in round_winners else "candidate"),
                    "summary": evaluation.get("summary", ""),
                }
            )
            for parent_id in parent_ids:
                edges.append({"from": parent_id, "to": variant_id})
        store.write_champion_tree(
            {
                "run_id": self.run_id,
                "global_champion_variant_id": self._champion_variant_id,
                "global_champion_score": self._champion_score,
                "nodes": nodes,
                "edges": edges,
            }
        )

    def _should_stop_outer_loop(
        self,
        termination_signal: str,
        best_eval: Optional[VariantEvaluation],
        outer_iteration: int,
    ) -> bool:
        if self.task_mode == "optimize" and termination_signal == "score_plateau" and self.continue_on_optimize_plateau:
            return outer_iteration >= self.max_outer_iterations
        if termination_signal in {"score_plateau", "coverage_plateau"} and self.objective.no_stop_until_target:
            return outer_iteration >= self.max_outer_iterations
        if termination_signal == "score_threshold" and self.objective.has_explicit_target:
            return self._generic_objective_met(best_eval)
        # Mirror _should_stop_query_loop: enforce at least 2 rounds for research
        # so a single high-scoring-but-irrelevant retrieval can't exit early.
        if self.task_mode == "research" and termination_signal == "claim_corroboration_threshold":
            min_rounds = min(2, self.max_outer_iterations)
            return outer_iteration >= min_rounds
        return termination_signal in {"score_threshold", "claim_corroboration_threshold", "score_plateau", "coverage_plateau"}

    def _should_stop_query_loop(self, termination_signal: str, outer_iteration: int) -> bool:
        minimum_query_rounds = 2 if self.objective.no_stop_until_target else 1
        if outer_iteration < min(self.max_outer_iterations, minimum_query_rounds):
            return False
        if termination_signal == "coverage_plateau" and self.objective.no_stop_until_target:
            return outer_iteration >= self.max_outer_iterations
        return termination_signal in {"claim_corroboration_threshold", "coverage_plateau"}

    def _should_stop_optimizer_loop(
        self,
        termination_signal: str,
        best_eval: Optional[VariantEvaluation],
        round_index: int,
    ) -> bool:
        if self.objective.has_explicit_target:
            if termination_signal in {"profit_target", "score_threshold"}:
                return self._objective_met(best_eval)
            return round_index >= self.max_outer_iterations
        if termination_signal == "score_plateau" and self.objective.no_stop_until_target:
            return round_index >= self.max_outer_iterations
        return termination_signal in {"score_threshold", "profit_target"}

    def _objective_met(self, best_eval: Optional[VariantEvaluation]) -> bool:
        if self.evaluator_name == "prediction_market":
            return self._prediction_market_objective_met(best_eval)
        return self._generic_objective_met(best_eval)

    def _generic_objective_met(self, best_eval: Optional[VariantEvaluation]) -> bool:
        if not best_eval:
            return False
        if self.objective.target is None:
            return best_eval.score >= 0.8
        return best_eval.score >= self.objective.target

    def _prediction_market_objective_met(self, best_eval: Optional[VariantEvaluation]) -> bool:
        if not best_eval:
            return False
        if not best_eval.metrics.get("score_eligible", best_eval.metrics.get("official_measured", False)):
            return False
        edge = _pm_edge_from_eval(best_eval)
        if self.objective.target is None:
            return best_eval.score >= 0.8
        return edge >= self.objective.target

    async def _evaluate_prediction_market_variant_capped(
        self,
        variant: Variant,
        store: ArtifactStore,
        round_index: int,
        semaphore: asyncio.Semaphore,
    ) -> VariantEvaluation:
        async with semaphore:
            return await self._evaluate_prediction_market_variant(variant, store, round_index)

    async def _record_literature_grounding(self, store: ArtifactStore, reason: str) -> None:
        if any(
            claim.get("created_by_agent") == "literature_grounding_policy"
            and str(claim.get("text", "")).startswith(f"Literature grounding ({reason})")
            for claim in store.list("claims")
        ):
            return
        started = time.perf_counter()
        started_at = now_iso()
        query = self._literature_grounding_query(reason, store)
        strategy_index = 0
        if "plateau" in reason and self.source_strategy:
            strategy_index = min(len(self.source_strategy) - 1, self._recovery_retriever_index % len(self.source_strategy))
        item = self.source_strategy[strategy_index] if self.source_strategy else None
        retriever_name = item.retriever if item else "local"
        limit = min(3, item.limit if item else 3)
        notes: list[str] = []
        store.append_progress(f"Literature grounding ({reason}): searching {retriever_name} for existing evidence")
        try:
            backend = self.search_factory(retriever_name)
            results = await asyncio.to_thread(backend.search, query, limit)
        except Exception as exc:
            notes.append(f"{retriever_name} failed ({type(exc).__name__}: {exc})")
            store.append_progress(f"Retriever fallback: {retriever_name} failed during literature grounding: {type(exc).__name__}: {exc}")
            backend = self.search_factory("local")
            results = await asyncio.to_thread(backend.search, query, limit)
        sources = []
        for document, relevance in results[:limit]:
            source = store.add_source(backend.to_source(document, relevance))
            sources.append((source, document))
        for source, document in sources:
            claim_text = document.claims[0] if document.claims else document.summary
            store.add_claim(
                Claim(
                    text=f"Literature grounding ({reason}) found: {claim_text}",
                    source_ids=[source.id],
                    confidence=0.74,
                    support_level="retrieved",
                    created_by_agent="literature_grounding_policy",
                    run_id=self.run_id,
                )
            )
        store.append_progress(
            f"Literature grounding ({reason}): query='{query}' retrieved {len(sources)} source(s)"
            + (f"; notes={' ; '.join(notes)}" if notes else "")
        )
        _record_timing_trace(
            store,
            self.run_id,
            agent_name=f"memory:literature_grounding:{reason}",
            role="memory",
            prompt=query,
            model="deterministic-memory-policy",
            started_at=started_at,
            started=started,
            status="completed",
            output_summary=f"Grounded optimization context with {len(sources)} source(s).",
        )

    def _literature_grounding_query(self, reason: str, store: Optional[ArtifactStore] = None) -> str:
        if self.evaluator_name == "prediction_market" and ("plateau" in reason or "stall" in reason):
            return _prediction_market_plateau_literature_query(store)
        context_parts = [self.goal, reason.replace("_", " ")]
        if store:
            for item in _score_history(store, mode="optimize", limit=3):
                context_parts.append(str(item.get("summary", "")))
                context_parts.append(str(item.get("payload", ""))[:240])
            context_parts.extend(_recent_literature_grounding_notes(store, limit=3))
            context_parts.extend(_recent_user_steering_notes(store, limit=3))
        terms = _context_terms(" ".join(context_parts), limit=28)
        return " ".join(terms) if terms else self.goal

    def _apply_plateau_recovery(self, plateau: PlateauDetector, store: ArtifactStore, round_index: int, reason: str) -> None:
        """Apply a context-derived recovery action and set one-shot flags for the next proposal round."""
        # Deduplicate: don't re-apply the same reason twice.
        already_applied = any(
            str(s.get("url", "")).startswith(f"memory://plateau-recovery/{self.run_id}/")
            and str(s.get("url", "")).endswith(f"/{reason}")
            for s in store.list("sources")
        )
        if already_applied:
            return

        # Reset all flags; the chosen action will set exactly one.
        self._recovery_forced_retriever = None
        self._recovery_temperature = 0.7
        self._recovery_inject_mutation = False
        self._recovery_entropy_intent = None

        history_mode = "optimize" if self.task_mode in {"optimize", "optimize_query"} else "research"
        score_context = json.dumps(_score_history(store, mode=history_mode, limit=4), sort_keys=True)
        action = plateau.next_recovery(f"{self.goal}|{reason}|{score_context}")
        entropy_intent = _plateau_entropy_intent(action, self.goal, score_context, self.evaluator_name)
        if history_mode == "optimize":
            old_population = self.population_size
            self.population_size = min(64, max(64 if self.population_size >= 48 else 32, self.population_size + self.parent_count, int(self.population_size * 1.25)))
            self._recovery_temperature = 1.15
            entropy_intent["population_before"] = old_population
            entropy_intent["population_after"] = self.population_size

        if action == "fresh_search_context":
            retrievers = [item.retriever for item in self.source_strategy] or ["local"]
            retriever = retrievers[self._recovery_retriever_index % len(retrievers)]
            self._recovery_retriever_index += 1
            self._recovery_forced_retriever = retriever
            entropy_intent["search_context"] = retriever
            action_note = f"fresh search context via {retriever}"
        elif action == "new_strategy_family":
            self._recovery_inject_mutation = True
            action_note = str(entropy_intent["exploration_path"])
        else:
            self._recovery_temperature = max(self._recovery_temperature, 1.05)
            action_note = str(entropy_intent["exploration_path"])
        self._recovery_entropy_intent = entropy_intent

        store.append_progress(
            f"Plateau entropy round {round_index} ({reason}): {action_note}; "
            f"expected_generalization={entropy_intent['expected_generalization']}"
        )

        # Record a traceable source/claim so the recovery appears in the artifact trail.
        source = store.add_source(
            Source(
                url=f"memory://plateau-recovery/{self.run_id}/{round_index}/{reason}",
                title=f"Plateau entropy: {action} at round {round_index}",
                author="research-harness",
                date=now_iso().split("T")[0],
                source_type="memory",
                summary=(
                    f"Loop plateaued ({reason}). Introduced meaningful entropy through {action_note}. "
                    f"Expected to improve generalization: {entropy_intent['expected_generalization']}. "
                    "This is not a hyperparameter-only retry."
                ),
                relevance_score=0.9,
                credibility_score=0.8,
                evidence_sections={
                    "exploration_path": str(entropy_intent["exploration_path"]),
                    "expected_generalization": str(entropy_intent["expected_generalization"]),
                    "evidence_basis": str(entropy_intent["evidence_basis"]),
                    "anti_reward_hack_rule": str(entropy_intent["anti_reward_hack_rule"]),
                },
            )
        )
        store.add_claim(
            Claim(
                text=(
                    f"Round {round_index} plateaued ({reason}); exploration path '{action}' is expected to improve "
                    f"generalization because {entropy_intent['expected_generalization']}"
                ),
                source_ids=[source.id],
                confidence=0.82,
                support_level="instrumented",
                created_by_agent="plateau_recovery_policy",
                run_id=self.run_id,
            )
        )

    async def _evaluate_prediction_market_variant(
        self,
        variant: Variant,
        store: ArtifactStore,
        round_index: int,
    ) -> VariantEvaluation:
        code = self._render_optimal_code(variant.payload)
        candidate_path = store.candidates_dir / f"round_{round_index:02d}_{variant.id}.py"
        candidate_path.write_text(code, encoding="utf-8")
        store.append_progress(f"  Candidate eval start {variant.id}: {candidate_path}")
        result = await asyncio.to_thread(_run_prediction_market_official, candidate_path)
        edge = float(result.get("mean_edge", 0.0))
        score_eligible = bool(result.get("score_eligible", result.get("official_measured", False)))
        score = _normalize_prediction_market_edge(edge) if score_eligible else 0.0
        result["candidate_path"] = str(candidate_path)
        result["score_eligible"] = score_eligible
        result["direction"] = _variant_direction_metadata(variant)
        store.append_progress(f"  Candidate eval done {variant.id}: mean_edge={edge:.3f} score={score:.3f} eligible={score_eligible}")
        score_source = str(result.get("score_source", "unknown"))
        measured_label = "upstream orderbook-pm" if result.get("official_measured") else "local challenge fallback"
        return VariantEvaluation(
            run_id=self.run_id,
            variant_id=variant.id,
            inner_loop="optimize",
            score=score,
            metrics=result,
            judge_scores=[score],
            summary=(
                f"{measured_label} mean_edge={edge:.3f}; "
                f"score_source={score_source}; "
                f"score_eligible={score_eligible}; "
                f"successes={int(result.get('success_count', 0))}; failures={int(result.get('failure_count', 0))}; "
                f"candidate={candidate_path}."
            ),
            passed=score_eligible and score >= 0.8,
        )

    def _write_optimization_outputs(
        self,
        store: ArtifactStore,
        variants: list[Variant],
        best_eval: Optional[VariantEvaluation],
    ) -> None:
        if not best_eval:
            if self.evaluator_name == "prediction_market":
                self._write_unmeasured_prediction_market_result(store)
            return
        if self.evaluator_name == "prediction_market" and not best_eval.metrics.get("score_eligible", False):
            self._write_unmeasured_prediction_market_result(store, best_eval)
            return
        started = time.perf_counter()
        started_at = now_iso()
        best_variant = next((variant for variant in variants if variant.id == best_eval.variant_id), None)
        candidate = best_variant.payload if best_variant else ""
        store.write_optimized_candidate(candidate + "\n")
        store.append_progress(f"Optimized candidate: {store.optimized_candidate_path}")
        optimal_code = self._render_optimal_code(candidate)
        optimal_code_path = str(store.write_optimal_code(optimal_code))
        store.append_progress(f"Optimal code: {store.optimal_code_path}")
        solution = self._render_solution(candidate)
        solution_path = None
        if solution:
            solution_path = str(store.write_solution(solution))
            store.append_progress(f"Solution: {store.solution_path}")
        objective = _objective_metadata(self.evaluator_name)
        official_result = objective["official_result"]
        if self.evaluator_name == "prediction_market":
            official_result = {
                "measured": bool(best_eval.metrics.get("official_measured", False)),
                "score_eligible": bool(best_eval.metrics.get("score_eligible", False)),
                "profit_usd": best_eval.metrics.get("mean_edge"),
                "target_profit_usd": self.objective.target,
                "target_met": self._prediction_market_objective_met(best_eval),
                "score_source": best_eval.metrics.get("score_source", "unknown"),
                "sandbox_executed": best_eval.metrics.get("sandbox_executed", False),
                "docker_sandbox": best_eval.metrics.get("docker_sandbox", False),
                "actions_seen": best_eval.metrics.get("actions_seen"),
                "simulations": best_eval.metrics.get("simulations"),
                "required_evaluator": "https://github.com/danrobinson/prediction-market-challenge",
                "candidate_path": best_eval.metrics.get("candidate_path"),
                "success_count": best_eval.metrics.get("success_count"),
                "failure_count": best_eval.metrics.get("failure_count"),
            }
        payload = {
            "run_id": self.run_id,
            "evaluator_name": self.evaluator_name or "unknown",
            "objective_name": objective["objective_name"],
            "objective_direction": objective["objective_direction"],
            "score_variable": "score",
            "score": best_eval.score,
            "metrics": best_eval.metrics,
            "evaluator_responses": _json_evaluator_responses(store, mode="optimize"),
            "best_variant_id": best_eval.variant_id,
            "best_candidate_path": str(store.optimized_candidate_path),
            "optimal_code_path": optimal_code_path,
            "solution_path": solution_path,
            "candidate": candidate,
            "official_result": official_result,
            "objective_target": {
                "kind": self.objective.kind,
                "target": self.objective.target,
                "no_stop_until_target": self.objective.no_stop_until_target,
                "met": self._objective_met(best_eval),
            },
            "note": objective["note"],
        }
        store.write_optimization_result(payload)
        store.append_progress(
            f"Optimization result: {store.optimization_result_path} "
            f"({payload['objective_direction']} {payload['objective_name']}={best_eval.score:.3f})"
        )
        _record_timing_trace(
            store,
            self.run_id,
            agent_name="orchestration:write_optimization_outputs",
            role="orchestration",
            prompt="Write optimized candidate, optimal code, solution, and optimization result",
            model="deterministic-orchestrator",
            started_at=started_at,
            started=started,
            status="completed",
            output_summary=f"Wrote optimization outputs for best variant {best_eval.variant_id}.",
        )

    def _write_unmeasured_prediction_market_result(
        self,
        store: ArtifactStore,
        best_eval: Optional[VariantEvaluation] = None,
    ) -> None:
        objective = _objective_metadata(self.evaluator_name)
        metrics = best_eval.metrics if best_eval else {}
        candidate_path = metrics.get("candidate_path")
        payload = {
            "run_id": self.run_id,
            "evaluator_name": self.evaluator_name or "unknown",
            "objective_name": objective["objective_name"],
            "objective_direction": objective["objective_direction"],
            "score_variable": "score",
            "score": 0.0,
            "metrics": metrics,
            "evaluator_responses": _json_evaluator_responses(store, mode="optimize"),
            "best_variant_id": best_eval.variant_id if best_eval else None,
            "best_candidate_path": None,
            "optimal_code_path": None,
            "official_result": {
                "measured": False,
                "score_eligible": False,
                "profit_usd": metrics.get("mean_edge"),
                "target_profit_usd": self.objective.target,
                "target_met": False,
                "score_source": metrics.get("score_source", "unmeasured_official_scorer_unavailable"),
                "sandbox_executed": metrics.get("sandbox_executed", False),
                "docker_sandbox": metrics.get("docker_sandbox", False),
                "required_evaluator": "https://github.com/danrobinson/prediction-market-challenge",
                "candidate_path": candidate_path,
                "error": metrics.get("error") or metrics.get("sandbox_error"),
            },
            "objective_target": {
                "kind": self.objective.kind,
                "target": self.objective.target,
                "no_stop_until_target": self.objective.no_stop_until_target,
                "met": False,
            },
            "note": (
                "Prediction-market official scorer did not produce an eligible sandboxed score. "
                "No optimal_code.py or solution.py was promoted from fallback/proxy scoring."
            ),
        }
        store.write_optimization_result(payload)
        store.append_progress(
            "Optimization result marked unmeasured: official sandbox scorer unavailable or failed; "
            "no fallback winner promoted."
        )

    def _build_optimizer_seed_context(self, store: ArtifactStore, result: Optional[InnerLoopResult]) -> dict[str, object]:
        evaluations = result.ranked_evaluations if result else []
        variant_lookup = {row["id"]: row for row in store.list("variants")}
        claim_rows = store.list("claims")
        source_lookup = {row["id"]: row for row in store.list("sources")}
        top_items = []
        for evaluation in evaluations[:5]:
            variant = variant_lookup.get(evaluation.variant_id, {})
            variant_claims = [
                claim for claim in claim_rows if str(claim.get("created_by_agent", "")).endswith(f":{evaluation.variant_id}")
            ][:8]
            source_ids = {
                str(source_id)
                for claim in variant_claims
                for source_id in claim.get("source_ids", [])
                if str(source_id) in source_lookup
            }
            supporting_sources = [
                {
                    "id": source_id,
                    "title": source_lookup[source_id].get("title", ""),
                    "url": source_lookup[source_id].get("url", ""),
                    "summary": source_lookup[source_id].get("summary", ""),
                    "source_type": source_lookup[source_id].get("source_type", ""),
                }
                for source_id in sorted(source_ids)
            ][:6]
            top_items.append(
                {
                    "variant_id": evaluation.variant_id,
                    "query": variant.get("payload", ""),
                    "score": evaluation.score,
                    "metrics": evaluation.metrics,
                    "summary": evaluation.summary,
                    "supporting_claims": [
                        {
                            "id": claim.get("id", ""),
                            "text": claim.get("text", ""),
                            "confidence": claim.get("confidence", 0.0),
                            "source_ids": claim.get("source_ids", []),
                        }
                        for claim in variant_claims
                    ],
                    "supporting_sources": supporting_sources,
                }
            )
        summary_parts = []
        for item in top_items[:3]:
            claims = item.get("supporting_claims", [])
            claim_text = ""
            if isinstance(claims, list) and claims and isinstance(claims[0], dict):
                claim_text = f" -> {str(claims[0].get('text', ''))[:180]}"
            summary_parts.append(f"{item['query']}{claim_text}")
        user_steering = _recent_user_steering_notes(store, limit=8)
        summary = "; ".join(summary_parts)
        if user_steering:
            summary = "; ".join([summary, *user_steering]).strip("; ")
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "mode": "optimize_query",
            "summary": summary,
            "top_query_findings": top_items,
            "user_steering": user_steering,
            "optimizer_instruction": (
                "Use the retrieved supporting claims and source summaries as strategy context when proposing "
                "optimization variants. Treat user_steering as live user-provided context, not as proof of success. "
                "Do not rely on query wording alone."
            ),
            "has_evaluator": self.evaluator is not None,
            "evaluator_name": self.evaluator_name or None,
        }

    def _seed_context_variants(self, seed_context: dict[str, object]) -> list[Variant]:
        parents = []
        for item in seed_context.get("top_query_findings", []) if isinstance(seed_context.get("top_query_findings"), list) else []:
            if not isinstance(item, dict):
                continue
            parents.append(
                Variant(
                    run_id=self.run_id,
                    outer_iteration=0,
                    kind="query",
                    payload=str(item.get("query", "")),
                    parent_ids=[],
                    metadata={
                        "seed_score": item.get("score", 0.0),
                        "seed_summary": item.get("summary", ""),
                        "seed_literature": {
                            "claims": item.get("supporting_claims", []),
                            "sources": item.get("supporting_sources", []),
                        },
                    },
                )
            )
        return parents

    def _propose_query_variants(
        self,
        outer_iteration: int,
        parents: list[Variant],
        store: Optional[ArtifactStore] = None,
    ) -> list[Variant]:
        # Consume one-shot recovery flags so they only affect this round.
        forced_retriever = self._recovery_forced_retriever
        temperature = self._recovery_temperature
        entropy_intent = self._recovery_entropy_intent
        self._recovery_forced_retriever = None
        self._recovery_temperature = 0.7
        self._recovery_inject_mutation = False  # not used for query variants
        self._recovery_entropy_intent = None
        directions = self._direction_specs(outer_iteration, parents, store)

        llm_variants = self._llm_query_variants(
            outer_iteration,
            parents,
            temperature=temperature,
            forced_retriever=forced_retriever,
            directions=directions,
            store=store,
        )
        if llm_variants:
            self._apply_direction_specs(llm_variants, directions, kind="query")
            if entropy_intent:
                for variant in llm_variants:
                    variant.metadata["meaningful_entropy_intent"] = entropy_intent
            return llm_variants

        # Fallback (no live LLM): build variants from source strategy or parent mutations.
        if not parents:
            variants = []
            source_items = self.source_strategy or [
                SourceStrategyItem(
                    name="forced_direction_fallback",
                    retriever="local",
                    purpose="forced direction fallback",
                    queries=[self.goal],
                    limit=6,
                )
            ]
            for index, direction in enumerate(directions[: self.population_size]):
                item = source_items[index % len(source_items)]
                retriever = forced_retriever or item.retriever
                variants.append(
                    Variant(
                        run_id=self.run_id,
                        outer_iteration=outer_iteration,
                        kind="query",
                        payload=_directional_query_payload(item.queries[0], direction),
                        parent_ids=[],
                        metadata={
                            "retriever": retriever,
                            "purpose": item.purpose,
                            "limit": item.limit,
                            "research_role": "parallel_research_subagent",
                            "search_phase": "broad" if outer_iteration == 1 else "narrow",
                            **({"recovery": "fresh_search_context"} if forced_retriever else {}),
                            **({"meaningful_entropy_intent": entropy_intent} if entropy_intent else {}),
                        },
                    )
                )
            self._apply_direction_specs(variants, directions, kind="query")
            return variants

        suffixes = _contextual_query_suffixes(self.goal, parents, self.population_size)
        unresolved = [
            str(item)
            for item in self.prior_run_memory.get("unresolved_directions", [])
            if str(item).strip()
        ]
        user_steering = _recent_user_steering_notes(store, limit=4)
        if user_steering:
            steering_suffixes = [" ".join(_context_terms(note, limit=6)) for note in user_steering]
            suffixes = [suffix for suffix in steering_suffixes if suffix] + suffixes
        if unresolved:
            memory_suffixes = [_context_terms(item, limit=4) for item in unresolved]
            suffixes = [
                " ".join(terms)
                for terms in memory_suffixes
                if terms
            ] + suffixes
        variants = []
        for index, direction in enumerate(directions[: self.population_size]):
            suffix = suffixes[index % len(suffixes)] if suffixes else direction.strategy_family
            parent_pool = [] if direction.parent_policy == "ignore_parents" else parents
            parent = parent_pool[index % len(parent_pool)] if parent_pool else parents[index % len(parents)]
            retriever = forced_retriever or str(parent.metadata.get("retriever", "local"))
            variants.append(
                Variant(
                    run_id=self.run_id,
                    outer_iteration=outer_iteration,
                    kind="query",
                    payload=_directional_query_payload(f"{parent.payload} {suffix}", direction),
                    parent_ids=[] if direction.parent_policy == "ignore_parents" else [parent.id],
                    metadata={
                        **parent.metadata,
                        "retriever": retriever,
                        "search_phase": "narrow",
                        "narrowing_suffix": suffix,
                        "suffix_provenance": "context-derived from goal and parent variants",
                        **({"recovery": "fresh_search_context"} if forced_retriever else {}),
                        **({"meaningful_entropy_intent": entropy_intent} if entropy_intent else {}),
                    },
                )
            )
        self._apply_direction_specs(variants, directions, kind="query")
        return variants

    def _propose_code_variants(
        self,
        outer_iteration: int,
        parents: list[Variant],
        store: Optional[ArtifactStore] = None,
    ) -> list[Variant]:
        # Consume one-shot recovery flags.
        temperature = self._recovery_temperature
        inject_mutation = self._recovery_inject_mutation
        entropy_intent = self._recovery_entropy_intent
        self._recovery_forced_retriever = None
        self._recovery_temperature = 0.7
        self._recovery_inject_mutation = False
        self._recovery_entropy_intent = None
        directions = self._direction_specs(outer_iteration, parents, store)

        llm_variants = self._llm_code_variants(outer_iteration, parents, temperature=temperature, directions=directions, store=store)
        if llm_variants:
            self._apply_direction_specs(llm_variants, directions, kind="code")
            if entropy_intent:
                suffix = _entropy_payload_suffix(entropy_intent)
                for variant in llm_variants:
                    variant.payload = f"{variant.payload} {suffix}"
                    variant.metadata["meaningful_entropy_intent"] = entropy_intent
            return llm_variants

        # Fallback seeds (no live LLM).
        if not parents:
            seeds = [
                _directional_code_seed("baseline direct implementation", direction)
                for direction in directions
            ]
        else:
            if all(parent.kind == "query" for parent in parents):
                challenge_context = self._optimizer_seed_prefix()
                seeds = [
                    _directional_code_seed(
                        f"{challenge_context} Optimization variant informed by query finding: {parents[index % len(parents)].payload} refined pass {outer_iteration}",
                        direction,
                    )
                    for index, direction in enumerate(directions)
                ]
            else:
                seeds = []
                for index, direction in enumerate(directions):
                    parent_pool = [] if direction.parent_policy == "ignore_parents" else parents
                    parent = parent_pool[index % len(parent_pool)] if parent_pool else parents[index % len(parents)]
                    base = f"{parent.payload} refined pass {outer_iteration}" if direction.parent_policy != "ignore_parents" else f"fresh non-descendant implementation for {self.goal}"
                    seeds.append(_directional_code_seed(base, direction))
        if self._champion_payload:
            champion_prefix = (
                f"diff_against_champion champion_score={self._champion_score:.3f} "
                f"champion_variant={self._champion_variant_id} champion_payload='{self._champion_payload[:300]}'"
            )
            seeds = [
                payload if directions[index % len(directions)].parent_policy == "ignore_champion"
                else f"{champion_prefix} mutation={payload}"
                for index, payload in enumerate(seeds)
            ]

        if entropy_intent:
            suffix = _entropy_payload_suffix(entropy_intent)
            seeds = [f"{payload} {suffix}" for payload in seeds]
        user_steering = _recent_user_steering_notes(store, limit=2)
        if user_steering:
            seeds = [f"{payload} user_steering='{user_steering[index % len(user_steering)]}'" for index, payload in enumerate(seeds)]

        variants = [
            Variant(
                run_id=self.run_id,
                outer_iteration=outer_iteration,
                kind="code",
                payload=payload,
                parent_ids=[parent.id for parent in parents],
                metadata={
                    "goal": self.goal,
                    **({"meaningful_entropy_intent": entropy_intent} if entropy_intent else {}),
                },
            )
            for payload in seeds[: self.population_size]
        ]

        if inject_mutation:
            variants = [_randomly_mutate_variant(v, outer_iteration) for v in variants]

        self._apply_direction_specs(variants, directions, kind="code")
        return variants

    def _optimizer_seed_prefix(self) -> str:
        return "Optimization sketch derived only from the user goal, retrieved evidence, parent variants, and score feedback:"

    def _propose_prediction_market_variants(
        self,
        outer_iteration: int,
        parents: list[Variant],
        store: Optional[ArtifactStore] = None,
    ) -> list[Variant]:
        temperature = self._recovery_temperature
        inject_mutation = self._recovery_inject_mutation
        entropy_intent = self._recovery_entropy_intent
        self._recovery_forced_retriever = None
        self._recovery_temperature = 0.7
        self._recovery_inject_mutation = False
        self._recovery_entropy_intent = None
        directions = self._direction_specs(outer_iteration, parents, store)

        score_history = _score_history(store, mode="optimize") if store else []
        literature_notes = _recent_literature_grounding_notes(store) if store else []
        user_steering_notes = _recent_user_steering_notes(store) if store else []
        context_text = " ".join([self.goal, *literature_notes, *user_steering_notes])
        templates = [
            _contextual_prediction_market_payload(context_text, outer_iteration, index)
            for index in range(max(self.population_size, 4))
        ]
        if score_history:
            best = score_history[0]
            best_payload = str(best.get("payload") or "")
            best_edge = float(best.get("mean_edge") or 0.0)
            params = _prediction_market_params(best_payload)
            base_spread = int(params["spread"])
            base_size = float(params["size"])
            base_inventory = float(params["inventory"])
            base_skew = float(params["skew_divisor"])
            templates = [
                (
                    f"pm_strategy=contextual_score_memory prior_best_edge={best_edge:.3f} "
                    f"spread={max(2, base_spread + delta)} size={max(0.1, base_size + size_delta):.2f} "
                    f"inventory={base_inventory:.1f} skew={base_skew:.1f} parent='{best_payload[:120]}'"
                )
                for delta, size_delta in [(-2, -0.25), (2, -0.5), (4, -0.25), (8, -0.75)]
            ] + [
                (
                    f"pm_strategy=contextual_score_explore prior_best_edge={best_edge:.3f} "
                    f"spread={max(2, base_spread + delta)} size={max(0.1, base_size * 0.75):.2f} "
                    f"inventory={max(5.0, base_inventory * 0.75):.1f} skew={base_skew + skew_delta:.1f} "
                    f"parent='{best_payload[:120]}'"
                )
                for delta, skew_delta in [(6, 3), (10, 5)]
            ] + templates
        if parents and all(parent.kind == "code" for parent in parents):
            parent_mutations = []
            for parent in parents:
                params = _prediction_market_params(parent.payload)
                for delta, size_factor, inventory_factor, skew_delta in [
                    (2, 0.80, 0.90, 2),
                    (4, 0.65, 0.75, 4),
                    (8, 0.50, 0.60, 6),
                    (12, 0.35, 0.50, 8),
                ]:
                    parent_mutations.append(
                        f"pm_strategy=contextual_parent_mutation mutation_round={outer_iteration} "
                        f"spread={max(2, int(params['spread']) + delta)} size={max(0.05, float(params['size']) * size_factor):.2f} "
                        f"inventory={max(1.0, float(params['inventory']) * inventory_factor):.1f} "
                        f"skew={float(params['skew_divisor']) + skew_delta:.1f} "
                        f"parent='{parent.payload[:120]}'"
                    )
            templates = parent_mutations + templates
        if self._champion_payload:
            champion_params = _prediction_market_params(self._champion_payload)
            champion_templates = []
            for delta, size_factor, inventory_factor, skew_delta in [
                (-1, 1.05, 1.00, -1),
                (1, 0.95, 0.95, 1),
                (3, 0.80, 0.80, 3),
                (6, 0.60, 0.70, 5),
            ]:
                champion_templates.append(
                    f"pm_strategy=champion_diff champion_variant={self._champion_variant_id} "
                    f"champion_score={self._champion_score:.3f} "
                    f"spread={max(2, int(champion_params['spread']) + delta)} "
                    f"size={max(0.05, float(champion_params['size']) * size_factor):.2f} "
                    f"inventory={max(1.0, float(champion_params['inventory']) * inventory_factor):.1f} "
                    f"skew={max(1.0, float(champion_params['skew_divisor']) + skew_delta):.1f} "
                    f"parent='{self._champion_payload[:120]}'"
                )
            templates = champion_templates + templates
        elif parents:
            context = self._optimizer_seed_prefix()
            templates = [
                (
                    f"{context} {_literature_seed_note(parents[index % len(parents)])} "
                    f"{template} query_seed={parents[index % len(parents)].payload}"
                )
                for index, template in enumerate(templates)
            ]
        if literature_notes:
            templates = [
                f"{template} fresh_literature='{literature_notes[index % len(literature_notes)]}'"
                for index, template in enumerate(templates)
            ]
        if user_steering_notes:
            templates = [
                f"{template} user_steering='{user_steering_notes[index % len(user_steering_notes)]}'"
                for index, template in enumerate(templates)
            ]
        if entropy_intent:
            suffix = _entropy_payload_suffix(entropy_intent)
            templates = [f"{template} {suffix}" for template in templates]
        deterministic_variants = [
            Variant(
                run_id=self.run_id,
                outer_iteration=outer_iteration,
                kind="code",
                payload=_directional_code_seed(payload, directions[index % len(directions)]) if directions else payload,
                parent_ids=[parent.id for parent in parents],
                metadata={
                    "goal": self.goal,
                    "challenge": "prediction_market",
                    **({"meaningful_entropy_intent": entropy_intent} if entropy_intent else {}),
                },
            )
            for index, payload in enumerate(templates[: self.population_size])
        ]
        if inject_mutation:
            deterministic_variants = [_randomly_mutate_variant(variant, outer_iteration) for variant in deterministic_variants]
        self._apply_direction_specs(deterministic_variants, directions, kind="code")

        llm_variants = self._llm_prediction_market_code_variants(
            outer_iteration,
            parents,
            temperature=temperature,
            directions=directions,
            store=store,
        )
        self._apply_direction_specs(llm_variants, directions, kind="code")
        if entropy_intent:
            suffix = _entropy_payload_suffix(entropy_intent)
            for variant in llm_variants:
                variant.payload = f"{variant.payload}\n# {suffix}"
                variant.metadata["meaningful_entropy_intent"] = entropy_intent
        return _dedupe_prediction_market_variants(
            [*llm_variants, *deterministic_variants],
            store=store,
            population_size=self.population_size,
            outer_iteration=outer_iteration,
        )

    def _render_solution(self, payload: str) -> str:
        if self.evaluator_name != "prediction_market":
            return ""
        # If the LLM already wrote a complete Strategy class, use it directly.
        if "class Strategy" in payload and "BaseStrategy" in payload:
            return payload
        return _prediction_market_solution(payload)

    def _render_optimal_code(self, payload: str) -> str:
        if self.evaluator_name == "prediction_market":
            # LLM-generated real Python code takes precedence over the template.
            if "class Strategy" in payload and "BaseStrategy" in payload:
                return payload
            return _prediction_market_solution(payload)
        return _generic_optimal_code(payload, self.evaluator_name)

    def _llm_query_variants(
        self,
        outer_iteration: int,
        parents: list[Variant],
        *,
        temperature: float = 0.7,
        forced_retriever: Optional[str] = None,
        directions: Optional[list[DirectionSpec]] = None,
        store: Optional[ArtifactStore] = None,
    ) -> list[Variant]:
        if not self.llm.is_live:
            return []
        started = time.perf_counter()
        started_at = now_iso()
        tokens_before = self.llm.total_prompt_tokens + self.llm.total_completion_tokens
        parent_payloads = [parent.payload for parent in parents]
        strategy = [
            {"retriever": item.retriever, "purpose": item.purpose, "query": item.queries[0], "limit": item.limit}
            for item in self.source_strategy[: self.population_size]
        ]
        already_tried = list({p.payload for p in parents})
        avoid_directions = [
            str(item)
            for item in self.prior_run_memory.get("avoid_directions", [])
            if str(item).strip()
        ][:8]
        unresolved_directions = [
            str(item)
            for item in self.prior_run_memory.get("unresolved_directions", [])
            if str(item).strip()
        ][:8]
        recovery_note = (
            f"\nPLATEAU RECOVERY ACTIVE: forced_retriever={forced_retriever!r} — "
            "every variant MUST use this retriever to escape the current search angle."
            if forced_retriever else ""
        )
        system = (
            "You are the outer orchestrator in an evolutionary research harness. "
            "Propose diverse, independent query variants for parallel research subagents.\n\n"
            "DIVERSITY RULES (strictly enforced):\n"
            "- Each variant MUST cover a different aspect, angle, or information source.\n"
            "- No two variants may be semantically equivalent or differ only in wording.\n"
            "- Assign a different `retriever` to each variant when possible (use the available_strategy list).\n"
            "- Do NOT repeat or closely rephrase any query in the already_tried list.\n"
            "- Do NOT repeat prior_report_avoid_directions unless the user explicitly asked for replication.\n"
            "- Prefer prior_report_unresolved_directions when they are directly related to the current goal.\n"
            "- Iteration 1: start broad and wide. Later iterations: narrow based on parent findings.\n"
            "- Obey forced_directions exactly: each returned variant must test its assigned strategy_family and mechanism_hypothesis.\n"
            + recovery_note + "\n\n"
            "Return JSON only: {\"variants\": [{\"query\": str, \"retriever\": str, \"purpose\": str}]}"
        )
        user = json.dumps(
            {
                "goal": self.goal,
                "outer_iteration": outer_iteration,
                "parents": parent_payloads,
                "already_tried": already_tried,
                "prior_report_avoid_directions": avoid_directions,
                "prior_report_unresolved_directions": unresolved_directions,
                "user_steering": _recent_user_steering_notes(store, limit=6),
                "available_strategy": strategy,
                "forced_directions": [_direction_metadata(direction, kind="query") for direction in (directions or [])],
                "population_size": self.population_size,
                **({"forced_retriever": forced_retriever} if forced_retriever else {}),
            },
            indent=2,
            sort_keys=True,
        )
        try:
            payload = self.llm.complete_json(system, user, max_output_tokens=800, temperature=temperature)
        except Exception as exc:
            if store:
                tokens_after = self.llm.total_prompt_tokens + self.llm.total_completion_tokens
                _record_timing_trace(
                    store,
                    self.run_id,
                    agent_name=f"llm_propose_queries:round_{outer_iteration}",
                    role="llm_thinking",
                    prompt=user,
                    model=self.llm.model_label,
                    started_at=started_at,
                    started=started,
                    status="failed",
                    output_summary="LLM query proposal failed; fallback variants will be used.",
                    token_usage=tokens_after - tokens_before,
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
            return []
        rows = payload.get("variants", [])
        if not isinstance(rows, list):
            return []
        seen_queries: set[str] = set(already_tried)
        variants = []
        for index, row in enumerate(rows[: self.population_size]):
            if not isinstance(row, dict):
                continue
            query = str(row.get("query") or "").strip()
            if not query or query in seen_queries:
                continue
            seen_queries.add(query)
            retriever = forced_retriever or str(row.get("retriever") or "")
            if not retriever and self.source_strategy:
                retriever = self.source_strategy[index % len(self.source_strategy)].retriever
            elif not retriever:
                retriever = "local"
            variants.append(
                Variant(
                    run_id=self.run_id,
                    outer_iteration=outer_iteration,
                    kind="query",
                    payload=query,
                    parent_ids=[parent.id for parent in parents],
                    metadata={
                        "retriever": retriever,
                        "purpose": str(row.get("purpose") or "llm-proposed query"),
                        "limit": 8,
                        "research_role": "parallel_research_subagent",
                        "search_phase": "broad" if outer_iteration == 1 else "narrow",
                        **({"recovery_temperature": temperature} if temperature != 0.7 else {}),
                        **({"recovery": "fresh_search_context"} if forced_retriever else {}),
                    },
                )
            )
        if store:
            tokens_after = self.llm.total_prompt_tokens + self.llm.total_completion_tokens
            _record_timing_trace(
                store,
                self.run_id,
                agent_name=f"llm_propose_queries:round_{outer_iteration}",
                role="llm_thinking",
                prompt=user,
                model=self.llm.model_label,
                started_at=started_at,
                started=started,
                status="completed",
                output_summary=f"Proposed {len(variants)} query variants.",
                token_usage=tokens_after - tokens_before,
            )
        return variants

    def _llm_code_variants(
        self,
        outer_iteration: int,
        parents: list[Variant],
        *,
        temperature: float = 0.7,
        directions: Optional[list[DirectionSpec]] = None,
        store: Optional[ArtifactStore] = None,
    ) -> list[Variant]:
        if not self.llm.is_live:
            return []
        if self.evaluator_name == "prediction_market":
            return self._llm_prediction_market_code_variants(
                outer_iteration,
                parents,
                temperature=temperature,
                directions=directions,
                store=store,
            )
        started = time.perf_counter()
        started_at = now_iso()
        tokens_before = self.llm.total_prompt_tokens + self.llm.total_completion_tokens
        system = (
            "You are the outer orchestrator in an evolutionary optimization harness. "
            "Propose candidate code or strategy variants as JSON only: {\"variants\": [{\"payload\": str}]}. "
            "Obey forced_directions exactly: every variant must test its assigned strategy_family and mechanism_hypothesis."
        )
        user = json.dumps(
            {
                "goal": self.goal,
                "outer_iteration": outer_iteration,
                "parents": [parent.payload for parent in parents],
                "champion": {
                    "variant_id": self._champion_variant_id,
                    "score": self._champion_score,
                    "payload": self._champion_payload,
                    "instruction": "Every proposed variant should be a deliberate diff against the champion unless exploring a new strategy family.",
                },
                "forced_directions": [_direction_metadata(direction, kind="code") for direction in (directions or [])],
                "population_size": self.population_size,
                "score_history": _score_history(store, mode="optimize") if store else [],
                "user_steering": _recent_user_steering_notes(store, limit=6),
            },
            indent=2,
            sort_keys=True,
        )
        try:
            payload = self.llm.complete_json(system, user, max_output_tokens=800, temperature=temperature)
        except Exception as exc:
            if store:
                tokens_after = self.llm.total_prompt_tokens + self.llm.total_completion_tokens
                _record_timing_trace(
                    store,
                    self.run_id,
                    agent_name=f"llm_propose_code:round_{outer_iteration}",
                    role="llm_thinking",
                    prompt=user,
                    model=self.llm.model_label,
                    started_at=started_at,
                    started=started,
                    status="failed",
                    output_summary="LLM code proposal failed; fallback variants will be used.",
                    token_usage=tokens_after - tokens_before,
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
            return []
        rows = payload.get("variants", [])
        if not isinstance(rows, list):
            return []
        variants = []
        for row in rows[: self.population_size]:
            if not isinstance(row, dict):
                continue
            candidate = str(row.get("payload") or "").strip()
            if not candidate:
                continue
            variants.append(
                Variant(
                    run_id=self.run_id,
                    outer_iteration=outer_iteration,
                    kind="code",
                    payload=candidate,
                    parent_ids=[parent.id for parent in parents],
                    metadata={"goal": self.goal, "proposal_source": "llm"},
                )
            )
        if store:
            tokens_after = self.llm.total_prompt_tokens + self.llm.total_completion_tokens
            _record_timing_trace(
                store,
                self.run_id,
                agent_name=f"llm_propose_code:round_{outer_iteration}",
                role="llm_thinking",
                prompt=user,
                model=self.llm.model_label,
                started_at=started_at,
                started=started,
                status="completed",
                output_summary=f"Proposed {len(variants)} code variants.",
                token_usage=tokens_after - tokens_before,
            )
        return variants

    def _llm_prediction_market_code_variants(
        self,
        outer_iteration: int,
        parents: list[Variant],
        *,
        temperature: float = 0.7,
        directions: Optional[list[DirectionSpec]] = None,
        store: Optional[ArtifactStore] = None,
    ) -> list[Variant]:
        """Ask the LLM to write complete, executable Python Strategy classes.

        Unlike the generic variant path which produces parameter strings, this
        generates real `class Strategy(BaseStrategy)` implementations that can be
        written to disk and run directly against the upstream orderbook-pm grader.
        """
        parent_snippets = [p.payload[:600] for p in parents if "class Strategy" in p.payload]
        literature_seed_context = [_parent_literature_context(parent) for parent in parents]
        literature_seed_context = [item for item in literature_seed_context if item]
        system = (
            "You are an expert market-making strategy developer for prediction markets. "
            "Generate complete, executable Python Strategy classes for the upstream orderbook_pm_challenge evaluator.\n\n"
            "INTERFACE (must be respected exactly):\n"
            "```python\n"
            "from orderbook_pm_challenge.strategy import BaseStrategy\n"
            "from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState\n\n"
            "class Strategy(BaseStrategy):\n"
            "    def on_step(self, state: StepState):  # return list of actions\n"
            "        # state fields: competitor_best_bid_ticks, competitor_best_ask_ticks,\n"
            "        #   buy_filled_quantity, sell_filled_quantity,\n"
            "        #   yes_inventory, no_inventory, free_cash\n"
            "        # actions: CancelAll(), PlaceOrder(side=Side.BUY/SELL, price_ticks=int, quantity=float)\n"
            "        # price_ticks: 1-99 (cents)\n"
            "        ...\n"
            "```\n\n"
            "SCORING: mean_edge is computed by the evaluator from fills and market state. Higher is better. "
            "Infer strategy choices only from the user goal, parent strategies, retrieved evidence, and score history; "
            "do not assume a fixed named trading doctrine unless the prompt or retrieved evidence supports it. "
            "Obey forced_directions exactly: every variant must test its assigned strategy_family and mechanism_hypothesis.\n\n"
            f"Return JSON only: {{\"variants\": [{{\"payload\": \"<complete Python source>\", \"description\": str}}]}} "
            f"with exactly {self.population_size} variants. Each must be a fully self-contained Python file "
            "with the imports and class definition. No placeholders or TODO comments."
        )
        user = json.dumps(
            {
                "goal": self.goal,
                "outer_iteration": outer_iteration,
                "parent_strategies": parent_snippets,
                "champion": {
                    "variant_id": self._champion_variant_id,
                    "score": self._champion_score,
                    "payload": self._champion_payload[:1200],
                    "instruction": "Each strategy should either be a deliberate diff against this champion or a clearly labeled new strategy family.",
                },
                "literature_seed_context": literature_seed_context,
                "user_steering": _recent_user_steering_notes(store, limit=6),
                "forced_directions": [_direction_metadata(direction, kind="code") for direction in (directions or [])],
                "population_size": self.population_size,
                "instruction": (
                    "Generate diverse strategies that differ meaningfully in logic. "
                    "When literature_seed_context is present, each strategy must name the specific retrieved claim "
                    "or source insight that inspired its quoting, cancellation, inventory, or sizing logic. "
                    "Try meaningfully different approaches, but derive the differences from the prompt, parents, "
                    "retrieved evidence, and observed failure modes rather than a prewritten strategy list. "
                    "Each variant MUST compile as valid Python."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        started = time.perf_counter()
        started_at = now_iso()
        tokens_before = self.llm.total_prompt_tokens + self.llm.total_completion_tokens
        try:
            payload = self.llm.complete_json(system, user, max_output_tokens=3000, temperature=temperature)
        except Exception as exc:
            if store:
                tokens_after = self.llm.total_prompt_tokens + self.llm.total_completion_tokens
                _record_timing_trace(
                    store,
                    self.run_id,
                    agent_name=f"llm_propose_prediction_market_code:round_{outer_iteration}",
                    role="llm_thinking",
                    prompt=user,
                    model=self.llm.model_label,
                    started_at=started_at,
                    started=started,
                    status="failed",
                    output_summary="LLM prediction-market code proposal failed; fallback variants will be used.",
                    token_usage=tokens_after - tokens_before,
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
            return []
        rows = payload.get("variants", [])
        if not isinstance(rows, list):
            return []
        variants = []
        for row in rows[: self.population_size]:
            if not isinstance(row, dict):
                continue
            code = str(row.get("payload") or "").strip()
            if not code or "class Strategy" not in code:
                continue
            # Validate syntax before accepting the variant.
            try:
                compile(code, "<llm_variant>", "exec")
            except SyntaxError:
                continue
            variants.append(
                Variant(
                    run_id=self.run_id,
                    outer_iteration=outer_iteration,
                    kind="code",
                    payload=code,
                    parent_ids=[parent.id for parent in parents],
                    metadata={
                        "goal": self.goal,
                        "proposal_source": "llm_python",
                        "description": str(row.get("description", "")),
                        "challenge": "prediction_market",
                        **({"recovery_temperature": temperature} if temperature != 0.7 else {}),
                    },
                )
            )
        if store:
            tokens_after = self.llm.total_prompt_tokens + self.llm.total_completion_tokens
            _record_timing_trace(
                store,
                self.run_id,
                agent_name=f"llm_propose_prediction_market_code:round_{outer_iteration}",
                role="llm_thinking",
                prompt=user,
                model=self.llm.model_label,
                started_at=started_at,
                started=started,
                status="completed",
                output_summary=f"Proposed {len(variants)} prediction-market code variants.",
                token_usage=tokens_after - tokens_before,
            )
        return variants


def _record_timing_trace(
    store: ArtifactStore,
    run_id: str,
    *,
    agent_name: str,
    role: str,
    prompt: str,
    model: str,
    started_at: str,
    started: float,
    status: str,
    output_summary: str,
    token_usage: int = 0,
    tools_used: Optional[list[str]] = None,
    tool_calls: Optional[list[dict[str, object]]] = None,
    errors: Optional[list[str]] = None,
) -> None:
    store.add_trace(
        AgentTrace(
            run_id=run_id,
            agent_name=agent_name,
            role=role,
            prompt=prompt,
            model=model,
            tools_used=tools_used or [],
            tool_calls=tool_calls or [],
            token_usage=max(0, token_usage),
            runtime_ms=max(0, int((time.perf_counter() - started) * 1000)),
            status=status,
            errors=errors or [],
            output_summary=output_summary,
            started_at=started_at,
            prompt_version=hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
            prompt_tokens=max(0, token_usage),
            failure_component=_trace_component(role, agent_name),
        )
    )


def _trace_component(role: str, agent_name: str) -> str:
    text = f"{role} {agent_name}".lower()
    if any(term in text for term in ["search", "literature", "retriever", "memory"]):
        return "retrieval"
    if "hypothesis" in text:
        return "hypothesis_generation"
    if "critic" in text:
        return "critic"
    if "synthesis" in text:
        return "synthesis"
    if any(term in text for term in ["optimize", "evaluator", "prediction_market"]):
        return "optimizer"
    if "loop_controller" in text:
        return "loop_control"
    if "orchestration" in text:
        return "orchestration"
    return "unknown"


def _continuation_reason(
    termination_signal: str,
    best_eval: Optional[VariantEvaluation],
    plateau_count: int,
    iteration: int,
    max_iterations: int,
) -> str:
    score = best_eval.score if best_eval else 0.0
    if iteration >= max_iterations and termination_signal not in {"score_threshold", "claim_corroboration_threshold", "profit_target"}:
        return f"Iteration budget reached ({iteration}/{max_iterations}); exiting with best score {score:.3f}."
    if termination_signal in {"score_threshold", "claim_corroboration_threshold"}:
        return f"Quality threshold reached with best score {score:.3f}; exiting loop."
    if termination_signal == "profit_target":
        return f"Explicit profit target met with best score {score:.3f}; exiting loop."
    if termination_signal in {"score_plateau", "coverage_plateau"}:
        return f"Plateau detected after {plateau_count} stalled round(s); recovery may run, then loop exits unless objective requires more iterations."
    return f"More research/evaluation is needed; best score {score:.3f}, plateau count {plateau_count}."


def _score_history(store: Optional[ArtifactStore], *, mode: str, limit: int = 8) -> list[dict[str, object]]:
    if store is None:
        return []
    variants = {str(row.get("id")): row for row in store.list("variants")}
    rows = [row for row in store.list("variant_evaluations") if str(row.get("inner_loop")) == mode]
    rows.sort(key=lambda row: float(row.get("score", 0.0) or 0.0), reverse=True)
    history: list[dict[str, object]] = []
    for row in rows[:limit]:
        variant = variants.get(str(row.get("variant_id")), {})
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        history.append(
            {
                "variant_id": row.get("variant_id"),
                "score": row.get("score"),
                "mean_edge": metrics.get("mean_edge"),
                "score_source": metrics.get("score_source"),
                "payload": str(variant.get("payload", ""))[:900],
                "summary": str(row.get("summary", ""))[:400],
            }
        )
    return history


def _json_evaluator_responses(store: ArtifactStore, *, mode: str) -> list[dict[str, object]]:
    rows = [row for row in store.list("variant_evaluations") if str(row.get("inner_loop")) == mode]
    responses: list[dict[str, object]] = []
    for row in rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        response = metrics.get("json_response") if isinstance(metrics, dict) else None
        if not isinstance(response, dict):
            response = {
                "status": metrics.get("evaluator_status", "completed") if isinstance(metrics, dict) else "completed",
                "score": float(row.get("score", 0.0) or 0.0),
                "metrics": metrics,
                "diagnostics": metrics.get("diagnostics", {}) if isinstance(metrics, dict) else {},
                "loss_reason": metrics.get("loss_reason", "") if isinstance(metrics, dict) else "",
                "summary": row.get("summary", ""),
            }
        responses.append(
            {
                "variant_id": row.get("variant_id"),
                "inner_loop": row.get("inner_loop"),
                **response,
            }
        )
    return responses


def _forced_direction_specs(
    *,
    goal: str,
    task_mode: TaskMode,
    population_size: int,
    outer_iteration: int,
    novelty_fraction: float,
    champion_payload: str,
    parents: list[Variant],
    store: Optional[ArtifactStore],
) -> list[DirectionSpec]:
    if population_size <= 0:
        return []
    terms = _context_terms(goal, limit=8)
    topic = " ".join(terms[:4]) or "objective"
    score_patterns = _score_history(store, mode="optimize" if task_mode in {"optimize", "optimize_query"} else "research", limit=4) if store else []
    observed_failure = " ".join(str(item.get("summary", "")) for item in score_patterns)[:240]
    novelty_slots = max(1, int(round(population_size * novelty_fraction))) if population_size >= 4 else 1
    base_families = [
        ("control_baseline", f"establish a simple baseline for {topic}"),
        ("champion_diff", "make one deliberate change to the current champion and keep everything else fixed"),
        ("new_strategy_family", f"test a structurally different approach to {topic}"),
        ("mechanism_transfer", "import a mechanism from retrieved evidence or prior findings"),
        ("failure_mode_targeting", f"target the most visible failure mode: {observed_failure or 'unknown failure'}"),
        ("regime_specific_probe", "specialize behavior for a distinct input/data/evaluation regime"),
        ("constraint_relaxation", "relax or invert one assumption held by parent variants"),
        ("risk_control", "add a conservative guardrail or kill-switch for brittle cases"),
        ("ablation", "remove or disable one champion component to identify load-bearing features"),
        ("multi_model_convergence", "solve independently from a different reasoning lane and compare mechanisms"),
    ]
    specs: list[DirectionSpec] = []
    for slot in range(population_size):
        family, mechanism = base_families[slot % len(base_families)]
        entropy_role = family
        parent_policy = "use_parents"
        if slot >= population_size - novelty_slots or family == "new_strategy_family":
            entropy_role = "novelty"
            parent_policy = "ignore_parents"
        if family == "champion_diff" and champion_payload:
            parent_policy = "use_champion"
        if family == "ablation":
            entropy_role = "ablation"
            parent_policy = "use_champion" if champion_payload else "use_parents"
        if family == "multi_model_convergence":
            entropy_role = "multi_model_convergence"
            parent_policy = "ignore_champion"
        if outer_iteration > 1 and slot % 7 == 0:
            entropy_role = "regime_probe"
            family = "regime_specific_probe"
        regime_focus = [
            "common_random_seeds",
            "edge_case_regime",
            "high_variance_regime",
            "small_input_regime",
            "large_input_regime",
            "post_failure_regime",
        ][slot % 6]
        specs.append(
            DirectionSpec(
                slot=slot,
                strategy_family=family,
                mechanism_hypothesis=mechanism,
                entropy_role=entropy_role,
                parent_policy=parent_policy,
                eval_protocol="paired_crn_same_seeds_across_variants",
                regime_focus=regime_focus,
                convergence_lane=f"lane_{slot % 3}",
                ablation_target=_ablation_target(champion_payload, parents, slot) if family == "ablation" else "",
            )
        )
    return specs


def _ablation_target(champion_payload: str, parents: list[Variant], slot: int) -> str:
    text = champion_payload or " ".join(parent.payload for parent in parents[:2])
    terms = _context_terms(text, limit=12)
    if not terms:
        return "largest_or_most_recent_component"
    return terms[slot % len(terms)]


def _direction_metadata(direction: DirectionSpec, *, kind: str) -> dict[str, object]:
    return {
        "direction_slot": direction.slot,
        "strategy_family": direction.strategy_family,
        "mechanism_hypothesis": direction.mechanism_hypothesis,
        "entropy_role": direction.entropy_role,
        "parent_policy": direction.parent_policy,
        "eval_protocol": direction.eval_protocol,
        "paired_crn": True,
        "regime_focus": direction.regime_focus,
        "regime_metrics_requested": [direction.regime_focus, "overall", "failure_cases"],
        "convergence_lane": direction.convergence_lane,
        "ablation_target": direction.ablation_target,
        "variant_contract": f"{kind}_variant_must_test_named_mechanism",
    }


def _variant_direction_metadata(variant: Variant) -> dict[str, object]:
    keys = {
        "strategy_family",
        "mechanism_hypothesis",
        "entropy_role",
        "eval_protocol",
        "paired_crn",
        "regime_focus",
        "regime_metrics_requested",
        "convergence_lane",
        "ablation_target",
    }
    return {key: variant.metadata.get(key) for key in keys if key in variant.metadata}


def _directional_query_payload(payload: str, direction: DirectionSpec) -> str:
    return (
        f"{payload} strategy_family:{direction.strategy_family} "
        f"mechanism_hypothesis:{direction.mechanism_hypothesis} "
        f"regime_focus:{direction.regime_focus} entropy_role:{direction.entropy_role}"
    )


def _directional_code_seed(payload: str, direction: DirectionSpec) -> str:
    ablation = f" ablation_target={direction.ablation_target}" if direction.ablation_target else ""
    return (
        f"{payload} strategy_family={direction.strategy_family} "
        f"mechanism_hypothesis='{direction.mechanism_hypothesis}' "
        f"entropy_role={direction.entropy_role} parent_policy={direction.parent_policy} "
        f"eval_protocol={direction.eval_protocol} regime_focus={direction.regime_focus}{ablation}"
    )


def _prediction_market_plateau_literature_query(store: Optional[ArtifactStore]) -> str:
    history = _score_history(store, mode="optimize", limit=6)
    observations: list[str] = []
    for item in history:
        observations.append(_strip_run_artifacts(str(item.get("summary", ""))))
        observations.append(_strip_run_artifacts(str(item.get("payload", ""))[:360]))
        if item.get("mean_edge") is not None:
            observations.append(f"observed edge {item.get('mean_edge')}")
    context = " ".join(observations)
    terms = _context_terms(context, limit=24)
    if not terms:
        terms = _context_terms("prediction market observed scorer feedback stalled evaluation", limit=12)
    return " ".join([*terms[:16], "literature", "mechanism", "generalization"])


def _strip_run_artifacts(text: str) -> str:
    text = re.sub(r"\b(?:outputs|eval_outputs)/\S+", " ", text)
    text = re.sub(r"\b(?:variant|round)_[A-Za-z0-9_]+\b", " ", text)
    text = re.sub(r"\b[a-f0-9]{12,}\b", " ", text)
    text = re.sub(r"\b\w+\.py\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parent_literature_context(parent: Variant) -> dict[str, object]:
    seed = parent.metadata.get("seed_literature")
    if not isinstance(seed, dict):
        return {}
    claims = seed.get("claims") if isinstance(seed.get("claims"), list) else []
    sources = seed.get("sources") if isinstance(seed.get("sources"), list) else []
    trimmed_claims = [
        {
            "text": str(claim.get("text", ""))[:500],
            "confidence": claim.get("confidence", 0.0),
            "source_ids": claim.get("source_ids", []),
        }
        for claim in claims
        if isinstance(claim, dict)
    ][:5]
    trimmed_sources = [
        {
            "title": str(source.get("title", ""))[:240],
            "url": str(source.get("url", "")),
            "summary": str(source.get("summary", ""))[:500],
            "source_type": str(source.get("source_type", "")),
        }
        for source in sources
        if isinstance(source, dict)
    ][:4]
    if not trimmed_claims and not trimmed_sources:
        return {}
    return {
        "query": parent.payload,
        "claims": trimmed_claims,
        "sources": trimmed_sources,
    }


def _recent_literature_grounding_notes(store: Optional[ArtifactStore], limit: int = 4) -> list[str]:
    if not store:
        return []
    notes = [
        str(claim.get("text", ""))
        for claim in store.list("claims")
        if claim.get("created_by_agent") == "literature_grounding_policy"
    ]
    return [_shorten(note, 220) for note in notes[-limit:] if note]


def _recent_user_steering_notes(store: Optional[ArtifactStore], limit: int = 4) -> list[str]:
    if not store:
        return []
    notes = [
        str(claim.get("text", ""))
        for claim in store.list("claims")
        if claim.get("created_by_agent") == "user_steering"
    ]
    return [_shorten(note, 260) for note in notes[-limit:] if note]


def _literature_seed_note(parent: Variant) -> str:
    context = _parent_literature_context(parent)
    claims = context.get("claims", []) if context else []
    sources = context.get("sources", []) if context else []
    claim_note = ""
    source_note = ""
    if isinstance(claims, list) and claims and isinstance(claims[0], dict):
        claim_note = str(claims[0].get("text", ""))[:180]
    if isinstance(sources, list) and sources and isinstance(sources[0], dict):
        source_note = str(sources[0].get("title", ""))[:120]
    if claim_note:
        return f"literature_inspiration='{claim_note}'"
    if source_note:
        return f"literature_source='{source_note}'"
    return "literature_inspiration='none retrieved'"


def _dedupe_prediction_market_variants(
    variants: list[Variant],
    *,
    store: Optional[ArtifactStore],
    population_size: int,
    outer_iteration: int,
) -> list[Variant]:
    existing_signatures = set()
    if store:
        for row in store.list("variants"):
            payload = str(row.get("payload", ""))
            if row.get("kind") == "code" and payload:
                existing_signatures.add(_prediction_market_code_signature(payload))
    selected: list[Variant] = []
    selected_signatures: set[str] = set()
    for variant in variants:
        candidate = variant
        signature = _prediction_market_code_signature(candidate.payload)
        for attempt in range(1, 6):
            if signature not in existing_signatures and signature not in selected_signatures:
                break
            candidate = _randomly_mutate_variant(candidate, (outer_iteration * 97) + attempt)
            signature = _prediction_market_code_signature(candidate.payload)
        if signature in existing_signatures or signature in selected_signatures:
            continue
        candidate.metadata["rendered_code_hash"] = signature
        selected.append(candidate)
        selected_signatures.add(signature)
        if len(selected) >= population_size:
            break
    while len(selected) < population_size:
        index = len(selected)
        context = " ".join(variant.payload for variant in variants[-3:]) if variants else ""
        payload = _contextual_prediction_market_payload(context, outer_iteration, index + 17)
        fallback = Variant(
            run_id=variants[0].run_id if variants else "",
            outer_iteration=outer_iteration,
            kind="code",
            payload=payload,
            parent_ids=variants[0].parent_ids if variants else [],
            metadata={"challenge": "prediction_market", "proposal_source": "contextual_recovery"},
        )
        signature = _prediction_market_code_signature(fallback.payload)
        if signature not in existing_signatures and signature not in selected_signatures:
            fallback.metadata["rendered_code_hash"] = signature
            selected.append(fallback)
            selected_signatures.add(signature)
        else:
            break
    return selected


def _contextual_prediction_market_payload(context: str, outer_iteration: int, index: int) -> str:
    seed_material = f"{context}|{outer_iteration}|{index}"
    digest = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    raw = int(digest[:12], 16)
    spread = 2 + (raw % 29)
    size = round(0.1 + ((raw >> 5) % 24) / 10.0, 2)
    inventory = 1 + ((raw >> 11) % 120)
    skew = 1 + ((raw >> 17) % 30)
    terms = " ".join(_context_terms(context, limit=8))
    return (
        f"pm_strategy=contextual_candidate round={outer_iteration} index={index} "
        f"spread={spread} size={size:.2f} inventory={inventory} skew={skew} "
        f"context_terms='{terms}'"
    )


def _contextual_query_suffixes(goal: str, parents: list[Variant], limit: int) -> list[str]:
    context = " ".join([goal, *[parent.payload for parent in parents]])
    terms = _context_terms(context, limit=max(8, limit * 3))
    suffixes: list[str] = []
    for index in range(limit):
        start = index * 2
        chunk = terms[start : start + 3]
        if chunk:
            suffixes.append(" ".join(chunk))
    if len(suffixes) < limit:
        digest = hashlib.sha256(context.encode("utf-8")).hexdigest()
        for index in range(len(suffixes), limit):
            start = (int(digest[index * 2 : index * 2 + 2] or "0", 16) % max(len(terms), 1)) if terms else 0
            chunk = terms[start : start + 2] if terms else []
            suffixes.append(" ".join(chunk) if chunk else f"context-{digest[index * 4:index * 4 + 8]}")
    return suffixes[:limit]


CONTEXT_STOPWORDS = {
    "and",
    "are",
    "but",
    "candidate",
    "challenge",
    "code",
    "for",
    "from",
    "goal",
    "into",
    "none",
    "optimization",
    "optimize",
    "query",
    "research",
    "round",
    "score",
    "strategy",
    "the",
    "this",
    "variant",
    "with",
}


def _context_terms(text: str, limit: int = 12) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_]{2,}", text.lower().replace("-", " ")):
        if token in CONTEXT_STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= limit:
            break
    return terms


def _plateau_entropy_intent(action: str, goal: str, score_context: str, evaluator_name: str) -> dict[str, object]:
    context_terms = _context_terms(f"{goal} {score_context}", limit=10)
    topic = " ".join(context_terms[:5]) or "the objective"
    evaluator = evaluator_name or "deterministic evaluator"
    plans = {
        "new_strategy_family": {
            "exploration_path": f"try a new strategy family for {topic}",
            "evidence_basis": "evolutionary search literature favors structural mutation when local refinements stall",
            "expected_generalization": "a structurally different family tests assumptions outside the current local optimum instead of tuning the same knobs",
        },
        "literature_mechanism": {
            "exploration_path": f"derive a mechanism from freshly grounded literature for {topic}",
            "evidence_basis": "literature-backed mechanisms add causal or empirical constraints that are independent of the current score trace",
            "expected_generalization": "mechanism-guided variants should survive distribution shifts better than score-only parameter changes",
        },
        "alternative_evaluator": {
            "exploration_path": f"judge candidates with an auxiliary robustness lens alongside {evaluator}",
            "evidence_basis": "multi-objective and out-of-sample validation reduce overfitting to a single proxy score",
            "expected_generalization": "an alternative evaluator lens penalizes brittle gains that only exploit the current proxy",
        },
        "fresh_search_context": {
            "exploration_path": f"refresh search context for {topic}",
            "evidence_basis": "retrieval-augmented optimization benefits from new evidence when existing candidates stop improving",
            "expected_generalization": "fresh context can reveal untried constraints, failure modes, or mechanisms not represented in parent variants",
        },
    }
    selected = plans.get(action, plans["new_strategy_family"])
    return {
        "action": action,
        "exploration_path": selected["exploration_path"],
        "expected_generalization": selected["expected_generalization"],
        "evidence_basis": selected["evidence_basis"],
        "anti_reward_hack_rule": "Do not count temperature changes, random seeds, or scalar hyperparameter nudges unless tied to a new family, mechanism, evaluator, or search context.",
    }


def _entropy_payload_suffix(intent: dict[str, object]) -> str:
    return (
        f"meaningful_entropy_action={intent.get('action', '')} "
        f"exploration_path='{intent.get('exploration_path', '')}' "
        f"expected_generalization='{intent.get('expected_generalization', '')}'"
    )


def _prediction_market_code_signature(payload: str) -> str:
    code = payload if "class Strategy" in payload and "BaseStrategy" in payload else _prediction_market_solution(payload)
    normalized = re.sub(r"SOURCE_VARIANT = \"\"\".*?\"\"\"", "", code, flags=re.S)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _randomly_mutate_variant(variant: Variant, seed: int) -> Variant:
    """Return a copy of variant with context-derived numeric perturbations."""
    digest = hashlib.sha256(f"{variant.payload}|{seed}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))

    def _perturb(match: re.Match) -> str:
        original = float(match.group(1))
        if original == 0.0:
            return match.group(1)
        factor = rng.uniform(0.6, 1.4)
        perturbed = original * factor
        # Preserve int vs float representation.
        if "." in match.group(1):
            return str(round(perturbed, 2))
        return str(int(round(perturbed)))

    mutated_payload = re.sub(r"(-?\d+(?:\.\d+)?)", _perturb, variant.payload)
    return Variant(
        run_id=variant.run_id,
        outer_iteration=variant.outer_iteration,
        kind=variant.kind,
        payload=mutated_payload,
        parent_ids=variant.parent_ids,
        metadata={**variant.metadata, "recovery": "context_derived_numeric_mutation", "mutation_seed": digest[:16]},
    )


def _support_level(confidence: float) -> str:
    if confidence >= 0.75:
        return "strong"
    if confidence >= 0.55:
        return "moderate"
    return "weak"


def _retriever_fallbacks(retriever_name: str) -> list[str]:
    scholarly = {"arxiv", "openalex", "semantic_scholar"}
    if retriever_name in scholarly:
        return [name for name in ["semantic_scholar", "openalex", "arxiv", "wikipedia", "local"] if name != retriever_name]
    if retriever_name in {"docs_blogs", "web", "wikipedia"}:
        return ["openalex", "semantic_scholar", "arxiv", "local"]
    return ["local"]


async def _search_backend_with_retry(backend: SearchBackend, query: str, limit: int) -> list[tuple[object, float]]:
    attempts = 2 if _is_live_retriever(backend.tool_name) else 1
    for attempt in range(attempts):
        try:
            return await asyncio.to_thread(backend.search, query, limit)
        except Exception as exc:
            if attempt + 1 >= attempts or not _is_rate_limit_error(exc):
                raise
            await asyncio.sleep(0.75 * (attempt + 1))
    return []


def _is_live_retriever(tool_name: str) -> bool:
    return any(term in tool_name for term in ["api", "web", "wikipedia", "github"])


def _is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError) and exc.code == 429:
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return "429" in text or "too many requests" in text or "rate limit" in text


def _stable_judge_score(payload: str, metrics: dict[str, float]) -> float:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    jitter = int(digest[:4], 16) / 0xFFFF
    weighted = (metrics["coverage"] * 0.35) + (metrics["corroboration"] * 0.35) + (metrics["credibility"] * 0.25)
    return round(min(1.0, weighted + (jitter * 0.05)), 3)


def _looks_like_optimization_query(goal: str) -> bool:
    normalized = goal.lower()
    query_terms = {"research", "find", "query", "search", "investigate", "explore", "look for"}
    return any(term in normalized for term in query_terms) and any(term in normalized for term in OPTIMIZE_HINTS)


def _product_agent_for(selected_mode: TaskMode, goal: str, evaluator_name: Optional[str]) -> ProductAgent:
    normalized = goal.lower()
    if evaluator_name == "prediction_market" or "challenge" in normalized:
        return "challenge"
    if selected_mode == "research":
        return "research"
    return "optimize"


def _loop_objective_from_goal(goal: str, evaluator_name: Optional[str]) -> LoopObjective:
    normalized = goal.lower()
    no_stop = bool(re.search(r"\bdo\s*not\s*stop\b|\bdon't\s*stop\b|\bdont\s*stop\b|until\s+you", normalized))
    if evaluator_name == "prediction_market":
        target = _profit_target_from_goal(goal)
        return LoopObjective(kind="profit_usd", target=target, no_stop_until_target=no_stop and target is not None)
    return LoopObjective(kind="score", target=None, no_stop_until_target=no_stop)


def _profit_target_from_goal(goal: str) -> Optional[float]:
    normalized = goal.lower()
    patterns = [
        r"(?:get\s+to|reach|hit|achieve|make|earn)\s*\$?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:\$|usd|dollars?)?\s*(?:profit|edge)?",
        r"\$+\s*([0-9]+(?:\.[0-9]+)?)\s*(?:profit|edge|usd|dollars?)",
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:\$|usd|dollars?)\s*(?:profit|edge)",
    ]
    if not any(term in normalized for term in ["profit", "profitable", "edge", "$", "usd", "dollar"]):
        return None
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return float(match.group(1))
    return None


def _implementability_score(text: str) -> float:
    terms = {"implement", "code", "algorithm", "strategy", "heuristic", "benchmark", "test", "optimize", "latency", "throughput"}
    tokens = set(text.lower().replace("-", " ").split())
    return round(min(1.0, 0.35 + (len(tokens & terms) * 0.12)), 3)


def _novelty_score(text: str) -> float:
    normalized = text.lower().replace("-", " ")
    tokens = [token for token in normalized.split() if len(token) > 3]
    if not tokens:
        return 0.4
    distinct_ratio = len(set(tokens)) / len(tokens)
    novelty_terms = {"novel", "alternative", "contradictory", "recent", "mechanism", "frontier", "unusual", "ablation"}
    term_bonus = min(0.25, len(set(tokens) & novelty_terms) * 0.08)
    return round(min(1.0, 0.35 + (distinct_ratio * 0.35) + term_bonus), 3)


def _evaluator_relevance_score(text: str, evaluator_name: str) -> float:
    if not evaluator_name:
        return 0.45
    evaluator_terms = set(evaluator_name.lower().replace("_", " ").split())
    tokens = set(text.lower().replace("-", " ").split())
    overlap = len(tokens & evaluator_terms)
    return round(min(1.0, 0.55 + (overlap * 0.15)), 3)


def _prediction_market_solution(payload: str) -> str:
    escaped_payload = payload.replace('"""', '\\"\\"\\"')
    params = _prediction_market_params(payload)
    return f'''from __future__ import annotations

from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    """Adaptive passive prediction-market market maker.

    Generated by research-harness from the best optimize-query variant.
    This file targets the public upstream challenge API:
    https://github.com/danrobinson/prediction-market-challenge

    Source variant:
    """  # noqa: D205

    quote_size = {params["size"]!r}
    base_spread_ticks = {int(params["spread"])}
    inventory_limit = {params["inventory"]!r}
    skew_divisor = {params["skew_divisor"]!r}
    quote_mode = {params["quote_mode"]!r}

    def __init__(self) -> None:
        self.estimated_mid_ticks = 50.0
        self.last_buy_fill = 0.0
        self.last_sell_fill = 0.0

    def on_step(self, state: StepState):
        competitor_bid = state.competitor_best_bid_ticks
        competitor_ask = state.competitor_best_ask_ticks
        if competitor_bid is None and competitor_ask is None:
            return [CancelAll()]
        if competitor_bid is None:
            competitor_bid = max(1, competitor_ask - 8)
        if competitor_ask is None:
            competitor_ask = min(99, competitor_bid + 8)
        competitor_mid = (competitor_bid + competitor_ask) / 2.0

        buy_flow = state.buy_filled_quantity
        sell_flow = state.sell_filled_quantity
        net_flow = buy_flow - sell_flow
        self.last_buy_fill = state.buy_filled_quantity
        self.last_sell_fill = state.sell_filled_quantity

        midpoint_jump = abs(competitor_mid - self.estimated_mid_ticks)
        self.estimated_mid_ticks = (self.estimated_mid_ticks * 0.9) + (competitor_mid * 0.1) + (net_flow * 0.04)

        inventory = state.yes_inventory - state.no_inventory
        inventory_skew = max(-12.0, min(12.0, inventory / self.skew_divisor))
        spread = self.base_spread_ticks + (4 if midpoint_jump >= 4 else 0)
        if self.quote_mode == "none":
            return [CancelAll()]
        if self.quote_mode == "extreme":
            bid_reference = min(competitor_bid, self.estimated_mid_ticks)
            ask_reference = max(competitor_ask, self.estimated_mid_ticks)
            bid_ticks = int(max(1, min(98, round(bid_reference - spread - inventory_skew))))
            ask_ticks = int(max(bid_ticks + 1, min(99, round(ask_reference + spread - inventory_skew))))
        else:
            bid_ticks = int(max(1, min(98, round(competitor_bid - spread - inventory_skew))))
            ask_ticks = int(max(bid_ticks + 1, min(99, round(competitor_ask + spread - inventory_skew))))

        actions = [CancelAll()]
        ask_cost = max(0.01, ask_ticks / 100.0)
        bid_cost = max(0.01, bid_ticks / 100.0)
        if self.quote_size <= 0:
            return actions
        size = max(0.01, min(self.quote_size, state.free_cash / ask_cost))

        if state.yes_inventory < self.inventory_limit and state.free_cash >= bid_cost * size:
            actions.append(PlaceOrder(side=Side.BUY, price_ticks=bid_ticks, quantity=size))
        if state.no_inventory < self.inventory_limit:
            actions.append(PlaceOrder(side=Side.SELL, price_ticks=ask_ticks, quantity=size))
        return actions


SOURCE_VARIANT = """{escaped_payload}"""
'''


def _generic_optimal_code(payload: str, evaluator_name: str) -> str:
    return f'''from __future__ import annotations

"""Best optimization candidate emitted by research-harness.

This module is written for every optimization or challenge run so downstream
evaluators can always find the agent-selected code artifact at optimal_code.py.
When a domain adapter can render executable code, it should replace this generic
representation with evaluator-ready code.
"""

EVALUATOR_NAME = {evaluator_name!r}
OPTIMAL_CANDIDATE = {payload!r}


def selected_candidate() -> str:
    """Return the exact candidate payload that achieved the best score."""
    return OPTIMAL_CANDIDATE
'''


def _prediction_market_params(payload: str) -> dict[str, object]:
    params = {match.group("name").lower(): float(match.group("value")) for match in re.finditer(r"(?P<name>spread|size|quantity|inventory|limit|skew)\s*[=:]\s*(?P<value>-?\d+(?:\.\d+)?)", payload, re.I)}
    text = payload.lower()
    spread = int(max(2, min(30, params.get("spread", 12.0))))
    size = max(0.0, min(5.0, params.get("quantity", params.get("size", 1.0))))
    inventory = max(0.0, min(150.0, params.get("inventory", params.get("limit", 30.0))))
    skew = max(1.0, min(30.0, params.get("skew", 8.0)))
    if "quote_mode=none" in text or "no_trade" in text:
        quote_mode = "none"
    elif "quote_mode=extreme" in text or "extreme" in text:
        quote_mode = "extreme"
    else:
        quote_mode = "contextual"
    return {
        "spread": spread,
        "size": size,
        "inventory": inventory,
        "skew_divisor": skew,
        "quote_mode": quote_mode,
    }


def _normalize_prediction_market_edge(edge: float) -> float:
    return round(max(0.0, min(1.0, (edge + 30.0) / 60.0)), 3)


def _pm_edge_from_eval(evaluation: Optional[VariantEvaluation]) -> float:
    if not evaluation:
        return 0.0
    return float(evaluation.metrics.get("mean_edge", 0.0))


def _run_prediction_market_official(strategy_path: Path) -> dict[str, object]:
    upstream_path = _find_pm_upstream_path()
    if upstream_path is None:
        if os.environ.get("PREDICTION_MARKET_ALLOW_LOCAL_FALLBACK") == "1":
            result = _run_prediction_market_sandbox(strategy_path)
            result["score_eligible"] = False
            result["error"] = (
                "Upstream repo not found. Used debug-only local fallback because "
                "PREDICTION_MARKET_ALLOW_LOCAL_FALLBACK=1."
            )
            return result
        return _prediction_market_unmeasured_result(
            "upstream_repo_missing",
            "Upstream repo not found. Install prediction-market-challenge at a known path or "
            "set PREDICTION_MARKET_CHALLENGE_PATH. No fallback score was used for optimization."
        )

    simulations = os.environ.get("PREDICTION_MARKET_SIMULATIONS", "200")
    steps = os.environ.get("PREDICTION_MARKET_STEPS", "600")
    seed_start = os.environ.get("PREDICTION_MARKET_SEED_START", "0")
    workers = os.environ.get("PREDICTION_MARKET_WORKERS", "4")
    if os.environ.get("PREDICTION_MARKET_ALLOW_UNSANDBOXED_UPSTREAM") == "1":
        completed = _run_prediction_market_upstream_on_host(
            upstream_path,
            strategy_path,
            simulations=simulations,
            steps=steps,
            seed_start=seed_start,
            workers=workers,
        )
        docker_sandbox = False
    else:
        runner = DockerSandboxRunner()
        completed = runner.execute_prediction_market(
            upstream_path=upstream_path,
            strategy_path=strategy_path,
            simulations=simulations,
            steps=steps,
            seed_start=seed_start,
            workers=workers,
        )
        docker_sandbox = True

    if completed.returncode != 0:
        return _prediction_market_unmeasured_result(
            "official_sandbox_failed",
            (completed.stderr or completed.stdout or "official scorer failed").strip()[:2000],
            docker_sandbox=docker_sandbox,
            exit_code=completed.returncode,
            timed_out=getattr(completed, "timed_out", False),
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return _prediction_market_unmeasured_result(
            "official_scorer_json_error",
            f"JSONDecodeError: {exc}; stdout={completed.stdout[:500]}",
            docker_sandbox=docker_sandbox,
        )
    results = payload.get("simulation_results", [])
    successes = [result for result in results if not result.get("failed")]
    if not results or not successes:
        return _prediction_market_unmeasured_result(
            "official_scorer_no_successes",
            f"official scorer returned {len(results)} result(s), {len(successes)} successful.",
            docker_sandbox=docker_sandbox,
        )
    mean_edge = sum(float(result.get("total_edge", 0.0)) for result in successes) / max(len(successes), 1)
    mean_arb_edge = sum(float(result.get("arb_edge", 0.0)) for result in successes) / max(len(successes), 1)
    mean_retail_edge = sum(float(result.get("retail_edge", 0.0)) for result in successes) / max(len(successes), 1)
    return {
        "official_measured": True,
        "score_eligible": True,
        "sandbox_executed": True,
        "docker_sandbox": docker_sandbox,
        "mean_edge": round(mean_edge, 6),
        "mean_arb_edge": round(mean_arb_edge, 6),
        "mean_retail_edge": round(mean_retail_edge, 6),
        "success_count": len(successes),
        "failure_count": len(results) - len(successes),
        "simulations": len(results),
        "score_source": "upstream_orderbook_pm_challenge",
    }


def _run_prediction_market_upstream_on_host(
    upstream_path: Path,
    strategy_path: Path,
    *,
    simulations: str,
    steps: str,
    seed_start: str,
    workers: str,
):
    cmd = [
        "uv",
        "run",
        "--project",
        str(upstream_path),
        "orderbook-pm",
        "run",
        str(strategy_path),
        "--simulations",
        simulations,
        "--steps",
        steps,
        "--seed-start",
        seed_start,
        "--workers",
        workers,
        "--json",
    ]
    try:
        env = dict(os.environ)
        env.setdefault("UV_CACHE_DIR", "/private/tmp/research-harness-uv-cache")
        env.setdefault("UV_PYTHON_INSTALL_DIR", "/private/tmp/research-harness-uv-python")
        env.setdefault("UV_TOOL_DIR", "/private/tmp/research-harness-uv-tools")
        return subprocess.run(
            cmd,
            check=False,
            text=True,
            capture_output=True,
            env=env,
            timeout=float(os.environ.get("PREDICTION_MARKET_TIMEOUT_SECONDS", "300")),
        )
    except subprocess.TimeoutExpired as exc:
        return type("Completed", (), {"returncode": 124, "stdout": exc.stdout or "", "stderr": exc.stderr or "host upstream execution timed out", "timed_out": True})()
    except Exception as exc:
        return type("Completed", (), {"returncode": 1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}", "timed_out": False})()


def _prediction_market_unmeasured_result(
    score_source: str,
    error: str,
    *,
    docker_sandbox: bool = False,
    exit_code: Optional[int] = None,
    timed_out: bool = False,
) -> dict[str, object]:
    return {
        "official_measured": False,
        "score_eligible": False,
        "sandbox_executed": False,
        "docker_sandbox": docker_sandbox,
        "mean_edge": 0.0,
        "mean_arb_edge": 0.0,
        "mean_retail_edge": 0.0,
        "success_count": 0,
        "failure_count": 0,
        "simulations": 0,
        "score_source": score_source,
        "error": error,
        "exit_code": exit_code,
        "timed_out": timed_out,
    }


def _run_prediction_market_sandbox(strategy_path: Path) -> dict[str, object]:
    sandbox_root = strategy_path.parent / "sandbox"
    sandbox_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pm_", dir=sandbox_root) as directory:
        runner_path = Path(directory) / "sandbox_runner.py"
        runner_path.write_text(_prediction_market_sandbox_runner(), encoding="utf-8")
        completed = subprocess.run(
            [sys.executable, str(runner_path), str(strategy_path)],
            check=False,
            text=True,
            capture_output=True,
            cwd=directory,
            timeout=float(os.environ.get("PREDICTION_MARKET_SANDBOX_TIMEOUT_SECONDS", "30")),
        )
    if completed.returncode != 0:
        result = _prediction_market_local_semantic_score(strategy_path.read_text(encoding="utf-8"))
        result["sandbox_executed"] = False
        result["score_eligible"] = False
        result["score_source"] = "local_semantic_fallback_after_sandbox_failure"
        result["sandbox_error"] = (completed.stderr or completed.stdout).strip()[:1000]
        return result
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        result = _prediction_market_local_semantic_score(strategy_path.read_text(encoding="utf-8"))
        result["sandbox_executed"] = False
        result["score_eligible"] = False
        result["score_source"] = "local_semantic_fallback_after_sandbox_json_error"
        result["sandbox_error"] = f"JSONDecodeError: {exc}; stdout={completed.stdout[:500]}"
        return result
    payload["official_measured"] = False
    payload["sandbox_executed"] = True
    payload["score_eligible"] = False
    payload["score_source"] = "local_sandbox_strategy_execution"
    return payload


def _prediction_market_sandbox_runner() -> str:
    return r'''
from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import sys
import types
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class BaseStrategy:
    pass


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class CancelAll:
    pass


@dataclass
class PlaceOrder:
    side: Side
    price_ticks: int
    quantity: float


@dataclass
class StepState:
    competitor_best_bid_ticks: int
    competitor_best_ask_ticks: int
    buy_filled_quantity: float
    sell_filled_quantity: float
    yes_inventory: float
    no_inventory: float
    free_cash: float


root = types.ModuleType("orderbook_pm_challenge")
strategy_mod = types.ModuleType("orderbook_pm_challenge.strategy")
types_mod = types.ModuleType("orderbook_pm_challenge.types")
strategy_mod.BaseStrategy = BaseStrategy
types_mod.CancelAll = CancelAll
types_mod.PlaceOrder = PlaceOrder
types_mod.Side = Side
types_mod.StepState = StepState
sys.modules["orderbook_pm_challenge"] = root
sys.modules["orderbook_pm_challenge.strategy"] = strategy_mod
sys.modules["orderbook_pm_challenge.types"] = types_mod


def main(path: str) -> None:
    spec = importlib.util.spec_from_file_location("candidate_strategy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load strategy module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    strategy_cls = getattr(module, "Strategy")
    rng = random.Random(20260509)
    simulations = int(os.environ.get("PREDICTION_MARKET_SIMULATIONS", "200"))
    edge = 0.0
    retail_edge = 0.0
    arb_edge = 0.0
    failures = 0
    actions_seen = 0
    for sim in range(simulations):
        strategy = strategy_cls()
        yes_inventory = 0.0
        no_inventory = 0.0
        free_cash = 1000.0
        true_prob = max(0.02, min(0.98, 0.5 + rng.uniform(-0.2, 0.2)))
        buy_filled = 0.0
        sell_filled = 0.0
        for step in range(300):
            true_prob = max(0.01, min(0.99, true_prob + rng.gauss(0.0, 0.018)))
            competitor_mid = int(max(2, min(98, round(true_prob * 100 + rng.gauss(0, 1.2)))))
            competitor_bid = max(1, competitor_mid - rng.choice([1, 2, 3]))
            competitor_ask = min(99, competitor_mid + rng.choice([1, 2, 3]))
            state = StepState(
                competitor_best_bid_ticks=competitor_bid,
                competitor_best_ask_ticks=competitor_ask,
                buy_filled_quantity=buy_filled,
                sell_filled_quantity=sell_filled,
                yes_inventory=yes_inventory,
                no_inventory=no_inventory,
                free_cash=free_cash,
            )
            try:
                actions = strategy.on_step(state)
            except Exception:
                failures += 1
                continue
            if actions is None:
                actions = []
            if not isinstance(actions, list):
                failures += 1
                continue
            for action in actions[:8]:
                if isinstance(action, CancelAll):
                    continue
                if not isinstance(action, PlaceOrder):
                    failures += 1
                    continue
                actions_seen += 1
                price_ticks = int(action.price_ticks)
                quantity = max(0.0, min(float(action.quantity), 10.0))
                if price_ticks < 1 or price_ticks > 99 or quantity <= 0 or not math.isfinite(quantity):
                    failures += 1
                    continue
                price = price_ticks / 100.0
                if action.side == Side.SELL:
                    retail_fill = rng.random() < max(0.01, min(0.25, 0.08 + (price - true_prob) * 0.8))
                    arb_fill = price < true_prob and rng.random() < 0.85
                    if retail_fill:
                        gain = quantity * (price - true_prob)
                        edge += gain
                        retail_edge += gain
                        no_inventory += quantity
                        sell_filled += quantity
                    if arb_fill:
                        loss = quantity * (price - true_prob)
                        edge += loss
                        arb_edge += loss
                elif action.side == Side.BUY:
                    retail_fill = rng.random() < max(0.01, min(0.25, 0.08 + (true_prob - price) * 0.8))
                    arb_fill = price > true_prob and rng.random() < 0.85
                    if retail_fill and free_cash >= price * quantity:
                        gain = quantity * (true_prob - price)
                        edge += gain
                        retail_edge += gain
                        yes_inventory += quantity
                        buy_filled += quantity
                        free_cash -= price * quantity
                    if arb_fill:
                        loss = quantity * (true_prob - price)
                        edge += loss
                        arb_edge += loss
    print(json.dumps({
        "mean_edge": round(edge / float(simulations), 6),
        "mean_arb_edge": round(arb_edge / float(simulations), 6),
        "mean_retail_edge": round(retail_edge / float(simulations), 6),
        "success_count": simulations,
        "failure_count": failures,
        "simulations": simulations,
        "actions_seen": actions_seen,
    }))


if __name__ == "__main__":
    main(sys.argv[1])
'''


def _prediction_market_local_semantic_score(strategy_text: str, simulations: int = 200, steps: int = 800) -> dict[str, object]:
    params = _params_from_strategy_text(strategy_text)
    rng = random.Random(20260507)
    edges = []
    retail_edges = []
    arb_edges = []
    failures = 0
    for sim in range(simulations):
        true_prob = max(0.02, min(0.98, 0.5 + rng.uniform(-0.22, 0.22)))
        competitor_mid = true_prob
        competitor_spread = rng.choice([1, 2, 3, 4])
        inventory = 0.0
        edge = 0.0
        retail_edge = 0.0
        arb_edge = 0.0
        for step in range(steps):
            if rng.random() < rng.uniform(0.0008, 0.003):
                true_prob += rng.gauss(0.0, rng.uniform(0.2, 0.6))
            true_prob += rng.gauss(0.0, 0.02)
            true_prob = max(0.01, min(0.99, true_prob))

            lower = max(1, min(99, int(true_prob * 100)))
            competitor_bid = max(1, lower - (competitor_spread - 1))
            competitor_ask = min(99, lower + 1 + (competitor_spread - 1))
            if params["quote_mode"] == "none" or params["size"] <= 0:
                continue
            skew = max(-12.0, min(12.0, inventory / params["skew_divisor"]))
            if params["quote_mode"] == "extreme":
                bid = int(max(1, min(98, round(min(competitor_bid, competitor_mid * 100) - params["spread"] - skew))))
                ask = int(max(bid + 1, min(99, round(max(competitor_ask, competitor_mid * 100) + params["spread"] - skew))))
            else:
                bid = int(max(1, min(98, round(competitor_bid - params["spread"] - skew))))
                ask = int(max(bid + 1, min(99, round(competitor_ask + params["spread"] - skew))))

            size = min(params["size"], max(0.01, params["inventory"] - abs(inventory)))
            if size <= 0:
                continue
            bid_price = bid / 100.0
            ask_price = ask / 100.0
            if ask_price < true_prob:
                fill_edge = size * (ask_price - true_prob)
                edge += fill_edge
                arb_edge += fill_edge
                inventory -= size
            if bid_price > true_prob:
                fill_edge = size * (true_prob - bid_price)
                edge += fill_edge
                arb_edge += fill_edge
                inventory += size

            arrivals = 1 if rng.random() < rng.uniform(0.154, 0.352) else 0
            for _ in range(arrivals):
                if rng.random() < 0.5:
                    # Retail buy crosses our ask only when we improve or equal
                    # the hidden competitor's visible ask.
                    if ask <= competitor_ask + 1:
                        q = min(size, rng.lognormvariate(1.0, 1.2))
                        fill_edge = q * (ask_price - true_prob)
                        edge += fill_edge
                        retail_edge += fill_edge
                        inventory -= q
                else:
                    if bid >= competitor_bid - 1:
                        q = min(size, rng.lognormvariate(1.0, 1.2) / max(true_prob, 0.05))
                        fill_edge = q * (true_prob - bid_price)
                        edge += fill_edge
                        retail_edge += fill_edge
                        inventory += q
        edges.append(edge)
        retail_edges.append(retail_edge)
        arb_edges.append(arb_edge)
    mean_edge = sum(edges) / len(edges)
    return {
        "official_measured": False,
        "mean_edge": round(mean_edge, 6),
        "mean_arb_edge": round(sum(arb_edges) / len(arb_edges), 6),
        "mean_retail_edge": round(sum(retail_edges) / len(retail_edges), 6),
        "success_count": simulations - failures,
        "failure_count": failures,
        "simulations": simulations,
        "score_source": "local_official_semantics_fallback",
    }


def _params_from_strategy_text(strategy_text: str) -> dict[str, object]:
    values = {}
    for name in ["quote_size", "base_spread_ticks", "inventory_limit", "skew_divisor"]:
        match = re.search(rf"{name}\s*=\s*([0-9.]+)", strategy_text)
        if match:
            values[name] = float(match.group(1))
    mode_match = re.search(r"quote_mode\s*=\s*['\"]([^'\"]+)['\"]", strategy_text)
    return {
        "size": float(values.get("quote_size", 1.0)),
        "spread": int(values.get("base_spread_ticks", 12)),
        "inventory": float(values.get("inventory_limit", 30.0)),
        "skew_divisor": max(1.0, float(values.get("skew_divisor", 8.0))),
        "quote_mode": mode_match.group(1) if mode_match else "contextual",
    }


def _objective_metadata(evaluator_name: str) -> dict[str, object]:
    if evaluator_name == "prediction_market":
        return {
            "objective_name": "prediction_market_mean_edge",
            "objective_direction": "maximize",
            "official_result": {
                "measured": True,
                "profit_usd": None,
                "score_source": "upstream_orderbook_pm_challenge_when_available",
                "required_evaluator": "https://github.com/danrobinson/prediction-market-challenge",
                "reason": "Optimization evaluates generated candidates with the upstream orderbook-pm runner when available.",
            },
            "note": (
                "Prediction-market optimization uses upstream mean edge when the orderbook_pm_challenge repo is available. "
                "The normalized score is derived from mean edge for harness aggregation."
            ),
        }
    if evaluator_name == "length_score":
        return {
            "objective_name": "length_score",
            "objective_direction": "maximize",
            "official_result": {"measured": True, "score_source": "local_deterministic_evaluator"},
            "note": "This evaluator maximizes 1 / token_count for smoke-test optimization.",
        }
    return {
        "objective_name": evaluator_name or "deterministic_score",
        "objective_direction": "maximize",
        "official_result": {"measured": True, "score_source": "local_deterministic_evaluator"},
        "note": "Optimization result from the registered deterministic evaluator.",
    }


def _shorten(text: str, limit: int = 140) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
