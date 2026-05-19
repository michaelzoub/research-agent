from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

from .agents import (
    AgentResult,
    CriticAgent,
    HarnessDebuggerAgent,
    HypothesisAgent,
    LiteratureAgent,
    SynthesisAgent,
)
from .llm import LLMClient
from .loops import EvaluatorRegistry, EvolutionaryOuterLoop, TaskRouter, _loop_objective_from_goal
from .run_benchmarks import write_run_benchmarks
from .schemas import AgentBudget, AgentTrace, CostEvent, LoopIteration, LoopTask, ResearchPlan, RunRecord, SourceStrategyItem, TaskType, now_iso, to_dict
from .search import AlchemySearch, ArxivSearch, LocalCorpusSearch, SearchBackend
from .search import DocsBlogsSearch
from .search import GitHubSearch
from .search import OpenAlexSearch
from .search import PriorArtifactMemorySearch
from .search import SemanticScholarSearch
from .search import SocialWebSearch
from .search import WebSearch
from .search import WikipediaSearch
from .sessions import SessionStore, default_session_projects_dir
from .steering import start_cli_steering
from .store import ArtifactStore


STOPPING_SIGNALS = [
    "no meaningful new sources after N cycles",
    "no new high-confidence claims",
    "hypotheses converge",
    "critic finds no new major objections",
    "benchmark/evaluation score plateaus",
    "cost budget reached",
    "time budget reached",
    "human approval needed",
]


BOUNDED_MARKERS = {
    "optimize",
    "debug",
    "fix",
    "implement",
    "compare",
    "benchmark",
    "choose",
    "rank",
    "who founded",
}


RUN_SLUG_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "based",
    "be",
    "can",
    "for",
    "how",
    "in",
    "of",
    "on",
    "ones",
    "please",
    "research",
    "the",
    "to",
    "used",
    "what",
    "which",
    "will",
}


@dataclass
class HarnessConfig:
    id: str = "phase2-research-harness-v1"
    mode: str = "evolutionary"
    retriever: str = "auto"
    search_agent_count: int = 7
    hypothesis_agent_count: int = 2
    include_debugger: bool = True
    max_loop_iterations: int = 12
    max_loop_iterations_explicit: bool = False
    max_task_attempts: int = 2
    task_mode: str = "auto"
    evaluator_name: Optional[str] = None
    evolution_population_size: int = 4
    query_population_size: Optional[int] = None
    optimization_preset: str = "standard"
    optimizer_parent_count: int = 2
    parallel_evaluator_cap: int = 8
    optimize_plateau_patience: int = 2
    continue_on_optimize_plateau: bool = False
    force_direction_entropy: bool = True
    novelty_fraction: float = 0.25
    llm_provider: str = "auto"
    llm_model: str = "gpt-5.2"
    research_lead_model: Optional[str] = None
    research_subagent_model: Optional[str] = None
    session_projects_dir: Optional[Path] = None
    resume_session_id: Optional[str] = None
    fork_session_id: Optional[str] = None
    enable_sessions: bool = True
    echo_progress: bool = True
    enable_steering: bool = False
    default_budget: AgentBudget = field(default_factory=AgentBudget)


class Orchestrator:
    def __init__(self, corpus_path: Path, output_root: Path, config: Optional[HarnessConfig] = None):
        self.corpus_path = corpus_path
        self.output_root = output_root
        self.config = config or HarnessConfig()
        if self.config.optimization_preset == "challenge":
            if not self.config.max_loop_iterations_explicit:
                self.config.max_loop_iterations = max(self.config.max_loop_iterations, 20)
            self.config.evolution_population_size = max(self.config.evolution_population_size, 48)
            if self.config.query_population_size is None:
                self.config.query_population_size = 16
            self.config.optimizer_parent_count = max(self.config.optimizer_parent_count, 4)
            self.config.parallel_evaluator_cap = 16 if self.config.parallel_evaluator_cap == 8 else max(1, self.config.parallel_evaluator_cap)
            self.config.optimize_plateau_patience = max(self.config.optimize_plateau_patience, 5)
            self.config.continue_on_optimize_plateau = True
        self.evaluator_registry = EvaluatorRegistry()
        self.llm = LLMClient(provider=self.config.llm_provider, model=self.config.llm_model)
        if self.config.llm_provider == "openai" and not self.llm.is_live:
            raise ValueError("--llm-provider openai requires OPENAI_API_KEY.")
        if self.config.llm_provider == "anthropic" and not self.llm.is_live:
            raise ValueError("--llm-provider anthropic requires ANTHROPIC_API_KEY.")

    async def run(self, goal: str, mode: Optional[str] = None) -> Tuple[RunRecord, ArtifactStore]:
        selected_mode = mode or self.config.mode
        prior_run_memory = self.load_prior_run_memory(goal)
        plan = self.create_plan(goal, prior_run_memory=prior_run_memory)
        source_strategy = self.create_source_strategy(goal, plan, prior_run_memory=prior_run_memory)
        primary_retriever = self._retriever_for(source_strategy[0].retriever)
        run = RunRecord(
            id=self._next_run_id(goal),
            user_goal=goal,
            task_type=plan.task_type,
            harness_config_id=f"{self.config.id}:{selected_mode}:{primary_retriever.tool_name}:source_strategy_v1",
            prompt_versions=_prompt_versions(),
            harness_config_snapshot=_config_snapshot(self.config, selected_mode, primary_retriever.tool_name),
        )
        session_store = self._start_session_store(goal, run)
        store = ArtifactStore(self.output_root / run.id, echo_progress=self.config.echo_progress, session_store=session_store)
        steering_handle = start_cli_steering(store) if self.config.enable_steering else None
        store.add_run(run)
        store.write_prior_run_memory(prior_run_memory)
        store.append_progress(f"Starting run {run.id}")
        store.append_progress(f"Execution mode: {selected_mode}")
        store.append_progress(f"Goal: {goal}")
        store.append_progress(
            f"Prior output memory: checked {prior_run_memory.get('checked_run_count', 0)} run(s); "
            f"{len(prior_run_memory.get('avoid_directions', []))} prior direction(s) to avoid."
        )
        self._write_prd(store, run, plan, source_strategy, selected_mode, stage="planned")
        store.append_progress(f"PRD: {store.prd_path}")
        try:
            if selected_mode == "deterministic":
                await self._run_phase1(run, store, plan, source_strategy)
            elif selected_mode in {"standard", "fanout"}:
                await self._run_phase2(run, store, plan, source_strategy)
            elif selected_mode in {"evolutionary", "loop"}:
                await self._run_loop(run, store, plan, source_strategy)
            else:
                raise ValueError(f"Unknown mode: {selected_mode}")
            run.status = "failed" if _has_incomplete_required_loop_tasks(store) else "completed"
        except (KeyboardInterrupt, asyncio.CancelledError):
            run.status = "cancelled"
            await self._synthesize_interrupted_run(run, store)
        except Exception:
            run.status = "failed"
            raise
        finally:
            final_started = time.perf_counter()
            final_started_at = now_iso()
            # Use real accumulated token counts from the LLM client.
            run.total_tokens = self.llm.total_prompt_tokens + self.llm.total_completion_tokens
            run.total_cost = round(self.llm.total_cost(), 6)
            self._write_prd(store, run, plan, source_strategy, selected_mode, stage="completed")
            # Write a per-run cost breakdown so it's easy to audit spend.
            cost_payload = self.llm.cost_breakdown()
            cost_payload["run_id"] = run.id
            cost_payload["completed_at"] = None
            self._record_cost_events(store, run)
            store.write_cost(cost_payload)
            store.add_trace(
                AgentTrace(
                    run_id=run.id,
                    agent_name="orchestration:finalize_run_outputs",
                    role="orchestration",
                    prompt="Write final PRD and cost artifacts before benchmark generation.",
                    model="deterministic-orchestrator",
                    tools_used=[],
                    tool_calls=[],
                    token_usage=0,
                    runtime_ms=max(0, int((time.perf_counter() - final_started) * 1000)),
                    status="completed",
                    errors=[],
                    output_summary="Final PRD and cost artifacts written.",
                    started_at=final_started_at,
                    failure_component="orchestration",
                )
            )
            run.completed_at = now_iso()
            store.update_run(run)
            self._write_prd(store, run, plan, source_strategy, selected_mode, stage="completed")
            cost_payload["completed_at"] = run.completed_at
            store.write_cost(cost_payload)
            write_run_benchmarks(store)
            store.append_progress(
                f"Cost: ${run.total_cost:.4f} ({run.total_tokens} tokens) — see {store.cost_path}"
            )
            if session_store is not None:
                session_store.complete_session(status=run.status, summary=f"Run {run.id} {run.status}.")
            if steering_handle is not None:
                steering_handle.stop()
        return run, store

    async def _synthesize_interrupted_run(self, run: RunRecord, store: ArtifactStore) -> None:
        """Turn a first Ctrl+C into "package current findings" instead of a hard stop."""
        store.append_progress("Interrupt received: stopping exploration and synthesizing partial run artifacts.")
        self._mark_running_loop_tasks_interrupted(store)
        task = self._synthesis_task_for_interrupt(store)
        iteration = len(store.list("loop_iterations")) + 1
        synthesis = SynthesisAgent(
            name="interrupt_synthesis",
            role="synthesis_agent",
            prompt_template=_prompt("synthesis_agent"),
            budget=self._budget("append_only"),
            llm=self.llm,
        )
        try:
            result = await synthesis.execute(run, store)
        except Exception as exc:
            summary = f"Interrupt synthesis failed: {type(exc).__name__}: {exc}"
            store.write_report(_interrupted_report(run, store, summary))
            result = AgentResult("interrupt_synthesis", summary, [summary])
        await self._record_task_result(run, store, task, iteration, result)
        if store.report_path.exists():
            report = store.report_path.read_text(encoding="utf-8", errors="replace")
            if "## Run Interrupted" not in report:
                store.write_report(
                    report.rstrip()
                    + "\n\n## Run Interrupted\n"
                    + "- This report was synthesized after an interrupt signal. Treat conclusions as partial and inspect `progress.txt`, `prd.json`, and optimization artifacts for the exact stopping point.\n"
                )
        store.append_progress("Interrupt synthesis complete: partial final_report.md and run artifacts were written.")

    def _mark_running_loop_tasks_interrupted(self, store: ArtifactStore) -> None:
        for row in store.list("loop_tasks"):
            if row.get("status") == "running":
                task = LoopTask(**row)
                task.status = "failed"
                task.passes = False
                task.last_error = "Interrupted by user before this task completed."
                task.result_summary = task.last_error
                task.completed_at = now_iso()
                store.update_loop_task(task)
        self._update_prd_tasks(store)

    def _synthesis_task_for_interrupt(self, store: ArtifactStore) -> LoopTask:
        tasks = [LoopTask(**row) for row in store.list("loop_tasks")]
        for task in tasks:
            if task.action == "synthesize":
                task.status = "running"
                store.update_loop_task(task)
                return task
        task = LoopTask(
            title="Synthesize interrupted run report",
            action="synthesize",
            priority=(max([task.priority for task in tasks], default=0) + 1),
            params={"interrupted": True},
            acceptance_criteria=["Partial final_report.md was written from available artifacts"],
            status="running",
        )
        return store.add_loop_task(task)

    def _start_session_store(self, goal: str, run: RunRecord) -> Optional[SessionStore]:
        if not self.config.enable_sessions:
            return None
        projects_dir = self.config.session_projects_dir or default_session_projects_dir()
        try:
            session_store = SessionStore(Path.cwd(), projects_dir)
            record = session_store.start_session(
                goal=goal,
                run_id=run.id,
                output_dir=self.output_root / run.id,
                resume_from=self.config.resume_session_id,
                fork_from=self.config.fork_session_id,
            )
        except OSError:
            return None
        run.session_id = record.id
        run.session_jsonl_path = str(record.jsonl_path)
        run.session_metadata_path = str(record.metadata_path)
        return session_store

    def _record_cost_events(self, store: ArtifactStore, run: RunRecord) -> None:
        if store.list("cost_events"):
            return
        for index, call in enumerate(self.llm.call_history, start=1):
            store.add_cost_event(
                CostEvent(
                    run_id=run.id,
                    component=f"model_call_{index}",
                    provider=str(call.get("provider") or self.llm.provider),
                    model=str(call.get("model") or self.llm.model),
                    prompt_tokens=int(call.get("prompt_tokens") or 0),
                    completion_tokens=int(call.get("completion_tokens") or 0),
                    cost_usd=float(call.get("cost_usd") or 0.0),
                    call_type="model",
                    metadata={
                        key: value
                        for key, value in call.items()
                        if key not in {"provider", "model", "prompt_tokens", "completion_tokens", "cost_usd"}
                    },
                )
            )

    def load_prior_run_memory(self, goal: str, limit: int = 6) -> dict[str, Any]:
        return _load_prior_run_memory(self.output_root, goal, limit=limit)

    def _write_prd(
        self,
        store: ArtifactStore,
        run: RunRecord,
        plan: ResearchPlan,
        source_strategy: list[SourceStrategyItem],
        selected_mode: str,
        stage: str,
    ) -> None:
        loop_tasks = store.list("loop_tasks")
        if loop_tasks:
            tasks = [_prd_task_from_loop_task(task, index) for index, task in enumerate(loop_tasks, start=1)]
        else:
            tasks = self._standard_prd_tasks(selected_mode, plan, source_strategy, store.list("agent_traces"))
        payload = {
            "schema_version": "research_harness_prd_v1",
            "stage": stage,
            "run": to_dict(run),
            "execution_mode": selected_mode,
            "task_mode": run.task_mode,
            "product_agent": run.product_agent,
            "goal": run.user_goal,
            "agent_harness": {
                "definition": "A product agent is model + harness: model client, loop policy, tools/evaluators, artifact store, budgets, traces, and stopping rules.",
                "runtime_mode": run.task_mode,
                "parallelism_policy": "The orchestrator may fan out role agents or variant evaluations with asyncio.gather when task dependencies allow it.",
            },
            "ralph_loop": {
                "definition": "PRD-driven long-running loop: persist tasks in prd.json, append progress.txt, run fresh loop iterations, and stop only when PRD items pass or the iteration budget is reached.",
                "state_files": ["prd.json", "progress.txt", "trace.jsonl"],
                "completion_rule": "All organized_tasks must have passes=true. Explicit objective targets, such as profit_usd, keep optimizer tasks incomplete until reached.",
                "max_iterations": self.config.max_loop_iterations,
            },
            "research_architecture": _research_architecture_payload(self.config, run.task_mode),
            "objective": _objective_payload(run.user_goal, self.config.evaluator_name, store),
            "plan": to_dict(plan),
            "source_strategy": [to_dict(item) for item in source_strategy],
            "organized_tasks": tasks,
            "artifacts": {
                "final_report": str(store.report_path),
                "final_report_tex": str(store.report_tex_path),
                "final_report_pdf": str(store.report_pdf_path),
                "final_report_preview": str(store.report_preview_path),
                "prior_run_memory": str(store.prior_run_memory_path),
                "optimizer_seed_context": str(store.optimizer_seed_context_path),
                "optimization_result": str(store.optimization_result_path),
                "optimized_candidate": str(store.optimized_candidate_path),
                "optimal_code": str(store.optimal_code_path),
                "solution": str(store.solution_path),
                "progress": str(store.progress_path),
                "run_benchmark": str(store.run_benchmark_path),
                "decision_dag": str(store.decision_dag_path),
                "agent_timeline": str(store.agent_timeline_path),
                "run_notebook": str(store.run_notebook_path),
                "harness_diagnosis": str(store.harness_diagnosis_path),
                "world_model_sqlite": str(store.sqlite_path),
                "loop_continuation_decisions": str(store.loop_continuation_path),
                "trace": str(store.trace_log_path),
                "session_jsonl": run.session_jsonl_path,
                "session_metadata": run.session_metadata_path,
            },
            "observability": {
                "prompt_versions": run.prompt_versions,
                "harness_config_snapshot": run.harness_config_snapshot,
                "cost_path": str(store.cost_path),
                "failure_taxonomy_path": str(store.harness_diagnosis_path),
            },
            "notes": [
                "This file is the run-local PRD/task map.",
                "Use organized_tasks for the ordered work plan, status, dependencies, and acceptance criteria.",
            ],
        }
        store.write_prd(payload)

    def _standard_prd_tasks(
        self,
        selected_mode: str,
        plan: ResearchPlan,
        source_strategy: list[SourceStrategyItem],
        traces: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        tasks: list[dict[str, object]] = []
        if selected_mode == "deterministic":
            first = source_strategy[0]
            tasks.append(
                _prd_task(
                    1,
                    "Search initial evidence",
                    "search",
                    {"retriever": first.retriever, "purpose": first.purpose, "queries": first.queries},
                    ["Retrieve sources", "Extract traceable claims"],
                    [],
                    traces,
                    "search_literature",
                )
            )
            tasks.append(
                _prd_task(2, "Critique evidence", "critique", {}, ["Review claims", "Record contradictions or questions"], ["US-001"], traces, "critic_reviewer")
            )
            tasks.append(
                _prd_task(3, "Synthesize report", "synthesize", {}, ["Write final_report.md"], ["US-002"], traces, "synthesis_agent")
            )
            return tasks

        for index, item in enumerate(source_strategy[: self.config.search_agent_count], start=1):
            tasks.append(
                _prd_task(
                    index,
                    f"Search {item.purpose}",
                    "search",
                    {"retriever": item.retriever, "purpose": item.purpose, "queries": item.queries},
                    ["Retrieve sources", "Extract claims", "Record failed paths if evidence is unavailable"],
                    [],
                    traces,
                    "search_literature",
                )
            )
        offset = len(tasks)
        for index, angle in enumerate(plan.hypothesis_angles[: self.config.hypothesis_agent_count], start=offset + 1):
            tasks.append(
                _prd_task(
                    index,
                    f"Generate hypotheses for {angle}",
                    "hypothesize",
                    {"hypothesis_angle": angle},
                    ["Inspect claims", "Create hypotheses or open questions"],
                    [task["id"] for task in tasks if task["kind"] == "search"],
                    traces,
                    "hypothesis_generation",
                )
            )
        tasks.append(
            _prd_task(
                len(tasks) + 1,
                "Critique claims and hypotheses",
                "critique",
                {},
                ["Review claims", "Record contradictions and open questions"],
                [task["id"] for task in tasks],
                traces,
                "critic_reviewer",
            )
        )
        tasks.append(
            _prd_task(
                len(tasks) + 1,
                "Synthesize final report",
                "synthesize",
                {},
                ["Write final_report.md", "Summarize sources, claims, hypotheses, and caveats"],
                [tasks[-1]["id"]],
                traces,
                "synthesis_agent",
            )
        )
        if self.config.include_debugger:
            tasks.append(
                _prd_task(
                    len(tasks) + 1,
                    "Inspect harness behavior",
                    "debug_harness",
                    {},
                    ["Record constrained harness-change proposal"],
                    [tasks[-1]["id"]],
                    traces,
                    "harness_debugger",
                )
            )
        return tasks

    def classify_task(self, goal: str) -> TaskType:
        normalized = goal.lower()
        if any(marker in normalized for marker in BOUNDED_MARKERS):
            return "bounded"
        return "open_ended"

    def create_plan(self, goal: str, prior_run_memory: Optional[dict[str, Any]] = None) -> ResearchPlan:
        interpretation = self.interpret_goal(goal)
        task_type = interpretation["task_type"] if interpretation["task_type"] in {"bounded", "open_ended"} else self.classify_task(goal)
        topics = set(str(topic) for topic in interpretation.get("topics", []) if topic)
        if "prediction_market" in topics and _is_prediction_market_challenge_goal(goal, self.config.evaluator_name):
            search_angles = [
                "challenge-specific evidence requested by the prompt",
                "candidate design constraints from retrieved sources",
                "evaluation failures and alternative approaches",
            ]
            hypothesis_angles = ["evidence-backed strategy direction", "risk mitigation", "alternative approach"]
            strategy = "Ground the challenge in prompt-derived and retrieved evidence before optimizing strategy code."
        elif task_type == "bounded":
            search_angles = _derive_search_angles(goal, interpretation, task_type, limit=3)
            hypothesis_angles = _derive_hypothesis_angles(goal, interpretation, task_type, limit=2)
            strategy = "Assign bounded roles and collect evidence against explicit success criteria."
        else:
            search_angles = _derive_search_angles(goal, interpretation, task_type, limit=4)
            hypothesis_angles = _derive_hypothesis_angles(goal, interpretation, task_type, limit=2)
            strategy = "Use a lead research agent to start wide across independent subagent searches, then narrow with critic-driven convergence."
        search_angles = _apply_prior_memory_to_angles(search_angles, prior_run_memory, limit=len(search_angles))
        hypothesis_angles = _apply_prior_memory_to_angles(hypothesis_angles, prior_run_memory, limit=len(hypothesis_angles))
        return ResearchPlan(
            task_type=task_type,
            goal=goal,
            strategy=strategy,
            search_angles=search_angles,
            hypothesis_angles=hypothesis_angles,
            stopping_signals=STOPPING_SIGNALS,
            topics=sorted(topics),
            topic_queries=[str(query) for query in interpretation.get("topic_queries", []) if query],
            planner=str(interpretation.get("planner", "deterministic-fallback")),
        )

    def interpret_goal(self, goal: str) -> dict[str, Any]:
        if self.llm.is_live:
            system = (
                "You interpret research-harness goals for planning and source selection. "
                "Infer the user's domain and intent semantically, including typos and abbreviations. "
                "Return JSON only with: task_type ('bounded' or 'open_ended'), topics (short snake_case strings), "
                "topic_queries (4-8 specific paper-search keyword queries or exact phrases), and rationale.\n\n"
                "For topic_queries: write the actual search strings a literature retriever should send to "
                "arXiv/OpenAlex/Semantic Scholar style tools. Prefer precise technical phrases, canonical method "
                "names, benchmark/dataset names, and survey phrases. Avoid generic words like 'find papers', "
                "'understand in depth', 'research', or broad product categories unless they are part of a known term."
            )
            user = json.dumps(
                {
                    "goal": goal,
                    "requested_task_mode": self.config.task_mode,
                    "selected_evaluator": self.config.evaluator_name,
                    "known_evaluators": ["length_score", "prediction_market"],
                    "note": (
                        "If selected_evaluator is prediction_market, treat the task as prediction-market "
                        "market-making even when the goal has typos such as 'predictionm arket' or shorthand like mm'ing."
                    ),
                },
                sort_keys=True,
            )
            try:
                payload = self.llm.complete_json(system, user, max_output_tokens=700, temperature=0.1)
                return _normalize_goal_interpretation(payload, fallback_task_type=self.classify_task(goal), planner="llm")
            except Exception:
                pass
        return _fallback_goal_interpretation(goal, self.config.evaluator_name, self.classify_task(goal))

    def create_source_strategy(
        self,
        goal: str,
        plan: ResearchPlan,
        prior_run_memory: Optional[dict[str, Any]] = None,
    ) -> list[SourceStrategyItem]:
        retriever = self.config.retriever.lower()
        strategy_goal = goal
        strategy_goal = _goal_with_prior_memory_context(strategy_goal, prior_run_memory)
        if "prediction_market" in set(plan.topics) and _is_prediction_market_challenge_goal(goal, self.config.evaluator_name):
            strategy_goal = _goal_with_prior_memory_context(goal, prior_run_memory)
        if retriever == "local":
            return [
                SourceStrategyItem(
                    name=f"local_{index + 1}",
                    retriever="local",
                    purpose=angle,
                    queries=[f"{strategy_goal} {angle}"],
                    limit=self.config.default_budget.max_tool_calls,
                )
                for index, angle in enumerate(plan.search_angles)
            ]
        if retriever == "auto":
            return _mixed_source_strategy(strategy_goal, plan)
        if retriever in {"arxiv", "openalex", "semantic_scholar", "github", "web", "docs_blogs", "twitter", "memory", "wikipedia", "alchemy"}:
            return _single_retriever_strategy(strategy_goal, plan, retriever)
        if retriever not in {"auto", "arxiv"}:
            raise ValueError(f"Unknown retriever: {self.config.retriever}")
        return _mixed_source_strategy(strategy_goal, plan)

    async def _run_phase1(
        self, run: RunRecord, store: ArtifactStore, plan: ResearchPlan, source_strategy: list[SourceStrategyItem]
    ) -> None:
        item = source_strategy[0]
        literature = LiteratureAgent(
            name="deterministic_literature_agent",
            query_angle=item.purpose,
            corpus=self._retriever_for(item.retriever),
            prompt_template=_prompt("literature_agent"),
            budget=self._budget("append_only"),
            search_queries=item.queries,
            llm=self.llm,
        )
        critic = CriticAgent(
            name="deterministic_critic",
            role="critic_reviewer",
            prompt_template=_prompt("critic_agent"),
            budget=self._budget("append_only"),
            llm=self.llm,
        )
        synth = SynthesisAgent(
            name="deterministic_synthesis",
            role="synthesis_agent",
            prompt_template=_prompt("synthesis_agent"),
            budget=self._budget("append_only"),
            llm=self.llm,
        )
        for agent in [literature, critic, synth]:
            await agent.execute(run, store)

    async def _run_phase2(
        self, run: RunRecord, store: ArtifactStore, plan: ResearchPlan, source_strategy: list[SourceStrategyItem]
    ) -> None:
        search_agents = [
            LiteratureAgent(
                name=f"literature_agent_{index + 1}",
                query_angle=item.purpose,
                corpus=self._retriever_for(item.retriever),
                prompt_template=_prompt("literature_agent"),
                budget=self._budget("upsert_by_url"),
                search_queries=item.queries,
                llm=self.llm,
            )
            for index, item in enumerate(source_strategy[: self.config.search_agent_count])
        ]
        await asyncio.gather(*(agent.execute(run, store) for agent in search_agents))

        hypothesis_agents = [
            HypothesisAgent(
                name=f"hypothesis_agent_{index + 1}",
                hypothesis_angle=angle,
                prompt_template=_prompt("hypothesis_agent"),
                budget=self._budget("append_only"),
                llm=self.llm,
            )
            for index, angle in enumerate(plan.hypothesis_angles[: self.config.hypothesis_agent_count])
        ]
        await asyncio.gather(*(agent.execute(run, store) for agent in hypothesis_agents))

        critic = CriticAgent(
            name="critic_reviewer",
            role="critic_reviewer",
            prompt_template=_prompt("critic_agent"),
            budget=self._budget("append_only"),
            llm=self.llm,
        )
        synthesis = SynthesisAgent(
            name="synthesis_agent",
            role="synthesis_agent",
            prompt_template=_prompt("synthesis_agent"),
            budget=self._budget("append_only"),
            llm=self.llm,
        )
        await critic.execute(run, store)
        await synthesis.execute(run, store)

        if self.config.include_debugger:
            debugger = HarnessDebuggerAgent(
                name="harness_debugger",
                role="harness_debugger",
                prompt_template=_prompt("harness_debugger"),
                budget=self._budget("append_only"),
                llm=self.llm,
            )
            await debugger.execute(run, store)

    async def _run_loop(
        self, run: RunRecord, store: ArtifactStore, plan: ResearchPlan, source_strategy: list[SourceStrategyItem]
    ) -> None:
        router = TaskRouter(self.evaluator_registry, llm=self.llm)
        decision = router.decide(run.user_goal, self.config.task_mode, self.config.evaluator_name)
        store.add_task_ingestion_decision(decision)
        run.task_mode = decision.selected_mode
        run.product_agent = decision.product_agent
        store.update_run(run)

        tasks = self.create_evolution_tasks(decision.selected_mode, decision.product_agent)
        for task in tasks:
            store.add_loop_task(task)
        store.append_progress(f"# Progress for {run.id}")
        store.append_progress(f"Prompt: {run.user_goal}")
        store.append_progress(f"Task ingestion: {decision.selected_mode} mode. {decision.reason}")

        await self._pass_task(
            run,
            store,
            tasks[0],
            1,
            "task_router",
            f"Selected {decision.selected_mode} mode. {decision.reason}",
        )

        tasks[1].status = "running"
        tasks[1].attempts += 1
        store.update_loop_task(tasks[1])
        population_size = self.config.evolution_population_size
        if decision.selected_mode == "research" and plan.task_type == "bounded":
            population_size = 1
        if self.config.optimization_preset == "challenge" and decision.selected_mode in {"optimize", "optimize_query"}:
            population_size = max(population_size, 48)
        outer_loop = EvolutionaryOuterLoop(
            run_id=run.id,
            goal=run.user_goal,
            task_mode=decision.selected_mode,
            source_strategy=source_strategy,
            search_factory=self._retriever_for,
            evaluator=self.evaluator_registry.get(decision.evaluator_name),
            evaluator_name=decision.evaluator_name,
            max_outer_iterations=self.config.max_loop_iterations,
            population_size=population_size,
            query_population_size=self.config.query_population_size,
            parent_count=self.config.optimizer_parent_count,
            parallel_evaluator_cap=self.config.parallel_evaluator_cap,
            optimize_plateau_patience=self.config.optimize_plateau_patience,
            continue_on_optimize_plateau=self.config.continue_on_optimize_plateau,
            force_direction_entropy=self.config.force_direction_entropy,
            novelty_fraction=self.config.novelty_fraction,
            llm=self.llm,
            prior_run_memory=_read_json_if_exists(store.prior_run_memory_path),
        )
        await outer_loop.run(store)
        await self._pass_task(
            run,
            store,
            tasks[1],
            2,
            "outer_orchestrator",
            f"Completed {len(store.list('evolution_rounds'))} evolutionary outer rounds.",
        )

        iteration = 3
        task_cursor = 2
        if decision.selected_mode == "optimize_query":
            await self._pass_task(
                run,
                store,
                tasks[task_cursor],
                iteration,
                "optimizer_seed_context",
                f"Compiled optimizer seed context at {store.optimizer_seed_context_path}.",
            )
            iteration += 1
            task_cursor += 1
            seed_context = _read_json_if_exists(store.optimizer_seed_context_path)
            if seed_context.get("has_evaluator"):
                optimize_eval_count = len([row for row in store.list("variant_evaluations") if row.get("inner_loop") == "optimize"])
                artifact_bits = []
                if store.optimization_result_path.exists():
                    artifact_bits.append("optimization_result.json")
                if store.optimal_code_path.exists():
                    artifact_bits.append("optimal_code.py")
                if store.solution_path.exists():
                    artifact_bits.append("solution.py")
                optimizer_summary = (
                    f"Ran {optimize_eval_count} optimizer variant evaluations using query-derived seed context"
                    + (f"; emitted {', '.join(artifact_bits)}." if artifact_bits else ".")
                )
                objective_status = _objective_status(run.user_goal, decision.evaluator_name, store)
                if objective_status.get("requires_target") and not objective_status.get("met"):
                    await self._fail_task(
                        run,
                        store,
                        tasks[task_cursor],
                        iteration,
                        "optimize_query_optimizer",
                        optimizer_summary
                        + f" Objective incomplete: {objective_status.get('summary')}. Continue the Ralph loop with a larger --max-iterations budget.",
                    )
                    store.append_progress("Optimization objective remains incomplete; stopping before downstream review tasks.")
                    return
                else:
                    await self._pass_task(run, store, tasks[task_cursor], iteration, "optimize_query_optimizer", optimizer_summary)
            else:
                await self._skip_task(
                    run,
                    store,
                    tasks[task_cursor],
                    iteration,
                    "Skipped optimizer phase because no evaluator was registered.",
                )
            iteration += 1
            task_cursor += 1
        if decision.selected_mode == "research":
            hypothesis = HypothesisAgent(
                name="loop_research_hypothesis",
                hypothesis_angle=plan.hypothesis_angles[0],
                prompt_template=_prompt("hypothesis_agent"),
                budget=self._budget("append_only"),
                llm=self.llm,
            )
            result = await hypothesis.execute(run, store)
            await self._record_task_result(run, store, tasks[task_cursor], iteration, result)
            iteration += 1
            task_cursor += 1

        critic = CriticAgent(
            name="loop_evolution_critic",
            role="critic_reviewer",
            prompt_template=_prompt("critic_agent"),
            budget=self._budget("append_only"),
            llm=self.llm,
        )
        result = await critic.execute(run, store)
        await self._record_task_result(run, store, tasks[task_cursor], iteration, result)
        iteration += 1
        task_cursor += 1

        synthesis = SynthesisAgent(
            name="loop_evolution_synthesis",
            role="synthesis_agent",
            prompt_template=_prompt("synthesis_agent"),
            budget=self._budget("append_only"),
            llm=self.llm,
        )
        result = await synthesis.execute(run, store)
        await self._record_task_result(run, store, tasks[task_cursor], iteration, result)
        iteration += 1
        task_cursor += 1

        if self.config.include_debugger:
            debugger = HarnessDebuggerAgent(
                name="loop_evolution_debugger",
                role="harness_debugger",
                prompt_template=_prompt("harness_debugger"),
                budget=self._budget("append_only"),
                llm=self.llm,
            )
            result = await debugger.execute(run, store)
            await self._record_task_result(run, store, tasks[task_cursor], iteration, result)

        remaining = [row for row in store.list("loop_tasks") if not row.get("passes")]
        if remaining:
            store.append_progress(f"Stopped with {len(remaining)} incomplete loop tasks.")
        else:
            store.append_progress("<promise>COMPLETE</promise>")

    def create_evolution_tasks(self, selected_mode: str, product_agent: str = "research") -> list[LoopTask]:
        if selected_mode == "optimize_query":
            return self._create_optimize_query_tasks(product_agent)
        tasks = [
            LoopTask(
                title="Ingest prompt and select task mode",
                action="debug_harness",
                priority=1,
                params={"selected_mode": selected_mode, "product_agent": product_agent},
                acceptance_criteria=[
                    "Explicit flags, evaluator availability, and prompt heuristics were checked",
                    "A task ingestion decision was persisted",
                    f"The `{product_agent}` product agent was selected",
                ],
            ),
            LoopTask(
                title=f"Run {product_agent} agent loop",
                action="debug_harness",
                priority=2,
                params={"selected_mode": selected_mode, "product_agent": product_agent},
                acceptance_criteria=[
                    "The agent harness ran model/tool/state loop steps for this product option",
                    "Research mode used breadth-first parallel query variants before narrowing",
                    "The outer loop proposed code or query variants",
                    "The selected inner loop returned ranked variants with scalar scores",
                    "Plateau or threshold stopping signals were evaluated",
                ],
            ),
        ]
        if selected_mode == "research":
            tasks.append(
                LoopTask(
                    title="Generate hypotheses from winning evidence",
                    action="hypothesize",
                    priority=3,
                    params={},
                    acceptance_criteria=["Claims from research variants were inspected", "Hypotheses or open questions were created"],
                )
            )
        tasks.extend(
            [
                LoopTask(
                    title="Critique ranked variants and claims",
                    action="critique",
                    priority=4,
                    params={},
                    acceptance_criteria=["Artifacts were reviewed for contradictions, weak evidence, or missing checks"],
                ),
                LoopTask(
                    title="Synthesize evolutionary run report",
                    action="synthesize",
                    priority=5,
                    params={},
                    acceptance_criteria=["Final report was written from run artifacts"],
                ),
            ]
        )
        if self.config.include_debugger:
            tasks.append(
                LoopTask(
                    title="Inspect harness behavior and propose improvements",
                    action="debug_harness",
                    priority=6,
                    params={},
                    acceptance_criteria=["A constrained harness-change proposal was recorded"],
                )
            )
        return tasks

    def _create_optimize_query_tasks(self, product_agent: str = "optimize") -> list[LoopTask]:
        optimizer_label = "challenge" if product_agent == "challenge" else "optimization"
        tasks = [
            LoopTask(
                title="Ingest prompt and select task mode",
                action="debug_harness",
                priority=1,
                params={"selected_mode": "optimize_query", "product_agent": product_agent},
                acceptance_criteria=[
                    "Explicit mode or auto-router selected optimization-query exploration",
                    "A task ingestion decision was persisted",
                    f"The `{product_agent}` product agent was selected",
                ],
            ),
            LoopTask(
                title=f"Generate and evaluate {optimizer_label} strategy queries",
                action="search",
                priority=2,
                params={"selected_mode": "optimize_query", "product_agent": product_agent},
                acceptance_criteria=[
                    "The agent harness ran a query-research loop before optimization",
                    "Outer loop proposed query variants about the optimization target",
                    "OptimizationQueryLoop scored query variants for evidence and implementability",
                    "Ranked query evaluations were persisted",
                ],
            ),
            LoopTask(
                title="Compile optimizer seed context",
                action="debug_harness",
                priority=3,
                params={},
                acceptance_criteria=[
                    "Top query findings were summarized",
                    "optimizer_seed_context.json was written",
                ],
            ),
            LoopTask(
                title="Run optimizer variants from query seed context",
                action="debug_harness",
                priority=4,
                params={"product_agent": product_agent},
                acceptance_criteria=[
                    "If evaluator exists, optimize/code variants were evaluated",
                    "If evaluator is missing, optimizer phase was explicitly skipped",
                    "Challenge agents may additionally emit a runnable solution artifact for official grading",
                    "Explicit objective targets from the prompt must be met before this PRD item passes",
                ],
            ),
            LoopTask(
                title="Critique ranked query and optimizer results",
                action="critique",
                priority=5,
                params={},
                acceptance_criteria=["Artifacts were reviewed for contradictions, weak evidence, or missing checks"],
            ),
            LoopTask(
                title="Synthesize optimize-query run report",
                action="synthesize",
                priority=6,
                params={},
                acceptance_criteria=["Final report included query findings and optimizer seed context"],
            ),
        ]
        if self.config.include_debugger:
            tasks.append(
                LoopTask(
                    title="Inspect harness behavior and propose improvements",
                    action="debug_harness",
                    priority=7,
                    params={},
                    acceptance_criteria=["A constrained harness-change proposal was recorded"],
                )
            )
        return tasks

    async def _record_task_result(
        self, run: RunRecord, store: ArtifactStore, task: LoopTask, iteration: int, result: AgentResult
    ) -> None:
        if result.errors:
            task.status = "failed"
            task.passes = False
            task.last_error = "; ".join(result.errors)
        else:
            task.status = "passed"
            task.passes = True
            task.completed_at = now_iso()
        task.attempts += 1
        task.result_summary = result.summary
        store.update_loop_task(task)
        store.add_loop_iteration(
            LoopIteration(
                run_id=run.id,
                iteration=iteration,
                task_id=task.id,
                task_title=task.title,
                agent_name=result.agent_name,
                status=task.status,
                summary=result.summary,
                errors=result.errors,
                completed_at=now_iso(),
            )
        )
        store.append_progress(f"Task {iteration}: {task.status} - {result.summary}")
        self._update_prd_tasks(store)

    def _update_prd_tasks(self, store: ArtifactStore) -> None:
        """Refresh organized_tasks in prd.json after each story completes."""
        loop_tasks = store.list("loop_tasks")
        if not loop_tasks or not store.prd_path.exists():
            return
        tasks = [_prd_task_from_loop_task(task, index) for index, task in enumerate(loop_tasks, start=1)]
        try:
            existing = json.loads(store.prd_path.read_text(encoding="utf-8"))
            existing["organized_tasks"] = tasks
            store.write_prd(existing)
        except (json.JSONDecodeError, OSError):
            pass

    async def _pass_task(
        self,
        run: RunRecord,
        store: ArtifactStore,
        task: LoopTask,
        iteration: int,
        agent_name: str,
        summary: str,
    ) -> None:
        await self._record_task_result(run, store, task, iteration, AgentResult(agent_name, summary, []))

    async def _fail_task(
        self,
        run: RunRecord,
        store: ArtifactStore,
        task: LoopTask,
        iteration: int,
        agent_name: str,
        summary: str,
    ) -> None:
        await self._record_task_result(run, store, task, iteration, AgentResult(agent_name, summary, [summary]))

    async def _skip_task(
        self,
        run: RunRecord,
        store: ArtifactStore,
        task: LoopTask,
        iteration: int,
        summary: str,
    ) -> None:
        task.status = "skipped"
        task.passes = True
        task.attempts += 1
        task.result_summary = summary
        task.completed_at = now_iso()
        store.update_loop_task(task)
        store.add_loop_iteration(
            LoopIteration(
                run_id=run.id,
                iteration=iteration,
                task_id=task.id,
                task_title=task.title,
                agent_name="optimize_query_router",
                status=task.status,
                summary=summary,
                errors=[],
                completed_at=now_iso(),
            )
        )
        store.append_progress(f"Task {iteration}: skipped - {summary}")
        self._update_prd_tasks(store)

    def create_loop_tasks(self, plan: ResearchPlan, source_strategy: list[SourceStrategyItem]) -> list[LoopTask]:
        tasks: list[LoopTask] = []
        for index, item in enumerate(source_strategy[: self.config.search_agent_count], start=1):
            tasks.append(
                LoopTask(
                    title=f"Search {item.purpose}",
                    action="search",
                    priority=index,
                    params={
                        "query_angle": item.purpose,
                        "retriever": item.retriever,
                        "queries": item.queries,
                    },
                    acceptance_criteria=[
                        "Retriever was called for the assigned angle",
                        "Relevant sources and claims were written when evidence exists",
                        "Failure paths were recorded when no evidence exists",
                    ],
                )
            )
        base_priority = len(tasks) + 1
        for index, angle in enumerate(plan.hypothesis_angles[: self.config.hypothesis_agent_count], start=base_priority):
            tasks.append(
                LoopTask(
                    title=f"Generate hypotheses for {angle}",
                    action="hypothesize",
                    priority=index,
                    params={"hypothesis_angle": angle},
                    acceptance_criteria=[
                        "Existing claims were inspected",
                        "Hypotheses or open questions were created",
                        "Next experiments were captured for testable hypotheses",
                    ],
                )
            )
        tasks.extend(
            [
                LoopTask(
                    title="Critique claims and hypotheses",
                    action="critique",
                    priority=len(tasks) + 1,
                    params={},
                    acceptance_criteria=[
                        "Claims and hypotheses were reviewed",
                        "Contradictions or open questions were recorded when found",
                    ],
                ),
                LoopTask(
                    title="Synthesize final report",
                    action="synthesize",
                    priority=len(tasks) + 2,
                    params={},
                    acceptance_criteria=[
                        "Final report was written",
                        "Sources, claims, hypotheses, contradictions, and questions were summarized",
                    ],
                ),
            ]
        )
        if self.config.include_debugger:
            tasks.append(
                LoopTask(
                    title="Inspect harness behavior and propose improvements",
                    action="debug_harness",
                    priority=len(tasks) + 1,
                    params={},
                    acceptance_criteria=[
                        "Run traces and artifact yield were inspected",
                        "A constrained harness-change proposal was recorded",
                    ],
                )
            )
        return tasks

    def _next_loop_task(self, store: ArtifactStore) -> Optional[LoopTask]:
        candidates = []
        for row in store.list("loop_tasks"):
            if row.get("passes"):
                continue
            if row.get("status") == "failed" and int(row.get("attempts", 0)) >= self.config.max_task_attempts:
                continue
            candidates.append(LoopTask(**row))
        if not candidates:
            return None
        return sorted(candidates, key=lambda task: (task.priority, task.created_at))[0]

    async def _execute_loop_task(self, run: RunRecord, store: ArtifactStore, task: LoopTask):
        if task.action == "search":
            agent = LiteratureAgent(
                name=f"loop_search_{task.id}",
                query_angle=str(task.params["query_angle"]),
                corpus=self._retriever_for(str(task.params["retriever"])),
                prompt_template=_prompt("literature_agent"),
                budget=self._budget("upsert_by_url"),
                search_queries=[str(query) for query in task.params.get("queries", [])],
                llm=self.llm,
            )
        elif task.action == "hypothesize":
            agent = HypothesisAgent(
                name=f"loop_hypothesis_{task.id}",
                hypothesis_angle=str(task.params["hypothesis_angle"]),
                prompt_template=_prompt("hypothesis_agent"),
                budget=self._budget("append_only"),
                llm=self.llm,
            )
        elif task.action == "critique":
            agent = CriticAgent(
                name=f"loop_critic_{task.id}",
                role="critic_reviewer",
                prompt_template=_prompt("critic_agent"),
                budget=self._budget("append_only"),
                llm=self.llm,
            )
        elif task.action == "synthesize":
            agent = SynthesisAgent(
                name=f"loop_synthesis_{task.id}",
                role="synthesis_agent",
                prompt_template=_prompt("synthesis_agent"),
                budget=self._budget("append_only"),
                llm=self.llm,
            )
        elif task.action == "debug_harness":
            agent = HarnessDebuggerAgent(
                name=f"loop_debugger_{task.id}",
                role="harness_debugger",
                prompt_template=_prompt("harness_debugger"),
                budget=self._budget("append_only"),
                llm=self.llm,
            )
        else:  # pragma: no cover - guarded by LoopTaskAction literal
            raise ValueError(f"Unknown loop action: {task.action}")
        return await agent.execute(run, store)

    def _budget(self, write_policy: str) -> AgentBudget:
        return AgentBudget(
            max_steps=self.config.default_budget.max_steps,
            max_tokens=self.config.default_budget.max_tokens,
            max_tool_calls=self.config.default_budget.max_tool_calls,
            max_runtime_seconds=self.config.default_budget.max_runtime_seconds,
            write_policy=write_policy,  # type: ignore[arg-type]
            reporting_schema=self.config.default_budget.reporting_schema,
        )

    def _retriever_for(self, retriever: str) -> SearchBackend:
        retriever = retriever.lower()
        if retriever == "local":
            return LocalCorpusSearch(self.corpus_path)
        if retriever == "arxiv":
            return ArxivSearch(llm=self.llm)
        if retriever == "openalex":
            return OpenAlexSearch()
        if retriever == "semantic_scholar":
            return SemanticScholarSearch()
        if retriever == "github":
            return GitHubSearch()
        if retriever == "web":
            return WebSearch()
        if retriever == "docs_blogs":
            return DocsBlogsSearch()
        if retriever == "twitter":
            return SocialWebSearch()
        if retriever == "memory":
            return PriorArtifactMemorySearch(self.output_root)
        if retriever == "wikipedia":
            return WikipediaSearch()
        if retriever == "alchemy":
            return AlchemySearch()
        if retriever == "auto":
            return OpenAlexSearch()
        raise ValueError(f"Unknown retriever: {retriever}")

    def _next_run_id(self, goal: str) -> str:
        base = goal_slug(goal)
        run_number = self._next_run_number()
        candidate = f"{run_number:03d}_run_{base}"
        if not (self.output_root / candidate).exists():
            return candidate
        while True:
            run_number += 1
            numbered = f"{run_number:03d}_run_{base}"
            if not (self.output_root / numbered).exists():
                return numbered

    def _next_run_number(self) -> int:
        self.output_root.mkdir(parents=True, exist_ok=True)
        numbers = [_run_number_from_name(path.name) for path in self.output_root.iterdir() if path.is_dir()]
        explicit_numbers = [number for number in numbers if number is not None]
        if explicit_numbers:
            return max(explicit_numbers) + 1
        legacy_runs = [path for path in self.output_root.iterdir() if path.is_dir() and path.name.startswith("run_")]
        return len(legacy_runs) + 1


def _prompt(name: str) -> str:
    path = Path(__file__).resolve().parent.parent / "prompts" / f"{name}.md"
    return path.read_text(encoding="utf-8")


def _prompt_versions() -> dict[str, str]:
    prompt_dir = Path(__file__).resolve().parent.parent / "prompts"
    versions = {}
    for path in sorted(prompt_dir.glob("*.md")):
        versions[path.stem] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return versions


def _config_snapshot(config: HarnessConfig, selected_mode: str, primary_retriever: str) -> dict[str, Any]:
    payload = asdict(config)
    payload["selected_mode"] = selected_mode
    payload["primary_retriever"] = primary_retriever
    payload["session_projects_dir"] = str(config.session_projects_dir) if config.session_projects_dir else None
    payload["default_budget"] = to_dict(config.default_budget)
    return json.loads(json.dumps(payload, default=str))


def goal_slug(goal: str, max_length: int = 72) -> str:
    words = re.findall(r"[a-zA-Z0-9]+", goal.lower())
    selected = [word for word in words if word not in RUN_SLUG_STOPWORDS]
    if not selected:
        selected = words or ["research-run"]
    slug = "-".join(selected)
    slug = slug[:max_length].strip("-")
    return slug or "research-run"


def _run_number_from_name(name: str) -> Optional[int]:
    match = re.match(r"^(\d+)_run_", name)
    if not match:
        return None
    return int(match.group(1))


def _is_run_dir(name: str) -> bool:
    return name.startswith("run_") or bool(re.match(r"^\d+_run_", name))


def _objective_payload(goal: str, evaluator_name: Optional[str], store: ArtifactStore) -> dict[str, object]:
    status = _objective_status(goal, evaluator_name, store)
    return {
        "kind": status.get("kind"),
        "target": status.get("target"),
        "current": status.get("current"),
        "met": status.get("met"),
        "no_stop_until_target": status.get("no_stop_until_target"),
        "summary": status.get("summary"),
    }


def _objective_status(goal: str, evaluator_name: Optional[str], store: ArtifactStore) -> dict[str, object]:
    objective = _loop_objective_from_goal(goal, evaluator_name)
    current = None
    if store.optimization_result_path.exists():
        try:
            payload = json.loads(store.optimization_result_path.read_text(encoding="utf-8"))
            official = payload.get("official_result") if isinstance(payload.get("official_result"), dict) else {}
            if objective.kind == "profit_usd":
                current = official.get("profit_usd") if isinstance(official, dict) else None
            else:
                current = payload.get("score")
        except Exception:
            current = None
    met = False
    if objective.target is None:
        met = current is not None
    elif current is not None:
        try:
            met = float(current) >= objective.target
        except (TypeError, ValueError):
            met = False
    summary = (
        f"{objective.kind} current={current} target={objective.target} met={met}"
        if objective.target is not None
        else f"{objective.kind} current={current}"
    )
    return {
        "kind": objective.kind,
        "target": objective.target,
        "current": current,
        "met": met,
        "no_stop_until_target": objective.no_stop_until_target,
        "requires_target": objective.target is not None,
        "summary": summary,
    }


def _has_incomplete_required_loop_tasks(store: ArtifactStore) -> bool:
    return any(not row.get("passes") and row.get("status") != "skipped" for row in store.list("loop_tasks"))


def _load_prior_run_memory(output_root: Path, goal: str, limit: int = 6) -> dict[str, Any]:
    goal_terms = _meaningful_memory_terms(goal)
    goal_topics = _fallback_prompt_topics(goal)
    checked: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    if output_root.exists():
        candidates = sorted(
            (path for path in output_root.iterdir() if path.is_dir() and _is_run_dir(path.name)),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    else:
        candidates = []
    for run_dir in candidates:
        if len(checked) >= limit:
            break
        report_path = run_dir / "final_report.md"
        if not report_path.exists():
            continue
        run_record = _first_json_row(run_dir / "runs.json")
        report_text = report_path.read_text(encoding="utf-8", errors="replace")
        prior_goal = str(run_record.get("user_goal", run_dir.name))
        directions = _prior_report_directions(report_text)
        open_questions = _prior_report_open_questions(report_text)
        report_terms = _meaningful_memory_terms(" ".join([prior_goal, *directions, *open_questions]))
        report_topics = _fallback_prompt_topics(" ".join([prior_goal, *directions, *open_questions]))
        overlap = sorted(goal_terms & report_terms)
        topic_overlap = sorted(goal_topics & report_topics)
        relevance = _prior_memory_relevance(goal_terms, report_terms, goal_topics, report_topics)
        if relevance < 0.34:
            skipped.append(
                {
                    "run_id": run_dir.name,
                    "goal": prior_goal,
                    "report_path": str(report_path),
                    "reason": "not_related_to_current_goal",
                    "overlap_terms": overlap[:8],
                    "overlap_topics": topic_overlap[:8],
                    "relevance": round(relevance, 3),
                }
            )
            continue
        checked.append(
            {
                "run_id": run_dir.name,
                "goal": prior_goal,
                "report_path": str(report_path),
                "completed_at": run_record.get("completed_at"),
                "overlap_terms": overlap[:12],
                "overlap_topics": topic_overlap[:8],
                "relevance": round(relevance, 3),
                "directions": directions[:8],
                "open_questions": open_questions[:8],
                "report_sha256": hashlib.sha256(report_text.encode("utf-8")).hexdigest()[:16],
            }
        )
    avoid = _dedupe_strings(
        [
            direction
            for item in checked
            for direction in item.get("directions", [])
            if isinstance(direction, str) and direction.strip()
        ]
    )[:16]
    unresolved = _dedupe_strings(
        [
            question
            for item in checked
            for question in item.get("open_questions", [])
            if isinstance(question, str) and question.strip()
        ]
    )[:12]
    return {
        "schema_version": "prior_run_memory_v1",
        "goal": goal,
        "checked_run_count": len(checked),
        "checked_reports": checked,
        "skipped_reports": skipped[:12],
        "avoid_directions": avoid,
        "unresolved_directions": unresolved,
        "policy": (
            "Every run checks prior final_report.md artifacts before planning. "
            "Only reports related to the current goal are allowed into memory. "
            "Use unresolved directions first when relevant, and avoid repeating prior directions unless the prompt asks for replication."
        ),
    }


MEMORY_STOPWORDS = RUN_SLUG_STOPWORDS | {
    "adoption",
    "agent",
    "agents",
    "analysis",
    "answer",
    "approach",
    "artifact",
    "artifacts",
    "claim",
    "claims",
    "current",
    "direction",
    "directions",
    "effect",
    "evidence",
    "final",
    "goal",
    "hypothesis",
    "intelligence",
    "question",
    "questions",
    "report",
    "run",
    "source",
    "sources",
    "system",
    "systems",
}


def _meaningful_memory_terms(text: str) -> set[str]:
    terms = set(_goal_terms(text))
    return {term for term in terms if term not in MEMORY_STOPWORDS and len(term) >= 4}


def _prior_memory_relevance(
    goal_terms: set[str],
    report_terms: set[str],
    goal_topics: set[str],
    report_topics: set[str],
) -> float:
    topic_overlap = goal_topics & report_topics
    term_overlap = goal_terms & report_terms
    if topic_overlap:
        return 1.0
    if not goal_terms or not report_terms:
        return 0.0
    overlap_ratio = len(term_overlap) / max(1, min(len(goal_terms), len(report_terms)))
    if len(term_overlap) >= 2:
        return max(0.5, overlap_ratio)
    if len(term_overlap) == 1 and len(goal_terms) <= 3:
        return 0.34
    return overlap_ratio


def _prior_report_directions(report_text: str) -> list[str]:
    directions: list[str] = []
    for heading in ["Key Takeaways", "Hypothesis Evidence Matrix", "Executive Synthesis"]:
        directions.extend(_markdown_section_bullets(report_text, heading, limit=6))
    return [_short_memory_text(item) for item in _dedupe_strings(directions) if item.strip()]


def _prior_report_open_questions(report_text: str) -> list[str]:
    return [_short_memory_text(item) for item in _markdown_section_bullets(report_text, "Open Questions", limit=8)]


def _markdown_section_bullets(markdown: str, heading: str, limit: int) -> list[str]:
    pattern = re.compile(rf"^##+\s+{re.escape(heading)}\s*$", re.I | re.M)
    match = pattern.search(markdown)
    if not match:
        return []
    rest = markdown[match.end() :]
    next_heading = re.search(r"^##+\s+", rest, flags=re.M)
    section = rest[: next_heading.start()] if next_heading else rest
    bullets = []
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith(("-", "*", "•")):
            bullets.append(re.sub(r"^[-*•]\s*", "", stripped).strip())
        elif stripped and heading.lower() == "executive synthesis":
            bullets.append(stripped)
        if len(bullets) >= limit:
            break
    return bullets


def _short_memory_text(text: str, max_chars: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:max_chars].rstrip()


def _apply_prior_memory_to_angles(
    angles: list[str],
    prior_run_memory: Optional[dict[str, Any]],
    *,
    limit: int,
) -> list[str]:
    if not prior_run_memory or not prior_run_memory.get("checked_reports"):
        return angles
    unresolved = [str(item) for item in prior_run_memory.get("unresolved_directions", []) if str(item).strip()]
    avoid = [str(item) for item in prior_run_memory.get("avoid_directions", []) if str(item).strip()]
    memory_angles = [f"unresolved prior question: {_short_query_label(item, max_words=9)}" for item in unresolved[:limit]]
    retained = [
        angle
        for angle in angles
        if not any(_direction_overlap(angle, prior) >= 0.65 for prior in avoid[:8])
    ]
    return _dedupe_strings([*memory_angles, *retained, *angles])[:limit]


def _direction_overlap(left: str, right: str) -> float:
    left_terms = set(_goal_terms(left))
    right_terms = set(_goal_terms(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(1, min(len(left_terms), len(right_terms)))


def _goal_with_prior_memory_context(goal: str, prior_run_memory: Optional[dict[str, Any]]) -> str:
    if not prior_run_memory or not prior_run_memory.get("checked_reports"):
        return goal
    unresolved = [str(item) for item in prior_run_memory.get("unresolved_directions", []) if str(item).strip()]
    avoid = [str(item) for item in prior_run_memory.get("avoid_directions", []) if str(item).strip()]
    if unresolved:
        return f"{goal} unresolved prior question {_short_query_label(unresolved[0], max_words=10)}"
    if avoid:
        return f"{goal} novel direction beyond {_short_query_label(avoid[0], max_words=10)}"
    return goal


def _research_architecture_payload(config: HarnessConfig, task_mode: Optional[str]) -> dict[str, object]:
    return {
        "enabled_for_mode": task_mode == "research",
        "lead_agent": {
            "role": "lead_research_orchestrator",
            "model": config.research_lead_model or config.llm_model,
            "responsibilities": [
                "decompose the question into independent directions",
                "fan out subagent searches in parallel when directions are independent",
                "start with broad landscape queries before narrowing",
                "merge evidence and route weak spots to critic or follow-up search",
            ],
        },
        "subagents": {
            "role": "parallel_research_subagents",
            "model": config.research_subagent_model or config.llm_model,
            "parallelism": "asyncio.gather over independent search agents or query variants",
            "search_policy": "start wide, evaluate source yield, then progressively narrow",
        },
        "judge_rubric": _research_judge_rubric(),
    }


def _research_judge_rubric() -> list[dict[str, object]]:
    return [
        {"name": "factual_accuracy", "question": "Do claims match retrieved sources?", "weight": 0.25},
        {"name": "citation_accuracy", "question": "Do cited sources support the cited claims?", "weight": 0.2},
        {"name": "completeness", "question": "Are all requested aspects covered?", "weight": 0.2},
        {"name": "source_quality", "question": "Were primary or authoritative sources preferred?", "weight": 0.2},
        {"name": "tool_efficiency", "question": "Were the right tools used a reasonable number of times?", "weight": 0.15},
    ]


def _derive_search_angles(goal: str, interpretation: dict[str, Any], task_type: TaskType, limit: int) -> list[str]:
    topic_queries = [str(query) for query in interpretation.get("topic_queries", []) if str(query).strip()]
    terms = _goal_terms(goal)
    anchors = _phrase_chunks_from_goal(goal, terms, limit=limit)
    angles: list[str] = []
    for query in topic_queries:
        angles.append(f"prompt-derived query: {_short_query_label(query)}")
    for anchor in anchors:
        angles.append(f"evidence about {anchor}")
    if task_type == "bounded":
        angles.append(f"constraints in {anchors[0] if anchors else 'the requested objective'}")
    else:
        angles.append(f"counter-evidence about {anchors[0] if anchors else 'the requested claim'}")
    return _dedupe_strings(angles)[:limit] or ["prompt-derived evidence"]


def _derive_hypothesis_angles(goal: str, interpretation: dict[str, Any], task_type: TaskType, limit: int) -> list[str]:
    terms = _goal_terms(goal)
    anchors = _phrase_chunks_from_goal(goal, terms, limit=max(limit, 2))
    topic_queries = [str(query) for query in interpretation.get("topic_queries", []) if str(query).strip()]
    angles = [f"claim about {anchor}" for anchor in anchors]
    angles.extend(f"direction from {_short_query_label(query)}" for query in topic_queries[:limit])
    if task_type == "bounded":
        angles.append(f"way to satisfy {anchors[0] if anchors else 'the objective'}")
    else:
        angles.append(f"open question from {anchors[0] if anchors else 'the prompt'}")
    return _dedupe_strings(angles)[:limit] or ["prompt-derived direction"]


def _phrase_chunks_from_goal(goal: str, terms: list[str], limit: int) -> list[str]:
    quoted = [match.strip() for match in re.findall(r"[\"']([^\"']{4,80})[\"']", goal) if match.strip()]
    chunks = quoted[:limit]
    window = 3
    while len(chunks) < limit and terms:
        start = len(chunks) * window
        chunk_terms = terms[start : start + window]
        if not chunk_terms:
            break
        chunks.append(" ".join(chunk_terms))
    return chunks[:limit]


def _short_query_label(query: str, max_words: int = 7) -> str:
    words = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", query)
    return " ".join(words[:max_words]) or query[:80]


def _mixed_source_strategy(goal: str, plan: ResearchPlan) -> list[SourceStrategyItem]:
    concepts = _goal_terms(goal)
    core = " ".join(concepts[:6]) or goal
    broad = _broad_landscape_query(goal, plan)
    topic_queries = plan.topic_queries
    topics = set(plan.topics)
    if topic_queries:
        first_query = topic_queries[0]
    else:
        first_query = f"{core} sources evidence"
    method_query = topic_queries[1] if len(topic_queries) > 1 else f"{core} evidence methods"
    implementation_query = topic_queries[2] if len(topic_queries) > 2 else f"{core} resources data"
    limitations_query = topic_queries[3] if len(topic_queries) > 3 else f"{core} limitations counter evidence"
    adoption_query = topic_queries[4] if len(topic_queries) > 4 else f"{core} applications resources"
    social_query = topic_queries[5] if len(topic_queries) > 5 else first_query
    scholarly_run = _looks_like_scholarly_request(goal, plan)
    if scholarly_run:
        return [
            SourceStrategyItem(
                name="scholarly_literature",
                retriever="openalex",
                purpose="primary papers and surveys",
                queries=[first_query, method_query],
                limit=8,
            ),
            SourceStrategyItem(
                name="semantic_scholar_literature",
                retriever="semantic_scholar",
                purpose="paper API retrieval and citation-oriented abstracts",
                queries=[method_query, implementation_query],
                limit=8,
            ),
            SourceStrategyItem(
                name="preprints_and_benchmarks",
                retriever="arxiv",
                purpose="preprints, benchmarks, and empirical evidence",
                queries=[implementation_query, limitations_query],
                limit=8,
            ),
            SourceStrategyItem(
                name="curated_overview",
                retriever="wikipedia",
                purpose="overview pages and curated external references",
                queries=[first_query, broad],
                limit=5,
            ),
            SourceStrategyItem(
                name="targeted_web_sources",
                retriever="web",
                purpose="only targeted academic pages, datasets, or course bibliographies",
                queries=[f"{first_query} site:edu OR site:ac.uk OR site:arxiv.org", f"{method_query} bibliography"],
                limit=5,
            ),
        ]
    return [
        SourceStrategyItem(
            name="broad_landscape",
            retriever="openalex",
            purpose="prompt-derived source overview",
            queries=[broad, first_query],
            limit=8,
        ),
        SourceStrategyItem(
            name="preprints_and_evidence",
            retriever="arxiv",
            purpose="prompt-related preprints and evidence",
            queries=[method_query, f"{core} evidence evaluation"],
            limit=8,
        ),
        SourceStrategyItem(
            name="resource_signals",
            retriever="github",
            purpose="prompt-related resources and artifacts",
            queries=[implementation_query, f"{core} resources artifacts"],
            limit=8,
        ),
        SourceStrategyItem(
            name="docs_blogs_sources",
            retriever="docs_blogs",
            purpose="prompt-related practitioner sources",
            queries=[adoption_query, f"{core} practical resources"],
            limit=8,
        ),
        SourceStrategyItem(
            name="wikipedia_overview",
            retriever="wikipedia",
            purpose="encyclopedic overview and curated external references",
            queries=[core, first_query],
            limit=6,
        ),
        SourceStrategyItem(
            name="social_trend_signals",
            retriever="twitter",
            purpose="public social trend signals",
            queries=[social_query, f"{core} public discussion"],
            limit=6,
        ),
        SourceStrategyItem(
            name="counterevidence_limitations",
            retriever="web",
            purpose="counter-evidence and limitations",
            queries=[limitations_query, f"{core} limitations"],
            limit=8,
        ),
        SourceStrategyItem(
            name="prior_artifact_memory",
            retriever="memory",
            purpose="prior related harness artifacts",
            queries=[core],
            limit=6,
        ),
    ]


def _single_retriever_strategy(goal: str, plan: ResearchPlan, retriever: str) -> list[SourceStrategyItem]:
    mixed = _mixed_source_strategy(goal, plan)
    selected = [item for item in mixed if item.retriever == retriever]
    if selected:
        return selected
    concepts = _goal_terms(goal)
    core = " ".join(concepts[:6]) or goal
    return [
        SourceStrategyItem(
            name=f"{retriever}_general",
            retriever=retriever,
            purpose=angle,
            queries=[f"{core} {angle}", f"{core} evidence limitations"],
            limit=8,
        )
        for angle in plan.search_angles
    ]


def _goal_terms(goal: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9]+", goal.lower())
    return [word for word in words if word not in RUN_SLUG_STOPWORDS]


def _looks_like_scholarly_request(goal: str, plan: ResearchPlan) -> bool:
    normalized = goal.lower()
    if any(term in normalized for term in ["paper", "papers", "literature", "academic", "arxiv", "doi", "sources"]):
        return True
    if any("paper" in query.lower() or "survey" in query.lower() for query in plan.topic_queries):
        return True
    return False


def _is_prediction_market_challenge_goal(goal: str, evaluator_name: Optional[str] = None) -> bool:
    normalized = goal.lower()
    if evaluator_name == "prediction_market":
        return True
    return any(term in normalized for term in ["challenge", "orderbook", "order book", "market-making", "market making", "mm'ing"])


def _normalize_goal_interpretation(payload: dict[str, object], *, fallback_task_type: TaskType, planner: str) -> dict[str, Any]:
    raw_topics = payload.get("topics", [])
    topics = [str(topic).strip().lower().replace("-", "_").replace(" ", "_") for topic in raw_topics if str(topic).strip()] if isinstance(raw_topics, list) else []
    raw_queries = payload.get("topic_queries", [])
    topic_queries = [str(query).strip() for query in raw_queries if str(query).strip()] if isinstance(raw_queries, list) else []
    task_type = str(payload.get("task_type") or fallback_task_type)
    if task_type not in {"bounded", "open_ended"}:
        task_type = fallback_task_type
    return {
        "task_type": task_type,
        "topics": _dedupe_strings(topics),
        "topic_queries": _dedupe_strings(topic_queries),
        "rationale": str(payload.get("rationale", "")),
        "planner": planner,
    }


def _fallback_goal_interpretation(goal: str, evaluator_name: Optional[str], fallback_task_type: TaskType) -> dict[str, Any]:
    topics = _fallback_prompt_topics(goal)
    if evaluator_name == "prediction_market":
        topics.add("prediction_market")
    return {
        "task_type": fallback_task_type,
        "topics": sorted(topics),
        "topic_queries": _topic_query_lenses(goal, topics),
        "rationale": "Offline deterministic fallback used because live LLM interpretation was unavailable.",
        "planner": "deterministic-fallback",
    }


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(value)
    return deduped


def _broad_landscape_query(goal: str, plan: ResearchPlan) -> str:
    topics = set(plan.topics)
    if "prediction_market" in topics and _is_prediction_market_challenge_goal(goal):
        return "prediction markets"
    terms = _goal_terms(goal)
    if not terms:
        return goal
    return " ".join(terms[: min(4, len(terms))])


def _fallback_prompt_topics(goal: str) -> set[str]:
    normalized = goal.lower()
    topics: set[str] = set()
    if "prediction market" in normalized or "prediction-market" in normalized:
        topics.add("prediction_market" if _is_prediction_market_challenge_goal(goal) else "prediction_markets")
    if any(term in normalized for term in ["machine learning", "deep learning", "ml "]) and any(
        term in normalized for term in ["stock", "finance", "financial", "trading", "market", "markets"]
    ):
        topics.add("finance_ml")
    if any(term in normalized for term in ["amm", "lmsr", "automated market maker", "constant product"]):
        topics.add("amm")
    if any(term in normalized for term in ["options", "option pricing", "black scholes", "volatility"]):
        topics.add("options")
    if any(term in normalized for term in ["entropy", "exploration", "regularization"]):
        topics.add("entropy")
    if any(term in normalized for term in ["agent", "llm", "multi-agent", "tool use", "react", "reflexion"]):
        topics.add("agents")
    if any(term in normalized for term in ["brain", "neuroscience", "cognitive"]):
        topics.add("neuroscience")
    return topics


def _topic_query_lenses(goal: str, topics: set[str]) -> list[str]:
    queries: list[str] = []
    core_terms = _goal_terms(goal)
    core = _prompt_phrase_core(goal, core_terms)
    if "prediction_market" in topics:
        queries.append("prediction market challenge evaluation strategy implementation")
        queries.append("prediction market trading strategy empirical evaluation")
        queries.append("prediction market simulator strategy benchmark")
    if "prediction_markets" in topics:
        queries.append(f"{core} forecasting calibration evidence")
        queries.append(f"{core} market prices probability evidence")
    if "finance_ml" in topics:
        queries.append(f"{core} empirical evidence")
        queries.append(f"{core} forecasting evaluation")
    if "amm" in topics:
        queries.append(f"{core} liquidity cost function evidence")
        queries.append(f"{core} arbitrage inventory risk")
    if "entropy" in topics:
        queries.append(f"{core} entropy exploration evidence")
    if "options" in topics:
        queries.append(f"{core} volatility hedging risk evidence")
    if "agents" in topics:
        queries.append(f"{core} agent evidence evaluation")
    if "neuroscience" in topics:
        queries.append(f"{core} neuroscience cognitive evidence")
    deduped: list[str] = []
    for query in queries:
        if query not in deduped:
            deduped.append(query)
    return deduped


def _prompt_phrase_core(goal: str, core_terms: list[str]) -> str:
    normalized = goal.lower()
    phrases = []
    for phrase in [
        "machine learning",
        "automated market maker",
        "prediction market",
        "stock",
        "artificial intelligence",
        "large language model",
    ]:
        if phrase in normalized:
            phrases.append(phrase)
    words = phrases + core_terms[:8]
    deduped: list[str] = []
    for word in words:
        if word not in deduped:
            deduped.append(word)
    return " ".join(deduped) or goal


def _read_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _interrupted_report(run: RunRecord, store: ArtifactStore, reason: str) -> str:
    rounds = store.list("evolution_rounds")
    evaluations = store.list("variant_evaluations")
    best_eval = max(evaluations, key=lambda row: float(row.get("score", 0.0) or 0.0), default={})
    lines = [
        f"# Partial Report: {run.user_goal}",
        "",
        "## Key Takeaways",
        "- The run was interrupted before normal completion, so this report packages the artifacts available at the stopping point.",
        f"- Completed evolutionary rounds: {len(rounds)}.",
        f"- Recorded variant evaluations: {len(evaluations)}.",
    ]
    if best_eval:
        lines.append(
            f"- Best retained evaluation: `{best_eval.get('variant_id', 'unknown')}` with score {float(best_eval.get('score', 0.0) or 0.0):.3f}."
        )
    lines.extend(
        [
            "",
            "## Important Artifacts",
            f"- Progress log: `{store.progress_path.name}`",
            f"- PRD/status: `{store.prd_path.name}`",
            f"- Cost: `{store.cost_path.name}`",
            f"- Failed paths: `{store.failed_paths_path.name}`",
            f"- Harness changes: `{store.harness_changes_path.name}`",
        ]
    )
    if store.optimization_result_path.exists():
        lines.append(f"- Optimization result: `{store.optimization_result_path.name}`")
    if store.optimal_code_path.exists():
        lines.append(f"- Optimal code: `{store.optimal_code_path.name}`")
    if store.champion_tree_path.exists():
        lines.append(f"- Champion tree: `{store.champion_tree_path.name}`")
    lines.extend(
        [
            "",
            "## Run Interrupted",
            f"- Reason: {reason}",
            "- Treat conclusions as partial and use the artifact trail to resume or inspect the exact stopping point.",
            "",
        ]
    )
    return "\n".join(lines)


def _first_json_row(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else {}


def _prd_task_from_loop_task(task: dict[str, object], index: int) -> dict[str, object]:
    priority = int(task.get("priority", index) or index)
    dependencies = [] if priority <= 1 else [f"US-{priority - 1:03d}"]
    return {
        "id": f"US-{priority:03d}",
        "source_task_id": task.get("id"),
        "title": task.get("title"),
        "kind": task.get("action"),
        "priority": priority,
        "status": task.get("status"),
        "passes": bool(task.get("passes")),
        "attempts": int(task.get("attempts", 0) or 0),
        "dependencies": dependencies,
        "params": task.get("params", {}),
        "acceptanceCriteria": task.get("acceptance_criteria", []),
        "result_summary": task.get("result_summary"),
        "last_error": task.get("last_error"),
    }


def _prd_task(
    index: int,
    title: str,
    kind: str,
    params: dict[str, object],
    acceptance_criteria: list[str],
    dependencies: list[str],
    traces: list[dict[str, object]],
    role: str,
) -> dict[str, object]:
    matching = [trace for trace in traces if trace.get("role") == role]
    failed = [trace for trace in matching if trace.get("status") != "completed"]
    status = "pending"
    if matching:
        status = "failed" if failed else "passed"
    return {
        "id": f"US-{index:03d}",
        "title": title,
        "kind": kind,
        "priority": index,
        "status": status,
        "passes": bool(matching and not failed),
        "attempts": len(matching),
        "dependencies": dependencies,
        "params": params,
        "acceptanceCriteria": acceptance_criteria,
        "result_summary": matching[-1].get("output_summary") if matching else None,
        "last_error": "; ".join(str(error) for trace in failed for error in trace.get("errors", [])) or None,
        "trace_ids": [trace.get("id") for trace in matching],
    }
