"""Run initialization and finalization around one model-directed agent loop."""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

from .llm import LLMClient
from .external_services import default_external_service_registry
from .research_agent import AgentRunConfig, ResearchAgent
from .schemas import AgentBudget, CostEvent, RunRecord, now_iso, to_dict
from .search import AlchemySearch, ArxivSearch, DocsBlogsSearch, GitHubSearch, LocalCorpusSearch, OpenAlexSearch, SearchBackend, SemanticScholarSearch, SocialWebSearch, WebSearch, WikipediaSearch
from .sessions import SessionStore, default_session_projects_dir
from .store import ArtifactStore
from .tools import evaluator_context
from .run_visuals import write_research_run_visuals


@dataclass
class HarnessConfig:
    """Safety, budgets, and available capabilities—not a trajectory selector."""
    id: str = "model-directed-agent-v1"
    retriever: str = "auto"
    max_iterations: Optional[int] = None
    evaluator_name: Optional[str] = None
    max_grader_calls: Optional[int] = None
    llm_provider: str = "auto"
    llm_model: str = "gpt-5.2"
    llm_seed: Optional[int] = None
    session_projects_dir: Optional[Path] = None
    resume_session_id: Optional[str] = None
    fork_session_id: Optional[str] = None
    enable_sessions: bool = True
    echo_progress: bool = True
    default_budget: AgentBudget = field(default_factory=AgentBudget)
    max_cost_usd: Optional[float] = None
    workspace_roots: tuple[Path, ...] = ()


class Orchestrator:
    """Initialize one run, invoke ResearchAgent once, then persist its real history."""
    def __init__(self, corpus_path: Path, output_root: Path, config: Optional[HarnessConfig] = None):
        self.corpus_path = corpus_path
        self.output_root = output_root
        self.config = config or HarnessConfig()
        self.llm = LLMClient(provider=self.config.llm_provider, model=self.config.llm_model, seed=self.config.llm_seed)
        if self.config.llm_provider in {"openai", "anthropic"} and not self.llm.is_live:
            raise ValueError(f"--llm-provider {self.config.llm_provider} requires configured credentials.")

    async def run(self, goal: str) -> Tuple[RunRecord, ArtifactStore]:
        run = RunRecord(
            id=self._next_run_id(goal), user_goal=goal, task_type="open_ended",
            harness_config_id=self.config.id, prompt_versions=_prompt_versions(),
            harness_config_snapshot=_config_snapshot(self.config),
        )
        session_store = self._start_session_store(goal, run)
        store = ArtifactStore(self.output_root / run.id, echo_progress=self.config.echo_progress, session_store=session_store)
        store.add_run(run)
        store.append_progress(f"Starting model-directed run {run.id}")
        store.append_progress(f"Goal: {goal}")
        self._write_run_state(store, run, stage="started")
        try:
            grader_bootstrap: Optional[dict[str, Any]] = None
            if self.config.evaluator_name:
                store.append_progress(
                    f"Grader configured ({self.config.evaluator_name}); exposing its registered tools to the model-directed loop."
                )
                grader_bootstrap = _grader_bootstrap_context(self.config.evaluator_name)
                store.write_grader_preflight(grader_bootstrap)
                store.append_progress(
                    f"Grader preflight: ok={grader_bootstrap['ok']} mode={grader_bootstrap['execution_mode']} "
                    f"upstream={grader_bootstrap['upstream_path']}"
                )
                if not grader_bootstrap["ok"]:
                    raise RuntimeError(f"Official grader preflight failed: {grader_bootstrap['reason']}")
            agent_objective = _agent_objective(goal, self.config.evaluator_name, grader_bootstrap)
            agent = ResearchAgent.with_research_tools(
                self.llm,
                [self._retriever_for(name) for name in self._enabled_retrievers()],
                AgentRunConfig(max_iterations=self.config.max_iterations, max_tool_calls=self.config.default_budget.max_tool_calls, max_grader_calls=self.config.max_grader_calls, max_runtime_seconds=self.config.default_budget.max_runtime_seconds, max_cost_usd=self.config.max_cost_usd),
                evaluator_name=self.config.evaluator_name,
            )
            result = await agent.arun(agent_objective, workspace=Path.cwd(), readable_roots=list(self.config.workspace_roots or (Path.cwd(),)), store=store, run_id=run.id)
            run.status = result.status if result.status in {"completed", "needs_input", "partial", "budget_exhausted", "cancelled", "safety_stopped", "failed"} else "failed"
            write_research_run_visuals(store, result.events)
            store.append_progress(f"Agent termination: {result.termination_reason}")
        except Exception as exc:
            run.status = "failed"
            store.append_progress(f"Agent runtime failure: {type(exc).__name__}: {exc}")
            raise
        finally:
            run.total_tokens = self.llm.total_prompt_tokens + self.llm.total_completion_tokens
            run.total_cost = round(self.llm.total_cost(), 6)
            run.completed_at = now_iso()
            self._record_cost_events(store, run)
            cost = self.llm.cost_breakdown()
            cost.update({"run_id": run.id, "completed_at": run.completed_at})
            store.write_cost(cost)
            store.update_run(run)
            self._write_run_state(store, run, stage="completed")
            if session_store is not None:
                session_store.complete_session(status=run.status, summary=f"Run {run.id} {run.status}.")
        return run, store

    def _enabled_retrievers(self) -> list[str]:
        """Return source capabilities enabled by configuration, never a search plan.

        The model receives these as tools and selects whether to call one, which
        one to call, and what query to make.  This boundary exists only so the
        harness can enforce permissions and expose an auditable capability set.
        """
        if self.config.retriever != "auto":
            return [self.config.retriever]
        # The local corpus is intentionally fixture-only.  Mixing example.org
        # records into a live research run makes a failed discovery pass look
        # grounded and trains the agent to report fabricated-looking evidence.
        return ["arxiv", "openalex", "semantic_scholar", "github", "web", "docs_blogs"]

    def _retriever_for(self, retriever: str) -> SearchBackend:
        registry: dict[str, Any] = {"local": lambda: LocalCorpusSearch(self.corpus_path), "arxiv": lambda: ArxivSearch(llm=self.llm), "openalex": OpenAlexSearch, "semantic_scholar": SemanticScholarSearch, "github": GitHubSearch, "web": WebSearch, "docs_blogs": DocsBlogsSearch, "twitter": SocialWebSearch, "wikipedia": WikipediaSearch, "alchemy": AlchemySearch}
        try:
            return registry[retriever.lower()]()
        except KeyError as exc:
            raise ValueError(f"Unknown retriever: {retriever}") from exc

    def _write_run_state(self, store: ArtifactStore, run: RunRecord, *, stage: str) -> None:
        transcript: dict[str, Any] = {}
        if store.agent_transcript_path.exists():
            try:
                transcript = json.loads(store.agent_transcript_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                transcript = {}
        store.write_run_state({
            "schema_version": "model_directed_run_state_v1", "stage": stage, "run": to_dict(run), "goal": run.user_goal,
            "available_tools": self._enabled_retrievers() + ["fetch_document", "inspect_document_figures", "read_workspace_file", "execute_python_analysis", "execute_terminal"] + [tool.name for tool in default_external_service_registry().tools()] + ["consult_specialist", "delegate_task"] + (["evaluate_prediction_market_candidate", "spawn_optimization_agents", "run_parameter_sweep", "save_learning"] if self.config.evaluator_name == "prediction_market" else []),
            "observed_counts": {"events": len(transcript.get("events", [])), "messages": len(transcript.get("messages", [])), "tool_calls": len(transcript.get("tool_calls", []))},
            "termination": transcript.get("termination_reason"),
            "artifacts": {
                "agent_transcript": str(store.agent_transcript_path),
                "event_log": str(store.agent_event_log_path),
                "final_report": str(store.report_path),
                "cost": str(store.cost_path),
                "learnings": str(store.learnings_path),
                "learning_log": str(store.learning_log_path),
                **({"grader_preflight": str(store.grader_preflight_path)} if store.grader_preflight_path.exists() else {}),
                **({"agent_timeline": str(store.agent_timeline_path)} if store.agent_timeline_path.exists() else {}),
                **({"candidate_graph": str(store.candidate_graph_path), "champion_history": str(store.champion_history_path)} if store.candidate_graph_path.exists() else {}),
            },
            "notes": ["This event history is append-only from actual model turns and tool results.", "No predefined plan, source strategy, or execution mode is recorded or used."],
        })

    def _record_cost_events(self, store: ArtifactStore, run: RunRecord) -> None:
        for index, call in enumerate(self.llm.call_history, 1):
            store.add_cost_event(CostEvent(run_id=run.id, component=f"model_turn_{index}", provider=str(call.get("provider") or self.llm.provider), model=str(call.get("model") or self.llm.model), prompt_tokens=int(call.get("prompt_tokens") or 0), completion_tokens=int(call.get("completion_tokens") or 0), cost_usd=float(call.get("cost_usd") or 0.0), metadata={key: value for key, value in call.items() if key not in {"provider", "model", "prompt_tokens", "completion_tokens", "cost_usd"}}))

    def _start_session_store(self, goal: str, run: RunRecord) -> Optional[SessionStore]:
        if not self.config.enable_sessions:
            return None
        try:
            session_store = SessionStore(Path.cwd(), self.config.session_projects_dir or default_session_projects_dir())
            record = session_store.start_session(goal=goal, run_id=run.id, output_dir=self.output_root / run.id, resume_from=self.config.resume_session_id, fork_from=self.config.fork_session_id)
            run.session_id, run.session_jsonl_path, run.session_metadata_path = record.id, str(record.jsonl_path), str(record.metadata_path)
            return session_store
        except OSError:
            return None

    def _next_run_id(self, goal: str) -> str:
        self.output_root.mkdir(parents=True, exist_ok=True)
        numbers = [int(match.group(1)) for path in self.output_root.iterdir() if (match := re.match(r"^(\d+)_run_", path.name))]
        return f"{(max(numbers) + 1 if numbers else 1):03d}_run_{goal_slug(goal)}"


def _agent_objective(
    goal: str,
    evaluator_name: Optional[str],
    grader_bootstrap: Optional[dict[str, Any]] = None,
) -> str:
    """Expose evaluator identity as context, never as a prescribed trajectory."""
    context = evaluator_context(evaluator_name)
    if grader_bootstrap and grader_bootstrap.get("ok"):
        context = (
            f"{context}\n\n"
            "The grader adapter already completed runtime preflight successfully. Do not spend tool calls locating the "
            "challenge repository or starter file. The exact adapter-resolved baseline and path are supplied below; use "
            "this public interface to author a complete candidate, then call evaluate_prediction_market_candidate.\n"
            f"Execution mode: {grader_bootstrap.get('execution_mode')}\n"
            f"Upstream path: {grader_bootstrap.get('upstream_path')}\n"
            f"Baseline path: {grader_bootstrap.get('baseline_path')}\n"
            "Baseline source:\n```python\n"
            f"{grader_bootstrap.get('baseline_code', '').rstrip()}\n```"
        )
    return f"{context}\n\nUser request:\n{goal}" if context else goal


def _grader_bootstrap_context(identifier: str) -> dict[str, Any]:
    from optimization_graders import get_optimization_grader

    grader = get_optimization_grader(identifier)
    preflight = grader.preflight()
    baselines = grader.registered_baselines()
    baseline_path = baselines.get("starter_strategy")
    baseline_code = baseline_path.read_text(encoding="utf-8") if baseline_path and baseline_path.is_file() else ""
    return {
        "grader_id": identifier,
        "ok": bool(preflight.ok),
        "reason": preflight.reason,
        "execution_mode": preflight.execution_mode,
        "docker_sandbox": bool(preflight.docker_sandbox),
        "upstream_path": preflight.upstream_path,
        "baseline_path": str(baseline_path) if baseline_path else None,
        "baseline_code": baseline_code,
    }


def _config_snapshot(config: HarnessConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["workspace_roots"] = [str(path) for path in config.workspace_roots]
    payload["session_projects_dir"] = str(config.session_projects_dir) if config.session_projects_dir else None
    payload["default_budget"] = to_dict(config.default_budget)
    return payload


def _prompt_versions() -> dict[str, str]:
    prompt_dir = Path(__file__).resolve().parent.parent / "prompts"
    return {path.stem: hashlib.sha256(path.read_bytes()).hexdigest()[:16] for path in sorted(prompt_dir.glob("*.md"))}


def _terms(value: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9]+", value.lower()) if len(term) > 2]


def goal_slug(goal: str, max_length: int = 72) -> str:
    words = _terms(goal)
    return ("-".join(words)[:max_length].strip("-") or "research-run")
