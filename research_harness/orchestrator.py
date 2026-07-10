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
from .research_agent import AgentRunConfig, ResearchAgent
from .schemas import AgentBudget, CostEvent, RunRecord, now_iso, to_dict
from .search import AlchemySearch, ArxivSearch, DocsBlogsSearch, GitHubSearch, LocalCorpusSearch, OpenAlexSearch, PriorArtifactMemorySearch, SearchBackend, SemanticScholarSearch, SocialWebSearch, WebSearch, WikipediaSearch
from .sessions import SessionStore, default_session_projects_dir
from .store import ArtifactStore


RUN_SLUG_STOPWORDS = {"a", "an", "and", "are", "be", "for", "how", "in", "of", "on", "please", "research", "the", "to", "what", "which", "will"}


@dataclass
class HarnessConfig:
    """Safety, budgets, and available capabilities—not a trajectory selector."""
    id: str = "model-directed-agent-v1"
    retriever: str = "auto"
    max_iterations: int = 12
    evaluator_name: Optional[str] = None
    llm_provider: str = "auto"
    llm_model: str = "gpt-5.2"
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
        self.llm = LLMClient(provider=self.config.llm_provider, model=self.config.llm_model)
        if self.config.llm_provider in {"openai", "anthropic"} and not self.llm.is_live:
            raise ValueError(f"--llm-provider {self.config.llm_provider} requires configured credentials.")

    async def run(self, goal: str) -> Tuple[RunRecord, ArtifactStore]:
        prior_memory = self.load_prior_run_memory(goal)
        run = RunRecord(
            id=self._next_run_id(goal), user_goal=goal, task_type="open_ended",
            harness_config_id=self.config.id, prompt_versions=_prompt_versions(),
            harness_config_snapshot=_config_snapshot(self.config),
        )
        session_store = self._start_session_store(goal, run)
        store = ArtifactStore(self.output_root / run.id, echo_progress=self.config.echo_progress, session_store=session_store)
        store.add_run(run)
        store.write_prior_run_memory(prior_memory)
        store.append_progress(f"Starting model-directed run {run.id}")
        store.append_progress(f"Goal: {goal}")
        self._write_run_state(store, run, prior_memory, stage="started")
        agent = ResearchAgent.with_research_tools(
            self.llm,
            [self._retriever_for(name) for name in self._registered_retrievers()],
            AgentRunConfig(max_iterations=self.config.max_iterations, max_tool_calls=self.config.default_budget.max_tool_calls, max_runtime_seconds=self.config.default_budget.max_runtime_seconds, max_cost_usd=self.config.max_cost_usd),
        )
        try:
            result = await agent.arun(goal, workspace=Path.cwd(), readable_roots=list(self.config.workspace_roots or (Path.cwd(),)), store=store, run_id=run.id)
            run.status = result.status if result.status in {"completed", "needs_input", "partial", "budget_exhausted", "cancelled", "safety_stopped", "failed"} else "failed"
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
            self._write_run_state(store, run, prior_memory, stage="completed")
            if session_store is not None:
                session_store.complete_session(status=run.status, summary=f"Run {run.id} {run.status}.")
        return run, store

    def load_prior_run_memory(self, goal: str, limit: int = 6) -> dict[str, Any]:
        matches: list[dict[str, Any]] = []
        for path in sorted(self.output_root.glob("*_run_*/run_state.json"), reverse=True):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            previous_goal = str(state.get("goal") or "")
            overlap = len(set(_terms(goal)) & set(_terms(previous_goal)))
            if overlap:
                matches.append({"run_id": state.get("run", {}).get("id"), "goal": previous_goal, "status": state.get("run", {}).get("status"), "path": str(path), "overlap": overlap})
            if len(matches) >= limit:
                break
        return {"checked_run_count": len(matches), "relevant_trajectories": matches}

    def _registered_retrievers(self) -> list[str]:
        if self.config.retriever != "auto":
            return [self.config.retriever]
        return ["local", "arxiv", "openalex", "semantic_scholar", "github", "web", "docs_blogs", "memory"]

    def _retriever_for(self, retriever: str) -> SearchBackend:
        registry: dict[str, Any] = {"local": lambda: LocalCorpusSearch(self.corpus_path), "arxiv": lambda: ArxivSearch(llm=self.llm), "openalex": OpenAlexSearch, "semantic_scholar": SemanticScholarSearch, "github": GitHubSearch, "web": WebSearch, "docs_blogs": DocsBlogsSearch, "twitter": SocialWebSearch, "memory": lambda: PriorArtifactMemorySearch(self.output_root), "wikipedia": WikipediaSearch, "alchemy": AlchemySearch}
        try:
            return registry[retriever.lower()]()
        except KeyError as exc:
            raise ValueError(f"Unknown retriever: {retriever}") from exc

    def _write_run_state(self, store: ArtifactStore, run: RunRecord, prior_memory: dict[str, Any], *, stage: str) -> None:
        transcript: dict[str, Any] = {}
        if store.agent_transcript_path.exists():
            try:
                transcript = json.loads(store.agent_transcript_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                transcript = {}
        store.write_run_state({
            "schema_version": "model_directed_run_state_v1", "stage": stage, "run": to_dict(run), "goal": run.user_goal,
            "available_tools": self._registered_retrievers() + ["fetch_document", "read_workspace_file", "execute_python_analysis"],
            "prior_relevant_memory": prior_memory,
            "events": transcript.get("events", []), "messages": transcript.get("messages", []), "tool_calls": transcript.get("tool_calls", []),
            "termination": transcript.get("termination_reason"),
            "artifacts": {"agent_transcript": str(store.agent_transcript_path), "final_report": str(store.report_path), "cost": str(store.cost_path), "prior_memory": str(store.prior_run_memory_path)},
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
    words = [word for word in _terms(goal) if word not in RUN_SLUG_STOPWORDS]
    return ("-".join(words)[:max_length].strip("-") or "research-run")
