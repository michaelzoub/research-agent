"""Approved, bounded nested model loops exposed through the normal tool layer."""
from __future__ import annotations

import asyncio
import json
import time
import threading
import weakref
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .agent_loop import AgentLoop, AgentRunConfig
from .schemas import new_id, now_iso
from .store import ArtifactStore
from .tools.base import ToolContext, ToolRegistry, ToolResult


@dataclass(frozen=True)
class WorkerBudget:
    max_iterations: int = 4
    max_tokens: int = 4000
    max_tool_calls: int = 8
    max_runtime_seconds: float = 120.0
    max_cost_usd: Optional[float] = None


@dataclass(frozen=True)
class WorkerProfile:
    name: str
    prompt: str
    allowed_tools: tuple[str, ...] = ()
    model_provider: Optional[str] = None
    model_name: Optional[str] = None
    budget: WorkerBudget = field(default_factory=WorkerBudget)
    readable_roots: tuple[Path, ...] = ()


@dataclass(frozen=True)
class WorkerResult:
    parent_run_id: str
    worker_run_id: str
    profile: str
    status: str
    termination_reason: str
    findings: str
    artifacts_path: str
    events_path: str
    runtime_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    tool_calls: int
    error: Optional[str] = None


class WorkerRegistry:
    """Own worker definitions and enforce aggregate nested-loop limits."""

    def __init__(self, profiles: Sequence[WorkerProfile], decider_factory: Callable[[WorkerProfile], Any],
                 *, max_workers_per_parent: int = 8, max_parallel_workers: int = 2):
        self._profiles = {profile.name: profile for profile in profiles}
        if len(self._profiles) != len(profiles):
            raise ValueError("worker profile names must be unique")
        self._decider_factory = decider_factory
        self.max_workers_per_parent = max(1, max_workers_per_parent)
        self._max_parallel_workers = max(1, max_parallel_workers)
        self._semaphores: Any = weakref.WeakKeyDictionary()
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def resolve(self, name: str) -> WorkerProfile:
        try:
            return self._profiles[name]
        except KeyError as exc:
            raise KeyError(f"Unknown worker profile '{name}'. Approved profiles: {', '.join(sorted(self._profiles))}") from exc

    def profile_names(self) -> list[str]:
        return sorted(self._profiles)

    async def delegate(self, profile_name: str, assignment: str, parent_context: ToolContext,
                       parent_tools: ToolRegistry) -> WorkerResult:
        if parent_context.delegation_depth:
            raise PermissionError("Recursive delegation is disabled.")
        profile = self.resolve(profile_name)
        parent_id = parent_context.run_id or parent_context.parent_run_id or "untracked_parent"
        with self._lock:
            count = self._counts.get(parent_id, 0)
            if count >= self.max_workers_per_parent:
                raise RuntimeError(f"Worker-count limit exhausted for parent run '{parent_id}'.")
            self._counts[parent_id] = count + 1
        loop = asyncio.get_running_loop()
        semaphore = self._semaphores.get(loop)
        if semaphore is None:
            semaphore = self._semaphores[loop] = asyncio.Semaphore(self._max_parallel_workers)
        async with semaphore:
            return await self._run(profile, assignment, parent_context, parent_tools, parent_id)

    async def _run(self, profile: WorkerProfile, assignment: str, parent_context: ToolContext,
                   parent_tools: ToolRegistry, parent_id: str) -> WorkerResult:
        worker_id = new_id("worker")
        started = time.perf_counter()
        root = (Path(parent_context.store.root) if parent_context.store else Path(parent_context.workspace)) / "workers" / worker_id
        store = ArtifactStore(root, echo_progress=False, sqlite_path=root / "world.sqlite")
        allowed_roots = _scoped_roots(parent_context.readable_roots, profile.readable_roots)
        registry = parent_tools.subset(profile.allowed_tools)
        decider = self._decider_factory(profile)
        llm = getattr(decider, "llm", None)
        token_reader = lambda: int(getattr(llm, "total_prompt_tokens", 0)) + int(getattr(llm, "total_completion_tokens", 0))
        context = ToolContext(
            workspace=parent_context.workspace, readable_roots=allowed_roots, store=store,
            run_id=worker_id, parent_run_id=parent_id, worker_run_id=worker_id,
            delegation_depth=parent_context.delegation_depth + 1, token_reader=token_reader,
        )
        error: Optional[str] = None
        try:
            result = await AgentLoop(
                decider, registry,
                AgentRunConfig(max_iterations=profile.budget.max_iterations, max_tokens=profile.budget.max_tokens,
                               max_tool_calls=profile.budget.max_tool_calls, max_runtime_seconds=profile.budget.max_runtime_seconds,
                               max_cost_usd=profile.budget.max_cost_usd),
                system_messages=[profile.prompt, "Complete only the bounded assignment. Return findings to the lead agent; do not assume control of the parent run."],
            ).run(assignment, context)
        except Exception as exc:
            result = None
            error = f"{type(exc).__name__}: {exc}"
        if result is not None and result.status != "completed" and error is None:
            error = next((event.error for event in reversed(result.events) if event.error), result.termination_reason)
        prompt_tokens = int(getattr(llm, "total_prompt_tokens", 0))
        completion_tokens = int(getattr(llm, "total_completion_tokens", 0))
        cost = float(llm.total_cost()) if llm and hasattr(llm, "total_cost") else 0.0
        payload = WorkerResult(
            parent_run_id=parent_id, worker_run_id=worker_id, profile=profile.name,
            status=result.status if result else "failed", termination_reason=result.termination_reason if result else "failed",
            findings=result.final_answer if result else "", artifacts_path=str(root), events_path=str(store.agent_event_log_path),
            runtime_ms=int((time.perf_counter() - started) * 1000), prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens,
            cost_usd=round(cost, 6), tool_calls=len(result.tool_calls) if result else 0, error=error,
        )
        (root / "worker_result.json").write_text(json.dumps(asdict(payload), indent=2, sort_keys=True), encoding="utf-8")
        return payload


class DelegateTaskTool:
    name = "delegate_task"
    is_read_only = False
    description = "Delegate one bounded assignment to an approved worker profile. The worker returns findings; the lead agent remains the controller and final synthesizer."

    def __init__(self, workers: WorkerRegistry, parent_tools: ToolRegistry):
        self.workers, self.parent_tools = workers, parent_tools
        self.input_schema = {"type": "object", "required": ["profile", "assignment"], "properties": {
            "profile": {"type": "string", "enum": workers.profile_names()},
            "assignment": {"type": "string", "minLength": 1, "maxLength": 12000},
        }, "additionalProperties": False}

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            result = await self.workers.delegate(str(arguments["profile"]), str(arguments["assignment"]), context, self.parent_tools)
        except (KeyError, PermissionError, RuntimeError, ValueError) as exc:
            return ToolResult("error", error=str(exc), retryable=False)
        data = asdict(result)
        return ToolResult("ok" if result.status == "completed" else "error", data, error=result.error or (None if result.status == "completed" else result.termination_reason))


def _scoped_roots(parent: Sequence[Any], requested: Sequence[Path]) -> tuple[Path, ...]:
    parent_roots = tuple(Path(item).resolve() for item in parent)
    if not requested:
        return parent_roots
    scoped = tuple(Path(item).resolve() for item in requested)
    if any(not any(root == approved or root.is_relative_to(approved) for approved in parent_roots) for root in scoped):
        raise PermissionError("Worker readable roots must be contained by the parent permissions.")
    return scoped
