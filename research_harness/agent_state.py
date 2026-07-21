"""Mutable state for one model-directed agent trajectory."""
from __future__ import annotations

import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from .schemas import now_iso
from .tools import ToolContext


@dataclass
class AgentEvent:
    sequence: int
    event_type: str
    actor: str
    timestamp: str = field(default_factory=now_iso)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    runtime_ms: Optional[int] = None
    model_call_id: Optional[str] = None
    projected_messages: Optional[list[dict[str, Any]]] = None
    model_turn: Optional[dict[str, Any]] = None
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    arguments: Optional[dict[str, Any]] = None
    result_status: Optional[str] = None
    observation: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    decision_summary: Optional[str] = None
    run_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    worker_run_id: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None


@dataclass
class AgentRunResult:
    final_answer: str
    termination_reason: str
    status: str
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    events: list[AgentEvent]
    sources: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AgentState:
    """All evolving data owned by a single run of the core loop."""

    objective: str
    context: ToolContext
    messages: list[dict[str, Any]]
    initial_cost: float
    started_at_monotonic: float = field(default_factory=time.monotonic)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    events: list[AgentEvent] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    canonical_source_urls: set[str] = field(default_factory=set)
    answer_chunks: list[str] = field(default_factory=list)
    current_iteration: int = 0
    source_stall_count: int = 0
    grader_action_nudge_sent: bool = False
    termination_reason: Optional[str] = None
    termination_error: Optional[str] = None
    model_started_at: Optional[str] = None
    model_started_perf: Optional[float] = None
    model_call_id: Optional[str] = None
    consecutive_model_failures: int = 0

    @classmethod
    def initialize(
        cls,
        *,
        objective: str,
        context: ToolContext,
        initial_cost: float,
        system_messages: Sequence[str],
    ) -> "AgentState":
        return cls(
            objective=objective,
            context=context,
            initial_cost=initial_cost,
            messages=[
                *({"role": "system", "content": content} for content in system_messages),
                {"role": "user", "content": objective},
            ],
        )

    def begin_iteration(self, iteration: int) -> None:
        self.current_iteration = iteration

    def add_sources(self, sources: Sequence[dict[str, Any]]) -> bool:
        added_new = False
        for source in sources:
            url = canonical_citation_url(str(source.get("url") or ""))
            if url and url not in self.canonical_source_urls:
                self.canonical_source_urls.add(url)
                added_new = True
        self.sources.extend(sources)
        return added_new

    def finish_tool_iteration(self, added_new_source: bool) -> None:
        self.source_stall_count = 0 if added_new_source else self.source_stall_count + 1


def canonical_citation_url(value: str) -> str:
    """Compare source URLs independent of HTTP(S) presentation details."""
    parsed = urllib.parse.urlsplit(value.rstrip(".,;"))
    if not parsed.netloc:
        return value.rstrip(".,;")
    return urllib.parse.urlunsplit(("", parsed.netloc.lower(), parsed.path.rstrip("/"), parsed.query, ""))
