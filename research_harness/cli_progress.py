"""Event-only terminal progress projection; never owns execution state."""
from __future__ import annotations

import os
import sys
import time
from typing import Any, Optional, TextIO

from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.text import Text


class _LiveProgressView:
    """A dynamic Rich renderable so Live refreshes time and frames between events."""

    def __init__(self, renderer: "CLIProgressRenderer") -> None:
        self.renderer = renderer

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield Text(self.renderer.render(), no_wrap=True, overflow="crop")


class CLIProgressRenderer:
    SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, *, stream: TextIO = sys.stdout, enabled: bool = True):
        self.stream = stream
        self.enabled = bool(enabled and stream.isatty() and not os.environ.get("CI") and not os.environ.get("NO_COLOR"))
        self.events: list[dict[str, Any]] = []
        self.started = time.monotonic()
        self._live: Optional[Live] = None
        if self.enabled:
            console = Console(
                file=stream,
                force_terminal=True,
                force_interactive=True,
                force_jupyter=False,
                color_system=None,
            )
            self._live = Live(
                _LiveProgressView(self),
                console=console,
                refresh_per_second=12,
                transient=False,
                auto_refresh=True,
            )
            self._live.start(refresh=True)

    def consume(self, event: dict[str, Any]) -> None:
        self.events.append(dict(event))
        if self._live is not None:
            self._live.refresh()

    def render(self) -> str:
        pending: dict[str, dict[str, Any]] = {}
        completed: list[dict[str, Any]] = []
        for event in self.events:
            kind, call_id = event.get("event_type"), str(event.get("tool_call_id") or event.get("model_call_id") or "")
            if kind in {"model_request", "tool_requested"}:
                pending[call_id] = event
            elif kind in {"model_turn", "tool_result"}:
                start = pending.pop(call_id, event)
                completed.append({**start, **event})
        rows = completed[-4:] + list(pending.values())[-4:]
        lines = [f"{self.SPINNER[int((time.monotonic()-self.started)*8) % len(self.SPINNER)] if pending else '✓'} Research run · {time.monotonic()-self.started:.1f}s"]
        for index, row in enumerate(rows):
            active = str(row.get("tool_call_id") or row.get("model_call_id") or "") in pending
            status = "active" if active else str(row.get("result_status") or "completed")
            icon = self.SPINNER[int((time.monotonic()-self.started)*8) % len(self.SPINNER)] if active else {"ok": "✓", "completed": "✓", "error": "✗", "failed": "✗", "skipped": "–", "cancelled": "■"}.get(status, "✓")
            name = str(row.get("tool_name") or ("Model reasoning" if row.get("model_call_id") else row.get("event_type") or "operation"))
            if name == "delegate_task":
                profile = (row.get("arguments") or {}).get("profile")
                name = f"Literature worker · {profile}" if profile else "Delegated worker"
            prefix = "└─" if index == len(rows) - 1 else "├─"
            lines.append(f"{prefix} {icon} {name}")
        return "\n".join(lines)

    def close(self) -> None:
        if self._live is not None:
            self._live.update(_LiveProgressView(self), refresh=True)
            self._live.stop()
            self._live = None

    def __enter__(self):
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
