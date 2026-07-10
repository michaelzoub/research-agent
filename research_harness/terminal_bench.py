from __future__ import annotations

import inspect
import shlex
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Optional

try:  # Harbor is optional for local unit tests; it is present in real Terminal-Bench runs.
    from harbor.agents.base import BaseAgent as _HarborBaseAgent
except Exception:  # pragma: no cover - exercised only when Harbor is installed.
    _HarborBaseAgent = object


@dataclass(frozen=True)
class TerminalBenchRunConfig:
    llm_provider: str = "auto"
    llm_model: Optional[str] = None
    max_iterations: int = 50
    quiet: bool = True


class ResearchHarnessTerminalBenchAgent(_HarborBaseAgent):
    """External Harbor agent adapter for running the research harness on Terminal-Bench tasks."""

    def __init__(self, config: Optional[TerminalBenchRunConfig] = None):
        self.config = config or TerminalBenchRunConfig()

    @staticmethod
    def name() -> str:
        return "research-harness"

    def version(self) -> Optional[str]:
        try:
            return metadata.version("research-harness")
        except metadata.PackageNotFoundError:
            return None

    async def setup(self, environment: Any) -> None:
        await _maybe_await(environment.exec("python3 -m research_harness.cli --help"))

    async def run(self, instruction: str, environment: Any, context: Any) -> None:
        command = self._command_for_instruction(instruction)
        _record_context(
            context,
            "research_harness_command",
            command,
        )
        result = await _maybe_await(environment.exec(command))
        _record_context(context, "research_harness_exit_code", _result_attr(result, "returncode", 0))
        _record_context(context, "research_harness_stdout", _result_attr(result, "stdout", ""))
        _record_context(context, "research_harness_stderr", _result_attr(result, "stderr", ""))

    def _command_for_instruction(self, instruction: str) -> str:
        parts = [
            "python3",
            "-m",
            "research_harness.cli",
            shlex.quote(instruction),
            "--llm-provider",
            shlex.quote(self.config.llm_provider),
            "--max-iterations",
            str(self.config.max_iterations),
        ]
        if self.config.llm_model:
            parts.extend(["--llm-model", shlex.quote(self.config.llm_model)])
        if self.config.quiet:
            parts.append("--quiet")
        return " ".join(parts)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _record_context(context: Any, key: str, value: Any) -> None:
    if isinstance(context, dict):
        context[key] = value
        return
    if hasattr(context, "metadata") and isinstance(context.metadata, dict):
        context.metadata[key] = value
        return
    setattr(context, key, value)


def _result_attr(result: Any, name: str, default: Any) -> Any:
    if result is None:
        return default
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)
