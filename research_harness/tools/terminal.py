"""Bounded host-terminal capability for model-directed research runs.

This is intentionally argv-based: it never invokes a shell or inherits the
parent environment.  It is suitable for real read-only commands that can
recover an evidence run (for example ``curl``ing a known paper URL or querying
npm package metadata), not for executing arbitrary downloaded programs.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import tempfile
import time
import urllib.parse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

from .base import ToolContext, ToolResult
from .research import _public_url_error
from ..schemas import Source


_DEFAULT_COMMANDS = frozenset({"curl", "npm", "git", "rg"})
_CURL_WRITE_OR_CREDENTIAL_FLAGS = {
    "-b", "-c", "-d", "-F", "-K", "-o", "-T", "-u", "-x",
    "--config", "--cookie", "--cookie-jar", "--data", "--data-raw",
    "--data-binary", "--form", "--netrc", "--oauth2-bearer", "--output",
    "--proxy", "--remote-name", "--upload-file", "--user",
}
_GIT_READ_ONLY_SUBCOMMANDS = frozenset({"branch", "diff", "log", "ls-files", "remote", "rev-parse", "show", "status", "tag"})
_NPM_READ_ONLY_SUBCOMMANDS = frozenset({"help", "info", "ping", "search", "view"})


class TerminalExecutionTool:
    """Execute a real, read-only terminal command inside the workspace boundary."""

    name = "execute_terminal"
    is_read_only = True
    description = (
        "Execute a real read-only terminal command without a shell. Available commands are curl, npm, git, and rg. "
        "Use curl for known public HTTP(S) URLs or npm for registry metadata. Commands, arguments, stdout, stderr, and exit status are recorded exactly. "
        "This does not evade bot challenges, execute arbitrary packages, use credentials, or modify files."
    )
    input_schema = {
        "type": "object",
        "required": ["command"],
        "properties": {
            "command": {"type": "string", "enum": sorted(_DEFAULT_COMMANDS)},
            "args": {"type": "array", "items": {"type": "string", "maxLength": 4000}, "maxItems": 80},
            "working_directory": {"type": "string", "minLength": 1, "maxLength": 512},
            "timeout_seconds": {"type": "number", "minimum": 1, "maximum": 60},
        },
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        allowed_commands: Sequence[str] = tuple(_DEFAULT_COMMANDS),
        runner: Callable[[list[str], Path, float], subprocess.CompletedProcess[str]] | None = None,
    ):
        self.allowed_commands = frozenset(allowed_commands)
        self._runner = runner or _run_command

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        command = str(arguments["command"])
        args = [str(value) for value in arguments.get("args", [])]
        if command not in self.allowed_commands:
            return ToolResult("error", error=f"Terminal command '{command}' is not enabled by this harness.")
        safety_error = _command_safety_error(command, args)
        if safety_error:
            return ToolResult("error", error=safety_error)
        executable = shutil.which(command)
        if executable is None:
            return ToolResult("error", error=f"Terminal command '{command}' is not installed on PATH.", retryable=False)
        try:
            workdir = _workspace_directory(context.workspace, arguments.get("working_directory"))
        except ValueError as exc:
            return ToolResult("error", error=str(exc))
        timeout_seconds = float(arguments.get("timeout_seconds", 30))
        argv = [executable, *args]
        started = time.perf_counter()
        try:
            completed = await asyncio.to_thread(self._runner, argv, workdir, timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                "error",
                {"argv": [command, *args], "stdout": _truncate(exc.stdout), "stderr": _truncate(exc.stderr), "exit_code": 124, "duration_ms": int((time.perf_counter() - started) * 1000)},
                error=f"Terminal command exceeded its {timeout_seconds:g}-second timeout.",
                retryable=True,
            )
        except OSError as exc:
            return ToolResult("error", error=f"Could not start terminal command: {exc}", retryable=True)
        duration_ms = int((time.perf_counter() - started) * 1000)
        data = {
            "argv": [command, *args],
            "stdout": _truncate(completed.stdout),
            "stderr": _truncate(completed.stderr),
            "exit_code": completed.returncode,
            "duration_ms": duration_ms,
        }
        if completed.returncode == 0:
            metadata = _curl_sources(args, data["stdout"], context) if command == "curl" else []
            return ToolResult("ok", data, source_metadata=metadata)
        return ToolResult(
            "error",
            data,
            error=_truncate(completed.stderr, 1500) or f"Terminal command exited with code {completed.returncode}.",
            retryable=completed.returncode in {6, 7, 28, 56},
        )


def _run_command(argv: list[str], working_directory: Path, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    """Run one direct executable with an ephemeral home and no inherited secrets."""
    executable_dir = str(Path(argv[0]).parent)
    safe_path = os.pathsep.join(dict.fromkeys([executable_dir, "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]))
    with tempfile.TemporaryDirectory(prefix="research_harness_terminal_") as home:
        return subprocess.run(
            argv,
            cwd=working_directory,
            env={"HOME": home, "PATH": safe_path, "LANG": "C", "LC_ALL": "C", "NO_COLOR": "1"},
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )


def _workspace_directory(workspace: Any, requested: object) -> Path:
    root = Path(workspace).resolve()
    candidate = root if requested is None else (root / str(requested)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Terminal working_directory must remain inside the approved workspace.") from exc
    if not candidate.is_dir():
        raise ValueError("Terminal working_directory does not exist or is not a directory.")
    return candidate


def _command_safety_error(command: str, args: Sequence[str]) -> str | None:
    if command == "curl":
        for index, arg in enumerate(args):
            flag = arg.split("=", 1)[0]
            if flag in _CURL_WRITE_OR_CREDENTIAL_FLAGS:
                return f"curl flag '{flag}' is not available in the read-only terminal tool."
            if flag in {"-L", "--location"}:
                return "curl redirects are not available here; use fetch_document, which validates every redirect destination."
            if arg.startswith(("file:", "ftp:", "gopher:", "smb:")):
                return "curl accepts public HTTP(S) URLs only."
            if arg.startswith(("http://", "https://")):
                error = _public_url_error(arg)
                if error:
                    return error
            if index and args[index - 1] == "--request" and arg.upper() not in {"GET", "HEAD"}:
                return "curl is limited to GET and HEAD requests."
        return None
    if command == "npm":
        if any(arg == "--registry" or arg.startswith("--registry=") for arg in args):
            return "npm custom registries are not available in the read-only terminal tool."
        subcommand = _first_non_option(args)
        if subcommand not in _NPM_READ_ONLY_SUBCOMMANDS and args not in (["--version"], ["-v"], ["--help"]):
            return "npm is limited to view, info, search, ping, help, and version output in the read-only terminal tool."
        return None
    if command == "git":
        subcommand = _first_non_option(args)
        if subcommand not in _GIT_READ_ONLY_SUBCOMMANDS:
            return "git is limited to read-only inspection subcommands in the terminal tool."
        return None
    # rg does not mutate when invoked without an external preprocessor.
    if any(arg.startswith("--pre=") or arg == "--pre" for arg in args):
        return "rg preprocessors are not available in the terminal tool."
    return None


def _first_non_option(args: Sequence[str]) -> str | None:
    for arg in args:
        if not arg.startswith("-"):
            return arg
    return None


def _curl_sources(args: Sequence[str], stdout: str, context: ToolContext) -> list[dict[str, Any]]:
    """Make a successful public curl request citeable by the final-answer gate."""
    sources: list[Source] = []
    for url in dict.fromkeys(arg for arg in args if arg.startswith(("http://", "https://"))):
        source = Source(
            url=url,
            title=f"Terminal curl: {url[:220]}",
            author=urllib.parse.urlsplit(url).netloc,
            date="",
            source_type="terminal_curl",
            summary=stdout[:800],
            relevance_score=1.0,
            credibility_score=0.70,
        )
        if context.store is not None:
            source = context.store.add_source(source)
        sources.append(source)
    return [asdict(source) for source in sources]


def _truncate(value: str | bytes | None, limit: int = 20_000) -> str:
    text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value or "")
    return text[:limit]
