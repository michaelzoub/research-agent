from __future__ import annotations

import asyncio
from typing import Any, Optional

from ..sandbox import DockerSandboxRunner
from .base import ToolContext, ToolResult


class CodeExecutionTool:
    name = "execute_python_analysis"
    is_read_only = True
    description = (
        "Run a short Python analysis in a network-isolated, read-only sandbox. "
        "The approved workspace is available at /workspace and is also the working directory, "
        "so relative paths such as outputs/<run>/optimization_trials/<trial>.json can be read."
    )
    input_schema = {"type": "object", "required": ["code"], "properties": {"code": {"type": "string", "minLength": 1, "maxLength": 20000}}, "additionalProperties": False}

    def __init__(self, sandbox: Optional[DockerSandboxRunner] = None):
        self.sandbox = sandbox or DockerSandboxRunner(timeout_seconds=60.0)

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        result = await asyncio.to_thread(
            self.sandbox.execute_python,
            str(arguments["code"]),
            workspace_path=context.workspace,
        )
        if result.exit_code == 0:
            return ToolResult("ok", {"stdout": result.stdout[:20000], "stderr": result.stderr[:5000], "exit_code": result.exit_code})
        return ToolResult("error", {"stdout": result.stdout[:20000], "stderr": result.stderr[:5000], "exit_code": result.exit_code}, error=result.stderr[:1000] or f"Sandboxed Python exited with code {result.exit_code}.", retryable=result.timed_out or result.exit_code in {125, 127})
