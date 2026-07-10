from __future__ import annotations

from typing import Any, Optional

from ..sandbox import DockerSandboxRunner
from .base import ToolContext, ToolResult


class CodeExecutionTool:
    name = "execute_python_analysis"
    description = (
        "Run a short self-contained Python analysis in a network-isolated Docker sandbox. "
        "Use for calculations or inspecting data already supplied in the script. Do not use for web access, "
        "filesystem exploration, package installation, or modifying the workspace."
    )
    input_schema = {
        "type": "object",
        "required": ["code"],
        "properties": {"code": {"type": "string", "description": "Self-contained Python source; example: print(sum([1, 2, 3]))"}},
        "additionalProperties": False,
    }

    def __init__(self, sandbox: Optional[DockerSandboxRunner] = None):
        self.sandbox = sandbox or DockerSandboxRunner(timeout_seconds=60.0)

    def execute(self, arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
        code = str(arguments["code"])
        if not code.strip() or len(code) > 20000:
            return ToolResult("error", error="code must contain 1 to 20,000 characters.", retryable=False)
        result = self.sandbox.execute_python(code)
        if result.exit_code == 0:
            return ToolResult("ok", {"stdout": result.stdout[:20000], "stderr": result.stderr[:5000], "exit_code": result.exit_code})
        return ToolResult(
            "error", {"stdout": result.stdout[:20000], "stderr": result.stderr[:5000], "exit_code": result.exit_code},
            error=result.stderr[:1000] or "Sandboxed Python exited with code %s." % result.exit_code,
            retryable=result.timed_out or result.exit_code in {125, 127},
        )
