from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import ToolContext, ToolResult


class FileReadTool:
    name = "read_local_file"
    description = (
        "Read a UTF-8 text file inside the configured workspace. Use for local specifications, artifacts, "
        "or source files. Do not use for web URLs, binary files, or files outside the workspace."
    )
    input_schema = {
        "type": "object",
        "required": ["path"],
        "properties": {
            "path": {"type": "string", "description": "Workspace-relative UTF-8 text path; example: docs/architecture.md"},
            "max_characters": {"type": "integer", "minimum": 1, "maximum": 50000},
        },
        "additionalProperties": False,
    }

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        root = Path(context.workspace).resolve()
        candidate = (root / str(arguments["path"])).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return ToolResult("error", error="Path must remain inside the configured workspace.", retryable=False)
        if not candidate.is_file():
            return ToolResult("error", error="File does not exist or is not a regular file.", retryable=False)
        limit = min(50000, max(1, int(arguments.get("max_characters", 12000))))
        try:
            text = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult("error", error="File is not UTF-8 text.", retryable=False)
        return ToolResult("ok", {"path": str(candidate.relative_to(root)), "content": text[:limit], "truncated": len(text) > limit})
