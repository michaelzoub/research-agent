from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .base import ToolContext, ToolResult


class FileReadTool:
    name = "read_workspace_file"
    is_read_only = True
    description = "Read UTF-8 text from an explicitly approved workspace root. Never use for secrets, .git data, or paths outside approved roots."
    input_schema = {"type": "object", "required": ["path"], "properties": {"path": {"type": "string", "minLength": 1}, "max_characters": {"type": "integer", "minimum": 1, "maximum": 50000}}, "additionalProperties": False}
    _DENIED_NAMES = {".env", ".git", ".ssh", "credentials", "secrets"}

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        return await asyncio.to_thread(self._read, arguments, context)

    def _read(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        roots = [Path(root).resolve() for root in (context.readable_roots or [context.workspace])]
        raw_path = Path(str(arguments["path"]))
        candidate = (Path(context.workspace) / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
        if any(part in self._DENIED_NAMES or part.startswith(".env") for part in candidate.parts):
            return ToolResult("error", error="The workspace policy denies this sensitive path.")
        root = next((allowed for allowed in roots if _is_within(candidate, allowed)), None)
        if root is None:
            return ToolResult("error", error="Path must remain inside an explicitly approved workspace root.")
        if not candidate.is_file():
            return ToolResult("error", error="File does not exist or is not a regular file.")
        limit = min(50000, max(1, int(arguments.get("max_characters", 12000))))
        try:
            text = candidate.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult("error", error="File is not UTF-8 text.")
        return ToolResult("ok", {"path": str(candidate.relative_to(root)), "content": text[:limit], "truncated": len(text) > limit})


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False
