"""Evidence-backed learning capture for configured optimization graders."""
from __future__ import annotations

from typing import Any

from .base import ToolContext, ToolResult


class SaveLearningTool:
    name = "save_learning"
    is_read_only = False
    description = "Save a confirmed breakthrough, dead end, or robust parameter finding for later optimization agents. Include the actual evaluation evidence; do not record speculation as confirmed."
    input_schema = {"type": "object", "required": ["title", "finding", "evidence", "status"], "properties": {
        "title": {"type": "string", "minLength": 3, "maxLength": 160}, "finding": {"type": "string", "minLength": 12, "maxLength": 4000},
        "evidence": {"type": "string", "minLength": 6, "maxLength": 4000}, "status": {"type": "string", "enum": ["confirmed", "dead_end", "hypothesis"]},
    }, "additionalProperties": False}

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        if context.store is None:
            return ToolResult("error", error="Saving a learning requires an artifact store.")
        path = context.store.append_learning(run_id=context.run_id, **{key: str(arguments[key]) for key in ("title", "finding", "evidence", "status")})
        return ToolResult("ok", {"learning_path": str(path), "status": arguments["status"]})
