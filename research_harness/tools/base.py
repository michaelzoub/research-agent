"""Async, schema-validated capability boundary for the research agent."""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Optional, Protocol, Sequence


@dataclass(frozen=True)
class ToolResult:
    status: str
    data: Any = None
    source_metadata: Sequence[dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    retryable: bool = False

    def as_message(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "data": self.data,
            # Full source records remain durable in the artifact store. Sending
            # full abstracts for every search result back each turn causes
            # context blowups and does not improve tool selection.
            "source_metadata": [
                {
                    key: (str(source.get(key) or "")[:400] if key == "summary" else source.get(key))
                    for key in ("id", "title", "url", "source_type", "relevance_score", "summary")
                    if source.get(key) is not None
                }
                for source in self.source_metadata
            ],
            "error": self.error,
            "retryable": self.retryable,
        }


@dataclass
class ToolContext:
    workspace: Any
    readable_roots: Sequence[Any] = field(default_factory=tuple)
    store: Optional[Any] = None
    run_id: str = ""
    cancelled: bool = False


class BaseTool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]
    is_read_only: bool

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult: ...


class ToolRegistry:
    """The only capability surface exposed to a model-directed run."""
    def __init__(self, tools: Sequence[BaseTool]):
        self._tools = {tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("tool names must be unique")

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, arguments: Any, context: ToolContext) -> ToolResult:
        if context.cancelled:
            return ToolResult("cancelled", error="Tool execution was cancelled.")
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult("error", error=f"Unknown tool '{name}'. Select only a registered tool.")
        if not isinstance(arguments, dict):
            return ToolResult("error", error="Tool arguments must be a JSON object.")
        validation_error = _validate_arguments(arguments, tool.input_schema)
        if validation_error:
            return ToolResult("error", error=validation_error)
        try:
            outcome = tool.execute(arguments, context)
            return await outcome if inspect.isawaitable(outcome) else outcome
        except Exception as exc:
            return ToolResult("error", error=f"{type(exc).__name__}: {exc}", retryable=True)

    async def execute_many(self, calls: Sequence[tuple[str, Any]], context: ToolContext) -> list[ToolResult]:
        """Parallelize only explicitly-declared read-only calls, preserving call order."""
        indexed = list(enumerate(calls))
        results: list[Optional[ToolResult]] = [None] * len(indexed)
        readonly = [(index, call) for index, call in indexed if getattr(self._tools.get(call[0]), "is_read_only", False)]
        mutating = [(index, call) for index, call in indexed if not getattr(self._tools.get(call[0]), "is_read_only", False)]
        if readonly:
            completed = await asyncio.gather(*(self.execute(name, arguments, context) for _, (name, arguments) in readonly))
            for (index, _), result in zip(readonly, completed):
                results[index] = result
        for index, (name, arguments) in mutating:
            results[index] = await self.execute(name, arguments, context)
        return [result if result is not None else ToolResult("error", error="Tool result was not produced.") for result in results]


def _validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> Optional[str]:
    """Validate the object-schema subset used by registered tools.

    Tool schemas are intentionally small and closed.  This validator checks nested
    object/array/string/number constraints rather than silently accepting fields.
    """
    error = _validate_value(arguments, schema, "arguments")
    return error


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> Optional[str]:
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            return f"{path} must be an object."
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            return "Missing required argument(s): %s." % ", ".join(missing)
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unexpected = sorted(set(value) - set(properties))
            if unexpected:
                return "Unexpected argument(s): %s." % ", ".join(unexpected)
        for key, child in value.items():
            rule = properties.get(key)
            if rule:
                error = _validate_value(child, rule, key)
                if error:
                    return error
        return None
    if expected == "array":
        if not isinstance(value, list):
            return f"{path} must be an array."
        if "minItems" in schema and len(value) < int(schema["minItems"]):
            return f"{path} must contain at least {schema['minItems']} item(s)."
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                error = _validate_value(item, item_schema, f"{path}[{index}]")
                if error:
                    return error
        return None
    if expected == "string":
        if not isinstance(value, str):
            return f"{path} must be a string."
        if "minLength" in schema and len(value) < int(schema["minLength"]):
            return f"{path} is shorter than the minimum length."
        if "maxLength" in schema and len(value) > int(schema["maxLength"]):
            return f"{path} exceeds the maximum length."
        if "enum" in schema and value not in schema["enum"]:
            return f"{path} must be one of: {', '.join(map(str, schema['enum']))}."
        return None
    if expected == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return f"{path} must be an integer."
    elif expected == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"{path} must be a number."
    elif expected == "boolean" and not isinstance(value, bool):
        return f"{path} must be a boolean."
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return f"{path} is below the minimum value."
        if "maximum" in schema and value > schema["maximum"]:
            return f"{path} is above the maximum value."
    return None
