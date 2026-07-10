from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence


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
            "source_metadata": list(self.source_metadata),
            "error": self.error,
            "retryable": self.retryable,
        }


@dataclass
class ToolContext:
    workspace: Any
    store: Optional[Any] = None
    run_id: str = ""
    cancelled: bool = False


class BaseTool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult: ...


class ToolRegistry:
    """Only registered tools are exposed to and callable by an agent."""
    def __init__(self, tools: Sequence[BaseTool]):
        self._tools = {tool.name: tool for tool in tools}
        if len(self._tools) != len(tools):
            raise ValueError("tool names must be unique")

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {"name": tool.name, "description": tool.description, "input_schema": tool.input_schema}
            for tool in self._tools.values()
        ]

    def execute(self, name: str, arguments: Any, context: ToolContext) -> ToolResult:
        if context.cancelled:
            return ToolResult("cancelled", error="Tool execution was cancelled.", retryable=False)
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult("error", error="Unknown tool '%s'. Select only a registered tool." % name, retryable=False)
        if not isinstance(arguments, dict):
            return ToolResult("error", error="Tool arguments must be a JSON object.", retryable=False)
        validation_error = _validate_arguments(arguments, tool.input_schema)
        if validation_error:
            return ToolResult("error", error=validation_error, retryable=False)
        try:
            return tool.execute(arguments, context)
        except Exception as exc:  # Tools report recoverable environmental feedback to the model.
            return ToolResult("error", error="%s: %s" % (type(exc).__name__, exc), retryable=True)


def _validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> Optional[str]:
    required = schema.get("required", [])
    missing = [key for key in required if key not in arguments]
    if missing:
        return "Missing required argument(s): %s." % ", ".join(missing)
    properties = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        unexpected = sorted(set(arguments) - set(properties))
        if unexpected:
            return "Unexpected argument(s): %s." % ", ".join(unexpected)
    for key, value in arguments.items():
        rule = properties.get(key, {})
        expected = rule.get("type")
        if expected == "string" and not isinstance(value, str):
            return "%s must be a string." % key
        if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            return "%s must be an integer." % key
        if isinstance(value, str) and "minLength" in rule and len(value) < int(rule["minLength"]):
            return "%s is shorter than the minimum length." % key
        if isinstance(value, int) and "minimum" in rule and value < int(rule["minimum"]):
            return "%s is below the minimum value." % key
        if isinstance(value, int) and "maximum" in rule and value > int(rule["maximum"]):
            return "%s is above the maximum value." % key
    return None
