"""Registry for optional external research-service adapters."""
from __future__ import annotations

from typing import Iterable

from ..tools.base import BaseTool
from .base import ExternalServiceAdapter, ExternalServiceDescriptor


class ExternalServiceRegistry:
    """Own the configured service adapters and flatten their tool surfaces."""

    def __init__(self, adapters: Iterable[ExternalServiceAdapter] = ()):
        self._adapters: dict[str, ExternalServiceAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: ExternalServiceAdapter) -> None:
        name = adapter.descriptor.name.strip().lower()
        if not name:
            raise ValueError("external service names must not be empty")
        if name in self._adapters:
            raise ValueError(f"external service '{name}' is already registered")
        self._adapters[name] = adapter

    def get(self, name: str) -> ExternalServiceAdapter:
        try:
            return self._adapters[name.strip().lower()]
        except KeyError as exc:
            raise KeyError(f"unknown external service '{name}'") from exc

    def descriptors(self) -> list[ExternalServiceDescriptor]:
        return [adapter.descriptor for adapter in self._adapters.values()]

    def tools(self) -> list[BaseTool]:
        tools = [tool for adapter in self._adapters.values() for tool in adapter.tools()]
        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise ValueError("external service tool names must be unique")
        return tools


def default_external_service_registry() -> ExternalServiceRegistry:
    """Return the first-party service directory used by research runs."""
    from .firecrawl import FirecrawlAdapter

    return ExternalServiceRegistry([FirecrawlAdapter()])

