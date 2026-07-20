"""Contracts for optional external research-service adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from ..tools.base import BaseTool


@dataclass(frozen=True)
class ExternalServiceDescriptor:
    """Stable, non-secret metadata exposed by an external service adapter."""

    name: str
    origin: str
    capabilities: tuple[str, ...]
    credential_environment_variable: str | None = None
    supports_keyless: bool = False


class ExternalServiceAdapter(Protocol):
    """Translate one external service into model-callable harness tools."""

    descriptor: ExternalServiceDescriptor

    def tools(self) -> Sequence[BaseTool]: ...

