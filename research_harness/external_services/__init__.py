"""Optional external service integrations."""

from .base import ExternalServiceAdapter, ExternalServiceDescriptor
from .firecrawl import FirecrawlAdapter, FirecrawlClient
from .registry import ExternalServiceRegistry, default_external_service_registry

__all__ = [
    "ExternalServiceAdapter",
    "ExternalServiceDescriptor",
    "ExternalServiceRegistry",
    "FirecrawlAdapter",
    "FirecrawlClient",
    "default_external_service_registry",
]
