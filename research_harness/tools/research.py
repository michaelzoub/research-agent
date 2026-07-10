from __future__ import annotations

import re
import ipaddress
import urllib.parse
import urllib.error
import urllib.request
from dataclasses import asdict
from typing import Any

from .base import ToolContext, ToolResult


class SearchTool:
    """One explicit search backend, selected by the caller/model by name."""
    def __init__(self, backend: Any):
        self.backend = backend
        self.name = str(backend.tool_name)
        self.description = (
            "Search the %s source collection for evidence. Use when you need candidate sources; "
            "do not use to read a known local file or fetch a known URL." % self.name
        )
        self.input_schema = {
            "type": "object",
            "required": ["query"],
            "properties": {"query": {"type": "string", "minLength": 2}, "limit": {"type": "integer", "minimum": 1, "maximum": 10}},
            "additionalProperties": False,
        }

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        query = str(arguments["query"]).strip()
        if len(query) < 2:
            return ToolResult("error", error="query must contain at least two characters.", retryable=False)
        documents = self.backend.search(query, limit=min(10, max(1, int(arguments.get("limit", 4)))))
        sources = [self.backend.to_source(document, relevance) for document, relevance in documents]
        if context.store is not None:
            sources = [context.store.add_source(source) for source in sources]
        metadata = [asdict(source) for source in sources]
        return ToolResult(
            "ok",
            {"query": query, "result_count": len(metadata), "results": [{"title": row["title"], "url": row["url"], "summary": row["summary"]} for row in metadata]},
            source_metadata=metadata,
        )


class WebFetchTool:
    name = "fetch_web_page"
    description = (
        "Fetch a known public HTTP(S) page as plain text after a search result identifies it. "
        "Use to inspect a specific URL; do not use to discover sources or access private/local addresses."
    )
    input_schema = {
        "type": "object",
        "required": ["url"],
        "properties": {"url": {"type": "string", "description": "Public http(s) URL from a promising source; example: https://arxiv.org/abs/1706.03762"}},
        "additionalProperties": False,
    }

    def __init__(self, timeout_seconds: float = 15.0, max_characters: int = 20000):
        self.timeout_seconds = timeout_seconds
        self.max_characters = max_characters

    def execute(self, arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
        url = str(arguments["url"]).strip()
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return ToolResult("error", error="Only public http(s) URLs are supported.", retryable=False)
        hostname = parsed.hostname.lower()
        try:
            if ipaddress.ip_address(hostname).is_private or ipaddress.ip_address(hostname).is_loopback:
                return ToolResult("error", error="Private or loopback addresses are not permitted.", retryable=False)
        except ValueError:
            if hostname == "localhost" or hostname.endswith(".local"):
                return ToolResult("error", error="Local hostnames are not permitted.", retryable=False)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-harness/0.1.0"}), timeout=self.timeout_seconds) as response:
                content_type = str(response.headers.get("Content-Type") or "")
                if "text" not in content_type and "json" not in content_type and content_type:
                    return ToolResult("error", error="URL did not return a text-compatible content type.", retryable=False)
                text = response.read(self.max_characters + 1).decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            return ToolResult("error", error="HTTP %s while fetching URL." % exc.code, retryable=exc.code >= 500)
        except (urllib.error.URLError, TimeoutError) as exc:
            return ToolResult("error", error="Network error: %s" % exc, retryable=True)
        return ToolResult("ok", {"url": url, "content": text[: self.max_characters], "truncated": len(text) > self.max_characters})
