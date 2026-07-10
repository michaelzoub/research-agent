from __future__ import annotations

import asyncio
import ipaddress
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from typing import Any

from .base import ToolContext, ToolResult


class SearchTool:
    """Read-only evidence discovery from one registered backend."""
    is_read_only = True

    def __init__(self, backend: Any):
        self.backend = backend
        self.name = str(backend.tool_name)
        self.description = f"Search {self.name} for candidate evidence. Use to discover sources, not to read a known file or URL."
        self.input_schema = {"type": "object", "required": ["query"], "properties": {"query": {"type": "string", "minLength": 2}, "limit": {"type": "integer", "minimum": 1, "maximum": 10}}, "additionalProperties": False}

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        query = str(arguments["query"]).strip()
        documents = await asyncio.to_thread(self.backend.search, query, limit=min(10, max(1, int(arguments.get("limit", 4)))))
        sources = [self.backend.to_source(document, relevance) for document, relevance in documents]
        if context.store is not None:
            sources = [context.store.add_source(source) for source in sources]
        metadata = [asdict(source) for source in sources]
        return ToolResult("ok", {"query": query, "result_count": len(metadata), "results": [{"title": row["title"], "url": row["url"], "summary": row["summary"]} for row in metadata]}, source_metadata=metadata)


class WebFetchTool:
    """Read-only public document fetcher with DNS and redirect SSRF protection."""
    name = "fetch_document"
    is_read_only = True
    description = "Fetch a known public HTTP(S) document after discovery. Rejects private, local, and unsafe redirect destinations."
    input_schema = {"type": "object", "required": ["url"], "properties": {"url": {"type": "string", "minLength": 8}, "prefer_markdown": {"type": "boolean"}}, "additionalProperties": False}

    def __init__(self, timeout_seconds: float = 15.0, max_characters: int = 20000, max_redirects: int = 5):
        self.timeout_seconds, self.max_characters, self.max_redirects = timeout_seconds, max_characters, max_redirects

    async def execute(self, arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
        return await asyncio.to_thread(self._fetch, str(arguments["url"]).strip(), bool(arguments.get("prefer_markdown", True)))

    def _fetch(self, url: str, prefer_markdown: bool) -> ToolResult:
        for _ in range(self.max_redirects + 1):
            error = _public_url_error(url)
            if error:
                return ToolResult("error", error=error)
            opener = urllib.request.build_opener(_NoRedirect())
            try:
                response = opener.open(urllib.request.Request(url, headers={"User-Agent": "research-harness/0.2.0"}), timeout=self.timeout_seconds)
            except urllib.error.HTTPError as exc:
                if exc.code in {301, 302, 303, 307, 308}:
                    target = exc.headers.get("Location")
                    if not target:
                        return ToolResult("error", error="Redirect response did not include a Location header.")
                    url = urllib.parse.urljoin(url, target)
                    continue
                return ToolResult("error", error=f"HTTP {exc.code} while fetching URL.", retryable=exc.code >= 500)
            except (urllib.error.URLError, TimeoutError) as exc:
                return ToolResult("error", error=f"Network error: {exc}", retryable=True)
            with response:
                content_type = str(response.headers.get("Content-Type") or "")
                if content_type and not any(kind in content_type.lower() for kind in ("text", "json", "pdf")):
                    return ToolResult("error", error="URL did not return a supported document content type.")
                raw = response.read(self.max_characters + 1)
            text = raw.decode("utf-8", errors="replace")
            if prefer_markdown and "html" in content_type.lower():
                rendered = self._fetch_curl_markdown(url)
                if rendered is not None:
                    return ToolResult("ok", {"url": url, "content_type": "text/markdown", "content": rendered[: self.max_characters], "truncated": len(rendered) > self.max_characters, "renderer": "curl.md"})
            return ToolResult("ok", {"url": url, "content_type": content_type, "content": text[: self.max_characters], "truncated": len(raw) > self.max_characters, "renderer": "direct"})
        return ToolResult("error", error="Too many redirects.")

    def _fetch_curl_markdown(self, url: str) -> str | None:
        """Optional curl.md renderer for compact agent-readable public HTML."""
        try:
            request = urllib.request.Request(f"https://curl.md/{url}", headers={"User-Agent": "research-harness/0.2.0"})
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                if response.status != 200:
                    return None
                return response.read(self.max_characters + 1).decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _public_url_error(url: str) -> str | None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return "Only public http(s) URLs are supported."
    hostname = parsed.hostname.lower()
    if hostname == "localhost" or hostname.endswith(".local"):
        return "Private or loopback addresses are not permitted."
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(hostname, parsed.port or 443, type=socket.SOCK_STREAM)}
    except socket.gaierror:
        return "URL hostname could not be resolved."
    for address in addresses:
        candidate = ipaddress.ip_address(address)
        if not candidate.is_global:
            return "Private or loopback addresses are not permitted."
    return None
