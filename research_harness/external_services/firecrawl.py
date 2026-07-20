"""Firecrawl v2 adapter for optional web search and document scraping."""
from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from typing import Any, Callable, Mapping

from ..schemas import Source
from ..tools.base import ToolContext, ToolResult
from ..tools.research import _public_url_error
from .base import ExternalServiceDescriptor


FIRECRAWL_ORIGIN = "https://api.firecrawl.dev"
FIRECRAWL_BASE_URL = f"{FIRECRAWL_ORIGIN}/v2"


class FirecrawlClient:
    """Small dependency-free client with credentials resolved at call time."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = FIRECRAWL_BASE_URL,
        timeout_seconds: float = 60.0,
        opener: Callable[..., Any] | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._opener = opener or urllib.request.urlopen

    @property
    def access_mode(self) -> str:
        key = self.api_key if self.api_key is not None else os.environ.get("FIRECRAWL_API_KEY", "")
        return "authenticated" if key else "keyless"

    def post(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if endpoint not in {"/search", "/scrape"}:
            raise ValueError(f"unsupported Firecrawl endpoint: {endpoint}")
        key = self.api_key if self.api_key is not None else os.environ.get("FIRECRAWL_API_KEY", "")
        headers = {"Content-Type": "application/json", "User-Agent": "research-harness/0.1.0"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(dict(payload)).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            response = self._opener(request, timeout=self.timeout_seconds)
            with response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            detail = _firecrawl_error_detail(raw)
            suffix = f": {detail}" if detail else ""
            raise FirecrawlRequestError(
                f"Firecrawl returned HTTP {exc.code}{suffix}",
                status_code=exc.code,
                retryable=exc.code == 429 or exc.code >= 500,
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise FirecrawlRequestError(f"Could not reach Firecrawl: {exc}", retryable=True) from exc
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FirecrawlRequestError("Firecrawl returned an invalid JSON response.", retryable=True) from exc
        if not isinstance(data, dict):
            raise FirecrawlRequestError("Firecrawl returned a non-object response.", retryable=True)
        if data.get("success") is False:
            raise FirecrawlRequestError(
                f"Firecrawl request failed: {str(data.get('error') or 'unknown error')[:500]}",
                retryable=False,
            )
        return data


class FirecrawlRequestError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class FirecrawlSearchTool:
    name = "firecrawl_search"
    is_read_only = True
    description = (
        "Search the live web through the optional Firecrawl service when ordinary registered retrieval is insufficient. "
        "Results are discovery leads and must be scraped or fetched before supporting factual claims."
    )
    input_schema = {
        "type": "object",
        "required": ["query"],
        "properties": {
            "query": {"type": "string", "minLength": 2, "maxLength": 500},
            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
            "include_domains": {"type": "array", "items": {"type": "string", "minLength": 1}},
            "exclude_domains": {"type": "array", "items": {"type": "string", "minLength": 1}},
        },
        "additionalProperties": False,
    }

    def __init__(self, client: FirecrawlClient):
        self.client = client

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        del context
        if arguments.get("include_domains") and arguments.get("exclude_domains"):
            return ToolResult("error", error="include_domains and exclude_domains are mutually exclusive.", executed=False)
        payload: dict[str, Any] = {
            "query": str(arguments["query"]).strip(),
            "limit": int(arguments.get("limit", 5)),
            "sources": ["web"],
            "ignoreInvalidURLs": True,
        }
        if arguments.get("include_domains"):
            payload["includeDomains"] = list(arguments["include_domains"])
        if arguments.get("exclude_domains"):
            payload["excludeDomains"] = list(arguments["exclude_domains"])
        try:
            response = await asyncio.to_thread(self.client.post, "/search", payload)
        except FirecrawlRequestError as exc:
            return ToolResult("error", error=str(exc), retryable=exc.retryable)
        web_results = response.get("data", {}).get("web", []) if isinstance(response.get("data"), dict) else []
        records = [row for row in web_results if isinstance(row, dict) and str(row.get("url") or "").startswith(("http://", "https://"))]
        provenance = _provenance("POST /v2/search", payload, response, self.client.access_mode)
        sources = [
            Source(
                url=str(row["url"]),
                title=str(row.get("title") or row["url"])[:300],
                author=urllib.parse.urlsplit(str(row["url"])).netloc,
                date="",
                source_type="firecrawl_search",
                summary=str(row.get("description") or "")[:800],
                relevance_score=0.70,
                credibility_score=0.60,
                evidence_sections=provenance,
                evidence_kind="lead",
            )
            for row in records
        ]
        return ToolResult(
            "ok",
            {
                "provider": "firecrawl",
                "query": payload["query"],
                "result_count": len(records),
                "results": [
                    {"title": source.title, "url": source.url, "summary": source.summary}
                    for source in sources
                ],
                "job_id": response.get("id"),
                "credits_used": response.get("creditsUsed"),
            },
            source_metadata=[asdict(source) for source in sources],
        )


class FirecrawlScrapeTool:
    name = "firecrawl_scrape"
    is_read_only = True
    description = (
        "Scrape one known public URL through the optional Firecrawl service and return clean Markdown. "
        "Use after discovery when direct document fetching is blocked or insufficient."
    )
    input_schema = {
        "type": "object",
        "required": ["url"],
        "properties": {
            "url": {"type": "string", "minLength": 8},
            "only_main_content": {"type": "boolean"},
            "max_age_ms": {"type": "integer", "minimum": 0},
        },
        "additionalProperties": False,
    }

    def __init__(self, client: FirecrawlClient, max_characters: int = 20_000):
        self.client = client
        self.max_characters = max_characters

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        del context
        url = str(arguments["url"]).strip()
        safety_error = await asyncio.to_thread(_public_url_error, url)
        if safety_error:
            return ToolResult("error", error=safety_error, executed=False)
        payload: dict[str, Any] = {
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": bool(arguments.get("only_main_content", True)),
        }
        if "max_age_ms" in arguments:
            payload["maxAge"] = int(arguments["max_age_ms"])
        try:
            response = await asyncio.to_thread(self.client.post, "/scrape", payload)
        except FirecrawlRequestError as exc:
            return ToolResult("error", error=str(exc), retryable=exc.retryable)
        data = response.get("data") if isinstance(response.get("data"), dict) else {}
        markdown = str(data.get("markdown") or "")
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        resolved_url = str(metadata.get("sourceURL") or metadata.get("url") or url)
        provenance = _provenance("POST /v2/scrape", payload, response, self.client.access_mode)
        source = Source(
            url=resolved_url,
            title=str(metadata.get("title") or f"Firecrawl scrape: {resolved_url}")[:300],
            author=urllib.parse.urlsplit(resolved_url).netloc,
            date=str(metadata.get("publishedTime") or ""),
            source_type="firecrawl_scrape",
            summary=markdown[:800],
            relevance_score=1.0,
            credibility_score=0.70,
            evidence_sections=provenance,
            evidence_kind="verified_document",
        )
        return ToolResult(
            "ok",
            {
                "provider": "firecrawl",
                "url": resolved_url,
                "content": markdown[: self.max_characters],
                "truncated": len(markdown) > self.max_characters,
                "metadata": metadata,
                "job_id": metadata.get("scrapeId") or response.get("id"),
                "credits_used": response.get("creditsUsed"),
            },
            source_metadata=[asdict(source)],
        )


class FirecrawlAdapter:
    descriptor = ExternalServiceDescriptor(
        name="firecrawl",
        origin=FIRECRAWL_ORIGIN,
        capabilities=("search", "scrape"),
        credential_environment_variable="FIRECRAWL_API_KEY",
        supports_keyless=True,
    )

    def __init__(self, client: FirecrawlClient | None = None):
        self.client = client or FirecrawlClient()

    def tools(self) -> list[FirecrawlSearchTool | FirecrawlScrapeTool]:
        return [FirecrawlSearchTool(self.client), FirecrawlScrapeTool(self.client)]


def _provenance(
    endpoint: str,
    payload: Mapping[str, Any],
    response: Mapping[str, Any],
    access_mode: str,
) -> dict[str, str]:
    request = {key: value for key, value in payload.items() if key.lower() not in {"authorization", "api_key"}}
    return {
        "provider": "firecrawl",
        "service_origin": FIRECRAWL_ORIGIN,
        "endpoint": endpoint,
        "request": json.dumps(request, sort_keys=True),
        "price": f"{response.get('creditsUsed')} credits" if response.get("creditsUsed") is not None else "not reported",
        "access_mode": access_mode,
    }


def _firecrawl_error_detail(raw: bytes) -> str:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw.decode("utf-8", errors="replace")[:500]
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("error") or payload.get("message") or "")[:500]
