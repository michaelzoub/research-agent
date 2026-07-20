from __future__ import annotations

import asyncio
import io
import json
import urllib.error
import unittest
from pathlib import Path
from unittest import mock

from research_harness.external_services import (
    ExternalServiceRegistry,
    FirecrawlAdapter,
    FirecrawlClient,
    default_external_service_registry,
)
from research_harness.external_services.firecrawl import FirecrawlRequestError
from research_harness.research_agent import ResearchAgent
from research_harness.tools import ToolContext


class _Response:
    def __init__(self, payload: dict):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return self.payload


class ExternalServicesTest(unittest.TestCase):
    def test_default_registry_contains_firecrawl_tools(self) -> None:
        registry = default_external_service_registry()

        self.assertEqual(registry.get("FIRECRAWL").descriptor.capabilities, ("search", "scrape"))
        self.assertEqual([tool.name for tool in registry.tools()], ["firecrawl_search", "firecrawl_scrape"])

    def test_registry_rejects_duplicate_services(self) -> None:
        adapter = FirecrawlAdapter(FirecrawlClient(api_key=""))

        with self.assertRaisesRegex(ValueError, "already registered"):
            ExternalServiceRegistry([adapter, adapter])

    def test_client_uses_runtime_key_without_putting_it_in_payload(self) -> None:
        observed: dict = {}

        def open_request(request, *, timeout):
            observed["request"] = request
            observed["timeout"] = timeout
            return _Response({"success": True, "data": {"web": []}})

        client = FirecrawlClient(api_key="fc-test-secret", opener=open_request)
        client.post("/search", {"query": "agent research"})

        request = observed["request"]
        self.assertEqual(request.full_url, "https://api.firecrawl.dev/v2/search")
        self.assertEqual(request.get_header("Authorization"), "Bearer fc-test-secret")
        self.assertNotIn("fc-test-secret", request.data.decode("utf-8"))
        self.assertEqual(client.access_mode, "authenticated")

    def test_keyless_search_normalizes_leads_and_provenance(self) -> None:
        response = {
            "success": True,
            "id": "job-search-1",
            "creditsUsed": 2,
            "data": {"web": [{
                "url": "https://example.com/research",
                "title": "Research result",
                "description": "A useful discovery lead.",
            }]},
        }
        client = FirecrawlClient(api_key="", opener=lambda *_args, **_kwargs: _Response(response))
        tool = FirecrawlAdapter(client).tools()[0]

        result = asyncio.run(tool.execute(
            {"query": "research agents", "limit": 3},
            ToolContext(workspace=Path.cwd()),
        ))

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["job_id"], "job-search-1")
        self.assertEqual(result.source_metadata[0]["evidence_kind"], "lead")
        provenance = result.source_metadata[0]["evidence_sections"]
        self.assertEqual(provenance["endpoint"], "POST /v2/search")
        self.assertEqual(provenance["access_mode"], "keyless")
        self.assertEqual(provenance["price"], "2 credits")
        self.assertNotIn("Authorization", provenance["request"])

    def test_scrape_rejects_private_targets_without_calling_firecrawl(self) -> None:
        opener = mock.Mock()
        tool = FirecrawlAdapter(FirecrawlClient(api_key="", opener=opener)).tools()[1]

        result = asyncio.run(tool.execute(
            {"url": "http://127.0.0.1/private"},
            ToolContext(workspace=Path.cwd()),
        ))

        self.assertEqual(result.status, "error")
        self.assertFalse(result.executed)
        opener.assert_not_called()

    def test_scrape_returns_verified_markdown_source(self) -> None:
        response = {
            "success": True,
            "creditsUsed": 1,
            "data": {
                "markdown": "# Primary page\n\nVerified content.",
                "metadata": {
                    "title": "Primary page",
                    "sourceURL": "https://example.com/primary",
                    "scrapeId": "scrape-1",
                },
            },
        }
        client = FirecrawlClient(api_key="fc-test", opener=lambda *_args, **_kwargs: _Response(response))
        tool = FirecrawlAdapter(client).tools()[1]

        with mock.patch("research_harness.external_services.firecrawl._public_url_error", return_value=None):
            result = asyncio.run(tool.execute(
                {"url": "https://example.com/primary"},
                ToolContext(workspace=Path.cwd()),
            ))

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["content"], "# Primary page\n\nVerified content.")
        self.assertEqual(result.source_metadata[0]["evidence_kind"], "verified_document")
        self.assertEqual(result.source_metadata[0]["evidence_sections"]["endpoint"], "POST /v2/scrape")
        self.assertEqual(result.source_metadata[0]["evidence_sections"]["access_mode"], "authenticated")

    def test_http_rate_limit_is_retryable_and_preserves_remote_error(self) -> None:
        error = urllib.error.HTTPError(
            "https://api.firecrawl.dev/v2/search",
            429,
            "rate limited",
            {},
            io.BytesIO(b'{"error":"rate limit exceeded"}'),
        )
        client = FirecrawlClient(api_key="", opener=mock.Mock(side_effect=error))

        with self.assertRaises(FirecrawlRequestError) as raised:
            client.post("/search", {"query": "research"})

        self.assertTrue(raised.exception.retryable)
        self.assertIn("HTTP 429: rate limit exceeded", str(raised.exception))

    def test_research_agent_exposes_registered_external_service_tools(self) -> None:
        llm = mock.Mock()
        agent = ResearchAgent.with_research_tools(llm, [])
        names = [schema["name"] for schema in agent.tools.schemas()]

        self.assertIn("firecrawl_search", names)
        self.assertIn("firecrawl_scrape", names)


if __name__ == "__main__":
    unittest.main()
