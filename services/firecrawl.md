# Firecrawl

Use Firecrawl when ordinary registered retrieval cannot reliably discover or
extract a public web page. The runtime adapter exposes `firecrawl_search` for
discovery and `firecrawl_scrape` for verified document extraction.

## Setup

Upstream onboarding skill:
`https://www.firecrawl.dev/agent-onboarding/SKILL.md`

The harness uses Firecrawl's v2 REST API directly, so it does not require a CLI
or SDK install. Set the optional credential in the process environment:

```bash
export FIRECRAWL_API_KEY=fc-...
```

Without a key, search and scrape use Firecrawl's rate-limited keyless fallback.
Do not store the key in source control or pass it in model tool arguments.

## Registered Tools

- `firecrawl_search`: `POST https://api.firecrawl.dev/v2/search`. Returns leads;
  scrape or fetch a selected result before citing it.
- `firecrawl_scrape`: `POST https://api.firecrawl.dev/v2/scrape`. Returns clean
  Markdown and persists the result as a verified document source.

The adapter records the provider, service origin, endpoint, sanitized request,
access mode, retrieval timestamp, and reported credits with source artifacts.

## Selection Rules

- Prefer ordinary retrieval and direct public-document fetching when sufficient.
- Use Firecrawl when a page needs stronger extraction or ordinary discovery is
  blocked or incomplete.
- Search before scrape when no URL is known.
- Never send private, loopback, link-local, or reserved URLs to the service.
- Treat search results as leads and scraped documents as claim-eligible evidence.
- On `401`, configure `FIRECRAWL_API_KEY`; on `429`, stop or retry later rather
  than repeatedly consuming the external-tool budget.

