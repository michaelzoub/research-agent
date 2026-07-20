# Research Service Directory

This directory lists optional advanced and paid data services that research-mode
agents may use when normal local, web, or corpus retrieval is not enough.

These services are discretionary. Do not call them by default. Use them only
when the task benefits from live, premium, structured, or paywalled data and the
expected value is worth the cost.

## Selection Policy

Use an advanced service when at least one condition is true:

- The user explicitly asks for paid APIs, mpp.dev, AgentCash, Tempo, Orthogonal,
  Tako, Product Hunt, Hacker News launch data, funding data, live odds, or a
  specific premium source.
- The research question depends on fresh data that standard search may miss,
  such as live prediction-market odds, current startup launches, recent funding
  events, social/platform activity, onchain activity, or structured company and
  people enrichment.
- A chart, embeddable knowledge card, structured data table, or API-grade output
  is materially better than a prose web-search summary.
- The agent has already tried cheaper sources and an unresolved question remains.

Avoid these services when:

- The local corpus or ordinary web retrieval is sufficient.
- The task can be completed deterministically for tests or benchmarks.
- The result is not worth a paid request.
- The user has asked for offline, reproducible, or no-cost execution.

## Providers

| Provider | Use for | Setup state |
| --- | --- | --- |
| [mpp.dev / Tempo](mpp-dev.md) | MPP service discovery and paid requests, including Tako, market data, scraping, live odds, and other service marketplace endpoints. | Tempo CLI installed and wallet ready. |
| [AgentCash](agentcash.md) | x402/MPP APIs, Stable* services, enrichment, social data, search, file upload, email, and other pay-per-call APIs. | Onboarded; balance available. Invite code was already redeemed. |
| [Orthogonal](orthogonal.md) | Curated skills first, then API marketplace search and paid integrations. | CLI installed; requires `orth login` with an API key before use. |
| [Firecrawl](firecrawl.md) | Live web search and clean Markdown scraping through the runtime external-service registry. | Keyless fallback available; set `FIRECRAWL_API_KEY` for authenticated limits. |

## Notable Research Uses

- Live prediction-market odds: use Tempo service search first, especially
  `codex` or `nansen` when onchain or prediction-market data is relevant.
- Tako: use Tempo service `tako` for dataset search, charts, research reports,
  and embeddable knowledge-card style outputs.
- Recent startup launches: search AgentCash or Tempo for Product Hunt and
  Hacker News endpoints. Current discovered options include Scout Product Hunt
  and Hacker News endpoints, plus Product Hunt daily launches through x402
  helper services.
- Recent startup funding: search AgentCash for `startup funding` or
  `recent funding`; discovered options include recent funding/funding rounds
  endpoints through x402 helper services and Messari crypto funding data.

## Cost And Provenance Rules

- Check wallet readiness and balance before paid calls.
- Discover endpoint schemas before calling paid endpoints; do not guess paths or
  parameter names.
- Use dry runs when supported before expensive or ambiguous requests.
- Store paid-source outputs in the normal research artifacts with source IDs,
  request metadata, timestamps, prices when available, and citations.
- If a paid call fails because of balance, auth, or spending limits, report the
  blocker and continue with cheaper sources when possible.
