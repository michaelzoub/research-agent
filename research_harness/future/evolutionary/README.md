# Archived evolutionary prototype

This directory contains the former population-based research and optimization
prototype. It is **not part of the active harness**: `autore`,
`Orchestrator`, and `ResearchAgent` do not import it.

It is retained because its candidate-evaluation, scoring, and artifact ideas
may be useful in a future, deliberately designed implementation. Do not add
production features or dependencies here. Any reintroduction must first split
the prototype into focused modules with an explicit production entry point,
bounded concurrency, and dedicated tests.
