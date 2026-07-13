# Prediction-market optimization grader

- **Upstream authority:** https://github.com/danrobinson/prediction-market-challenge
- **Vendored source:** `challenges/prediction-market-challenge` (read-only)
- **Candidate contract:** a Python `Strategy(BaseStrategy)` implementation.
- **Registered baseline:** `starter_strategy` resolves to the upstream
  `examples/starter_strategy.py` file and demonstrates the accepted public API.
- **Official score:** mean realized `total_edge` from the upstream CLI's JSON
  simulation results on a fixed seed range.
- **Default execution:** Docker with no network, constrained CPU/memory, a
  read-only upstream mount, and an isolated candidate copy.

The local adapter only renders candidates, invokes the upstream CLI, and
normalizes its output. It does not duplicate the upstream simulator or scoring
logic. Every trial records candidate code, exact command, upstream identity,
stdout, stderr, metrics, and promotion eligibility.

Candidate source is parsed before execution. Dynamic builtins including
`getattr()`, `setattr()`, `delattr()`, `vars()`, `eval()`, `exec()`, and
`__import__()` are rejected as non-sandbox-compatible. The official command
also includes upstream `--sandbox`; a validation failure is an unmeasured,
non-promotable zero-score result.
