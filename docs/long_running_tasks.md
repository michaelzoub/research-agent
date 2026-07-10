# Long-Running Tasks

The harness is designed to make the model the bottleneck, not the scheduler.
Long-running work should be expressed as repeated agent trials with durable
artifacts, not as a single opaque model call.

## Core Pattern

1. Route the goal to a product agent: `research`, `optimize`, or `challenge`.
2. Initialize `run_state.json` before execution so the run can record evidence,
   actions actually taken, budgets, and stopping rationale.
3. Run loop rounds until a score threshold, plateau, budget, or external stop
   condition fires.
4. Evaluate candidate trials in parallel whenever they are independent.
5. Persist every trial trajectory, score, and artifact under `outputs/<run>/`.
6. Promote the best candidate into stable output names such as
   `optimal_code.py`, `optimized_candidate.txt`, and `optimization_result.json`.

## Prediction-Market Challenge

Use the challenge agent with the optimization-query harness:

```bash
./autore "Get to $10 profit in the prediction market challenge, don't stop until you're profitable. Make sure to introduce entropy (AMM, PM, options, etc) literature if you start tweaking hyperparameters." --task-mode optimize_query --evaluator prediction_market --max-iterations 12
```

Generated strategies are never source files. They are written to:

```text
outputs/<run>/candidates/
outputs/<run>/optimal_code.py
outputs/<run>/solution.py
outputs/<run>/optimization_result.json
```

By default, the prediction-market loop uses the local challenge-semantics
fallback scorer so iteration remains fast and parallel. To run the upstream
grader, provide a local checkout and opt in explicitly:

```bash
PREDICTION_MARKET_USE_UPSTREAM=1 \
PREDICTION_MARKET_CHALLENGE_PATH=/private/tmp/prediction-market-challenge-src \
PREDICTION_MARKET_SIMULATIONS=200 \
PREDICTION_MARKET_STEPS=2000 \
PREDICTION_MARKET_WORKERS=4 \
./autore "Get to $10 profit in the prediction market challenge" --task-mode optimize_query --evaluator prediction_market --max-iterations 12
```

If upstream scoring is unavailable, `optimization_result.json` will show
`official_result.measured = false`. When upstream scoring runs successfully, it
will show `official_result.measured = true` and `score_source =
upstream_orderbook_pm_challenge`.

## Resume Shape

Full resume support should load a prior `outputs/<run>/` directory, reconstruct
parents from the best prior variants, and continue loop rounds into the same
artifact store. Until that is implemented, every run is durable but not resumable:
you can inspect prior candidates and pass the winning `optimal_code.py` back into
a new run as context, but the CLI does not yet continue a previous run in place.

## Parallelism Rule

Parallelize across independent candidates, simulations, and product agents. Do
not parallelize stateful mutation inside one candidate trajectory unless the
environment explicitly supports it. For prediction-market runs, this means:

- candidate strategies are evaluated concurrently by the harness;
- upstream simulations are parallelized by `PREDICTION_MARKET_WORKERS`;
- each candidate writes to a unique file under `outputs/<run>/candidates/`;
- the best candidate is selected only after all round results are collected.
