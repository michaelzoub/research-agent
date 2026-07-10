# Prediction Market Optimization Challenge

Upstream source of truth: https://github.com/danrobinson/prediction-market-challenge/tree/main

## Objective

Design a passive market-making strategy for a binary YES prediction-market contract.
The strategy should earn edge from uninformed retail order flow while avoiding
negative edge from an informed arbitrageur sweeping stale quotes.

## Market Mechanics

- Trade one YES share on an integer-tick FIFO limit order book.
- The contract settles to 1 if a latent score finishes above zero, otherwise 0.
- The strategy may only post resting limit orders and cancel existing orders.
- An informed arbitrageur knows true fair value and sweeps stale quotes.
- Retail flow is uninformed and is the main source of positive edge.
- A static competitor maintains a hidden liquidity ladder and does not re-anchor
  to fair value after jumps.

## Upstream Challenge Contract

The local adapter should model the upstream repository directly:

- Package: `orderbook_pm_challenge`.
- CLI: `uv run orderbook-pm run <strategy.py>`.
- Parallel scoring: `uv run orderbook-pm run <strategy.py> --workers 4`.
- Sandboxed scoring: `uv run orderbook-pm run <strategy.py> --sandbox`.
- Programmatic scorer: `orderbook_pm_challenge.runner.run_batch`.
- Public strategy API:

```python
from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState


class Strategy(BaseStrategy):
    def on_step(self, state: StepState):
        return [CancelAll(), PlaceOrder(side=Side.BUY, price_ticks=48, quantity=1.0)]
```

`StepState` includes `step`, `steps_remaining`, `yes_inventory`,
`no_inventory`, `cash`, `reserved_cash`, `free_cash`,
`competitor_best_bid_ticks`, `competitor_best_ask_ticks`,
`buy_filled_quantity`, `sell_filled_quantity`, and `own_orders`.

## Local Evaluator

The local harness evaluator is registered as `prediction_market`.
It scores strategy descriptions or generated code/strategy variants by mapping
them into a parameterized passive market maker, then running deterministic
regime simulations.

The evaluator rewards variants that mention or implement:

- Adaptive fair-value estimation from fills, order flow, and competitor midpoint.
- Frequent cancellation or quote widening after jumps, volatility, or stale-fill risk.
- Inventory skew and position limits.
- Moderate order sizes and offset/spread choices.
- Passive liquidity provision only.

The evaluator penalizes:

- Static ladders that never re-anchor.
- Large size without inventory controls.
- Tight quotes without jump/adverse-selection guards.
- Strategies that ignore arbitrageur sweeps or stale quote risk.

The proxy score is not the official score. Official measurement must run the
emitted `optimal_code.py` or `solution.py` against the upstream
`orderbook-pm`/`run_batch` path and record mean edge.

## Suggested Harness Command

```bash
./autore "Research approaches for the prediction market challenge: adaptive passive market making against stale quote arbitrage and retail flow" --retriever local
```
