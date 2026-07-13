from __future__ import annotations

from .prediction_market.adapter import PredictionMarketGrader


_GRADERS = {"prediction_market": PredictionMarketGrader}


def get_optimization_grader(identifier: str):
    try:
        return _GRADERS[identifier]()
    except KeyError as exc:
        available = ", ".join(sorted(_GRADERS))
        raise ValueError(f"Unknown optimization grader {identifier!r}; available: {available}") from exc


def list_optimization_graders() -> tuple[str, ...]:
    return tuple(sorted(_GRADERS))


def optimization_grader_baselines(identifier: str):
    """Return upstream example strategies registered for an optimization grader."""
    return get_optimization_grader(identifier).registered_baselines()
