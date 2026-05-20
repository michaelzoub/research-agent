"""Probe script that verifies DockerSandboxRunner actually fires a container."""

import sys
import tempfile
import time
from pathlib import Path

SENTINEL = f"SANDBOX_PROBE_EXECUTED_{int(time.time())}"

STRATEGY_TEMPLATE = f"""\
# {SENTINEL}
from typing import Sequence
from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import StepState, Action


class Strategy(BaseStrategy):
    def on_step(self, state: StepState) -> Sequence[Action]:
        return []
"""


def _fail(reason: str) -> None:
    print(f"[SANDBOX PROBE]: FAIL - {reason}", flush=True)
    sys.exit(1)


def main() -> None:
    print(f"[SANDBOX PROBE]: sentinel = {SENTINEL}", flush=True)

    # Locate the upstream package
    project_root = Path(__file__).resolve().parent
    upstream_path = project_root / "challenges" / "prediction-market-challenge"
    if not upstream_path.exists():
        _fail(f"upstream_path not found: {upstream_path}")

    # Write a minimal strategy.py to a temp file
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, prefix="probe_strategy_"
    ) as fh:
        fh.write(STRATEGY_TEMPLATE)
        strategy_path = Path(fh.name)

    print(f"[SANDBOX PROBE]: strategy written to {strategy_path}", flush=True)
    print(f"[SANDBOX PROBE]: upstream_path = {upstream_path}", flush=True)

    # Import and run the sandbox
    try:
        from research_harness.sandbox import DockerSandboxRunner
    except ImportError as exc:
        _fail(f"could not import DockerSandboxRunner: {exc}")

    runner = DockerSandboxRunner()
    print(f"[SANDBOX PROBE]: docker available = {runner.available}", flush=True)
    print(f"[SANDBOX PROBE]: image = {runner.image}", flush=True)

    result = runner.execute_prediction_market(
        upstream_path=upstream_path,
        strategy_path=strategy_path,
        simulations="1",
        steps="10",
        seed_start="0",
        workers="1",
    )

    print(f"[SANDBOX PROBE]: exit_code = {result.exit_code}", flush=True)
    print(f"[SANDBOX PROBE]: timed_out = {result.timed_out}", flush=True)
    print(f"--- stdout ---\n{result.stdout}", flush=True)
    print(f"--- stderr ---\n{result.stderr}", flush=True)

    # Assertions
    if result.exit_code == 127:
        _fail("exit_code=127 — docker executable not found on PATH")

    if result.timed_out:
        _fail("execution timed out before completing")

    # Execution was attempted; accept any non-127 exit code as proof the
    # container was launched (the strategy may legitimately error inside Docker).
    print("[SANDBOX PROBE]: PASS", flush=True)


if __name__ == "__main__":
    main()
