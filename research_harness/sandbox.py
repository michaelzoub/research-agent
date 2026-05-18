from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SandboxExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False

    @property
    def returncode(self) -> int:
        return self.exit_code


class DockerSandboxRunner:
    """Run commands in an isolated local Docker container.

    This runner intentionally keeps secrets out of the container, mounts only
    the upstream evaluator and candidate directory, and disables networking by
    default. It is a small boundary object, not an agent framework dependency.
    """

    def __init__(
        self,
        *,
        image: Optional[str] = None,
        network: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> None:
        self.image = image or os.environ.get("RESEARCH_HARNESS_DOCKER_IMAGE", "ghcr.io/astral-sh/uv:python3.11-bookworm")
        self.network = network or os.environ.get("RESEARCH_HARNESS_DOCKER_NETWORK", "none")
        self.timeout_seconds = timeout_seconds or float(os.environ.get("PREDICTION_MARKET_TIMEOUT_SECONDS", "300"))

    @property
    def available(self) -> bool:
        return shutil.which("docker") is not None

    def execute_prediction_market(
        self,
        *,
        upstream_path: Path,
        strategy_path: Path,
        simulations: str,
        steps: str,
        seed_start: str,
        workers: str,
    ) -> SandboxExecutionResult:
        if not self.available:
            return SandboxExecutionResult("", "docker executable not found on PATH", 127)
        upstream_path = upstream_path.resolve()
        strategy_path = strategy_path.resolve()
        with tempfile.TemporaryDirectory(prefix="research_harness_pm_docker_") as directory:
            sandbox_dir = Path(directory)
            candidate_path = sandbox_dir / "strategy.py"
            candidate_path.write_bytes(strategy_path.read_bytes())
            command = [
                "docker",
                "run",
                "--rm",
                "--network",
                self.network,
                "--cpus",
                os.environ.get("RESEARCH_HARNESS_DOCKER_CPUS", "2"),
                "--memory",
                os.environ.get("RESEARCH_HARNESS_DOCKER_MEMORY", "2g"),
                "-e",
                "PYTHONPATH=/workspace/upstream",
                "-v",
                f"{upstream_path}:/workspace/upstream:ro",
                "-v",
                f"{sandbox_dir}:/workspace/candidate:rw",
                "-w",
                "/workspace/candidate",
                self.image,
                "python",
                "-m",
                "orderbook_pm_challenge.cli",
                "run",
                "/workspace/candidate/strategy.py",
                "--simulations",
                simulations,
                "--steps",
                steps,
                "--seed-start",
                seed_start,
                "--workers",
                workers,
                "--json",
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                return SandboxExecutionResult(exc.stdout or "", exc.stderr or "docker execution timed out", 124, timed_out=True)
            except Exception as exc:
                return SandboxExecutionResult("", f"{type(exc).__name__}: {exc}", 1)
        return SandboxExecutionResult(completed.stdout, completed.stderr, completed.returncode)
