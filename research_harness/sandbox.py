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
    command: tuple[str, ...] = ()

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

    @property
    def daemon_available(self) -> bool:
        if not self.available:
            return False
        try:
            completed = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                check=False,
                text=True,
                capture_output=True,
                timeout=5,
            )
        except Exception:
            return False
        return completed.returncode == 0

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
        if not self.daemon_available:
            return SandboxExecutionResult(
                "",
                "docker daemon is not reachable; start Docker Desktop or set PREDICTION_MARKET_ALLOW_UNSANDBOXED_UPSTREAM=1 for an explicit host run",
                125,
            )
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
                "--sandbox",
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
                return SandboxExecutionResult(exc.stdout or "", exc.stderr or "docker execution timed out", 124, timed_out=True, command=tuple(command))
            except Exception as exc:
                return SandboxExecutionResult("", f"{type(exc).__name__}: {exc}", 1, command=tuple(command))
        return SandboxExecutionResult(completed.stdout, completed.stderr, completed.returncode, command=tuple(command))

    def execute_python(
        self,
        code: str,
        *,
        timeout_seconds: Optional[float] = None,
        workspace_path: Optional[Path] = None,
    ) -> SandboxExecutionResult:
        """Execute a short analysis script in the same network-isolated boundary."""
        if not self.available:
            return SandboxExecutionResult("", "docker executable not found on PATH", 127)
        if not self.daemon_available:
            return SandboxExecutionResult("", "docker daemon is not reachable", 125)
        with tempfile.TemporaryDirectory(prefix="research_harness_code_docker_") as directory:
            sandbox_dir = Path(directory)
            script = sandbox_dir / "analysis.py"
            script.write_text(code, encoding="utf-8")
            workspace = Path(workspace_path).resolve() if workspace_path is not None else sandbox_dir
            command = [
                "docker", "run", "--rm", "--network", self.network,
                "--cpus", os.environ.get("RESEARCH_HARNESS_DOCKER_CPUS", "2"),
                "--memory", os.environ.get("RESEARCH_HARNESS_DOCKER_MEMORY", "2g"),
                "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
                "-v", f"{sandbox_dir}:/analysis:ro",
                "-v", f"{workspace}:/workspace:ro", "-w", "/workspace", self.image,
                "python", "-I", "/analysis/analysis.py",
            ]
            try:
                completed = subprocess.run(
                    command, check=False, text=True, capture_output=True,
                    timeout=timeout_seconds or min(self.timeout_seconds, 60.0),
                )
            except subprocess.TimeoutExpired as exc:
                return SandboxExecutionResult(exc.stdout or "", exc.stderr or "analysis execution timed out", 124, timed_out=True, command=tuple(command))
            except Exception as exc:
                return SandboxExecutionResult("", "%s: %s" % (type(exc).__name__, exc), 1, command=tuple(command))
        return SandboxExecutionResult(completed.stdout, completed.stderr, completed.returncode, command=tuple(command))
