from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from research_harness.sandbox import DockerSandboxRunner, SandboxExecutionResult


DEFAULT_SIMULATIONS = "24"
DEFAULT_SEED_START = "0"
EVAL_PROTOCOL = "paired_crn_fixed_seed_range"
UPSTREAM_URL = "https://github.com/danrobinson/prediction-market-challenge"


@dataclass(frozen=True)
class PredictionMarketPreflight:
    ok: bool
    reason: str
    upstream_path: Optional[str] = None
    execution_mode: str = "unavailable"
    docker_sandbox: bool = False


class PredictionMarketGrader:
    """Adapter that invokes—not reimplements—the upstream official scorer."""

    identifier = "prediction_market"
    upstream_url = UPSTREAM_URL

    def registered_baselines(self) -> dict[str, Path]:
        """Known-good upstream strategy sources, keyed for prompt/tool use."""
        upstream = self.find_upstream_path()
        if upstream is None:
            return {}
        starter = upstream / "examples" / "starter_strategy.py"
        return {"starter_strategy": starter} if starter.is_file() else {}

    @staticmethod
    def validate_candidate(code: str) -> Optional[str]:
        """Reject source the upstream sandbox cannot safely execute."""
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"Python syntax error: {exc.msg} (line {exc.lineno})."
        has_strategy = any(isinstance(node, ast.ClassDef) and node.name == "Strategy" for node in tree.body)
        if not has_strategy:
            return "Candidate must define class Strategy(BaseStrategy)."
        forbidden = {"getattr", "setattr", "delattr", "vars", "eval", "exec", "__import__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in forbidden:
                return f"Use of '{node.func.id}()' is not allowed by the prediction-market sandbox."
        return None

    def find_upstream_path(self) -> Optional[Path]:
        if os.environ.get("PREDICTION_MARKET_USE_UPSTREAM") == "0":
            return None
        root = Path(__file__).resolve().parents[2]
        candidates = [
            os.environ.get("PREDICTION_MARKET_CHALLENGE_PATH"),
            str(root / "challenges" / "prediction-market-challenge"),
            str(root / "challenges" / "prediction_market_challenge"),
            str(root / "vendor" / "prediction-market-challenge"),
            "/private/tmp/prediction-market-challenge-src",
            str(Path.home() / "prediction-market-challenge"),
            str(Path.home() / "src" / "prediction-market-challenge"),
            str(Path.cwd() / "prediction-market-challenge"),
        ]
        return next((Path(path) for path in candidates if path and self._is_upstream_repo(Path(path))), None)

    def preflight(self) -> PredictionMarketPreflight:
        upstream = self.find_upstream_path()
        if upstream is None:
            return PredictionMarketPreflight(False, "prediction-market upstream scorer not found; set PREDICTION_MARKET_CHALLENGE_PATH or keep the vendored challenge present")
        if os.environ.get("PREDICTION_MARKET_ALLOW_UNSANDBOXED_UPSTREAM") == "1":
            if shutil.which("uv") is None:
                return PredictionMarketPreflight(False, "PREDICTION_MARKET_ALLOW_UNSANDBOXED_UPSTREAM=1 but `uv` is not on PATH", str(upstream), "host", False)
            return PredictionMarketPreflight(True, "official upstream scorer available via host `uv run`", str(upstream), "host", False)
        sandbox = DockerSandboxRunner()
        if not sandbox.available:
            return PredictionMarketPreflight(False, "docker executable not found; start Docker or set PREDICTION_MARKET_ALLOW_UNSANDBOXED_UPSTREAM=1", str(upstream), "docker", True)
        if not sandbox.daemon_available:
            return PredictionMarketPreflight(False, "docker daemon is not reachable; start Docker Desktop or set PREDICTION_MARKET_ALLOW_UNSANDBOXED_UPSTREAM=1", str(upstream), "docker", True)
        return PredictionMarketPreflight(True, "official upstream scorer available through Docker sandbox", str(upstream), "docker", True)

    def evaluate(self, candidate_path: Path, *, simulations: Optional[str] = None, seed_start: Optional[str] = None) -> dict[str, Any]:
        upstream = self.find_upstream_path()
        candidate_code = candidate_path.read_text(encoding="utf-8")
        invalid = self.validate_candidate(candidate_code)
        if invalid:
            return self._unmeasured("candidate_contract_invalid", invalid, candidate_code=candidate_code)
        if upstream is None:
            return self._unmeasured("upstream_repo_missing", "Upstream repo not found. No fallback score was used for optimization.", candidate_code=candidate_code)
        simulations = simulations or os.environ.get("PREDICTION_MARKET_SIMULATIONS", DEFAULT_SIMULATIONS)
        steps = os.environ.get("PREDICTION_MARKET_STEPS", "600")
        seed_start = seed_start or os.environ.get("PREDICTION_MARKET_SEED_START", DEFAULT_SEED_START)
        workers = os.environ.get("PREDICTION_MARKET_WORKERS", "4")
        if os.environ.get("PREDICTION_MARKET_ALLOW_UNSANDBOXED_UPSTREAM") == "1":
            completed = self._run_on_host(upstream, candidate_path, simulations, steps, seed_start, workers)
            docker_sandbox = False
        else:
            completed = DockerSandboxRunner().execute_prediction_market(
                upstream_path=upstream, strategy_path=candidate_path, simulations=simulations,
                steps=steps, seed_start=seed_start, workers=workers,
            )
            docker_sandbox = True
        provenance = {
            "grader_id": self.identifier,
            "upstream_url": self.upstream_url,
            "upstream_path": str(upstream.resolve()),
            "upstream_revision": self._git_revision(upstream),
            "execution_mode": "docker" if docker_sandbox else "host",
            "command": list(completed.command),
        }
        base = {"candidate_code": candidate_code, "stdout": completed.stdout, "stderr": completed.stderr, "upstream": provenance}
        if completed.returncode != 0:
            return self._unmeasured("official_sandbox_failed", (completed.stderr or completed.stdout or "official scorer failed").strip()[:2000], docker_sandbox=docker_sandbox, exit_code=completed.returncode, timed_out=completed.timed_out, **base)
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            return self._unmeasured("official_scorer_json_error", f"JSONDecodeError: {exc}; stdout={completed.stdout[:500]}", docker_sandbox=docker_sandbox, **base)
        results = payload.get("simulation_results", [])
        successes = [result for result in results if not result.get("failed")]
        if not results or not successes:
            return self._unmeasured("official_scorer_no_successes", f"official scorer returned {len(results)} result(s), {len(successes)} successful.", docker_sandbox=docker_sandbox, **base)
        mean = lambda key: round(sum(float(row.get(key, 0.0)) for row in successes) / len(successes), 6)
        return {
            **base,
            "official_measured": True, "score_eligible": True, "sandbox_executed": True,
            "docker_sandbox": docker_sandbox, "paired_crn": True, "eval_protocol": EVAL_PROTOCOL,
            "seed_start": int(seed_start) if seed_start.isdigit() else seed_start,
            "steps": int(steps) if steps.isdigit() else steps,
            "mean_edge": mean("total_edge"), "mean_arb_edge": mean("arb_edge"), "mean_retail_edge": mean("retail_edge"),
            "success_count": len(successes), "failure_count": len(results) - len(successes),
            "simulations": len(results), "score_source": "upstream_orderbook_pm_challenge", "exit_code": completed.returncode,
            "timed_out": completed.timed_out,
        }

    @staticmethod
    def _is_upstream_repo(path: Path) -> bool:
        pyproject, package = path / "pyproject.toml", path / "orderbook_pm_challenge"
        return path.is_dir() and pyproject.exists() and package.is_dir() and 'orderbook-pm = "orderbook_pm_challenge.cli:main"' in pyproject.read_text(encoding="utf-8")

    @staticmethod
    def _git_revision(path: Path) -> Optional[str]:
        result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=False, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None

    @staticmethod
    def _run_on_host(upstream: Path, candidate: Path, simulations: str, steps: str, seed_start: str, workers: str) -> SandboxExecutionResult:
        command = ("uv", "run", "--project", str(upstream), "orderbook-pm", "run", str(candidate), "--simulations", simulations, "--steps", steps, "--seed-start", seed_start, "--workers", workers, "--sandbox", "--json")
        env = dict(os.environ)
        env.setdefault("UV_CACHE_DIR", "/private/tmp/research-harness-uv-cache")
        try:
            completed = subprocess.run(command, check=False, text=True, capture_output=True, env=env, timeout=float(os.environ.get("PREDICTION_MARKET_TIMEOUT_SECONDS", "300")))
            return SandboxExecutionResult(completed.stdout, completed.stderr, completed.returncode, command=command)
        except subprocess.TimeoutExpired as exc:
            return SandboxExecutionResult(exc.stdout or "", exc.stderr or "host upstream execution timed out", 124, timed_out=True, command=command)
        except Exception as exc:
            return SandboxExecutionResult("", f"{type(exc).__name__}: {exc}", 1, command=command)

    @staticmethod
    def _unmeasured(score_source: str, error: str, *, candidate_code: str = "", docker_sandbox: bool = False, exit_code: Optional[int] = None, timed_out: bool = False, stdout: str = "", stderr: str = "", upstream: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        return {
            "official_measured": False, "score_eligible": False, "sandbox_executed": False,
            "docker_sandbox": docker_sandbox, "paired_crn": False, "eval_protocol": "unmeasured",
            "mean_edge": 0.0, "mean_arb_edge": 0.0, "mean_retail_edge": 0.0,
            "success_count": 0, "failure_count": 0, "simulations": 0, "score_source": score_source,
            "error": error, "exit_code": exit_code, "timed_out": timed_out, "candidate_code": candidate_code,
            "stdout": stdout, "stderr": stderr, "upstream": upstream or {"grader_id": "prediction_market", "upstream_url": UPSTREAM_URL, "command": []},
        }
