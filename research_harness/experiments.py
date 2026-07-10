"""Durable, domain-neutral experiment coordination primitives.

The module deliberately treats strategies, evaluator options, and metrics as
opaque domain data.  Importing projects supply the adapters that interpret
them; this layer owns lifecycle, isolation, deduplication, and persistence.
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence
from uuid import uuid4


ExperimentStatus = str
PromotionStatus = str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id(prefix: str) -> str:
    return "%s_%s" % (prefix, uuid4().hex[:12])


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def _canonical(value: Any) -> str:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), default=str)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    direction: str = "maximize"
    weight: Optional[float] = None
    target: Optional[float] = None
    constraint: Optional[Any] = None


@dataclass(frozen=True)
class EvaluationProtocol:
    identifier: str
    stage: str = "screening"
    episodes: Optional[Any] = None
    seeds: Optional[Sequence[Any]] = None
    datasets: Optional[Any] = None
    scenarios: Optional[Any] = None
    repetitions: Optional[int] = None
    timeout_seconds: Optional[float] = None
    concurrency: Optional[int] = None
    objective_definition: Any = None
    constraints: Any = None
    evaluator_configuration: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRun:
    identifier: str
    metrics: dict[str, Any]
    constraints: dict[str, Any] = field(default_factory=dict)
    episode_identifier: Optional[str] = None
    seed: Optional[Any] = None
    scenario_identifier: Optional[str] = None
    duration_seconds: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationResult:
    evaluator_identifier: str
    evaluator_version: str
    protocol_identifier: str
    individual_runs: Sequence[EvaluationRun]
    aggregate_metrics: dict[str, Any]
    artifacts: Sequence[str] = field(default_factory=list)
    raw_output_reference: Optional[str] = None
    reproducible: bool = False
    warnings: Sequence[str] = field(default_factory=list)


@dataclass
class ExperimentSpecification:
    role: str
    hypothesis: str
    rationale: str
    strategy_domain: str
    strategy_adapter_identifier: str
    evaluator_identifier: str
    evaluator_version: str
    base_strategy_reference: str
    base_strategy_hash: str
    evaluation_protocol: EvaluationProtocol
    promotion_policy: Any
    allowed_changes: Any = None
    frozen_components: Any = None
    parameter_space: Any = None
    reserved_region: Any = None
    baseline_metrics: Any = None
    parent_experiment_id: Optional[str] = None
    replication_target_id: Optional[str] = None
    id: str = field(default_factory=lambda: _id("experiment"))
    status: ExperimentStatus = "proposed"
    assigned_worker: Optional[str] = None
    lease_expiration: Optional[str] = None
    created_at: str = field(default_factory=_now)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class ExperimentResult:
    experiment_id: str
    exact_configuration: Any
    exact_diff: Any
    changed_components: Sequence[str]
    operations: Sequence[str]
    evaluator_identifier: str
    evaluator_version: str
    evaluation_protocol_identifier: str
    random_state_records: dict[str, Any]
    evaluation_result: EvaluationResult
    hypothesis_supported: Optional[bool]
    target_satisfied: bool
    conclusion: str
    artifacts: Sequence[str]
    reproducible: bool
    candidate_strategy_reference: Optional[str] = None
    candidate_strategy_hash: Optional[str] = None
    failure_mode: Optional[str] = None
    follow_up_hypotheses: Sequence[str] = field(default_factory=list)


@dataclass(frozen=True)
class PromotionDecision:
    decision: PromotionStatus
    reasons: Sequence[str]
    passed_constraints: Sequence[str] = field(default_factory=list)
    failed_constraints: Sequence[str] = field(default_factory=list)


class StrategyAdapter(Protocol):
    identifier: str
    def load(self, reference: str) -> Any: ...
    def clone(self, strategy: Any, isolation_context: Any) -> Any: ...
    def read_configuration(self, strategy: Any) -> Any: ...
    def apply_configuration(self, strategy: Any, changes: Any) -> Any: ...
    def apply_patch(self, strategy: Any, structural_patch: Any) -> Any: ...
    def calculate_diff(self, base: Any, candidate: Any) -> Any: ...
    def calculate_fingerprint(self, strategy: Any) -> str: ...
    def save_immutable_candidate(self, strategy: Any, metadata: dict[str, Any]) -> str: ...


class EvaluatorAdapter(Protocol):
    identifier: str
    version: str
    def validate(self, protocol: EvaluationProtocol) -> Any: ...
    def evaluate(self, strategy: Any, protocol: EvaluationProtocol, context: Any = None) -> EvaluationResult: ...
    def compare(self, baseline: Any, candidate: Any, protocol: EvaluationProtocol, context: Any = None) -> Any: ...
    def randomness_policy(self) -> Any: ...


class SearchSpaceAdapter(Protocol):
    def normalize(self, configuration: Any) -> Any: ...
    def generate_configurations(self, parameter_space: Any) -> Iterable[Any]: ...
    def sample_configuration(self, parameter_space: Any, random_source: "RandomSource") -> Any: ...
    def calculate_distance(self, first: Any, second: Any) -> float: ...
    def overlaps(self, first_region: Any, second_region: Any) -> bool: ...
    def assign_bucket(self, configuration: Any, tolerance_policy: Any) -> Any: ...


class ObjectiveAdapter(Protocol):
    def aggregate(self, runs: Sequence[EvaluationRun], definition: Any) -> dict[str, Any]: ...
    def compare(self, baseline: dict[str, Any], candidate: dict[str, Any], policy: Any) -> Any: ...
    def rank(self, results: Sequence[ExperimentResult]) -> Sequence[ExperimentResult]: ...


class PromotionPolicy(Protocol):
    def assess(self, baseline_evaluation: Any, candidate_evaluation: Any, independent_validation: Any,
               adversarial_findings: Any, required_constraints: Any) -> PromotionDecision: ...


class RandomSource(Protocol):
    def next_float(self) -> float: ...
    def next_integer(self, minimum: int, maximum: int) -> int: ...
    def sample(self, values: Sequence[Any]) -> Any: ...
    def weighted_sample(self, values: Sequence[Any], probabilities: Sequence[float]) -> Any: ...
    def fork(self, namespace: str) -> "RandomSource": ...
    def snapshot_state(self) -> dict[str, Any]: ...


class SeededRandomSource:
    """Independent replayable streams, derived without shared mutable state."""
    def __init__(self, seed: Optional[int] = None, namespace: str = "root"):
        self.seed = seed if seed is not None else random.SystemRandom().randrange(2**63)
        self.namespace = namespace
        self._random = random.Random(self.seed)

    def next_float(self) -> float:
        return self._random.random()

    def next_integer(self, minimum: int, maximum: int) -> int:
        return self._random.randint(minimum, maximum)

    def sample(self, values: Sequence[Any]) -> Any:
        return self._random.choice(list(values))

    def weighted_sample(self, values: Sequence[Any], probabilities: Sequence[float]) -> Any:
        return self._random.choices(list(values), weights=list(probabilities), k=1)[0]

    def fork(self, namespace: str) -> "SeededRandomSource":
        seed = int(_hash({"parent": self.seed, "namespace": self.namespace, "child": namespace})[:16], 16)
        return SeededRandomSource(seed, "%s/%s" % (self.namespace, namespace))

    def snapshot_state(self) -> dict[str, Any]:
        return {"seed": self.seed, "namespace": self.namespace, "state": repr(self._random.getstate())}


class DurableExperimentStore:
    """SQLite persistence with atomic claims and promotion transactions."""
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS durable_experiments (
                  id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL UNIQUE, status TEXT NOT NULL,
                  worker_id TEXT, lease_expires_at TEXT, payload_json TEXT NOT NULL,
                  created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS durable_results (
                  experiment_id TEXT PRIMARY KEY, payload_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiment_reservations (
                  bucket TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, region_json TEXT NOT NULL,
                  active INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS canonical_strategies (
                  domain TEXT PRIMARY KEY, strategy_reference TEXT NOT NULL, strategy_hash TEXT NOT NULL,
                  experiment_id TEXT NOT NULL, promoted_at TEXT NOT NULL
                );
            """)

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    def create(self, specification: ExperimentSpecification, fingerprint: str) -> bool:
        payload = _canonical(specification)
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO durable_experiments VALUES (?, ?, ?, NULL, NULL, ?, ?, ?)",
                    (specification.id, fingerprint, "proposed", payload, specification.created_at, _now()),
                )
                connection.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def reserve(self, experiment_id: str, bucket: Optional[Any] = None, region: Any = None) -> bool:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT status FROM durable_experiments WHERE id = ?", (experiment_id,)).fetchone()
            if row is None or row["status"] != "proposed":
                connection.rollback()
                return False
            if bucket is not None:
                try:
                    connection.execute("INSERT INTO experiment_reservations VALUES (?, ?, ?, 1)", (_canonical(bucket), experiment_id, _canonical(region)))
                except sqlite3.IntegrityError:
                    connection.rollback()
                    return False
            connection.execute("UPDATE durable_experiments SET status = 'reserved', updated_at = ? WHERE id = ?", (_now(), experiment_id))
            connection.commit()
            return True

    def claim_next(self, worker_id: str, lease_seconds: int = 300) -> Optional[ExperimentSpecification]:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT * FROM durable_experiments WHERE status = 'reserved' ORDER BY created_at LIMIT 1").fetchone()
            if row is None:
                connection.rollback()
                return None
            expiry = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
            changed = connection.execute(
                "UPDATE durable_experiments SET status='running', worker_id=?, lease_expires_at=?, updated_at=? WHERE id=? AND status='reserved'",
                (worker_id, expiry, _now(), row["id"]),
            ).rowcount
            if changed != 1:
                connection.rollback()
                return None
            connection.commit()
            payload = json.loads(row["payload_json"])
            payload.update({"status": "running", "assigned_worker": worker_id, "lease_expiration": expiry, "started_at": _now()})
            return _specification_from_dict(payload)

    def receive_result(self, result: ExperimentResult) -> bool:
        _validate_result(result)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT status FROM durable_experiments WHERE id=?", (result.experiment_id,)).fetchone()
            if row is None or row["status"] != "running":
                connection.rollback()
                return False
            connection.execute("INSERT INTO durable_results VALUES (?, ?, ?)", (result.experiment_id, _canonical(result), _now()))
            next_status = "pending_validation" if result.candidate_strategy_reference else "completed"
            connection.execute("UPDATE durable_experiments SET status=?, worker_id=NULL, lease_expires_at=NULL, updated_at=? WHERE id=?", (next_status, _now(), result.experiment_id))
            connection.execute("UPDATE experiment_reservations SET active=0 WHERE experiment_id=?", (result.experiment_id,))
            connection.commit()
            return True

    def result(self, experiment_id: str) -> Optional[ExperimentResult]:
        with self._connection() as connection:
            row = connection.execute("SELECT payload_json FROM durable_results WHERE experiment_id=?", (experiment_id,)).fetchone()
        return _result_from_dict(json.loads(row["payload_json"])) if row else None

    def status(self, experiment_id: str) -> Optional[str]:
        with self._connection() as connection:
            row = connection.execute("SELECT status FROM durable_experiments WHERE id=?", (experiment_id,)).fetchone()
        return str(row["status"]) if row else None

    def experiments_with_status(self, status: str) -> list[ExperimentSpecification]:
        with self._connection() as connection:
            rows = connection.execute("SELECT payload_json FROM durable_experiments WHERE status=? ORDER BY created_at", (status,)).fetchall()
        return [_specification_from_dict(json.loads(row["payload_json"])) for row in rows]

    def promote(self, domain: str, result: ExperimentResult) -> bool:
        if not result.candidate_strategy_reference or not result.candidate_strategy_hash:
            raise ValueError("promotion requires an immutable candidate reference and hash")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT status FROM durable_experiments WHERE id=?", (result.experiment_id,)).fetchone()
            if row is None or row["status"] not in {"pending_validation", "validating"}:
                connection.rollback()
                return False
            connection.execute("INSERT OR REPLACE INTO canonical_strategies VALUES (?, ?, ?, ?, ?)", (domain, result.candidate_strategy_reference, result.candidate_strategy_hash, result.experiment_id, _now()))
            connection.execute("UPDATE durable_experiments SET status='promoted', updated_at=? WHERE id=?", (_now(), result.experiment_id))
            connection.commit()
            return True

    def write_knowledge_view(self, path: Path) -> Path:
        with self._connection() as connection:
            rows = connection.execute("SELECT status, COUNT(*) AS count FROM durable_experiments GROUP BY status ORDER BY status").fetchall()
        lines = ["# Experiment knowledge", "", "## Lifecycle counts", ""]
        lines.extend("- %s: %s" % (row["status"], row["count"]) for row in rows)
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        return Path(path)


class ExperimentCoordinator:
    def __init__(self, store: DurableExperimentStore, search_space: SearchSpaceAdapter, promotion_policy: PromotionPolicy):
        self.store = store
        self.search_space = search_space
        self.promotion_policy = promotion_policy

    def fingerprint(self, specification: ExperimentSpecification) -> str:
        return _hash({
            "role": specification.role, "strategy_adapter": specification.strategy_adapter_identifier,
            "evaluator": specification.evaluator_identifier, "evaluator_version": specification.evaluator_version,
            "base_hash": specification.base_strategy_hash, "hypothesis": specification.hypothesis,
            "parameters": self.search_space.normalize(specification.parameter_space),
            "frozen": specification.frozen_components, "protocol": specification.evaluation_protocol,
        })

    def create_distinct_assignments(self, specifications: Iterable[ExperimentSpecification], tolerance_policy: Any = None) -> list[ExperimentSpecification]:
        accepted: list[ExperimentSpecification] = []
        for specification in specifications:
            bucket = self.search_space.assign_bucket(specification.parameter_space, tolerance_policy) if specification.parameter_space is not None else None
            if self.store.create(specification, self.fingerprint(specification)) and self.store.reserve(specification.id, bucket, specification.reserved_region):
                accepted.append(specification)
        return accepted

    def claim_next_experiment(self, worker_id: str, lease_seconds: int = 300) -> Optional[ExperimentSpecification]:
        return self.store.claim_next(worker_id, lease_seconds)

    def receive_result(self, result: ExperimentResult) -> bool:
        return self.store.receive_result(result)

    def select_candidates_for_validation(self) -> list[ExperimentResult]:
        return [result for spec in self.store.experiments_with_status("pending_validation") if (result := self.store.result(spec.id)) is not None]

    def assess_promotion(self, result: ExperimentResult, baseline: Any, independent_validation: Any, adversarial_findings: Any, constraints: Any) -> PromotionDecision:
        return self.promotion_policy.assess(baseline, result.evaluation_result, independent_validation, adversarial_findings, constraints)

    def promote(self, specification: ExperimentSpecification, result: ExperimentResult, decision: PromotionDecision) -> bool:
        return decision.decision == "promote" and self.store.promote(specification.strategy_domain, result)


class SweepEngine:
    def __init__(self, strategy: StrategyAdapter, evaluator: EvaluatorAdapter, search_space: SearchSpaceAdapter,
                 objective: ObjectiveAdapter, random_source: RandomSource):
        self.strategy, self.evaluator, self.search_space, self.objective, self.random_source = strategy, evaluator, search_space, objective, random_source

    def run(self, base_reference: str, parameter_space: Any, protocol: EvaluationProtocol, frozen_components: Any = None) -> list[ExperimentResult]:
        validation = self.evaluator.validate(protocol)
        if validation is False:
            raise ValueError("evaluator rejected the evaluation protocol")
        base = self.strategy.load(base_reference)
        results: list[ExperimentResult] = []
        for index, configuration in enumerate(self.search_space.generate_configurations(parameter_space)):
            candidate = self.strategy.clone(base, {"experiment": index})
            candidate = self.strategy.apply_configuration(candidate, configuration)
            evaluation = self.evaluator.evaluate(candidate, protocol, {"random": self.random_source.fork("sweep-%d" % index).snapshot_state()})
            candidate_hash = self.strategy.calculate_fingerprint(candidate)
            reference = self.strategy.save_immutable_candidate(candidate, {"configuration": configuration, "evaluation": _plain(evaluation)})
            results.append(ExperimentResult(
                experiment_id=_id("sweep"), exact_configuration=configuration, exact_diff=self.strategy.calculate_diff(base, candidate),
                changed_components=[], operations=["clone", "apply_configuration", "evaluate"], evaluator_identifier=evaluation.evaluator_identifier,
                evaluator_version=evaluation.evaluator_version, evaluation_protocol_identifier=protocol.identifier,
                random_state_records={"experiment": index}, evaluation_result=evaluation, hypothesis_supported=None,
                target_satisfied=False, conclusion="Sweep evaluation completed.", artifacts=list(evaluation.artifacts),
                reproducible=evaluation.reproducible, candidate_strategy_reference=reference, candidate_strategy_hash=candidate_hash,
            ))
        return list(self.objective.rank(results))


class WorkerPool:
    """Small worker factory wrapper; workers receive claimed specifications only."""
    def __init__(self, worker_factory: Callable[[str], Callable[[ExperimentSpecification], ExperimentResult]], workers: int = 1):
        self.worker_factory, self.workers = worker_factory, max(1, workers)

    def run_once(self, coordinator: ExperimentCoordinator) -> list[bool]:
        def work(index: int) -> bool:
            specification = coordinator.claim_next_experiment("worker-%d" % index)
            if specification is None:
                return False
            return coordinator.receive_result(self.worker_factory("worker-%d" % index)(specification))
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            return list(executor.map(work, range(self.workers)))


@dataclass
class ExperimentSystem:
    strategy_adapter: StrategyAdapter
    evaluator_adapter: EvaluatorAdapter
    search_space_adapter: SearchSpaceAdapter
    objective_adapter: ObjectiveAdapter
    promotion_policy: PromotionPolicy
    experiment_store: DurableExperimentStore
    worker_factory: Callable[[str], Callable[[ExperimentSpecification], ExperimentResult]]
    random_source: RandomSource

    def coordinator(self) -> ExperimentCoordinator:
        return ExperimentCoordinator(self.experiment_store, self.search_space_adapter, self.promotion_policy)

    def sweep_engine(self) -> SweepEngine:
        return SweepEngine(self.strategy_adapter, self.evaluator_adapter, self.search_space_adapter, self.objective_adapter, self.random_source)

    def run_iteration(
        self,
        specification: ExperimentSpecification,
        *,
        baseline_evaluation: Any,
        independent_validation: Any,
        adversarial_findings: Any,
        required_constraints: Any,
        tolerance_policy: Any = None,
    ) -> PromotionDecision:
        """Run one factory iteration without allowing workers to self-promote.

        The worker factory builds a candidate and records its quality-control
        evaluation.  The coordinator then sends that evidence to the promotion
        policy, which is the only path that can update the canonical strategy.
        """
        coordinator = self.coordinator()
        accepted = coordinator.create_distinct_assignments([specification], tolerance_policy)
        if not accepted:
            return PromotionDecision("inconclusive", ["Experiment was duplicate or could not reserve its region."])
        WorkerPool(self.worker_factory, 1).run_once(coordinator)
        result = self.experiment_store.result(specification.id)
        if result is None:
            return PromotionDecision("inconclusive", ["Worker did not produce an accepted structured result."])
        decision = coordinator.assess_promotion(
            result, baseline_evaluation, independent_validation, adversarial_findings, required_constraints
        )
        coordinator.promote(specification, result, decision)
        return decision


def _validate_result(result: ExperimentResult) -> None:
    missing = []
    for name in ("exact_configuration", "exact_diff", "changed_components", "operations", "random_state_records", "evaluation_result", "conclusion", "artifacts"):
        if getattr(result, name, None) is None:
            missing.append(name)
    if not result.evaluator_identifier or not result.evaluator_version or not result.evaluation_protocol_identifier:
        missing.append("evaluator identity")
    if missing:
        raise ValueError("incomplete experiment result: %s" % ", ".join(missing))


def _specification_from_dict(value: dict[str, Any]) -> ExperimentSpecification:
    protocol = value.get("evaluation_protocol") or {}
    value = dict(value)
    value["evaluation_protocol"] = EvaluationProtocol(**protocol)
    return ExperimentSpecification(**value)


def _result_from_dict(value: dict[str, Any]) -> ExperimentResult:
    value = dict(value)
    raw_evaluation = dict(value["evaluation_result"])
    raw_evaluation["individual_runs"] = [EvaluationRun(**run) for run in raw_evaluation.get("individual_runs", [])]
    value["evaluation_result"] = EvaluationResult(**raw_evaluation)
    return ExperimentResult(**value)
