from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from research_harness.experiments import (
    DurableExperimentStore,
    EvaluationProtocol,
    EvaluationResult,
    EvaluationRun,
    ExperimentCoordinator,
    ExperimentResult,
    ExperimentSpecification,
    ExperimentSystem,
    PromotionDecision,
    SeededRandomSource,
    SweepEngine,
)


class DictSearchSpace:
    def normalize(self, value: Any) -> Any:
        return value

    def generate_configurations(self, parameter_space: Any):
        return list(parameter_space)

    def sample_configuration(self, parameter_space: Any, _random: Any) -> Any:
        return parameter_space[0]

    def calculate_distance(self, _first: Any, _second: Any) -> float:
        return 0.0

    def overlaps(self, _first: Any, _second: Any) -> bool:
        return False

    def assign_bucket(self, configuration: Any, _tolerance: Any) -> Any:
        return configuration


class StringStrategy:
    identifier = "string-strategy"

    def load(self, reference: str) -> str:
        return reference

    def clone(self, strategy: str, _context: Any) -> str:
        return str(strategy)

    def read_configuration(self, strategy: str) -> str:
        return strategy

    def apply_configuration(self, strategy: str, changes: str) -> str:
        return "%s:%s" % (strategy, changes)

    def apply_patch(self, strategy: str, patch: str) -> str:
        return "%s:%s" % (strategy, patch)

    def calculate_diff(self, base: str, candidate: str) -> str:
        return "%s -> %s" % (base, candidate)

    def calculate_fingerprint(self, strategy: str) -> str:
        return strategy

    def save_immutable_candidate(self, strategy: str, _metadata: dict[str, Any]) -> str:
        return "candidate://%s" % strategy


class MeasuringEvaluator:
    identifier = "measuring-evaluator"
    version = "1"

    def validate(self, _protocol: EvaluationProtocol) -> bool:
        return True

    def evaluate(self, strategy: str, protocol: EvaluationProtocol, _context: Any = None) -> EvaluationResult:
        score = float(strategy.rsplit(":", 1)[-1])
        run = EvaluationRun(identifier="run", metrics={"score": score})
        return EvaluationResult(self.identifier, self.version, protocol.identifier, [run], {"score": score}, reproducible=True)

    def compare(self, _baseline: Any, _candidate: Any, _protocol: Any, _context: Any = None) -> Any:
        return {}

    def randomness_policy(self) -> str:
        return "explicit-state-only"


class ScoreObjective:
    def aggregate(self, _runs: Any, _definition: Any) -> dict[str, Any]:
        return {}

    def compare(self, _baseline: Any, _candidate: Any, _policy: Any) -> Any:
        return {}

    def rank(self, results: Any) -> Any:
        return sorted(results, key=lambda result: result.evaluation_result.aggregate_metrics["score"], reverse=True)


class ApprovePolicy:
    def assess(self, _baseline: Any, _candidate: Any, _validation: Any, _adversarial: Any, _constraints: Any) -> PromotionDecision:
        return PromotionDecision("promote", ["Independent board accepted the evidence."])


def specification() -> ExperimentSpecification:
    return ExperimentSpecification(
        role="parameter_sweep",
        hypothesis="A higher setting improves the measured score.",
        rationale="Controlled experiment.",
        strategy_domain="test",
        strategy_adapter_identifier="string-strategy",
        evaluator_identifier="measuring-evaluator",
        evaluator_version="1",
        base_strategy_reference="base",
        base_strategy_hash="base-hash",
        evaluation_protocol=EvaluationProtocol("screen"),
        promotion_policy={"minimum": 0.5},
        parameter_space={"setting": 1},
    )


def result_for(experiment_id: str) -> ExperimentResult:
    evaluation = EvaluationResult("measuring-evaluator", "1", "screen", [EvaluationRun("run", {"score": 1.0})], {"score": 1.0}, reproducible=True)
    return ExperimentResult(
        experiment_id=experiment_id,
        exact_configuration={"setting": 1},
        exact_diff="base -> candidate",
        changed_components=["strategy"],
        operations=["build", "measure"],
        evaluator_identifier="measuring-evaluator",
        evaluator_version="1",
        evaluation_protocol_identifier="screen",
        random_state_records={"worker": "state"},
        evaluation_result=evaluation,
        hypothesis_supported=True,
        target_satisfied=True,
        conclusion="Measured improvement.",
        artifacts=["result.json"],
        reproducible=True,
        candidate_strategy_reference="candidate://best",
        candidate_strategy_hash="best-hash",
    )


class ExperimentFactoryTest(unittest.TestCase):
    def test_atomic_claim_deduplication_and_result_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = DurableExperimentStore(Path(directory) / "factory.sqlite")
            coordinator = ExperimentCoordinator(store, DictSearchSpace(), ApprovePolicy())
            spec = specification()

            self.assertEqual(coordinator.create_distinct_assignments([spec]), [spec])
            self.assertEqual(coordinator.create_distinct_assignments([specification()]), [])
            claimed = coordinator.claim_next_experiment("worker-a")
            self.assertIsNotNone(claimed)
            self.assertIsNone(coordinator.claim_next_experiment("worker-b"))
            self.assertTrue(coordinator.receive_result(result_for(spec.id)))
            self.assertEqual(store.status(spec.id), "pending_validation")
            self.assertEqual([item.experiment_id for item in coordinator.select_candidates_for_validation()], [spec.id])

    def test_system_uses_separate_board_for_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = DurableExperimentStore(Path(directory) / "factory.sqlite")

            def worker(_worker_id: str):
                return lambda assigned: result_for(assigned.id)

            system = ExperimentSystem(
                StringStrategy(), MeasuringEvaluator(), DictSearchSpace(), ScoreObjective(), ApprovePolicy(), store, worker, SeededRandomSource(7)
            )
            spec = specification()
            decision = system.run_iteration(
                spec,
                baseline_evaluation={"score": 0.0},
                independent_validation={"passed": True},
                adversarial_findings=[],
                required_constraints={"score": ">= 0.5"},
            )

            self.assertEqual(decision.decision, "promote")
            self.assertEqual(store.status(spec.id), "promoted")

    def test_sweep_persists_independent_candidates_in_score_order(self) -> None:
        results = SweepEngine(StringStrategy(), MeasuringEvaluator(), DictSearchSpace(), ScoreObjective(), SeededRandomSource(3)).run(
            "base", ["0.2", "0.9"], EvaluationProtocol("screen")
        )

        self.assertEqual([result.evaluation_result.aggregate_metrics["score"] for result in results], [0.9, 0.2])
        self.assertTrue(all(result.candidate_strategy_reference for result in results))

    def test_random_forks_are_replayable_and_isolated(self) -> None:
        source = SeededRandomSource(11)
        self.assertEqual(source.fork("strategy").next_float(), source.fork("strategy").next_float())
        self.assertNotEqual(source.fork("strategy").next_float(), source.fork("environment").next_float())

    def test_incomplete_result_is_rejected(self) -> None:
        invalid = result_for("missing")
        invalid.exact_diff = None
        with self.assertRaisesRegex(ValueError, "exact_diff"):
            from research_harness.experiments import _validate_result
            _validate_result(invalid)


if __name__ == "__main__":
    unittest.main()
