from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import sqlite3
import tempfile
import textwrap
import unittest
import unittest.mock
import urllib.error
from pathlib import Path

from challenges.prediction_market import prediction_market_score
from research_harness.benchmark import collect_runs, write_outputs
from research_harness.cli import build_parser, configure_interactive_run
from research_harness.evals import (
    EvaluationHarness,
    EvalTask,
    GraderResult,
    aggregate_results,
    default_eval_suite,
    default_graders,
    edge_eval_suite,
    preflight_eval_suite,
    select_eval_tasks,
    graph_trajectory_match,
    trajectory_optimizer_flow,
    trajectory_match,
)
from research_harness.evals.cli import build_parser as build_eval_parser
from research_harness.agents import SynthesisAgent
from research_harness.llm import LLMClient, LLMError
from research_harness.loops import EvaluatorRegistry, EvaluatorResult, EvolutionaryOuterLoop, InnerLoopResult, OptimizeLoop, PlateauDetector, ResearchLoop, TaskRouter
from research_harness.orchestrator import HarnessConfig, Orchestrator, goal_slug
from research_harness.schemas import AgentTrace, Claim, Contradiction, EvolutionRound, Hypothesis, RunRecord, Source, SourceStrategyItem, Variant, VariantEvaluation
from research_harness.search import CorpusDocument, LocalCorpusSearch, OpenAlexSearch, SemanticScholarSearch, _parse_arxiv_feed, _score_documents
from research_harness.sessions import SessionStore
from research_harness.store import ArtifactStore


class SmokeTest(unittest.TestCase):
    def test_llm_failures_are_recorded_in_cost_history(self) -> None:
        import urllib.error

        body = io.BytesIO(b'{"error":{"message":"max_tokens is not supported"}}')
        error = urllib.error.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=body,
        )
        client = LLMClient(provider="openai", model="gpt-5.2", api_key="sk-test")

        with unittest.mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(LLMError):
                client.complete("system", "user", max_output_tokens=10)

        self.assertEqual(client.cost_breakdown()["model_call_count"], 1)
        self.assertEqual(client.call_history[0]["status"], "failed")
        self.assertIn("max_tokens is not supported", client.call_history[0]["error"])

    def test_llm_model_catalog_resolves_openai_and_anthropic_labs(self) -> None:
        from research_harness.model_catalog import configured_model_pool, model_choices, resolve_model_selection

        with unittest.mock.patch.dict(os.environ, {"RESEARCH_HARNESS_LLM_MODELS": ""}, clear=False):
            choices = dict(model_choices())

            self.assertIn("all-configured", choices)
            self.assertIn("openai/gpt-5.5", choices)
            self.assertIn("anthropic/claude-opus-4-6", choices)
            self.assertIn("anthropic/claude-sonnet-4-6", choices)
            self.assertIn("openai/gpt-5.2", choices)
            self.assertIn("kimi/kimi-k2.6", choices)
            self.assertIn("ollama/qwen3.5:latest", choices)
            self.assertIn("ollama/qwen", choices)
            self.assertIn("anthropic/claude-sonnet-4-5", choices)
            self.assertEqual(resolve_model_selection("auto", "anthropic/claude-sonnet-4-5"), ("anthropic", "claude-sonnet-4-5"))
            self.assertEqual(resolve_model_selection("auto", "openai/gpt-5.2"), ("openai", "gpt-5.2"))
            self.assertEqual(resolve_model_selection("auto", "kimi/kimi-k2.6"), ("kimi", "kimi-k2.6"))
            self.assertEqual(resolve_model_selection("auto", "ollama/qwen3.5:latest"), ("ollama", "qwen3.5:latest"))
            self.assertEqual(resolve_model_selection("auto", "ollama/qwen"), ("ollama", "qwen"))
            self.assertEqual(resolve_model_selection("openai", "kimi/kimi-k2.6"), ("kimi", "kimi-k2.6"))
            self.assertEqual(resolve_model_selection("auto", "all-configured"), ("multi", "all-configured"))

        with unittest.mock.patch.dict(os.environ, {"RESEARCH_HARNESS_LLM_MODELS": "openai/custom-a,local/local-deterministic-fallback"}, clear=False):
            choices = dict(model_choices())
            pool = configured_model_pool()

            self.assertIn("openai/custom-a", choices)
            self.assertIn("openai/gpt-5.5", choices)
            self.assertEqual([option.id for option in pool], ["openai/custom-a", "local/local-deterministic-fallback"])

    def test_llm_client_uses_anthropic_key_for_claude_model(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test", "OPENAI_API_KEY": ""}, clear=False):
            client = LLMClient(provider="auto", model="anthropic/claude-sonnet-4-5")

        self.assertEqual(client.provider, "anthropic")
        self.assertEqual(client.model, "claude-sonnet-4-5")
        self.assertTrue(client.is_live)

    def test_llm_client_uses_moonshot_key_for_kimi_model(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"MOONSHOT_API_KEY": "moonshot-test", "OPENAI_API_KEY": "", "ANTHROPIC_API_KEY": ""}, clear=False):
            client = LLMClient(provider="auto", model="kimi/kimi-k2.6")

        self.assertEqual(client.provider, "kimi")
        self.assertEqual(client.model, "kimi-k2.6")
        self.assertTrue(client.is_live)

    def test_llm_client_uses_ollama_chat_api(self) -> None:
        captured: dict[str, object] = {}

        class FakeResponse:
            def __init__(self, payload: dict[str, object]):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            url = request.full_url
            if url.endswith("/api/tags"):
                return FakeResponse({"models": [{"name": "qwen"}]})
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse(
                {
                    "model": "qwen",
                    "message": {"role": "assistant", "content": "local answer"},
                    "prompt_eval_count": 7,
                    "eval_count": 3,
                }
            )

        with unittest.mock.patch.dict(os.environ, {"OLLAMA_HOST": "localhost:11434"}, clear=False):
            client = LLMClient(provider="auto", model="ollama/qwen")
            with unittest.mock.patch("urllib.request.urlopen", fake_urlopen):
                response = client.complete("system", "user", max_output_tokens=12, temperature=0.2)

        self.assertEqual(client.provider, "ollama")
        self.assertEqual(response.provider, "ollama")
        self.assertEqual(response.text, "local answer")
        self.assertEqual(response.cost, 0.0)
        self.assertEqual(captured["payload"]["model"], "qwen")
        self.assertEqual(captured["payload"]["options"]["num_predict"], 12)

    def test_kimi_requests_force_temperature_one(self) -> None:
        captured: dict[str, object] = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": {}}).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        with unittest.mock.patch.dict(os.environ, {"MOONSHOT_API_KEY": "moonshot-test"}, clear=False):
            client = LLMClient(provider="kimi", model="kimi-k2.6")
            with unittest.mock.patch("urllib.request.urlopen", fake_urlopen):
                client.complete("system", "user", temperature=0.2)

        self.assertEqual(captured["payload"]["temperature"], 1)

    def test_kimi_retries_rate_limit(self) -> None:
        calls = {"count": 0}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": {}}).encode("utf-8")

        def fake_urlopen(_request, timeout=0):
            calls["count"] += 1
            if calls["count"] < 3:
                raise urllib.error.HTTPError(
                    url="https://api.moonshot.ai/v1/chat/completions",
                    code=429,
                    msg="rate limited",
                    hdrs={"Retry-After": "0"},
                    fp=io.BytesIO(b'{"error":{"message":"rate limit"}}'),
                )
            return FakeResponse()

        with unittest.mock.patch.dict(os.environ, {"MOONSHOT_API_KEY": "moonshot-test"}, clear=False):
            client = LLMClient(provider="kimi", model="kimi-k2.6")
            with unittest.mock.patch("urllib.request.urlopen", fake_urlopen), unittest.mock.patch("time.sleep", lambda _seconds: None):
                response = client.complete("system", "user")

        self.assertEqual(response.text, "ok")
        self.assertEqual(calls["count"], 3)

    def test_llm_client_loads_kimi_key_from_env_local_defaults(self) -> None:
        from research_harness import llm as llm_module

        with tempfile.TemporaryDirectory() as directory:
            env_local = Path(directory) / ".env.local"
            env_local.write_text("MOONSHOT_API_KEY=moonshot-from-file\n", encoding="utf-8")
            with unittest.mock.patch.dict(os.environ, {}, clear=True):
                with contextlib.ExitStack() as stack:
                    stack.enter_context(unittest.mock.patch.object(llm_module, "_ENV_DEFAULTS_LOADED", False))
                    stack.enter_context(unittest.mock.patch.object(llm_module, "Path", lambda value="": env_local if value == ".env.local" else Path(directory) / str(value)))
                    client = LLMClient(provider="kimi", model="kimi-k2.6")

        self.assertEqual(client.kimi_api_key, "moonshot-from-file")
        self.assertTrue(client.is_live)

    def test_llm_client_all_configured_round_robins_available_models(self) -> None:
        env = {
            "RESEARCH_HARNESS_LLM_MODELS": "openai/gpt-test-a,anthropic/claude-test-b,local/local-deterministic-fallback",
            "OPENAI_API_KEY": "sk-test",
            "ANTHROPIC_API_KEY": "sk-ant-test",
        }
        with unittest.mock.patch.dict(os.environ, env, clear=False):
            client = LLMClient(provider="auto", model="all-configured")

        def fake_openai(_system: str, _user: str, **_kwargs: object):
            return type("Response", (), {"text": "openai", "model": client.model, "provider": "openai", "prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0})()

        def fake_anthropic(_system: str, _user: str, **_kwargs: object):
            return type("Response", (), {"text": "anthropic", "model": client.model, "provider": "anthropic", "prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0})()

        with unittest.mock.patch.object(client, "_openai_response", side_effect=fake_openai), unittest.mock.patch.object(client, "_anthropic_response", side_effect=fake_anthropic):
            responses = [client.complete("s", "u") for _ in range(4)]

        self.assertEqual(client.provider, "multi")
        self.assertEqual(client.model, "all-configured")
        self.assertEqual([response.provider for response in responses], ["openai", "anthropic", "local", "openai"])
        self.assertEqual([call["model"] for call in client.call_history], ["gpt-test-a", "claude-test-b", "local-deterministic-fallback", "gpt-test-a"])

    def test_cli_allows_selection_based_setup_without_goal(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])

        self.assertIsNone(args.goal)
        self.assertFalse(args.interactive)
        self.assertFalse(args.preflight)

    def test_cli_preflight_is_explicit_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--preflight", "--preflight-eval", "research_uses_at_least_four_source_families", "Research agents"])

        self.assertTrue(args.preflight)
        self.assertEqual(args.preflight_suite, "preflight")
        self.assertEqual(args.preflight_eval_ids, ["research_uses_at_least_four_source_families"])
        self.assertFalse(args.no_steering)

    def test_user_steering_inbox_ingests_articles_as_sources_and_claims(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            store.append_user_steering(
                "/article https://example.edu/paper | Useful mechanism paper | This suggests a fresh evaluation angle."
            )
            ingested = store.ingest_pending_user_steering("run_steering")

            sources = store.list("sources")
            claims = store.list("claims")

        self.assertEqual(ingested, 1)
        self.assertEqual(sources[0]["url"], "https://example.edu/paper")
        self.assertEqual(sources[0]["source_type"], "user_steering")
        self.assertEqual(claims[0]["created_by_agent"], "user_steering")
        self.assertIn("fresh evaluation angle", claims[0]["text"])

    def test_interactive_cli_setup_collects_run_choices(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--interactive", "--retriever", "local"])
        text_answers = iter(
            [
                "Research agent workflow evaluation",
                "5",
            ]
        )
        keys = iter(["down", "down", "enter", "down", "enter", "enter", "enter"])
        prompts: list[str] = []

        with contextlib.redirect_stdout(io.StringIO()):
            configured = configure_interactive_run(
                args,
                input_func=lambda prompt: prompts.append(prompt) or next(text_answers),
                output_func=lambda _message: None,
                key_reader=lambda: next(keys),
            )

        self.assertEqual(configured.goal, "Research agent workflow evaluation")
        self.assertEqual(configured.task_mode, "optimize")
        self.assertEqual(configured.evaluator, "length_score")
        self.assertEqual(configured.retriever, "local")
        self.assertEqual(configured.max_iterations, 5)
        self.assertEqual(configured.llm_provider, "openai")
        self.assertEqual(configured.llm_model, "gpt-5.2")
        self.assertFalse(configured.quiet)
        self.assertFalse(any("Choose a number" in prompt for prompt in prompts))

    def test_interactive_prediction_market_routes_to_challenge_mode(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--interactive", "--retriever", "local"])
        text_answers = iter(["optimizing prediction market making", "12"])
        keys = iter(["down", "down", "enter", "down", "down", "enter", "enter", "enter"])

        with contextlib.redirect_stdout(io.StringIO()):
            configured = configure_interactive_run(
                args,
                input_func=lambda _prompt: next(text_answers),
                output_func=lambda _message: None,
                key_reader=lambda: next(keys),
            )

        self.assertEqual(configured.task_mode, "optimize_query")
        self.assertEqual(configured.evaluator, "prediction_market")
        self.assertEqual(configured.optimization_preset, "challenge")
        self.assertTrue(configured.max_iterations_explicit)

    def test_cli_accepts_optimization_challenge_knobs(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "optimize kernel challenge",
                "--task-mode",
                "optimize",
                "--evaluator",
                "length_score",
                "--optimization-preset",
                "challenge",
                "--population-size",
                "64",
                "--parent-count",
                "5",
                "--parallel-evaluator-cap",
                "12",
            ]
        )

        self.assertEqual(args.optimization_preset, "challenge")
        self.assertEqual(args.population_size, 64)
        self.assertEqual(args.parent_count, 5)
        self.assertEqual(args.parallel_evaluator_cap, 12)

    def test_challenge_preset_expands_optimizer_defaults(self) -> None:
        config = HarnessConfig(
            optimization_preset="challenge",
            max_loop_iterations=12,
            evolution_population_size=4,
            optimizer_parent_count=2,
            parallel_evaluator_cap=8,
        )
        orchestrator = Orchestrator(
            Path("examples/corpus/research_corpus.json"),
            Path("outputs"),
            config=config,
        )

        self.assertEqual(orchestrator.config.max_loop_iterations, 20)
        self.assertEqual(orchestrator.config.evolution_population_size, 48)
        self.assertEqual(orchestrator.config.query_population_size, 16)
        self.assertEqual(orchestrator.config.optimizer_parent_count, 4)
        self.assertEqual(orchestrator.config.parallel_evaluator_cap, 16)
        self.assertEqual(orchestrator.config.optimize_plateau_patience, 5)
        self.assertTrue(orchestrator.config.continue_on_optimize_plateau)

    def test_challenge_preset_respects_explicit_iteration_cap(self) -> None:
        config = HarnessConfig(
            optimization_preset="challenge",
            max_loop_iterations=2,
            max_loop_iterations_explicit=True,
        )
        orchestrator = Orchestrator(
            Path("examples/corpus/research_corpus.json"),
            Path("outputs"),
            config=config,
        )

        self.assertEqual(orchestrator.config.max_loop_iterations, 2)
        self.assertEqual(orchestrator.config.evolution_population_size, 48)
        self.assertEqual(orchestrator.config.query_population_size, 16)

    def test_optimize_query_research_fanout_can_be_capped_separately(self) -> None:
        async def fake_evaluate(_self, variants, _store):
            return InnerLoopResult(
                ranked_evaluations=[
                    VariantEvaluation(
                        run_id="run_query_cap",
                        variant_id=variants[0].id,
                        inner_loop="optimize_query",
                        score=0.5,
                        metrics={},
                        judge_scores=[0.5],
                        summary="fake",
                        passed=False,
                    )
                ],
                termination_signal="continue",
            )

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            outer = EvolutionaryOuterLoop(
                run_id="run_query_cap",
                goal="prediction market challenge",
                task_mode="optimize_query",
                source_strategy=[],
                search_factory=lambda _name: LocalCorpusSearch(Path("examples/corpus/research_corpus.json")),
                evaluator=lambda payload: len(payload) / 100.0,
                evaluator_name="prediction_market",
                max_outer_iterations=1,
                population_size=48,
                query_population_size=3,
            )
            observed_populations: list[int] = []

            def propose_query(round_index, _parents, _store):
                observed_populations.append(outer.population_size)
                return [
                    Variant(
                        run_id=outer.run_id,
                        outer_iteration=round_index,
                        kind="query",
                        payload=f"query {index}",
                        parent_ids=[],
                        metadata={},
                    )
                    for index in range(outer.population_size)
                ]

            async def skip_optimizer(_store, _parents, _seed_context):
                observed_populations.append(outer.population_size)

            outer._propose_query_variants = propose_query  # type: ignore[method-assign]
            outer._run_prediction_market_optimizer = skip_optimizer  # type: ignore[method-assign]
            with unittest.mock.patch("research_harness.loops.OptimizationQueryLoop.evaluate", fake_evaluate):
                asyncio.run(outer._run_optimize_query(store))
            variant_count = len([row for row in store.list("variants") if row["kind"] == "query"])

        self.assertEqual(observed_populations, [3, 1])
        self.assertEqual(variant_count, 3)

    def test_prediction_market_evaluator_overrides_plain_optimize_to_challenge_flow(self) -> None:
        router = TaskRouter(EvaluatorRegistry())

        decision = router.decide(
            "optimizing the prediction market market making strategy, get to 10 dollars",
            requested_mode="optimize",
            evaluator_name="prediction_market",
        )

        self.assertEqual(decision.selected_mode, "optimize_query")
        self.assertEqual(decision.product_agent, "challenge")
        self.assertEqual(decision.evaluator_name, "prediction_market")

    def test_optimize_loop_records_json_evaluator_responses_and_failures(self) -> None:
        def evaluator(payload: str) -> EvaluatorResult:
            if payload == "bad":
                raise TimeoutError("benchmark timed out")
            return EvaluatorResult(
                score=0.42,
                status="completed",
                metrics={"latency_ms": 12.5},
                diagnostics={"correctness": "passed"},
                loss_reason="slower_than_baseline",
                summary="Variant was correct but slow.",
            )

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            variants = [
                Variant(
                    run_id="run_json_eval",
                    outer_iteration=1,
                    kind="code",
                    payload="good",
                    parent_ids=[],
                    metadata={"strategy_family": "control_baseline", "mechanism_hypothesis": "test direction metadata"},
                ),
                Variant(run_id="run_json_eval", outer_iteration=1, kind="code", payload="bad", parent_ids=[], metadata={}),
            ]
            for variant in variants:
                store.add_variant(variant)
            result = asyncio.run(OptimizeLoop("run_json_eval", evaluator, parallel_evaluator_cap=1).evaluate(variants, store))
            evaluations = store.list("variant_evaluations")

        self.assertEqual(len(result.ranked_evaluations), 2)
        self.assertEqual(len(evaluations), 2)
        json_responses = [row["metrics"]["json_response"] for row in evaluations]
        self.assertTrue(all(isinstance(response, dict) for response in json_responses))
        self.assertIn("slower_than_baseline", {response["loss_reason"] for response in json_responses})
        self.assertIn("timeout", {response["loss_reason"] for response in json_responses})
        self.assertTrue(all(row["summary"].startswith("{") for row in evaluations))
        self.assertEqual(evaluations[0]["metrics"]["direction"]["strategy_family"], "control_baseline")

    def test_direction_entropy_forces_multiple_code_families_and_novelty_slots(self) -> None:
        outer = EvolutionaryOuterLoop(
            run_id="run_direction_entropy",
            goal="optimize a matrix multiplication kernel for latency and correctness",
            task_mode="optimize",
            source_strategy=[],
            search_factory=lambda _name: LocalCorpusSearch(Path("examples/corpus/research_corpus.json")),
            evaluator_name="length_score",
            population_size=12,
            novelty_fraction=0.25,
        )

        variants = outer._propose_code_variants(1, [], None)
        families = {variant.metadata.get("strategy_family") for variant in variants}
        roles = {variant.metadata.get("entropy_role") for variant in variants}

        self.assertEqual(len(variants), 12)
        self.assertGreaterEqual(len(families), 6)
        self.assertIn("novelty", roles)
        self.assertIn("ablation", roles)
        self.assertTrue(all(variant.metadata.get("mechanism_hypothesis") for variant in variants))
        self.assertTrue(all(variant.metadata.get("paired_crn") is True for variant in variants))
        self.assertTrue(any("strategy_family=" in variant.payload for variant in variants))

    def test_direction_entropy_forces_multiple_research_query_directions(self) -> None:
        outer = EvolutionaryOuterLoop(
            run_id="run_query_direction_entropy",
            goal="research robustness methods for automated theorem proving agents",
            task_mode="research",
            source_strategy=[
                SourceStrategyItem(
                    name="local",
                    retriever="local",
                    purpose="local corpus",
                    queries=["automated theorem proving agent robustness"],
                    limit=6,
                )
            ],
            search_factory=lambda _name: LocalCorpusSearch(Path("examples/corpus/research_corpus.json")),
            population_size=8,
        )

        variants = outer._propose_query_variants(1, [], None)
        families = {variant.metadata.get("strategy_family") for variant in variants}

        self.assertEqual(len(variants), 8)
        self.assertGreaterEqual(len(families), 6)
        self.assertTrue(all("mechanism_hypothesis:" in variant.payload for variant in variants))
        self.assertTrue(all(variant.metadata.get("eval_protocol") == "paired_crn_same_seeds_across_variants" for variant in variants))

    def test_plan_uses_llm_interpretation_for_typoed_prediction_market_goal(self) -> None:
        class FakePlannerLLM:
            is_live = True

            def complete_json(self, _system: str, user: str, **_kwargs: object) -> dict[str, object]:
                self.user_payload = json.loads(user)
                return {
                    "task_type": "bounded",
                    "topics": ["prediction_market", "market_making"],
                    "topic_queries": [
                        "prediction market challenge evaluation strategy implementation",
                        "prediction market trading strategy empirical evaluation",
                    ],
                    "rationale": "Interpreted typoed predictionm arket and mm'ing as prediction-market market making.",
                }

        orchestrator = Orchestrator(
            corpus_path=Path("examples/corpus/research_corpus.json"),
            output_root=Path("outputs"),
            config=HarnessConfig(retriever="auto"),
        )
        fake_llm = FakePlannerLLM()
        orchestrator.llm = fake_llm  # type: ignore[assignment]

        plan = orchestrator.create_plan("optimizing the predictionm arket mm'ing, get to 10$")
        strategy = orchestrator.create_source_strategy(plan.goal, plan)
        queries = " ".join(query for item in strategy for query in item.queries).lower()

        self.assertEqual(plan.planner, "llm")
        self.assertIn("prediction_market", plan.topics)
        self.assertIn("prediction market challenge evaluation", queries)
        self.assertIn("selected_evaluator", fake_llm.user_payload)

    def test_offline_research_plan_avoids_fixed_agentic_lenses(self) -> None:
        orchestrator = Orchestrator(
            corpus_path=Path("examples/corpus/research_corpus.json"),
            output_root=Path("outputs"),
            config=HarnessConfig(retriever="local"),
        )

        plan = orchestrator.create_plan(
            'When we say "introduce entropy in AI agent systems", do we mean introduce varied information?'
        )
        strategy = orchestrator.create_source_strategy(plan.goal, plan)
        combined = " ".join(
            [*plan.search_angles, *plan.hypothesis_angles, *[query for item in strategy for query in item.queries]]
        ).lower()

        self.assertIn("entropy", combined)
        for leaked in [
            "breadth-first landscape scan",
            "primary-source mechanisms",
            "recent empirical evidence",
            "contradictory evidence limitations",
            "agentic ai",
        ]:
            self.assertNotIn(leaked, combined)

    def test_prior_run_memory_uses_only_related_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            neuro = ArtifactStore(root / "001_run_neuroscience")
            neuro_run = RunRecord(
                id="001_run_neuroscience",
                user_goal="Research neuroscience and artificial intelligence",
                task_type="open_ended",
                harness_config_id="test-config",
                prompt_versions={},
                harness_config_snapshot={},
            )
            neuro.add_run(neuro_run)
            neuro.write_report(
                "# Research Report: neuroscience\n\n"
                "## Key Takeaways\n"
                "- Brain-inspired representations connect cognitive neuroscience and artificial intelligence.\n"
            )
            pm = ArtifactStore(root / "002_run_prediction_market")
            pm_run = RunRecord(
                id="002_run_prediction_market",
                user_goal="Research prediction market maker inventory controls",
                task_type="open_ended",
                harness_config_id="test-config",
                prompt_versions={},
                harness_config_snapshot={},
            )
            pm.add_run(pm_run)
            pm.write_report(
                "# Research Report: prediction markets\n\n"
                "## Key Takeaways\n"
                "- Prediction market makers need inventory-aware quoting and spread controls.\n"
                "## Open Questions\n"
                "- Which inventory controls improve market-maker edge without killing fill rate?\n"
            )
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=root,
                config=HarnessConfig(retriever="auto"),
            )

            memory = orchestrator.load_prior_run_memory("Research prediction market quoting strategies")

            self.assertEqual(memory["checked_run_count"], 1)
            self.assertIn("prediction_market", memory["checked_reports"][0]["run_id"])
            self.assertTrue(any("neuroscience" in row["run_id"] for row in memory["skipped_reports"]))
            self.assertTrue(memory["avoid_directions"])
            self.assertTrue(memory["unresolved_directions"])

    def test_phase2_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(mode="standard", retriever="local", echo_progress=False),
            )
            run, store = asyncio.run(
                orchestrator.run(
                    "Research how multi-agent systems improve automated literature review quality",
                    mode="standard",
                )
            )

            self.assertEqual(run.status, "completed")
            self.assertTrue(store.report_path.exists())
            self.assertGreaterEqual(len(store.list("sources")), 2)
            self.assertGreaterEqual(len(store.list("claims")), 4)
            self.assertGreaterEqual(len(store.list("hypotheses")), 1)
            self.assertGreaterEqual(len(store.list("agent_traces")), 6)
            self.assertEqual(len(store.list("harness_changes")), 1)
            self.assertTrue(run.id.startswith("001_run_multi-agent-systems-improve-automated-literature-review-quality"))
            self.assertTrue(store.run_state_path.exists())
            self.assertEqual(json.loads(store.run_state_path.read_text(encoding="utf-8"))["stage"], "completed")

    def test_interrupt_synthesizes_partial_artifacts(self) -> None:
        async def interrupted_outer_loop(_loop, _store):
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(retriever="local", max_loop_iterations=3, include_debugger=False, echo_progress=False),
            )
            with unittest.mock.patch("research_harness.orchestrator.EvolutionaryOuterLoop.run", interrupted_outer_loop):
                run, store = asyncio.run(
                    orchestrator.run(
                        "Research how multi-agent systems improve automated literature review quality",
                    )
                )

            self.assertEqual(run.status, "cancelled")
            self.assertTrue(store.report_path.exists())
            self.assertIn("## Run Interrupted", store.report_path.read_text(encoding="utf-8"))
            self.assertTrue(store.run_state_path.exists())
            self.assertTrue(store.cost_path.exists())
            self.assertTrue(store.run_benchmark_path.exists())
            progress = store.progress_path.read_text(encoding="utf-8")
            self.assertIn("Interrupt received", progress)
            self.assertIn("Interrupt synthesis complete", progress)

    def test_world_model_dedup_provenance_and_observability_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            store_a = ArtifactStore(output_root / "001_run_world")
            run_a = RunRecord(
                id="001_run_world",
                user_goal="Research provenance",
                task_type="open_ended",
                harness_config_id="test-config",
                prompt_versions={"literature_agent": "abc123"},
                harness_config_snapshot={"mode": "test"},
            )
            store_a.add_run(run_a)
            source_a = store_a.add_source(
                Source(
                    url="https://example.com/paper",
                    title="World model paper",
                    author="Ada",
                    date="2026",
                    source_type="paper",
                    summary="A paper about persistent world models.",
                    relevance_score=0.9,
                    credibility_score=0.9,
                )
            )
            claim_a = store_a.add_claim(
                Claim(
                    text="Persistent stores improve cross-run memory.",
                    source_ids=[source_a.id],
                    confidence=0.8,
                    support_level="strong",
                    created_by_agent="test",
                    run_id=run_a.id,
                )
            )
            hypothesis_a = store_a.add_hypothesis(
                Hypothesis(
                    text="Cross-run dedupe should reduce repeated claims.",
                    supporting_claim_ids=[claim_a.id],
                    contradicting_claim_ids=[],
                    confidence=0.7,
                    novelty_score=0.6,
                    testability_score=0.8,
                    next_experiment="Run the same source twice.",
                )
            )
            store_a.add_contradiction(
                Contradiction(
                    claim_a=claim_a.id,
                    claim_b=claim_a.id,
                    explanation="Self-check edge for provenance coverage.",
                    severity="low",
                )
            )
            store_a.write_report("Report citing the persistent world model claim.\n")
            store_a.write_harness_diagnosis()

            store_b = ArtifactStore(output_root / "002_run_world")
            source_b = store_b.add_source(
                Source(
                    url="https://example.com/paper",
                    title="World model paper",
                    author="Ada",
                    date="2026",
                    source_type="paper",
                    summary="A paper about persistent world models.",
                    relevance_score=0.9,
                    credibility_score=0.9,
                )
            )
            claim_b = store_b.add_claim(
                Claim(
                    text="Persistent stores improve cross-run memory.",
                    source_ids=[source_b.id],
                    confidence=0.8,
                    support_level="strong",
                    created_by_agent="test",
                    run_id="002_run_world",
                )
            )

            self.assertEqual(source_b.duplicate_of, source_a.id)
            self.assertEqual(claim_b.duplicate_of, claim_a.id)
            self.assertTrue(store_a.sqlite_path.exists())
            self.assertTrue(store_a.harness_diagnosis_path.exists())
            self.assertGreaterEqual(len(store_a.list("provenance_edges")), 4)
            self.assertEqual(hypothesis_a.run_id, run_a.id)
            with sqlite3.connect(store_a.sqlite_path) as connection:
                artifact_count = connection.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
                migration_count = connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
            self.assertGreaterEqual(artifact_count, 4)
            self.assertGreaterEqual(migration_count, 1)

    def test_duplicate_run_names_are_numbered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(mode="deterministic", retriever="local", echo_progress=False),
            )
            first_run, _ = asyncio.run(orchestrator.run("Research agent memory systems", mode="deterministic"))
            second_run, _ = asyncio.run(orchestrator.run("Research agent memory systems", mode="deterministic"))

            self.assertEqual(first_run.id, "001_run_agent-memory-systems")
            self.assertEqual(second_run.id, "002_run_agent-memory-systems")

    def test_sessions_are_plaintext_jsonl_and_snapshot_previous_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=root / "outputs",
                config=HarnessConfig(
                    mode="deterministic",
                    retriever="local",
                    session_projects_dir=root / "autore" / "projects",
                    echo_progress=False,
                ),
            )
            run, store = asyncio.run(orchestrator.run("Research agent session memory", mode="deterministic"))

            self.assertIsNotNone(run.session_id)
            self.assertIsNotNone(run.session_jsonl_path)
            session_jsonl = Path(run.session_jsonl_path or "")
            self.assertTrue(session_jsonl.exists())
            rows = [json.loads(line) for line in session_jsonl.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["event"], "session_start")
            self.assertTrue(any(row["event"] == "progress" for row in rows))
            self.assertTrue(any(row["event"] == "agent_trace" for row in rows))
            self.assertTrue(any(row["event"] == "artifact_write" for row in rows))
            self.assertTrue(any(row["event"] == "snapshot" for row in rows))
            run_state = json.loads(store.run_state_path.read_text(encoding="utf-8"))
            self.assertEqual(run_state["artifacts"]["session_jsonl"], str(session_jsonl))

    def test_session_store_records_resume_and_fork_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root / "workspace", root / "autore" / "projects")
            record = store.start_session(
                goal="Optimize from prior artifacts",
                run_id="001_run_optimize",
                output_dir=root / "outputs" / "001_run_optimize",
                resume_from="session_old",
                fork_from="session_branch",
            )
            store.complete_session(status="completed", summary="ok")

            metadata = json.loads(record.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["resume_from"], "session_old")
            self.assertEqual(metadata["fork_from"], "session_branch")
            self.assertEqual(metadata["status"], "completed")
            events = [json.loads(line)["event"] for line in record.jsonl_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events, ["session_start", "session_complete"])

    def test_loop_mode_runs_nested_research_evolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(retriever="local", max_loop_iterations=3, echo_progress=False),
            )
            run, store = asyncio.run(
                orchestrator.run(
                    "Research how multi-agent systems improve automated literature review quality",
                )
            )

            tasks = store.list("loop_tasks")
            iterations = store.list("loop_iterations")

            self.assertEqual(run.status, "completed")
            self.assertEqual(run.task_mode, "research")
            self.assertEqual(run.product_agent, "research")
            self.assertGreaterEqual(len(tasks), 5)
            self.assertTrue(all(task["passes"] for task in tasks))
            self.assertEqual(len(iterations), len(tasks))
            self.assertEqual(store.list("task_ingestion_decisions")[0]["selected_mode"], "research")
            self.assertGreaterEqual(len(store.list("variants")), 1)
            self.assertGreaterEqual(len(store.list("variant_evaluations")), 1)
            self.assertGreaterEqual(len(store.list("evolution_rounds")), 1)
            self.assertEqual(len(store.list("loop_continuation_decisions")), len(store.list("evolution_rounds")))
            self.assertTrue(all(row["decision"] in {"continue", "exit"} for row in store.list("loop_continuation_decisions")))
            self.assertTrue(store.report_path.exists())
            self.assertTrue(store.report_tex_path.exists())
            self.assertTrue(store.report_pdf_path.exists())
            self.assertTrue(store.report_preview_path.exists())
            self.assertGreater(store.report_preview_path.stat().st_size, 1000)
            self.assertIn(r"\section{Key Takeaways}", store.report_tex_path.read_text(encoding="utf-8"))
            self.assertIn("## Key Takeaways", store.report_path.read_text(encoding="utf-8"))
            self.assertTrue(any(source.get("evidence_sections") for source in store.list("sources")))
            self.assertTrue(store.run_state_path.exists())
            run_state = json.loads(store.run_state_path.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(run_state["observed_actions"]), 5)
            self.assertTrue(run_state["research_architecture"]["enabled_for_mode"])
            self.assertEqual(run_state["research_architecture"]["lead_agent"]["role"], "lead_research_orchestrator")
            self.assertEqual(run_state["research_architecture"]["subagents"]["role"], "parallel_research_subagents")
            self.assertIn("asyncio.gather", run_state["research_architecture"]["subagents"]["parallelism"])
            self.assertEqual(
                {item["name"] for item in run_state["research_architecture"]["judge_rubric"]},
                {"factual_accuracy", "citation_accuracy", "completeness", "source_quality", "tool_efficiency"},
            )
            research_evaluations = [row for row in store.list("variant_evaluations") if row["inner_loop"] == "research"]
            self.assertTrue(research_evaluations)
            for metric in ["factual_accuracy", "citation_accuracy", "completeness", "source_quality", "tool_efficiency"]:
                self.assertIn(metric, research_evaluations[0]["metrics"])
            self.assertTrue(store.run_benchmark_path.exists())
            self.assertTrue(store.decision_dag_path.exists())
            self.assertTrue(store.agent_timeline_path.exists())
            self.assertTrue(store.agent_timeline_svg_path.exists())
            self.assertTrue(store.score_improvement_path.exists())
            self.assertTrue(store.score_improvement_svg_path.exists())
            self.assertTrue((store.root / "run_benchmark_summary.json").exists())
            self.assertIn("<promise>COMPLETE</promise>", store.progress_path.read_text(encoding="utf-8"))

    def test_loop_mode_can_route_to_optimize_with_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(
                    retriever="local",
                    max_loop_iterations=2,
                    task_mode="optimize",
                    evaluator_name="length_score",
                    include_debugger=False,
                    echo_progress=False,
                ),
            )
            run, store = asyncio.run(orchestrator.run("Optimize a tiny scoring function"))

            self.assertEqual(run.status, "completed")
            self.assertEqual(run.task_mode, "optimize")
            self.assertEqual(run.product_agent, "optimize")
            self.assertEqual(store.list("task_ingestion_decisions")[0]["selected_mode"], "optimize")
            self.assertEqual(store.list("task_ingestion_decisions")[0]["product_agent"], "optimize")
            self.assertTrue(all(row["inner_loop"] == "optimize" for row in store.list("variant_evaluations")))
            self.assertTrue(all(task["passes"] for task in store.list("loop_tasks")))
            self.assertTrue(store.optimal_code_path.exists())
            optimization_result = json.loads(store.optimization_result_path.read_text(encoding="utf-8"))
            self.assertEqual(optimization_result["optimal_code_path"], str(store.optimal_code_path))
            optimizer_trace = json.loads((store.root / "optimizer_trace.json").read_text(encoding="utf-8"))
            self.assertTrue(optimizer_trace)
            self.assertTrue((store.root / "optimizer_flow.mmd").exists())
            self.assertTrue((store.root / "optimizer_flow.png").exists())
            self.assertTrue(store.optimizer_agent_summary_path.exists())
            self.assertTrue(store.role_trajectory_contract_path.exists())
            self.assertTrue(store.champion_tree_graph_path.exists())
            self.assertTrue(store.champion_tree_svg_path.exists())
            self.assertTrue(store.champion_tree_mermaid_path.exists())
            self.assertIn("Optimizer Trace", store.run_benchmark_path.read_text(encoding="utf-8"))
            self.assertIn("Optimizer Agent Controller", store.run_benchmark_path.read_text(encoding="utf-8"))
            self.assertIn("Optimization Controller", store.role_trajectory_contract_path.read_text(encoding="utf-8"))

    def test_optimize_query_mode_feeds_optimizer_with_evaluator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(
                    retriever="local",
                    max_loop_iterations=1,
                    task_mode="optimize_query",
                    evaluator_name="length_score",
                    include_debugger=False,
                    echo_progress=False,
                ),
            )
            run, store = asyncio.run(orchestrator.run("Research optimization strategies for a tiny scoring benchmark"))

            seed_context = json.loads(store.optimizer_seed_context_path.read_text(encoding="utf-8"))
            inner_loops = {row["inner_loop"] for row in store.list("variant_evaluations")}
            run_state = json.loads(store.run_state_path.read_text(encoding="utf-8"))

            self.assertEqual(run.task_mode, "optimize_query")
            self.assertEqual(run.product_agent, "optimize")
            self.assertTrue(seed_context["has_evaluator"])
            self.assertIn("optimize_query", inner_loops)
            self.assertIn("optimize", inner_loops)
            query_evaluations = [row for row in store.list("variant_evaluations") if row["inner_loop"] == "optimize_query"]
            self.assertTrue(all("novelty" in row["metrics"] for row in query_evaluations))
            self.assertTrue(all("implementability" in row["metrics"] for row in query_evaluations))
            self.assertTrue(all("evaluator_relevance" in row["metrics"] for row in query_evaluations))
            query_variants = [row for row in store.list("variants") if row["kind"] == "query"]
            self.assertTrue(any(row["metadata"].get("evaluator_name") == "length_score" for row in query_variants))
            self.assertIn("optimizer_seed_context", run_state["artifacts"])
            self.assertTrue(any(task["params"].get("runtime_phase") == "seed_context" for task in run_state["observed_actions"]))

    def test_optimize_query_mode_without_evaluator_skips_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(
                    retriever="local",
                    max_loop_iterations=1,
                    task_mode="optimize_query",
                    include_debugger=False,
                    echo_progress=False,
                ),
            )
            run, store = asyncio.run(orchestrator.run("Research optimization strategies for a tiny scoring benchmark"))

            seed_context = json.loads(store.optimizer_seed_context_path.read_text(encoding="utf-8"))
            inner_loops = {row["inner_loop"] for row in store.list("variant_evaluations")}
            tasks = store.list("loop_tasks")

            self.assertEqual(run.task_mode, "optimize_query")
            self.assertEqual(run.product_agent, "optimize")
            self.assertFalse(seed_context["has_evaluator"])
            self.assertIn("optimize_query", inner_loops)
            self.assertNotIn("optimize", inner_loops)
            self.assertTrue(any(task["status"] == "skipped" for task in tasks))

    def test_prediction_market_evaluator_rewards_adaptive_strategy(self) -> None:
        static_ladder = "Static ladder around midpoint with size=12 spread=2 and no inventory controls."
        adaptive_guarded = (
            "Adaptive fair value estimate from fills and competitor midpoint, CancelAll after repeated "
            "loss-making fills or jump volatility, size=5 spread=4 inventory limit=90 with position controls."
        )

        self.assertGreater(prediction_market_score(adaptive_guarded), prediction_market_score(static_ladder))

    def test_research_loop_falls_back_to_local_when_live_retriever_fails(self) -> None:
        class FailingSearch:
            tool_name = "failing_search"

            def search(self, query: str, limit: int = 4):
                raise RuntimeError("rate limited")

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory), echo_progress=False)

            def search_factory(name: str):
                if name == "local":
                    return LocalCorpusSearch(Path("examples/corpus/research_corpus.json"))
                return FailingSearch()

            loop = ResearchLoop("run_test", search_factory)
            variant = Variant(
                run_id="run_test",
                outer_iteration=1,
                kind="query",
                payload="prediction market stale arbitrageur retail flow",
                parent_ids=[],
                metadata={"retriever": "arxiv", "limit": 8},
            )

            result = asyncio.run(loop.evaluate([variant], store))

            self.assertEqual(len(result.ranked_evaluations), 1)
            self.assertGreater(len(store.list("sources")), 0)
            self.assertGreater(len(store.list("failed_paths")), 0)
            self.assertEqual(store.list("variant_evaluations")[0]["metrics"]["fallback_used"], 1.0)

    def test_retriever_rate_limits_and_empty_results_fall_back_to_local(self) -> None:
        import urllib.error

        class FakeBackend:
            def __init__(self, name: str):
                self.tool_name = name

            def search(self, _query: str, _limit: int):
                if self.tool_name in {"semantic_scholar", "arxiv"}:
                    raise urllib.error.HTTPError("https://example.test", 429, "Too Many Requests", None, None)
                if self.tool_name in {"openalex", "wikipedia"}:
                    return []
                return [
                    (
                        CorpusDocument(
                            url="local://rate-limit-fallback",
                            title="Local fallback evidence",
                            author="Fixture",
                            date="2026",
                            source_type="paper",
                            summary="Local fallback evidence keeps a rate-limited run from becoming empty.",
                            claims=["Local fallback evidence supports continuity after live API rate limits."],
                            tags=["fallback"],
                            credibility_score=0.6,
                        ),
                        0.5,
                    )
                ]

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory), echo_progress=False)
            loop = ResearchLoop("run_test", lambda name: FakeBackend(name))
            variant = Variant(
                run_id="run_test",
                outer_iteration=1,
                kind="query",
                payload="agent reasoning benchmark literature",
                parent_ids=[],
                metadata={},
            )

            backend, results, notes = asyncio.run(loop._search_with_fallback("semantic_scholar", variant, 8, store))

            self.assertEqual(backend.tool_name, "local")
            self.assertEqual(len(results), 1)
            self.assertTrue(any("semantic_scholar failed" in note for note in notes))
            self.assertTrue(any("openalex fallback used" in note for note in notes))
            self.assertTrue(any("local fallback used" in note for note in notes))

    def test_optimize_query_prediction_market_challenge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(
                    retriever="local",
                    max_loop_iterations=1,
                    task_mode="optimize_query",
                    evaluator_name="prediction_market",
                    include_debugger=False,
                    echo_progress=False,
                ),
            )
            run, store = asyncio.run(
                orchestrator.run(
                    "Research approaches for the prediction market challenge: adaptive passive market making against stale quote arbitrage and retail flow"
                )
            )

            seed_context = json.loads(store.optimizer_seed_context_path.read_text(encoding="utf-8"))
            progress = store.progress_path.read_text(encoding="utf-8")
            inner_loops = {row["inner_loop"] for row in store.list("variant_evaluations")}

            self.assertEqual(run.task_mode, "optimize_query")
            self.assertEqual(run.product_agent, "challenge")
            self.assertTrue(seed_context["has_evaluator"])
            self.assertIn("optimize_query", inner_loops)
            self.assertIn("optimize", inner_loops)
            self.assertIn("prediction_market", progress)
            self.assertTrue(store.optimization_result_path.exists())
            optimization_result = json.loads(store.optimization_result_path.read_text(encoding="utf-8"))
            self.assertEqual(optimization_result["objective_direction"], "maximize")
            self.assertEqual(optimization_result["objective_name"], "prediction_market_mean_edge")
            score_source = optimization_result["official_result"]["score_source"]
            measured = optimization_result["official_result"]["measured"]
            self.assertIn(score_source, {"upstream_repo_missing", "official_sandbox_failed", "official_scorer_json_error", "official_scorer_no_successes", "upstream_orderbook_pm_challenge", "unmeasured_official_scorer_unavailable"})
            # measured must be truthful: True iff the upstream runner was used.
            if score_source == "upstream_orderbook_pm_challenge":
                self.assertTrue(measured)
                self.assertTrue(optimization_result["official_result"]["score_eligible"])
                self.assertTrue(store.optimized_candidate_path.exists())
                self.assertTrue(store.optimal_code_path.exists())
                self.assertTrue(store.solution_path.exists())
                self.assertIn("class Strategy", store.solution_path.read_text(encoding="utf-8"))
                self.assertIn("class Strategy", store.optimal_code_path.read_text(encoding="utf-8"))
                self.assertEqual(optimization_result["optimal_code_path"], str(store.optimal_code_path))
            else:
                self.assertFalse(measured)
                self.assertFalse(optimization_result["official_result"]["score_eligible"])
                self.assertFalse(store.optimized_candidate_path.exists())
                self.assertFalse(store.optimal_code_path.exists())
                self.assertFalse(store.solution_path.exists())
            candidate_path = optimization_result["official_result"].get("candidate_path")
            if candidate_path:
                self.assertTrue(Path(candidate_path).exists())
            self.assertTrue(any("Prediction Market" in source["title"] for source in store.list("sources")))
            run_state = json.loads(store.run_state_path.read_text(encoding="utf-8"))
            self.assertEqual(run_state["product_agent"], "challenge")
            self.assertEqual(run_state["agent_harness"]["runtime_mode"], "optimize_query")

    def test_prediction_market_optimizer_reuses_query_ids_as_stable_strategy_lanes(self) -> None:
        def fake_official(_candidate_path: Path) -> dict[str, object]:
            return {
                "mean_edge": 1.0,
                "score_eligible": True,
                "official_measured": True,
                "score_source": "upstream_orderbook_pm_challenge",
                "success_count": 1,
                "failure_count": 0,
                "actions_seen": ["PlaceOrder"],
                "simulations": 1,
            }

        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            store = ArtifactStore(run_root)
            llm = LLMClient(provider="local")
            llm.provider = "multi"
            llm.model = "all-configured"
            llm.model_pool = [("local", "lane-a"), ("local", "lane-b")]
            outer = EvolutionaryOuterLoop(
                run_id="run_stable_lanes",
                goal="prediction market challenge",
                task_mode="optimize_query",
                source_strategy=[],
                search_factory=lambda _name: LocalCorpusSearch(Path("examples/corpus/research_corpus.json")),
                evaluator_name="prediction_market",
                llm=llm,
                max_outer_iterations=2,
                population_size=2,
            )
            parents = [
                Variant(
                    run_id="run_stable_lanes",
                    outer_iteration=0,
                    kind="query",
                    payload="query lane a",
                    parent_ids=[],
                    metadata={"query_variant_id": "query_a"},
                    id="query_a",
                ),
                Variant(
                    run_id="run_stable_lanes",
                    outer_iteration=0,
                    kind="query",
                    payload="query lane b",
                    parent_ids=[],
                    metadata={"query_variant_id": "query_b"},
                    id="query_b",
                ),
            ]

            with unittest.mock.patch("research_harness.loops._run_prediction_market_official", fake_official):
                asyncio.run(outer._run_prediction_market_optimizer(store, parents, {"summary": "seed"}))

            code_variants = [row for row in store.list("variants") if row["kind"] == "code"]
            rounds = store.list("evolution_rounds")

            self.assertEqual([round_row["variant_ids"] for round_row in rounds], [["query_a", "query_b"], ["query_a", "query_b"]])
            self.assertEqual({row["id"] for row in code_variants}, {"query_a", "query_b"})
            self.assertTrue(all(row["metadata"].get("stable_strategy_lane") for row in code_variants))
            self.assertTrue((run_root / "candidates" / "query_a.py").exists())
            self.assertTrue((run_root / "candidates" / "query_b.py").exists())
            self.assertFalse(any((run_root / "candidates").glob("round_*_query_*.py")))

    def test_prediction_market_report_filters_unrelated_sources_for_typos(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(
                    retriever="local",
                    max_loop_iterations=1,
                    task_mode="optimize",
                    evaluator_name="prediction_market",
                    include_debugger=False,
                    echo_progress=False,
                    llm_provider="local",
                ),
            )
            run, store = asyncio.run(orchestrator.run("optimizing the predictionm arket mm'ing, get to 10$"))

            self.assertEqual(run.task_mode, "optimize_query")
            self.assertEqual(run.product_agent, "challenge")
            result = json.loads(store.optimization_result_path.read_text(encoding="utf-8"))
            self.assertFalse(result["official_result"]["measured"])
            self.assertFalse(result["official_result"]["score_eligible"])
            self.assertFalse(store.report_path.exists())
            progress = store.progress_path.read_text(encoding="utf-8")
            self.assertIn("prediction_market", progress)

    def test_research_report_filters_placeholder_and_challenge_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            run = RunRecord(
                id="run_ai_agents",
                user_goal="find me research/data which proves that ai agents will be everywhere and white collar work will be done by research agents",
                task_type="open_ended",
                task_mode="research",
                product_agent="research",
                harness_config_id="test",
                prompt_versions={},
                harness_config_snapshot={},
            )
            good = store.add_source(
                Source(
                    url="https://doi.org/10.1007/s11704-024-40231-1",
                    title="A survey on large language model based autonomous agents",
                    author="Frontiers of Computer Science",
                    date="2024",
                    source_type="paper",
                    summary="Large language model based autonomous agents use tools, planning, and reasoning.",
                    relevance_score=0.92,
                    credibility_score=0.9,
                    evidence_sections={"abstract": "LLM based autonomous agents can plan, use tools, and perform multi-step tasks."},
                )
            )
            placeholder = store.add_source(
                Source(
                    url="https://example.org/multi-agent-review-quality-2025",
                    title="Parallel Agent Review Improves Evidence Recall In Literature Tasks",
                    author="Demo Corpus",
                    date="2025",
                    source_type="demo",
                    summary="Synthetic fixture source that should not appear beside real literature.",
                    relevance_score=0.5,
                    credibility_score=0.2,
                )
            )
            challenge = store.add_source(
                Source(
                    url="challenges/prediction_market/spec.md",
                    title="Prediction Market Strategy Design Notes",
                    author="Challenge",
                    date="2026",
                    source_type="local",
                    summary="Prediction market challenge notes.",
                    relevance_score=0.4,
                    credibility_score=0.3,
                )
            )
            store.add_claim(
                Claim(
                    text="LLM autonomous agents are defined by planning, tool use, and multi-step task execution.",
                    source_ids=[good.id],
                    confidence=0.85,
                    support_level="strong",
                    created_by_agent="test",
                    run_id=run.id,
                )
            )
            store.add_claim(
                Claim(
                    text="This synthetic demo article says reviewer agents improve recall.",
                    source_ids=[placeholder.id],
                    confidence=0.6,
                    support_level="medium",
                    created_by_agent="test",
                    run_id=run.id,
                )
            )
            store.add_claim(
                Claim(
                    text="Prediction market orderbook strategies need inventory controls.",
                    source_ids=[challenge.id],
                    confidence=0.6,
                    support_level="medium",
                    created_by_agent="test",
                    run_id=run.id,
                )
            )

            agent = SynthesisAgent(
                name="test_synthesis",
                role="synthesis_agent",
                prompt_template="test",
                llm=LLMClient(provider="local"),
            )
            asyncio.run(agent.run(run, store))

            report = store.report_path.read_text(encoding="utf-8")
            tex = store.report_tex_path.read_text(encoding="utf-8")
            combined = report + "\n" + tex
            self.assertIn("Key Takeaways", tex)
            self.assertIn(good.url, combined)
            self.assertNotIn("example.org", combined)
            self.assertNotIn("Prediction Market Strategy Design Notes", combined)
            self.assertNotIn("challenges/prediction_market", combined)

            task = EvalTask(
                id="pure_research",
                name="pure research",
                prompt=run.user_goal,
                task_mode="research",
                success_criteria=[],
                grader_ids=["report_no_fabricated_sources"],
            )
            result = default_graders()["report_no_fabricated_sources"].grade(task, store)
            self.assertTrue(result.passed, result.assertions)

    def test_local_corpus_sources_are_labeled_as_fixtures_not_external_literature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            run = RunRecord(
                id="run_local_fixture_report",
                user_goal="Research how multi-agent systems improve automated literature review quality.",
                task_type="open_ended",
                task_mode="research",
                product_agent="research",
                harness_config_id="test",
                prompt_versions={},
                harness_config_snapshot={},
            )
            source = store.add_source(
                Source(
                    url="https://example.org/multi-agent-review-quality-2025",
                    title="Parallel Agent Review Improves Evidence Recall In Literature Tasks",
                    author="A. Rivera and S. Chen",
                    date="2025-02-14",
                    source_type="paper",
                    summary="Bundled deterministic fixture.",
                    relevance_score=0.7,
                    credibility_score=0.5,
                )
            )
            store.add_claim(
                Claim(
                    text="Parallel query framings can increase recall in a deterministic fixture.",
                    source_ids=[source.id],
                    confidence=0.7,
                    support_level="medium",
                    created_by_agent="test",
                    run_id=run.id,
                )
            )

            asyncio.run(
                SynthesisAgent(
                    name="test_synthesis",
                    role="synthesis_agent",
                    prompt_template="test",
                    llm=LLMClient(provider="local"),
                ).run(run, store)
            )

            report = store.report_path.read_text(encoding="utf-8")
            tex = store.report_tex_path.read_text(encoding="utf-8")

        self.assertIn("No externally verifiable sources were retained", report)
        self.assertIn("Local Corpus Fixtures", report)
        self.assertIn("not external sources", report)
        self.assertNotIn("](https://example.org", report)
        self.assertNotIn(r"\url{https://example.org", tex)

    def test_kernel_challenge_report_does_not_import_prediction_market_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run_kernel")
            run = RunRecord(
                id="run_kernel",
                user_goal="High-performance GPU kernel generation is increasingly important for large-model systems.",
                task_type="bounded",
                task_mode="research",
                product_agent="challenge",
                harness_config_id="test",
                prompt_versions={},
                harness_config_snapshot={"evaluator_name": "length_score"},
            )
            gpu_source = store.add_source(
                Source(
                    url="https://doi.org/10.1145/gpu-kernel-generation",
                    title="GPU Kernel Generation For Large Models",
                    author="Kernel Systems Lab",
                    date="2026",
                    source_type="paper",
                    summary="High-performance GPU kernels improve latency and throughput for large model inference.",
                    relevance_score=0.94,
                    credibility_score=0.9,
                    evidence_sections={"abstract": "GPU kernel generation targets latency, throughput, and correctness for large-model systems."},
                )
            )
            pm_source = store.add_source(
                Source(
                    url="https://github.com/danrobinson/prediction-market-challenge",
                    title="Orderbook Prediction Market Challenge",
                    author="Dan Robinson",
                    date="2026-05-04",
                    source_type="optimization_challenge",
                    summary="A binary prediction-market optimization challenge with retail flow and stale quotes.",
                    relevance_score=0.8,
                    credibility_score=0.88,
                )
            )
            gpu_claim = store.add_claim(
                Claim(
                    text="GPU kernel generation should be evaluated on latency, throughput, and correctness.",
                    source_ids=[gpu_source.id],
                    confidence=0.82,
                    support_level="strong",
                    created_by_agent="test",
                    run_id=run.id,
                )
            )
            pm_claim = store.add_claim(
                Claim(
                    text="Positive edge comes from capturing spread against uninformed retail order flow.",
                    source_ids=[pm_source.id],
                    confidence=0.68,
                    support_level="moderate",
                    created_by_agent="test",
                    run_id=run.id,
                )
            )
            store.add_hypothesis(
                Hypothesis(
                    text="Kernel optimization should focus on measured latency and correctness.",
                    supporting_claim_ids=[gpu_claim.id],
                    contradicting_claim_ids=[],
                    confidence=0.7,
                    novelty_score=0.4,
                    testability_score=0.9,
                    next_experiment="Run the generated kernel against benchmark inputs.",
                    run_id=run.id,
                )
            )
            store.add_hypothesis(
                Hypothesis(
                    text="Prediction-market retail flow should guide the final strategy.",
                    supporting_claim_ids=[pm_claim.id],
                    contradicting_claim_ids=[],
                    confidence=0.66,
                    novelty_score=0.4,
                    testability_score=0.8,
                    next_experiment="Backtest against orderbook fills.",
                    run_id=run.id,
                )
            )

            asyncio.run(
                SynthesisAgent(
                    name="test_synthesis",
                    role="synthesis_agent",
                    prompt_template="test",
                    llm=LLMClient(provider="local"),
                ).run(run, store)
            )

            combined = (
                store.report_path.read_text(encoding="utf-8")
                + "\n"
                + store.report_tex_path.read_text(encoding="utf-8")
            ).lower()

        self.assertIn("gpu kernel generation", combined)
        self.assertIn("latency, throughput, and correctness", combined)
        self.assertNotIn("prediction-market", combined)
        self.assertNotIn("prediction market", combined)
        self.assertNotIn("uninformed retail order flow", combined)
        self.assertNotIn("sources: none retained", combined)

    def test_research_key_takeaways_do_not_import_unrelated_agentic_storyline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run_entropy")
            run = RunRecord(
                id="run_entropy",
                user_goal='When we say "introduce entropy in AI agent systems", do we mean introduce varied information?',
                task_type="open_ended",
                harness_config_id="test-config",
                prompt_versions={},
                harness_config_snapshot={},
                task_mode="research",
                product_agent="research",
            )
            store.add_run(run)
            source = store.add_source(
                Source(
                    url="https://example.edu/entropy",
                    title="Entropy And Exploration In Agent Systems",
                    author="Ada",
                    date="2026",
                    source_type="paper",
                    summary="Entropy can describe exploration pressure or diversity in sampled information.",
                    relevance_score=0.9,
                    credibility_score=0.8,
                )
            )
            claim = store.add_claim(
                Claim(
                    text="Entropy can operationalize diversity in candidate information when tied to a task objective.",
                    source_ids=[source.id],
                    confidence=0.76,
                    support_level="strong",
                    created_by_agent="test",
                    run_id=run.id,
                )
            )
            store.add_hypothesis(
                Hypothesis(
                    text="claim about entropy: varied information may improve exploration when it remains goal-relevant.",
                    supporting_claim_ids=[claim.id],
                    contradicting_claim_ids=[],
                    confidence=0.66,
                    novelty_score=0.5,
                    testability_score=0.7,
                    next_experiment="Compare retrieval diversity against answer quality.",
                )
            )

            agent = SynthesisAgent(
                name="test_synthesis",
                role="synthesis_agent",
                prompt_template="test",
                llm=LLMClient(provider="local"),
            )
            asyncio.run(agent.run(run, store))

            report = store.report_path.read_text(encoding="utf-8").lower()
            self.assertIn("entropy", report)
            for leaked in ["static chat", "workflow execution", "document-heavy", "white-collar", "labor-impact"]:
                self.assertNotIn(leaked, report)

    def test_research_report_filters_broad_technology_false_positives(self) -> None:
        prompt = (
            "What is the current evidence that enterprise AI agent adoption follows the historical SaaS "
            "internalization-then-outsourcing pattern? Find literature on proprietary agent harnesses, "
            "multi-agent self-modification, autonomous trading agents, evolutionary computation, and LLM self-improvement."
        )
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            run = RunRecord(
                id="run_enterprise_agents",
                user_goal=prompt,
                task_type="open_ended",
                task_mode="research",
                product_agent="research",
                harness_config_id="test",
                prompt_versions={},
                harness_config_snapshot={},
            )
            good_sources = [
                ("https://doi.org/10.1007/s11704-024-40231-1", "A survey on large language model based autonomous agents", "LLM autonomous agents use planning, tool use, and multi-agent coordination."),
                ("https://arxiv.org/abs/2310.11511", "Self-Refine: Iterative Refinement with Self-Feedback", "LLMs can improve outputs through self-feedback and iterative refinement."),
                ("https://doi.org/10.1109/TEVC.2002.804320", "Evolutionary computation in dynamic optimization", "Evolutionary algorithms adapt strategies through selection and mutation."),
            ]
            bad_sources = [
                ("https://doi.org/10.1109/access.2019.2932609", "Internet-of-Things (IoT)-Based Smart Agriculture", "Wireless sensors and IoT platforms support irrigation, crop surveillance, and harvesting."),
                ("https://doi.org/10.1109/access.2019.2953499", "A Survey on Digital Twin", "Digital twin platforms model industrial assets and applications."),
                ("https://doi.org/10.1186/s40537-020-00369-8", "CatBoost for big data", "Gradient boosting supports big data classification across domains."),
                ("https://doi.org/10.1109/ojcoms.2021.3071496", "Survey on 6G Frontiers", "6G communications enable future wireless applications and services."),
            ]
            for url, title, summary in good_sources + bad_sources:
                source = store.add_source(
                    Source(
                        url=url,
                        title=title,
                        author="Researcher",
                        date="2024",
                        source_type="paper",
                        summary=summary,
                        relevance_score=0.8,
                        credibility_score=0.8,
                        evidence_sections={"abstract": summary},
                    )
                )
                store.add_claim(
                    Claim(
                        text=summary,
                        source_ids=[source.id],
                        confidence=0.8,
                        support_level="strong",
                        created_by_agent="test",
                        run_id=run.id,
                    )
                )

            asyncio.run(
                SynthesisAgent(
                    name="test_synthesis",
                    role="synthesis_agent",
                    prompt_template="test",
                    llm=LLMClient(provider="local"),
                ).run(run, store)
            )

            tex = store.report_tex_path.read_text(encoding="utf-8")
            self.assertIn("large language model based autonomous agents", tex)
            self.assertIn("Self-Refine", tex)
            self.assertNotIn("Smart Agriculture", tex)
            self.assertNotIn("Digital Twin", tex)
            self.assertNotIn("CatBoost", tex)
            self.assertNotIn("6G Frontiers", tex)

    def test_prediction_market_dont_stop_profit_target_keeps_action_incomplete_until_met(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=Path(directory),
                config=HarnessConfig(
                    retriever="local",
                    max_loop_iterations=3,
                    task_mode="optimize_query",
                    evaluator_name="prediction_market",
                    include_debugger=False,
                    echo_progress=False,
                ),
            )
            run, store = asyncio.run(
                orchestrator.run(
                    "Get to $10 profit in the prediction market challenge, don't stop until you're profitable."
                )
            )

            rounds = store.list("evolution_rounds")
            optimize_rounds = [row for row in rounds if row["mode"] == "optimize"]
            query_rounds = [row for row in rounds if row["mode"] == "optimize_query"]
            run_state = json.loads(store.run_state_path.read_text(encoding="utf-8"))
            optimization_result = json.loads(store.optimization_result_path.read_text(encoding="utf-8"))
            optimizer_task = next(task for task in run_state["observed_actions"] if task["title"] == "Run optimizer variants from query seed context")

            self.assertEqual(run.product_agent, "challenge")
            self.assertGreaterEqual(len(query_rounds), 2)
            self.assertEqual(run_state["objective"]["kind"], "profit_usd")
            self.assertEqual(run_state["objective"]["target"], 10.0)
            self.assertEqual(optimization_result["objective_target"]["target"], 10.0)
            if optimization_result["objective_target"]["met"]:
                self.assertEqual(run.status, "completed")
                self.assertEqual(optimizer_task["status"], "passed")
                self.assertTrue(optimizer_task["passes"])
                self.assertTrue(run_state["objective"]["met"])
            else:
                self.assertEqual(run.status, "failed")
                self.assertEqual(len(optimize_rounds), 3)
                self.assertFalse(run_state["objective"]["met"])
                self.assertEqual(optimizer_task["status"], "failed")
                self.assertFalse(optimizer_task["passes"])
                downstream = [
                    task for task in run_state["observed_actions"]
                    if task["title"] in {"Critique ranked query and optimizer results", "Synthesize optimize-query run report"}
                ]
                self.assertTrue(downstream)
                self.assertTrue(all(task["status"] == "pending" for task in downstream))

    def test_goal_slug(self) -> None:
        self.assertEqual(
            goal_slug("Please research new agent paradigms on arxive and determine workplace trends"),
            "new-agent-paradigms-arxive-determine-workplace-trends",
        )


class BenchmarkTest(unittest.TestCase):
    def test_benchmark_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            orchestrator = Orchestrator(
                corpus_path=Path("examples/corpus/research_corpus.json"),
                output_root=root / "outputs",
                config=HarnessConfig(mode="standard", retriever="local", echo_progress=False),
            )
            asyncio.run(
                orchestrator.run(
                    "Research how multi-agent systems improve automated literature review quality",
                    mode="standard",
                )
            )

            runs = collect_runs(root / "outputs")
            write_outputs(runs, root / "benchmarks")

            self.assertEqual(len(runs), 1)
            self.assertTrue((root / "benchmarks" / "index.html").exists())
            self.assertTrue((root / "benchmarks" / "summary.json").exists())
            self.assertTrue((root / "benchmarks" / "charts" / "artifact_counts.svg").exists())


class EvaluationHarnessTest(unittest.TestCase):
    def test_core_eval_suite_defines_all_run_types(self) -> None:
        suite = default_eval_suite()
        task_ids = {task.id for task in suite.tasks}

        self.assertIn("research_open_ended", task_ids)
        self.assertIn("optimize_direct", task_ids)
        self.assertIn("optimize_query_seeded", task_ids)
        self.assertIn("challenge_prediction_market", task_ids)
        self.assertTrue(all(task.success_criteria for task in suite.tasks))
        self.assertTrue(all(task.grader_ids for task in suite.tasks))
        self.assertTrue(all("run_actions_executed" in task.grader_ids for task in suite.tasks))
        research_task = next(task for task in suite.tasks if task.id == "research_open_ended")
        self.assertIn("run_actions_executed_deterministic", research_task.grader_ids)
        self.assertIn("llm_research_quality_challenger", research_task.grader_ids)
        self.assertIn("llm_hypothesis_novelty_challenger", research_task.grader_ids)
        self.assertIn("llm_open_ended_judgment_challenger", research_task.grader_ids)
        self.assertIn("literature_section_evidence", research_task.grader_ids)
        self.assertIn("hypothesis_evidence_matrix", research_task.grader_ids)
        with tempfile.TemporaryDirectory() as directory:
            registry = EvaluationHarness(output_root=Path(directory)).grader_registry
            self.assertEqual(registry["run_actions_executed_deterministic"].grader_type, "code")
            self.assertEqual(registry["literature_section_evidence"].grader_type, "code")
            self.assertEqual(registry["hypothesis_evidence_matrix"].grader_type, "code")
            self.assertEqual(registry["llm_research_quality_challenger"].grader_type, "model")

    def test_edge_eval_suite_defines_failure_prone_cases(self) -> None:
        suite = edge_eval_suite()
        task_ids = {task.id for task in suite.tasks}

        self.assertIn("optimize_query_missing_evaluator_skips_optimizer", task_ids)
        self.assertIn("prediction_market_outputs_are_contained", task_ids)
        self.assertIn("prediction_market_unmeasured_official_status", task_ids)
        self.assertIn("challenge_prediction_market_official_unavailable_records_unmeasured", task_ids)
        self.assertIn("challenge_prediction_market_candidate_files_only_in_outputs", task_ids)
        self.assertIn("parallel_trials_do_not_share_tmp_or_outputs", task_ids)
        self.assertIn("challenge_prediction_market_no_repo_root_strategy_files", task_ids)
        self.assertIn("research_should_not_oversearch", task_ids)
        self.assertIn("nested_loop_multiple_iterations_no_regression", task_ids)
        self.assertIn("stuck_loop_triggers_literature_search", task_ids)
        self.assertIn("trajectory_match_modes_are_enforced", task_ids)
        self.assertIn("optimize_runs_start_with_literature_grounding", task_ids)
        self.assertTrue(any("trajectory_modes" in task.grader_ids for task in suite.tasks))
        self.assertTrue(any("prediction_market_artifact_containment" in task.grader_ids for task in suite.tasks))
        self.assertTrue(any("parallel_trial_isolation" in task.grader_ids for task in suite.tasks))
        self.assertTrue(any("research_search_budget" in task.grader_ids for task in suite.tasks))
        self.assertTrue(any("trajectory_graph_artifact" in task.grader_ids for task in suite.tasks))
        self.assertTrue(any("literature_refresh_on_stuck" in task.grader_ids for task in suite.tasks))
        self.assertTrue(any("literature_grounding_present" in task.grader_ids for task in suite.tasks))
        self.assertTrue(any("trajectory_match_modes" in task.grader_ids for task in suite.tasks))
        self.assertTrue(any("graph_trajectory_match" in task.grader_ids for task in suite.tasks))
        challenge_task = next(task for task in default_eval_suite().tasks if task.id == "challenge_prediction_market")
        self.assertIn("prediction_market_agentic_optimizer", challenge_task.grader_ids)

    def test_preflight_eval_suite_defines_source_diversity_gate(self) -> None:
        suite = preflight_eval_suite()
        task_ids = {task.id for task in suite.tasks}

        self.assertIn("research_uses_at_least_four_source_families", task_ids)
        research_task = next(task for task in suite.tasks if task.id == "research_uses_at_least_four_source_families")
        self.assertEqual(research_task.retriever, "auto")
        self.assertIn("research_source_diversity", research_task.grader_ids)
        self.assertEqual(research_task.metadata["min_distinct_source_families"], 4)

    def test_trajectory_optimizer_flow_marks_post_round_entropy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            source = store.add_source(
                Source(
                    url="memory://entropy/round-1",
                    title="Fresh Bandit Literature",
                    author="research-harness",
                    date="2026-05-19",
                    source_type="paper",
                    summary="Fresh exploration literature.",
                    relevance_score=0.8,
                    credibility_score=0.8,
                    retrieved_at="2026-05-19T10:00:05+00:00",
                )
            )
            store.add_claim(
                Claim(
                    text="Literature grounding (optimizer_entropy_after_round_1) found: bandit exploration can add useful entropy.",
                    source_ids=[source.id],
                    confidence=0.8,
                    support_level="retrieved",
                    created_by_agent="literature_grounding_policy",
                    run_id="run_flow",
                )
            )
            store.add_evolution_round(
                EvolutionRound(
                    run_id="run_flow",
                    outer_iteration=1,
                    mode="optimize",
                    variant_ids=[],
                    best_variant_id=None,
                    best_score=0.4,
                    termination_signal="continue",
                    plateau_count=0,
                    completed_at="2026-05-19T10:00:00+00:00",
                )
            )

            flow = trajectory_optimizer_flow(store)

        self.assertIn("Post-round entropy introduced", flow)
        self.assertIn("optimizer_entropy_after_round_1", flow)
        self.assertIn("Fresh Bandit Literature", flow)

    def test_trajectory_optimizer_flow_flags_missing_post_round_entropy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            store.add_evolution_round(
                EvolutionRound(
                    run_id="run_flow_missing",
                    outer_iteration=1,
                    mode="optimize",
                    variant_ids=[],
                    best_variant_id=None,
                    best_score=0.4,
                    termination_signal="continue",
                    plateau_count=0,
                    completed_at="2026-05-19T10:00:00+00:00",
                )
            )

            flow = trajectory_optimizer_flow(store)

        self.assertIn("Missing post-round entropy", flow)
        self.assertIn("expected optimizer_entropy_after_round_1", flow)

    def test_eval_cli_can_select_specific_eval_ids(self) -> None:
        parser = build_eval_parser()
        args = parser.parse_args(["--suite", "preflight", "--eval", "research_uses_at_least_four_source_families"])
        suite = select_eval_tasks(preflight_eval_suite(), args.eval_ids)

        self.assertEqual([task.id for task in suite.tasks], ["research_uses_at_least_four_source_families"])

    def test_research_source_diversity_grader_requires_four_families(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            task = EvalTask(
                id="source_diversity",
                name="Source diversity",
                prompt="Research transformer efficiency",
                task_mode="research",
                success_criteria=[],
                metadata={"min_distinct_source_families": 4},
            )
            store.add_trace(
                AgentTrace(
                    run_id="run_test",
                    agent_name="literature_agent",
                    role="search_literature",
                    prompt="",
                    model="local",
                    tools_used=["openalex_api_search", "semantic_scholar_api_search"],
                    tool_calls=[
                        {"tool": "openalex_api_search", "query": "transformer efficiency", "results": 2},
                        {"tool": "semantic_scholar_api_search", "query": "transformer efficiency", "results": 2},
                        {"tool": "arxiv_api_search", "query": "transformer efficiency", "results": 2},
                        {"tool": "web_search", "query": "transformer efficiency benchmarks", "results": 2},
                    ],
                    token_usage=0,
                    runtime_ms=1,
                    status="completed",
                    errors=[],
                    output_summary="searched",
                )
            )
            for index, source_type in enumerate(["openalex_work", "semantic_scholar_paper", "arxiv_paper", "web_result"], start=1):
                store.add_source(
                    Source(
                        url=f"https://example.test/{index}",
                        title=f"Source {index}",
                        author="author",
                        date="2026",
                        source_type=source_type,
                        summary="summary",
                        relevance_score=0.9,
                        credibility_score=0.9,
                    )
                )

            result = default_graders()["research_source_diversity"].grade(task, store)

            self.assertTrue(result.passed)
            self.assertEqual(result.assertions[0]["actual"], 4)
            self.assertEqual(result.assertions[0]["families"], ["arxiv", "openalex", "semantic_scholar", "web"])

    def test_research_source_diversity_does_not_count_requested_failed_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            task = EvalTask(
                id="source_diversity",
                name="Source diversity",
                prompt="Research transformer efficiency",
                task_mode="research",
                success_criteria=[],
                metadata={"min_distinct_source_families": 4},
            )
            store.add_trace(
                AgentTrace(
                    run_id="run_test",
                    agent_name="research_eval",
                    role="research_variant_agent",
                    prompt="",
                    model="local",
                    tools_used=["local_corpus_search"],
                    tool_calls=[
                        {
                            "tool": "local_corpus_search",
                            "requested_tool": "github_repo_search",
                            "query": "transformer efficiency",
                            "results": 1,
                            "fallback_used": True,
                        }
                    ],
                    token_usage=0,
                    runtime_ms=1,
                    status="completed",
                    errors=[],
                    output_summary="github failed; local fallback used",
                )
            )
            store.add_source(
                Source(
                    url="https://example.test/local",
                    title="Local fallback source",
                    author="author",
                    date="2026",
                    source_type="paper",
                    summary="summary",
                    relevance_score=0.9,
                    credibility_score=0.9,
                )
            )

            result = default_graders()["research_source_diversity"].grade(task, store)

            self.assertFalse(result.passed)
            self.assertEqual(result.assertions[0]["families"], ["local"])
            self.assertEqual(result.assertions[1]["families"], ["local"])

    def test_native_trajectory_match_modes(self) -> None:
        actual = [
            {"type": "router", "name": "optimize"},
            {"type": "outer_loop", "name": "optimize"},
            {"type": "inner_loop", "name": "optimize"},
            {"type": "selection", "name": "variant"},
            {"type": "signal", "name": "score_plateau"},
            {"type": "outcome", "name": "completed"},
        ]
        reference = [
            {"type": "router", "name": "optimize"},
            {"type": "outer_loop", "name": "optimize"},
            {"type": "inner_loop", "name": "optimize"},
        ]

        self.assertTrue(trajectory_match(actual, reference, "strict")["passed"])
        self.assertTrue(trajectory_match(list(reversed(actual)), reference, "unordered")["passed"])
        self.assertTrue(trajectory_match(actual, reference + [{"type": "outcome", "name": "completed"}], "superset")["passed"])
        self.assertFalse(trajectory_match(actual, reference, "subset")["passed"])

    def test_graph_trajectory_match(self) -> None:
        graph = {"edges": [{"from": "prompt", "to": "router"}, {"from": "router", "to": "outer"}]}

        self.assertTrue(graph_trajectory_match(graph, [["prompt", "router"]])["passed"])
        self.assertFalse(graph_trajectory_match(graph, [["inner", "select"]])["passed"])

    def test_eval_harness_runs_prediction_market_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            suite = default_eval_suite()
            suite.tasks = [task for task in suite.tasks if task.id == "challenge_prediction_market"]
            summary = asyncio.run(
                EvaluationHarness(
                    corpus_path=Path("examples/corpus/research_corpus.json"),
                    output_root=Path(directory),
                ).run_suite(suite)
            )

            self.assertEqual(summary.trial_count, 1)
            self.assertEqual(summary.passed_trials, 0)
            self.assertLess(summary.aggregate_score, 0.8)
            self.assertTrue((Path(directory) / "core_summary.json").exists())
            trial = summary.trials[0]
            isolation = trial["isolation"]
            self.assertTrue(isolation["clean_start"])
            self.assertTrue(Path(isolation["trial_root"]).exists())
            self.assertTrue(Path(isolation["tmpdir"]).exists())
            self.assertIn("Orchestrator", isolation["production_agent_path"])
            graders = {result["grader_id"]: result for result in trial["grader_results"]}
            self.assertIn("optimization_code_artifact", graders)
            self.assertFalse(graders["optimization_code_artifact"]["passed"])
            self.assertIn("prediction_market_solution", graders)
            self.assertFalse(graders["prediction_market_solution"]["passed"])
            self.assertIn("isolation_clean_trial", graders)
            self.assertTrue(graders["isolation_clean_trial"]["passed"])

    def test_prediction_market_agentic_optimizer_grader_uses_real_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            task = EvalTask(
                id="pm_agentic",
                name="PM agentic optimizer",
                prompt="Optimize prediction market strategy",
                task_mode="optimize_query",
                evaluator_name="prediction_market",
                success_criteria=[],
            )
            step_payload = [
                {
                    "round_index": 2,
                    "reflection": "negative mean_edge stayed flat, so change the strategy mechanism instead of nudging parameters",
                    "literature_required": True,
                    "actions": [
                        {"tool": "read_champion", "status": "completed"},
                        {"tool": "read_failures", "status": "completed"},
                        {"tool": "read_evaluator_summary", "status": "completed"},
                        {"tool": "compare_variants", "status": "completed"},
                        {"tool": "fetch_literature", "status": "completed"},
                        {"tool": "propose_strategy", "status": "completed"},
                    ],
                }
            ]
            (store.root / "optimization_agent_steps.json").write_text(json.dumps(step_payload), encoding="utf-8")
            round_one = Variant(
                run_id="run_pm_agentic",
                outer_iteration=1,
                kind="code",
                payload="pm_strategy=wide_passive spread=12 size=0.2",
                parent_ids=[],
                metadata={},
            )
            round_two = Variant(
                run_id="run_pm_agentic",
                outer_iteration=2,
                kind="code",
                payload="pm_strategy=flow_gate optimizer_agent_next_mechanism='new flow toxicity gate' PlaceOrder filled_quantity",
                parent_ids=[round_one.id],
                metadata={},
            )
            store.add_variant(round_one)
            store.add_variant(round_two)
            store.add_variant_evaluation(
                VariantEvaluation(
                    run_id="run_pm_agentic",
                    variant_id=round_one.id,
                    inner_loop="optimize",
                    score=0.49,
                    metrics={"mean_edge": -0.2, "score_eligible": True},
                    judge_scores=[0.49],
                    summary="negative mean_edge",
                    passed=False,
                )
            )
            store.add_trace(
                AgentTrace(
                    run_id="run_pm_agentic",
                    agent_name="llm_propose_prediction_market_code:round_2",
                    role="llm_thinking",
                    prompt='{"optimizer_agent_context": {}, "literature_refresh_notes": ["paper claim"]}',
                    model="local",
                    tools_used=[],
                    tool_calls=[],
                    token_usage=0,
                    runtime_ms=1,
                    status="completed",
                    errors=[],
                    output_summary="proposed",
                )
            )

            result = default_graders()["prediction_market_agentic_optimizer"].grade(task, store)

        self.assertTrue(result.passed)

    def test_prediction_market_agentic_optimizer_grader_rejects_loop_only_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            task = EvalTask(
                id="pm_agentic_negative",
                name="PM agentic optimizer negative",
                prompt="Optimize prediction market strategy",
                task_mode="optimize_query",
                evaluator_name="prediction_market",
                success_criteria=[],
            )
            round_one = Variant(
                run_id="run_pm_loop_only",
                outer_iteration=1,
                kind="code",
                payload="pm_strategy=same spread=10 size=0.5 PlaceOrder",
                parent_ids=[],
                metadata={},
            )
            round_two = Variant(
                run_id="run_pm_loop_only",
                outer_iteration=2,
                kind="code",
                payload="pm_strategy=same spread=10 size=0.5 PlaceOrder",
                parent_ids=[round_one.id],
                metadata={},
            )
            store.add_variant(round_one)
            store.add_variant(round_two)
            store.add_variant_evaluation(
                VariantEvaluation(
                    run_id="run_pm_loop_only",
                    variant_id=round_one.id,
                    inner_loop="optimize",
                    score=0.49,
                    metrics={"mean_edge": -0.2, "score_eligible": True},
                    judge_scores=[0.49],
                    summary="negative mean_edge",
                    passed=False,
                )
            )

            result = default_graders()["prediction_market_agentic_optimizer"].grade(task, store)

        self.assertFalse(result.passed)
        failed = {assertion["check"] for assertion in result.assertions if not assertion["passed"]}
        self.assertIn("controller_steps_exist", failed)
        self.assertIn("consecutive_rounds_not_same_signature", failed)

    def test_eval_aggregation_modes(self) -> None:
        suite = default_eval_suite()
        task = suite.tasks[0]
        task.aggregation = "weighted"
        results = [
            GraderResult("a", "code", "exact", 1.0, True, 1.0, "pass", []),
            GraderResult("b", "model", "rubric", 0.5, False, 1.0, "partial", []),
        ]
        score, passed = aggregate_results(task, results)

        self.assertEqual(score, 0.75)
        self.assertFalse(passed)

    def test_model_style_graders_emit_dag_right_wrong_judgments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            store.progress_path.write_text("<promise>complete</promise>\n", encoding="utf-8")
            store.write_report(
                "# Summary\nFindings cite source evidence and claims. "
                "The synthesis includes caveats, limitations, confidence, and recommendations. "
                "This report discusses source quality and evidence across enough detail to be judged. " * 4
            )
            source = store.add_source(
                Source(
                    url="https://example.test/paper",
                    title="Grounded Research",
                    author="A. Reviewer",
                    date="2026-05-19",
                    source_type="paper",
                    summary="Evidence.",
                    relevance_score=0.9,
                    credibility_score=0.9,
                )
            )
            claim = store.add_claim(
                Claim(
                    text="Grounded claims are supported by sources.",
                    source_ids=[source.id],
                    confidence=0.9,
                    support_level="strong",
                    created_by_agent="test",
                    run_id="run",
                )
            )
            store.add_hypothesis(
                Hypothesis(
                    text="Independent retrieval checks improve evidence reliability over single-pass summaries.",
                    supporting_claim_ids=[claim.id],
                    contradicting_claim_ids=[],
                    confidence=0.8,
                    novelty_score=0.9,
                    testability_score=0.9,
                    next_experiment="Compare one-pass and multi-pass retrieval on citation accuracy.",
                    run_id="run",
                )
            )
            store.add_variant_evaluation(
                VariantEvaluation(
                    run_id="run",
                    variant_id="variant_research",
                    inner_loop="research",
                    score=0.9,
                    metrics={
                        "factual_accuracy": 0.9,
                        "citation_accuracy": 0.9,
                        "completeness": 0.9,
                        "source_quality": 0.9,
                        "tool_efficiency": 0.9,
                    },
                    judge_scores=[0.9],
                    summary="research metrics",
                    passed=True,
                )
            )
            task = EvalTask(
                id="dag_model",
                name="DAG model graders",
                prompt="Research grounded evidence reliability",
                task_mode="research",
                success_criteria=[],
            )
            graders = default_graders()

            results = [
                graders["model_report_rubric"].grade(task, store),
                graders["llm_research_quality_challenger"].grade(task, store),
                graders["llm_hypothesis_novelty_challenger"].grade(task, store),
                graders["llm_open_ended_judgment_challenger"].grade(task, store),
            ]

        for result in results:
            self.assertEqual(result.grader_type, "model")
            self.assertTrue(result.assertions)
            dag = result.assertions[0]
            self.assertEqual(dag["type"], "deep_acyclic_graph")
            self.assertIn("right_behaviors", dag)
            self.assertIn("wrong_behaviors", dag)
            self.assertIn("nodes", dag)
            self.assertGreater(len(dag["right_behaviors"]), 0)
            self.assertIn("Right:", result.summary)
            self.assertIn("Wrong:", result.summary)

    def test_model_style_dag_graders_negative_control_reports_wrong_behaviors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            store.write_report("Tiny unrelated text.")
            task = EvalTask(
                id="dag_model_negative",
                name="DAG model negative",
                prompt="Research grounded evidence reliability",
                task_mode="research",
                success_criteria=[],
            )

            result = default_graders()["llm_open_ended_judgment_challenger"].grade(task, store)

        dag = result.assertions[0]
        self.assertEqual(dag["type"], "deep_acyclic_graph")
        self.assertFalse(result.passed)
        self.assertGreater(len(dag["wrong_behaviors"]), 0)
        self.assertIn("Report appears off-topic.", dag["wrong_behaviors"])

class ArxivRetrieverTest(unittest.TestCase):
    def test_parse_arxiv_feed(self) -> None:
        payload = b"""<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/2601.00001v1</id>
            <updated>2026-01-01T00:00:00Z</updated>
            <published>2026-01-01T00:00:00Z</published>
            <title>Agent Paradigms For Workplace Automation</title>
            <summary>We introduce a benchmark for agentic workflows. Results suggest planner-executor systems improve reliability.</summary>
            <author><name>A. Researcher</name></author>
            <category term="cs.AI" />
          </entry>
        </feed>"""

        documents = _parse_arxiv_feed(payload)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].source_type, "arxiv_paper")
        self.assertEqual(documents[0].url, "http://arxiv.org/abs/2601.00001v1")
        self.assertIn("planner-executor", " ".join(documents[0].claims))

    def test_source_strategy_fans_out_for_general_research(self) -> None:
        orchestrator = Orchestrator(
            corpus_path=Path("examples/corpus/research_corpus.json"),
            output_root=Path("outputs"),
            config=HarnessConfig(retriever="auto"),
        )

        plan = orchestrator.create_plan("find research studying the human brain and artificial intelligence")
        strategy = orchestrator.create_source_strategy(
            "find research studying the human brain and artificial intelligence",
            plan,
        )

        self.assertGreaterEqual(len(strategy), 7)
        self.assertIn("openalex", {item.retriever for item in strategy})
        self.assertIn("arxiv", {item.retriever for item in strategy})
        self.assertIn("github", {item.retriever for item in strategy})
        self.assertIn("memory", {item.retriever for item in strategy})
        self.assertIn("brain", strategy[0].queries[0])
        self.assertEqual(strategy[0].name, "broad_landscape")
        self.assertLessEqual(len(strategy[0].queries[0].split()), 4)
        self.assertIsInstance(orchestrator._retriever_for(strategy[0].retriever), OpenAlexSearch)

    def test_source_strategy_uses_prompt_domain_lenses(self) -> None:
        orchestrator = Orchestrator(
            corpus_path=Path("examples/corpus/research_corpus.json"),
            output_root=Path("outputs"),
            config=HarnessConfig(retriever="auto"),
        )

        goal = "Get to $10 profit in the prediction market challenge using automated market maker cost-function literature"
        plan = orchestrator.create_plan(goal)
        strategy = orchestrator.create_source_strategy(goal, plan)
        queries = " ".join(query for item in strategy for query in item.queries).lower()

        self.assertIn("challenge", plan.strategy)
        self.assertIn("automated market maker", queries)
        self.assertIn("prediction market", queries)
        self.assertNotIn("workplace automation", queries)

    def test_prediction_market_evaluator_forces_prediction_market_lens(self) -> None:
        orchestrator = Orchestrator(
            corpus_path=Path("examples/corpus/research_corpus.json"),
            output_root=Path("outputs"),
            config=HarnessConfig(retriever="auto", evaluator_name="prediction_market"),
        )

        goal = "Optimize an image compression routine with a deterministic benchmark"
        plan = orchestrator.create_plan(goal)
        strategy = orchestrator.create_source_strategy(goal, plan)
        queries = " ".join(query for item in strategy for query in item.queries).lower()

        self.assertIn("challenge", plan.strategy)
        self.assertIn("prediction market", queries)

    def test_source_strategy_does_not_force_prediction_market_lens_without_prompt_or_evaluator(self) -> None:
        orchestrator = Orchestrator(
            corpus_path=Path("examples/corpus/research_corpus.json"),
            output_root=Path("outputs"),
            config=HarnessConfig(retriever="auto"),
        )

        goal = "Optimize an image compression routine with a deterministic benchmark"
        plan = orchestrator.create_plan(goal)
        strategy = orchestrator.create_source_strategy(goal, plan)
        queries = " ".join(query for item in strategy for query in item.queries).lower()

        self.assertNotIn("prediction-market", plan.strategy)
        self.assertNotIn("lmsr", queries)
        self.assertNotIn("challenge evaluation strategy", queries)

    def test_llm_planner_keywords_drive_paper_search_queries(self) -> None:
        class FakeKeywordLLM:
            is_live = True

            def complete_json(self, _system: str, user: str, **_kwargs: object) -> dict[str, object]:
                self.user_payload = json.loads(user)
                return {
                    "task_type": "open_ended",
                    "topics": ["statistical_learning"],
                    "topic_queries": [
                        '"statistical learning theory" "generalization bounds"',
                        '"supervised learning" "bias variance tradeoff"',
                        '"benchmark datasets" "model evaluation"',
                        '"representation learning" "deep neural networks"',
                    ],
                    "rationale": "Converted the slug-like user request into precise literature-search phrases.",
                }

        orchestrator = Orchestrator(
            corpus_path=Path("examples/corpus/research_corpus.json"),
            output_root=Path("outputs"),
            config=HarnessConfig(retriever="auto"),
        )
        fake_llm = FakeKeywordLLM()
        orchestrator.llm = fake_llm  # type: ignore[assignment]

        goal = "find-me-data-or-papers-help-me-understand-machine-learning-depth"
        plan = orchestrator.create_plan(goal)
        strategy = orchestrator.create_source_strategy(goal, plan)
        queries = " ".join(query for item in strategy for query in item.queries).lower()

        self.assertIn("statistical_learning", plan.topics)
        self.assertIn('"statistical learning theory"', queries)
        self.assertIn('"benchmark datasets"', queries)
        self.assertIn("selected_evaluator", fake_llm.user_payload)
        self.assertNotIn("workplace automation", queries)
        self.assertNotIn("autonomous multi-agent", queries)

    def test_search_scoring_uses_specific_content_terms_not_prompt_filler(self) -> None:
        documents = [
            CorpusDocument(
                url="https://doi.org/10.1007/s10994-011-5255-4",
                title="Scikit-learn: Machine Learning in Python",
                author="Pedregosa et al.",
                date="2011",
                source_type="paper",
                summary="A software library for supervised and unsupervised machine learning.",
                claims=["Scikit-learn supports machine learning model evaluation."],
                tags=["machine learning", "python"],
                credibility_score=0.8,
            ),
            CorpusDocument(
                url="https://doi.org/10.1093/nar/gkv007",
                title="limma powers differential expression analyses for RNA-sequencing and microarray studies",
                author="Ritchie et al.",
                date="2015",
                source_type="paper",
                summary="An R/Bioconductor package for gene expression experiments and RNA sequencing.",
                claims=["limma provides differential expression analysis for genomics."],
                tags=["bioinformatics", "genomics"],
                credibility_score=0.76,
            ),
        ]

        scored = _score_documents("machine learning papers datasets foundations", documents)

        self.assertEqual([document.title for document, _score in scored], ["Scikit-learn: Machine Learning in Python"])

    def test_search_scoring_rejects_broad_iot_false_positive_for_agent_prompt(self) -> None:
        documents = [
            CorpusDocument(
                url="https://doi.org/10.1007/s11704-024-40231-1",
                title="A survey on large language model based autonomous agents",
                author="Xi et al.",
                date="2024",
                source_type="paper",
                summary="LLM autonomous agents plan, use tools, communicate in multi-agent systems, and adapt through feedback.",
                claims=["Autonomous agents can use tools and coordinate with other agents."],
                tags=["llm", "autonomous agents", "multi-agent systems"],
                credibility_score=0.82,
            ),
            CorpusDocument(
                url="https://doi.org/10.1109/access.2019.2932609",
                title="Internet-of-Things (IoT)-Based Smart Agriculture",
                author="Researcher",
                date="2019",
                source_type="paper",
                summary="Wireless sensors and IoT platforms support irrigation, crop surveillance, and harvesting.",
                claims=["IoT devices and communication techniques are used in agriculture applications."],
                tags=["iot", "agriculture", "sensors"],
                credibility_score=0.8,
            ),
        ]

        prompt = (
            "enterprise AI agent adoption proprietary agent harnesses multi-agent self-modification "
            "inter-agent communication autonomous trading agents internal evals evolutionary computation LLM self-improvement"
        )
        scored = _score_documents(prompt, documents)

        self.assertEqual([document.title for document, _score in scored], ["A survey on large language model based autonomous agents"])

    def test_paper_requests_use_scholarly_api_strategy_not_local_memory(self) -> None:
        orchestrator = Orchestrator(
            corpus_path=Path("examples/corpus/research_corpus.json"),
            output_root=Path("outputs"),
            config=HarnessConfig(retriever="auto"),
        )

        goal = "find me papers or sources which go in depth on machine learning for stock or prediction markets"
        plan = orchestrator.create_plan(goal)
        strategy = orchestrator.create_source_strategy(goal, plan)
        retrievers = {item.retriever for item in strategy}
        queries = " ".join(query for item in strategy for query in item.queries).lower()

        self.assertIn("openalex", retrievers)
        self.assertIn("semantic_scholar", retrievers)
        self.assertIn("arxiv", retrievers)
        self.assertNotIn("local", retrievers)
        self.assertNotIn("memory", retrievers)
        self.assertNotIn("github", retrievers)
        self.assertIn("machine learning", queries)
        self.assertIn("stock", queries)
        self.assertIsInstance(orchestrator._retriever_for("semantic_scholar"), SemanticScholarSearch)

    def test_challenge_seed_context_carries_retrieved_literature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            run = RunRecord(
                id="001_run_challenge",
                user_goal="prediction market challenge",
                task_type="bounded",
                harness_config_id="test",
                prompt_versions={},
                harness_config_snapshot={},
            )
            store.add_run(run)
            source = store.add_source(
                Source(
                    url="https://example.org/lmsr",
                    title="Prediction Market Scoring Rules",
                    author="Researcher",
                    date="2020",
                    source_type="paper",
                    summary="Market scoring rules motivate liquidity costs, position limits, and risk controls.",
                    relevance_score=0.9,
                    credibility_score=0.8,
                )
            )
            variant = Variant(
                run_id=run.id,
                outer_iteration=1,
                kind="query",
                payload="prediction market evaluation strategy risk controls",
                parent_ids=[],
                metadata={"retriever": "local"},
            )
            store.add_variant(variant)
            claim = store.add_claim(
                Claim(
                    text="Risk-aware quoting can reduce loss-making fills in prediction-market simulations.",
                    source_ids=[source.id],
                    confidence=0.78,
                    support_level="strong",
                    created_by_agent=f"research_loop:{variant.id}",
                    run_id=run.id,
                )
            )
            evaluation = VariantEvaluation(
                run_id=run.id,
                variant_id=variant.id,
                inner_loop="optimize_query",
                score=0.8,
                metrics={},
                judge_scores=[0.8],
                summary="Retrieved source-backed market-making evidence.",
                passed=True,
            )
            outer = EvolutionaryOuterLoop(
                run_id=run.id,
                goal=run.user_goal,
                task_mode="optimize_query",
                source_strategy=[],
                search_factory=lambda _name: LocalCorpusSearch(Path("examples/corpus/research_corpus.json")),
                evaluator_name="prediction_market",
            )

            seed_context = outer._build_optimizer_seed_context(store, type("Result", (), {"ranked_evaluations": [evaluation]})())
            seed_variant = outer._seed_context_variants(seed_context)[0]

        top_finding = seed_context["top_query_findings"][0]
        self.assertEqual(top_finding["supporting_claims"][0]["id"], claim.id)
        self.assertEqual(top_finding["supporting_sources"][0]["title"], "Prediction Market Scoring Rules")
        self.assertIn("Risk-aware quoting", seed_variant.metadata["seed_literature"]["claims"][0]["text"])

    def test_code_generation_prompt_sees_optimizer_literature_context(self) -> None:
        class CapturingLLM:
            is_live = True
            total_prompt_tokens = 0
            total_completion_tokens = 0
            model_label = "capturing"

            def __init__(self) -> None:
                self.user = ""

            def complete_json(self, _system: str, user: str, **_kwargs: object) -> dict[str, object]:
                self.user = user
                return {"variants": [{"payload": "use source claim about risk-aware quoting"}]}

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            store.write_optimizer_seed_context(
                {
                    "summary": "Risk-aware quoting seed context",
                    "top_query_findings": [
                        {
                            "query": "prediction market strategy risk",
                            "score": 0.8,
                            "summary": "Source-backed evidence.",
                            "supporting_claims": [
                                {"text": "Risk-aware quoting can reduce loss-making fills.", "confidence": 0.78, "source_ids": ["s1"]}
                            ],
                            "supporting_sources": [
                                {"title": "Prediction Market Scoring Rules", "summary": "Liquidity costs and risk controls.", "source_type": "paper"}
                            ],
                        }
                    ],
                    "optimizer_instruction": "Use retrieved claims as strategy context.",
                }
            )
            llm = CapturingLLM()
            outer = EvolutionaryOuterLoop(
                run_id="run_lit_prompt",
                goal="optimize strategy",
                task_mode="optimize",
                source_strategy=[],
                search_factory=lambda _name: LocalCorpusSearch(Path("examples/corpus/research_corpus.json")),
                evaluator_name="length_score",
                llm=llm,
            )

            variants = outer._llm_code_variants(1, [], store=store)
            prompt_payload = json.loads(llm.user)

        self.assertTrue(variants)
        self.assertIn("optimizer_seed_context", prompt_payload)
        self.assertIn("Risk-aware quoting can reduce loss-making fills.", json.dumps(prompt_payload))
        self.assertIn("Prediction Market Scoring Rules", json.dumps(prompt_payload))

    def test_prediction_market_code_prompt_sees_recent_failures(self) -> None:
        class CapturingLLM:
            is_live = True
            total_prompt_tokens = 0
            total_completion_tokens = 0
            model_label = "capturing"

            def __init__(self) -> None:
                self.user = ""

            def complete_json(self, _system: str, user: str, **_kwargs: object) -> dict[str, object]:
                self.user = user
                return {
                    "variants": [
                        {
                            "description": "responds to negative mean_edge by changing quote gating",
                            "payload": textwrap.dedent("""\
                                from orderbook_pm_challenge.strategy import BaseStrategy
                                from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState

                                class Strategy(BaseStrategy):
                                    def on_step(self, state: StepState):
                                        return [CancelAll()]
                            """),
                        }
                    ]
                }

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            variant = Variant(run_id="run_pm_prompt", outer_iteration=2, kind="code", payload="bad stale quoting logic", parent_ids=[], metadata={})
            store.add_variant(variant)
            store.add_variant_evaluation(
                VariantEvaluation(
                    run_id="run_pm_prompt",
                    variant_id=variant.id,
                    inner_loop="optimize",
                    score=0.5,
                    metrics={"mean_edge": -0.027, "score_source": "upstream_orderbook_pm_challenge", "score_eligible": True},
                    judge_scores=[0.5],
                    summary="upstream orderbook-pm mean_edge=-0.027",
                    passed=False,
                )
            )
            llm = CapturingLLM()
            outer = EvolutionaryOuterLoop(
                run_id="run_pm_prompt",
                goal="find positive mean edge prediction market strategy",
                task_mode="optimize_query",
                source_strategy=[],
                search_factory=lambda _name: LocalCorpusSearch(Path("examples/corpus/research_corpus.json")),
                evaluator_name="prediction_market",
                llm=llm,
            )

            variants = outer._llm_prediction_market_code_variants(3, [variant], store=store)
            prompt_payload = json.loads(llm.user)

        self.assertTrue(variants)
        self.assertIn("recent_failure_context", prompt_payload)
        self.assertIn("negative_mean_edge", json.dumps(prompt_payload))
        self.assertIn("-0.027", json.dumps(prompt_payload))
        self.assertIn("challenge_contract", prompt_payload)
        self.assertIn("starter_code", prompt_payload)
        self.assertIn("baseline_rendered_code", prompt_payload)

    def test_prediction_market_code_prompt_receives_entropy_intent_at_generation_site(self) -> None:
        class CapturingLLM:
            is_live = True
            total_prompt_tokens = 0
            total_completion_tokens = 0
            model_label = "capturing"

            def __init__(self) -> None:
                self.user = ""

            def complete_json(self, _system: str, user: str, **_kwargs: object) -> dict[str, object]:
                self.user = user
                return {
                    "variants": [
                        {
                            "description": "uses entropy intent to test a new flow-toxicity gate",
                            "payload": textwrap.dedent("""\
                                from orderbook_pm_challenge.strategy import BaseStrategy
                                from orderbook_pm_challenge.types import CancelAll, StepState

                                class Strategy(BaseStrategy):
                                    def on_step(self, state: StepState):
                                        return [CancelAll()]
                            """),
                        }
                    ]
                }

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            llm = CapturingLLM()
            outer = EvolutionaryOuterLoop(
                run_id="run_pm_entropy_prompt",
                goal="find positive mean edge prediction market strategy",
                task_mode="optimize_query",
                source_strategy=[],
                search_factory=lambda _name: LocalCorpusSearch(Path("examples/corpus/research_corpus.json")),
                evaluator_name="prediction_market",
                llm=llm,
            )
            entropy_intent = {
                "action": "fresh_search_context",
                "exploration_path": "flow toxicity gate from new literature",
                "expected_generalization": "avoid fills after adverse order flow",
            }

            variants = outer._llm_prediction_market_code_variants(4, [], store=store, entropy_intent=entropy_intent)
            prompt_payload = json.loads(llm.user)

        self.assertTrue(variants)
        self.assertEqual(prompt_payload["post_round_entropy_intent"]["exploration_path"], "flow toxicity gate from new literature")
        self.assertIn("starter_code", prompt_payload)

    def test_prediction_market_llm_failure_gets_structural_not_parameter_only_fallbacks(self) -> None:
        class FailingLLM:
            is_live = True
            total_prompt_tokens = 0
            total_completion_tokens = 0
            model_label = "failing"

            def complete_json(self, *_args: object, **_kwargs: object) -> dict[str, object]:
                raise LLMError("model unavailable")

        outer = EvolutionaryOuterLoop(
            run_id="run_pm_structural_fallback",
            goal="find positive mean edge prediction market strategy",
            task_mode="optimize_query",
            source_strategy=[],
            search_factory=lambda _name: LocalCorpusSearch(Path("examples/corpus/research_corpus.json")),
            evaluator_name="prediction_market",
            llm=FailingLLM(),
            population_size=6,
        )
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            variants = outer._propose_prediction_market_variants(1, [], store)

        structural = [variant for variant in variants if variant.metadata.get("proposal_source") == "deterministic_structural"]
        self.assertGreaterEqual(len(structural), 2)
        self.assertTrue(all("class Strategy" in variant.payload for variant in structural))
        self.assertGreater(len({variant.metadata.get("structural_archetype") for variant in structural}), 1)

    def test_prediction_market_rank_key_prefers_less_negative_edge_over_same_score(self) -> None:
        from research_harness.loops import _prediction_market_rank_key

        worse = VariantEvaluation(
            run_id="rank",
            variant_id="worse",
            inner_loop="optimize",
            score=0.5,
            metrics={"mean_edge": -0.03},
            judge_scores=[0.5],
            summary="worse",
            passed=False,
        )
        better = VariantEvaluation(
            run_id="rank",
            variant_id="better",
            inner_loop="optimize",
            score=0.5,
            metrics={"mean_edge": -0.008},
            judge_scores=[0.5],
            summary="better",
            passed=False,
        )

        self.assertGreater(_prediction_market_rank_key(better), _prediction_market_rank_key(worse))

    def test_prediction_market_cancel_only_is_not_score_eligible(self) -> None:
        from research_harness.loops import _prediction_market_no_trade_baseline

        code = """from orderbook_pm_challenge.strategy import BaseStrategy
from orderbook_pm_challenge.types import CancelAll

class Strategy(BaseStrategy):
    def on_step(self, state):
        return [CancelAll()]
"""
        variant = Variant(
            run_id="run_no_trade",
            outer_iteration=1,
            kind="code",
            payload=code,
            parent_ids=[],
            metadata={"structural_archetype": "cancel_only"},
        )
        result = {"mean_edge": 0.0, "mean_arb_edge": 0.0, "mean_retail_edge": 0.0}

        self.assertTrue(_prediction_market_no_trade_baseline(code, result, variant))

    def test_prediction_market_unmeasured_candidates_do_not_become_parents_or_structural_fallbacks(self) -> None:
        from research_harness.loops import _prediction_market_parent_variants, _prediction_market_structural_fallback_variants

        variants = [
            Variant(run_id="run_pm_parent", outer_iteration=1, kind="code", payload="candidate a", parent_ids=[], metadata={}, id="variant_a"),
            Variant(run_id="run_pm_parent", outer_iteration=1, kind="code", payload="candidate b", parent_ids=[], metadata={}, id="variant_b"),
        ]
        unmeasured = [
            VariantEvaluation(
                run_id="run_pm_parent",
                variant_id="variant_a",
                inner_loop="optimize",
                score=0.0,
                metrics={"score_eligible": False, "score_source": "official_sandbox_failed"},
                judge_scores=[0.0],
                summary="docker failed",
                passed=False,
            )
        ]

        self.assertEqual(_prediction_market_parent_variants(variants, [], parent_count=2), [])
        self.assertEqual(_prediction_market_parent_variants(variants, unmeasured[:0], parent_count=2), [])

        fallbacks = _prediction_market_structural_fallback_variants(
            run_id="run_pm_parent",
            goal="prediction market challenge",
            outer_iteration=1,
            parents=[],
            directions=[],
            entropy_intent=None,
            population_size=6,
        )

        self.assertFalse(any(row.metadata.get("structural_archetype") == "cancel_only" for row in fallbacks))
        self.assertFalse(any("return [CancelAll()]" in row.payload for row in fallbacks))

    def test_docker_sandbox_runner_requires_reachable_daemon(self) -> None:
        from research_harness.sandbox import DockerSandboxRunner

        completed = type("Completed", (), {"returncode": 1, "stdout": "", "stderr": "daemon unavailable"})()
        with unittest.mock.patch("research_harness.sandbox.shutil.which", return_value="/usr/bin/docker"):
            with unittest.mock.patch("research_harness.sandbox.subprocess.run", return_value=completed):
                runner = DockerSandboxRunner()
                result = runner.execute_prediction_market(
                    upstream_path=Path("/tmp/upstream"),
                    strategy_path=Path("/tmp/strategy.py"),
                    simulations="1",
                    steps="1",
                    seed_start="0",
                    workers="1",
                )

        self.assertEqual(result.exit_code, 125)
        self.assertIn("docker daemon is not reachable", result.stderr)

    def test_prediction_market_non_positive_edge_does_not_write_optimal_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            outer = EvolutionaryOuterLoop(
                run_id="run_non_positive_edge",
                goal="optimize prediction market strategy",
                task_mode="optimize",
                source_strategy=[],
                search_factory=lambda _item: LocalCorpusSearch([]),
                evaluator_name="prediction_market",
            )
            evaluation = VariantEvaluation(
                run_id="run_non_positive_edge",
                variant_id="variant_flat",
                inner_loop="optimize",
                score=0.5,
                metrics={
                    "mean_edge": 0.0,
                    "official_measured": True,
                    "score_eligible": True,
                    "score_source": "upstream_orderbook_pm_challenge",
                    "candidate_path": "candidate.py",
                },
                judge_scores=[0.5],
                summary="flat no-profit result",
                passed=False,
            )
            variant = Variant(
                run_id="run_non_positive_edge",
                outer_iteration=1,
                kind="code",
                payload="from orderbook_pm_challenge.strategy import BaseStrategy\nclass Strategy(BaseStrategy):\n    pass\n",
                parent_ids=[],
                metadata={},
            )

            outer._write_optimization_outputs(store, [variant], evaluation)
            result = json.loads(store.optimization_result_path.read_text(encoding="utf-8"))

            self.assertFalse(store.optimal_code_path.exists())
            self.assertFalse(result["official_result"]["score_eligible"])
            self.assertEqual(result["official_result"]["promotion_rejected_reason"], "no_positive_mean_edge")

    def test_optimizer_rounds_refresh_literature_before_next_code_round(self) -> None:
        class FakeOptimizeLoop:
            async def evaluate(self, variants, store):
                score = 0.4 + (0.1 * len(store.list("evolution_rounds")))
                return InnerLoopResult(
                    ranked_evaluations=[
                        VariantEvaluation(
                            run_id="run_round_entropy",
                            variant_id=variants[0].id,
                            inner_loop="optimize",
                            score=score,
                            metrics={},
                            judge_scores=[score],
                            summary="fake optimizer result",
                            passed=False,
                        )
                    ],
                    termination_signal="continue",
                )

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            store.write_optimizer_seed_context({"summary": "seed"})
            outer = EvolutionaryOuterLoop(
                run_id="run_round_entropy",
                goal="prediction market challenge",
                task_mode="optimize_query",
                source_strategy=[],
                search_factory=lambda _name: LocalCorpusSearch(Path("examples/corpus/research_corpus.json")),
                evaluator=lambda payload: len(payload) / 100.0,
                evaluator_name="length_score",
                max_outer_iterations=2,
                population_size=2,
            )
            refresh_reasons: list[str] = []

            async def record_literature(_store, reason):
                refresh_reasons.append(reason)

            outer._record_literature_grounding = record_literature  # type: ignore[method-assign]
            asyncio.run(outer._run_generic_optimizer_rounds(store, FakeOptimizeLoop(), [], {"summary": "seed"}))

        self.assertEqual(refresh_reasons, ["optimizer_entropy_after_round_1"])

    def test_prediction_market_entropy_literature_axis_is_chosen_by_llm(self) -> None:
        class AxisLLM:
            is_live = True
            total_prompt_tokens = 0
            total_completion_tokens = 0
            model_label = "axis-llm"

            def __init__(self) -> None:
                self.prompts: list[str] = []

            def complete_json(self, _system: str, user: str, **_kwargs: object) -> dict[str, object]:
                self.prompts.append(user)
                payload = json.loads(user)
                reason = str(payload["reason"])
                return {
                    "axis": f"LLM selected fresh mechanism for {reason}",
                    "rationale": "Use a new mechanism instead of a fixed axis.",
                    "query_terms": ["causal failure mode", "out of sample robustness"],
                }

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            llm = AxisLLM()
            outer = EvolutionaryOuterLoop(
                run_id="run_query_entropy",
                goal="prediction market challenge",
                task_mode="optimize_query",
                source_strategy=[],
                search_factory=lambda _name: LocalCorpusSearch(Path("examples/corpus/research_corpus.json")),
                evaluator_name="prediction_market",
                llm=llm,
            )

            query = outer._literature_grounding_query("prediction_market_entropy_after_round_1", store)
            traces = store.list("agent_traces")

        self.assertTrue(llm.prompts)
        self.assertLessEqual(len(query), 180)
        self.assertIn("llm selected fresh", query)
        self.assertIn("prediction_market_entropy_after_round_1", query)
        self.assertIn("causal failure mode", query)
        self.assertTrue(any(trace.get("agent_name") == "llm_entropy_literature_axis:prediction_market_entropy_after_round_1" for trace in traces))

    def test_prediction_market_entropy_literature_axis_fallback_is_context_derived(self) -> None:
        outer = EvolutionaryOuterLoop(
            run_id="run_query_entropy",
            goal="prediction market challenge",
            task_mode="optimize_query",
            source_strategy=[],
            search_factory=lambda _name: LocalCorpusSearch(Path("examples/corpus/research_corpus.json")),
            evaluator_name="prediction_market",
        )

        first = outer._literature_grounding_query("prediction_market_entropy_after_round_1")
        second = outer._literature_grounding_query("prediction_market_entropy_after_round_2")

        self.assertIn("literature", first)
        self.assertIn("literature", second)

    def test_optimizer_trace_records_score_spread_and_change_summary(self) -> None:
        from research_harness.run_benchmarks import build_optimizer_trace

        variant = Variant(
            run_id="run_trace",
            outer_iteration=1,
            kind="code",
            payload="candidate strategy_family=risk_control",
            parent_ids=[],
            metadata={
                "strategy_family": "risk_control",
                "mechanism_hypothesis": "add guardrail",
                "entropy_role": "risk_control",
            },
        )
        other = Variant(run_id="run_trace", outer_iteration=1, kind="code", payload="other", parent_ids=[], metadata={})
        round_record = EvolutionRound(
            run_id="run_trace",
            outer_iteration=1,
            mode="optimize",
            variant_ids=[variant.id, other.id],
            best_variant_id=variant.id,
            best_score=0.7,
            termination_signal="continue",
            plateau_count=0,
        )
        evaluations = [
            VariantEvaluation(run_id="run_trace", variant_id=variant.id, inner_loop="optimize", score=0.7, metrics={}, judge_scores=[0.7], summary="", passed=True),
            VariantEvaluation(run_id="run_trace", variant_id=other.id, inner_loop="optimize", score=0.2, metrics={}, judge_scores=[0.2], summary="", passed=False),
        ]

        trace = build_optimizer_trace(
            [round_record.__dict__],
            [variant.__dict__, other.__dict__],
            [evaluation.__dict__ for evaluation in evaluations],
        )

        self.assertAlmostEqual(trace[0]["round_score_spread"], 0.5)
        self.assertGreater(trace[0]["round_score_stddev"], 0)
        self.assertIn("family=risk_control", trace[0]["round_change_summary"][0])

    def test_champion_tree_svg_renders_actual_lineage_tree(self) -> None:
        from research_harness.run_benchmarks import champion_tree_mermaid, champion_tree_svg

        tree = {
            "global_champion_variant_id": "variant_root",
            "nodes": [
                {"id": "variant_root", "outer_iteration": 1, "score": 0.9, "highlight": "global_champion", "is_global_champion": True},
                {"id": "variant_left", "outer_iteration": 2, "score": 0.7, "highlight": "candidate"},
                {"id": "variant_right", "outer_iteration": 2, "score": 0.5, "highlight": "round_winner", "is_round_winner": True},
            ],
            "edges": [
                {"from": "variant_root", "to": "variant_left"},
                {"from": "variant_root", "to": "variant_right"},
            ],
        }

        svg = champion_tree_svg(tree)
        mermaid = champion_tree_mermaid(tree)

        self.assertIn("flowchart TD", mermaid)
        self.assertIn("<circle", svg)
        self.assertIn("<path", svg)
        self.assertIn("#dc2626", svg)
        self.assertIn("Actual parent-to-child lineage", svg)

    def test_prediction_market_optimizer_forces_distinct_rendered_code_across_rounds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            outer = EvolutionaryOuterLoop(
                run_id="run_pm_diversity",
                goal="prediction market challenge",
                task_mode="optimize_query",
                source_strategy=[],
                search_factory=lambda _name: LocalCorpusSearch(Path("examples/corpus/research_corpus.json")),
                evaluator_name="prediction_market",
                population_size=4,
            )
            round_one = outer._propose_prediction_market_variants(1, [], store)
            for variant in round_one:
                store.add_variant(variant)
            round_two = outer._propose_prediction_market_variants(2, round_one[:2], store)
            hashes = [variant.metadata.get("rendered_code_hash") for variant in round_one + round_two]

        self.assertEqual(len(hashes), len(set(hashes)))
        self.assertTrue(all(hashes))
        self.assertTrue(
            any("contextual_parent_mutation" in variant.payload for variant in round_two)
            or any(variant.metadata.get("proposal_source") == "deterministic_structural" for variant in round_two)
        )
        self.assertFalse(any("lmsr" in variant.payload.lower() for variant in round_two))
        self.assertFalse(any("adverse" in variant.payload.lower() for variant in round_two))

    def test_optimizer_champion_promotion_writes_highlighted_tree_and_guides_next_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            outer = EvolutionaryOuterLoop(
                run_id="run_champion",
                goal="optimize tiny kernel",
                task_mode="optimize",
                source_strategy=[],
                search_factory=lambda _name: LocalCorpusSearch(Path("examples/corpus/research_corpus.json")),
                evaluator_name="length_score",
                population_size=4,
                parent_count=3,
            )
            variants = [
                Variant(run_id="run_champion", outer_iteration=1, kind="code", payload=f"candidate {index}", parent_ids=[], metadata={})
                for index in range(4)
            ]
            for variant in variants:
                store.add_variant(variant)
            evaluations = [
                VariantEvaluation(
                    run_id="run_champion",
                    variant_id=variants[2].id,
                    inner_loop="optimize",
                    score=0.7,
                    metrics={"json_response": {"score": 0.7, "status": "completed"}},
                    judge_scores=[0.7],
                    summary="{}",
                    passed=False,
                ),
                VariantEvaluation(
                    run_id="run_champion",
                    variant_id=variants[1].id,
                    inner_loop="optimize",
                    score=0.5,
                    metrics={"json_response": {"score": 0.5, "status": "completed"}},
                    judge_scores=[0.5],
                    summary="{}",
                    passed=False,
                ),
            ]
            for evaluation in evaluations:
                store.add_variant_evaluation(evaluation)
            store.add_evolution_round(
                EvolutionRound(
                    run_id="run_champion",
                    outer_iteration=1,
                    mode="optimize",
                    variant_ids=[variant.id for variant in variants],
                    best_variant_id=variants[2].id,
                    best_score=0.7,
                    termination_signal="continue",
                    plateau_count=0,
                )
            )
            outer._promote_round_champion(store, 1, variants, evaluations, loop_name="optimizer_loop")
            tree = json.loads(store.champion_tree_path.read_text(encoding="utf-8"))
            current_champion_exists = store.current_champion_path.exists()
            next_variants = outer._propose_code_variants(2, variants[:3], store)

        self.assertEqual(tree["global_champion_variant_id"], variants[2].id)
        champion_nodes = [node for node in tree["nodes"] if node["highlight"] == "global_champion"]
        self.assertEqual(len(champion_nodes), 1)
        self.assertTrue(current_champion_exists)
        self.assertTrue(all("diff_against_champion" in variant.payload for variant in next_variants))

    def test_prediction_market_plateau_grounding_can_refresh_literature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            outer = EvolutionaryOuterLoop(
                run_id="run_pm_refresh",
                goal="prediction market market making strategy",
                task_mode="optimize_query",
                source_strategy=[],
                search_factory=lambda _name: LocalCorpusSearch(Path("examples/corpus/research_corpus.json")),
                evaluator_name="prediction_market",
            )
            asyncio.run(outer._record_literature_grounding(store, "initial"))
            asyncio.run(outer._record_literature_grounding(store, "prediction_market_plateau_round_2"))
            grounding_claims = [
                claim for claim in store.list("claims")
                if claim.get("created_by_agent") == "literature_grounding_policy"
            ]

        self.assertGreaterEqual(len(grounding_claims), 2)
        self.assertTrue(any("prediction_market_plateau_round_2" in claim["text"] for claim in grounding_claims))

    def test_optimizer_literature_grounding_uses_requested_query_first(self) -> None:
        calls: list[str] = []

        class QueryCaptureSearch:
            tool_name = "query_capture"

            def search(self, query: str, limit: int = 4):
                calls.append(query)
                if "inventory toxicity cancel threshold" not in query:
                    return []
                document = CorpusDocument(
                    url="https://example.com/requested-query-paper",
                    title="Requested Query Paper",
                    author="Researcher",
                    date="2026",
                    source_type="paper",
                    summary=f"Evidence for {query}",
                    claims=["Requested query evidence claim"],
                    tags=["market-making"],
                    credibility_score=0.8,
                )
                return [(document, 0.9)][:limit]

            def to_source(self, document, relevance_score):
                return Source(
                    url=document.url,
                    title=document.title,
                    author=document.author,
                    date=document.date,
                    source_type=document.source_type,
                    summary=document.summary,
                    relevance_score=relevance_score,
                    credibility_score=document.credibility_score,
                )

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            outer = EvolutionaryOuterLoop(
                run_id="run_requested_query",
                goal="prediction market market making strategy",
                task_mode="optimize_query",
                source_strategy=[],
                search_factory=lambda _name: QueryCaptureSearch(),
                evaluator_name="prediction_market",
            )

            asyncio.run(
                outer._record_literature_grounding(
                    store,
                    "optimization_agent_round_2_fetch_literature",
                    requested_queries=["inventory toxicity cancel threshold queue position adverse selection"],
                )
            )
            traces = store.list("agent_traces")
            sources = store.list("sources")

        self.assertTrue(calls)
        self.assertIn("inventory toxicity cancel threshold", calls[0])
        self.assertTrue(any(source.get("title") == "Requested Query Paper" for source in sources))
        self.assertTrue(any(trace.get("prompt") == calls[0] for trace in traces))

    def test_prediction_market_entropy_grounding_fans_out_across_strategy_retrievers(self) -> None:
        calls: list[str] = []

        class FanoutSearch:
            def __init__(self, name: str) -> None:
                self.tool_name = name
                self.name = name

            def search(self, query: str, limit: int = 4):
                calls.append(self.name)
                if self.name not in {"openalex", "arxiv"}:
                    return []
                document = CorpusDocument(
                    url=f"https://example.com/{self.name}/{len(calls)}",
                    title=f"{self.name} paper",
                    author="Researcher",
                    date="2026",
                    source_type="paper",
                    summary=f"{self.name} evidence for {query}",
                    claims=[f"{self.name} evidence claim"],
                    tags=[self.name],
                    credibility_score=0.8,
                )
                return [(document, 0.8)][:limit]

            def to_source(self, document, relevance_score):
                return Source(
                    url=document.url,
                    title=document.title,
                    author=document.author,
                    date=document.date,
                    source_type=document.source_type,
                    summary=document.summary,
                    relevance_score=relevance_score,
                    credibility_score=document.credibility_score,
                )

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            outer = EvolutionaryOuterLoop(
                run_id="run_pm_fanout",
                goal="prediction market market making",
                task_mode="optimize_query",
                source_strategy=[
                    SourceStrategyItem(name="a", retriever="openalex", purpose="papers", queries=["prediction market"], limit=3),
                    SourceStrategyItem(name="b", retriever="arxiv", purpose="papers", queries=["market making"], limit=3),
                ],
                search_factory=lambda name: FanoutSearch(name),
                evaluator_name="prediction_market",
            )

            asyncio.run(outer._record_literature_grounding(store, "prediction_market_entropy_after_round_3"))
            claims = [claim for claim in store.list("claims") if claim.get("created_by_agent") == "literature_grounding_policy"]

        self.assertIn("openalex", calls)
        self.assertIn("arxiv", calls)
        self.assertGreaterEqual(len(claims), 2)

    def test_prediction_market_first_stall_fetches_fresh_literature(self) -> None:
        strategy_code = textwrap.dedent("""\
            from orderbook_pm_challenge.strategy import BaseStrategy
            from orderbook_pm_challenge.types import CancelAll, StepState

            class Strategy(BaseStrategy):
                def on_step(self, state: StepState):
                    return [CancelAll()]
        """)
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            outer = EvolutionaryOuterLoop(
                run_id="run_pm_first_stall",
                goal="prediction market market making strategy",
                task_mode="optimize_query",
                source_strategy=[],
                search_factory=lambda _name: LocalCorpusSearch(Path("examples/corpus/research_corpus.json")),
                evaluator_name="prediction_market",
                max_outer_iterations=2,
            )
            recorded_reasons: list[str] = []

            def propose(round_index: int, parents: list[Variant], _store: ArtifactStore) -> list[Variant]:
                return [
                    Variant(
                        run_id=outer.run_id,
                        outer_iteration=round_index,
                        kind="code",
                        payload=strategy_code,
                        parent_ids=[parent.id for parent in parents],
                        metadata={"challenge": "prediction_market"},
                    )
                ]

            async def evaluate(variant: Variant, _store: ArtifactStore, _round_index: int) -> VariantEvaluation:
                return VariantEvaluation(
                    run_id=outer.run_id,
                    variant_id=variant.id,
                    inner_loop="optimize",
                    score=0.5,
                    metrics={
                        "mean_edge": -0.02,
                        "score_eligible": True,
                        "score_source": "upstream_orderbook_pm_challenge",
                    },
                    judge_scores=[0.5],
                    summary="upstream orderbook-pm mean_edge=-0.020",
                    passed=False,
                )

            async def record_literature(_store: ArtifactStore, reason: str) -> None:
                recorded_reasons.append(reason)

            outer._propose_prediction_market_variants = propose  # type: ignore[method-assign]
            outer._evaluate_prediction_market_variant = evaluate  # type: ignore[method-assign]
            outer._record_literature_grounding = record_literature  # type: ignore[method-assign]

            asyncio.run(outer._run_prediction_market_optimizer(store, [], {"summary": ""}))
            query = outer._literature_grounding_query("prediction_market_first_stall_round_2", store)

        self.assertIn("prediction_market_first_stall_round_2", recorded_reasons)
        self.assertIn("literature", query)
        self.assertIn("mechanism", query)
        self.assertNotIn("adverse selection", query)
        self.assertNotIn("inventory skew", query)
        self.assertNotIn("avellaneda", query.lower())
        self.assertNotIn("glosten", query.lower())
        self.assertNotIn("outputs", query)

    def test_plateau_recovery_records_meaningful_entropy_intent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            outer = EvolutionaryOuterLoop(
                run_id="run_entropy_recovery",
                goal="prediction market market making strategy",
                task_mode="optimize_query",
                source_strategy=[],
                search_factory=lambda _name: LocalCorpusSearch(Path("examples/corpus/research_corpus.json")),
                evaluator_name="prediction_market",
                population_size=4,
            )
            plateau = PlateauDetector("optimize")
            plateau.update(0.5)
            plateau.update(0.5)
            store.add_evolution_round(
                EvolutionRound(
                    run_id="run_entropy_recovery",
                    outer_iteration=2,
                    mode="optimize",
                    variant_ids=[],
                    best_variant_id=None,
                    best_score=0.5,
                    termination_signal="score_plateau",
                    plateau_count=2,
                )
            )
            outer._apply_plateau_recovery(plateau, store, 2, "score_plateau")
            variants = outer._propose_prediction_market_variants(3, [], store)
            for variant in variants:
                store.add_variant(variant)

            result = default_graders()["plateau_entropy_exploration"].grade(
                EvalTask(
                    id="entropy",
                    name="entropy",
                    prompt="prediction market",
                    task_mode="optimize_query",
                    success_criteria=[],
                ),
                store,
            )
            claims = store.list("claims")
            persisted_variants = store.list("variants")

        self.assertTrue(result.passed)
        self.assertTrue(any("expected to improve generalization" in claim["text"].lower() for claim in claims))
        self.assertTrue(
            any(
                isinstance(variant.get("metadata", {}).get("meaningful_entropy_intent"), dict)
                for variant in persisted_variants
            )
        )

    def test_broad_optimizer_plateau_branches_instead_of_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            outer = EvolutionaryOuterLoop(
                run_id="run_broad_plateau",
                goal="optimization challenge",
                task_mode="optimize",
                source_strategy=[],
                search_factory=lambda _name: LocalCorpusSearch(Path("examples/corpus/research_corpus.json")),
                evaluator_name="length_score",
                population_size=48,
                parent_count=4,
                max_outer_iterations=20,
                optimize_plateau_patience=5,
                continue_on_optimize_plateau=True,
            )
            plateau = PlateauDetector("optimize", patience=5)
            signal = "continue"
            for _ in range(6):
                signal = plateau.update(0.2)
            should_stop = outer._should_stop_outer_loop(
                "score_plateau",
                VariantEvaluation(
                    run_id="run_broad_plateau",
                    variant_id="variant_best",
                    inner_loop="optimize",
                    score=0.2,
                    metrics={},
                    judge_scores=[0.2],
                    summary="{}",
                    passed=False,
                ),
                5,
            )
            outer._apply_plateau_recovery(plateau, store, 5, signal)

        self.assertEqual(signal, "score_plateau")
        self.assertFalse(should_stop)
        self.assertEqual(outer.population_size, 64)
        self.assertGreaterEqual(outer._recovery_temperature, 1.05)

    def test_plateau_entropy_grader_rejects_hyperparameter_only_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            store.add_evolution_round(
                EvolutionRound(
                    run_id="run_bad_entropy",
                    outer_iteration=2,
                    mode="optimize",
                    variant_ids=[],
                    best_variant_id=None,
                    best_score=0.4,
                    termination_signal="score_plateau",
                    plateau_count=2,
                )
            )
            store.add_variant(
                Variant(
                    run_id="run_bad_entropy",
                    outer_iteration=3,
                    kind="code",
                    payload="same strategy temperature=1.2 seed=44",
                    parent_ids=[],
                    metadata={
                        "meaningful_entropy_intent": {
                            "action": "boost_temperature",
                            "exploration_path": "raise temperature",
                            "expected_generalization": "",
                        }
                    },
                )
            )

            result = default_graders()["plateau_entropy_exploration"].grade(
                EvalTask(
                    id="bad_entropy",
                    name="bad entropy",
                    prompt="prediction market",
                    task_mode="optimize_query",
                    success_criteria=[],
                ),
                store,
            )

        self.assertFalse(result.passed)
        failed_checks = {assertion["check"] for assertion in result.assertions if not assertion["passed"]}
        self.assertIn("not_hyperparameter_only", failed_checks)
        self.assertIn("intent_records_expected_generalization", failed_checks)


class SkillSpecTest(unittest.TestCase):
    def test_repo_skills_follow_agent_skills_frontmatter_spec(self) -> None:
        skills_root = Path("skills")
        skill_dirs = sorted(path for path in skills_root.iterdir() if path.is_dir())

        self.assertGreaterEqual(len(skill_dirs), 1)
        for skill_dir in skill_dirs:
            skill_file = skill_dir / "SKILL.md"
            self.assertTrue(skill_file.exists(), f"{skill_dir} is missing SKILL.md")
            text = skill_file.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("---\n"), f"{skill_file} must start with YAML frontmatter")
            _, frontmatter, body = text.split("---", 2)
            fields = _simple_frontmatter(frontmatter)
            name = fields.get("name", "")
            description = fields.get("description", "")

            self.assertEqual(name, skill_dir.name)
            self.assertRegex(name, r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
            self.assertLessEqual(len(name), 64)
            self.assertTrue(description.strip(), f"{skill_file} description is required")
            self.assertLessEqual(len(description), 1024)
            self.assertGreater(len(body.strip()), 20)


def _simple_frontmatter(frontmatter: str) -> dict[str, str]:
    fields = {}
    for line in frontmatter.splitlines():
        if not line.strip() or line.startswith(" "):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip().strip('"').strip("'")
    return fields


class PredictionMarketSandboxTest(unittest.TestCase):
    """Tests for the local sandbox evaluation path.

    These run entirely offline — no upstream repo, no uv, no network.
    They exercise _run_prediction_market_sandbox and its fallback chain
    (_prediction_market_local_semantic_score) directly.
    """

    _NOOP = textwrap.dedent("""\
        from orderbook_pm_challenge.strategy import BaseStrategy
        from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState

        class Strategy(BaseStrategy):
            def on_step(self, state: StepState):
                return [CancelAll()]
    """)

    _PASSIVE_MM = textwrap.dedent("""\
        from orderbook_pm_challenge.strategy import BaseStrategy
        from orderbook_pm_challenge.types import CancelAll, PlaceOrder, Side, StepState

        class Strategy(BaseStrategy):
            def on_step(self, state: StepState):
                bid = max(1, (state.competitor_best_bid_ticks or 45) - 6)
                ask = min(99, (state.competitor_best_ask_ticks or 55) + 6)
                return [
                    CancelAll(),
                    PlaceOrder(side=Side.BUY,  price_ticks=bid, quantity=0.5),
                    PlaceOrder(side=Side.SELL, price_ticks=ask, quantity=0.5),
                ]
    """)

    _RUNTIME_ERROR = textwrap.dedent("""\
        from orderbook_pm_challenge.strategy import BaseStrategy
        from orderbook_pm_challenge.types import CancelAll, StepState

        class Strategy(BaseStrategy):
            def on_step(self, state: StepState):
                raise RuntimeError("intentional failure in on_step")
    """)

    def _strategy_path(self, directory: str, code: str, name: str = "strategy.py") -> Path:
        path = Path(directory) / name
        path.write_text(code, encoding="utf-8")
        return path

    # ------------------------------------------------------------------ happy path

    def test_sandbox_noop_strategy_returns_zero_edge(self) -> None:
        from research_harness.loops import PREDICTION_MARKET_DEFAULT_SIMULATION_COUNT, _run_prediction_market_sandbox
        with tempfile.TemporaryDirectory() as d:
            result = _run_prediction_market_sandbox(self._strategy_path(d, self._NOOP))

        self.assertTrue(result["sandbox_executed"])
        self.assertFalse(result["official_measured"])
        self.assertEqual(result["score_source"], "local_sandbox_strategy_execution")
        self.assertAlmostEqual(float(result["mean_edge"]), 0.0, places=4)
        self.assertEqual(int(result["simulations"]), PREDICTION_MARKET_DEFAULT_SIMULATION_COUNT)
        self.assertTrue(result["paired_crn"])
        self.assertEqual(result["seed_start"], 0)
        self.assertIn("success_count", result)
        self.assertIn("failure_count", result)
        self.assertIn("actions_seen", result)

    def test_sandbox_passive_mm_strategy_produces_positive_edge(self) -> None:
        from research_harness.loops import _run_prediction_market_sandbox
        with tempfile.TemporaryDirectory() as d:
            result = _run_prediction_market_sandbox(self._strategy_path(d, self._PASSIVE_MM))

        self.assertTrue(result["sandbox_executed"])
        self.assertGreater(int(result["actions_seen"]), 0)
        self.assertGreater(
            float(result["mean_edge"]),
            0.0,
            "A market-making strategy quoting outside the competitor ladder should yield positive mean edge; "
            "negative edge indicates inventory/cash state is leaking across simulations.",
        )

    def test_sandbox_result_keys_match_upstream_schema(self) -> None:
        from research_harness.loops import _run_prediction_market_sandbox
        with tempfile.TemporaryDirectory() as d:
            result = _run_prediction_market_sandbox(self._strategy_path(d, self._NOOP))

        required = {"official_measured", "mean_edge", "mean_arb_edge", "mean_retail_edge",
                    "success_count", "failure_count", "simulations", "score_source"}
        self.assertTrue(required.issubset(result.keys()), f"Missing keys: {required - result.keys()}")

    # ------------------------------------------------------------------ fallback paths

    def test_sandbox_falls_back_on_syntax_error(self) -> None:
        from research_harness.loops import _run_prediction_market_sandbox
        with tempfile.TemporaryDirectory() as d:
            path = self._strategy_path(d, "class Strategy: def on_step(self ???")
            result = _run_prediction_market_sandbox(path)

        self.assertFalse(result.get("sandbox_executed", True))
        self.assertIn("sandbox_error", result)
        self.assertIn("mean_edge", result)

    def test_sandbox_counts_runtime_errors_as_failures_not_crash(self) -> None:
        # on_step exceptions are caught per-step, so the process still exits 0.
        from research_harness.loops import _run_prediction_market_sandbox
        with tempfile.TemporaryDirectory() as d:
            result = _run_prediction_market_sandbox(self._strategy_path(d, self._RUNTIME_ERROR))

        self.assertIn("mean_edge", result)
        self.assertAlmostEqual(float(result["mean_edge"]), 0.0, places=4)

    def test_sandbox_fallback_when_missing_strategy_class(self) -> None:
        from research_harness.loops import _run_prediction_market_sandbox
        no_class = "# no Strategy class defined here\nprint('oops')\n"
        with tempfile.TemporaryDirectory() as d:
            result = _run_prediction_market_sandbox(self._strategy_path(d, no_class))

        self.assertFalse(result.get("sandbox_executed", True))
        self.assertIn("mean_edge", result)

    # ------------------------------------------------------------------ official → sandbox integration

    def test_official_path_uses_sandbox_when_upstream_disabled(self) -> None:
        from research_harness.loops import _run_prediction_market_official
        with tempfile.TemporaryDirectory() as d:
            path = self._strategy_path(d, self._NOOP)
            prev = os.environ.get("PREDICTION_MARKET_USE_UPSTREAM")
            os.environ["PREDICTION_MARKET_USE_UPSTREAM"] = "0"
            try:
                result = _run_prediction_market_official(path)
            finally:
                if prev is None:
                    del os.environ["PREDICTION_MARKET_USE_UPSTREAM"]
                else:
                    os.environ["PREDICTION_MARKET_USE_UPSTREAM"] = prev

        self.assertFalse(result.get("official_measured", True))
        self.assertIn("error", result)
        self.assertIn("mean_edge", result)

    def test_official_path_result_is_not_measured_when_sandbox_used(self) -> None:
        from research_harness.loops import _run_prediction_market_official
        with tempfile.TemporaryDirectory() as d:
            path = self._strategy_path(d, self._PASSIVE_MM)
            prev = os.environ.get("PREDICTION_MARKET_USE_UPSTREAM")
            os.environ["PREDICTION_MARKET_USE_UPSTREAM"] = "0"
            try:
                result = _run_prediction_market_official(path)
            finally:
                if prev is None:
                    del os.environ["PREDICTION_MARKET_USE_UPSTREAM"]
                else:
                    os.environ["PREDICTION_MARKET_USE_UPSTREAM"] = prev

        # official_measured must be False; only the upstream runner sets it True.
        self.assertFalse(result["official_measured"])
        self.assertNotEqual(result.get("score_source"), "upstream_orderbook_pm_challenge")

    def test_prediction_market_preflight_fails_when_upstream_disabled(self) -> None:
        from research_harness.loops import _prediction_market_official_preflight

        prev = os.environ.get("PREDICTION_MARKET_USE_UPSTREAM")
        os.environ["PREDICTION_MARKET_USE_UPSTREAM"] = "0"
        try:
            result = _prediction_market_official_preflight()
        finally:
            if prev is None:
                del os.environ["PREDICTION_MARKET_USE_UPSTREAM"]
            else:
                os.environ["PREDICTION_MARKET_USE_UPSTREAM"] = prev

        self.assertFalse(result.ok)
        self.assertIn("upstream scorer not found", result.reason)
        self.assertEqual(result.execution_mode, "unavailable")

    def test_prediction_market_preflight_accepts_host_mode_with_uv(self) -> None:
        from research_harness import loops

        with tempfile.TemporaryDirectory() as d:
            repo = Path(d) / "prediction-market-challenge"
            package_dir = repo / "orderbook_pm_challenge"
            package_dir.mkdir(parents=True)
            (repo / "pyproject.toml").write_text(
                '[project]\nname = "orderbook-pm-challenge"\n'
                '[project.scripts]\norderbook-pm = "orderbook_pm_challenge.cli:main"\n',
                encoding="utf-8",
            )
            old_path = os.environ.get("PREDICTION_MARKET_CHALLENGE_PATH")
            old_unsandboxed = os.environ.get("PREDICTION_MARKET_ALLOW_UNSANDBOXED_UPSTREAM")
            os.environ["PREDICTION_MARKET_CHALLENGE_PATH"] = str(repo)
            os.environ["PREDICTION_MARKET_ALLOW_UNSANDBOXED_UPSTREAM"] = "1"
            try:
                with unittest.mock.patch.object(loops.shutil, "which", return_value="/usr/local/bin/uv"):
                    result = loops._prediction_market_official_preflight()
            finally:
                if old_path is None:
                    del os.environ["PREDICTION_MARKET_CHALLENGE_PATH"]
                else:
                    os.environ["PREDICTION_MARKET_CHALLENGE_PATH"] = old_path
                if old_unsandboxed is None:
                    del os.environ["PREDICTION_MARKET_ALLOW_UNSANDBOXED_UPSTREAM"]
                else:
                    os.environ["PREDICTION_MARKET_ALLOW_UNSANDBOXED_UPSTREAM"] = old_unsandboxed

        self.assertTrue(result.ok)
        self.assertEqual(result.execution_mode, "host")
        self.assertFalse(result.docker_sandbox)

    def test_official_scorer_path_requires_upstream_orderbook_repo(self) -> None:
        from research_harness.loops import _find_pm_upstream_path, _is_pm_upstream_repo

        project_root = Path(__file__).resolve().parents[1]
        self.assertFalse(_is_pm_upstream_repo(project_root / "challenges" / "prediction_market"))

        with tempfile.TemporaryDirectory() as d:
            repo = Path(d) / "prediction-market-challenge"
            package_dir = repo / "orderbook_pm_challenge"
            package_dir.mkdir(parents=True)
            (repo / "pyproject.toml").write_text(
                '[project]\nname = "orderbook-pm-challenge"\n'
                '[project.scripts]\norderbook-pm = "orderbook_pm_challenge.cli:main"\n',
                encoding="utf-8",
            )
            prev = os.environ.get("PREDICTION_MARKET_CHALLENGE_PATH")
            os.environ["PREDICTION_MARKET_CHALLENGE_PATH"] = str(repo)
            try:
                self.assertEqual(_find_pm_upstream_path(), repo)
            finally:
                if prev is None:
                    del os.environ["PREDICTION_MARKET_CHALLENGE_PATH"]
                else:
                    os.environ["PREDICTION_MARKET_CHALLENGE_PATH"] = prev


@unittest.skipUnless(os.environ.get("RUN_PM_UPSTREAM") == "1", "Set RUN_PM_UPSTREAM=1 to test the uv run path against the upstream repo.")
class PredictionMarketUpstreamLiveTest(unittest.TestCase):
    """Gated live test: exercises the actual uv run … orderbook-pm … subprocess.

    Requires the upstream prediction-market-challenge repo to be checked out at
    one of the paths _find_pm_upstream_path() probes, or PREDICTION_MARKET_CHALLENGE_PATH set.
    """

    _NOOP = textwrap.dedent("""\
        from orderbook_pm_challenge.strategy import BaseStrategy
        from orderbook_pm_challenge.types import CancelAll, StepState

        class Strategy(BaseStrategy):
            def on_step(self, state: StepState):
                return [CancelAll()]
    """)

    def test_uv_run_returns_official_measured_result(self) -> None:
        import shutil
        from research_harness.loops import _find_pm_upstream_path, _run_prediction_market_official
        self.assertIsNotNone(shutil.which("uv"), "uv must be on PATH to run the upstream evaluator.")
        self.assertIsNotNone(_find_pm_upstream_path(), "prediction-market-challenge repo not found.")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "strategy.py"
            path.write_text(self._NOOP, encoding="utf-8")
            result = _run_prediction_market_official(path)

        self.assertTrue(result["official_measured"])
        self.assertEqual(result["score_source"], "upstream_orderbook_pm_challenge")
        self.assertIn("mean_edge", result)
        self.assertIsInstance(float(result["mean_edge"]), float)
        self.assertGreaterEqual(int(result["simulations"]), 1)


if __name__ == "__main__":
    unittest.main()
