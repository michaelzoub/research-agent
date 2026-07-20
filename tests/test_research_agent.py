from __future__ import annotations

import json
import asyncio
import io
import os
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
from unittest import mock
from pathlib import Path
from typing import Any

from research_harness.cli import _apply_grader_budget_defaults, build_parser, configure_interactive_run
from research_harness.llm import LLMClient, LLMError, ModelToolCall, ModelTurn, _post_json, _validate_tool_history
from research_harness.agents import SpecialistConsultationTool
from research_harness.orchestrator import HarnessConfig
from research_harness.agent_loop import AgentLoop, AgentMiddleware, MiddlewareStack
from research_harness.agent_state import AgentState
from research_harness.context_projection import WorkingStateProjector
from research_harness.research_agent import AgentRunConfig, FinalAnswerValidator, ResearchAgent, _SYSTEM_INSTRUCTIONS, _join_answer_chunks, _partial_synthesis
from research_harness.search import ArxivSearch, LocalCorpusSearch, WebSearch, _arxiv_identifier, _arxiv_query, _retrieval_query
from research_harness.schemas import Source
from research_harness.store import ArtifactStore
from research_harness.tools import DocumentFigureTool, OptimizationSwarmTool, ParameterSweepTool, PredictionMarketEvaluationTool, SaveLearningTool, SearchTool, TerminalExecutionTool, ToolContext, ToolRegistry, ToolResult, WebFetchTool
from research_harness.llm import LLMResponse
from research_harness.tools.research import _FigureHTMLParser, _arxiv_pdf_url, _image_dimensions, _inspection_urls


class ScriptedDecider:
    def __init__(self, responses: list[dict[str, Any]]):
        self.responses = list(responses)
        self.observed_messages: list[list[dict[str, Any]]] = []

    def decide(self, messages: list[dict[str, Any]], _tools: list[dict[str, Any]]) -> dict[str, Any]:
        self.observed_messages.append(list(messages))
        return self.responses.pop(0)


class ResearchAgentTest(unittest.TestCase):
    def test_shared_inference_transport_retries_a_transient_timeout(self) -> None:
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = b'{"ok": true}'
        with mock.patch(
            "research_harness.llm.urllib.request.urlopen",
            # macOS system Python 3.9 exposes socket.timeout as OSError rather
            # than TimeoutError; this is the exact failure seen in run 143.
            side_effect=[socket.timeout("read timed out"), response],
        ) as urlopen, mock.patch("research_harness.llm.time.sleep") as sleep:
            result = _post_json("https://provider.test/v1", {"model": "test"}, {}, 1)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(0.5)

    def test_shared_inference_transport_does_not_retry_a_bad_request(self) -> None:
        error = urllib.error.HTTPError(
            "https://provider.test/v1", 400, "bad request", {}, io.BytesIO(b'invalid payload')
        )
        with mock.patch("research_harness.llm.urllib.request.urlopen", side_effect=error) as urlopen:
            with self.assertRaisesRegex(LLMError, "HTTP 400.*invalid payload"):
                _post_json("https://provider.test/v1", {"model": "test"}, {}, 1)

        self.assertEqual(urlopen.call_count, 1)

    def test_kimi_uses_longer_default_timeout_and_allows_override(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RESEARCH_HARNESS_LLM_TIMEOUT_SECONDS", None)
            client = LLMClient(provider="kimi", model="kimi/kimi-k2.6", api_key="sk-test")
        self.assertEqual(client.timeout_seconds, 120.0)

        with mock.patch.dict(os.environ, {"RESEARCH_HARNESS_LLM_TIMEOUT_SECONDS": "45"}):
            overridden = LLMClient(provider="kimi", model="kimi/k2.6", api_key="sk-test")
        self.assertEqual(overridden.timeout_seconds, 45.0)

    def test_kimi_native_tool_turn_forces_temperature_one(self) -> None:
        client = LLMClient(provider="kimi", model="kimi/kimi-k2.6", api_key="sk-test")
        response = {
            "model": "kimi-k2.6",
            "choices": [{"finish_reason": "stop", "message": {"content": "Done.", "tool_calls": []}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
        with mock.patch("research_harness.llm._post_json", return_value=response) as post_json:
            turn = client.complete_turn(
                [{"role": "user", "content": "Use a tool if needed."}],
                [],
                temperature=0.35,
            )

        self.assertEqual(turn.provider, "kimi")
        self.assertEqual(post_json.call_args.args[1]["temperature"], 1)
        self.assertEqual(post_json.call_args.args[1]["max_completion_tokens"], 8192)

    def test_kimi_native_tool_turn_serializes_empty_assistant_tool_call_content(self) -> None:
        client = LLMClient(provider="kimi", model="kimi/kimi-k2.6", api_key="sk-test")
        response = {
            "model": "kimi-k2.6",
            "choices": [{"finish_reason": "stop", "message": {"content": "Done.", "tool_calls": []}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
        messages = [
            {"role": "user", "content": "Inspect the challenge."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "name": "read_workspace_file", "arguments": {"path": "challenge.py"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "name": "read_workspace_file", "content": {"status": "ok"}},
        ]
        with mock.patch("research_harness.llm._post_json", return_value=response) as post_json:
            client.complete_turn(messages, [])

        serialized = post_json.call_args.args[1]["messages"]
        self.assertEqual(serialized[1]["content"], "Tool calls requested.")

    def test_kimi_continuation_serializes_empty_assistant_content(self) -> None:
        client = LLMClient(provider="kimi", model="kimi/kimi-k2.6", api_key="sk-test")
        response = {
            "model": "kimi-k2.6",
            "choices": [{"finish_reason": "stop", "message": {"content": "Finished.", "tool_calls": []}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
        messages = [
            {"role": "user", "content": "Write the answer."},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Continue after the output limit."},
        ]
        with mock.patch("research_harness.llm._post_json", return_value=response) as post_json:
            client.complete_turn(messages, [])

        serialized = post_json.call_args.args[1]["messages"]
        self.assertEqual(
            serialized[1]["content"],
            "The previous response reached its output limit before emitting visible text.",
        )

    def test_openai_history_does_not_invent_empty_assistant_content(self) -> None:
        client = LLMClient(provider="openai", model="gpt-4o-mini", api_key="sk-test")
        response = {
            "model": "gpt-4o-mini",
            "choices": [{"finish_reason": "stop", "message": {"content": "Finished.", "tool_calls": []}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
        messages = [
            {"role": "user", "content": "Write the answer."},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Continue."},
        ]
        with mock.patch("research_harness.llm._post_json", return_value=response) as post_json:
            client.complete_turn(messages, [])

        self.assertIsNone(post_json.call_args.args[1]["messages"][1]["content"])
        self.assertEqual(post_json.call_args.args[1]["max_completion_tokens"], 1200)

    def test_incomplete_provider_tool_arguments_are_not_silently_coerced_to_empty(self) -> None:
        client = LLMClient(provider="openai", model="gpt-4o-mini", api_key="sk-test")
        response = {
            "model": "gpt-4o-mini",
            "choices": [{
                "finish_reason": "length",
                "message": {"content": "", "tool_calls": [{
                    "id": "grader_1",
                    "function": {
                        "name": "evaluate_prediction_market_candidate",
                        "arguments": '{"code":',
                    },
                }]},
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1200},
        }
        with mock.patch("research_harness.llm._post_json", return_value=response):
            turn = client.complete_turn([{"role": "user", "content": "Evaluate code."}], [])

        self.assertEqual(turn.tool_calls[0].raw_arguments, '{"code":')
        self.assertIn("incomplete or invalid JSON", turn.tool_calls[0].argument_error or "")
        self.assertEqual(turn.tool_calls[0].arguments, {})

    def test_native_tool_history_rejects_interleaved_non_tool_message(self) -> None:
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "name": "first", "arguments": {}},
                {"id": "call_2", "name": "second", "arguments": {}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": {"status": "ok"}},
            {"role": "user", "content": "Continue optimizing."},
            {"role": "tool", "tool_call_id": "call_2", "content": {"status": "error"}},
        ]

        with self.assertRaisesRegex(LLMError, "interrupted pending tool results"):
            _validate_tool_history(messages)

    def test_native_tool_history_accepts_contiguous_success_and_error_results(self) -> None:
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "name": "first", "arguments": {}},
                {"id": "call_2", "name": "second", "arguments": {}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": {"status": "ok"}},
            {"role": "tool", "tool_call_id": "call_2", "content": {"status": "error"}},
            {"role": "user", "content": "Continue optimizing."},
        ]

        _validate_tool_history(messages)

    def _agent(self, decider: ScriptedDecider) -> ResearchAgent:
        backend = LocalCorpusSearch(Path("examples/corpus/research_corpus.json"))
        return ResearchAgent(decider, ToolRegistry([SearchTool(backend)]), AgentRunConfig(max_iterations=4))

    def test_agent_can_answer_without_a_tool_call(self) -> None:
        agent = self._agent(ScriptedDecider([{"type": "final", "answer": "No external evidence is needed."}]))

        result = agent.run("Rewrite this sentence.", workspace=Path.cwd())

        self.assertEqual(result.termination_reason, "completed")
        self.assertEqual(result.final_answer, "No external evidence is needed.")
        self.assertEqual(result.tool_calls, [])

    def test_agent_state_initializes_the_provider_neutral_trajectory(self) -> None:
        context = ToolContext(workspace=Path.cwd(), run_id="state-test")
        state = AgentState.initialize(
            objective="Investigate.", context=context, initial_cost=0.25,
            system_messages=["system one", "system two"],
        )

        self.assertEqual(state.objective, "Investigate.")
        self.assertEqual([message["role"] for message in state.messages], ["system", "system", "user"])
        self.assertEqual(state.initial_cost, 0.25)
        self.assertEqual(state.current_iteration, 0)
        self.assertEqual(state.tool_calls, [])

    def test_middleware_stack_composes_forward_before_and_reverse_after(self) -> None:
        calls: list[str] = []

        class RecordingMiddleware(AgentMiddleware):
            def __init__(self, name: str) -> None:
                self.name = name

            async def before_agent(self, _state: AgentState) -> None:
                calls.append(f"before:{self.name}")

            async def after_agent(self, _state: AgentState) -> None:
                calls.append(f"after:{self.name}")

        state = AgentState.initialize(
            objective="Investigate.", context=ToolContext(workspace=Path.cwd()),
            initial_cost=0.0, system_messages=[],
        )
        stack = MiddlewareStack([RecordingMiddleware("first"), RecordingMiddleware("second")])

        async def exercise() -> None:
            await stack.before_agent(state)
            await stack.after_agent(state)

        asyncio.run(exercise())
        self.assertEqual(calls, ["before:first", "before:second", "after:second", "after:first"])

    def test_runtime_policy_terminates_before_a_model_call(self) -> None:
        class UncalledDecider:
            def __init__(self) -> None:
                self.called = False

            def decide(self, _messages: Any, _tools: Any) -> ModelTurn:
                self.called = True
                raise AssertionError("runtime policy should stop before the model")

        decider = UncalledDecider()
        result = asyncio.run(AgentLoop(
            decider, ToolRegistry([]), AgentRunConfig(max_runtime_seconds=-1)
        ).run("Do work.", ToolContext(workspace=Path.cwd())))

        self.assertFalse(decider.called)
        self.assertEqual(result.termination_reason, "budget_exhausted")
        self.assertIn("Wall-clock runtime budget exhausted", result.final_answer)
        self.assertEqual([event.event_type for event in result.events], ["termination"])

    def test_cost_policy_terminates_before_a_model_call(self) -> None:
        class CostLLM:
            def total_cost(self) -> float:
                return 1.0

        decider = mock.Mock(llm=CostLLM())
        result = asyncio.run(AgentLoop(
            decider, ToolRegistry([]), AgentRunConfig(max_cost_usd=0.0)
        ).run("Do work.", ToolContext(workspace=Path.cwd())))

        decider.decide.assert_not_called()
        self.assertEqual(result.termination_reason, "budget_exhausted")

    def test_model_turn_records_request_start_response_end_and_duration(self) -> None:
        class DelayedDecider:
            async def decide(self, _messages: list[dict[str, Any]], _tools: list[dict[str, Any]]) -> dict[str, Any]:
                await asyncio.sleep(0.01)
                return {"type": "final", "answer": "Done."}

        result = ResearchAgent(DelayedDecider(), ToolRegistry([]), AgentRunConfig(max_iterations=1)).run(
            "Rewrite this sentence.", workspace=Path.cwd()
        )
        self.assertEqual(result.events[0].event_type, "model_request")
        event = result.events[1]

        self.assertEqual(event.event_type, "model_turn")
        self.assertEqual(result.events[0].model_call_id, event.model_call_id)
        self.assertIsNotNone(event.started_at)
        self.assertIsNotNone(event.completed_at)
        self.assertGreaterEqual(event.runtime_ms or 0, 5)
        self.assertLess(event.started_at or "", event.completed_at or "")

    def test_model_error_progress_exposes_the_actual_error(self) -> None:
        class BrokenDecider:
            def decide(self, _messages: Any, _tools: Any) -> ModelTurn:
                raise LLMError("provider rejected malformed history")

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            result = ResearchAgent(
                BrokenDecider(), ToolRegistry([]), AgentRunConfig(max_iterations=1)
            ).run("Test error reporting.", workspace=Path.cwd(), store=store, run_id="error-test")
            progress = store.progress_path.read_text(encoding="utf-8")

        self.assertEqual(result.status, "partial")
        self.assertIn("error — Model error: LLMError: provider rejected malformed history", progress)
        self.assertNotIn("unknown/unknown", progress)

    def test_agent_retries_transport_exhaustion_without_losing_state(self) -> None:
        class TimeoutThenAnswer:
            def __init__(self) -> None:
                self.calls = 0
                self.observed_messages: list[list[dict[str, Any]]] = []

            def decide(self, messages: list[dict[str, Any]], _tools: Any) -> ModelTurn:
                self.calls += 1
                self.observed_messages.append(list(messages))
                if self.calls == 1:
                    raise LLMError("Could not reach model provider after 3 attempt(s): read timed out")
                return ModelTurn("Recovered.", [], "stop", "test", "test")

        decider = TimeoutThenAnswer()
        result = ResearchAgent(
            decider, ToolRegistry([]), AgentRunConfig(max_iterations=3, max_consecutive_model_failures=2)
        ).run("Recover from a transient provider failure.", workspace=Path.cwd())

        self.assertEqual(result.status, "completed")
        self.assertEqual(decider.calls, 2)
        self.assertEqual(decider.observed_messages[0], decider.observed_messages[1])
        retries = [event for event in result.events if event.event_type == "model_retry"]
        self.assertEqual(len(retries), 1)

    def test_agent_does_not_retry_non_transport_model_error(self) -> None:
        class RejectedRequest:
            def __init__(self) -> None:
                self.calls = 0

            def decide(self, _messages: Any, _tools: Any) -> ModelTurn:
                self.calls += 1
                raise LLMError("provider rejected malformed history")

        decider = RejectedRequest()
        result = ResearchAgent(
            decider, ToolRegistry([]), AgentRunConfig(max_iterations=3, max_consecutive_model_failures=2)
        ).run("Do not hide a deterministic provider rejection.", workspace=Path.cwd())

        self.assertEqual(result.status, "partial")
        self.assertEqual(decider.calls, 1)

    def test_prediction_market_evaluation_is_a_model_selected_real_grader_call(self) -> None:
        code = "from orderbook_pm_challenge.strategy import BaseStrategy\nclass Strategy(BaseStrategy):\n    pass\n"
        decider = ScriptedDecider([
            {"type": "tool_call", "tool_name": "evaluate_prediction_market_candidate", "arguments": {"code": code, "rationale": "measure this candidate"}},
            {"type": "final", "answer": "The official grader measured the candidate."},
        ])
        agent = ResearchAgent(decider, ToolRegistry([PredictionMarketEvaluationTool()]), AgentRunConfig(max_iterations=3))
        measured = {
            "official_measured": True, "score_eligible": True, "mean_edge": 1.25,
            "mean_arb_edge": 0.5, "mean_retail_edge": 0.75, "score_source": "upstream_orderbook_pm_challenge",
            "stdout": "{}", "stderr": "", "upstream": {"command": ["docker", "run"]},
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch("research_harness.tools.graders.get_optimization_grader") as get_grader:
            get_grader.return_value.evaluate.return_value = measured
            store = ArtifactStore(Path(directory) / "run")
            result = agent.run("Evaluate the configured PM challenge.", workspace=Path.cwd(), store=store, run_id="run_pm")

            self.assertEqual(result.tool_calls[0]["status"], "ok")
            trial_path = next(store.optimization_trials_dir.glob("*.json"))
            trial = json.loads(trial_path.read_text(encoding="utf-8"))
            self.assertEqual(trial["score"], 1.25)
            self.assertEqual(trial["round_index"], 1)
            trial_code_path = Path(trial["trial_code_path"])
            self.assertEqual(trial_code_path.suffix, ".py")
            self.assertEqual(trial_code_path.read_text(encoding="utf-8"), code)
            champion = json.loads(store.current_champion_path.read_text(encoding="utf-8"))
            self.assertEqual(champion["global_champion"]["variant_id"], trial["trial_id"])
            self.assertTrue(champion["global_champion"]["promoted_this_round"])

    def test_prediction_market_tool_rejects_non_candidate_without_a_trial(self) -> None:
        tool = PredictionMarketEvaluationTool()
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            result = asyncio.run(tool.execute({"code": "not a strategy candidate"}, ToolContext(workspace=Path.cwd(), store=store)))

            self.assertEqual(result.status, "error")
            self.assertFalse(store.optimization_trials_dir.exists())

    def test_grader_tools_save_learnings_and_sweep_official_variants(self) -> None:
        base = "from orderbook_pm_challenge.strategy import BaseStrategy\nPARAM = 1\nclass Strategy(BaseStrategy):\n    pass\n"
        measured = {"official_measured": True, "score_eligible": True, "mean_edge": 1.5}
        with tempfile.TemporaryDirectory() as directory, mock.patch("research_harness.tools.swarm.get_optimization_grader") as get_grader:
            root, store = Path(directory), ArtifactStore(Path(directory) / "run")
            (root / "base.py").write_text(base, encoding="utf-8")
            get_grader.return_value.evaluate.return_value = measured
            context = ToolContext(workspace=root, readable_roots=[root], store=store, run_id="run_pm")
            learning = asyncio.run(SaveLearningTool().execute({"title": "Parameter stable", "finding": "PARAM=2 matched the best score.", "evidence": "official mean_edge=1.5", "status": "confirmed"}, context))
            sweep = asyncio.run(ParameterSweepTool().execute({"base_strategy_path": "base.py", "old_value": "PARAM = 1", "values": ["PARAM = 2", "PARAM = 3"], "seed_start": 4, "simulations": 8}, context))
            self.assertEqual(learning.status, "ok")
            self.assertIn("Parameter stable", store.learnings_path.read_text(encoding="utf-8"))
            self.assertEqual(sweep.status, "ok")
            self.assertTrue(Path(sweep.data["winner"]["winner_path"]).is_file())
            get_grader.return_value.evaluate.assert_called_with(mock.ANY, simulations="8", seed_start="4")

    def test_grader_swarm_runs_independent_model_workers_and_persists_results(self) -> None:
        base = "from orderbook_pm_challenge.strategy import BaseStrategy\nclass Strategy(BaseStrategy):\n    pass\n"
        class WorkerLLM:
            def complete(self, *_args, **_kwargs):
                return LLMResponse(base, "test-model", "test", 3, 4, 0.01)
        measured = {"official_measured": True, "score_eligible": True, "mean_edge": 2.0}
        with tempfile.TemporaryDirectory() as directory, mock.patch("research_harness.tools.swarm.get_optimization_grader") as get_grader:
            root, store = Path(directory), ArtifactStore(Path(directory) / "run")
            (root / "base.py").write_text(base, encoding="utf-8")
            get_grader.return_value.evaluate.return_value = measured
            result = asyncio.run(OptimizationSwarmTool(WorkerLLM()).execute({"base_strategy_path": "base.py", "agents": [
                {"hypothesis": "Reduce stale quote risk.", "evaluation_protocol": "8 fixed seeds", "target_to_beat": 1.0},
                {"hypothesis": "Improve inventory skew.", "evaluation_protocol": "8 fixed seeds", "target_to_beat": 1.0},
            ]}, ToolContext(workspace=root, readable_roots=[root], store=store, run_id="run_pm")))
            self.assertEqual(result.status, "ok")
            self.assertEqual(len(result.data["workers"]), 2)
            self.assertTrue((store.root / "swarm_results.json").is_file())
            self.assertEqual(len(store.list("agent_traces")), 2)

    def test_grader_swarm_allows_a_from_scratch_worker_without_a_base_file(self) -> None:
        code = "from orderbook_pm_challenge.strategy import BaseStrategy\nclass Strategy(BaseStrategy):\n    pass\n"
        class WorkerLLM:
            def __init__(self): self.prompt = ""
            def complete(self, _system, prompt, **_kwargs):
                self.prompt = prompt
                return LLMResponse(code, "test-model", "test")
        with tempfile.TemporaryDirectory() as directory, mock.patch("research_harness.tools.swarm.get_optimization_grader") as get_grader:
            llm, store = WorkerLLM(), ArtifactStore(Path(directory) / "run")
            get_grader.return_value.evaluate.return_value = {"official_measured": True, "score_eligible": True, "mean_edge": 2.0}
            result = asyncio.run(OptimizationSwarmTool(llm).execute({"agents": [{"hypothesis": "Build an inventory-aware strategy from first principles.", "evaluation_protocol": "8 fixed seeds", "target_to_beat": 1.0, "strategy_mode": "from_scratch"}]}, ToolContext(workspace=Path(directory), readable_roots=[Path(directory)], store=store, run_id="run_pm")))
            self.assertEqual(result.status, "ok")
            self.assertNotIn("Base strategy", llm.prompt)

    def test_prediction_market_promotion_keeps_the_best_measured_candidate(self) -> None:
        first = "from orderbook_pm_challenge.strategy import BaseStrategy\nclass Strategy(BaseStrategy):\n    pass\n"
        second = first + "# lower-scoring revision\n"
        measured = [
            {"official_measured": True, "score_eligible": True, "mean_edge": 2.0, "mean_arb_edge": 1.0, "mean_retail_edge": 1.0},
            {"official_measured": True, "score_eligible": True, "mean_edge": 1.0, "mean_arb_edge": 0.4, "mean_retail_edge": 0.6},
        ]
        with tempfile.TemporaryDirectory() as directory, mock.patch("research_harness.tools.graders.get_optimization_grader") as get_grader:
            get_grader.return_value.evaluate.side_effect = measured
            store = ArtifactStore(Path(directory) / "run")
            tool = PredictionMarketEvaluationTool()
            first_result = asyncio.run(tool.execute({"code": first}, ToolContext(workspace=Path.cwd(), store=store, run_id="run_pm")))
            second_result = asyncio.run(tool.execute({"code": second}, ToolContext(workspace=Path.cwd(), store=store, run_id="run_pm")))

            self.assertTrue(first_result.data["promotion"]["promoted_this_round"])
            self.assertFalse(second_result.data["promotion"]["promoted_this_round"])
            self.assertEqual(second_result.data["promotion"]["global_champion"]["score"], 2.0)
            self.assertEqual(len(second_result.data["promotion"]["measured_history"]), 2)

    def test_agent_recovers_after_bad_tool_selection_using_observation(self) -> None:
        decider = ScriptedDecider([
            {"type": "tool_call", "tool_name": "not_registered", "arguments": {}},
            {"type": "final", "answer": "I cannot access that capability, so here is the limitation."},
        ])
        result = self._agent(decider).run("Find evidence.", workspace=Path.cwd())

        self.assertEqual(result.termination_reason, "completed")
        self.assertEqual(result.tool_calls[0]["status"], "error")
        self.assertIn("Unknown tool", decider.observed_messages[1][-1]["content"]["error"])

    def test_tool_error_is_persisted_to_failed_paths(self) -> None:
        decider = ScriptedDecider([
            {"type": "tool_call", "tool_name": "not_registered", "arguments": {}},
            {"type": "final", "answer": "I cannot access that capability."},
        ])
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            self._agent(decider).run("Find evidence.", workspace=Path.cwd(), store=store, run_id="run_error")
            failures = store.list("failed_paths")

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["failure_component"], "tool")
        self.assertIn("Unknown tool", failures[0]["reason"])

    def test_selected_search_tool_persists_real_sources_and_trace(self) -> None:
        decider = ScriptedDecider([
            {"type": "tool_call", "tool_name": "local_corpus_search", "arguments": {"query": "multi-agent systems", "limit": 2}},
            {"type": "final", "answer": "Grounded synthesis. https://example.org/single-agent-baseline-limitations"},
        ])
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            result = self._agent(decider).run("Research multi-agent systems.", workspace=Path.cwd(), store=store, run_id="run_agent")

            self.assertEqual(result.termination_reason, "partial")
            self.assertGreater(len(store.list("sources")), 0)
            self.assertEqual(store.list("sources")[0]["evidence_kind"], "lead")
            transcript = json.loads(store.agent_transcript_path.read_text(encoding="utf-8"))
            self.assertEqual(transcript["termination_reason"], "partial")
            self.assertEqual(transcript["tool_calls"][0]["tool"], "local_corpus_search")
            trace = store.list("agent_traces")[0]
            self.assertEqual(trace["tools_used"], ["local_corpus_search"])
            self.assertEqual(trace["tool_calls"][0]["results"], len(store.list("sources")))
            event_rows = [json.loads(line) for line in store.agent_event_log_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(
                [event["event_type"] for event in event_rows][:7],
                ["model_request", "model_turn", "tool_requested", "tool_result", "model_request", "model_turn", "final_validation"],
            )
            self.assertEqual(event_rows[0]["model_call_id"], event_rows[1]["model_call_id"])
            self.assertEqual(event_rows[4]["model_call_id"], event_rows[5]["model_call_id"])

    def test_iteration_limit_is_not_reported_as_a_final_answer(self) -> None:
        decider = ScriptedDecider([{"type": "tool_call", "tool_name": "not_registered", "arguments": {}}] * 3)
        result = ResearchAgent(decider, ToolRegistry([]), AgentRunConfig(max_iterations=2)).run("Do work.", workspace=Path.cwd())

        self.assertEqual(result.termination_reason, "budget_exhausted")
        self.assertIn("Incomplete evidence packet", result.final_answer)

    def test_web_fetch_rejects_private_network_targets(self) -> None:
        result = asyncio.run(WebFetchTool().execute({"url": "http://127.0.0.1/private"}, ToolContext(workspace=Path.cwd())))

        self.assertEqual(result.status, "error")
        self.assertIn("Private or loopback", result.error or "")

    def test_terminal_executes_direct_argv_and_preserves_output(self) -> None:
        captured: dict[str, Any] = {}

        def runner(argv: list[str], cwd: Path, timeout_seconds: float):
            captured.update({"argv": argv, "cwd": cwd, "timeout": timeout_seconds})
            return subprocess.CompletedProcess(argv, 0, stdout="real command output", stderr="")

        tool = TerminalExecutionTool(runner=runner)
        with mock.patch("research_harness.tools.terminal.shutil.which", return_value="/usr/bin/curl"), mock.patch("research_harness.tools.terminal._public_url_error", return_value=None):
            result = asyncio.run(tool.execute(
                {"command": "curl", "args": ["https://example.org/paper"], "timeout_seconds": 12},
                ToolContext(workspace=Path.cwd()),
            ))

        self.assertEqual(result.status, "ok")
        self.assertEqual(captured["argv"], ["/usr/bin/curl", "https://example.org/paper"])
        self.assertEqual(captured["cwd"], Path.cwd().resolve())
        self.assertEqual(captured["timeout"], 12)
        self.assertEqual(result.data["stdout"], "real command output")
        self.assertEqual(result.source_metadata[0]["url"], "https://example.org/paper")

    def test_terminal_rejects_private_curl_and_mutating_npm(self) -> None:
        tool = TerminalExecutionTool()
        private_curl = asyncio.run(tool.execute(
            {"command": "curl", "args": ["http://127.0.0.1/private"]}, ToolContext(workspace=Path.cwd())
        ))
        npm_install = asyncio.run(tool.execute(
            {"command": "npm", "args": ["install", "some-package"]}, ToolContext(workspace=Path.cwd())
        ))

        self.assertEqual(private_curl.status, "error")
        self.assertIn("Private or loopback", private_curl.error or "")
        self.assertEqual(npm_install.status, "error")
        self.assertIn("limited", npm_install.error or "")

    def test_registry_rejects_invalid_tool_arguments_before_execution(self) -> None:
        backend = LocalCorpusSearch(Path("examples/corpus/research_corpus.json"))
        result = asyncio.run(ToolRegistry([SearchTool(backend)]).execute(
            "local_corpus_search", {"query": "ok", "limit": 0}, ToolContext(workspace=Path.cwd())
        ))

        self.assertEqual(result.status, "error")
        self.assertIn("below the minimum", result.error or "")

    def test_cli_and_config_have_no_execution_mode(self) -> None:
        parser = build_parser()
        self.assertNotIn("--mode", parser.format_help())
        self.assertNotIn("--task-mode", parser.format_help())
        self.assertFalse(hasattr(HarnessConfig(), "mode"))

    def test_cli_grader_flag_defaults_to_prediction_market(self) -> None:
        args = build_parser().parse_args(["optimize pm challenge", "--grader", "--grader-loops", "6"])
        self.assertEqual(args.grader, "prediction_market")
        self.assertEqual(args.grader_loops, 6)
        self.assertNotIn("--evaluator", build_parser().format_help())
        retriever_action = next(action for action in build_parser()._actions if action.dest == "retriever")
        self.assertNotIn("memory", retriever_action.choices or ())

    def test_guided_cli_can_select_an_official_grader(self) -> None:
        args = build_parser().parse_args([])
        answers = iter(["Optimize the PM challenge.", "2", "3", "", "", ""])

        configured = configure_interactive_run(
            args,
            input_func=lambda _prompt: next(answers),
            output_func=lambda _text: None,
            key_reader=None,
        )

        self.assertEqual(configured.grader, "prediction_market")
        self.assertEqual(configured.grader_loops, 3)

    def test_cli_accepts_a_separate_grader_loop_limit(self) -> None:
        args = build_parser().parse_args(["optimize pm challenge", "--grader", "prediction_market", "--grader-loops", "3"])
        self.assertEqual(args.grader_loops, 3)

    def test_cli_scales_implicit_budgets_for_more_grader_rounds(self) -> None:
        args = build_parser().parse_args(["optimize pm challenge", "--grader", "--grader-loops", "8"])
        args.max_iterations_explicit = False
        args.max_runtime_seconds_explicit = False

        _apply_grader_budget_defaults(args)

        self.assertEqual(args.max_iterations, 28)
        self.assertEqual(args.max_runtime_seconds, 960.0)

    def test_cli_preserves_explicit_tight_grader_budgets(self) -> None:
        args = build_parser().parse_args([
            "optimize pm challenge", "--grader", "--grader-loops", "8",
            "--max-iterations", "12", "--max-runtime-seconds", "240",
        ])
        args.max_iterations_explicit = True
        args.max_runtime_seconds_explicit = True

        _apply_grader_budget_defaults(args)

        self.assertEqual(args.max_iterations, 12)
        self.assertEqual(args.max_runtime_seconds, 240.0)

    def test_grader_loop_limit_blocks_a_second_official_candidate_attempt(self) -> None:
        class CountingGrader:
            name = "evaluate_prediction_market_candidate"
            description = "test grader"
            is_read_only = False
            input_schema = {"type": "object", "required": [], "properties": {}, "additionalProperties": False}

            def __init__(self) -> None:
                self.calls = 0

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                self.calls += 1
                return ToolResult("ok", {"measured": True})

        tool = CountingGrader()
        decider = ScriptedDecider([
            {"type": "tool_call", "tool_name": tool.name, "arguments": {}},
            {"type": "tool_call", "tool_name": tool.name, "arguments": {}},
            {"type": "final", "answer": "One official evaluation was used."},
        ])
        agent = ResearchAgent(decider, ToolRegistry([tool]), AgentRunConfig(max_iterations=3, max_grader_calls=1))
        result = agent.run("Optimize the challenge.", workspace=Path.cwd())

        self.assertEqual(tool.calls, 1)
        self.assertEqual([call["status"] for call in result.tool_calls], ["ok", "skipped"])
        self.assertIn("--grader-loops=1", result.tool_calls[1]["error"])

    def test_grader_run_nudges_zero_execution_wandering_after_one_turn(self) -> None:
        class NoopTool:
            name = "inspect_context"
            description = "inspect context"
            is_read_only = True
            input_schema = {"type": "object", "required": [], "properties": {}, "additionalProperties": False}

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                return ToolResult("ok", {"inspected": True})

        class GraderTool(NoopTool):
            name = "evaluate_prediction_market_candidate"
            is_read_only = False

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                return ToolResult("ok", {"official_measured": True, "mean_edge": 1.0})

        decider = ScriptedDecider([
            {"type": "tool_call", "tool_name": "inspect_context", "arguments": {}},
            {"type": "tool_call", "tool_name": "evaluate_prediction_market_candidate", "arguments": {}},
            {"type": "final", "answer": "The candidate was officially measured."},
        ])
        result = ResearchAgent(
            decider,
            ToolRegistry([NoopTool(), GraderTool()]),
            AgentRunConfig(max_iterations=3, max_grader_calls=1),
        ).run("Adapter baseline is already available.", workspace=Path.cwd())

        second_turn_messages = decider.observed_messages[1]
        nudges = [
            message["content"] for message in second_turn_messages
            if message["role"] == "user" and "zero grader attempts" in str(message.get("content"))
        ]
        self.assertEqual(len(nudges), 1)
        self.assertIn("Stop trying to locate challenge files", nudges[0])
        self.assertTrue(result.tool_calls[1]["official_measured"])
        self.assertEqual(result.status, "completed")

    def test_non_grader_run_never_receives_a_grader_action_nudge(self) -> None:
        decider = ScriptedDecider([{"type": "final", "answer": "Done."}])
        result = ResearchAgent(decider, ToolRegistry([]), AgentRunConfig(max_iterations=1)).run(
            "Rewrite this.", workspace=Path.cwd()
        )

        self.assertFalse(any(
            "zero grader attempts" in str(message.get("content"))
            for messages in decider.observed_messages for message in messages
        ))
        self.assertEqual(result.status, "completed")

    def test_requested_grader_trials_keep_the_model_in_the_feedback_loop(self) -> None:
        class NegativeGrader:
            name = "evaluate_prediction_market_candidate"
            description = "test grader"
            is_read_only = False
            input_schema = {"type": "object", "required": [], "properties": {}, "additionalProperties": False}

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                return ToolResult("ok", {"mean_edge": -0.25, "mean_arb_edge": -0.5})

        decider = ScriptedDecider([
            {"type": "tool_call", "tool_name": "evaluate_prediction_market_candidate", "arguments": {}},
            {"type": "tool_call", "tool_name": "evaluate_prediction_market_candidate", "arguments": {}},
            {"type": "final", "answer": "Both requested candidates were measured."},
        ])
        result = ResearchAgent(
            decider, ToolRegistry([NegativeGrader()]), AgentRunConfig(max_iterations=3, max_grader_calls=2)
        ).run("Optimize the challenge.", workspace=Path.cwd())

        self.assertEqual([call["status"] for call in result.tool_calls], ["ok", "ok"])
        feedback = decider.observed_messages[1][-1]
        self.assertEqual(feedback["role"], "user")
        self.assertIn("non-positive (-0.25)", feedback["content"])
        self.assertIn("1 requested official evaluation", feedback["content"])
        self.assertIn("fresh, high-relevance evidence", feedback["content"])

    def test_grader_context_projects_state_without_replaying_old_tool_calls(self) -> None:
        code = "class Strategy(BaseStrategy):\n    def on_step(self, state):\n        return []\n"

        class Grader:
            name, description, is_read_only = "evaluate_prediction_market_candidate", "grader", False
            input_schema = {"type": "object", "required": ["code"], "properties": {"code": {"type": "string"}}, "additionalProperties": False}

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                return ToolResult("ok", {
                    "candidate_id": "candidate_a", "official_measured": True, "score_eligible": True,
                    "mean_edge": 1.25, "mean_arb_edge": 0.75, "mean_retail_edge": 0.5,
                    "trial_path": "optimization_trials/candidate_a.json",
                    "promotion": {"promoted_this_round": True},
                })

        class Fetch:
            name, description, is_read_only = "fetch_document", "fetch", True
            input_schema = {"type": "object", "required": [], "properties": {}, "additionalProperties": False}

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                return ToolResult("ok", {"content": "paper"}, source_metadata=[{
                    "id": "paper", "url": "https://arxiv.org/pdf/1009.1446v1", "title": "Prediction market paper",
                    "source_type": "fetched_document", "summary": "Inventory risk evidence.",
                    "relevance_score": 1.0, "credibility_score": 0.8, "evidence_kind": "verified_document",
                    "evidence_sections": {"page_1": "Inventory risk changes optimal market-making quotes."},
                    "evidence_locators": {"page_1": [{"kind": "pdf_page", "page": 1}]},
                }])

        decider = ScriptedDecider([
            {"type": "tool_call", "tool_name": "evaluate_prediction_market_candidate", "arguments": {"code": code}},
            {"type": "tool_call", "tool_name": "fetch_document", "arguments": {}},
            {"type": "final", "answer": "The official candidate and fetched paper were retained. https://arxiv.org/pdf/1009.1446v1"},
        ])
        result = ResearchAgent(
            decider, ToolRegistry([Grader(), Fetch()]), AgentRunConfig(max_iterations=3, max_grader_calls=1)
        ).run("Optimize with literature.", workspace=Path.cwd())

        third_context = decider.observed_messages[2]
        _validate_tool_history(third_context)
        checkpoint = next(message["content"] for message in third_context if "Harness working-state checkpoint" in str(message.get("content")))
        self.assertIn(code, checkpoint)
        self.assertIn("1.250000", checkpoint)
        self.assertIn("Inventory risk changes optimal market-making quotes", checkpoint)
        self.assertFalse(any(
            call.get("name") == "evaluate_prediction_market_candidate"
            for message in third_context for call in message.get("tool_calls") or []
        ))
        self.assertTrue(any(
            call.get("name") == "fetch_document"
            for message in third_context for call in message.get("tool_calls") or []
        ))
        self.assertTrue(any(
            call.get("name") == "evaluate_prediction_market_candidate"
            for message in result.messages for call in message.get("tool_calls") or []
        ))

    def test_context_projection_does_not_promote_failed_trials_or_search_leads(self) -> None:
        state = AgentState.initialize(
            objective="Optimize.", context=ToolContext(workspace=Path.cwd()), initial_cost=0.0,
            system_messages=["system"],
        )
        state.begin_iteration(3)
        code = "class Strategy(BaseStrategy):\n    pass\n"
        state.messages.extend([
            {"role": "assistant", "content": "try", "tool_calls": [{"id": "failed", "name": "evaluate_prediction_market_candidate", "arguments": {"code": code}}]},
            {"role": "tool", "tool_call_id": "failed", "name": "evaluate_prediction_market_candidate", "content": {"status": "error", "data": {"mean_edge": 99.0}, "error": "scorer failed"}},
        ])
        state.tool_calls.append({
            "id": "failed", "tool": "evaluate_prediction_market_candidate", "status": "error",
            "official_measured": False, "error": "scorer failed",
        })
        state.sources.append({
            "url": "https://example.org/lead", "title": "Search lead", "summary": "Unfetched snippet",
            "evidence_kind": "lead", "relevance_score": 1.0,
        })

        projection = WorkingStateProjector().project(state, max_grader_calls=2)
        checkpoint = next(message["content"] for message in projection.messages if "Harness working-state checkpoint" in str(message.get("content")))

        self.assertIn("error (not eligible)", checkpoint)
        self.assertNotIn("Current champion strategy", checkpoint)
        self.assertNotIn("Unfetched snippet", checkpoint)
        self.assertEqual(projection.fetched_document_count, 0)

    def test_completed_grader_budget_tells_model_to_finish_without_extra_trials(self) -> None:
        class Grader:
            name = "evaluate_prediction_market_candidate"
            description = "test grader"
            is_read_only = False
            input_schema = {"type": "object", "required": [], "properties": {}, "additionalProperties": False}

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                return ToolResult("ok", {"official_measured": True, "mean_edge": 1.0})

        decider = ScriptedDecider([
            {"type": "tool_call", "tool_name": "evaluate_prediction_market_candidate", "arguments": {}},
            {"type": "final", "answer": "The requested candidate was measured."},
        ])
        result = ResearchAgent(
            decider, ToolRegistry([Grader()]), AgentRunConfig(max_iterations=2, max_grader_calls=1)
        ).run("Optimize the challenge.", workspace=Path.cwd())

        completion_guidance = decider.observed_messages[1][-1]
        self.assertEqual(completion_guidance["role"], "user")
        self.assertIn("All requested official evaluations are complete", completion_guidance["content"])
        self.assertIn("Do not call evaluate_prediction_market_candidate again", completion_guidance["content"])
        self.assertEqual(result.status, "completed")

    def test_optimizer_system_prompt_requires_fresh_failure_specific_evidence(self) -> None:
        self.assertIn("after every scorer observation", _SYSTEM_INSTRUCTIONS)
        self.assertIn("score delta, component metric, failure trace", _SYSTEM_INSTRUCTIONS)
        self.assertIn("never broaden into an unrelated domain", _SYSTEM_INSTRUCTIONS)

    def test_harness_nudges_after_five_source_free_iterations(self) -> None:
        class NoSourceTool:
            name = "analyze"
            description = "analysis only"
            is_read_only = True
            input_schema = {"type": "object", "required": [], "properties": {}, "additionalProperties": False}

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                return ToolResult("ok", {"result": "no source"})

        decider = ScriptedDecider([
            *[{"type": "tool_call", "tool_name": "analyze", "arguments": {}} for _ in range(5)],
            {"type": "final", "answer": "Existing evidence is sufficient."},
        ])
        result = ResearchAgent(decider, ToolRegistry([NoSourceTool()]), AgentRunConfig(max_iterations=6)).run(
            "Optimize the challenge.", workspace=Path.cwd()
        )

        self.assertEqual(result.status, "completed")
        nudge_events = [event for event in result.events if event.event_type == "source_refresh_nudge"]
        self.assertEqual(len(nudge_events), 1)
        self.assertEqual(nudge_events[0].observation["iterations_without_new_sources"], 5)
        self.assertIn("Harness nudge", decider.observed_messages[5][-1]["content"])

    def test_harness_does_not_nudge_when_a_new_source_resets_the_counter(self) -> None:
        class AlternatingTool:
            name = "inspect"
            description = "returns a source only on its first call"
            is_read_only = True
            input_schema = {"type": "object", "required": [], "properties": {}, "additionalProperties": False}

            def __init__(self) -> None:
                self.calls = 0

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                self.calls += 1
                metadata = [{"url": "https://example.org/new", "title": "new"}] if self.calls == 1 else []
                return ToolResult("ok", {"result": self.calls}, source_metadata=metadata)

        decider = ScriptedDecider([
            *[{"type": "tool_call", "tool_name": "inspect", "arguments": {}} for _ in range(5)],
            {"type": "final", "answer": "Existing evidence is sufficient. https://example.org/new"},
        ])
        result = ResearchAgent(decider, ToolRegistry([AlternatingTool()]), AgentRunConfig(max_iterations=6)).run(
            "Optimize the challenge.", workspace=Path.cwd()
        )

        self.assertEqual(result.status, "completed")
        self.assertFalse(any(event.event_type == "source_refresh_nudge" for event in result.events))

    def test_search_tool_discards_low_relevance_discovery_hits(self) -> None:
        class MixedRelevanceBackend:
            tool_name = "mixed_search"

            def search(self, _query: str, limit: int = 4):
                return [("relevant", 0.75), ("noise", 0.45)][:limit]

            def to_source(self, document: str, relevance: float):
                from research_harness.schemas import Source

                return Source(
                    url=f"https://example.org/{document}", title=document, author="test", date="",
                    source_type="test", summary=document, relevance_score=relevance, credibility_score=0.8,
                )

        result = asyncio.run(SearchTool(MixedRelevanceBackend()).execute({"query": "prediction market", "limit": 4}, ToolContext(workspace=Path.cwd())))

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["result_count"], 1)
        self.assertEqual(result.data["discarded_low_relevance_count"], 1)
        self.assertEqual([source["title"] for source in result.source_metadata], ["relevant"])

    def test_final_answer_is_deferred_until_requested_grader_trials_are_used(self) -> None:
        class Grader:
            name = "evaluate_prediction_market_candidate"
            description = "test grader"
            is_read_only = False
            input_schema = {"type": "object", "required": [], "properties": {}, "additionalProperties": False}

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                return ToolResult("ok", {"mean_edge": 0.1})

        decider = ScriptedDecider([
            {"type": "final", "answer": "Stopping too soon."},
            {"type": "tool_call", "tool_name": "evaluate_prediction_market_candidate", "arguments": {}},
            {"type": "final", "answer": "Measured the requested candidate."},
        ])
        result = ResearchAgent(
            decider, ToolRegistry([Grader()]), AgentRunConfig(max_iterations=3, max_grader_calls=1)
        ).run("Optimize the challenge.", workspace=Path.cwd())

        self.assertEqual(result.final_answer, "Measured the requested candidate.")
        self.assertEqual([call["status"] for call in result.tool_calls], ["ok"])
        self.assertTrue(any(
            "Do not finalize while 1 requested official evaluation" in str(message.get("content"))
            for message in decider.observed_messages[1]
        ))

    def test_schema_rejected_grader_call_does_not_consume_measured_trial_budget(self) -> None:
        class Grader:
            name = "evaluate_prediction_market_candidate"
            description = "test grader"
            is_read_only = False
            input_schema = {
                "type": "object",
                "required": ["code"],
                "properties": {"code": {"type": "string", "minLength": 5}},
                "additionalProperties": False,
            }

            def __init__(self) -> None:
                self.executions = 0

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                self.executions += 1
                return ToolResult("ok", {"official_measured": True, "mean_edge": 1.0})

        grader = Grader()
        decider = ScriptedDecider([
            {"type": "tool_call", "tool_name": grader.name, "arguments": {}},
            {"type": "tool_call", "tool_name": grader.name, "arguments": {"code": "class Strategy: pass"}},
            {"type": "final", "answer": "One candidate was officially measured."},
        ])
        result = ResearchAgent(
            decider, ToolRegistry([grader]), AgentRunConfig(max_iterations=3, max_grader_calls=1)
        ).run("Optimize the challenge.", workspace=Path.cwd())

        self.assertEqual([call["status"] for call in result.tool_calls], ["error", "ok"])
        self.assertEqual([call["executed"] for call in result.tool_calls], [False, True])
        self.assertEqual([call["official_measured"] for call in result.tool_calls], [False, True])
        self.assertEqual(grader.executions, 1)
        self.assertEqual(result.status, "completed")

    def test_grader_guidance_never_interrupts_a_later_parallel_tool_result_batch(self) -> None:
        class Tool:
            description = "test tool"
            is_read_only = True
            input_schema = {"type": "object", "required": [], "properties": {}, "additionalProperties": False}

            def __init__(self, name: str):
                self.name = name

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                return ToolResult("ok", {"tool": self.name})

        class Grader(Tool):
            is_read_only = False

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                return ToolResult("ok", {"mean_edge": -1.0})

        class NativeScript:
            def __init__(self) -> None:
                self.observed_messages: list[list[dict[str, Any]]] = []
                self.turns = [
                    ModelTurn("", [ModelToolCall("grader_1", "evaluate_prediction_market_candidate", {})], "tool_calls", "test", "test"),
                    ModelTurn("", [ModelToolCall("search_1", "search_one", {}), ModelToolCall("search_2", "search_two", {})], "tool_calls", "test", "test"),
                    ModelTurn("", [ModelToolCall("grader_2", "evaluate_prediction_market_candidate", {})], "tool_calls", "test", "test"),
                    ModelTurn("Measured both candidates.", [], "stop", "test", "test"),
                ]

            def decide(self, messages: list[dict[str, Any]], _tools: list[dict[str, Any]]) -> ModelTurn:
                self.observed_messages.append(list(messages))
                return self.turns.pop(0)

        decider = NativeScript()
        result = ResearchAgent(
            decider,
            ToolRegistry([Grader("evaluate_prediction_market_candidate"), Tool("search_one"), Tool("search_two")]),
            AgentRunConfig(max_iterations=4, max_grader_calls=2),
        ).run("Optimize the challenge.", workspace=Path.cwd())

        messages_before_second_grader = decider.observed_messages[2]
        parallel_call_index = next(
            index for index, message in enumerate(messages_before_second_grader)
            if [call["id"] for call in message.get("tool_calls") or []] == ["search_1", "search_2"]
        )
        self.assertEqual(
            [message["role"] for message in messages_before_second_grader[parallel_call_index:parallel_call_index + 3]],
            ["assistant", "tool", "tool"],
        )
        followups = [
            message for message in messages_before_second_grader
            if message["role"] == "user" and "requested official evaluation(s) remain" in str(message.get("content"))
        ]
        self.assertEqual(len(followups), 1)
        self.assertEqual(result.status, "completed")

    def test_multiple_read_only_model_requested_tools_run_concurrently(self) -> None:
        class DelayedTool:
            is_read_only = True
            input_schema = {"type": "object", "required": [], "properties": {}, "additionalProperties": False}

            def __init__(self, name: str):
                self.name, self.description = name, name

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                await asyncio.sleep(0.08)
                return ToolResult("ok", {"name": self.name})

        class NativeScript:
            def __init__(self) -> None:
                self.turns = [
                    ModelTurn("", [ModelToolCall("a", "first", {}), ModelToolCall("b", "second", {})], "tool_calls", "test", "test"),
                    ModelTurn("Done.", [], "stop", "test", "test"),
                ]

            def decide(self, _messages: Any, _tools: Any) -> ModelTurn:
                return self.turns.pop(0)

        agent = ResearchAgent(NativeScript(), ToolRegistry([DelayedTool("first"), DelayedTool("second")]), AgentRunConfig(max_iterations=3))
        started = time.monotonic()
        result = agent.run("Do two independent reads.", workspace=Path.cwd())
        self.assertLess(time.monotonic() - started, 0.14)
        self.assertEqual(result.status, "completed")
        self.assertEqual([call["id"] for call in result.tool_calls], ["a", "b"])
        self.assertEqual(
            [message["tool_call_id"] for message in result.messages if message["role"] == "tool"],
            ["a", "b"],
        )
        self.assertEqual(
            [event.tool_call_id for event in result.events if event.event_type == "tool_result"],
            ["a", "b"],
        )

    def test_unsupported_citation_is_returned_for_revision(self) -> None:
        decider = ScriptedDecider([
            {"type": "tool_call", "tool_name": "local_corpus_search", "arguments": {"query": "multi-agent systems", "limit": 1}},
            {"type": "final", "answer": "Unsupported claim https://invalid.example/not-retrieved"},
            {"type": "final", "answer": "Supported claim https://example.org/single-agent-baseline-limitations"},
        ])
        result = self._agent(decider).run("Find evidence.", workspace=Path.cwd())
        self.assertEqual(result.status, "partial")
        feedback = [event for event in result.events if event.event_type == "final_validation"]
        self.assertEqual(feedback[0].observation["status"], "REVISE")

    def test_length_limited_turn_is_continued_not_accepted_as_final(self) -> None:
        class LengthThenFinal:
            def __init__(self) -> None:
                self.turns = [
                    ModelTurn("First half.", [], "length", "test", "test"),
                    ModelTurn("Second half.", [], "stop", "test", "test"),
                ]

            def decide(self, _messages: Any, _tools: Any) -> ModelTurn:
                return self.turns.pop(0)

        result = ResearchAgent(LengthThenFinal(), ToolRegistry([]), AgentRunConfig(max_iterations=3)).run("Explain something.", workspace=Path.cwd())
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.final_answer, "First half.\n\nSecond half.")

    def test_budget_rejected_calls_receive_matching_tool_messages(self) -> None:
        class TwoCallsThenFinal:
            def __init__(self) -> None:
                self.turns = [
                    ModelTurn("Need two reads.", [ModelToolCall("first", "known", {}), ModelToolCall("second", "known", {})], "tool_calls", "test", "test"),
                    ModelTurn("Grounded answer.", [], "stop", "test", "test"),
                ]
                self.second_messages: list[dict[str, Any]] = []

            def decide(self, messages: Any, _tools: Any) -> ModelTurn:
                if len(self.turns) == 1:
                    self.second_messages = list(messages)
                return self.turns.pop(0)

        class KnownTool:
            name, description, is_read_only = "known", "known", True
            input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                return ToolResult("ok")

        decider = TwoCallsThenFinal()
        result = ResearchAgent(decider, ToolRegistry([KnownTool()]), AgentRunConfig(max_iterations=3, max_tool_calls=1)).run("Use tools.", workspace=Path.cwd())
        responses = [message for message in decider.second_messages if message.get("role") == "tool"]
        self.assertEqual(result.status, "completed")
        self.assertEqual({message["tool_call_id"] for message in responses}, {"first", "second"})
        self.assertEqual(result.tool_calls[1]["status"], "skipped")

    def test_retryable_tool_errors_do_not_consume_evidence_budget(self) -> None:
        class RecoveringTool:
            name, description, is_read_only = "recovering", "recovering", True
            input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

            def __init__(self) -> None:
                self.attempts = 0

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                self.attempts += 1
                if self.attempts == 1:
                    return ToolResult("error", error="temporary search backend failure", retryable=True)
                return ToolResult("ok", {"evidence": "retrieved"})

        class ErrorThenRecover:
            def __init__(self) -> None:
                self.turns = [
                    ModelTurn("Try discovery.", [ModelToolCall("one", "recovering", {})], "tool_calls", "test", "test"),
                    ModelTurn("Try an alternative route.", [ModelToolCall("two", "recovering", {})], "tool_calls", "test", "test"),
                    ModelTurn("Grounded result.", [], "stop", "test", "test"),
                ]

            def decide(self, _messages: Any, _tools: Any) -> ModelTurn:
                return self.turns.pop(0)

        result = ResearchAgent(ErrorThenRecover(), ToolRegistry([RecoveringTool()]), AgentRunConfig(max_iterations=4, max_tool_calls=1)).run("Recover from a temporary source failure.", workspace=Path.cwd())

        self.assertEqual(result.status, "completed")
        self.assertEqual([call["status"] for call in result.tool_calls], ["error", "ok"])

    def test_empty_discovery_does_not_consume_evidence_budget(self) -> None:
        class EmptyThenEvidenceTool:
            name, description, is_read_only = "discovery", "discovery", True
            input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

            def __init__(self) -> None:
                self.attempts = 0

            async def execute(self, _arguments: dict[str, Any], _context: ToolContext) -> ToolResult:
                self.attempts += 1
                if self.attempts == 1:
                    return ToolResult("ok", {"result_count": 0})
                return ToolResult("ok", {"result_count": 1}, source_metadata=[{"url": "https://example.org/evidence"}])

        class TwoDiscoveryPasses:
            def __init__(self) -> None:
                self.turns = [
                    ModelTurn("Try one source.", [ModelToolCall("one", "discovery", {})], "tool_calls", "test", "test"),
                    ModelTurn("Try a second source.", [ModelToolCall("two", "discovery", {})], "tool_calls", "test", "test"),
                    ModelTurn("Grounded result https://example.org/evidence", [], "stop", "test", "test"),
                ]

            def decide(self, _messages: Any, _tools: Any) -> ModelTurn:
                return self.turns.pop(0)

        result = ResearchAgent(TwoDiscoveryPasses(), ToolRegistry([EmptyThenEvidenceTool()]), AgentRunConfig(max_iterations=4, max_tool_calls=1)).run("Find external evidence.", workspace=Path.cwd())

        self.assertEqual(result.status, "completed")
        self.assertEqual([call["status"] for call in result.tool_calls], ["ok", "ok"])

    def test_html_figure_parser_preserves_caption_and_image_url(self) -> None:
        parser = _FigureHTMLParser("https://example.org/paper")
        parser.feed('<figure class="ltx_figure"><img src="images/plot.png"><figcaption>Figure 2: Agent performance over time.</figcaption></figure>')

        self.assertEqual(parser.figures, [{"image_url": "https://example.org/images/plot.png", "caption": "Figure 2: Agent performance over time."}])

    def test_figure_inspection_registers_direct_figure_assets_for_citation(self) -> None:
        tool = DocumentFigureTool()
        inspected = ToolResult("ok", {
            "source_url": "https://example.org/paper",
            "figures": [{"image_url": "https://example.org/figures/one.png", "caption": "Figure 1: Verified result."}],
        })
        with mock.patch.object(tool, "_inspect", return_value=inspected):
            result = asyncio.run(tool.execute({"url": "https://example.org/paper"}, ToolContext(workspace=Path.cwd())))

        self.assertEqual(result.status, "ok")
        self.assertEqual(
            {source["url"] for source in result.source_metadata},
            {"https://example.org/paper", "https://example.org/figures/one.png"},
        )

    def test_image_dimensions_reads_png_header(self) -> None:
        png = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + (1200).to_bytes(4, "big") + (800).to_bytes(4, "big")
        self.assertEqual(_image_dimensions(png), (1200, 800))

    def test_arxiv_query_preserves_exact_ids_and_filters_irrelevant_results(self) -> None:
        self.assertEqual(_arxiv_identifier("Concrete Problems in AI Safety 1606.06565"), "1606.06565")
        self.assertNotIn("%", _arxiv_query("Stochastic Parrots Bender Gebru 2021"))
        payload = b'''<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2607.08562v1</id><title>XShooter DESI Lens Program</title><summary>Astronomy lens observations.</summary><published>2026-07-09T00:00:00Z</published><author><name>Author</name></author><category term="astro-ph"/></entry></feed>'''

        response = mock.MagicMock()
        response.read.return_value = payload
        response.__enter__.return_value = response
        with mock.patch("research_harness.search.urllib.request.urlopen", return_value=response) as urlopen:
            results = ArxivSearch().search("Concrete Problems in AI Safety Amodei 2016", limit=5)

        self.assertEqual(results, [])
        self.assertNotIn("%25", urlopen.call_args.args[0].full_url)

    def test_arxiv_retrieval_query_drops_figure_format_noise(self) -> None:
        self.assertEqual(
            _retrieval_query("FunSearch figure chart algorithm discovery performance"),
            "funsearch algorithm discovery",
        )

    def test_arxiv_abstract_fetch_resolves_to_pdf(self) -> None:
        self.assertEqual(
            _arxiv_pdf_url("http://arxiv.org/abs/1009.1446v1"),
            "https://arxiv.org/pdf/1009.1446v1",
        )
        self.assertIsNone(_arxiv_pdf_url("https://example.org/abs/1009.1446v1"))

    def test_pdf_fetch_uses_document_byte_limit_not_text_character_limit(self) -> None:
        payload = b"%PDF-1.7\n" + b"x" * 50_000

        class Response:
            headers = {"Content-Type": "application/octet-stream"}
            status = 200

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_args: Any) -> None:
                return None

            def read(self, limit: int) -> bytes:
                self.limit = limit
                return payload

        response = Response()
        opener = mock.Mock()
        opener.open.return_value = response
        tool = WebFetchTool(max_characters=20_000, max_document_bytes=100_000)
        with mock.patch("research_harness.tools.research._public_url_error", return_value=None), mock.patch(
            "research_harness.tools.research.urllib.request.build_opener", return_value=opener
        ):
            result = tool._fetch("http://arxiv.org/abs/1009.1446v1", prefer_markdown=True)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["url"], "https://arxiv.org/pdf/1009.1446v1")
        self.assertEqual(result.data["raw_content"], payload)
        self.assertEqual(response.limit, 100_001)

    def test_html_fetch_uses_download_limit_then_truncates_extracted_text(self) -> None:
        payload = b"<html><body>" + b"useful evidence " * 3_000 + b"</body></html>"

        class Response:
            headers = {"Content-Type": "text/html; charset=utf-8"}

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_args: Any) -> None:
                return None

            def read(self, limit: int) -> bytes:
                self.limit = limit
                return payload

        response = Response()
        opener = mock.Mock()
        opener.open.return_value = response
        tool = WebFetchTool(max_characters=20_000, max_document_bytes=100_000)
        with mock.patch("research_harness.tools.research._public_url_error", return_value=None), mock.patch(
            "research_harness.tools.research.urllib.request.build_opener", return_value=opener
        ), mock.patch.object(tool, "_fetch_curl_markdown", return_value=None):
            fetched = tool._fetch("https://www.paradigm.xyz/writing/pm-amm", prefer_markdown=True)

        self.assertEqual(fetched.status, "ok")
        self.assertEqual(response.limit, 100_001)
        with mock.patch.object(tool, "_fetch", return_value=ToolResult("ok", {
            "url": "https://www.paradigm.xyz/writing/pm-amm",
            "content_type": "text/html; charset=utf-8",
            "raw_content": payload,
            "truncated": False,
            "renderer": "direct",
        })):
            ingested = asyncio.run(tool.execute(
                {"url": "https://www.paradigm.xyz/writing/pm-amm", "prefer_markdown": True},
                ToolContext(workspace=Path.cwd()),
            ))
        self.assertEqual(ingested.status, "ok")
        retained_characters = sum(len(value) for value in ingested.data["evidence_sections"].values())
        self.assertLessEqual(retained_characters, 20_000)
        self.assertTrue(ingested.data["truncated"])

    def test_pdf_fetch_rejects_oversized_document_instead_of_parsing_a_prefix(self) -> None:
        class Response:
            headers = {"Content-Type": "application/pdf"}

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_args: Any) -> None:
                return None

            def read(self, _limit: int) -> bytes:
                return b"%PDF" + b"x" * 101

        opener = mock.Mock()
        opener.open.return_value = Response()
        tool = WebFetchTool(max_document_bytes=100)
        with mock.patch("research_harness.tools.research._public_url_error", return_value=None), mock.patch(
            "research_harness.tools.research.urllib.request.build_opener", return_value=opener
        ):
            result = tool._fetch("https://arxiv.org/abs/1009.1446v1", prefer_markdown=True)

        self.assertEqual(result.status, "error")
        self.assertIn("exceeded", result.error or "")

    def test_fetch_document_reuses_cached_verified_arxiv_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            source = Source(
                url="https://arxiv.org/pdf/1009.1446v1", title="Fetched paper", author="arxiv.org", date="",
                source_type="fetched_document", summary="Extracted paper text.", relevance_score=1.0,
                credibility_score=0.8, evidence_kind="verified_document",
                evidence_sections={"page_1": "Exact extracted PDF evidence."},
                evidence_locators={"page_1": [{"kind": "pdf_page", "page": 1}]},
            )
            store.commit_tool_sources([source.__dict__])
            tool = WebFetchTool()
            with mock.patch.object(tool, "_fetch", side_effect=AssertionError("network fetch should not run")):
                result = asyncio.run(tool.execute(
                    {"url": "http://arxiv.org/abs/1009.1446v1", "prefer_markdown": True},
                    ToolContext(workspace=Path.cwd(), store=store),
                ))

        self.assertEqual(result.status, "ok")
        self.assertTrue(result.data["cached"])
        self.assertEqual(result.data["renderer"], "artifact_cache")
        self.assertIn("Exact extracted PDF evidence", result.data["content"])

    def test_figure_inspection_uses_arxiv_pdf_when_html_is_unavailable(self) -> None:
        self.assertEqual(
            _inspection_urls("https://arxiv.org/abs/2304.03442v2"),
            [
                "https://arxiv.org/html/2304.03442v2",
                "https://arxiv.org/pdf/2304.03442v2",
                "https://arxiv.org/abs/2304.03442v2",
            ],
        )

    def test_web_search_reports_duckduckgo_bot_challenge(self) -> None:
        response = mock.MagicMock()
        response.read.return_value = b"<html><div class='anomaly-modal'>Unfortunately, bots use DuckDuckGo too.</div></html>"
        response.__enter__.return_value = response
        with mock.patch("research_harness.search.urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "bot challenge"):
                WebSearch().search("AI safety")

    def test_external_source_objective_cannot_pass_without_sources(self) -> None:
        validation = FinalAnswerValidator().validate("A confident but uncited answer.", "Use external sources to explain AGI limitations.", [])
        self.assertEqual(validation.status, "revise")
        self.assertIn("external evidence", validation.feedback)

    def test_generic_source_handoff_is_not_accepted_as_final_research_answer(self) -> None:
        validation = FinalAnswerValidator().validate(
            "Pick one: Option A: send 5 candidate URLs. Option B: give permission to use another discovery source.",
            "Use external sources to find figures.",
            [],
        )
        self.assertEqual(validation.status, "revise")
        self.assertIn("generic request", validation.feedback)

    def test_citation_validation_normalizes_http_and_https(self) -> None:
        validation = FinalAnswerValidator().validate(
            "See https://arxiv.org/abs/1606.06565v2.",
            "Use external sources.",
            [{"url": "http://arxiv.org/abs/1606.06565v2"}],
        )
        self.assertEqual(validation.status, "pass")

    def test_length_continuation_does_not_break_url(self) -> None:
        self.assertEqual(
            _join_answer_chunks(["Source: https://arxiv.org", "/abs/1606.06565v2" ]),
            "Source: https://arxiv.org/abs/1606.06565v2",
        )

    def test_incomplete_evidence_packet_preserves_retrieved_summary(self) -> None:
        report = _partial_synthesis(
            [{"title": "Primary result", "url": "https://example.org/result", "summary": "The source contains the directly retrieved result."}],
            None,
        )

        self.assertIn("Incomplete evidence packet", report)
        self.assertIn("directly retrieved result", report)
        self.assertIn("https://example.org/result", report)

    def test_incomplete_report_ranks_verified_sources_and_retains_relevant_late_leads(self) -> None:
        sources = [
            {
                "title": f"Early lead {index}", "url": f"https://example.org/{index}",
                "summary": "Generic result.", "evidence_kind": "lead", "relevance_score": 0.5,
            }
            for index in range(15)
        ]
        sources.extend([
            {
                "title": "Market Making in Prediction Markets", "url": "https://www.quantvps.com/market-making",
                "summary": "Inventory risk and optimal quotes.", "evidence_kind": "lead", "relevance_score": 0.7,
            },
            {
                "title": "Fetched paper", "url": "https://arxiv.org/pdf/1009.1446v1",
                "summary": "Extracted paper text.", "evidence_kind": "verified_document", "relevance_score": 1.0,
            },
        ])

        report = _partial_synthesis(sources, "Wall-clock runtime budget exhausted.")

        self.assertLess(report.index("Fetched paper"), report.index("Market Making in Prediction Markets"))
        self.assertIn("**verified document:**", report)
        self.assertIn("**discovery lead:**", report)
        self.assertIn("quantvps.com", report)

    def test_controller_can_consult_a_model_chosen_specialist(self) -> None:
        class SpecialistLLM:
            model_label = "specialist-test"

            def complete(self, _system: str, _prompt: str, **_kwargs: Any):
                from research_harness.llm import LLMResponse
                return LLMResponse("Evidence is weak because the baseline is mismatched.", "specialist-test", "test", 12, 8, 0.01)

        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "run")
            tool = SpecialistConsultationTool(SpecialistLLM())
            result = asyncio.run(tool.execute(
                {"specialty": "evidence critic", "question": "Assess the baseline.", "evidence": ["Model A beat Model B."]},
                ToolContext(workspace=Path.cwd(), store=store, run_id="run_specialist"),
            ))
            self.assertEqual(result.status, "ok")
            self.assertIn("baseline is mismatched", result.data["response"])
            self.assertEqual(store.list("agent_traces")[0]["role"], "specialist_consultation")


if __name__ == "__main__":
    unittest.main()
