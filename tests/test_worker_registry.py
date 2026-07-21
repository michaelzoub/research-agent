import asyncio
import tempfile
import unittest
from pathlib import Path

from research_harness.llm import ModelTurn
from research_harness.store import ArtifactStore
from research_harness.tools import ToolContext, ToolRegistry
from research_harness.worker_registry import DelegateTaskTool, WorkerBudget, WorkerProfile, WorkerRegistry


class _LLM:
    total_prompt_tokens = 7
    total_completion_tokens = 3
    def total_cost(self): return 0.01


class _Decider:
    def __init__(self, answer="bounded findings", failure=None):
        self.llm, self.answer, self.failure = _LLM(), answer, failure
    async def decide(self, messages, tools):
        if self.failure: raise self.failure
        return ModelTurn(self.answer, [], "stop", "worker-model", "test", 7, 3)


class WorkerRegistryTests(unittest.TestCase):
    def _context(self, root):
        return ToolContext(workspace=root, readable_roots=[root], store=ArtifactStore(root / "parent"), run_id="parent_1")

    def test_resolution_and_unknown_profile(self):
        registry = WorkerRegistry([WorkerProfile("critic", "Review.")], lambda profile: _Decider())
        self.assertEqual(registry.resolve("critic").prompt, "Review.")
        with self.assertRaisesRegex(KeyError, "Unknown worker profile"):
            registry.resolve("missing")

    def test_delegate_returns_structured_isolated_result_and_nested_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workers = WorkerRegistry([WorkerProfile("critic", "Review.")], lambda profile: _Decider())
            result = asyncio.run(workers.delegate("critic", "Check this.", self._context(root), ToolRegistry([])))
            self.assertEqual(result.parent_run_id, "parent_1")
            self.assertTrue(result.worker_run_id.startswith("worker_"))
            self.assertEqual(result.findings, "bounded findings")
            self.assertEqual(result.total_tokens, 10)
            self.assertEqual(result.cost_usd, 0.01)
            self.assertTrue(Path(result.events_path).is_file())
            self.assertTrue((Path(result.artifacts_path) / "worker_result.json").is_file())

    def test_permission_scope_cannot_expand_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = WorkerProfile("bad", "Review.", readable_roots=(root.parent,))
            workers = WorkerRegistry([profile], lambda item: _Decider())
            with self.assertRaisesRegex(PermissionError, "contained"):
                asyncio.run(workers.delegate("bad", "Check.", self._context(root), ToolRegistry([])))

    def test_budgets_and_worker_count_are_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = WorkerProfile("tiny", "Review.", budget=WorkerBudget(max_iterations=0))
            workers = WorkerRegistry([profile], lambda item: _Decider(), max_workers_per_parent=1)
            result = asyncio.run(workers.delegate("tiny", "Check.", self._context(root), ToolRegistry([])))
            self.assertEqual(result.status, "budget_exhausted")
            with self.assertRaisesRegex(RuntimeError, "Worker-count limit"):
                asyncio.run(workers.delegate("tiny", "Again.", self._context(root), ToolRegistry([])))

    def test_token_and_runtime_budgets_terminate_before_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token_profile = WorkerProfile("tokens", "Review.", budget=WorkerBudget(max_tokens=0))
            runtime_profile = WorkerProfile("runtime", "Review.", budget=WorkerBudget(max_runtime_seconds=-1))
            workers = WorkerRegistry([token_profile, runtime_profile], lambda item: _Decider())
            token_result = asyncio.run(workers.delegate("tokens", "Check.", self._context(root), ToolRegistry([])))
            runtime_result = asyncio.run(workers.delegate("runtime", "Check.", self._context(root), ToolRegistry([])))
            self.assertEqual(token_result.status, "budget_exhausted")
            self.assertEqual(runtime_result.status, "budget_exhausted")

    def test_failure_propagates_through_delegate_tool_and_recursion_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workers = WorkerRegistry([WorkerProfile("broken", "Review.", budget=WorkerBudget(max_iterations=1))], lambda item: _Decider(failure=ValueError("boom")))
            parent_tools = ToolRegistry([])
            tool = DelegateTaskTool(workers, parent_tools)
            outcome = asyncio.run(tool.execute({"profile": "broken", "assignment": "Check."}, self._context(root)))
            self.assertEqual(outcome.status, "error")
            self.assertIn("Model error", outcome.error)
            recursive = self._context(root)
            recursive.delegation_depth = 1
            denied = asyncio.run(tool.execute({"profile": "broken", "assignment": "Check."}, recursive))
            self.assertIn("Recursive delegation", denied.error)


if __name__ == "__main__":
    unittest.main()
