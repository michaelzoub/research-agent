from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import shutil
import subprocess
import unittest
from dataclasses import dataclass
from typing import Optional

from research_harness.terminal_bench import ResearchHarnessTerminalBenchAgent, TerminalBenchRunConfig


TERMINAL_BENCH_DATASET = "terminal-bench/terminal-bench-2"
CURRENT_HARBOR_TERMINAL_BENCH_DATASET = "terminal-bench@2.0"


@dataclass(frozen=True)
class HarborCommand:
    argv: tuple[str, ...]

    def shell(self) -> str:
        return " ".join(self.argv)


def terminal_bench_oracle_command(dataset: str = TERMINAL_BENCH_DATASET) -> HarborCommand:
    return HarborCommand(("harbor", "run", "-d", dataset, "-a", "oracle"))


def terminal_bench_claude_code_daytona_command(
    *,
    dataset: str = TERMINAL_BENCH_DATASET,
    model: str = "anthropic/claude-haiku-4-5",
    n: int = 32,
) -> HarborCommand:
    return HarborCommand(
        (
            "harbor",
            "run",
            "-d",
            dataset,
            "-m",
            model,
            "-a",
            "claude-code",
            "--env",
            "daytona",
            "-n",
            str(n),
        )
    )


def terminal_bench_custom_agent_command(
    *,
    agent_import_path: str,
    dataset: str = TERMINAL_BENCH_DATASET,
    model: Optional[str] = None,
    n: Optional[int] = None,
) -> HarborCommand:
    argv = ["harbor", "run", "-d", dataset, "--agent-import-path", agent_import_path]
    if model:
        argv.extend(["-m", model])
    if n is not None:
        argv.extend(["-n", str(n)])
    return HarborCommand(tuple(argv))


class TerminalBenchHarborContractTest(unittest.TestCase):
    def test_oracle_command_matches_terminal_bench_docs(self) -> None:
        command = terminal_bench_oracle_command()

        self.assertEqual(
            command.argv,
            ("harbor", "run", "-d", "terminal-bench/terminal-bench-2", "-a", "oracle"),
        )
        self.assertEqual(command.shell(), "harbor run -d terminal-bench/terminal-bench-2 -a oracle")

    def test_current_harbor_dataset_alias_is_documented(self) -> None:
        command = terminal_bench_oracle_command(dataset=CURRENT_HARBOR_TERMINAL_BENCH_DATASET)

        self.assertEqual(command.shell(), "harbor run -d terminal-bench@2.0 -a oracle")

    def test_claude_code_daytona_command_matches_terminal_bench_docs(self) -> None:
        command = terminal_bench_claude_code_daytona_command()

        self.assertEqual(command.argv[0:4], ("harbor", "run", "-d", "terminal-bench/terminal-bench-2"))
        self.assertIn("-m", command.argv)
        self.assertIn("anthropic/claude-haiku-4-5", command.argv)
        self.assertIn("-a", command.argv)
        self.assertIn("claude-code", command.argv)
        self.assertIn("--env", command.argv)
        self.assertIn("daytona", command.argv)
        self.assertEqual(command.argv[-2:], ("-n", "32"))

    def test_daytona_run_requires_expected_environment_variables(self) -> None:
        required = {"DAYTONA_API_KEY", "ANTHROPIC_API_KEY"}
        documented_export_lines = {
            'export DAYTONA_API_KEY="<your-daytona-api-key>"',
            'export ANTHROPIC_API_KEY="<your-anthropic-api-key>"',
        }

        self.assertEqual(required, {line.split()[1].split("=")[0] for line in documented_export_lines})

    def test_custom_agent_command_uses_agent_import_path(self) -> None:
        command = terminal_bench_custom_agent_command(
            agent_import_path="research_harness.terminal_bench:ResearchHarnessTerminalBenchAgent",
            dataset=CURRENT_HARBOR_TERMINAL_BENCH_DATASET,
            model="openai/gpt-5.2",
            n=4,
        )

        self.assertEqual(command.argv[0:4], ("harbor", "run", "-d", "terminal-bench@2.0"))
        self.assertIn("--agent-import-path", command.argv)
        self.assertIn("research_harness.terminal_bench:ResearchHarnessTerminalBenchAgent", command.argv)
        self.assertEqual(command.argv[-2:], ("-n", "4"))


class TerminalBenchCustomAgentContractTest(unittest.TestCase):
    def test_agent_import_path_resolves_to_harbor_external_agent(self) -> None:
        module_name, class_name = "research_harness.terminal_bench:ResearchHarnessTerminalBenchAgent".split(":")
        module = importlib.import_module(module_name)
        agent_cls = getattr(module, class_name)

        self.assertIs(agent_cls, ResearchHarnessTerminalBenchAgent)
        self.assertEqual(agent_cls.name(), "research-harness")
        self.assertTrue(callable(getattr(agent_cls, "version")))
        self.assertTrue(inspect.iscoroutinefunction(agent_cls.setup))
        self.assertTrue(inspect.iscoroutinefunction(agent_cls.run))

    def test_external_agent_setup_checks_cli_inside_environment(self) -> None:
        environment = FakeTerminalBenchEnvironment()
        agent = ResearchHarnessTerminalBenchAgent()

        asyncio.run(agent.setup(environment))

        self.assertEqual(environment.commands, ["python3 -m research_harness.cli --help"])

    def test_external_agent_run_executes_harness_and_populates_context(self) -> None:
        environment = FakeTerminalBenchEnvironment(stdout="Run: run_terminal_bench_demo\nStatus: completed\n")
        context = FakeAgentContext()
        agent = ResearchHarnessTerminalBenchAgent(
            TerminalBenchRunConfig(llm_provider="local", llm_model="cheap-local", max_iterations=50, quiet=True)
        )

        asyncio.run(agent.run("Solve the terminal task, then write the answer.", environment, context))

        self.assertEqual(len(environment.commands), 1)
        command = environment.commands[0]
        self.assertIn("python3 -m research_harness.cli", command)
        self.assertIn("'Solve the terminal task, then write the answer.'", command)
        self.assertNotIn("--task-mode", command)
        self.assertIn("--llm-provider local", command)
        self.assertIn("--llm-model cheap-local", command)
        self.assertIn("--max-iterations 50", command)
        self.assertIn("--quiet", command)
        self.assertEqual(context.metadata["research_harness_command"], command)
        self.assertEqual(context.metadata["research_harness_exit_code"], 0)
        self.assertIn("Status: completed", context.metadata["research_harness_stdout"])

    def test_external_agent_run_records_failure_output_for_transcript_debugging(self) -> None:
        environment = FakeTerminalBenchEnvironment(returncode=2, stderr="missing dependency: docker")
        context = {}
        agent = ResearchHarnessTerminalBenchAgent(TerminalBenchRunConfig(llm_provider="local"))

        asyncio.run(agent.run("Diagnose failing shell task", environment, context))

        self.assertEqual(context["research_harness_exit_code"], 2)
        self.assertEqual(context["research_harness_stderr"], "missing dependency: docker")


@dataclass
class FakeCommandResult:
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


class FakeTerminalBenchEnvironment:
    def __init__(self, *, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.commands: list[str] = []
        self.result = FakeCommandResult(returncode=returncode, stdout=stdout, stderr=stderr)

    async def exec(self, command: str) -> FakeCommandResult:
        self.commands.append(command)
        return self.result


class FakeAgentContext:
    def __init__(self) -> None:
        self.metadata: dict[str, object] = {}


@unittest.skipUnless(os.environ.get("RUN_TERMINAL_BENCH") == "1", "Set RUN_TERMINAL_BENCH=1 to run live Harbor/Docker checks.")
class TerminalBenchLiveHarborSmokeTest(unittest.TestCase):
    def test_harbor_and_docker_are_available(self) -> None:
        self.assertIsNotNone(shutil.which("harbor"), "Harbor must be installed to run Terminal-Bench.")
        self.assertIsNotNone(shutil.which("docker"), "Docker must be installed and running to run Terminal-Bench.")
        completed = subprocess.run(["docker", "info"], text=True, capture_output=True, timeout=20)
        self.assertEqual(completed.returncode, 0, completed.stderr[-500:])

    def test_terminal_bench_oracle_command_starts(self) -> None:
        self.assertIsNotNone(shutil.which("harbor"), "Harbor must be installed to run Terminal-Bench.")
        command = terminal_bench_oracle_command(dataset=os.environ.get("TERMINAL_BENCH_DATASET", CURRENT_HARBOR_TERMINAL_BENCH_DATASET))
        completed = subprocess.run(command.argv, text=True, capture_output=True, timeout=300)
        self.assertEqual(completed.returncode, 0, (completed.stdout + completed.stderr)[-1000:])


if __name__ == "__main__":
    unittest.main()
