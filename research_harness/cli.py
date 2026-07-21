from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import shutil
import sys
import termios
import tty
from importlib import metadata
from pathlib import Path
from typing import Callable, Optional

from optimization_graders import list_optimization_graders

from .evals.suites import SUITE_CHOICES
from .model_catalog import format_model_catalog, model_choices, resolve_model_selection
from .orchestrator import HarnessConfig, Orchestrator
from .schemas import AgentBudget


RETRIEVER_CHOICES = ("auto", "local", "arxiv", "openalex", "semantic_scholar", "github", "web", "docs_blogs", "twitter", "alchemy")
LLM_PROVIDER_CHOICES = ("auto", "openai", "anthropic", "kimi", "ollama", "local", "multi")
OPTIMIZATION_GRADER_CHOICES = list_optimization_graders()
DEFAULT_GRADER_LOOPS = 8

ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[38;5;35m",
    "teal": "\033[38;5;43m",
    "blue": "\033[38;5;39m",
    "gray": "\033[38;5;245m",
    "yellow": "\033[38;5;221m",
    "red": "\033[38;5;203m",
    "amber": "\033[38;5;214m",
    "gold": "\033[38;5;220m",
}

LOGO = r"""
    _         _                 
   / \  _   _| |_ ___  _ __ ___ 
  / _ \| | | | __/ _ \| '__/ _ \
 / ___ \ |_| | || (_) | | |  __/
/_/   \_\__,_|\__\___/|_|  \___|
"""

HELP_EPILOG = """
Examples:
  autore
  autore "Research how multi-agent systems improve literature review quality"
  autore "optimize pm challenge" --grader --grader-loops 8

Useful companions:
  autore --list-llm-models
  autore-eval --suite preflight
  autore-bench --outputs outputs
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Autore, the research and optimization agent harness. Use no arguments for the guided setup.",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("goal", nargs="?", help="High-level research goal. Omit to use the interactive run setup.")
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Open the selection-based run setup, using any supplied flags as defaults.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path(os.environ.get("RESEARCH_HARNESS_CORPUS_PATH", "examples/corpus/research_corpus.json")),
        help="Path to local deterministic search corpus.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("RESEARCH_HARNESS_OUTPUT_DIR", "outputs")),
        help="Directory where run artifacts are written.",
    )
    parser.add_argument(
        "--retriever",
        choices=RETRIEVER_CHOICES,
        default=os.environ.get("RESEARCH_HARNESS_RETRIEVER", "auto"),
        help="Evidence retriever/source mix. Auto uses a mixed strategy. Use local for the offline demo corpus.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum model turns. Unbounded when omitted.",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=48,
        help="Maximum evidence-producing external-tool calls. Failed or empty discovery calls remain visible but do not spend this evidence budget.",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=None,
        help="Wall-clock limit for the run, including source and figure inspection. Unbounded when omitted.",
    )
    parser.add_argument(
        "--grader",
        dest="grader",
        nargs="?",
        const="prediction_market",
        choices=OPTIMIZATION_GRADER_CHOICES,
        default=os.environ.get("RESEARCH_HARNESS_GRADER"),
        help="Enable the official prediction-market grader. An explicit registered id is optional.",
    )
    parser.add_argument(
        "--grader-loops",
        type=int,
        default=int(os.environ["RESEARCH_HARNESS_GRADER_LOOPS"]) if os.environ.get("RESEARCH_HARNESS_GRADER_LOOPS") else None,
        help="Requested number of official candidate evaluations for --grader. Model-turn and runtime limits remain unbounded unless explicitly supplied.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=LLM_PROVIDER_CHOICES,
        default=os.environ.get("RESEARCH_HARNESS_LLM_PROVIDER", "auto"),
        help="LLM provider for agent proposal, judging, and synthesis. Auto infers provider from --llm-model when possible.",
    )
    parser.add_argument(
        "--llm-model",
        default=os.environ.get("RESEARCH_HARNESS_LLM_MODEL", "openai/gpt-5.2"),
        help="LLM model id/name. Use provider/model ids like openai/gpt-5.2, anthropic/claude-sonnet-4-6, ollama/qwen3.5:latest, or all-configured.",
    )
    parser.add_argument(
        "--list-llm-models",
        action="store_true",
        help="Print the configured model catalog and exit.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not stream run progress to the terminal; artifacts are still written.",
    )
    parser.add_argument("--no-animations", action="store_true", help="Disable the live TTY progress animation; progress.txt remains unchanged.")
    parser.add_argument(
        "--session-projects-dir",
        type=Path,
        default=Path(os.environ["AUTORE_PROJECTS_DIR"]) if os.environ.get("AUTORE_PROJECTS_DIR") else None,
        help="Directory for plaintext session JSONL logs. Defaults to ~/.autore/projects/.",
    )
    parser.add_argument(
        "--resume-session",
        default=os.environ.get("AUTORE_RESUME_SESSION"),
        help="Record this run as a fresh session resumed from an existing session id.",
    )
    parser.add_argument(
        "--fork-session",
        default=os.environ.get("AUTORE_FORK_SESSION"),
        help="Record this run as a fresh session forked from an existing session id.",
    )
    parser.add_argument(
        "--no-sessions",
        action="store_true",
        help="Disable ~/.autore/projects session JSONL logging for this run.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        default=_env_truthy("AUTORE_PREFLIGHT_EVALS"),
        help="Run the selected preflight eval gate before starting autore.",
    )
    parser.add_argument(
        "--preflight-suite",
        choices=SUITE_CHOICES,
        default=os.environ.get("AUTORE_PREFLIGHT_SUITE", "preflight"),
        help="Eval suite to run when --preflight is selected.",
    )
    parser.add_argument(
        "--preflight-eval",
        action="append",
        default=[],
        dest="preflight_eval_ids",
        help="With --preflight, run only the selected eval id. May be repeated or comma-separated.",
    )
    return parser


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _use_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _paint(text: str, color: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{ANSI[color]}{text}{ANSI['reset']}"


def _print_cli_banner(
    *,
    output_func: Callable[[str], None] = print,
    color: Optional[bool] = None,
    compact: bool = False,
) -> None:
    use_color = _use_color() if color is None else color
    version = _package_version()
    if compact:
        output_func(
            f"{_paint(f'Autore Agent v{version}', 'gold', enabled=use_color)} "
            f"{_paint('research + optimize agent harness', 'gray', enabled=use_color)}"
        )
        return
    width = max(72, min(116, shutil.get_terminal_size((100, 24)).columns - 4))
    left_width = max(31, min(46, width // 2 - 2))
    right_width = width - left_width - 3
    header = f" Autore Agent v{version} "
    output_func(_paint("─" * max(1, left_width - 1), "amber", enabled=use_color) + _paint(header, "gold", enabled=use_color) + _paint("─" * max(0, width - left_width - len(header) + 1), "amber", enabled=use_color))
    border = "+" + "-" * left_width + "+" + "-" * right_width + "+"
    output_func(_paint(border, "amber", enabled=use_color))
    left_lines = _banner_left_lines(left_width)
    right_lines = _banner_right_lines(right_width)
    max_lines = max(len(left_lines), len(right_lines))
    for index in range(max_lines):
        left = left_lines[index] if index < len(left_lines) else ""
        right = right_lines[index] if index < len(right_lines) else ""
        output_func(
            _paint("|", "amber", enabled=use_color)
            + _banner_cell(left, left_width, use_color=use_color)
            + _paint("|", "amber", enabled=use_color)
            + _banner_cell(right, right_width, use_color=use_color)
            + _paint("|", "amber", enabled=use_color)
        )
    output_func(_paint(border, "amber", enabled=use_color))
    output_func("")


def _package_version() -> str:
    try:
        return metadata.version("research-harness")
    except metadata.PackageNotFoundError:
        return "0.1.0"


def _banner_left_lines(width: int) -> list[str]:
    user = getpass.getuser() or "there"
    model = os.environ.get("RESEARCH_HARNESS_LLM_MODEL", "openai/gpt-5.2")
    retriever = os.environ.get("RESEARCH_HARNESS_RETRIEVER", "auto")
    cwd = _compact_path(Path.cwd(), max(18, width - 5))
    return [
        f"Welcome back, {user}",
        "",
        "     ___  _   _ _____ ___  ____  _____",
        r"    / _ \| | | |_   _/ _ \|  _ \| ____|",
        r"   | |_| | | | | | || | | | |_) |  _|",
        r"   |  _  | |_| | | || |_| |  _ <| |___",
        r"   |_| |_|\___/  |_| \___/|_| \_\_____|",
        "",
        "            <AUTORE>",
        "       autonomous research",
        "",
        f"model: {model}",
        f"retriever: {retriever}",
        f"workspace: {cwd}",
    ]


def _banner_right_lines(width: int) -> list[str]:
    return [
        "Available tools",
        "research: search, fetch documents",
        "documents: inspect figures, extract data",
        "analysis: analyze documents, python, charts",
        "workspace: read files, bounded terminal",
        "delegation: delegate_task, specialist",
        "services: Firecrawl + configured adapters",
        "optimization: grade candidates, save findings",
        "",
        "Run controls",
        "turns: unlimited by default",
        "runtime: unlimited by default",
        "limit turns: --max-iterations",
        "limit runtime: --max-runtime-seconds",
        "cancel: Ctrl-C; artifacts remain append-only",
    ]


def _compact_path(path: Path, width: int) -> str:
    text = str(path).replace(str(Path.home()), "~", 1)
    if len(text) <= width:
        return text
    return "..." + text[-max(1, width - 3):]


def _banner_cell(text: str, width: int, *, use_color: bool) -> str:
    clean = text[: max(0, width - 2)]
    color = "gold" if clean in {"Available tools", "Run controls"} else "gray"
    if clean.startswith("Welcome back"):
        color = "gold"
    elif ":" in clean:
        color = "amber"
    return " " + _paint(clean.ljust(width - 2), color, enabled=use_color) + " "


def _print_run_summary(run, store, *, output_func: Callable[[str], None] = print, color: Optional[bool] = None) -> None:
    use_color = _use_color() if color is None else color
    status_color = "green" if run.status == "completed" else "red"
    output_func("")
    output_func(_paint("Run complete", "bold", enabled=use_color))
    output_func(f"{_paint('status', 'gray', enabled=use_color)}   {_paint(run.status, status_color, enabled=use_color)}")
    output_func(f"{_paint('run', 'gray', enabled=use_color)}      {run.id}")
    output_func(f"{_paint('home', 'gray', enabled=use_color)}     {store.root}")
    if run.session_jsonl_path:
        output_func(f"{_paint('session', 'gray', enabled=use_color)}  {run.session_jsonl_path}")

    primary_artifacts = [
        ("report", store.report_path),
        ("run state", store.run_state_path),
        ("benchmark", store.run_benchmark_path),
        ("notebook", store.run_notebook_path),
    ]
    optional_artifacts = [
        ("seed context", store.optimizer_seed_context_path),
        ("optimization", store.optimization_result_path),
        ("candidate", store.optimized_candidate_path),
        ("optimal code", store.optimal_code_path),
        ("solution", store.solution_path),
        ("champion", store.current_champion_path),
        ("candidate graph", store.candidate_graph_path),
        ("candidate visual", store.candidate_graph_graph_path),
        ("champion history", store.champion_history_path),
    ]
    diagnostics = [
        ("diagnosis", store.harness_diagnosis_path),
        ("world db", store.sqlite_path),
        ("decision dag", store.decision_dag_path),
        ("timeline", store.agent_timeline_path),
        ("timeline svg", store.agent_timeline_svg_path),
        ("score graph", store.score_improvement_path),
    ]
    output_func("")
    output_func(_paint("Open first", "teal", enabled=use_color))
    for label, path in [(label, path) for label, path in primary_artifacts if path.exists()]:
        output_func(f"  {_paint(label.ljust(10), 'gray', enabled=use_color)} {path}")
    available_optional = [(label, path) for label, path in optional_artifacts if path.exists()]
    if available_optional:
        output_func("")
        output_func(_paint("Optimization artifacts", "teal", enabled=use_color))
        for label, path in available_optional:
            output_func(f"  {_paint(label.ljust(10), 'gray', enabled=use_color)} {path}")
    output_func("")
    output_func(_paint("Diagnostics", "teal", enabled=use_color))
    for label, path in [(label, path) for label, path in diagnostics if path.exists()]:
        output_func(f"  {_paint(label.ljust(10), 'gray', enabled=use_color)} {path}")


def prompt_choice(
    title: str,
    options: list[tuple[str, str]],
    *,
    default: str,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
    key_reader: Optional[Callable[[], str]] = None,
    use_arrows: Optional[bool] = None,
) -> str:
    if use_arrows is None:
        use_arrows = key_reader is not None or sys.stdin.isatty()
    if use_arrows:
        return prompt_arrow_choice(title, options, default=default, key_reader=key_reader)
    use_color = _use_color()
    output_func("")
    output_func(_paint(title, "teal", enabled=use_color))
    for index, (value, label) in enumerate(options, start=1):
        suffix = _paint(" [default]", "green", enabled=use_color) if value == default else ""
        output_func(f"  {_paint(str(index) + '.', 'gray', enabled=use_color)} {label}{suffix}")
    while True:
        answer = input_func(_paint("Choose a number: ", "blue", enabled=use_color)).strip()
        if not answer:
            return default
        if answer.isdigit():
            selected_index = int(answer)
            if 1 <= selected_index <= len(options):
                return options[selected_index - 1][0]
        output_func(_paint(f"Please enter 1-{len(options)}, or press Enter for the default.", "yellow", enabled=use_color))


def prompt_arrow_choice(
    title: str,
    options: list[tuple[str, str]],
    *,
    default: str,
    key_reader: Optional[Callable[[], str]] = None,
) -> str:
    if not options:
        raise ValueError("prompt_arrow_choice requires at least one option")
    selected_index = next((index for index, (value, _label) in enumerate(options) if value == default), 0)
    read_key = key_reader or read_terminal_key
    lines_rendered = 0
    use_color = _use_color()

    while True:
        if lines_rendered:
            sys.stdout.write(f"\033[{lines_rendered}F")
        lines = [
            _paint(title, "teal", enabled=use_color),
            _paint("Use Up/Down, then Enter. Vim keys work too.", "gray", enabled=use_color),
        ]
        for index, (_value, label) in enumerate(options):
            if index == selected_index:
                lines.append(f"{_paint('>', 'green', enabled=use_color)} {_paint(label, 'bold', enabled=use_color)}")
            else:
                lines.append(f"  {_paint(label, 'gray', enabled=use_color)}")
        for line in lines:
            sys.stdout.write(f"\033[2K\r{line}\n")
        sys.stdout.flush()
        lines_rendered = len(lines)

        key = read_key()
        if key in {"up", "k"}:
            selected_index = (selected_index - 1) % len(options)
        elif key in {"down", "j"}:
            selected_index = (selected_index + 1) % len(options)
        elif key in {"enter", "\r", "\n"}:
            sys.stdout.write("\n")
            sys.stdout.flush()
            return options[selected_index][0]


def read_terminal_key() -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        char = sys.stdin.read(1)
        if char == "\x1b":
            suffix = sys.stdin.read(2)
            if suffix == "[A":
                return "up"
            if suffix == "[B":
                return "down"
            return "escape"
        if char in {"\r", "\n"}:
            return "enter"
        return char
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def prompt_text(
    prompt: str,
    *,
    default: Optional[str] = None,
    required: bool = False,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> str:
    rendered = f"{prompt} [{default}]: " if default else f"{prompt}: "
    use_color = _use_color()
    while True:
        answer = input_func(_paint(rendered, "blue", enabled=use_color)).strip()
        if answer:
            return answer
        if default is not None:
            return default
        if not required:
            return ""
        output_func(_paint("Please enter a value.", "yellow", enabled=use_color))


def prompt_int(
    prompt: str,
    *,
    default: int,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> int:
    use_color = _use_color()
    while True:
        answer = input_func(_paint(f"{prompt} [{default}]: ", "blue", enabled=use_color)).strip()
        if not answer:
            return default
        try:
            value = int(answer)
        except ValueError:
            output_func(_paint("Please enter a whole number.", "yellow", enabled=use_color))
            continue
        if value > 0:
            return value
        output_func(_paint("Please enter a number greater than zero.", "yellow", enabled=use_color))


def prompt_optional_int(
    prompt: str,
    *,
    default: Optional[int] = None,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
) -> Optional[int]:
    use_color = _use_color()
    default_label = str(default) if default is not None else "unlimited"
    while True:
        answer = input_func(_paint(f"{prompt} [{default_label}]: ", "blue", enabled=use_color)).strip().lower()
        if not answer:
            return default
        if answer in {"none", "unlimited", "infinite", "inf"}:
            return None
        try:
            value = int(answer)
        except ValueError:
            output_func(_paint("Enter a positive whole number or 'unlimited'.", "yellow", enabled=use_color))
            continue
        if value > 0:
            return value
        output_func(_paint("Enter a positive whole number or 'unlimited'.", "yellow", enabled=use_color))


def configure_interactive_run(
    args: argparse.Namespace,
    *,
    input_func: Callable[[str], str] = input,
    output_func: Callable[[str], None] = print,
    key_reader: Optional[Callable[[], str]] = None,
) -> argparse.Namespace:
    _print_cli_banner(output_func=output_func)
    output_func("Ready to run. Existing flags are used as starting values.")
    output_func("Press Enter to accept defaults; use Up/Down in menus.")
    output_func("")
    args.goal = prompt_text(
        "What should the agent work on?",
        default=args.goal,
        required=True,
        input_func=input_func,
        output_func=output_func,
    )
    args.grader = prompt_choice(
        "How should this run be evaluated?",
        [("", "Research only (no official scorer)"), *[
            (grader, f"{grader.replace('_', ' ')} (official scorer)")
            for grader in OPTIMIZATION_GRADER_CHOICES
        ]],
        default=args.grader or "",
        input_func=input_func,
        output_func=output_func,
        key_reader=key_reader,
    ) or None
    if args.grader:
        args.grader_loops = prompt_int(
            "Official candidate evaluations",
            default=args.grader_loops or DEFAULT_GRADER_LOOPS,
            input_func=input_func,
            output_func=output_func,
        )
    else:
        args.grader_loops = None
    args.max_iterations = prompt_optional_int(
        "Maximum model turns",
        default=args.max_iterations,
        input_func=input_func,
        output_func=output_func,
    )
    args.max_iterations_explicit = args.max_iterations is not None
    selected_model = prompt_choice(
        "Which model/lab should run the harness?",
        model_choices(),
        default=args.llm_model or "openai/gpt-5.2",
        input_func=input_func,
        output_func=output_func,
        key_reader=key_reader,
    )
    args.llm_model = selected_model
    args.llm_provider, args.llm_model = resolve_model_selection(args.llm_provider, args.llm_model)
    output_func("")
    output_func("Starting run. The artifact trail will be waiting at the finish line.")
    return args


def load_dotenv(path: Path = Path(".env"), *, override: bool = False) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        clean_key = key.strip()
        clean_value = value.strip().strip('"').strip("'")
        if override:
            os.environ[clean_key] = clean_value
        else:
            os.environ.setdefault(clean_key, clean_value)


def main() -> None:
    load_dotenv()
    load_dotenv(Path(".env.local"), override=True)
    parser = build_parser()
    args = parser.parse_args()
    args.max_iterations_explicit = any(
        item == "--max-iterations" or item.startswith("--max-iterations=")
        for item in sys.argv[1:]
    )
    args.max_runtime_seconds_explicit = any(
        item == "--max-runtime-seconds" or item.startswith("--max-runtime-seconds=")
        for item in sys.argv[1:]
    )
    if args.list_llm_models:
        _print_cli_banner(compact=True)
        print(format_model_catalog())
        return
    args.llm_provider, args.llm_model = resolve_model_selection(args.llm_provider, args.llm_model)
    banner_printed = False
    if args.interactive or not args.goal:
        if not sys.stdin.isatty():
            parser.error(
                "a goal is required when stdin is not interactive; "
                "run `autore` in a terminal for the selection setup"
            )
        args = configure_interactive_run(args)
        banner_printed = True
    if not args.quiet and not banner_printed:
        _print_cli_banner()
        print(f"Ready to work on: {args.goal}")
        print("")
    if args.preflight:
        run_preflight_evals(args)
    if args.grader_loops is not None and args.grader is None:
        parser.error("--grader-loops requires --grader.")
    if args.grader_loops is not None and args.grader_loops < 1:
        parser.error("--grader-loops must be at least 1.")
    if args.max_iterations is not None and args.max_iterations < 1:
        parser.error("--max-iterations must be at least 1 when supplied.")
    if args.max_runtime_seconds is not None and args.max_runtime_seconds <= 0:
        parser.error("--max-runtime-seconds must be greater than zero when supplied.")
    config = HarnessConfig(
        retriever=args.retriever,
        max_iterations=args.max_iterations,
        evaluator_name=args.grader,
        max_grader_calls=args.grader_loops,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        session_projects_dir=args.session_projects_dir,
        resume_session_id=args.resume_session,
        fork_session_id=args.fork_session,
        enable_sessions=not args.no_sessions,
        echo_progress=not args.quiet,
        animations=not args.no_animations and not args.quiet and sys.stdout.isatty() and not os.environ.get("CI") and not os.environ.get("NO_COLOR"),
        default_budget=AgentBudget(
            max_tool_calls=args.max_tool_calls,
            max_runtime_seconds=args.max_runtime_seconds,
        ),
    )
    orchestrator = Orchestrator(args.corpus, args.output, config)
    try:
        run, store = asyncio.run(orchestrator.run(args.goal))
    except RuntimeError as exc:
        raise SystemExit(f"Optimization run stopped: {exc}") from exc
    _print_run_summary(run, store)


def _apply_grader_budget_defaults(args: argparse.Namespace) -> None:
    """Compatibility hook: omitted limits intentionally remain unbounded."""
    return


def run_preflight_evals(args: argparse.Namespace) -> None:
    from .evals.harness import EvaluationHarness
    from .evals.suites import eval_suite_by_id, select_eval_tasks

    try:
        suite = select_eval_tasks(eval_suite_by_id(args.preflight_suite), args.preflight_eval_ids)
    except ValueError as exc:
        raise SystemExit(f"Preflight eval selection failed: {exc}") from exc
    output_root = Path(os.environ.get("AUTORE_PREFLIGHT_OUTPUT_DIR", "eval_outputs/preflight"))
    print(f"Preflight evals: running {suite.id} ({len(suite.tasks)} eval(s))", flush=True)
    summary = asyncio.run(EvaluationHarness(corpus_path=args.corpus, output_root=output_root).run_suite(suite))
    if summary.passed_trials == summary.trial_count:
        print(f"Preflight evals: passed {summary.passed_trials}/{summary.trial_count}", flush=True)
        return
    failed = [trial for trial in summary.trials if not trial.get("passed")]
    lines = [
        "Preflight evals failed. Refusing to start autore run.",
        f"Passed {summary.passed_trials}/{summary.trial_count}; summary: {output_root / (suite.id + '_summary.json')}",
    ]
    for trial in failed[:5]:
        failed_graders = [
            grader.get("grader_id")
            for grader in trial.get("grader_results", [])
            if not grader.get("passed")
        ]
        lines.append(f"- {trial.get('task_id')}: failed graders={failed_graders}")
    raise SystemExit("\n".join(lines))


if __name__ == "__main__":
    main()
