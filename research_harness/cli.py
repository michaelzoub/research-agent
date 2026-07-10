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

from .evals.suites import SUITE_CHOICES
from .model_catalog import format_model_catalog, model_choices, resolve_model_selection
from .orchestrator import HarnessConfig, Orchestrator


RETRIEVER_CHOICES = ("auto", "local", "arxiv", "openalex", "semantic_scholar", "github", "web", "docs_blogs", "twitter", "memory", "alchemy")
LLM_PROVIDER_CHOICES = ("auto", "openai", "anthropic", "kimi", "ollama", "local", "multi")

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
  autore "Compare a proposed strategy with the registered evaluator" --evaluator length_score

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
        default=12,
        help="Maximum model turns for this run.",
    )
    parser.add_argument(
        "--evaluator",
        default=os.environ.get("RESEARCH_HARNESS_EVALUATOR"),
        help="Optional registered evaluator made available to the agent as a controlled capability.",
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
    parser.add_argument(
        "--no-steering",
        action="store_true",
        help="Disable live /article and /steer input while the agent is running.",
    )
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
            f"{_paint(f'Autore Code v{version}', 'teal', enabled=use_color)} "
            f"{_paint('research + optimize agent harness', 'gray', enabled=use_color)}"
        )
        return
    width = max(72, min(116, shutil.get_terminal_size((100, 24)).columns - 4))
    left_width = max(31, min(46, width // 2 - 2))
    right_width = width - left_width - 3
    header = f" Autore Code v{version} "
    output_func(_paint(header, "teal", enabled=use_color) + _paint("-" * max(0, width - len(header)), "teal", enabled=use_color))
    border = "+" + "-" * left_width + "+" + "-" * right_width + "+"
    output_func(_paint(border, "teal", enabled=use_color))
    left_lines = _banner_left_lines(left_width)
    right_lines = _banner_right_lines(right_width)
    max_lines = max(len(left_lines), len(right_lines))
    for index in range(max_lines):
        left = left_lines[index] if index < len(left_lines) else ""
        right = right_lines[index] if index < len(right_lines) else ""
        output_func(
            _paint("|", "teal", enabled=use_color)
            + _banner_cell(left, left_width, use_color=use_color)
            + _paint("|", "teal", enabled=use_color)
            + _banner_cell(right, right_width, use_color=use_color)
            + _paint("|", "teal", enabled=use_color)
        )
    output_func(_paint(border, "teal", enabled=use_color))
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
        f"Welcome back {user}!",
        "",
        "     _         _",
        "    / \\  _   _| |_ ___  _ __ ___",
        "   / _ \\| | | | __/ _ \\| '__/ _ \\",
        "  / ___ \\ |_| | || (_) | | |  __/",
        " /_/   \\_\\__,_|\\__\\___/|_|  \\___|",
        "",
        f"Model {model}",
        f"Retriever {retriever}",
        cwd,
    ]


def _banner_right_lines(width: int) -> list[str]:
    return [
        "Tips for getting started",
        "Run autore with no args for guided setup.",
        "Use autore \"goal\" for direct runs.",
        "Use --llm-model ollama/qwen3.5:latest locally.",
        "",
        "What's new",
        "Claude-style startup panel and guided flow.",
        "Ollama provider support for local models.",
        "Install command: python3 -m pip install -e .",
    ]


def _compact_path(path: Path, width: int) -> str:
    text = str(path).replace(str(Path.home()), "~", 1)
    if len(text) <= width:
        return text
    return "..." + text[-max(1, width - 3):]


def _banner_cell(text: str, width: int, *, use_color: bool) -> str:
    clean = text[: max(0, width - 2)]
    color = "teal" if clean in {"Tips for getting started", "What's new"} else "gray"
    if clean.startswith("Welcome back"):
        color = "bold"
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
        ("champ tree", store.champion_tree_path),
        ("champ graph", store.champion_tree_graph_path),
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
    for label, path in primary_artifacts:
        output_func(f"  {_paint(label.ljust(10), 'gray', enabled=use_color)} {path}")
    available_optional = [(label, path) for label, path in optional_artifacts if path.exists()]
    if available_optional:
        output_func("")
        output_func(_paint("Optimization artifacts", "teal", enabled=use_color))
        for label, path in available_optional:
            output_func(f"  {_paint(label.ljust(10), 'gray', enabled=use_color)} {path}")
    output_func("")
    output_func(_paint("Diagnostics", "teal", enabled=use_color))
    for label, path in diagnostics:
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
    args.retriever = prompt_choice(
        "Where should research evidence come from?",
        [
            ("auto", "Auto mix of available sources"),
            ("local", "Bundled offline corpus"),
            ("arxiv", "arXiv"),
            ("openalex", "OpenAlex"),
            ("semantic_scholar", "Semantic Scholar"),
            ("github", "GitHub"),
            ("web", "General web"),
            ("docs_blogs", "Docs and blogs"),
            ("twitter", "Twitter/X"),
            ("memory", "Stored run memory"),
            ("alchemy", "Alchemy blockchain data (requires ALCHEMY_API_KEY)"),
        ],
        default=args.retriever or "auto",
        input_func=input_func,
        output_func=output_func,
        key_reader=key_reader,
    )
    args.max_iterations = prompt_int(
        "Maximum model turns",
        default=args.max_iterations,
        input_func=input_func,
        output_func=output_func,
    )
    args.max_iterations_explicit = True
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
    config = HarnessConfig(
        retriever=args.retriever,
        max_iterations=args.max_iterations,
        evaluator_name=args.evaluator,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        session_projects_dir=args.session_projects_dir,
        resume_session_id=args.resume_session,
        fork_session_id=args.fork_session,
        enable_sessions=not args.no_sessions,
        echo_progress=not args.quiet,
        enable_steering=(not args.no_steering and not args.quiet and sys.stdin.isatty()),
    )
    orchestrator = Orchestrator(args.corpus, args.output, config)
    run, store = asyncio.run(orchestrator.run(args.goal))
    _print_run_summary(run, store)


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
