from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from .harness import EvaluationHarness
from .suites import SUITE_CHOICES, eval_suite_by_id, select_eval_tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run prewritten research-harness eval suites.")
    parser.add_argument("--suite", choices=SUITE_CHOICES, default="core")
    parser.add_argument(
        "--eval",
        action="append",
        default=[],
        dest="eval_ids",
        help="Run only the selected eval id. May be repeated or comma-separated.",
    )
    parser.add_argument("--list", action="store_true", help="List eval ids for the selected suite and exit.")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--seed", action="append", type=int, default=[], help="Seed cell to include; may be repeated.")
    parser.add_argument("--model", action="append", default=[], help="Provider/model cell to include (for example openai/gpt-5.2); may be repeated.")
    parser.add_argument("--output", type=Path, default=Path("eval_outputs"))
    parser.add_argument("--corpus", type=Path, default=Path("examples/corpus/research_corpus.json"))
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        suite = select_eval_tasks(eval_suite_by_id(args.suite), args.eval_ids)
    except ValueError as exc:
        parser.error(str(exc))
    if args.list:
        print(f"Eval suite: {suite.name}")
        for task in suite.tasks:
            print(f"{task.id}\t{task.name}")
        return
    suite.trials_per_task = args.trials
    if args.seed:
        suite.seeds = args.seed
    if args.model:
        suite.models = args.model
    summary = asyncio.run(EvaluationHarness(corpus_path=args.corpus, output_root=args.output).run_suite(suite))
    print(f"Eval suite: {summary.suite_name}")
    print(f"Trials: {summary.passed_trials}/{summary.trial_count} passed")
    print(f"Aggregate score: {summary.aggregate_score:.3f}")
    for trial in summary.trials:
        status = "PASS" if trial.get("passed") else "FAIL"
        print(
            f"- {trial.get('task_id')} {trial.get('model')} seed {trial.get('seed')} trial {trial.get('trial_index')}: "
            f"{float(trial.get('aggregate_score') or 0.0):.3f} {status}"
        )
        for grader in trial.get("grader_results", []):
            if grader.get("grader_type") != "model":
                continue
            print(f"  - {grader.get('grader_id')}: {float(grader.get('score') or 0.0):.3f}")
            for assertion in grader.get("assertions", []):
                if assertion.get("type") != "deep_acyclic_graph":
                    continue
                right = assertion.get("right_behaviors") or []
                wrong = assertion.get("wrong_behaviors") or []
                if right:
                    print(f"    right: {'; '.join(str(item) for item in right[:3])}")
                if wrong:
                    print(f"    wrong: {'; '.join(str(item) for item in wrong[:3])}")
    if summary.comparisons:
        print("Comparisons:")
        for cell, metrics in summary.comparisons.items():
            print(f"- {cell}: pass_rate={metrics['pass_rate']:.3f}, mean={metrics['mean_score']:.3f}, stdev={metrics['score_stdev']:.3f}, seeds={metrics['seeds']}")
    print(f"Summary: {args.output / (suite.id + '_summary.json')}")


if __name__ == "__main__":
    main()
