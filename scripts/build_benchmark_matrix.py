from __future__ import annotations

import argparse
from pathlib import Path

from benchies.metrics import empty_metric_row, write_metrics_csv, write_metrics_jsonl
from benchies.prompts import (
    CONTROL_PROMPT_CASES,
    CURATED_PROMPT_CASES,
    PromptCase,
    PRIMARY_PROMPT_CASES,
    PROMPT_CASES,
)


def select_prompt_cases(prompt_set: str) -> tuple[PromptCase, ...]:
    """Map a CLI prompt-set name to the corresponding prompt records."""
    if prompt_set == "primary":
        return PRIMARY_PROMPT_CASES
    if prompt_set == "curated":
        return CURATED_PROMPT_CASES
    if prompt_set == "control":
        return CONTROL_PROMPT_CASES
    return PROMPT_CASES


def parse_steps(raw_steps: str) -> list[int]:
    """Parse comma-separated step counts such as '1,2,4'."""
    steps = [int(step.strip()) for step in raw_steps.split(",") if step.strip()]
    if not steps:
        raise argparse.ArgumentTypeError("at least one step count is required")
    return steps


def build_parser() -> argparse.ArgumentParser:
    """Define CLI flags for expanding prompts into benchmark rows."""
    parser = argparse.ArgumentParser(
        description="Build an empty FLUX Schnell benchmark metric matrix.",
    )
    parser.add_argument(
        "--prompt-set",
        choices=("primary", "curated", "control", "all"),
        default="curated",
        help="Prompt set to expand. Defaults to the 50-prompt curated set.",
    )
    parser.add_argument(
        "--steps",
        type=parse_steps,
        default=[1, 2, 4],
        help="Comma-separated inference step counts. Defaults to 1,2,4.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Runs per prompt and step count. Defaults to 5.",
    )
    parser.add_argument(
        "--output",
        default="outputs/benchmark_matrix.csv",
        help="Output path ending in .csv or .jsonl.",
    )
    return parser


def main() -> None:
    """Expand prompt cases across step counts and repeated runs."""
    args = build_parser().parse_args()
    prompt_cases = select_prompt_cases(args.prompt_set)
    metrics = [
        empty_metric_row(
            prompt_id=prompt_case.prompt_id,
            prompt=prompt_case.prompt,
            category=prompt_case.category,
            steps=steps,
            run=run,
        )
        for prompt_case in prompt_cases
        for steps in args.steps
        for run in range(1, args.runs + 1)
    ]

    output_path = Path(args.output)
    if output_path.suffix == ".csv":
        saved_path = write_metrics_csv(metrics, output_path)
    elif output_path.suffix == ".jsonl":
        saved_path = write_metrics_jsonl(metrics, output_path)
    else:
        raise SystemExit("--output path must end in .csv or .jsonl")

    print(f"wrote {len(metrics)} benchmark rows to {saved_path}")


if __name__ == "__main__":
    main()
