from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from benchies.metrics import BENCHMARK_METRIC_FIELDS, BenchmarkMetrics, write_metrics_csv, write_metrics_jsonl
from benchies.prompts import (
    CONTROL_PROMPT_CASES,
    CURATED_PROMPT_CASES,
    PRIMARY_PROMPT_CASES,
    PROMPT_CASES,
    PromptCase,
    prompts_by_category,
)


DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-schnell"


def parse_csv_ints(raw_values: str) -> list[int]:
    """Parse comma-separated integers from CLI flags such as '1,2,4'."""
    values = [int(value.strip()) for value in raw_values.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer value is required")
    return values


def parse_csv_strings(raw_values: str) -> list[str]:
    """Parse comma-separated string filters from CLI flags."""
    return [value.strip() for value in raw_values.split(",") if value.strip()]


def select_prompt_set(prompt_set: str) -> tuple[PromptCase, ...]:
    """Map a prompt-set name to the corresponding prompt records."""
    if prompt_set == "primary":
        return PRIMARY_PROMPT_CASES
    if prompt_set == "curated":
        return CURATED_PROMPT_CASES
    if prompt_set == "control":
        return CONTROL_PROMPT_CASES
    return PROMPT_CASES


def filter_prompt_cases(
    prompt_cases: tuple[PromptCase, ...],
    *,
    categories: list[str] | None,
    prompt_ids: list[str] | None,
    one_per_category: bool,
    limit_per_category: int | None,
    limit: int | None,
) -> tuple[PromptCase, ...]:
    """Apply category, prompt id, and limit filters while preserving registry order."""
    selected = list(prompt_cases)

    if categories:
        requested_categories = set(categories)
        selected = [prompt_case for prompt_case in selected if prompt_case.category in requested_categories]

    if prompt_ids:
        requested_prompt_ids = set(prompt_ids)
        selected = [prompt_case for prompt_case in selected if prompt_case.prompt_id in requested_prompt_ids]

    if one_per_category:
        selected = [
            category_prompt_cases[0]
            for category_prompt_cases in prompts_by_category(selected).values()
            if category_prompt_cases
        ]

    if limit_per_category is not None:
        selected = [
            prompt_case
            for category_prompt_cases in prompts_by_category(selected).values()
            for prompt_case in category_prompt_cases[:limit_per_category]
        ]

    if limit is not None:
        selected = selected[:limit]

    return tuple(selected)


def cuda_memory_mb(torch_module) -> tuple[float | None, float | None]:
    """Return current and peak CUDA memory in MiB when CUDA is available."""
    if not torch_module.cuda.is_available():
        return None, None
    current_memory = torch_module.cuda.memory_allocated() / 1024 / 1024
    peak_memory = torch_module.cuda.max_memory_allocated() / 1024 / 1024
    return current_memory, peak_memory


def safe_name(value: str) -> str:
    """Create a filesystem-safe path segment for generated image names."""
    return "".join(character if character.isalnum() or character in ("-", "_") else "-" for character in value)


def build_image_path(output_dir: Path, prompt_case: PromptCase, steps: int, run: int, seed: int) -> Path:
    """Create a deterministic image path from benchmark dimensions."""
    filename = f"{prompt_case.prompt_id}_steps-{steps}_run-{run}_seed-{seed}.png"
    return output_dir / prompt_case.category / safe_name(filename)


def write_metrics(metrics: list[BenchmarkMetrics], output_path: Path) -> Path:
    """Write metrics in the format implied by the output path suffix."""
    if output_path.suffix == ".csv":
        return write_metrics_csv(metrics, output_path)
    if output_path.suffix == ".jsonl":
        return write_metrics_jsonl(metrics, output_path)
    raise SystemExit("--metrics-output path must end in .csv or .jsonl")


def build_parser() -> argparse.ArgumentParser:
    """Define CLI flags for running FLUX Schnell benchmark generations."""
    parser = argparse.ArgumentParser(
        description="Run selected FLUX Schnell benchmark prompts and write metric rows.",
    )
    parser.add_argument(
        "--prompt-set",
        choices=("primary", "curated", "control", "all"),
        default="curated",
        help="Prompt set to run. Defaults to curated.",
    )
    parser.add_argument(
        "--category",
        type=parse_csv_strings,
        help="Comma-separated category filter, e.g. simple_objects,counting.",
    )
    parser.add_argument(
        "--prompt-id",
        type=parse_csv_strings,
        help="Comma-separated prompt id filter, e.g. simple_objects_01,counting_01.",
    )
    parser.add_argument(
        "--one-per-category",
        action="store_true",
        help="Run the first selected prompt from each category.",
    )
    parser.add_argument(
        "--limit-per-category",
        type=int,
        help="Run at most N prompts from each selected category.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run at most N prompts after all other filters.",
    )
    parser.add_argument(
        "--steps",
        type=parse_csv_ints,
        default=[1, 2, 4],
        help="Comma-separated inference step counts. Defaults to 1,2,4.",
    )
    parser.add_argument("--runs", type=int, default=1, help="Runs per prompt and step count.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--seed-stride",
        type=int,
        default=1,
        help="Amount to add to the seed after each repeated run.",
    )
    parser.add_argument("--width", type=int, default=1024, help="Output width in pixels.")
    parser.add_argument("--height", type=int, default=1024, help="Output height in pixels.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale. FLUX Schnell commonly uses 0.0.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=256,
        help="Maximum text sequence length passed into the pipeline.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id to load.")
    parser.add_argument(
        "--output-dir",
        default="outputs/flux_benchmark/images",
        help="Directory where generated images will be saved.",
    )
    parser.add_argument(
        "--metrics-output",
        default="outputs/flux_benchmark/metrics.csv",
        help="Metric output path ending in .csv or .jsonl.",
    )
    return parser


def main() -> None:
    """Load FLUX once, run selected prompt combinations, and save metrics."""
    args = build_parser().parse_args()
    import torch

    from benchies.flux_schnell import (
        FluxSchnellConfig,
        generate_image,
        load_pipeline,
        save_image,
        select_device,
        select_dtype,
    )

    prompt_cases = filter_prompt_cases(
        select_prompt_set(args.prompt_set),
        categories=args.category,
        prompt_ids=args.prompt_id,
        one_per_category=args.one_per_category,
        limit_per_category=args.limit_per_category,
        limit=args.limit,
    )

    if not prompt_cases:
        raise SystemExit("no prompts matched the selected filters")

    output_dir = Path(args.output_dir)
    metrics_output = Path(args.metrics_output)
    device = select_device()
    dtype = select_dtype(device)

    load_start = perf_counter()
    pipeline = load_pipeline(args.model_id, device=device, dtype=dtype)
    load_time_ms = (perf_counter() - load_start) * 1000

    metrics: list[BenchmarkMetrics] = []
    total_rows = len(prompt_cases) * len(args.steps) * args.runs
    row_index = 0

    for prompt_case in prompt_cases:
        for steps in args.steps:
            for run in range(1, args.runs + 1):
                row_index += 1
                seed = args.seed + ((run - 1) * args.seed_stride)
                image_path = build_image_path(output_dir, prompt_case, steps=steps, run=run, seed=seed)
                config = FluxSchnellConfig(
                    prompt=prompt_case.prompt,
                    output=image_path,
                    model_id=args.model_id,
                    width=args.width,
                    height=args.height,
                    num_inference_steps=steps,
                    guidance_scale=args.guidance_scale,
                    seed=seed,
                    max_sequence_length=args.max_sequence_length,
                )

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()

                row_start = perf_counter()
                generation_start = perf_counter()
                image = generate_image(pipeline, config=config, device=device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                generation_time_ms = (perf_counter() - generation_start) * 1000

                saved_path = save_image(image, image_path)
                total_time_ms = (perf_counter() - row_start) * 1000
                gpu_memory_mb, peak_memory_mb = cuda_memory_mb(torch)
                metrics.append(
                    BenchmarkMetrics(
                        prompt_id=prompt_case.prompt_id,
                        prompt=prompt_case.prompt,
                        category=prompt_case.category,
                        steps=steps,
                        run=run,
                        load_time_ms=load_time_ms if row_index == 1 else None,
                        generation_time_ms=generation_time_ms,
                        total_time_ms=total_time_ms,
                        gpu_memory_mb=gpu_memory_mb,
                        peak_memory_mb=peak_memory_mb,
                        seed=seed,
                        guidance_scale=args.guidance_scale,
                        resolution=f"{args.width}x{args.height}",
                        model_id=args.model_id,
                        device=device,
                        dtype=str(dtype),
                        image_path=str(saved_path),
                    )
                )
                write_metrics(metrics, metrics_output)
                print(f"[{row_index}/{total_rows}] {prompt_case.prompt_id} steps={steps} run={run}")

    saved_metrics_path = write_metrics(metrics, metrics_output)
    print(f"wrote {len(metrics)} metric rows to {saved_metrics_path}")
    print(f"metric fields: {', '.join(BENCHMARK_METRIC_FIELDS)}")


if __name__ == "__main__":
    main()
