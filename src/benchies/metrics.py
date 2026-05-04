from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    """Comparable result schema for one generated image."""

    prompt_id: str
    prompt: str
    category: str
    steps: int
    run: int
    load_time_ms: float | None = None
    generation_time_ms: float | None = None
    total_time_ms: float | None = None
    clip_score: float | None = None
    aesthetic_score: float | None = None
    gpu_memory_mb: float | None = None
    peak_memory_mb: float | None = None
    seed: int | None = None
    guidance_scale: float | None = None
    resolution: str | None = None
    model_id: str | None = None
    device: str | None = None
    dtype: str | None = None
    image_path: str | None = None


BENCHMARK_METRIC_FIELDS: tuple[str, ...] = tuple(BenchmarkMetrics.__dataclass_fields__)


def empty_metric_row(
    *,
    prompt_id: str,
    prompt: str,
    category: str,
    steps: int,
    run: int,
) -> BenchmarkMetrics:
    """Create a placeholder metric row before generation/scoring has run."""
    return BenchmarkMetrics(
        prompt_id=prompt_id,
        prompt=prompt,
        category=category,
        steps=steps,
        run=run,
    )


def write_metrics_jsonl(metrics: list[BenchmarkMetrics], output_path: Path) -> Path:
    """Write benchmark rows as JSON Lines for append-friendly pipelines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for metric in metrics:
            file.write(json.dumps(asdict(metric)) + "\n")
    return output_path


def write_metrics_csv(metrics: list[BenchmarkMetrics], output_path: Path) -> Path:
    """Write benchmark rows as CSV for analysis in pandas or spreadsheets."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=BENCHMARK_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(asdict(metric) for metric in metrics)
    return output_path
