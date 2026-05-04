from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from benchies.metrics import BENCHMARK_METRIC_FIELDS


def select_device(requested_device: str):
    """Select the scoring device, preferring CUDA when available."""
    import torch

    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_metric_rows(input_path: Path) -> list[dict[str, Any]]:
    """Read benchmark rows from CSV or JSONL."""
    if input_path.suffix == ".csv":
        with input_path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))

    if input_path.suffix == ".jsonl":
        with input_path.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    raise SystemExit("--metrics-input path must end in .csv or .jsonl")


def write_metric_rows(rows: list[dict[str, Any]], output_path: Path) -> Path:
    """Write scored benchmark rows to CSV or JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(BENCHMARK_METRIC_FIELDS)
    extra_fields = sorted({field for row in rows for field in row if field not in fieldnames})
    fieldnames.extend(extra_fields)

    if output_path.suffix == ".csv":
        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return output_path

    if output_path.suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row) + "\n")
        return output_path

    raise SystemExit("--metrics-output path must end in .csv or .jsonl")


def resolve_image_path(image_path: str | None, root_dir: Path) -> Path:
    """Resolve image paths from the metrics file relative to the chosen root."""
    if not image_path:
        raise ValueError("metric row is missing image_path")
    path = Path(image_path)
    if path.is_absolute():
        return path
    return root_dir / path


def load_image(image_path: Path):
    """Load an image as RGB for CLIP preprocessing."""
    from PIL import Image

    with Image.open(image_path) as image:
        return image.convert("RGB")


class AestheticPredictor:
    """LAION-style aesthetic predictor over CLIP ViT-L/14 image embeddings."""

    def __init__(self, input_size: int = 768) -> None:
        import torch.nn as nn

        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def load(self, weights_path: Path, device) -> None:
        """Load local aesthetic-predictor weights."""
        import torch

        state_dict = torch.load(weights_path, map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        state_dict = {
            key.removeprefix("model.").removeprefix("layers."): value
            for key, value in state_dict.items()
        }
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

    def score(self, image_features):
        """Return aesthetic scores for normalized CLIP image embeddings."""
        import torch

        with torch.no_grad():
            return self.model(image_features).squeeze(-1)


def load_aesthetic_predictor(weights_path: Path | None, device) -> AestheticPredictor | None:
    """Load the aesthetic predictor only when local weights are provided."""
    if weights_path is None:
        return None
    predictor = AestheticPredictor()
    predictor.load(weights_path, device)
    return predictor


def load_image_reward_model(enabled: bool):
    """Load ImageReward only when explicitly requested."""
    if not enabled:
        return None
    patch_transformers_for_image_reward()
    try:
        import ImageReward as RM
    except ModuleNotFoundError as error:
        if error.name == "ImageReward":
            raise SystemExit(
                "ImageReward scoring requires the image-reward package. "
                "Run `uv sync` after pulling the branch, then retry."
            ) from error
        raise SystemExit(
            f"ImageReward is installed, but one of its import-time dependencies is missing: "
            f"{error.name}. Run `uv lock && uv sync` after pulling the latest branch."
        ) from error
    except ImportError as error:
        raise SystemExit(
            f"ImageReward failed to import because of a dependency conflict: {error}. "
            "Run `uv lock && uv sync` after pulling the latest branch."
        ) from error
    return RM.load("ImageReward-v1.0")


def patch_transformers_for_image_reward() -> None:
    """Patch old Transformers symbols expected by ImageReward's BLIP code."""
    try:
        import transformers.modeling_utils as modeling_utils
        from transformers.pytorch_utils import apply_chunking_to_forward
    except ImportError:
        return

    if not hasattr(modeling_utils, "apply_chunking_to_forward"):
        modeling_utils.apply_chunking_to_forward = apply_chunking_to_forward


def score_image_reward_rows(
    rows: list[dict[str, Any]],
    *,
    root_dir: Path,
    image_reward_model,
) -> None:
    """Populate ImageReward scores in-place, grouped by prompt for efficient scoring."""
    if image_reward_model is None:
        return

    rows_by_prompt: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_prompt.setdefault(str(row["prompt"]), []).append(row)

    scored_count = 0
    for prompt, prompt_rows in rows_by_prompt.items():
        image_paths = [
            str(resolve_image_path(row.get("image_path"), root_dir))
            for row in prompt_rows
        ]
        rewards = image_reward_model.score(prompt, image_paths)
        if not isinstance(rewards, list):
            rewards = [rewards]

        for row, reward in zip(prompt_rows, rewards, strict=True):
            row["image_reward_score"] = f"{float(reward):.6f}"
            scored_count += 1
        print(f"image_reward scored {scored_count}/{len(rows)} rows")


def score_rows(
    rows: list[dict[str, Any]],
    *,
    root_dir: Path,
    clip_model_name: str,
    aesthetic_weights: Path | None,
    enable_image_reward: bool,
    batch_size: int,
    device_name: str,
) -> list[dict[str, Any]]:
    """Populate CLIP, optional aesthetic, and optional ImageReward scores."""
    if batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor

    device = select_device(device_name)
    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model = CLIPModel.from_pretrained(clip_model_name).to(device)
    model.eval()
    aesthetic_predictor = load_aesthetic_predictor(aesthetic_weights, device)
    image_reward_model = load_image_reward_model(enable_image_reward)

    scored_rows: list[dict[str, Any]] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        images = [load_image(resolve_image_path(row.get("image_path"), root_dir)) for row in batch_rows]
        prompts = [str(row["prompt"]) for row in batch_rows]
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True, truncation=True)
        inputs = {name: value.to(device) for name, value in inputs.items()}

        with torch.no_grad():
            image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)
            clip_scores = (image_features * text_features).sum(dim=-1)
            aesthetic_scores = (
                aesthetic_predictor.score(image_features) if aesthetic_predictor is not None else None
            )

        for index, row in enumerate(batch_rows):
            scored_row = dict(row)
            scored_row["clip_score"] = f"{clip_scores[index].item():.6f}"
            if aesthetic_scores is not None:
                scored_row["aesthetic_score"] = f"{aesthetic_scores[index].item():.6f}"
            scored_rows.append(scored_row)

        print(f"scored {min(start + batch_size, len(rows))}/{len(rows)} rows")

    score_image_reward_rows(
        scored_rows,
        root_dir=root_dir,
        image_reward_model=image_reward_model,
    )
    return scored_rows


def build_parser() -> argparse.ArgumentParser:
    """Define CLI flags for scoring benchmark outputs."""
    parser = argparse.ArgumentParser(
        description="Score generated benchmark images with CLIP and optional aesthetic predictor.",
    )
    parser.add_argument("--metrics-input", required=True, help="Input benchmark CSV or JSONL.")
    parser.add_argument("--metrics-output", required=True, help="Output scored CSV or JSONL.")
    parser.add_argument(
        "--root-dir",
        default=".",
        help="Directory used to resolve relative image_path values. Defaults to repo root.",
    )
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-large-patch14",
        help="Hugging Face CLIP model id. Defaults to openai/clip-vit-large-patch14.",
    )
    parser.add_argument(
        "--aesthetic-weights",
        type=Path,
        help="Optional local .pth weights for a LAION-style CLIP aesthetic predictor.",
    )
    parser.add_argument(
        "--enable-imagereward",
        action="store_true",
        help="Populate image_reward_score using the optional image-reward package.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Scoring batch size.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Scoring device: auto, cuda, cpu, or mps. Defaults to auto.",
    )
    return parser


def main() -> None:
    """Read benchmark metrics, score referenced images, and write updated metrics."""
    args = build_parser().parse_args()
    rows = read_metric_rows(Path(args.metrics_input))
    if not rows:
        raise SystemExit("no metric rows found")

    scored_rows = score_rows(
        rows,
        root_dir=Path(args.root_dir),
        clip_model_name=args.clip_model,
        aesthetic_weights=args.aesthetic_weights,
        enable_image_reward=args.enable_imagereward,
        batch_size=args.batch_size,
        device_name=args.device,
    )
    saved_path = write_metric_rows(scored_rows, Path(args.metrics_output))
    print(f"wrote {len(scored_rows)} scored rows to {saved_path}")
    if args.aesthetic_weights is None:
        print("aesthetic_score was left unchanged because --aesthetic-weights was not provided")
    if not args.enable_imagereward:
        print("image_reward_score was left unchanged because --enable-imagereward was not provided")


if __name__ == "__main__":
    main()
