from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from diffusers import FluxPipeline
from PIL import Image

DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-schnell"


@dataclass(slots=True)
class FluxSchnellConfig:
    prompt: str
    output: Path
    model_id: str = DEFAULT_MODEL_ID
    negative_prompt: str | None = None
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 4
    guidance_scale: float = 0.0
    seed: int = 42
    max_sequence_length: int = 256


def select_device() -> str:
    """Pick the best available accelerator, preferring CUDA."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def select_dtype(device: str) -> torch.dtype:
    """Use fp16 on CUDA, otherwise stay on fp32 for compatibility."""
    if device == "cuda":
        return torch.float16
    return torch.float32


def build_generator(device: str, seed: int) -> torch.Generator:
    """Create a seeded generator compatible with the selected backend."""
    if device == "cuda":
        return torch.Generator(device="cuda").manual_seed(seed)
    return torch.Generator().manual_seed(seed)


def load_pipeline(model_id: str, device: str, dtype: torch.dtype) -> FluxPipeline:
    """Load FLUX Schnell and place it on the chosen device."""
    pipeline = FluxPipeline.from_pretrained(model_id, torch_dtype=dtype)
    if device == "cuda":
        pipeline.enable_model_cpu_offload()
    else:
        pipeline = pipeline.to(device)
    return pipeline


def generate_image(pipeline: FluxPipeline, config: FluxSchnellConfig, device: str) -> Image.Image:
    """Run one generation pass and return the first produced image."""
    generator = build_generator(device, config.seed)
    result = pipeline(
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        width=config.width,
        height=config.height,
        guidance_scale=config.guidance_scale,
        num_inference_steps=config.num_inference_steps,
        max_sequence_length=config.max_sequence_length,
        generator=generator,
    )
    return result.images[0]


def save_image(image: Image.Image, output_path: Path) -> Path:
    """Persist the generated image, creating parent directories if needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def run_generation(config: FluxSchnellConfig) -> Path:
    """Load the pipeline, generate one image, and print basic timings."""
    device = select_device()
    dtype = select_dtype(device)

    start = perf_counter()
    pipeline = load_pipeline(config.model_id, device=device, dtype=dtype)
    load_duration = perf_counter() - start

    start = perf_counter()
    image = generate_image(pipeline, config=config, device=device)
    generation_duration = perf_counter() - start

    saved_path = save_image(image, config.output)

    print(f"model: {config.model_id}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"seed: {config.seed}")
    print(f"steps: {config.num_inference_steps}")
    print(f"resolution: {config.width}x{config.height}")
    print(f"pipeline_load_seconds: {load_duration:.2f}")
    print(f"generation_seconds: {generation_duration:.2f}")
    print(f"saved_to: {saved_path}")

    return saved_path


def build_parser() -> argparse.ArgumentParser:
    """Define the CLI for a single FLUX Schnell generation run."""
    parser = argparse.ArgumentParser(
        description="Generate a single image with FLUX Schnell.",
    )
    parser.add_argument("--prompt", required=True, help="Text prompt for generation.")
    parser.add_argument(
        "--output",
        default="outputs/flux-schnell.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id to load.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Optional negative prompt.",
    )
    parser.add_argument("--width", type=int, default=1024, help="Output width in pixels.")
    parser.add_argument("--height", type=int, default=1024, help="Output height in pixels.")
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Number of diffusion steps. FLUX Schnell is usually run with a low step count.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale. FLUX Schnell commonly uses 0.0.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=256,
        help="Maximum text sequence length passed into the pipeline.",
    )
    return parser


def parse_args() -> FluxSchnellConfig:
    """Translate CLI flags into a typed generation config."""
    args = build_parser().parse_args()
    return FluxSchnellConfig(
        prompt=args.prompt,
        output=Path(args.output),
        model_id=args.model_id,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        max_sequence_length=args.max_sequence_length,
    )


def main() -> None:
    """CLI entrypoint for the standalone FLUX Schnell runner."""
    run_generation(parse_args())


if __name__ == "__main__":
    main()
