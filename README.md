# benchies

Benchmark harness for a FLUX Schnell inference optimization study.

## Current baseline

This repo is now set up for a CUDA-first workflow aimed at Google Colab style
environments, including T4 GPUs. The package uses Python 3.11 or 3.12 because
that is a much safer baseline for the PyTorch and diffusers stack than Python
3.13.

```bash
uv sync
uv run benchies --about
```

You can also run module or script commands through the same environment:

```bash
uv run python -m benchies.cli --about
```

## Installed core dependencies

- `torch`, `torchvision`
- `diffusers`
- `transformers`
- `accelerate`
- `safetensors`
- `pillow`
- `huggingface-hub`

Note: for Colab, the exact CUDA wheel for PyTorch is usually best handled in
the notebook environment

## Current entrypoint

The package exposes a small CLI:

```bash
uv run benchies --about
```

## Research prompt set

The benchmark prompt registry is stored in `src/benchies/prompts.py`.

Current sets:

- `primary`: 60 prompts across simple objects, two-object composition, spatial
  relationships, counting, complex scenes, artistic styles, photorealism, text
  rendering, and negation.
- `curated`: recommended 50-prompt subset for balanced step-count comparisons.
- `control`: 4 edge-case prompts for long-prompt and abstract stress testing.
- `all`: primary plus control prompts.

Inspect category counts:

```bash
uv run benchies --list-prompt-categories
```

Export prompts:

```bash
uv run benchies --export-prompts outputs/curated_prompts.csv --prompt-set curated
uv run benchies --export-prompts outputs/primary_prompts.json --prompt-set primary
```

Build an empty metric matrix for the recommended experiment:

```bash
uv run python scripts/build_benchmark_matrix.py \
  --prompt-set curated \
  --steps 1,2,4 \
  --runs 5 \
  --output outputs/benchmark_matrix.csv
```

The metric schema is stored in `src/benchies/metrics.py` and includes latency,
quality, resource, and metadata fields:

- `clip_score`
- `aesthetic_score`
- `image_reward_score`
- `load_time_ms`, `generation_time_ms`, `total_time_ms`
- `gpu_memory_mb`, `peak_memory_mb`
- `seed`, `guidance_scale`, `resolution`, `model_id`, `device`, `dtype`

## Lambda benchmark run commands

Run these from the repo root on the Lambda VM.

Install/sync the environment:

```bash
uv sync
```

Confirm the prompt registry:

```bash
uv run benchies --list-prompt-categories
```

Single prompt smoke test. This runs the first curated prompt only, at 4 steps:

```bash
uv run python scripts/run_flux_benchmark.py \
  --prompt-set curated \
  --limit 1 \
  --steps 4 \
  --runs 1 \
  --output-dir outputs/smoke/images \
  --metrics-output outputs/smoke/metrics.csv
```

One prompt from each primary category. This is the best first real GPU sanity
check because it covers the full category spread without running all 60
prompts:

```bash
uv run python scripts/run_flux_benchmark.py \
  --prompt-set primary \
  --one-per-category \
  --steps 1,2,4 \
  --runs 1 \
  --output-dir outputs/one_per_category/images \
  --metrics-output outputs/one_per_category/metrics.csv
```

Run one category only:

```bash
uv run python scripts/run_flux_benchmark.py \
  --prompt-set primary \
  --category spatial_relationships \
  --steps 1,2,4 \
  --runs 3 \
  --output-dir outputs/spatial_relationships/images \
  --metrics-output outputs/spatial_relationships/metrics.csv
```

Run the recommended curated benchmark. This is `50 prompts x 3 step counts x 5
runs = 750` generations:

```bash
uv run python scripts/run_flux_benchmark.py \
  --prompt-set curated \
  --steps 1,2,4 \
  --runs 5 \
  --output-dir outputs/curated_full/images \
  --metrics-output outputs/curated_full/metrics.csv
```

Run all 60 primary prompts:

```bash
uv run python scripts/run_flux_benchmark.py \
  --prompt-set primary \
  --steps 1,2,4 \
  --runs 5 \
  --output-dir outputs/primary_full/images \
  --metrics-output outputs/primary_full/metrics.csv
```

Run the 4 control prompts:

```bash
uv run python scripts/run_flux_benchmark.py \
  --prompt-set control \
  --steps 1,2,4 \
  --runs 3 \
  --output-dir outputs/control/images \
  --metrics-output outputs/control/metrics.csv
```

The runner currently records generation latency, CUDA memory when available,
image paths, seed, model, dtype, device, and resolution. Run the separate
scoring pass below to populate `clip_score` and, when weights are available,
`aesthetic_score` and `image_reward_score`.

Score generated outputs with CLIP:

```bash
uv run python scripts/score_benchmark_outputs.py \
  --metrics-input outputs/one_per_category/metrics.csv \
  --metrics-output outputs/one_per_category/metrics_scored.csv \
  --batch-size 16
```

ImageReward is a project dependency. If you add it on a branch or pull a commit
that changes dependencies, sync the VM environment first:

```bash
uv sync
```

Score generated outputs with CLIP plus ImageReward:

```bash
uv run python scripts/score_benchmark_outputs.py \
  --metrics-input outputs/one_per_category/metrics.csv \
  --metrics-output outputs/one_per_category/metrics_scored.csv \
  --enable-imagereward \
  --batch-size 16
```

Score generated outputs with CLIP plus a local aesthetic predictor checkpoint:

```bash
uv run python scripts/score_benchmark_outputs.py \
  --metrics-input outputs/curated_full/metrics.csv \
  --metrics-output outputs/curated_full/metrics_scored.csv \
  --aesthetic-weights models/aesthetic_predictor.pth \
  --batch-size 16
```

The scoring script uses `openai/clip-vit-large-patch14` by default and writes
cosine-similarity CLIP scores into the `clip_score` column. Aesthetic scoring
requires a local LAION-style CLIP aesthetic predictor checkpoint passed through
`--aesthetic-weights`; otherwise `aesthetic_score` is left empty. ImageReward
scoring requires the optional `image-reward` package and `--enable-imagereward`;
otherwise `image_reward_score` is left empty.

## Minimal FLUX Schnell run

The first real inference path is a single script that loads FLUX Schnell and
generates one image:

```bash
uv run python scripts/run_flux_schnell.py \
  --prompt "a cinematic photo of a lighthouse at dusk" \
  --output outputs/lighthouse.png
```

Useful options:

- `--seed 42`
- `--width 1024 --height 1024`
- `--steps 4`
- `--guidance-scale 0.0`
- `--max-sequence-length 256`
