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
