from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchies",
        description="Utilities for FLUX Schnell benchmarking experiments.",
    )
    parser.add_argument(
        "--about",
        action="store_true",
        help="Show the current project setup summary.",
    )
    parser.add_argument(
        "--flux-schnell-script",
        action="store_true",
        help="Show the command for the minimal FLUX Schnell generation script.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.about:
        print("benchies: FLUX Schnell inference benchmark scaffold")
        print("baseline target: NVIDIA CUDA (Colab T4 class GPU)")
        print("entrypoint script: uv run python scripts/run_flux_schnell.py --prompt '...'")
        return

    if args.flux_schnell_script:
        print("uv run python scripts/run_flux_schnell.py --prompt \"a cinematic photo of a lighthouse at dusk\"")
        return

    print("benchies is configured. Run `benchies --about` for setup details.")


if __name__ == "__main__":
    main()
