# AGENTS.md

## Purpose
This document provides context and guidelines for AI assistants working with this codebase. It explains the project structure, conventions, and important considerations for making changes.

## Project Overview

"FLUX Schnell Inference Optimization Study"

Experiments:

Step Count Analysis:

Quality metrics across 1-4 steps
Where does diminishing return hit?
Latency per step


Quantization Study:

FP16 baseline
FP8 (what Fal likely uses)
Mixed precision strategies
Quality delta measurement (CLIP score, aesthetic predictor)


Bottleneck Profiling:

Where does time actually go? (VAE decode, transformer blocks, flow solver)
Memory profile across resolutions
Batch size sweet spots


Attention Optimization:

Flash Attention 2 vs default
torch.compile() impact
Memory-efficient attention

**Tech Stack:**
This project uses uv as its project manager


## Directory Structure