from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal


PromptSplit = Literal["primary", "control"]


@dataclass(frozen=True, slots=True)
class PromptCase:
    """One benchmark prompt with its intended evaluation axis."""

    prompt_id: str
    category: str
    prompt: str
    purpose: str
    expected_clip_range: str
    split: PromptSplit = "primary"
    curated: bool = False


CATEGORY_PURPOSES: dict[str, str] = {
    "simple_objects": "Baseline quality ceiling and fast convergence.",
    "two_object_composition": "Attribute binding and basic two-object composition.",
    "spatial_relationships": "Left/right/above/below/inside positional reasoning.",
    "counting": "Known hard case for exact quantities and precision.",
    "complex_scenes": "Multi-object scene coherence and real-use composition.",
    "artistic_styles": "Style transfer coherence and aesthetic quality.",
    "photorealism": "Fine detail, lighting, texture, and realism.",
    "text_rendering": "Known hard case for legible generated text.",
    "negation": "Instruction following for absence constraints.",
    "long_prompts": "Long-context stress test.",
    "abstract": "Ambiguous concept rendering and semantic robustness.",
}


CATEGORY_EXPECTED_CLIP_RANGES: dict[str, str] = {
    "simple_objects": "0.85+",
    "two_object_composition": "0.75-0.80",
    "spatial_relationships": "0.70-0.75",
    "counting": "0.60-0.70",
    "complex_scenes": "0.70-0.75",
    "artistic_styles": "0.65-0.72",
    "photorealism": "0.72-0.78",
    "text_rendering": "0.50-0.65",
    "negation": "0.65-0.70",
    "long_prompts": "variable",
    "abstract": "variable",
}


_PROMPTS_BY_CATEGORY: dict[str, list[str]] = {
    "simple_objects": [
        "a red apple",
        "a blue coffee mug",
        "a wooden chair",
        "a green plant in a pot",
        "a yellow tennis ball",
        "a silver fork",
        "a white pillow",
        "a black leather shoe",
    ],
    "two_object_composition": [
        "a red cube and a blue sphere",
        "a small cat next to a large dog",
        "a metal spoon on a wooden table",
        "a green apple beside a yellow banana",
        "a glass bottle containing orange juice",
        "a white bird perched on a brown branch",
        "a round clock above a square painting",
        "a fluffy pillow on a leather couch",
    ],
    "spatial_relationships": [
        "a book to the left of a lamp",
        "a cat sitting inside a cardboard box",
        "a plant behind a window",
        "a ball between two chairs",
        "a cup on top of a saucer",
        "a painting hanging above a fireplace",
        "a dog lying under a table",
        "a bicycle leaning against a wall",
    ],
    "counting": [
        "exactly three apples in a row",
        "two cats and one dog",
        "five colorful balloons",
        "four books stacked vertically",
        "six eggs in a carton",
        "a hand with five fingers clearly visible",
    ],
    "complex_scenes": [
        "a cozy living room with a fireplace, armchair, and bookshelf",
        "a busy city street with cars, pedestrians, and traffic lights",
        "a kitchen counter with a cutting board, knife, and vegetables",
        "a desk with a laptop, coffee mug, and scattered papers",
        "a garden with flowers, a stone path, and a wooden bench",
        "a beach scene with umbrellas, people, and waves",
        "a forest clearing with trees, sunlight, and a deer",
        "a workspace with tools, blueprints, and a hardhat",
    ],
    "artistic_styles": [
        "a mountain landscape in the style of Vincent van Gogh",
        "a portrait of a woman in the style of Pablo Picasso",
        "a city skyline in minimalist geometric style",
        "a forest scene as a watercolor painting",
        "a cat in the style of Japanese woodblock prints",
        "a still life in the style of Renaissance oil painting",
        "an abstract composition with bold primary colors",
        "a steampunk airship with intricate mechanical details",
    ],
    "photorealism": [
        "professional product photography of a luxury watch on marble",
        "macro photography of water droplets on a leaf",
        "portrait photograph of a elderly man with weathered skin",
        "food photography of a gourmet burger with sesame bun",
        "architectural photography of a modern glass skyscraper",
        "wildlife photography of a snow leopard in mountains",
    ],
    "text_rendering": [
        "a neon sign that says OPEN",
        "a book cover with the title THE FUTURE in large letters",
        "a coffee shop menu board showing ESPRESSO 3 DOLLARS",
        "a street sign pointing to MAIN STREET",
    ],
    "negation": [
        "a table with books but no lamp",
        "a red apple without any leaves",
        "a beach scene with no people",
        "a living room with furniture but no television",
    ],
}


_CONTROL_PROMPTS_BY_CATEGORY: dict[str, list[str]] = {
    "long_prompts": [
        "a hyperrealistic photograph of a steampunk workshop interior featuring a wooden workbench covered with brass gears and copper pipes, illuminated by warm Edison bulbs hanging from exposed ceiling beams, with tools scattered across the surface including wrenches and screwdrivers, a leather apron hanging on a hook, and blueprints pinned to a corkboard on the brick wall",
    ],
    "abstract": [
        "the concept of time",
        "a feeling of nostalgia",
        "dreams and memories intertwined",
    ],
}


_CURATED_COUNTS: dict[str, int] = {
    "simple_objects": 6,
    "two_object_composition": 6,
    "spatial_relationships": 8,
    "counting": 4,
    "complex_scenes": 8,
    "artistic_styles": 6,
    "photorealism": 6,
    "text_rendering": 3,
    "negation": 3,
}


def _build_prompt_cases() -> tuple[PromptCase, ...]:
    """Normalize raw prompt strings into stable PromptCase records."""
    cases: list[PromptCase] = []
    for category, prompts in _PROMPTS_BY_CATEGORY.items():
        curated_count = _CURATED_COUNTS[category]
        for index, prompt in enumerate(prompts, start=1):
            cases.append(
                PromptCase(
                    prompt_id=f"{category}_{index:02d}",
                    category=category,
                    prompt=prompt,
                    purpose=CATEGORY_PURPOSES[category],
                    expected_clip_range=CATEGORY_EXPECTED_CLIP_RANGES[category],
                    curated=index <= curated_count,
                )
            )

    for category, prompts in _CONTROL_PROMPTS_BY_CATEGORY.items():
        for index, prompt in enumerate(prompts, start=1):
            cases.append(
                PromptCase(
                    prompt_id=f"{category}_{index:02d}",
                    category=category,
                    prompt=prompt,
                    purpose=CATEGORY_PURPOSES[category],
                    expected_clip_range=CATEGORY_EXPECTED_CLIP_RANGES[category],
                    split="control",
                )
            )

    return tuple(cases)


PROMPT_CASES: tuple[PromptCase, ...] = _build_prompt_cases()
PRIMARY_PROMPT_CASES: tuple[PromptCase, ...] = tuple(
    prompt_case for prompt_case in PROMPT_CASES if prompt_case.split == "primary"
)
CURATED_PROMPT_CASES: tuple[PromptCase, ...] = tuple(
    prompt_case for prompt_case in PRIMARY_PROMPT_CASES if prompt_case.curated
)
CONTROL_PROMPT_CASES: tuple[PromptCase, ...] = tuple(
    prompt_case for prompt_case in PROMPT_CASES if prompt_case.split == "control"
)


def prompts_by_category(prompt_cases: Iterable[PromptCase] = PROMPT_CASES) -> dict[str, list[PromptCase]]:
    """Group prompt cases by category while preserving their source order."""
    grouped: dict[str, list[PromptCase]] = {}
    for prompt_case in prompt_cases:
        grouped.setdefault(prompt_case.category, []).append(prompt_case)
    return grouped


def write_prompt_cases_json(prompt_cases: Iterable[PromptCase], output_path: Path) -> Path:
    """Write prompt metadata to JSON for inspection or VM transfer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([asdict(prompt_case) for prompt_case in prompt_cases], indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def write_prompt_cases_csv(prompt_cases: Iterable[PromptCase], output_path: Path) -> Path:
    """Write prompt metadata to CSV for spreadsheet-based review."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(prompt_case) for prompt_case in prompt_cases]
    fieldnames = list(PromptCase.__dataclass_fields__)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
