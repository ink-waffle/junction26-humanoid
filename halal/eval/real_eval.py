"""Dev-only evaluator with the real robot's words: sim_eval with the colours renamed and the real sort_fixed rule.

    python -m halal.eval.proxy --evaluator halal.eval.real_eval ...     (words and rule: halal/ops/real_sim.yaml)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from aaltonomous.interface import env as env_module
from aaltonomous.interface import tasks

from . import paraphrases, sim_eval

CONFIG = Path(__file__).resolve().parents[1] / "ops" / "real_sim.yaml"
Instruction = tuple[str, dict[str, str]]


def reword(text: str, words: dict[str, str]) -> str:
    """Canonical colour words replaced by the real ones, capitalisation kept."""
    def swap(match: re.Match) -> str:
        word = words[match.group().lower()]
        return word.capitalize() if match.group()[0].isupper() else word
    return re.sub(r"\b(" + "|".join(words) + r")\b", swap, text, flags=re.IGNORECASE)


def fixed_instruction(mapping: dict[str, str]) -> str:
    """The organisers' fixed sentence for another colour -> bin rule, bins named left to right."""
    by_bin = {b: c for c, b in mapping.items()}
    slots = {f"{kind}{i}": word for i, b in enumerate(tasks.BINS) for kind, word in (("c", by_bin[b]), ("b", b))}
    return tasks.TEMPLATES[0].format(**slots)


def real_instructions(make: Callable[[str, np.random.Generator], Instruction], cfg: dict) -> Callable[[str, np.random.Generator], Instruction]:
    """The training sampler with real words; sort_fixed gets the real rule for the instruction AND the scoring."""
    def sample(task: str, rng: np.random.Generator) -> Instruction:
        text, mapping = make(task, rng)
        if task == "sort_fixed":
            mapping = dict(cfg["fixed_mapping"])
            text = fixed_instruction(mapping)
        return reword(text, cfg["words"]), mapping
    return sample


def main(argv: list[str] | None = None) -> int:
    cfg = yaml.safe_load(CONFIG.read_text())
    render = paraphrases.render
    tasks.make_instruction = env_module.make_instruction = real_instructions(tasks.make_instruction, cfg)
    paraphrases.render = lambda *args: (lambda text, mapping: (reword(text, cfg["words"]), mapping))(*render(*args))
    return sim_eval.main(argv)


if __name__ == "__main__":
    sys.exit(main())
