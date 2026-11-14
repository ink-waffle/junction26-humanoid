"""The organisers' sim evaluator with our instruction source: `--instructions unseen` swaps training templates for held-out paraphrases."""
from __future__ import annotations

import argparse
import sys
import zlib

import numpy as np

from aaltonomous.interface import env as env_module
from aaltonomous.interface import tasks
from aaltonomous.sim import evaluate

from . import paraphrases

UNSEEN_CATEGORIES = ("color_full", "bin_synonym", "color_respectively")


def unseen_instruction(task: str, rng: np.random.Generator) -> tuple[str, dict[str, str]]:
    """Same rng draws and mapping as the training sampler (layouts stay paired across legs), reworded by a paraphrase."""
    text, mapping = tasks.make_instruction(task, rng)
    if task != "sort_language":
        return text, mapping
    pool = paraphrases.load(split="test", categories=UNSEEN_CATEGORIES)
    pick = pool[np.random.default_rng(zlib.crc32(text.encode())).integers(len(pool))]
    return paraphrases.render(pick, list(mapping), list(mapping.values()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instructions", choices=("train", "unseen"), default="train")
    args, rest = ap.parse_known_args(argv)
    if args.instructions == "unseen":
        env_module.make_instruction = unseen_instruction
    return evaluate.main(rest)


if __name__ == "__main__":
    sys.exit(main())
