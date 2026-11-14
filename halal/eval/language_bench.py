"""Instruction parsers against the held-out paraphrase benchmark: exact-mapping accuracy and expected points per object.

    python -m halal.eval.language_bench --split dev --failures 3
    python -m halal.eval.language_bench --split test --file halal/eval/paraphrases_fresh.yaml
    python -m halal.eval.language_bench --split test --file halal/eval/paraphrases_black.yaml    (a fourth colour, named literally)
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from halal.eval import paraphrases
from halal.language import PARSERS, parse_instruction

POINTS_CORRECT, POINTS_WRONG = 2.0, -1.0


@dataclass
class Tally:
    cases: int = 0
    exact: int = 0
    confident: int = 0
    confident_wrong: int = 0
    points: float = 0.0


def object_points(predicted: dict, truth: dict) -> float:
    """Mean sort_language points per object type if the robot executed `predicted` perfectly."""
    scores = [0.0 if p is None else POINTS_CORRECT if p == truth[k] else POINTS_WRONG for k, p in predicted.items()]
    return sum(scores) / len(scores)


def run(parser: str, split: str | None, threshold: float) -> tuple[dict[str, Tally], dict[str, list[tuple]]]:
    tallies: dict[str, Tally] = defaultdict(Tally)
    failures: dict[str, list[tuple]] = defaultdict(list)
    for entry, text, mapping in paraphrases.cases(split=split):
        parsed = parse_instruction(text, parser)
        predicted, truth = paraphrases.assignment(parsed.mapping, text), paraphrases.assignment(mapping, text)
        acted = predicted if parsed.confidence >= threshold else dict.fromkeys(predicted)
        tally = tallies[entry.category]
        tally.cases += 1
        tally.exact += predicted == truth
        tally.confident += parsed.confidence >= threshold
        tally.confident_wrong += parsed.confidence >= threshold and predicted != truth
        tally.points += object_points(acted, truth)
        if predicted != truth:
            failures[entry.category].append((entry.id, text, mapping, parsed.mapping, parsed.confidence))
    return tallies, failures


def report(tallies: dict[str, Tally]) -> None:
    print(f"{'category':<20}{'cases':>7}{'exact':>8}{'confident':>11}{'conf+wrong':>12}{'points/obj':>12}")
    total = Tally()
    for category in paraphrases.CATEGORIES:
        t = tallies.get(category)
        if t is None:
            continue
        print(f"{category:<20}{t.cases:>7}{t.exact / t.cases:>8.3f}{t.confident / t.cases:>11.3f}"
              f"{t.confident_wrong / t.cases:>12.3f}{t.points / t.cases:>12.3f}")
        for field in ("cases", "exact", "confident", "confident_wrong", "points"):
            setattr(total, field, getattr(total, field) + getattr(t, field))
    print(f"{'ALL':<20}{total.cases:>7}{total.exact / total.cases:>8.3f}{total.confident / total.cases:>11.3f}"
          f"{total.confident_wrong / total.cases:>12.3f}{total.points / total.cases:>12.3f}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parser", default="rules", choices=sorted(PARSERS))
    ap.add_argument("--split", default="dev", choices=("dev", "test", "all"))
    ap.add_argument("--threshold", type=float, default=0.5, help="confidence below which the robot would not act")
    ap.add_argument("--failures", type=int, default=0, help="print up to N distinct failing entries per category")
    ap.add_argument("--file", type=Path, help="another benchmark file, e.g. halal/eval/paraphrases_fresh.yaml")
    args = ap.parse_args(argv)
    if args.file:
        paraphrases.use(args.file)
    tallies, failures = run(args.parser, None if args.split == "all" else args.split, args.threshold)
    report(tallies)
    for category, rows in failures.items():
        seen: set[str] = set()
        for entry_id, text, truth, predicted, confidence in rows:
            if entry_id in seen or len(seen) >= args.failures:
                continue
            seen.add(entry_id)
            print(f"\n[{category}] {entry_id} (confidence {confidence:.2f})\n  text:      {text}\n  truth:     {truth}\n  predicted: {predicted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
