"""CLI: python -m halal.language "Put the red ones left and the rest in the middle" prints the parsed mapping."""
from __future__ import annotations

import argparse

from halal.language import PARSERS, parse_instruction


def main() -> None:
    cli = argparse.ArgumentParser(prog="python -m halal.language", description=__doc__)
    cli.add_argument("instruction", nargs="+", help="instruction text (quoted or as separate words)")
    cli.add_argument("--parser", default="rules", choices=sorted(PARSERS))
    args = cli.parse_args()
    parsed = parse_instruction(" ".join(args.instruction), args.parser)
    print(f"mapping:    {parsed.mapping}")
    print(f"confidence: {parsed.confidence:.2f}  (parser: {parsed.parser})")
    for note in parsed.notes:
        print(f"note:       {note}")


if __name__ == "__main__":
    main()
