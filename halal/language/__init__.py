"""Instruction -> mapping: which objects (by colour, shape or both) go to which bin; parsers sit behind one interface."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from halal.language.grammar.lexicon import DEFAULT_KEY, SHAPES

ObjectKey = str      # "red" | "cube" | "red/cube" | "default" (colour, shape, colour/shape, everything else)


@dataclass(frozen=True)
class ParsedInstruction:
    """What an instruction asks for; objects that match no key are not to be moved."""
    mapping: dict[ObjectKey, str]       # object key -> "left" | "center" | "right"
    confidence: float                   # 0..1; 1.0 = unambiguous and complete, below ~0.5 = treat as a guess
    parser: str                         # name of the adapter that produced it
    notes: tuple[str, ...] = ()         # what was matched and what stayed doubtful

    def bin_for(self, color: str, shape: str | None) -> str | None:
        """Bin for an object; with an unknown shape, the bin every possible shape agrees on, else None."""
        bins = {self._resolve(color, candidate) for candidate in ([shape] if shape else SHAPES)}
        return bins.pop() if len(bins) == 1 else None

    def _resolve(self, color: str, shape: str) -> str | None:
        """Most specific key wins: colour/shape, colour, shape, default."""
        for key in (f"{color}/{shape}", color, shape, DEFAULT_KEY):
            if key in self.mapping:
                return self.mapping[key]
        return None


class InstructionParser(Protocol):
    """Adapter interface: one way of turning instruction text into a mapping."""
    name: str

    def parse(self, text: str) -> ParsedInstruction: ...


from halal.language.parsers.rules import RulesParser  # noqa: E402  (adapters import the types defined above)

PARSERS: dict[str, InstructionParser] = {RulesParser.name: RulesParser()}


def parse_instruction(text: str, parser: str = "rules") -> ParsedInstruction:
    """Parse with the named adapter; adapters never raise on any text."""
    return PARSERS[parser].parse(text)
