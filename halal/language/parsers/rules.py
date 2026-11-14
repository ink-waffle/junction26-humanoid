"""Deterministic instruction parser: tag tokens, chunk mentions, pair objects with bins, read off the mapping."""
from __future__ import annotations

from halal.language import ParsedInstruction
from halal.language.grammar import tuning
from halal.language.grammar.association import Reading, candidate_readings
from halal.language.grammar.mentions import Mention, find_mentions
from halal.language.grammar.semantics import Interpretation, interpret
from halal.language.grammar.tokens import Issue, Token, tag


class RulesParser:
    """Rule-based parser; never raises, returns an empty mapping with confidence 0 for text it cannot read."""
    name = "rules"

    def parse(self, text: str) -> ParsedInstruction:
        try:
            return self._parse(text)
        except Exception as error:      # a parse failure must never take the policy down
            return ParsedInstruction({}, 0.0, self.name, (f"parser error: {error!r}",))

    def _parse(self, text: str) -> ParsedInstruction:
        tokens = tag(text) if isinstance(text, str) else []
        mentions = find_mentions(tokens)
        ranked = _ranked(mentions, tokens)
        reading, meaning = ranked[0]
        if not meaning.informative:
            return ParsedInstruction({}, 0.0, self.name, ("no object group could be paired with a bin",))
        issues = _issues(reading, meaning, mentions, ranked[1:])
        confidence = 1.0
        for issue in issues:
            confidence *= issue.factor
        notes = _matches(reading, mentions, tokens) + tuple(issue.note for issue in issues)
        return ParsedInstruction(meaning.mapping, round(confidence, 2), self.name, notes)


def _ranked(mentions: list[Mention], tokens: list[Token]) -> list[tuple[Reading, Interpretation]]:
    """Candidate readings with their meanings, cheapest total cost first."""
    interpreted = [(reading, interpret(reading, mentions)) for reading in candidate_readings(mentions, tokens)]
    return sorted(interpreted, key=lambda pair: pair[1].cost)


def _issues(reading: Reading, meaning: Interpretation, mentions: list[Mention],
            rivals: list[tuple[Reading, Interpretation]]) -> tuple[Issue, ...]:
    """Every doubt behind the chosen reading: guessed words, awkward pairings, loose ends, a close rival."""
    used = [mentions[index] for unit in reading.units for index in unit.objects] + \
           [mention for mention in mentions if mention.soft]
    issues = [issue for mention in mentions if mention in used or not mention.soft for issue in mention.issues]
    issues += [issue for unit in reading.units for issue in unit.issues]
    issues += meaning.issues
    close = [rival for _, rival in rivals
             if rival.mapping != meaning.mapping and rival.cost - meaning.cost < tuning.AMBIGUITY_MARGIN]
    if close:
        issues.append(Issue(f"another reading is as plausible: {close[0].mapping}", tuning.F_AMBIGUOUS))
    return tuple(dict.fromkeys(issues))


def _matches(reading: Reading, mentions: list[Mention], tokens: list[Token]) -> tuple[str, ...]:
    """The text span behind each unit, for the notes."""
    spans = []
    for unit in reading.units:
        members = [mentions[index] for index in unit.objects]
        start, end = min(m.start for m in members), max(m.end for m in members)
        spans.append(f"{' '.join(token.text for token in tokens[start:end])} -> {' / '.join(unit.bins) or 'no bin'}")
    return tuple(f"matched: {span}" for span in spans)
