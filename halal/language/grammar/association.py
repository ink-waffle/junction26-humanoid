"""Pair object mentions with bin mentions: beam search over segmentations of the mention sequence into units."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator

from halal.language.grammar import tuning
from halal.language.grammar.lexicon import DEFAULT_KEY, Tag
from halal.language.grammar.mentions import BinMention, Gap, Mention, Negation, ObjectMention, Scope, gap_between
from halal.language.grammar.tokens import Issue, Token

REMAINING = "?"      # placeholder for "the remaining bin", resolved once every other bin is known


@dataclass(frozen=True)
class Unit:
    """One clause's assignment: every object to one bin, or parallel lists paired in order."""
    objects: tuple[int, ...]
    bins: tuple[str, ...]
    forbidden: tuple[str, ...] = ()
    cost: float = 0.0
    issues: tuple[Issue, ...] = ()


@dataclass(frozen=True)
class Reading:
    """A segmentation of the mention sequence into units and unpaired mentions."""
    units: tuple[Unit, ...] = ()
    unpaired: tuple[int, ...] = ()
    cost: float = 0.0


@dataclass(frozen=True)
class _Sequence:
    mentions: list[Mention]
    gaps: list[Gap]              # gaps[i] stands between mentions i and i + 1
    cued: list[bool]             # mention i shares a sentence with "respectively"


def candidate_readings(mentions: list[Mention], tokens: list[Token]) -> list[Reading]:
    """Readings of the whole sequence, cheapest first by the costs known before interpretation."""
    sequence = _Sequence(mentions, [gap_between(tokens, a, b) for a, b in zip(mentions, mentions[1:])],
                         _cued(mentions, tokens))
    frontier: list[list[Reading]] = [[] for _ in range(len(mentions) + 1)]
    frontier[0].append(Reading())
    for start in range(len(mentions)):
        for partial in sorted(frontier[start], key=lambda reading: reading.cost)[:tuning.BEAM_WIDTH]:
            for end, reading in _extensions(sequence, partial, start):
                frontier[end].append(reading)
    return sorted(frontier[-1], key=lambda reading: reading.cost)[:tuning.MAX_CANDIDATES]


def _cued(mentions: list[Mention], tokens: list[Token]) -> list[bool]:
    """Per mention: does its sentence contain 'respectively' / 'in that order' / 'from left to right'?"""
    sentence, cued_sentences, sentence_of = 0, set(), []
    for token in tokens:
        sentence += token.tag is Tag.BREAK
        sentence_of.append(sentence)
        if token.tag is Tag.RESPECTIVELY:
            cued_sentences.add(sentence)
    return [sentence_of[mention.start] in cued_sentences for mention in mentions]


def _extensions(sequence: _Sequence, partial: Reading, start: int) -> Iterator[tuple[int, Reading]]:
    """Ways to consume the mention at `start`: leave it unpaired, attach it to the last bin, or open a unit."""
    mention = sequence.mentions[start]
    skipped = 0.0 if mention.soft else tuning.REDUNDANT
    yield start + 1, replace(partial, unpaired=partial.unpaired + (start,), cost=partial.cost + skipped)
    previous_bin = _previous_bin(partial)
    if isinstance(mention, ObjectMention) and mention.add and previous_bin:
        note = Issue(f"'{mention.key}' joins the previous group's bin", tuning.F_ATTACHED)
        yield start + 1, _with_unit(partial, Unit((start,), (previous_bin,), cost=tuning.ATTACHED, issues=(note,)))
    boundary = _boundary_cost(sequence, start)
    for end in range(start + 2, min(len(sequence.mentions), start + tuning.MAX_UNIT_SPAN) + 1):
        if _kind_changes(sequence.mentions[start:end]) > 1:
            break
        unit = _unit(sequence, start, end, previous_bin)
        if unit:
            yield end, _with_unit(partial, replace(unit, cost=unit.cost + boundary))


def _with_unit(partial: Reading, unit: Unit) -> Reading:
    return replace(partial, units=partial.units + (unit,), cost=partial.cost + unit.cost)


def _previous_bin(partial: Reading) -> str | None:
    """The single stated bin of the last unit, for 'the same bin' and 'blue too'."""
    bins = partial.units[-1].bins if partial.units else ()
    return bins[0] if len(bins) == 1 and bins[0] != REMAINING else None


def _kind_changes(mentions: list[Mention]) -> int:
    kinds = [isinstance(mention, ObjectMention) for mention in mentions]
    return sum(a != b for a, b in zip(kinds, kinds[1:]))


def _boundary_cost(sequence: _Sequence, start: int) -> float:
    """Clauses are normally set apart by punctuation or 'and'; a verb or preposition should not be left over."""
    if start == 0 or sequence.gaps[start - 1].separated:
        return 0.0
    dangling_link = sequence.gaps[start - 1].has(Tag.LINK, Tag.ARROW)
    return tuning.LINK_BETWEEN_CLAUSES if dangling_link else tuning.UNSEPARATED


def _unit(sequence: _Sequence, start: int, end: int, previous_bin: str | None) -> Unit | None:
    """The unit spanning mentions [start, end): objects then bins or bins then objects; None when ill-formed."""
    members = sequence.mentions[start:end]
    if _kind_changes(members) != 1:
        return None
    objects = tuple(i for i in range(start, end) if isinstance(sequence.mentions[i], ObjectMention))
    bins = [mention for mention in members if isinstance(mention, BinMention)]
    positions = [_position(mention, previous_bin) for mention in bins]
    allowed = _unique(p for p, mention in zip(positions, bins) if not mention.negated)
    forbidden = _unique(p for p, mention in zip(positions, bins) if mention.negated)
    if None in positions or (len(allowed) > 1 and (forbidden or len(allowed) != len(objects))):
        return None
    cost, issues = _unit_cost(sequence, start, end, len(allowed) > 1)
    return Unit(objects, allowed, forbidden, cost, issues)


def _position(mention: BinMention, previous_bin: str | None) -> str | None:
    if mention.anaphor == "same":
        return previous_bin
    return REMAINING if mention.anaphor == "remaining" else mention.position


def _unique(items: Iterator[str | None]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


def _unit_cost(sequence: _Sequence, start: int, end: int, distributes: bool) -> tuple[float, tuple[Issue, ...]]:
    """Cost of reading mentions [start, end) as one unit, with the doubts that cost stands for."""
    members = sequence.mentions[start:end]
    turn = next(i for i in range(start, end - 1) if _kind_changes(sequence.mentions[i:i + 2]))
    charges = [_run_charge(sequence, i) for i in range(start, end - 1) if i != turn]
    charges.append(_transition_charge(sequence.gaps[turn], isinstance(members[0], BinMention)))
    if distributes and not any(sequence.cued[start:end]):
        charges.append((tuning.DISTRIBUTION, Issue("lists paired in order without 'respectively'",
                                                   tuning.F_DISTRIBUTION)))
    charges.append(_object_charge([m for m in members if isinstance(m, ObjectMention)]))
    cost = sum(charge for charge, _ in charges)
    return cost, tuple(dict.fromkeys(issue for _, issue in charges if issue))


def _run_charge(sequence: _Sequence, index: int) -> tuple[float, Issue | None]:
    """A list should hold only commas and conjunctions; a contrasted bin ('left, not right') is exempt."""
    gap = sequence.gaps[index]
    pair = sequence.mentions[index:index + 2]
    if gap.list_like or any(isinstance(m, BinMention) and m.negated for m in pair):
        return 0.0, None
    if gap.has(Tag.BREAK, Tag.ARROW, Tag.CLAUSE):
        return tuning.RUN_BROKEN, Issue("a list runs across a sentence break or colon", tuning.F_RUN_BROKEN)
    mixed = tuning.RUN_UNKNOWN if gap.tags.isdisjoint({Tag.LINK, Tag.LEAVE, Tag.NOT, Tag.VOID}) else tuning.RUN_MIXED
    return mixed, Issue("a list is interrupted by other words", tuning.F_RUN_MIXED)


def _transition_charge(gap: Gap, bin_first: bool) -> tuple[float, Issue | None]:
    """Objects and their bin are normally joined by a verb, preposition or arrow, not split by punctuation."""
    linked = gap.has(Tag.LINK, Tag.ARROW, Tag.NOT, Tag.LEAVE)
    if gap.has(Tag.BREAK):
        return tuning.ACROSS_BREAK, Issue("objects and bin stand in different sentences", tuning.F_ACROSS_BREAK)
    if gap.has(Tag.CLAUSE):
        return tuning.ACROSS_CLAUSE, Issue("objects and bin stand on two sides of 'but'", tuning.F_ACROSS_CLAUSE)
    if gap.has(Tag.AND) and not linked:
        return tuning.ACROSS_AND, Issue("only 'and' joins objects and bin", tuning.F_ACROSS_AND)
    if gap.has(Tag.COMMA) and not linked:
        return tuning.ACROSS_COMMA, Issue("only a comma joins objects and bin", tuning.F_ACROSS_COMMA)
    if bin_first and gap.directional and not gap.has(Tag.ARROW):
        return tuning.BIN_FIRST_DIRECTIONAL, Issue("bin named before a preposition that points to it",
                                                   tuning.F_BIN_FIRST_DIRECTIONAL)
    if gap.has(Tag.OTHER):
        return 0.0, Issue("unknown wording between objects and bin", tuning.UNKNOWN_LINK)
    return (tuning.LINKED_AFTER_COMMA if gap.separated else 0.0), None


def _object_charge(objects: list[ObjectMention]) -> tuple[float, Issue | None]:
    """Doubtful object lists: a placement after 'leave', 'everything' beside specifics, a lone exception."""
    cost = tuning.DEPOSIT_AFTER_LEAVE if any(m.leave for m in objects) else 0.0
    quantifiers = [m for m in objects if is_quantifier(m)]
    if quantifiers and any(m.negated is Negation.NONE for m in objects if m not in quantifiers):
        cost += tuning.QUANTIFIER_IN_LIST
    if not quantifiers and any(m.negated is Negation.WHOLE for m in objects):
        return cost + tuning.LONE_EXCEPTION, Issue("an excepted group is given its own bin", tuning.F_LONE_EXCEPTION)
    return cost, None


def is_quantifier(mention: ObjectMention) -> bool:
    """'everything', 'the rest': a mention that stands for all objects not named more specifically."""
    return mention.scope is not Scope.PLAIN and mention.key == DEFAULT_KEY and mention.negated is Negation.NONE
