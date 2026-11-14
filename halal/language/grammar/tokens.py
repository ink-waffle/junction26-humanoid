"""Normalise, tokenise and tag an instruction; context rules settle the words that have several readings."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterator

from halal.language.grammar import tuning
from halal.language.grammar.lexicon import (BINS, COLOUR_EMOJI, FUZZY_TARGETS, INTENSIFIED, MAX_PHRASE,
                                            MODAL_FOLLOWERS, NOT_TYPOS, NUMBERS, PHRASES, RELATIVE_PRONOUNS,
                                            ROW_JOINERS, ROW_STARTERS, STAY_WORDS, SURFACES, WH_WORDS, WORDS, Tag)

_QUOTES = str.maketrans({"’": "'", "‘": "'", "`": "'", "“": '"', "”": '"',
                         **{emoji: f" {colour} " for emoji, colour in COLOUR_EMOJI.items()}})
_BULLET = re.compile(r"(?m)^\s*[-*•·]+\s+")
_ARROW = re.compile(r"<-+|-+>|=+>|[→⇒➔➜➝⟶➡↦←⇐⬅»>]")
_DASH = re.compile(r"[–—]|\s-+\s")
_CONTRACTIONS = (
    (re.compile(r"\bcan'?t\b|\bcannot\b"), "can not"),
    (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\b(do|does|did|is|are|was|were|should|must|would|could|need|have|has)n'?t\b"), r"\1 not"),
    (re.compile(r"\b(what|that|whatever|it|there|here|who|which)'s\b"), r"\1 is"),
    (re.compile(r"'(s|d)\b"), ""),
    (re.compile(r"'re\b"), " are"),
    (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'ve\b"), " have"),
)
_WORD = re.compile(r"[^\W\d_]+|\d+(?:st|nd|rd|th)?|[.;!?,:=→()&/+#]")
_SEPARATORS = (Tag.BREAK, Tag.COMMA, Tag.CLAUSE)


@dataclass(frozen=True)
class Issue:
    """Something the parser had to assume; multiplies confidence by `factor`."""
    note: str
    factor: float


@dataclass
class Token:
    """One tagged word, phrase or punctuation mark."""
    text: str
    tag: Tag
    value: str = ""
    issue: Issue | None = None
    soft: bool = False          # a bin word that may stay unpaired ("from left to right")


def tag(text: str) -> list[Token]:
    """Tagged tokens of an instruction, ambiguous words resolved by their context."""
    tokens = _lookup(_WORD.findall(normalise(text)))
    for rule in _CONTEXT_RULES:
        tokens = rule(tokens)
    return tokens


def normalise(text: str) -> str:
    """Lower-case text with unified quotes, arrows, dashes, colour emoji as words and expanded contractions."""
    text = unicodedata.normalize("NFKC", text).lower().translate(_QUOTES)
    text = _DASH.sub(" : ", _ARROW.sub(" → ", _BULLET.sub(" ", text)))
    for pattern, replacement in _CONTRACTIONS:
        text = pattern.sub(replacement, text)
    return text.replace("e.g.", "eg").replace("i.e.", "ie").replace("-", " ").replace("\n", " , ")


def _lookup(words: list[str]) -> list[Token]:
    """Tag words from the lexicon: the longest known phrase first, then single words."""
    tokens: list[Token] = []
    position = 0
    while position < len(words):
        size = _phrase_size(words, position)
        phrase = tuple(words[position:position + size])
        tokens.append(Token(" ".join(phrase), *PHRASES[phrase]) if size > 1 else _word(words[position]))
        position += size
    return tokens


def _phrase_size(words: list[str], position: int) -> int:
    """Length of the longest lexicon phrase starting at `position` (1 when there is none)."""
    for size in range(min(MAX_PHRASE, len(words) - position), 1, -1):
        if tuple(words[position:position + size]) in PHRASES:
            return size
    return 1


def _word(word: str) -> Token:
    """Tag one word; an unknown word may be a light typo of the core vocabulary."""
    if word in WORDS:
        return Token(word, *WORDS[word])
    if word[0].isdigit():
        return Token(word, Tag.FILL)
    target = _typo_target(word)
    if target is None:
        return Token(word, Tag.OTHER)
    return Token(word, *WORDS[target], issue=Issue(f"read '{word}' as '{target}'", tuning.TYPO))


def _typo_target(word: str) -> str | None:
    """The core-vocabulary word that `word` misspells, when exactly one reading fits."""
    if len(word) < 4 or word in NOT_TYPOS:
        return None
    hits = [target for target in FUZZY_TARGETS if _is_typo(word, target)]
    return hits[0] if len({WORDS[hit] for hit in hits}) == 1 else None


def _is_typo(word: str, target: str) -> bool:
    """One edit for short words, two for long ones that keep their first letter."""
    limit = 2 if len(target) >= 8 else 1
    if abs(len(word) - len(target)) > limit or (limit == 2 and word[0] != target[0]):
        return False
    return _edit_distance(word, target) <= limit


def _edit_distance(a: str, b: str) -> int:
    """Optimal string alignment distance: insert, delete, substitute, swap neighbours."""
    before: list[int] = []
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            cost = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (char_a != char_b))
            if before and j > 1 and char_a == b[j - 2] and a[i - 2] == char_b:
                cost = min(cost, before[j - 2] + 1)
            current.append(cost)
        before, previous = previous, current
    return previous[-1]


_OUTSIDE = Token("", Tag.BREAK)


def _at(tokens: list[Token], index: int) -> Token:
    """Token at `index`; a sentence break stands in for positions outside the text."""
    return tokens[index] if 0 <= index < len(tokens) else _OUTSIDE


def _skip(tokens: list[Token], index: int, tags: tuple[Tag, ...], step: int = 1) -> int:
    """First index from `index` (walking by `step`) whose token has none of `tags`."""
    while 0 <= index < len(tokens) and tokens[index].tag in tags:
        index += step
    return index


def _clause_ahead(tokens: list[Token], index: int) -> Iterator[Token]:
    """Tokens from `index` up to the next separator, conjunction or new object description."""
    while _at(tokens, index).tag not in (*_SEPARATORS, Tag.AND, Tag.COLOUR, Tag.SHAPE):
        yield tokens[index]
        index += 1


def _retag(token: Token, new_tag: Tag, value: str = "") -> None:
    token.tag, token.value = new_tag, value


def _expand_rows(tokens: list[Token]) -> list[Token]:
    """Rewrite 'left to right', 'from the leftmost bin to the rightmost', 'starting from the right' as the bin row."""
    out: list[Token] = []
    position = 0
    while position < len(tokens):
        side, end = _row_at(tokens, position)
        if side:
            out.append(Token("in row order", Tag.RESPECTIVELY, side))
            out.extend(Token(name, Tag.BIN, name, soft=True) for name in (BINS if side == "left" else BINS[::-1]))
            position = end
        else:
            out.append(tokens[position])
            position += 1
    return out


def _row_at(tokens: list[Token], position: int) -> tuple[str, int]:
    """(side the row starts from, index after the phrase) for a row phrase at `position`, else ('', position)."""
    first = position
    if tokens[position].text in ROW_STARTERS:
        first = position + 1
        while _at(tokens, first).text in ("from", "at", "the", "your"):
            first += 1
    side, far = _at(tokens, first), _row_end(tokens, first)
    stated_alone = first > position + 1 and _is_side(side)
    if far is None and not stated_alone:
        return "", position
    return side.value, _skip(tokens, (first if far is None else far) + 1, (Tag.CONTAINER,))


def _is_side(token: Token) -> bool:
    return token.tag is Tag.BIN and token.value != "center"


def _row_end(tokens: list[Token], first: int) -> int | None:
    """Index of the opposite side closing 'left [bin] to [the] [center to the] right', None when there is none."""
    if not _is_side(_at(tokens, first)):
        return None
    index = _skip(tokens, first + 1, (Tag.CONTAINER,))
    for _ in range(2):
        if _at(tokens, index).text not in ROW_JOINERS:
            return None
        index = _skip(tokens, index + 1, (Tag.FILL,))
        stop = _at(tokens, index)
        if _is_side(stop):
            return index if stop.value != tokens[first].value else None
        if stop.tag is not Tag.BIN:
            return None
        index = _skip(tokens, index + 1, (Tag.CONTAINER,))
    return None


def _resolve_left(tokens: list[Token]) -> list[Token]:
    """Read 'left' as 'remaining' or 'left alone' where it is a participle of 'leave'."""
    for index, token in enumerate(tokens):
        if token.text == "left" and token.tag is Tag.BIN:
            if _means_remaining(tokens, index):
                _retag(token, Tag.REST)
            elif _means_left_alone(tokens, index):
                _retag(token, Tag.LEAVE, "backward")
    return tokens


def _means_remaining(tokens: list[Token], index: int) -> bool:
    """'whatever is left', 'everything left goes ...', 'the only bin left'."""
    via_copula = _at(tokens, index - 1).value == "copula"
    head_index = index - 2 if via_copula else index - 1
    head, before, after = _at(tokens, head_index), _at(tokens, head_index - 1), _at(tokens, index + 1)
    if via_copula:
        return head.text in WH_WORDS or (head.tag is Tag.CONTAINER and before.text in WH_WORDS)
    if head.tag is Tag.CONTAINER:
        return before.text in ("only", "one", "last", "single") and after.text != "of"
    bare_noun = head.tag is Tag.OBJECT and before.tag not in (Tag.COLOUR, Tag.SHAPE)
    quantifier = head.tag in (Tag.ALL, Tag.REST) or head.text == "what"
    return (bare_noun or quantifier) and (after.tag is Tag.LINK or _stays_put(tokens, index + 1))


def _means_left_alone(tokens: list[Token], index: int) -> bool:
    """'should be left', 'are left on the table', 'is left untouched'."""
    auxiliary = _at(tokens, index - 1)
    return auxiliary.text in ("be", "been", "being") or (auxiliary.value == "copula" and _stays_put(tokens, index + 1))


def _stays_put(tokens: list[Token], index: int) -> bool:
    """A complement of 'leave' starts here: 'alone', 'where they are', 'on the table'."""
    word = _at(tokens, index)
    if word.text in STAY_WORDS:
        return True
    return word.text in ("on", "in", "at") and any(_at(tokens, index + k).text in SURFACES for k in (1, 2, 3))


def _resolve_right(tokens: list[Token]) -> list[Token]:
    """Drop 'right' used as an intensifier: 'right away', 'right into the left bin'."""
    for index, token in enumerate(tokens):
        if token.text == "right" and token.tag is Tag.BIN and _is_intensifier(tokens, index):
            _retag(token, Tag.FILL)
    return tokens


def _is_intensifier(tokens: list[Token], index: int) -> bool:
    following = _at(tokens, index + 1)
    if following.text in INTENSIFIED:
        return True
    points_on = following.value == "directional" or following.text in ("on", "at", "next", "beside")
    if not points_on or _at(tokens, index - 1).text in ("the", "your", "my", "its", "far"):
        return False
    return any(token.tag is Tag.BIN for token in _clause_ahead(tokens, index + 2))


def _resolve_between(tokens: list[Token]) -> list[Token]:
    """'between the left and right bins', 'in between' and 'between them' all name the center bin."""
    for index, token in enumerate(tokens):
        if token.text != "between":
            continue
        ends = [t for t in tokens[index + 1:index + 7] if t.tag is Tag.BIN][:2]
        after = _at(tokens, _skip(tokens, index + 1, (Tag.FILL,)))
        bare = not ends and (_at(tokens, index - 1).text == "in" or after.tag in (Tag.PRONOUN, Tag.REST, Tag.OBJECT))
        if {t.value for t in ends} == {"left", "right"} or bare:
            for end in ends:
                _retag(end, Tag.FILL)
            _retag(token, Tag.BIN, "center")
    return tokens


def _resolve_ordinals(tokens: list[Token]) -> list[Token]:
    """Turn 'second bin from the right', 'the first box' and an elliptic 'the third one' into bin positions."""
    declared = _declared_side(tokens)
    sides = [declared] if declared else []
    elliptic: list[int] = []
    for index, token in enumerate(tokens):
        if token.tag is not Tag.ORDINAL:
            continue
        noun_end = _skip(tokens, index + 1, (Tag.CONTAINER, Tag.OBJECT))
        attached = _counting_side(tokens, noun_end)
        if attached or any(t.tag is Tag.CONTAINER for t in tokens[index + 1:noun_end]):
            sides.append(attached)
            _set_ordinal(token, attached or declared)
        else:
            elliptic.append(index)
    for index in elliptic:
        after = _at(tokens, index + 1)
        names_object = after.tag in (Tag.COLOUR, Tag.SHAPE) or (after.tag is Tag.OBJECT and after.text != "one")
        if sides and _at(tokens, index - 1).text == "the" and not names_object:
            _set_ordinal(tokens[index], max(sides))
        else:
            _retag(tokens[index], Tag.FILL)
    return tokens


def _declared_side(tokens: list[Token]) -> str:
    """Side the row is counted from when the text says so: 'numbered from the right', 'left to right'."""
    return next((token.value for token in tokens if token.tag is Tag.RESPECTIVELY and token.value), "")


def _counting_side(tokens: list[Token], index: int) -> str:
    """Side named by 'from the left' / 'counting from your right' at `index` (consumed), else ''."""
    if _at(tokens, index).text not in ("from", "counting"):
        return ""
    end = index
    while _at(tokens, end).text in ("from", "counting", "the", "your", "my"):
        end += 1
    side = _at(tokens, end)
    if side.tag is not Tag.BIN or side.value == "center":
        return ""
    value = side.value
    _retag(side, Tag.FILL)
    return value


def _set_ordinal(token: Token, side: str) -> None:
    """Make an ordinal a bin, counted from `side` (from the left when no side is given)."""
    row = BINS[::-1] if side == "right" else BINS
    token.tag, token.value = Tag.BIN, row[int(token.value) - 1]
    if not side:
        token.issue = Issue(f"'{token.text}' bin counted from the left", tuning.ORDINAL_ASSUMED)


def _resolve_numbered(tokens: list[Token]) -> list[Token]:
    """'bin 2', 'box number three': bins numbered along the row from the left."""
    for index, token in enumerate(tokens):
        if token.tag is not Tag.CONTAINER or _at(tokens, index - 1).tag is Tag.BIN:
            continue
        number_index = index + 1
        while _at(tokens, number_index).text in ("number", "no", "nr", "#"):
            number_index += 1
        number = _at(tokens, number_index)
        if number.text in NUMBERS and _at(tokens, number_index + 1).tag not in (Tag.COLOUR, Tag.SHAPE, Tag.OBJECT):
            for label in tokens[index + 1:number_index]:
                _retag(label, Tag.FILL)
            number.value = str(NUMBERS[number.text])
            _set_ordinal(number, _declared_side(tokens))
    return tokens


def _resolve_relative(tokens: list[Token]) -> list[Token]:
    """'left of the center bin' is left; 'right of the left bin' is center; 'center of the left bin' is left."""
    for index, token in enumerate(tokens):
        if token.tag is not Tag.BIN:
            continue
        of_index = _skip(tokens, index + 1, (Tag.CONTAINER, Tag.OBJECT))
        anchor = _at(tokens, _skip(tokens, of_index + 1, (Tag.FILL,)))
        if _at(tokens, of_index).text != "of" or anchor.tag is not Tag.BIN:
            continue
        if token.value == "center":
            _retag(token, Tag.FILL)
        else:
            token.value = token.value if anchor.value in ("center", token.value) else "center"
            _retag(anchor, Tag.FILL)
    return tokens


def _resolve_can(tokens: list[Token]) -> list[Token]:
    """'the can', 'a red can' name a cylinder unless a verb follows the modal 'can'."""
    for index, token in enumerate(tokens):
        before, after = _at(tokens, index - 1), _at(tokens, index + 1)
        determined = before.tag is Tag.COLOUR or before.text in ("a", "the", "each", "every", "any", "this", "that")
        if token.text == "can" and determined and after.text not in MODAL_FOLLOWERS:
            _retag(token, Tag.SHAPE, "cylinder")
    return tokens


def _resolve_blocks(tokens: list[Token]) -> list[Token]:
    """'blocks' named beside cylinders or triangles, with no cube word around, are the cubes."""
    shapes = {token.value for token in tokens if token.tag is Tag.SHAPE}
    if not shapes or "cube" in shapes:
        return tokens
    for index, token in enumerate(tokens):
        if token.text in ("block", "blocks") and _at(tokens, index - 1).tag not in (Tag.ALL, Tag.REST):
            _retag(token, Tag.SHAPE, "cube")
            token.issue = Issue("read 'blocks' as cubes because other shapes are named", tuning.BLOCKS_AS_CUBES)
    return tokens


def _resolve_but(tokens: list[Token]) -> list[Token]:
    """'all but the red ones': 'but' after a quantifier is an exception, not a conjunction."""
    for index, token in enumerate(tokens):
        head = _at(tokens, _skip(tokens, index - 1, (Tag.OBJECT, Tag.FILL), step=-1))
        if token.text == "but" and head.tag in (Tag.ALL, Tag.REST):
            _retag(token, Tag.EXCEPT)
    return tokens


def _resolve_pronouns(tokens: list[Token]) -> list[Token]:
    """Keep pronouns that refer back; 'those cubes' is a determiner, 'it is red' describes a new object."""
    for index, token in enumerate(tokens):
        if token.tag is not Tag.PRONOUN:
            continue
        determiner = token.text in ("those", "these", "which")
        if determiner and _at(tokens, index + 1).tag in (Tag.COLOUR, Tag.SHAPE, Tag.CONTAINER):
            _retag(token, Tag.FILL)
        elif _describes(tokens, index + 1):
            _retag(token, Tag.OBJECT)
    return tokens


def _describes(tokens: list[Token], index: int) -> bool:
    """'(that) is/are [not] <colour or shape>' starts at `index`."""
    index += _at(tokens, index).text in RELATIVE_PRONOUNS
    if _at(tokens, index).value != "copula":
        return False
    return _at(tokens, _skip(tokens, index + 1, (Tag.FILL, Tag.NOT))).tag in (Tag.COLOUR, Tag.SHAPE)


def _resolve_not(tokens: list[Token]) -> list[Token]:
    """Decide whether a negation word negates a property ('not red') or a clause ('do not put ...')."""
    for index, token in enumerate(tokens):
        if token.tag is not Tag.NOT or token.value == "clause":
            continue
        before = _at(tokens, index - 1)
        if _at(tokens, _skip(tokens, index + 1, (Tag.FILL,))).tag not in (Tag.COLOUR, Tag.SHAPE):
            _retag(token, *((Tag.NOT, "clause") if token.text == "not" else (Tag.FILL, "")))
        elif token.text == "not" and before.tag in (Tag.COMMA, Tag.AND, Tag.CLAUSE):
            _retag(token, Tag.EXCEPT)
        elif token.text == "not" and not _modifies_noun(before):
            token.value = "clause"
        else:
            token.value = "constituent"
    return tokens


def _modifies_noun(before: Token) -> bool:
    """The word before 'not' lets it negate a property: 'that are not red', 'anything not blue'."""
    return before.value == "copula" or before.tag in (Tag.ALL, Tag.REST, Tag.OBJECT, Tag.SHAPE)


def _resolve_unless(tokens: list[Token]) -> list[Token]:
    """'unless it is red' describes the objects that are not red; other uses of 'unless' stay exceptions."""
    for index, token in enumerate(tokens):
        subject_end = _skip(tokens, index + 1, (Tag.FILL, Tag.OBJECT, Tag.ALL))
        if token.text == "unless" and subject_end > index + 1 and _describes(tokens, subject_end):
            _retag(token, Tag.NOT, "constituent")
    return tokens


def _flag_viewpoint(tokens: list[Token]) -> list[Token]:
    """'my left' is the speaker's side and may be mirrored for the robot."""
    for index, token in enumerate(tokens):
        if token.tag is Tag.BIN and _at(tokens, index - 1).text == "my":
            token.issue = Issue(f"'my {token.value}' may be mirrored for the robot", tuning.VIEWPOINT)
    return tokens


_CONTEXT_RULES = (_expand_rows, _resolve_left, _resolve_right, _resolve_between, _resolve_ordinals,
                  _resolve_numbered, _resolve_relative, _resolve_can, _resolve_blocks, _resolve_but,
                  _resolve_pronouns, _resolve_not, _resolve_unless, _flag_viewpoint)
