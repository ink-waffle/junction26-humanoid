"""Every tunable of the rule parser: search limits, costs that rank readings, confidence factors per kind of guess."""
from __future__ import annotations

# Search over segmentations of the mention sequence.
BEAM_WIDTH = 32                  # partial readings kept per position ...
BEAM_FULL_UNTIL = 12             # ... for up to this many mentions; longer texts get a proportionally narrower beam
MAX_CANDIDATES = 6               # complete readings interpreted per search pass
MAX_UNIT_SPAN = 12
ASIDE_RETRY_BELOW = 0.9          # below this confidence, text in parentheses is dropped and the parse retried

# Costs of a reading (lower is better). A mention left unpaired costs more than any awkward pairing.
UNPAIRED_OBJECT = 10.0           # a colour or shape that never gets a bin
UNPAIRED_BIN = 6.0               # a bin that never gets objects
REDUNDANT = 1.0                  # unpaired mention whose colour or bin is settled elsewhere ("sort red and blue: ...")
DESCRIPTIVE = 3.0                # unpaired mention in a sentence that names no bin (or no objects) at all
SKIPPED_PAIR = 3.0               # an explicit "red in the left bin" left unused in favour of another reading
CONFLICT = 4.0                   # one object key sent to two different bins
RUN_BROKEN = 6.0                 # arrow or "but" inside a list (a sentence break inside a list is never accepted)
RUN_DIRECTED = 12.0              # "to" / "into" inside a list: the first item had a destination of its own
RUN_MIXED = 3.0                  # other verb or preposition inside a list
RUN_UNKNOWN = 2.0                # unknown word inside a list
ACROSS_BREAK = 5.0               # objects and their bin in different sentences
ACROSS_CLAUSE = 4.0              # objects and their bin on two sides of "but" / "while"
ACROSS_AND = 3.0                 # only "and" between objects and their bin
ACROSS_COMMA = 1.5               # only a comma between objects and their bin
LINKED_AFTER_COMMA = 0.5         # comma plus verb or preposition between objects and their bin
UNSEPARATED = 1.0                # two clauses with nothing between them
LINK_BETWEEN_CLAUSES = 1.5       # a verb or preposition left between two clauses
DISTRIBUTION = 1.0               # parallel lists without "respectively"
WHOLE_SET_ONE_BIN = 2.0          # every colour listed for one bin, not as a plain "A, B and C go in X"
ORDER_IGNORED = 2.0              # "respectively" / "in order" in the sentence, yet a whole list goes to one bin
BIN_FIRST_DIRECTIONAL = 1.0      # "left bin into red": preposition points the other way
DEPOSIT_AFTER_LEAVE = 0.5        # "leave red in the left bin" read as a placement
QUANTIFIER_IN_LIST = 1.0         # "everything" listed beside specific groups
LONE_EXCEPTION = 1.0             # "except blue: right" read as blue -> right
ATTACHED = 1.5                   # "blue too" joins the previous bin
AMENDED = 0.5                    # "sorry, I meant the right bin" replaces the bin of the previous clause
COMPANION = 1.0                  # "blue goes with yellow": blue takes yellow's bin
UNDECIDED = 4.0                  # "red in the left or the right bin": several bins for one group, none chosen
NESTED = 2.0                     # "what is left after red went left goes right": one clause inside another
AMBIGUITY_MARGIN = 1.0           # a different mapping this close in cost makes the parse ambiguous

# Cue weights that decide between a colour rule and a shape rule claiming the same object type.
W_UNIVERSAL = 2                  # "regardless of colour", "whatever its shape"
W_QUANTIFIED = 1                 # "all cubes", "any triangle"
W_RESIDUAL = 2                   # subtracted: "the other red ones", "of the remaining objects, ..."
W_DECISIVE = 2                   # a lead this large settles the overlap without doubt

# Confidence factors (multiplied; 1.0 = no doubt).
TYPO = 0.9
ORDINAL_ASSUMED = 0.85           # "second bin" with no counting direction
LETTER_BIN = 0.85                # undefined one-letter bin names: "red -> L, blue -> R"
VIEWPOINT = 0.4                  # "my left" may be mirrored
BLOCKS_AS_CUBES = 0.85
GENERIC_CUBE = 0.8               # "cubes" used for every group: read as "objects"
SHARED_COLOUR = 0.8              # "red cubes and cylinders": cylinders read as red
UNKNOWN_LINK = 0.95              # unknown word between objects and their bin
F_ACROSS_BREAK = 0.7
F_ACROSS_CLAUSE = 0.7
F_ACROSS_AND = 0.8
F_ACROSS_COMMA = 0.95
F_RUN_BROKEN = 0.6
F_RUN_DIRECTED = 0.5
F_RUN_MIXED = 0.8
F_COMPANION = 0.9
F_CORRECTED = 0.9                # a self-correction was applied
F_CORRECTION_LOST = 0.45         # a correction marker was seen but nothing could be corrected
F_RUN_UNKNOWN = 0.9
F_NESTED = 0.8
F_DESCRIPTIVE = 0.9
F_DISTRIBUTION = 0.9
F_ORDER_IGNORED = 0.45
F_WHOLE_SET_ONE_BIN = 0.45
F_BIN_FIRST_DIRECTIONAL = 0.9
F_LONE_EXCEPTION = 0.9
F_ATTACHED = 0.85
F_UNPAIRED_OBJECT = 0.6
F_UNPAIRED_BIN = 0.8
F_CONFLICT = 0.4
F_UNDETERMINED = 0.7             # a group is mentioned but its bin cannot be worked out
F_ELIMINATION = 0.9              # bin inferred from two excluded bins
F_CORRECT_BIN = 0.8              # "the right bin" read as "the correct bin" while some colours stay unassigned
F_OVERLAP = 0.7                  # a colour rule and a shape rule disagree about one object type, no cue decides
F_OVERLAP_CUED = 0.85            # ... and only a weak cue ("all cubes") decides
F_AMBIGUOUS = 0.75
