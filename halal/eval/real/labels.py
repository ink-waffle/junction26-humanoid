"""Hand labels for drop 1 (`g1_pick_cube`), read off head and left-wrist frames; episode-local frame ranges, inclusive."""
from __future__ import annotations

import numpy as np

# Left hand holds the object clear of the counter (purple cube; purple cylinder in 11, 14, 17).
HOLD_AIR: dict[int, tuple[int, int]] = {
    0: (480, 560), 2: (290, 345), 3: (225, 315), 4: (350, 450), 5: (262, 424), 6: (272, 322), 7: (150, 218),
    8: (195, 278), 9: (318, 385), 11: (490, 565), 14: (355, 425), 17: (245, 330),
}
HELD_SHAPE: dict[int, str] = {11: "cylinder", 14: "cylinder", 17: "cylinder"}
# Left fingers commanded closed in the air with nothing between them.
CLOSED_EMPTY: dict[int, tuple[int, int]] = {0: (790, 835), 1: (362, 392), 13: (885, 950)}
# Left hand closed on the object while it still rests on the counter (before the lift, or a grasp that never lifts).
HOLD_COUNTER: dict[int, tuple[int, int]] = {0: (450, 470), 1: (275, 330), 4: (315, 340), 9: (285, 305)}
# Centre bin in the first head frame of one episode per view group, read by eye (+-3 px): rim corners BL, BR, FR, FL.
CENTRE_BIN_RIM_PX: dict[int, tuple[tuple[int, int], ...]] = {
    0: ((283, 33), (478, 25), (515, 172), (280, 174)),
    11: ((256, 62), (434, 60), (457, 178), (244, 182)),
}
# Same frames: bottom corners of the near wall, left then right.
CENTRE_BIN_FOOT_PX: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {
    0: ((285, 229), (498, 224)), 11: ((244, 236), (452, 236)), 17: ((256, 156), (460, 158)),
}


def mask(windows: dict[int, tuple[int, int]], episode: int, length: int, pad: int = 0) -> np.ndarray:
    """Boolean array over one episode, true inside its labelled window grown by `pad` frames."""
    out = np.zeros(length, bool)
    if episode in windows:
        first, last = windows[episode]
        out[max(first - pad, 0):last + pad + 1] = True
    return out
