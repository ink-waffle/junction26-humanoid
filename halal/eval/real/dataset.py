"""Loader for a LeRobot v3.0 real-robot drop: joint arrays per episode and decoded camera frames."""
from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = Path(os.environ.get("HALAL_REAL_DROP", REPO / "runs/drops/g1_pick_cube"))
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
FPS = 30.0
N_ARM = 14


@dataclass(frozen=True)
class Drop:
    """All frames of a drop as flat arrays plus the first global index of every episode."""
    root: Path
    state: np.ndarray
    action: np.ndarray
    timestamp: np.ndarray
    episode: np.ndarray
    starts: np.ndarray

    @property
    def n_episodes(self) -> int:
        return len(self.starts) - 1

    def span(self, ep: int) -> slice:
        """Global frame slice of one episode."""
        return slice(int(self.starts[ep]), int(self.starts[ep + 1]))

    def arm_q(self, ep: int) -> np.ndarray:
        return self.state[self.span(ep), :N_ARM]

    def hand_q(self, ep: int) -> np.ndarray:
        return self.state[self.span(ep), N_ARM:]

    def arm_cmd(self, ep: int) -> np.ndarray:
        return self.action[self.span(ep), :N_ARM]

    def hand_cmd(self, ep: int) -> np.ndarray:
        return self.action[self.span(ep), N_ARM:]


def _column(table: pa.Table, name: str) -> np.ndarray:
    return np.asarray(table.column(name).to_pylist())


def load(root: Path = DEFAULT_ROOT) -> Drop:
    """Read every data parquet of the drop into one Drop."""
    files = sorted(Path(root).glob("data/chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no data parquet under {root} (set HALAL_REAL_DROP)")
    table = pq.read_table(files)
    episode = _column(table, "episode_index")
    starts = np.r_[np.flatnonzero(np.r_[True, np.diff(episode) != 0]), len(episode)]
    return Drop(Path(root), _column(table, "observation.state").astype(np.float64),
                _column(table, "action").astype(np.float64), _column(table, "timestamp").astype(np.float64),
                episode, starts)


def video_path(root: Path, camera: str) -> Path:
    return Path(root) / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"


def iter_frames(root: Path, camera: str, indices: Iterable[int] | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (global index, RGB frame) in order; `indices` restricts the output (the file is still decoded in sequence)."""
    import av

    wanted = None if indices is None else set(int(i) for i in indices)
    last = None if wanted is None else max(wanted, default=-1)
    with av.open(str(video_path(root, camera))) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            if wanted is None or i in wanted:
                yield i, frame.to_ndarray(format="rgb24")
            if last is not None and i >= last:
                return


def frames(root: Path, camera: str, indices: Iterable[int]) -> dict[int, np.ndarray]:
    """Decoded RGB frames for the given global indices."""
    return dict(iter_frames(root, camera, indices))
