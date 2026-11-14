"""Background jobs: slow work (perception, planning) runs in a low-priority thread while `act()` keeps answering."""
from __future__ import annotations

import os
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

Cancelled = Callable[[], bool]


@dataclass
class Job:
    work: Callable[[Cancelled], Any]
    cancelled: bool = False
    finished: threading.Event = field(default_factory=threading.Event)
    value: Any = None
    error: BaseException | None = None


class Background:
    """One job at a time is of interest; a cancelled job runs out unobserved and later jobs queue behind it."""

    def __init__(self, nice: int) -> None:
        self.nice = nice
        self._serial = threading.Lock()   # jobs share planner and kinematics state
        self._job: Job | None = None

    @property
    def idle(self) -> bool:
        """No job thread is running, not even an abandoned one: the planner state is free for the caller."""
        return not self._serial.locked()

    def submit(self, work: Callable[[Cancelled], Any]) -> None:
        self._job = Job(work)
        threading.Thread(target=self._run, args=(self._job,), daemon=True).start()

    def poll(self, wait_s: float = 0.0) -> Job | None:
        """The submitted job once it has finished (exactly once), else None; waits up to `wait_s` for it."""
        job = self._job
        if job is None or not job.finished.wait(wait_s):
            return None
        self._job = None
        return job

    def cancel(self) -> None:
        if self._job is not None:
            self._job.cancelled = True
            self._job = None

    def _run(self, job: Job) -> None:
        try:
            self._lower_priority()
            with self._serial:
                if not job.cancelled:
                    job.value = job.work(lambda: job.cancelled)
        except Exception as error:
            job.error = error
        finally:
            job.finished.set()

    def _lower_priority(self) -> None:
        """Nice the calling thread (a per-thread setting on Linux only) so planning never starves `act()`."""
        if sys.platform == "linux":
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), self.nice)


def within(seconds: float, work: Callable[[], Any], nice: int) -> Any:
    """What `work` returns, run in a thread of its own; TimeoutError if it takes longer (that thread is left to run out)."""
    runner = Background(nice)
    runner.submit(lambda cancelled: work())
    job = runner.poll(seconds)
    if job is None:
        raise TimeoutError(f"no answer within {seconds:.1f} s")
    if job.error is not None:
        raise job.error
    return job.value
