"""Policy server: the sorter behind the organisers' websocket protocol, on port 8000 or $PORT.

    python -m halal.server        (AALTO_TIME_LIMIT = episode length in seconds, default from config.yaml)
"""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Mapping

from aaltonomous.deploy.policy_server import Policy, serve
from aaltonomous.deploy.protocol import PORT

from halal import config
from halal.language import parse_instruction
from halal.policy import SortPolicy, leave_everything, make_policy

log = logging.getLogger(__name__)


class SortServer(Policy):
    name = "halal"

    def __init__(self, sorter: SortPolicy) -> None:
        self.sorter = sorter
        self.settings = config.load()["policy"]
        self.action_horizon = self.settings["chunk_ticks"]

    def reset(self, instruction: str, task: str | None = None, seed: int | None = None) -> None:
        """Parse the instruction; a mapping we are not confident about means nothing gets moved."""
        parsed = parse_instruction(instruction)
        confident = parsed.confidence >= self.settings["min_instruction_confidence"]
        self.sorter.reset(parsed.bin_for if confident else leave_everything)
        log.info("reset task=%s seed=%s instruction=%r mapping=%s confidence=%.2f parser=%s", task, seed, instruction,
                 parsed.mapping, parsed.confidence, parsed.parser)

    def act(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        return self.sorter.act(obs)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    sys.setswitchinterval(config.load()["policy"]["gil_switch_s"])
    started = time.monotonic()
    limit = os.environ.get("AALTO_TIME_LIMIT")
    server = SortServer(make_policy(float(limit) if limit else None))
    log.info("policy ready in %.1f s", time.monotonic() - started)
    serve(server, port=int(os.environ.get("PORT", PORT)))


if __name__ == "__main__":
    main()
