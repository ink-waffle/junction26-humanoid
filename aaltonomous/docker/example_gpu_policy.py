"""GPU submission example: load the model on CUDA at startup (before serve(), so the 120 s startup limit covers it),
then hold the arms. Exits with an error if no GPU is visible (e.g. `docker run` without --gpus all)."""
import logging
import sys

import numpy as np
import torch

from aaltonomous.deploy.policy_server import HoldPolicy, serve


class GPUHoldPolicy(HoldPolicy):
    def __init__(self):
        dev = torch.device("cuda")
        self.net = torch.nn.Linear(28, 28).to(dev)  # stand-in for your model + weights
        with torch.no_grad():                       # warm-up: first CUDA call (kernel load) happens before serving
            self.net(torch.zeros(1, 28, device=dev))
        torch.cuda.synchronize()
        self.name = f"gpu-hold ({torch.cuda.get_device_name(0)}, torch {torch.__version__})"
        self.action_horizon = 1

    def act(self, obs):
        q = torch.as_tensor(np.concatenate([obs["arm_q"], obs["hand_q"]]), device="cuda").float()
        with torch.no_grad():
            self.net(q[None])                       # real inference would produce the targets
        return super().act(obs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not torch.cuda.is_available():
        sys.exit("no CUDA device visible: run with --gpus all (needs nvidia-container-toolkit on the host)")
    cap = torch.cuda.get_device_capability(0)
    logging.info("CUDA OK: %s, capability %s, torch %s (CUDA %s), %.1f GB", torch.cuda.get_device_name(0), cap,
                 torch.__version__, torch.version.cuda, torch.cuda.get_device_properties(0).total_memory / 1e9)
    serve(GPUHoldPolicy(), port=8000)
