"""Registers the Aalto-* gym ids on import (imported by aaltonomous/sim/aalto_sim_main.py after
upstream `tasks`). Run: TASK=Aalto-Sort-G129-Dex3-Joint bash aaltonomous/sim/run_sim.sh"""
import gymnasium as gym

from . import env_cfg

for _id, _cfg in (("Aalto-Sort-G129-Dex3-Joint", env_cfg.AaltoSortEnvCfg),
                  ("Aalto-Insert-G129-Dex3-Joint", env_cfg.AaltoInsertEnvCfg)):
    gym.register(id=_id, entry_point="isaaclab.envs:ManagerBasedRLEnv",
                 kwargs={"env_cfg_entry_point": _cfg}, disable_env_checker=True)
