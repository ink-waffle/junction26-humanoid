"""Aalto-Sort / Aalto-Insert env configs: upstream Dex3 pick-place MDP (same DDS observation terms, same joint
position action) on our scene. No success termination and no time-out: episodes are run by AaltoEnv / the eval,
which read the outcome from the scene service (hooks.py)."""
from __future__ import annotations

import torch

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from tasks.common_config.env_configs import UnitreeManagerBasedRLEnvCfg
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_observations.camera_state import get_camera_image
from tasks.common_observations.dex3_state import get_robot_dex3_joint_states
from tasks.common_observations.g1_29dof_state import get_robot_boy_joint_states

from . import hooks
from .scene_cfg import AaltoLidSceneCfg, AaltoSceneCfg


@configclass
class ActionsCfg:
    # ABSOLUTE targets (the DDS commands are absolute); uncommanded joints get the default pose from the
    # action-provider wrapper in aalto_sim_main.py. Upstream tasks use use_default_offset=True (fine only at 0 defaults).
    joint_pos = base_mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0,
                                                use_default_offset=False)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=get_robot_boy_joint_states)     # -> rt/lowstate
        robot_gipper_state = ObsTerm(func=get_robot_dex3_joint_states)  # -> rt/dex3/*/state
        camera_image = ObsTerm(func=get_camera_image)                    # -> teleimager shm (head mono + wrists)
        aalto = ObsTerm(func=hooks.aalto_step_obs)                       # scene service + head RGB-D

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    pass


@configclass
class RewardsCfg:
    pass


@configclass
class EventCfg:
    reset_layout = EventTermCfg(func=hooks.reset_random_layout, mode="reset", params={"seed": None})


@configclass
class AaltoSortEnvCfg(UnitreeManagerBasedRLEnvCfg):
    scene: AaltoSceneCfg = AaltoSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    commands = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 1.0e6       # never auto-reset; the scene service resets
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics.bounce_threshold_velocity = 0.01
        self.sim.physics.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physics.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physics.friction_correlation_distance = 0.003
        self.sim.physics.enable_ccd = True
        self.sim.physics.min_position_iteration_count = 16
        self.sim.physics.min_velocity_iteration_count = 4

        # sim_main triggers these on rt/reset_pose/cmd (category 1 = objects, 2 = everything)
        self.event_manager = SimpleEventManager()
        self.event_manager.register("reset_object_self", SimpleEvent(func=lambda env: hooks.reset_random_layout(env, None)))
        self.event_manager.register("reset_all_self", SimpleEvent(func=lambda env: (
            base_mdp.reset_scene_to_default(env, torch.arange(env.num_envs, device=str(env.device))),
            hooks.reset_random_layout(env, None))))


@configclass
class AaltoInsertEnvCfg(AaltoSortEnvCfg):
    scene: AaltoLidSceneCfg = AaltoLidSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
