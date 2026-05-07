# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG, MATRICE_CFG

from .track_position_direct_vel_env_cfg import TrackPositionDirectVelEnvCfg


@configclass
class ArlDirectVelEnvCfg(TrackPositionDirectVelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ARL_ROBOT_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["thrusters"].dt = self.sim.dt


@configclass
class ArlDirectVelEnvCfg_PLAY(ArlDirectVelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class MatriceDirectVelEnvCfg(TrackPositionDirectVelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = MATRICE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["thrusters"].dt = self.sim.dt

        # Gains derived from actual PhysX inertia (scripts/demos/arl_robot_1.py):
        #   mass=6.500 kg, I_xx=0.775, I_yy=0.756, I_zz=1.029 kg·m²
        #   tau_max=49.6 N·m  →  nominal K_rot_xy=80, K_angvel_xy=18.9, K_vel_xy=3.38
        #
        # Ranges use ±10 % to ensure ζ > 1.03 at every randomisation corner:
        #   worst case (K_rot=88, K_angvel=17.0, I=0.776): ζ = 1.03
        #   best  case (K_rot=72, K_angvel=20.8, I=0.776): ζ = 1.39
        #
        # Design equations for re-tuning after a model change:
        #   K_rot    = min(tau_max / (0.5 × I_att), 80)       tau_max = Δ_thrust × min(sum|pitch_arms|, sum|roll_arms|)
        #   K_angvel = 2 × 1.2 × sqrt(K_rot / I_att) × I_att
        #   K_vel    = sqrt(K_rot / I_att) / 3                 (3× cascade bandwidth margin)
        #   Yaw: K_rot_z = 0.4 × K_rot_xy, K_angvel_z from same ζ formula with I_zz
        ctrl = self.actions.velocity_command.controller_cfg
        ctrl.K_rot_range    = ((72.0, 72.0, 28.8), (88.0, 88.0, 35.2))
        ctrl.K_angvel_range = ((17.0, 17.0, 12.4), (20.8, 20.8, 15.1))
        ctrl.K_vel_range    = (( 3.0,  3.0,  4.6), ( 3.7,  3.7,  5.6))

@configclass
class MatriceDirectVelEnvCfg_PLAY(MatriceDirectVelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
