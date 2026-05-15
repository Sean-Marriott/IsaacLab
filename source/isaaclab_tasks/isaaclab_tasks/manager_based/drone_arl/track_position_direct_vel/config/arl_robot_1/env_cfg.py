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

        # Gains derived from actual PhysX inertia (scripts/demos/matrice_vel_demo.py):
        #   mass=8.665 kg, I_xx=2.640, I_yy=2.649, I_zz=1.065 kg·m²  (includes chainsaw payload)
        #   hover_thrust/motor=21.25 N, arm_pitch_sum=1.269 m, arm_roll_sum=1.434 m
        #   tau_max=42.81 N·m  →  nominal K_rot_xy=32.33, K_angvel_xy=15.73, K_vel_xy=3.49
        #
        # Ranges use ±10 % around nominal (ζ = 0.85 at nominal; corners: 0.73–0.99):
        #   worst case (K_rot=35.6, K_angvel=14.2, I_att=2.649): ζ = 0.73
        #   best  case (K_rot=29.1, K_angvel=17.3, I_att=2.649): ζ = 0.99
        #
        # Design equations for re-tuning after a model change:
        #   K_rot    = min(tau_max / (0.5 × I_att), 200)      tau_max = Δ_thrust × min(sum|pitch_arms|, sum|roll_arms|)
        #   K_angvel = 2 × 0.85 × sqrt(K_rot / I_att) × I_att (ζ = 0.85)
        #   K_vel    = sqrt(K_rot / I_att)                     (1:1 cascade bandwidth ratio)
        #   Yaw: K_rot_z = 0.4 × K_rot_xy, K_angvel_z from same ζ formula with I_zz
        ctrl = self.actions.velocity_command.controller_cfg
        ctrl.K_rot_range    = ((29.1, 29.1, 11.6), (35.6, 35.6, 14.2))
        ctrl.K_angvel_range = ((14.2, 14.2,  5.7), (17.3, 17.3,  6.9))
        ctrl.K_vel_range    = (( 3.1,  3.1,  1.9), ( 3.8,  3.8,  2.3))

@configclass
class MatriceDirectVelEnvCfg_PLAY(MatriceDirectVelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.push_pole = None
        self.events.push_robot = None
