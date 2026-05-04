# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

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

        # M350: Ixx=Iyy=0.7 kg·m², Izz=0.9 kg·m² vs ARL ~0.02 kg·m².
        # K_rot = I·ωn², K_angvel = 2ζ·I·ωn
        # roll/pitch: ωn=4 rad/s, ζ=0.8 → K_rot≈11.2, K_angvel≈4.5
        # yaw:        ωn=2 rad/s, ζ=0.8 → K_rot≈3.6,  K_angvel≈2.9
        ctrl = self.actions.velocity_command.controller_cfg
        ctrl.K_rot_range = ((10.0, 10.0, 3.0), (13.0, 13.0, 4.5))
        ctrl.K_angvel_range = ((4.0, 4.0, 2.5), (5.0, 5.0, 3.5))


@configclass
class MatriceDirectVelEnvCfg_PLAY(MatriceDirectVelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
