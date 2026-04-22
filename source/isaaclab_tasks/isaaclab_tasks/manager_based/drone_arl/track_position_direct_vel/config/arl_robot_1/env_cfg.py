# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_assets.robots.arl_robot_1 import ARL_ROBOT_1_CFG

from .track_position_direct_vel_env_cfg import TrackPositionDirectVelEnvCfg


@configclass
class DirectVelEnvCfg(TrackPositionDirectVelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ARL_ROBOT_1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["thrusters"].dt = self.sim.dt


@configclass
class DirectVelEnvCfg_PLAY(DirectVelEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
