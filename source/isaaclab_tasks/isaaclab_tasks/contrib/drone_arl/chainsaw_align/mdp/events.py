# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for the chainsaw alignment environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

    from .tool_pose_command import ToolPoseCommand


def lock_passive_joints(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg, stiffness: float
) -> None:
    """Hold unactuated joints at zero angle with a stiff drive.

    The drive target of a passive joint is zero, so a large drive stiffness locks it there. The drive is
    written directly because a :class:`~isaaclab_contrib.assets.Multirotor` cannot have joint actuators.
    It goes through the CPU tensor API, so the term is meant for ``mode="startup"``.

    Args:
        env: The manager-based RL environment instance.
        env_ids: Indices of the environments to write. All environments if None.
        asset_cfg: SceneEntityCfg identifying the asset and the joints to lock.
        stiffness: Drive stiffness [N·m/rad].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)
    joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int32, device=asset.device)
    stiffness_values = torch.full((len(env_ids), len(joint_ids)), stiffness, device=asset.device)
    asset.write_joint_stiffness_to_sim_index(stiffness=stiffness_values, joint_ids=joint_ids, env_ids=env_ids)


class RandomWind(ManagerTermBase):
    """Apply a random wind gust to a body as a constant force and torque, resampled at random intervals.

    Each environment holds its gust for a random interval, then draws a new one, which models sustained wind
    rather than an impulse. Every gust component is drawn uniformly from its range, so zero ranges mean no
    wind. The ranges are read on every call, so a curriculum can grow them during training.

    Run it every policy step (``mode="interval"`` with ``interval_range_s`` equal to the policy step and
    ``is_global_time=True``) so the hold intervals are timed per step. The wrench goes into the asset's
    permanent wrench buffer and stays applied between calls.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        device = env.device
        self._force = torch.zeros(env.num_envs, 3, device=device)
        self._torque = torch.zeros(env.num_envs, 3, device=device)
        self._time_left = torch.zeros(env.num_envs, device=device)
        self._all_env_ids = torch.arange(env.num_envs, device=device, dtype=torch.int32)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Draw a new gust for the reset environments on the next call."""
        self._time_left[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        force_range: tuple[float, float],
        torque_range: tuple[float, float],
        interval_range_s: tuple[float, float] = (2.0, 6.0),
    ) -> None:
        """Resample the gusts that have run out and write the wind wrench for every environment.

        Args:
            env: The manager-based RL environment instance.
            env_ids: Unused. The term always writes every environment.
            asset_cfg: SceneEntityCfg identifying the asset and the body the wind acts on.
            force_range: Range of each gust force component, in the world frame [N].
            torque_range: Range of each gust torque component, in the world frame [N·m].
            interval_range_s: Range of the time each gust is held for [s].
        """
        asset: Articulation = env.scene[asset_cfg.name]
        self._time_left -= env.step_dt
        expired = (self._time_left <= 0.0).nonzero().flatten()
        if len(expired) > 0:
            n = len(expired)
            self._force[expired] = math_utils.sample_uniform(*force_range, (n, 3), env.device)
            self._torque[expired] = math_utils.sample_uniform(*torque_range, (n, 3), env.device)
            self._time_left[expired] = math_utils.sample_uniform(*interval_range_s, (n,), env.device)
        asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=self._force.unsqueeze(1),
            torques=self._torque.unsqueeze(1),
            body_ids=asset_cfg.body_ids,
            env_ids=self._all_env_ids,
            is_global=True,
        )


def reset_tool_around_target(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    hang_offset: tuple[float, float, float],
    distance_range: tuple[float, float],
    vertical_range: tuple[float, float],
    max_azimuth: float,
    yaw_range: tuple[float, float],
    rp_range: tuple[float, float] = (0.0, 0.0),
    lin_vel_range: tuple[float, float] = (0.0, 0.0),
    ang_vel_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the robot so that its tool starts at a sampled bearing and distance from the target.

    The tool is placed ``distance`` from the target horizontally, at an azimuth within ``max_azimuth`` of the
    approach axis (0 is straight behind the target on the axis, pi is anywhere around it), and ``vertical``
    above or below it. The robot faces the target yaw turned by ``yaw``, and is placed so that the tool is
    there with the payload hanging at rest. The approach axis is taken to be horizontal.

    The target is read from the command term, whose target pose is fixed, so it is valid here even though
    the command is reset after the events.

    Args:
        env: The manager-based RL environment instance.
        env_ids: Indices of the environments to reset.
        command_name: Name of the :class:`ToolPoseCommand` term.
        hang_offset: Tool position relative to the root with the robot level and the payload hanging at
            rest, in the yaw-aligned vehicle frame [m].
        distance_range: Range of the horizontal distance of the tool from the target [m].
        vertical_range: Range of the height of the tool relative to the target [m].
        max_azimuth: Largest angle of the tool's bearing from the approach axis, either side [rad].
        yaw_range: Range of the robot yaw relative to the target yaw [rad].
        rp_range: Range of the initial roll and pitch [rad].
        lin_vel_range: Range of the initial linear velocity on each axis [m/s].
        ang_vel_range: Range of the initial angular velocity about each axis [rad/s].
        asset_cfg: SceneEntityCfg identifying the asset to reset.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    term: ToolPoseCommand = env.command_manager.get_term(command_name)
    # A full reset passes a slice, which cannot be counted or used to gather.
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=asset.device)[env_ids if env_ids is not None else slice(None)]
    n = len(env_ids)

    def _sample(value_range: tuple[float, float], size: tuple[int, ...]) -> torch.Tensor:
        return math_utils.sample_uniform(value_range[0], value_range[1], size, device=asset.device)

    # -- tool position around the target
    axis_w = term.approach_axis_w[env_ids]
    azimuth = torch.atan2(axis_w[:, 1], axis_w[:, 0]) + _sample((-max_azimuth, max_azimuth), (n,))
    distance = _sample(distance_range, (n,))
    tool_pos_w = term.target_pos_w[env_ids].clone()
    tool_pos_w[:, 0] += distance * torch.cos(azimuth)
    tool_pos_w[:, 1] += distance * torch.sin(azimuth)
    tool_pos_w[:, 2] += _sample(vertical_range, (n,))

    # -- robot placed so the tool is there with the payload at rest
    zeros = torch.zeros(n, device=asset.device)
    yaw = term.target_yaw[env_ids] + _sample(yaw_range, (n,))
    hang_offset_w = math_utils.quat_apply(
        math_utils.quat_from_euler_xyz(zeros, zeros, yaw),
        torch.tensor(hang_offset, device=asset.device).expand(n, 3),
    )
    positions = tool_pos_w - hang_offset_w
    orientations = math_utils.quat_from_euler_xyz(_sample(rp_range, (n,)), _sample(rp_range, (n,)), yaw)
    velocities = torch.cat((_sample(lin_vel_range, (n, 3)), _sample(ang_vel_range, (n, 3))), dim=-1)

    asset.write_root_pose_to_sim_index(root_pose=torch.cat((positions, orientations), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)
