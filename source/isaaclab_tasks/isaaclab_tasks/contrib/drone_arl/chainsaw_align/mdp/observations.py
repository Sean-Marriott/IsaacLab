# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the chainsaw alignment environments.

Vector quantities are expressed in the vehicle frame (the airframe's yaw-only frame, which is the frame
the velocity action is given in) unless they are about the target, which are in the target's yaw frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from .tool_pose_command import ToolPoseCommand


def tool_goal_error_v(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Current goal position minus tool position, in the vehicle frame [m]. Shape is (num_envs, 3)."""
    term: ToolPoseCommand = env.command_manager.get_term(command_name)
    vehicle_quat_w = math_utils.yaw_quat(term.robot.data.root_link_quat_w.torch)
    return math_utils.quat_apply_inverse(vehicle_quat_w, term.goal_error_w())


def rolled_hang_offset(
    robot: Articulation,
    hang_offset: tuple[float, float, float],
    env_ids: torch.Tensor | slice = slice(None),
    roll_cfg: SceneEntityCfg | None = None,
    roll_hang_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Tool position relative to the root with the payload hanging at rest, allowing for a rolled tool [m].

    Rolling the tool by ``theta`` on the joint in ``roll_cfg`` moves it about the roll axis, which does not pass
    through it, and shifts how the payload hangs. The offset is ``hang_offset`` plus
    ``(sx * (1 - cos(theta)), sy * sin(theta), sz * (1 - cos(theta)))`` for ``roll_hang_shift = (sx, sy, sz)``:
    a lever turning about the roll axis, which fits the measured rest pose of the Mk3 to within ~1 mm over the
    joint's 0-90 deg. The roll is read from the joint, as the vehicle reads it from its encoder.

    Args:
        robot: The robot carrying the tool.
        hang_offset: Tool position relative to the root with the robot level, the tool unrolled and the payload
            hanging at rest, in the yaw-aligned vehicle frame [m].
        env_ids: Environments to compute the offset for. Defaults to all.
        roll_cfg: SceneEntityCfg naming the robot's tool-roll joint, resolved by the manager. None for no roll.
        roll_hang_shift: Shift of the offset with the roll, as above [m].

    Returns:
        The offset in the yaw-aligned vehicle frame. Shape is (num_envs or len(env_ids), 3).
    """
    joint_pos = robot.data.joint_pos.torch[env_ids]
    offset = torch.tensor(hang_offset, device=joint_pos.device).expand(joint_pos.shape[0], 3)
    if roll_cfg is None:
        return offset
    roll = joint_pos[:, roll_cfg.joint_ids[0]]
    lever = 1.0 - torch.cos(roll)
    shift = torch.tensor(roll_hang_shift, device=joint_pos.device)
    return offset + torch.stack((shift[0] * lever, shift[1] * torch.sin(roll), shift[2] * lever), dim=-1)


def airframe_rest_error_v(
    env: ManagerBasedRLEnv,
    command_name: str,
    hang_offset: tuple[float, float, float],
    roll_cfg: SceneEntityCfg | None = None,
    roll_hang_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """Error of the airframe to where it rests with the tool on the current goal, in the vehicle frame [m].

    The rest position is the goal minus the hang offset rotated by the airframe's current yaw: where the
    airframe would put the tool on the goal, at its current heading, with the payload hanging still. It is
    the collocated error for the velocity command, free of the swing that the tool's own error carries.
    Shape is (num_envs, 3).

    Args:
        env: The environment instance.
        command_name: Name of the :class:`ToolPoseCommand` term.
        hang_offset: Tool position relative to the root with the robot level, the tool unrolled and the payload
            hanging at rest, in the yaw-aligned vehicle frame [m].
        roll_cfg: The tool-roll joint, if the tool can be rolled (see :func:`rolled_hang_offset`).
        roll_hang_shift: Shift of the hang offset with the roll [m] (see :func:`rolled_hang_offset`).
    """
    term: ToolPoseCommand = env.command_manager.get_term(command_name)
    vehicle_quat_w = math_utils.yaw_quat(term.robot.data.root_link_quat_w.torch)
    hang_offset_v = rolled_hang_offset(term.robot, hang_offset, roll_cfg=roll_cfg, roll_hang_shift=roll_hang_shift)
    rest_pos_w = term.goal_pos_w - math_utils.quat_apply(vehicle_quat_w, hang_offset_v)
    return math_utils.quat_apply_inverse(vehicle_quat_w, rest_pos_w - term.robot.data.root_link_pos_w.torch)


def tool_offset_t(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Tool position relative to the target, in the target's yaw frame [m]. Shape is (num_envs, 3).

    Places the tool relative to the approach axis directly: behind the target, beside it or ahead of it.
    """
    term: ToolPoseCommand = env.command_manager.get_term(command_name)
    return term.tool_offset_t()


def tool_staged(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """1 once the tool has been staged and the goal is the target, else 0. Shape is (num_envs, 1)."""
    term: ToolPoseCommand = env.command_manager.get_term(command_name)
    return term.staged.float().unsqueeze(-1)


def tool_yaw_error_sin_cos(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Tool yaw error to the target as (sin, cos), continuous across +/-pi. Shape is (num_envs, 2)."""
    term: ToolPoseCommand = env.command_manager.get_term(command_name)
    yaw_error = term.yaw_error()
    return torch.stack((torch.sin(yaw_error), torch.cos(yaw_error)), dim=-1)


def root_lin_vel_v(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Airframe linear velocity in the vehicle frame [m/s]. Shape is (num_envs, 3)."""
    asset: Articulation = env.scene[asset_cfg.name]
    vehicle_quat_w = math_utils.yaw_quat(asset.data.root_link_quat_w.torch)
    return math_utils.quat_apply_inverse(vehicle_quat_w, asset.data.root_link_lin_vel_w.torch)


def root_roll_pitch(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Airframe roll and pitch, wrapped to [-pi, pi] [rad]. Shape is (num_envs, 2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_link_quat_w.torch)
    return torch.stack((math_utils.wrap_to_pi(roll), math_utils.wrap_to_pi(pitch)), dim=-1)
