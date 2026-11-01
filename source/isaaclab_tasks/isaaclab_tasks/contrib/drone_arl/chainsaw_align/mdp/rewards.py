# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the chainsaw alignment environments. Each is a Gaussian kernel bounded to [0, 1]."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .tool_pose_command import ToolPoseCommand


def tool_route_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Kernel on the remaining route length to the target, ``exp(-D^2 / std^2)``. Shape is (num_envs,).

    The route runs through the staging point until the tool has been staged (see
    :meth:`ToolPoseCommand.route_distance`), so the kernel reaches 1 only on the target, and only after
    arriving along the approach axis.
    """
    term: ToolPoseCommand = env.command_manager.get_term(command_name)
    return torch.exp(-torch.square(term.route_distance()) / std**2)


def tool_cross_track_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Kernel on the tool's distance from the approach axis, paid only once staged. Shape is (num_envs,)."""
    term: ToolPoseCommand = env.command_manager.get_term(command_name)
    return torch.exp(-torch.square(term.cross_track_error()) / std**2) * term.staged.float()


def tool_still_exp(env: ManagerBasedRLEnv, command_name: str, std_speed: float, std_distance: float) -> torch.Tensor:
    """Kernel on the tool's speed, gated by a kernel on its distance to the target. Shape is (num_envs,).

    ``exp(-|v|^2 / std_speed^2) * exp(-d^2 / std_distance^2)``: it pays for holding the tool still only on the
    target, so it cannot be earned by hovering motionless anywhere else. Near the target the position kernels
    are flat, so a small payload swing costs them almost nothing; this term charges it at full rate.
    """
    term: ToolPoseCommand = env.command_manager.get_term(command_name)
    speed_sq = torch.sum(torch.square(term.tool_lin_vel_w), dim=-1)
    distance_sq = torch.sum(torch.square(term.target_error_w()), dim=-1)
    return torch.exp(-speed_sq / std_speed**2 - distance_sq / std_distance**2)


def tool_yaw_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    """Kernel on the tool's yaw error to the target, ``exp(-e^2 / std^2)``. Shape is (num_envs,)."""
    term: ToolPoseCommand = env.command_manager.get_term(command_name)
    return torch.exp(-torch.square(term.yaw_error()) / std**2)
