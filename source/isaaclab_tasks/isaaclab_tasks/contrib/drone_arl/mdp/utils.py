# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers shared by the drone ARL MDP terms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject

    from .commands import DroneTrajectoryCommand


def resolve_env_ids(
    env_ids: Sequence[int] | torch.Tensor | slice | None, num_envs: int, device: str | torch.device
) -> torch.Tensor:
    """Convert environment indices into an index tensor.

    The managers pass ``None`` or a :class:`slice` when all environments are selected, which cannot be
    counted with :func:`len` or used to gather per-environment values.

    Args:
        env_ids: Environment indices, a slice over all environments, or None for all environments.
        num_envs: Number of environments.
        device: Device of the returned tensor.

    Returns:
        The selected environment indices, shape ``(num_selected,)``, dtype ``torch.long``.
    """
    if env_ids is None:
        env_ids = slice(None)
    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device)[env_ids]
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


def hanging_tool_and_end_target(
    asset: RigidObject,
    body_id: int,
    tool_b: torch.Tensor,
    tool_rot_b: torch.Tensor | None,
    hang_offset_v: torch.Tensor,
    command_term: DroneTrajectoryCommand,
    env_origins: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the pose of a hanging tool and of its target at the trajectory end point.

    The target is where the tool would be if the vehicle sat level at the end point with the payload
    hanging at rest: the end position plus ``hang_offset_v`` rotated by the end yaw only.

    Args:
        asset: The asset carrying the tool body.
        body_id: Index of the tool body in the asset.
        tool_b: Tool point in the tool body frame [m], shape ``(num_envs, 3)``.
        tool_rot_b: Orientation of the tool frame in the tool body frame, as quaternions (x, y, z, w),
            shape ``(num_envs, 4)``, or None for the tool body orientation itself.
        hang_offset_v: Tool point relative to the tracked body when the payload hangs at rest below a
            level vehicle, in the yaw-aligned vehicle frame [m], shape ``(num_envs, 3)``.
        command_term: The trajectory command to read the end pose from.
        env_origins: Environment origins [m], shape ``(num_envs, 3)``.

    Returns:
        A tuple ``(tool_pos, tool_quat, target_pos, target_quat)``, in the environment-local world
        frame. Positions are in [m], shape ``(num_envs, 3)``; orientations are quaternions
        (x, y, z, w), shape ``(num_envs, 4)``. The target orientation is the end yaw.
    """
    body_pos = asset.data.body_pos_w.torch[:, body_id] - env_origins
    body_quat = asset.data.body_quat_w.torch[:, body_id]
    tool_pos = body_pos + math_utils.quat_apply(body_quat, tool_b)
    tool_quat = body_quat if tool_rot_b is None else math_utils.quat_mul(body_quat, tool_rot_b)

    zeros = torch.zeros_like(command_term.yaw_end)
    end_yaw_quat = math_utils.quat_from_euler_xyz(zeros, zeros, command_term.yaw_end)
    target_pos = command_term.pos_end_w + math_utils.quat_apply(end_yaw_quat, hang_offset_v)
    return tool_pos, tool_quat, target_pos, end_yaw_quat
