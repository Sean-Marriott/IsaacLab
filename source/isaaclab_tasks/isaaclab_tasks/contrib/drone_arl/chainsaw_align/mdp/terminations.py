# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the chainsaw alignment environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .tool_pose_command import ToolPoseCommand


def tool_distance_above_maximum(env: ManagerBasedRLEnv, command_name: str, max_distance: float) -> torch.Tensor:
    """Terminate when the tool is further than ``max_distance`` [m] from its target. Shape is (num_envs,)."""
    term: ToolPoseCommand = env.command_manager.get_term(command_name)
    return torch.linalg.norm(term.target_error_w(), dim=-1) > max_distance
