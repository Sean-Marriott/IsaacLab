# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create drone termination terms.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def position_error_above_maximum(
    env: ManagerBasedRLEnv,
    max_error: float,
    command_name: str = "trajectory",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the asset has fallen too far behind the commanded position.

    An episode that has diverged this far is not recoverable in any useful sense and only wastes
    samples, so it is cut short.

    Args:
        env: The manager-based RL environment instance.
        max_error: Position error at which to terminate [m].
        command_name: Name of the command to read the target position from. The function expects
            the target position in the first three columns of the command tensor, expressed in the
            environment-local world frame.
        asset_cfg: SceneEntityCfg identifying the asset. Must have ``body_ids`` configured to select
            the tracked body link.

    Returns:
        A boolean tensor of shape (num_envs,) that is True where the episode should terminate.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    current_position = asset.data.body_pos_w.torch[:, asset_cfg.body_ids[0]] - env.scene.env_origins  # type: ignore
    return torch.norm(command[:, :3] - current_position, dim=1) > max_error
