# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create drone termination terms.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_above_maximum(
    env: ManagerBasedRLEnv, maximum_height: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terminate when the asset's root height is above the maximum height.

    Note:
        This is currently only supported for flat terrains, i.e. the maximum height is in the world frame.

    Args:
        env: The environment instance.
        maximum_height: The maximum allowed root height in the world frame [m].
        asset_cfg: The configuration for the asset to check. Defaults to ``SceneEntityCfg("robot")``.

    Returns:
        A boolean tensor indicating whether the asset's root height is above the maximum height.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w.torch[:, 2] > maximum_height


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


def tracked_position_error_above_maximum(
    env: ManagerBasedRLEnv, max_error: float, command_name: str = "trajectory"
) -> torch.Tensor:
    """Terminate when the frame tracked by a trajectory command has fallen too far behind it.

    Unlike :func:`position_error_above_maximum`, which compares a named body with any position
    command, this compares the frame tracked by a ``DroneTrajectoryCommand`` (see its
    ``tracked_frame_state``), so it follows a tool frame when the command tracks one.

    Args:
        env: The manager-based RL environment instance.
        max_error: Position error at which to terminate [m].
        command_name: Name of the trajectory command to read the reference and tracked frame from.

    Returns:
        A boolean tensor of shape (num_envs,) that is True where the episode should terminate.
    """
    command_term = env.command_manager.get_term(command_name)
    pos_w, _, _ = command_term.tracked_frame_state()
    return torch.norm(command_term.command[:, :3] - pos_w, dim=1) > max_error


class PositionErrorAboveMaximumAfterCapture(ManagerTermBase):
    """Terminate when the asset leaves a safety ball around the commanded position, once it has reached it.

    :func:`position_error_above_maximum` has to allow for the spawn distance, so on a task that starts
    metres from the target it cannot also bound how far the vehicle strays once it is there. This term
    arms itself per environment the first time the asset comes within ``capture_radius`` of the target,
    and from then on terminates as soon as the error exceeds ``max_error``. It disarms on reset.

    The ``params`` of the term configuration are:

    * ``max_error``: Position error at which an armed environment terminates [m].
    * ``capture_radius``: Position error below which the term arms [m].
    * ``command_name``: Name of the command to read the target position from, in the first three
      columns of the command tensor, in the environment-local world frame.
    * ``asset_cfg``: SceneEntityCfg identifying the asset, with ``body_ids`` selecting the body link.
      None measures the frame tracked by the trajectory command instead (see its
      ``tracked_frame_state``), such as a tool frame. Defaults to the robot, which needs ``body_ids``.
    * ``window_s``: If given, the term is active, and can arm, only during the last ``window_s``
      seconds of the trajectory reference, such as the hold at its end point [s]. Defaults to None,
      active over the whole episode.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._armed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._armed[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        max_error: float,
        capture_radius: float,
        command_name: str = "trajectory",
        asset_cfg: SceneEntityCfg | None = SceneEntityCfg("robot"),
        window_s: float | None = None,
    ) -> torch.Tensor:
        """Compute the termination.

        Returns:
            A boolean tensor of shape (num_envs,) that is True where the episode should terminate.
        """
        if asset_cfg is None:
            command_term = env.command_manager.get_term(command_name)
            current_position, _, _ = command_term.tracked_frame_state()
            command = command_term.command
        else:
            asset: RigidObject = env.scene[asset_cfg.name]
            command = env.command_manager.get_command(command_name)
            current_position = asset.data.body_pos_w.torch[:, asset_cfg.body_ids[0]] - env.scene.env_origins  # type: ignore
        error = torch.norm(command[:, :3] - current_position, dim=1)
        active = error.new_ones(error.shape, dtype=torch.bool)
        if window_s is not None:
            cmd = env.command_manager.get_term(command_name)
            active = cmd.ref_time >= cmd.episode_length_s - window_s
        self._armed |= active & (error < capture_radius)
        return self._armed & active & (error > max_error)
