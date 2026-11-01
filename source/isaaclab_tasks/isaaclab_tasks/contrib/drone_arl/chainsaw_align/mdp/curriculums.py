# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for the chainsaw alignment environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .tool_pose_command import ToolPoseCommand


class SuccessDifficultyCurriculum(ManagerTermBase):
    """Raise a difficulty level with the success rate, and ramp event parameters with it.

    A level in [0, 1] goes up by ``step`` after every ``min_episodes`` finished episodes whose success rate
    (from the command term) is above ``promote_rate``, and down by ``step`` when it is below ``demote_rate``.

    Each entry of ``ramps`` maps ``"<event term>.<parameter>"`` to ``(at_start, at_end, level_start,
    level_end)``: the parameter goes linearly from ``at_start`` to ``at_end`` as the level goes from
    ``level_start`` to ``level_end``, and is held outside that window. Giving the ramps different windows
    staggers them, so one source of difficulty can be mastered before the next comes in. A parameter that is
    a range is written as the symmetric range ``(-value, value)``; any other is written as the value.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.level = 0.0
        self._successes = 0
        self._episodes = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int] | torch.Tensor,
        command_name: str,
        ramps: dict[str, tuple[float, float, float, float]],
        promote_rate: float = 0.6,
        demote_rate: float = 0.3,
        step: float = 0.1,
        min_episodes: int | None = None,
    ) -> dict[str, float]:
        """Update the level from the episodes that just ended and write the ramped parameters for the new ones.

        Args:
            env: The manager-based RL environment instance.
            env_ids: Indices of the environments being reset.
            command_name: Name of the :class:`ToolPoseCommand` term that records success.
            ramps: Parameters to ramp with the level; see the class description.
            promote_rate: Success rate above which the level goes up.
            demote_rate: Success rate below which the level goes down.
            step: Change of the level per evaluation.
            min_episodes: Number of finished episodes per evaluation. Defaults to the number of environments.

        Returns:
            The level and every ramped value, for logging.
        """
        # The curriculum runs before the command is reset, so the success flags are those of the ended episodes.
        # Only episodes that ran count, so the reset at startup is not scored as a wave of failures.
        term: ToolPoseCommand = env.command_manager.get_term(command_name)
        ran = env.episode_length_buf[env_ids] > 0
        self._successes += int((term.succeeded[env_ids] & ran).sum().item())
        self._episodes += int(ran.sum().item())
        if self._episodes >= (min_episodes or env.num_envs):
            rate = self._successes / self._episodes
            if rate > promote_rate:
                self.level = min(self.level + step, 1.0)
            elif rate < demote_rate:
                self.level = max(self.level - step, 0.0)
            self._successes = 0
            self._episodes = 0

        state = {"level": self.level}
        for address, (at_start, at_end, level_start, level_end) in ramps.items():
            event_name, param_name = address.split(".")
            fraction = min(max((self.level - level_start) / max(level_end - level_start, 1e-6), 0.0), 1.0)
            value = at_start + fraction * (at_end - at_start)
            params = env.event_manager.get_term_cfg(event_name).params
            params[param_name] = (-value, value) if isinstance(params[param_name], tuple) else value
            state[address] = value
        return state
