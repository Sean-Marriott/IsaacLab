# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluation environment flying the same geometric test figures as the vehicle.

``dji_tap_and_go/generate_test_trajectory.py`` flies a square, circle, line or zigzag on the real
drone. This environment puts a policy on those same figures in simulation, at the same size and
cruise speed, so the two can be compared directly instead of against the randomly sampled
sum-of-sinusoids reference the policy trains on.

This is an evaluation configuration only. It is deterministic by construction -- fixed figures,
fixed assignment to environments, no observation corruption and no gust -- so two runs of the same
checkpoint produce the same numbers.
"""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.drone_arl.mdp.commands import DroneTestShapeCommandCfg

from .env_cfg import MatriceCleanTrajectoryEnvCfg_PLAY

# The slowest figure is the zigzag, whose lap is far longer than the others at the same cruise
# speed (it covers the same span while weaving). One lap of it at the default 1.5 m / 0.5 m/s is
# about 25 s, so the episode has to be longer than the 20 s used for training or most environments
# would be cut off mid-figure and the metrics would score a partial lap.
EPISODE_LENGTH_SECONDS = 40.0


@configclass
class MatriceTestShapeEnvCfg_PLAY(MatriceCleanTrajectoryEnvCfg_PLAY):
    """Fly the ROS test figures: square, circle, line and zigzag, one per environment in turn."""

    def __post_init__(self):
        super().__post_init__()

        # A multiple of the four shapes, so each figure gets the same number of environments and
        # the aggregate metrics are not weighted towards whichever shape got the extra ones.
        self.scene.num_envs = 48
        # The figures span `size` (1.5 m), not the several metres a sampled trajectory reaches, so
        # they can be packed far tighter than the training PLAY config allows.
        self.scene.env_spacing = 6.0

        self.episode_length_s = EPISODE_LENGTH_SECONDS

        self.commands.trajectory = DroneTestShapeCommandCfg(
            asset_name="robot",
            body_name="base_link",
            # Strictly longer than an episode, matching the training config: the figure is fixed, so
            # a resample would draw the same shape again, but it would restart the lap clock
            # mid-episode and put a discontinuity in the reference.
            resampling_time_range=(EPISODE_LENGTH_SECONDS + 1.0, EPISODE_LENGTH_SECONDS + 1.0),
            debug_vis=True,
            # Matches generate_test_trajectory.py's --shape / --size / --max-speed / --num-points
            # defaults, so the two tools fly the same figures unless both are changed together.
            shapes=("square", "circle", "line", "zigzag"),
            size=1.5,
            speed=0.5,
            num_points=8,
            # The ROS tool defaults to --yaw-mode keep, a constant heading. "fixed" is its
            # equivalent and keeps the run deterministic; the policy trained on a per-episode random
            # constant yaw, of which this is one draw.
            yaw_mode="fixed",
        )
