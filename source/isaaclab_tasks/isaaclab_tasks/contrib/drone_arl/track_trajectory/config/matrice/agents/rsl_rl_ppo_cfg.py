# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
)


@configclass
class TrackTrajectoryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration for the drone trajectory-following task.

    At 4096 environments this collects about 131k transitions per iteration, so the default run is
    roughly 262M environment steps.
    """

    num_steps_per_env = 32  # 3.2 s of trajectory per rollout at 10 Hz
    max_iterations = 2000
    save_interval = 100
    experiment_name = "matrice_track_trajectory"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        # The initial std is in *normalized* action units: with max_velocity at 3 m/s rather than the
        # 0.5 m/s of the position-tracking task, a std of 0.5 would mean 1.5 m/s of velocity jitter,
        # which at K_vel = 4 is 6 m/s^2 of acceleration jitter and simply flips the drone. It is
        # deliberately left in normalized units when the speed envelope is dilated: the physical
        # jitter then scales with the envelope, which is what a spatial dilation asks for.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.3),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.002,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,  # 100-step (10 s) effective horizon at 10 Hz
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class DetunedTrajectoryPPORunnerCfg(TrackTrajectoryPPORunnerCfg):
    """PPO configuration for the detuned trajectory-following task, logged to its own folder."""

    experiment_name = "matrice_track_trajectory_detuned"


@configclass
class ChainsawTrajectoryPPORunnerCfg(TrackTrajectoryPPORunnerCfg):
    """PPO configuration for the chainsaw trajectory-following task, logged to its own folder."""

    experiment_name = "matrice_chainsaw_trajectory"


@configclass
class ChainsawSwingPPORunnerCfg(TrackTrajectoryPPORunnerCfg):
    """PPO configuration for the chainsaw swing-suppression task, logged to its own folder."""

    experiment_name = "matrice_chainsaw_swing"


@configclass
class ChainsawSwingRNNPPORunnerCfg(ChainsawSwingPPORunnerCfg):
    """Recurrent PPO configuration for the chainsaw swing-suppression task, logged to its own folder.

    The swing dynamics depend on parameters the policy cannot observe in a single frame: hinge damping
    and spring rate are randomized over 0.5-2x, the saw mass over +/-15%, the airframe mass over +/-20%,
    and the observations arrive with a random 0-1 step lag. A GRU in front of the MLP lets the policy
    infer them from the recent response of the payload, as an adaptive controller would, rather than
    having to act on the average over the randomization.

    Select it with ``--agent rsl_rl_rnn_cfg_entry_point``.
    """

    experiment_name = "matrice_chainsaw_swing_rnn"
    # Gradients are truncated at the rollout length, so the rollout has to cover enough of the swing to
    # identify it. The Mk3 pendulum runs at ~3 rad/s (a ~2.1 s period); 48 steps at 10 Hz is 4.8 s,
    # about two periods, where the 32-step default sees only one and a half.
    num_steps_per_env = 48

    # A smaller MLP than the feed-forward policy: the GRU state already carries the history.
    actor = RslRlRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        # Same initial std as the feed-forward runner, in normalized action units.
        distribution_cfg=RslRlRNNModelCfg.GaussianDistributionCfg(init_std=0.3),
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    # The critic is recurrent too: with the same partial observations it needs the history as much as
    # the actor does to value a state.
    critic = RslRlRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )


@configclass
class ChainsawToolPPORunnerCfg(TrackTrajectoryPPORunnerCfg):
    """PPO configuration for the chainsaw tool-tracking task, with a stronger entropy bonus.

    At 0.002 the action noise kept collapsing after the difficulty ramp ended: the summed entropy fell
    to about -21.5 by iteration 1200 (a std near 0.001, in normalized units) and was still falling,
    while the tool position error rose from 6.5 to 7.7 cm at the end of the hold. The swing task, with
    the same coefficient, levelled off near -15 with a flat error. rsl_rl has no lower bound on the
    std, so the bonus is raised instead.
    """

    experiment_name = "matrice_chainsaw_tool"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.005


@configclass
class ChainsawToolRNNPPORunnerCfg(ChainsawSwingRNNPPORunnerCfg):
    """Recurrent PPO configuration for the chainsaw tool-tracking task, logged to its own folder.

    The swing task's recurrent runner, for the same reason: leading the cutting mouth along the
    reference means leading a pendulum whose damping, spring rate, mass and rest offset are randomized
    and cannot be read from a single frame. The entropy bonus is that of :class:`ChainsawToolPPORunnerCfg`.

    Select it with ``--agent rsl_rl_rnn_cfg_entry_point``.
    """

    experiment_name = "matrice_chainsaw_tool_rnn"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.005
