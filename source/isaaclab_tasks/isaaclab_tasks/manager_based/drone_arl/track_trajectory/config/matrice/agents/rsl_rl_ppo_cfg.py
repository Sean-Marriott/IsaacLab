# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


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
    empirical_normalization = (
        True  # Not sure if this is correct, intuitivily it seems correct but is disabled on most examples?
    )
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    policy = RslRlPpoActorCriticCfg(
        # Note this is in *normalized* action units: with max_velocity at 3 m/s rather than the
        # 0.5 m/s of the position-tracking task, a std of 0.5 would mean 1.5 m/s of velocity jitter,
        # which at K_vel = 4 is 6 m/s^2 of acceleration jitter and simply flips the drone. It is
        # deliberately left in normalized units when the speed envelope is dilated: the physical
        # jitter then scales with the envelope, which is what a spatial dilation asks for.
        init_noise_std=0.3,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
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
