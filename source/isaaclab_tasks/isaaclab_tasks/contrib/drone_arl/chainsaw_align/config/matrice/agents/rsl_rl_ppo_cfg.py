# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlRNNModelCfg


@configclass
class ChainsawAlignPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    # Not "matrice_chainsaw_align": that folder holds checkpoints of the trajectory-based align task,
    # whose observation layout differs.
    experiment_name = "matrice_chainsaw_align_base"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    # The observation mixes metres, radians and rad/s, so it is normalized.
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 128, 64],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        # 150 policy steps per episode; a horizon of ~100 steps (10 s) covers an approach and settle.
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class ChainsawAlignRNNPPORunnerCfg(ChainsawAlignPPORunnerCfg):
    """Recurrent PPO configuration, logged to its own folder.

    A GRU in front of the MLP lets the policy infer the payload's swing state and dynamics from its
    recent response rather than from a single frame. Select it with ``--agent rsl_rl_rnn_cfg_entry_point``.
    """

    experiment_name = "matrice_chainsaw_align_base_rnn"
    # Gradients are truncated at the rollout length, so the rollout has to cover enough of the swing to
    # identify it. The Mk3 pendulum has a ~2.1 s period; 48 steps at 10 Hz is about two periods.
    num_steps_per_env = 48

    # A smaller MLP than the feed-forward policy: the GRU state already carries the history.
    actor = RslRlRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlRNNModelCfg.GaussianDistributionCfg(init_std=0.5),
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    # The critic sees the same partial observations, so it needs the history too.
    critic = RslRlRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
