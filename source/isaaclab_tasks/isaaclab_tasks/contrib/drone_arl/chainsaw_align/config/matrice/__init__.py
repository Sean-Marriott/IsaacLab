# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="IsaacContrib-ChainsawAlign-Matrice",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.chainsaw_align_env_cfg:MatriceChainsawAlignEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChainsawAlignPPORunnerCfg",
        "rsl_rl_rnn_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChainsawAlignRNNPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-ChainsawAlignBranch-Matrice",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.chainsaw_align_branch_env_cfg:MatriceChainsawAlignBranchEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChainsawAlignBranchPPORunnerCfg",
        "rsl_rl_rnn_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ChainsawAlignBranchRNNPPORunnerCfg",
    },
)
