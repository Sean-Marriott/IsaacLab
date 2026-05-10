# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained swing-suppression policy against a passive-settling baseline.

Purpose
-------
A chainsaw attached to a drone via a two-DOF pendulum joint (csTubePitch,
csTubeRoll) will swing when the drone manoeuvres or is disturbed. This script
quantifies how much a trained RL policy reduces that swing compared with doing
nothing: zero velocity commands, drone hovers in place, and the chainsaw settles
passively under gravity and damping alone.

Method
------
Two rollouts are executed back-to-back in the same physics session:

1. **Trained policy** — loads the supplied RSL-RL checkpoint and acts normally.
2. **Zero-action baseline** — outputs zero velocity commands every step. The
   Lee velocity controller still stabilises the drone's attitude and holds it
   at hover; the chainsaw receives no active suppression.

Both rollouts are seeded identically before reset so every environment starts
from the same randomly-sampled joint angles (±60° pitch, ±34° roll) and drone
pose, making the comparison controlled. The ``push_robot`` disturbance event is
disabled so the only variable between the two conditions is the policy itself.
Results are averaged over ``--num_envs`` parallel environments (default 50).

Outputs
-------
* A PNG plot of mean ± std of |joint angle| vs time for each joint and each
  condition, saved to ``<checkpoint_dir>/swing_comparison.png`` by default.
* A console summary reporting mean residual angle and mean first-passage settling
  time (threshold 0.05 rad) for each condition.

Usage::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/compare_swing_suppression.py \\
        --task Isaac-TrackPositionDirectVel-Matrice-Play-v0 \\
        --checkpoint /path/to/model.pt \\
        --num_envs 50 \\
        --num_steps 200 \\
        --headless
"""

import argparse
import importlib.metadata as metadata
import os
import sys

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from packaging import version

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.string import list_intersection

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

import cli_args  # isort: skip

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Compare swing suppression policy vs. zero-action baseline.")
parser.add_argument("--task", type=str, required=True, help="Gym task ID (use the Play variant for smaller num_envs).")
parser.add_argument("--num_envs", type=int, default=50, help="Number of parallel environments.")
parser.add_argument("--num_steps", type=int, default=200, help="Policy steps per rollout (200 steps = 20 s at 10 Hz).")
parser.add_argument("--output", type=str, default=None, help="Output PNG path. Defaults to <log_dir>/swing_comparison.png.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.seed = agent_cfg.seed

        # Disable disturbance pushes so the only difference between the two
        # rollouts is the policy — not random external forces.
        env_cfg.events.push_robot = None  # type: ignore[union-attr]

        # Resolve checkpoint
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
        print(f"[INFO] Loading checkpoint: {resume_path}")

        # Create and wrap env
        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # Load trained policy
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        trained_policy = runner.get_inference_policy(device=env.unwrapped.device)

        # Locate chainsaw joint indices
        robot = env.unwrapped.scene["robot"]
        joint_names = list(robot.joint_names)
        try:
            pitch_idx = joint_names.index("csTubePitch")
            roll_idx = joint_names.index("csTubeRoll")
        except ValueError as e:
            raise RuntimeError(
                f"Could not find chainsaw joints in {joint_names}. "
                "Check joint names match csTubePitch / csTubeRoll."
            ) from e

        num_steps = args_cli.num_steps
        num_envs = args_cli.num_envs
        device = env.unwrapped.device
        action_dim = env.unwrapped.action_manager.total_action_dim

        def run_rollout(policy_fn, label: str, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
            """Step the env for num_steps and return (T, N) joint angle arrays."""
            print(f"[INFO] Running rollout: {label} ...")

            pitch_log = np.zeros((num_steps, num_envs), dtype=np.float32)
            roll_log = np.zeros((num_steps, num_envs), dtype=np.float32)

            with torch.inference_mode():
                # Seed before reset so both rollouts start from identical initial
                # conditions, making the comparison apples-to-apples.
                torch.manual_seed(seed)
                np.random.seed(seed)
                # Reset inside inference_mode so command-manager metric buffers
                # (created as inference tensors in the previous rollout) can be
                # updated inplace without triggering a version-counter error.
                env.unwrapped.reset()
                obs = env.get_observations()

                for t in range(num_steps):
                    actions = policy_fn(obs)
                    obs, _, dones, _ = env.step(actions)

                    # Reset recurrent state for any envs that just terminated
                    if version.parse(installed_version) >= version.parse("4.0.0") and hasattr(trained_policy, "reset"):
                        try:
                            trained_policy.reset(dones)
                        except Exception:
                            pass

                    jp = robot.data.joint_pos.torch  # (num_envs, num_joints)
                    pitch_log[t] = jp[:, pitch_idx].cpu().numpy()
                    roll_log[t] = jp[:, roll_idx].cpu().numpy()

            return pitch_log, roll_log

        zero_policy = lambda obs: torch.zeros(num_envs, action_dim, device=device)

        trained_pitch, trained_roll = run_rollout(trained_policy, "Trained swing-suppression policy", seed=42)
        zero_pitch, zero_roll = run_rollout(zero_policy, "Zero-action baseline (passive settling)", seed=42)

        env.close()

        # ------------------------------------------------------------------ #
        # Plot
        # ------------------------------------------------------------------ #
        dt = env.unwrapped.step_dt
        t_axis = np.arange(num_steps) * dt

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        fig.suptitle("Chainsaw swing suppression: trained policy vs. zero-action baseline", fontsize=13)

        plot_cfg = [
            (axes[0], trained_pitch, zero_pitch, "csTubePitch — |angle| [rad]"),
            (axes[1], trained_roll, zero_roll, "csTubeRoll — |angle| [rad]"),
        ]

        for ax, trained, zero, ylabel in plot_cfg:
            for data, color, label in [
                (trained, "tab:blue", "Trained"),
                (zero, "tab:orange", "Zero-action"),
            ]:
                abs_data = np.abs(data)
                mean = abs_data.mean(axis=1)
                std = abs_data.std(axis=1)
                ax.plot(t_axis, mean, color=color, label=label, linewidth=1.8)
                ax.fill_between(t_axis, mean - std, mean + std, alpha=0.18, color=color)

            ax.set_ylabel(ylabel)
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout()

        out_path = args_cli.output if args_cli.output else os.path.join(log_dir, "swing_comparison.png")
        out_path = os.path.abspath(out_path)
        fig.savefig(out_path, dpi=150)
        print(f"[INFO] Plot saved: {out_path}")

        # Also print summary statistics
        def settling_time(log: np.ndarray, threshold: float = 0.05) -> float:
            """Mean first-passage time (in seconds) below threshold across envs."""
            times = []
            for env_i in range(log.shape[1]):
                idx = np.where(np.abs(log[:, env_i]) < threshold)[0]
                times.append(t_axis[idx[0]] if len(idx) else float("nan"))
            return float(np.nanmean(times))

        print("\n--- Summary (mean |angle| at end of rollout) ---")
        for name, pitch, roll in [("Trained", trained_pitch, trained_roll), ("Zero-action", zero_pitch, zero_roll)]:
            print(
                f"  {name:12s}  pitch={np.abs(pitch[-1]).mean():.4f} rad  "
                f"roll={np.abs(roll[-1]).mean():.4f} rad  "
                f"pitch-settle={settling_time(pitch):.2f}s  "
                f"roll-settle={settling_time(roll):.2f}s"
            )


if __name__ == "__main__":
    main()
