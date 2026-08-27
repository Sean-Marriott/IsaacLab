# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quantify how well a policy follows the reference trajectory.

Method
------
Each requested condition is run as its own rollout in the same physics session, seeded identically
before reset so every condition sees the same trajectories and the same initial states.

* ``policy`` -- a trained RSL-RL checkpoint.
* ``oracle`` -- a hand-coded proportional controller on top of the same action interface:
  ``a = clip((v_ref + kp * position_error) / max_velocity, -1, 1)``. This is the feasibility gate.
  If the oracle cannot track the reference, neither will a policy, and the trajectory limits or the
  controller gains are what need fixing -- not the training run.
* ``zero`` -- zero velocity commands, i.e. hover in place. The floor for any comparison.

Metrics
-------
Beyond RMS error, the along-track and cross-track split is the diagnostic one. A large *signed*
along-track error with a small cross-track error is pure phase lag, which more look-ahead or a
faster velocity loop fixes. A large cross-track error instead means insufficient lateral authority
or too strong an action-rate penalty. The lag estimate cross-correlates the flown path against
time-shifted references and refines the peak by parabolic interpolation, which gives sub-step
resolution from 10 Hz data.

Usage::

    # feasibility gate, no checkpoint needed
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_trajectory_tracking.py \\
        --task Isaac-TrackTrajectory-Matrice-Play-v0 --conditions oracle zero

    # full comparison against a trained policy
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/eval_trajectory_tracking.py \\
        --task Isaac-TrackTrajectory-Matrice-Play-v0 --checkpoint /path/to/model.pt \\
        --conditions policy oracle zero --num_envs 64
"""

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate trajectory-tracking performance.")
parser.add_argument("--task", type=str, required=True, help="Gym task ID (use the Play variant).")
parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
parser.add_argument("--num_steps", type=int, default=200, help="Policy steps per rollout (200 = 20 s at 10 Hz).")
parser.add_argument(
    "--conditions",
    type=str,
    nargs="+",
    default=["oracle", "zero"],
    choices=["policy", "oracle", "zero"],
    help="Which controllers to evaluate. 'policy' requires a checkpoint.",
)
parser.add_argument("--oracle_kp", type=float, default=2.0, help="Position gain of the oracle controller [1/s].")
parser.add_argument("--oracle_kyaw", type=float, default=2.0, help="Yaw gain of the oracle controller [1/s].")
parser.add_argument("--difficulty", type=float, default=None, help="Override the trajectory difficulty.")
parser.add_argument("--seed_rollout", type=int, default=42, help="Seed applied before every rollout.")
parser.add_argument("--output", type=str, default=None, help="Output PNG path.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_args

installed_version = metadata.version("rsl-rl-lib")


def estimate_lag(position: np.ndarray, reference: np.ndarray, dt: float, max_lag_s: float = 1.0) -> float:
    """Estimate how far the flown path lags the reference.

    The flown path is compared against the reference shifted back in time by a whole number of
    steps, and the best whole-step shift is refined by fitting a parabola through the three error
    values around it. That recovers sub-step resolution from a coarsely sampled log.

    Args:
        position: Flown position [m], shape ``(num_steps, num_envs, 3)``.
        reference: Reference position [m], same shape as :paramref:`position`.
        dt: Time between samples [s].
        max_lag_s: Largest lag considered [s].

    Returns:
        The estimated lag [s]. Zero when the best shift is at the edge of the search range.
    """
    max_shift = max(int(round(max_lag_s / dt)), 2)
    errors = []
    for shift in range(max_shift + 1):
        # compare p(t) against p_ref(t - shift)
        flown = position[shift:] if shift else position
        shifted_reference = reference[: len(reference) - shift] if shift else reference
        errors.append(float(np.sqrt(((flown - shifted_reference) ** 2).sum(axis=-1).mean())))
    errors = np.asarray(errors)
    best = int(errors.argmin())
    if best == 0 or best == len(errors) - 1:
        return best * dt
    # parabolic refinement through (best-1, best, best+1)
    left, mid, right = errors[best - 1], errors[best], errors[best + 1]
    denominator = left - 2.0 * mid + right
    offset = 0.0 if abs(denominator) < 1e-12 else 0.5 * (left - right) / denominator
    return (best + offset) * dt


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    with launch_simulation(env_cfg, args_cli):
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.seed = agent_cfg.seed
        if args_cli.difficulty is not None:
            env_cfg.commands.trajectory.difficulty = args_cli.difficulty  # type: ignore[union-attr]

        log_dir = os.path.abspath(".")
        resume_path = None
        if "policy" in args_cli.conditions:
            log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            if args_cli.checkpoint:
                resume_path = retrieve_file_path(args_cli.checkpoint)
            else:
                resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            log_dir = os.path.dirname(resume_path)
            print(f"[INFO] Loading checkpoint: {resume_path}")

        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        device = env.unwrapped.device
        num_envs, num_steps = args_cli.num_envs, args_cli.num_steps
        action_dim = env.unwrapped.action_manager.total_action_dim
        command_term = env.unwrapped.command_manager.get_term("trajectory")
        action_term = env.unwrapped.action_manager.get_term("velocity_command")
        max_velocity = action_term.cfg.max_velocity
        max_yaw_rate = action_term.cfg.max_yaw_rate
        robot = env.unwrapped.scene["robot"]

        trained_policy = None
        if resume_path is not None:
            from rsl_rl.runners import OnPolicyRunner

            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(resume_path)
            trained_policy = runner.get_inference_policy(device=device)

        def oracle_policy(_obs: torch.Tensor) -> torch.Tensor:
            """Proportional position control on top of the reference velocity feed-forward."""
            import isaaclab.utils.math as math_utils

            position = robot.data.root_pos_w.torch - env.unwrapped.scene.env_origins
            velocity_setpoint_w = command_term.vel_ref_w + args_cli.oracle_kp * (command_term.pos_ref_w - position)
            # the action is interpreted in the yaw-only vehicle frame
            vehicle_quat = math_utils.yaw_quat(robot.data.root_link_quat_w.torch)
            velocity_setpoint_v = math_utils.quat_apply_inverse(vehicle_quat, velocity_setpoint_w)

            _, _, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w.torch)
            yaw_rate = command_term.yaw_rate_ref + args_cli.oracle_kyaw * math_utils.wrap_to_pi(
                command_term.yaw_ref - yaw
            )
            actions = torch.cat([velocity_setpoint_v / max_velocity, (yaw_rate / max_yaw_rate).unsqueeze(-1)], dim=-1)
            return actions.clamp(-1.0, 1.0)

        def zero_policy(_obs: torch.Tensor) -> torch.Tensor:
            return torch.zeros(num_envs, action_dim, device=device)

        policies = {"policy": trained_policy, "oracle": oracle_policy, "zero": zero_policy}

        def run_rollout(label: str, policy_fn) -> dict[str, np.ndarray]:
            """Step the environment for ``num_steps`` and log the reference and the flown state."""
            print(f"[INFO] Running rollout: {label} ...")
            logs = {
                key: np.zeros((num_steps, num_envs, size), dtype=np.float32)
                for key, size in (
                    ("pos", 3),
                    ("pos_ref", 3),
                    ("vel", 3),
                    ("vel_ref", 3),
                    ("action", action_dim),
                )
            }
            logs["yaw"] = np.zeros((num_steps, num_envs), dtype=np.float32)
            logs["yaw_ref"] = np.zeros((num_steps, num_envs), dtype=np.float32)
            logs["done"] = np.zeros((num_steps, num_envs), dtype=np.float32)

            import isaaclab.utils.math as math_utils

            with torch.inference_mode():
                # Seed before reset so every condition sees identical trajectories and spawns.
                torch.manual_seed(args_cli.seed_rollout)
                np.random.seed(args_cli.seed_rollout)
                # Reset inside inference_mode so metric buffers created as inference tensors in a
                # previous rollout can still be written in place.
                env.unwrapped.reset()
                obs = env.get_observations()

                for t in range(num_steps):
                    actions = policy_fn(obs)
                    logs["pos"][t] = (robot.data.root_pos_w.torch - env.unwrapped.scene.env_origins).cpu().numpy()
                    logs["pos_ref"][t] = command_term.pos_ref_w.cpu().numpy()
                    logs["vel"][t] = robot.data.root_lin_vel_w.torch.cpu().numpy()
                    logs["vel_ref"][t] = command_term.vel_ref_w.cpu().numpy()
                    _, _, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w.torch)
                    logs["yaw"][t] = yaw.cpu().numpy()
                    logs["yaw_ref"][t] = command_term.yaw_ref.cpu().numpy()
                    logs["action"][t] = actions.cpu().numpy()

                    obs, _, dones, _ = env.step(actions)
                    logs["done"][t] = dones.float().cpu().numpy()

                    if trained_policy is not None and hasattr(trained_policy, "reset"):
                        with contextlib.suppress(Exception):
                            trained_policy.reset(dones)
            return logs

        results = {}
        for condition in args_cli.conditions:
            if policies[condition] is None:
                raise RuntimeError("Condition 'policy' requires --checkpoint or a resolvable run directory.")
            results[condition] = run_rollout(condition, policies[condition])

        env.close()

        dt = env.unwrapped.step_dt
        time_axis = np.arange(num_steps) * dt

        # ------------------------------------------------------------------ #
        # Metrics
        # ------------------------------------------------------------------ #
        summaries = {}
        for condition, logs in results.items():
            error = logs["pos_ref"] - logs["pos"]
            error_norm = np.linalg.norm(error, axis=-1)

            # split the error along and across the reference direction of travel
            speed = np.linalg.norm(logs["vel_ref"], axis=-1, keepdims=True)
            tangent = logs["vel_ref"] / np.maximum(speed, 1e-6)
            along = (error * tangent).sum(axis=-1)
            cross = np.linalg.norm(error - along[..., None] * tangent, axis=-1)

            action_rate = np.linalg.norm(np.diff(logs["action"], axis=0), axis=-1)
            yaw_error = np.arctan2(np.sin(logs["yaw_ref"] - logs["yaw"]), np.cos(logs["yaw_ref"] - logs["yaw"]))

            summaries[condition] = {
                "rms": float(np.sqrt((error_norm**2).mean())),
                "mean": float(error_norm.mean()),
                "p95": float(np.percentile(error_norm, 95)),
                "max": float(error_norm.max()),
                "per_axis_rms": np.sqrt((error**2).mean(axis=(0, 1))),
                "along_mean": float(along.mean()),
                "along_rms": float(np.sqrt((along**2).mean())),
                "cross_rms": float(np.sqrt((cross**2).mean())),
                "lag_s": estimate_lag(logs["pos"], logs["pos_ref"], dt),
                "vel_rms": float(np.sqrt(((logs["vel_ref"] - logs["vel"]) ** 2).sum(axis=-1).mean())),
                "yaw_rms": float(np.sqrt((yaw_error**2).mean())),
                "action_rate_rms": float(np.sqrt((action_rate**2).mean())),
                "saturation": float((np.abs(logs["action"]) > 0.98).mean()),
                "dones": float(logs["done"].sum()),
                "error_norm": error_norm,
            }

        header = (
            f"{'condition':10s} {'rms':>7s} {'mean':>7s} {'p95':>7s} {'max':>7s} {'along':>8s} {'cross':>7s}"
            f" {'lag':>6s} {'velRMS':>7s} {'yawRMS':>7s} {'dA':>6s} {'sat':>6s} {'dones':>6s}"
        )
        lines = [
            "=" * len(header),
            f"Trajectory tracking summary — {args_cli.task}",
            f"{num_envs} environments x {num_steps} steps, difficulty {env_cfg.commands.trajectory.difficulty}",  # type: ignore[union-attr]
            "position errors in m, lag in s, yaw in rad; 'along' is the signed along-track error",
            "=" * len(header),
            header,
        ]
        for condition, s in summaries.items():
            lines.append(
                f"{condition:10s} {s['rms']:7.3f} {s['mean']:7.3f} {s['p95']:7.3f} {s['max']:7.3f}"
                f" {s['along_mean']:+8.3f} {s['cross_rms']:7.3f} {s['lag_s']:6.3f} {s['vel_rms']:7.3f}"
                f" {s['yaw_rms']:7.3f} {s['action_rate_rms']:6.3f} {s['saturation']:6.3f} {s['dones']:6.0f}"
            )
        lines.append("=" * len(header))
        for condition, s in summaries.items():
            axis = s["per_axis_rms"]
            lines.append(f"{condition:10s} per-axis RMS: x={axis[0]:.3f}  y={axis[1]:.3f}  z={axis[2]:.3f}")
        summary_text = "\n".join(lines)
        print("\n" + summary_text + "\n")

        # ------------------------------------------------------------------ #
        # Plots
        # ------------------------------------------------------------------ #
        colors = {"policy": "tab:blue", "oracle": "tab:green", "zero": "tab:orange"}
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f"Trajectory tracking — {args_cli.task}", fontsize=13)

        reference = next(iter(results.values()))["pos_ref"]
        axes[0, 0].plot(reference[:, 0, 0], reference[:, 0, 1], "k--", linewidth=1.6, label="reference")
        for condition, logs in results.items():
            axes[0, 0].plot(
                logs["pos"][:, 0, 0], logs["pos"][:, 0, 1], color=colors[condition], linewidth=1.4, label=condition
            )
        axes[0, 0].set_title("xy path (environment 0)", fontsize=10)
        axes[0, 0].set_xlabel("x [m]")
        axes[0, 0].set_ylabel("y [m]")
        axes[0, 0].set_aspect("equal", adjustable="datalim")

        for condition, s in summaries.items():
            mean = s["error_norm"].mean(axis=1)
            std = s["error_norm"].std(axis=1)
            axes[0, 1].plot(time_axis, mean, color=colors[condition], linewidth=1.6, label=condition)
            axes[0, 1].fill_between(time_axis, mean - std, mean + std, color=colors[condition], alpha=0.15)
        axes[0, 1].set_title("position error, mean ± std across environments", fontsize=10)
        axes[0, 1].set_xlabel("t [s]")
        axes[0, 1].set_ylabel("‖e‖ [m]")

        for index, axis_name in enumerate("xyz"):
            axes[1, 0].plot(
                time_axis, reference[:, 0, index], "--", linewidth=1.2, color=f"C{index}", label=f"{axis_name} ref"
            )
        for condition, logs in results.items():
            for index in range(3):
                axes[1, 0].plot(
                    time_axis,
                    logs["pos"][:, 0, index],
                    color=f"C{index}",
                    alpha=0.85,
                    linewidth=1.2,
                    linestyle="-" if condition != "zero" else ":",
                )
        axes[1, 0].set_title("per-axis position (environment 0, dashed = reference)", fontsize=10)
        axes[1, 0].set_xlabel("t [s]")
        axes[1, 0].set_ylabel("position [m]")

        reference_speed = np.linalg.norm(next(iter(results.values()))["vel_ref"], axis=-1).mean(axis=1)
        axes[1, 1].plot(time_axis, reference_speed, "k--", linewidth=1.6, label="reference")
        for condition, logs in results.items():
            axes[1, 1].plot(
                time_axis,
                np.linalg.norm(logs["vel"], axis=-1).mean(axis=1),
                color=colors[condition],
                linewidth=1.4,
                label=condition,
            )
        axes[1, 1].set_title("speed, averaged across environments", fontsize=10)
        axes[1, 1].set_xlabel("t [s]")
        axes[1, 1].set_ylabel("‖v‖ [m/s]")

        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()

        out_path = os.path.abspath(
            args_cli.output if args_cli.output else os.path.join(log_dir, "trajectory_tracking.png")
        )
        fig.savefig(out_path, dpi=150)
        # Also persist the table: the simulator redirects stdout, so console output is easy to lose.
        summary_path = os.path.splitext(out_path)[0] + ".txt"
        with open(summary_path, "w") as summary_file:
            summary_file.write(summary_text + "\n")
        print(f"[INFO] Plot saved: {out_path}")
        print(f"[INFO] Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
