# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test a chainsaw swing-suppression policy on a Mk3 chainsaw that hangs at a fixed offset from vertical.

Purpose
-------
The swing-suppression policy (``IsaacContrib-TrackTrajectory-Matrice-ChainsawSwing``, M350 + Mk3 chainsaw) is
trained on a payload that hangs straight down at rest, so its hinge encoders read zero with no swing. On
the real vehicle the chainsaw hangs at a fixed offset, by default 5.4 deg forward and 3 deg right. This
script measures what that does to the policy: the hinge observation then reads a constant non-zero
angle at rest, which the policy has never seen without a swing.

Offset model
------------
The offset is a *static hang offset*: the payload's equilibrium is tilted, not its encoders. It is
made by moving the saw (``roll_link``, ~1.9 of the ~2.0 kg payload) COM sideways, in the saw's own
frame, until the payload COM lies straight below the pivot when the hinges are at the offset
angles. So:

* At rest the payload hangs at the offset, and the hinge observation reads the offset.
* The swing measured against gravity (:class:`PayloadSwingEnergy`, the reward's swing term) is
  zero at the new equilibrium, so the swing metrics below are about the true rest pose.
* Every reset puts the hinges at the offset, on top of whatever swing the task samples, so a
  payload "released at rest" really starts at rest.

Signs follow ``scripts/tools/check_chainsaw_joint_signs.py``: ``+csTubePitch`` swings the payload aft
and ``+csTubeRoll`` swings it left, so "forward" is ``-csTubePitch`` and "right" is ``-csTubeRoll``. Pass
negative values to ``--offset_forward_deg`` / ``--offset_right_deg`` for aft / left.

Method
------
The conditions run back to back in one physics session, seeded identically before reset:

* ``nominal`` -- the payload as trained (hangs straight down).
* ``offset``  -- the payload with the fixed hang offset.

Outputs
-------
* ``<checkpoint_dir>/swing_offset.png``: swing amplitude, hinge angles, airframe position error and
  airframe tilt vs time for each condition.
* ``<checkpoint_dir>/swing_offset.txt``: the summary table printed to the console.

Usage::

    # hold on the target pose, graded initial swing (default task)
    uv run python scripts/reinforcement_learning/rsl_rl/eval_chainsaw_swing_offset.py \\
        --checkpoint /path/to/model.pt

    # fly to the target with regular kicks, recurrent policy, 4 deg forward / 2 deg left
    uv run python scripts/reinforcement_learning/rsl_rl/eval_chainsaw_swing_offset.py \\
        --task IsaacContrib-TrackTrajectory-Matrice-ChainsawSwing --agent rsl_rl_rnn_cfg_entry_point \\
        --checkpoint /path/to/model.pt --offset_forward_deg 4.0 --offset_right_deg -2.0
"""

import argparse
import contextlib
import math
import os
from typing import TYPE_CHECKING

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

parser = argparse.ArgumentParser(description="Test chainsaw swing suppression with a fixed chainsaw hang offset.")
parser.add_argument(
    "--task",
    type=str,
    default="IsaacContrib-TrackTrajectory-Matrice-ChainsawSwingHold",
    help="Gym task ID of a chainsaw swing-suppression task (play-mode overrides are applied).",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--num_envs", type=int, default=50, help="Number of parallel environments.")
parser.add_argument("--num_steps", type=int, default=200, help="Policy steps per rollout (200 = 20 s at 10 Hz).")
parser.add_argument("--offset_forward_deg", type=float, default=5.4, help="Forward hang offset [deg]; <0 is aft.")
parser.add_argument("--offset_right_deg", type=float, default=3.0, help="Rightward hang offset [deg]; <0 is left.")
parser.add_argument(
    "--conditions",
    type=str,
    nargs="+",
    default=["nominal", "offset"],
    choices=["nominal", "offset"],
    help="Which payloads to evaluate.",
)
parser.add_argument("--settle_deg", type=float, default=3.0, help="Swing amplitude counted as settled [deg].")
parser.add_argument("--seed_rollout", type=int, default=42, help="Seed applied before every rollout.")
parser.add_argument("--output", type=str, default=None, help="Output PNG path.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
# Anything not parsed here is passed on to Hydra as config overrides.
args_cli, hydra_args = parser.parse_known_args()

HINGE_JOINT_NAMES = ["csTubePitch", "csTubeRoll"]
PAYLOAD_BODY_NAMES = ["main_tube", "base_link", "roll_link", "feed_arm_link"]
PIVOT_BODY_NAME = "top_hinge"
SAW_BODY_NAME = "roll_link"
OFFSET_EVENT_NAME = "reset_hinge_hang_offset"
SWING_TERM_NAME = "swing_suppression"


def add_hinge_offset(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    offset: tuple[float, float],
    asset_cfg: "SceneEntityCfg",
) -> None:
    """Shift the hinge angles by a fixed offset on reset, so the payload starts about its offset rest pose.

    Runs after the task's own reset terms and adds to the angles they wrote, keeping their velocities.

    Args:
        env: The manager-based RL environment instance.
        env_ids: Indices of the environments to reset.
        offset: Offset added to the two hinge joints, outer axis first [rad].
        asset_cfg: SceneEntityCfg identifying the asset and the two hinge joints.
    """
    if offset[0] == 0.0 and offset[1] == 0.0:
        return
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch[env_ids[:, None], asset_cfg.joint_ids].clone()
    joint_pos += torch.tensor(offset, device=joint_pos.device)
    limits = asset.data.soft_joint_pos_limits.torch[env_ids[:, None], asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])
    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def main():
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=True, overrides=hydra_args)
    with launch_simulation(env_cfg, args_cli):
        # USD-dependent modules can only be imported once the simulator is up.
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg

        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.seed = agent_cfg.seed
        # Added last, so it runs after the task's own hinge resets. Zero (inactive) until the offset
        # condition sets it.
        setattr(
            env_cfg.events,
            OFFSET_EVENT_NAME,
            EventTerm(
                func=add_hinge_offset,
                mode="reset",
                params={"offset": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot", joint_names=HINGE_JOINT_NAMES)},
            ),
        )

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
        print(f"[INFO] Loading checkpoint: {resume_path}")

        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        base_env: ManagerBasedRLEnv = env.unwrapped  # type: ignore[assignment]

        from rsl_rl.runners import OnPolicyRunner

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=base_env.device)

        import isaaclab.utils.math as math_utils

        device = base_env.device
        num_envs, num_steps = args_cli.num_envs, args_cli.num_steps
        robot = base_env.scene["robot"]
        command_term = base_env.command_manager.get_term("trajectory")
        swing_term = base_env.reward_manager.get_term_cfg(SWING_TERM_NAME).func
        offset_event = base_env.event_manager.get_term_cfg(OFFSET_EVENT_NAME)
        hinge_ids = [robot.find_joints(name)[0][0] for name in HINGE_JOINT_NAMES]
        pivot_id = robot.find_bodies(PIVOT_BODY_NAME)[0][0]
        saw_id = robot.find_bodies(SAW_BODY_NAME)[0][0]
        payload_ids = robot.find_bodies(PAYLOAD_BODY_NAMES, preserve_order=True)[0]
        # forward is -csTubePitch, right is -csTubeRoll (see check_chainsaw_joint_signs.py)
        hinge_offset = (-math.radians(args_cli.offset_forward_deg), -math.radians(args_cli.offset_right_deg))

        # ------------------------------------------------------------------ #
        # Offset payload: move the saw COM so the payload hangs at the offset
        # ------------------------------------------------------------------ #
        nominal_coms = robot.data.body_com_pose_b.torch.clone()
        all_env_ids = torch.arange(num_envs, dtype=torch.int, device=device)
        saw_body_ids = torch.tensor([saw_id], dtype=torch.int, device=device)

        def pose_payload(hinge_pos: tuple[float, float]) -> torch.Tensor:
            """Place every airframe level and at rest with the hinges at ``hinge_pos`` and return the
            payload COM relative to the pivot, in the world frame [m], shape (num_envs, 3)."""
            root_pose = robot.data.default_root_pose.torch.clone()
            root_pose[:, :3] = base_env.scene.env_origins + torch.tensor([0.0, 0.0, 3.0], device=device)
            joint_pos = robot.data.default_joint_pos.torch.clone()
            joint_pos[:, hinge_ids] += torch.tensor(hinge_pos, device=device)
            robot.write_root_pose_to_sim_index(root_pose=root_pose)
            robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(robot.data.default_root_vel.torch))
            robot.write_joint_position_to_sim_index(position=joint_pos)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos))
            # One physics step updates the body poses; the payload moves well under a millimetre in it.
            base_env.sim.step(render=False)
            robot.update(base_env.physics_dt)
            mass = robot.data.body_mass.torch[:, payload_ids].to(device)
            com_w = (mass.unsqueeze(-1) * robot.data.body_com_pos_w.torch[:, payload_ids]).sum(1) / mass.sum(
                1, keepdim=True
            )
            return com_w - robot.data.body_link_pos_w.torch[:, pivot_id]

        def tilt_from_vertical_deg(offset_w: torch.Tensor) -> np.ndarray:
            down = torch.tensor([0.0, 0.0, -1.0], device=device)
            cos = (offset_w @ down) / offset_w.norm(dim=-1)
            return np.degrees(torch.acos(cos.clamp(-1.0, 1.0)).cpu().numpy())

        def apply_hang_offset() -> None:
            """Write saw COMs that put the payload COM straight below the pivot at the offset hinge angles."""
            # Two passes: the first is exact up to the 13 mm between the hinge axes and the small change
            # in pendulum length, the second removes what is left.
            for _ in range(2):
                com_rel_w = pose_payload(hinge_offset)
                target_w = torch.zeros_like(com_rel_w)
                target_w[:, 2] = -com_rel_w.norm(dim=-1)
                mass = robot.data.body_mass.torch[:, payload_ids].to(device)
                # Move the whole payload COM by moving the saw COM alone.
                saw_shift_w = (target_w - com_rel_w) * (mass.sum(1) / robot.data.body_mass.torch[:, saw_id].to(device))[
                    :, None
                ]
                saw_quat_w = robot.data.body_link_quat_w.torch[:, saw_id]
                coms = robot.data.body_com_pose_b.torch.clone()
                coms[:, saw_id, :3] += math_utils.quat_apply_inverse(saw_quat_w, saw_shift_w)
                robot.set_coms_index(coms=coms[:, [saw_id]], body_ids=saw_body_ids, env_ids=all_env_ids)
            shift = (robot.data.body_com_pose_b.torch[:, saw_id, :3] - nominal_coms[:, saw_id, :3]).norm(dim=-1)
            print(
                f"[INFO] Saw COM moved by {shift.mean().item() * 1000:.1f} mm. Payload COM from vertical:"
                f" {tilt_from_vertical_deg(pose_payload(hinge_offset)).mean():.2f} deg at the offset hinge angles"
                f" (should be ~0), {tilt_from_vertical_deg(pose_payload((0.0, 0.0))).mean():.2f} deg at zero"
                f" (should be ~{math.degrees(math.hypot(*hinge_offset)):.1f})."
            )

        def restore_nominal() -> None:
            robot.set_coms_index(coms=nominal_coms[:, [saw_id]], body_ids=saw_body_ids, env_ids=all_env_ids)

        # ------------------------------------------------------------------ #
        # Rollouts
        # ------------------------------------------------------------------ #
        def run_rollout(condition: str) -> dict[str, np.ndarray]:
            print(f"[INFO] Running rollout: {condition} ...")
            logs = {
                "swing": np.zeros((num_steps, num_envs), dtype=np.float32),
                "hinge": np.zeros((num_steps, num_envs, 2), dtype=np.float32),
                "pos_error": np.zeros((num_steps, num_envs), dtype=np.float32),
                "tilt": np.zeros((num_steps, num_envs), dtype=np.float32),
                "roll_pitch": np.zeros((num_steps, num_envs, 2), dtype=np.float32),
                "done": np.zeros((num_steps, num_envs), dtype=np.float32),
            }
            with torch.inference_mode():
                if condition == "offset":
                    apply_hang_offset()
                    offset_event.params["offset"] = hinge_offset
                else:
                    restore_nominal()
                    offset_event.params["offset"] = (0.0, 0.0)

                # Seed before reset so every condition sees identical spawns, swings and kicks.
                torch.manual_seed(args_cli.seed_rollout)
                np.random.seed(args_cli.seed_rollout)
                base_env.reset()
                obs = env.get_observations()
                if hasattr(policy, "reset"):
                    with contextlib.suppress(Exception):
                        policy.reset(torch.ones(num_envs, dtype=torch.bool, device=device))

                for t in range(num_steps):
                    obs, _, dones, _ = env.step(policy(obs))
                    amplitude, _ = swing_term.swing_amplitude(base_env)
                    logs["swing"][t] = np.degrees(amplitude.cpu().numpy())
                    logs["hinge"][t] = np.degrees(robot.data.joint_pos.torch[:, hinge_ids].cpu().numpy())
                    pos = robot.data.root_pos_w.torch - base_env.scene.env_origins
                    logs["pos_error"][t] = (command_term.pos_ref_w - pos).norm(dim=-1).cpu().numpy()
                    quat = robot.data.root_quat_w.torch
                    up = math_utils.quat_apply(quat, torch.tensor([[0.0, 0.0, 1.0]], device=device).expand(num_envs, 3))
                    logs["tilt"][t] = np.degrees(torch.acos(up[:, 2].clamp(-1.0, 1.0)).cpu().numpy())
                    roll, pitch, _ = math_utils.euler_xyz_from_quat(quat)
                    logs["roll_pitch"][t, :, 0] = np.degrees(math_utils.wrap_to_pi(roll).cpu().numpy())
                    logs["roll_pitch"][t, :, 1] = np.degrees(math_utils.wrap_to_pi(pitch).cpu().numpy())
                    logs["done"][t] = dones.float().cpu().numpy()
                    if hasattr(policy, "reset"):
                        with contextlib.suppress(Exception):
                            policy.reset(dones)
            return logs

        results = {condition: run_rollout(condition) for condition in args_cli.conditions}
        dt = base_env.step_dt
        env.close()

        # ------------------------------------------------------------------ #
        # Metrics
        # ------------------------------------------------------------------ #
        time_axis = np.arange(num_steps) * dt
        tail = time_axis >= time_axis[-1] - 3.0  # last 3 s

        def settle_time(swing: np.ndarray) -> float:
            """Mean time after which each environment's swing stays below the threshold [s]."""
            times = []
            for env_i in range(swing.shape[1]):
                above = np.where(swing[:, env_i] >= args_cli.settle_deg)[0]
                if len(above) == 0:
                    times.append(0.0)
                elif above[-1] < len(time_axis) - 1:
                    times.append(time_axis[above[-1] + 1])
                else:
                    times.append(np.nan)
            return float(np.nanmean(times)) if not np.all(np.isnan(times)) else float("nan")

        summaries = {}
        for condition, logs in results.items():
            unsettled = int(np.sum(logs["swing"][-1] >= args_cli.settle_deg))
            summaries[condition] = {
                "swing_mean": float(logs["swing"].mean()),
                "swing_tail": float(logs["swing"][tail].mean()),
                "swing_p95_tail": float(np.percentile(logs["swing"][tail], 95)),
                "settle": settle_time(logs["swing"]),
                "unsettled": unsettled,
                "hinge_tail": logs["hinge"][tail].mean(axis=(0, 1)),
                "pos_rms": float(np.sqrt((logs["pos_error"] ** 2).mean())),
                "pos_tail": float(logs["pos_error"][tail].mean()),
                "tilt_tail": float(logs["tilt"][tail].mean()),
                "roll_pitch_tail": logs["roll_pitch"][tail].mean(axis=(0, 1)),
                "dones": float(logs["done"].sum()),
            }

        header = (
            f"{'condition':9s} {'swing':>6s} {'swing3s':>7s} {'p95_3s':>7s} {'settle':>7s} {'unset':>5s}"
            f" {'pitchJ':>7s} {'rollJ':>7s} {'posRMS':>7s} {'pos3s':>6s} {'tilt3s':>6s} {'roll3s':>7s}"
            f" {'pitch3s':>7s} {'dones':>5s}"
        )
        lines = [
            "=" * len(header),
            f"Chainsaw swing suppression with a fixed hang offset — {args_cli.task}",
            f"checkpoint: {resume_path}",
            f"offset: {args_cli.offset_forward_deg:+.2f} deg forward, {args_cli.offset_right_deg:+.2f} deg right"
            f" (csTubePitch {math.degrees(hinge_offset[0]):+.2f} deg,"
            f" csTubeRoll {math.degrees(hinge_offset[1]):+.2f} deg)",
            f"{num_envs} environments x {num_steps} steps ({num_steps * dt:.1f} s)",
            "angles in deg, positions in m, settle in s; '3s' columns are means over the last 3 s;",
            f"swing is the amplitude about the true rest pose; settled below {args_cli.settle_deg:.1f} deg;"
            " pitchJ/rollJ are the mean hinge readings; roll3s/pitch3s the mean airframe attitude",
            "=" * len(header),
            header,
        ]
        for condition, s in summaries.items():
            lines.append(
                f"{condition:9s} {s['swing_mean']:6.2f} {s['swing_tail']:7.2f} {s['swing_p95_tail']:7.2f}"
                f" {s['settle']:7.2f} {s['unsettled']:5d} {s['hinge_tail'][0]:+7.2f} {s['hinge_tail'][1]:+7.2f}"
                f" {s['pos_rms']:7.3f} {s['pos_tail']:6.3f} {s['tilt_tail']:6.2f} {s['roll_pitch_tail'][0]:+7.2f}"
                f" {s['roll_pitch_tail'][1]:+7.2f} {s['dones']:5.0f}"
            )
        lines.append("=" * len(header))
        summary_text = "\n".join(lines)
        print("\n" + summary_text + "\n")

        # ------------------------------------------------------------------ #
        # Plots
        # ------------------------------------------------------------------ #
        colors = {"nominal": "tab:blue", "offset": "tab:red"}
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
        fig.suptitle(
            f"Swing suppression, hang offset {args_cli.offset_forward_deg:+.1f} deg fwd /"
            f" {args_cli.offset_right_deg:+.1f} deg right — {args_cli.task}",
            fontsize=12,
        )

        def band(ax, data: np.ndarray, color: str, label: str) -> None:
            mean, std = data.mean(axis=1), data.std(axis=1)
            ax.plot(time_axis, mean, color=color, linewidth=1.6, label=label)
            ax.fill_between(time_axis, mean - std, mean + std, color=color, alpha=0.15)

        for condition, logs in results.items():
            band(axes[0, 0], logs["swing"], colors[condition], condition)
            band(axes[1, 0], logs["pos_error"], colors[condition], condition)
            band(axes[1, 1], logs["tilt"], colors[condition], condition)
            for index, (name, style) in enumerate(zip(HINGE_JOINT_NAMES, ("-", "--"))):
                axes[0, 1].plot(
                    time_axis,
                    logs["hinge"][:, :, index].mean(axis=1),
                    color=colors[condition],
                    linestyle=style,
                    linewidth=1.4,
                    label=f"{condition} {name}",
                )
        axes[0, 0].axhline(args_cli.settle_deg, color="k", linestyle=":", linewidth=1.0, label="settled")
        if "offset" in results:
            for value, style in zip(hinge_offset, ("-", "--")):
                axes[0, 1].axhline(math.degrees(value), color="k", linestyle=style, linewidth=0.8, alpha=0.5)

        axes[0, 0].set_title("swing amplitude about rest, mean ± std", fontsize=10)
        axes[0, 0].set_ylabel("θ_amp [deg]")
        axes[0, 1].set_title("hinge readings, env mean (grey = offset rest angles)", fontsize=10)
        axes[0, 1].set_ylabel("joint angle [deg]")
        axes[1, 0].set_title("airframe distance to target, mean ± std", fontsize=10)
        axes[1, 0].set_ylabel("‖e‖ [m]")
        axes[1, 1].set_title("airframe tilt from level, mean ± std", fontsize=10)
        axes[1, 1].set_ylabel("tilt [deg]")
        for ax in axes.flat:
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        for ax in axes[1]:
            ax.set_xlabel("t [s]")
        fig.tight_layout()

        out_path = os.path.abspath(args_cli.output if args_cli.output else os.path.join(log_dir, "swing_offset.png"))
        fig.savefig(out_path, dpi=150)
        # The simulator redirects stdout, so console output is easy to lose.
        summary_path = os.path.splitext(out_path)[0] + ".txt"
        with open(summary_path, "w") as summary_file:
            summary_file.write(summary_text + "\n")
        print(f"[INFO] Plot saved: {out_path}")
        print(f"[INFO] Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
