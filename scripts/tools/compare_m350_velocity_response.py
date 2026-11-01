# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay real M350 velocity commands in simulation and compare the response under two gain sets.

Purpose
-------
The Lee velocity gains of the M350 trajectory tasks were fitted to real flights. This script shows
the effect: it feeds the velocity commands recorded on the vehicle to the simulated M350 and plots
the simulated velocity under the previous gains and under the current config's gains against the
velocity the vehicle actually flew.

Method
------
Every segment from :mod:`extract_m350_velocity_logs` is replayed twice in one physics session, one
environment per segment and gain set:

* ``previous`` -- the gains before the fit (``--previous_gains``).
* ``current``  -- the centre of the gain ranges in the task config as it stands now.

Each environment starts level, at the vehicle's yaw and velocity at the start of its segment, with
the payload at rest. The command is held between samples and applied at the 100 Hz physics rate
(decimation 1). World-frame commands are rotated into the body frame with the simulated yaw, which
is how the vehicle's ground-fixed velocity mode treats them. Gains are pinned after reset, and every
randomized term is fixed at its nominal value, so the two runs of a segment differ only in gains.
The gust is off; drag is left on at the centre of its range.

Outputs
-------
* ``<output_dir>/<log>_seg<NN>.png``: command, measured and both simulated velocities (E, N, U).
* ``<output_dir>/summary.png``: tracking RMSE per segment and axis for both gain sets.
* ``<output_dir>/summary.txt``: the table printed to the console.

Usage::

    # chainsaw flights (payload logs are replayed on the Mk3 chainsaw model)
    uv run python scripts/tools/compare_m350_velocity_response.py --logs 02 03 04 05 06

    # open-loop steps (no chainsaw), with a 0.1 s command delay added to the current-gain run
    uv run python scripts/tools/compare_m350_velocity_response.py --logs 01 --current_cmd_delay 0.1

Run :mod:`extract_m350_velocity_logs` first to create the extracted data.
"""

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.math import quat_from_euler_xyz

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

parser = argparse.ArgumentParser(description="Compare simulated and real M350 velocity response.")
parser.add_argument("--data_dir", type=Path, default=Path("m350_velocity_logs/extracted"), help="Extracted logs.")
parser.add_argument("--logs", type=str, nargs="+", default=["02", "03", "04", "05", "06"], help="Log keys to replay.")
parser.add_argument(
    "--clean_task", type=str, default="IsaacContrib-TrackTrajectory-Matrice", help="Task without payload."
)
parser.add_argument(
    "--chainsaw_task", type=str, default="IsaacContrib-TrackTrajectory-Matrice-Chainsaw", help="Task with chainsaw."
)
parser.add_argument(
    "--previous_gains",
    type=float,
    nargs=4,
    default=[4.0, 4.0, 1.5, 2.5],
    metavar=("K_VEL_XY", "K_VEL_Z", "K_INT_XY", "K_INT_Z"),
    help="Gains of the 'previous' run.",
)
parser.add_argument("--previous_cmd_delay", type=float, default=0.0, help="Command delay of the 'previous' run [s].")
parser.add_argument("--current_cmd_delay", type=float, default=0.0, help="Command delay of the 'current' run [s].")
parser.add_argument("--output_dir", type=Path, default=Path("plots/m350_velocity_response"), help="Output directory.")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

AXES = ("E", "N", "U")
WARMUP_S = 2.0
"""Hover time before each replay, to bring the thrusters up to hover thrust [s]."""


def load_log(key: str) -> dict:
    """Load one extracted log."""
    d = args_cli.data_dir / key
    return {
        "key": key,
        "meta": json.loads((d / "meta.json").read_text()),
        "cmd": pd.read_csv(d / "cmd.csv"),
        "vel": pd.read_csv(d / "vel.csv"),
        "pose": pd.read_csv(d / "pose.csv"),
    }


def yaw_from_quat(q: np.ndarray) -> np.ndarray:
    """Yaw [rad] of (x, y, z, w) quaternions."""
    x, y, z, w = q.T
    return np.unwrap(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def build_segments(logs: list[dict], dt: float) -> list[dict]:
    """Resample every segment on the physics step: command (zero-order hold), measured velocity, yaw."""
    segments = []
    for log in logs:
        cmd, vel, pose, frame = log["cmd"], log["vel"], log["pose"], log["meta"]["frame"]
        yaw = yaw_from_quat(pose[["qx", "qy", "qz", "qw"]].values)
        for i, (a, b) in enumerate(log["meta"]["segments"]):
            t = np.arange(a, b, dt)
            idx = np.clip(np.searchsorted(cmd.t.values, t, side="right") - 1, 0, None)
            c = cmd[["vx", "vy", "vz", "yaw_rate"]].values[idx]
            psi = np.interp(t, pose.t, yaw)
            if frame == "body_flu":
                cmd_w = np.stack(
                    [
                        np.cos(psi) * c[:, 0] - np.sin(psi) * c[:, 1],
                        np.sin(psi) * c[:, 0] + np.cos(psi) * c[:, 1],
                        c[:, 2],
                    ],
                    1,
                )
            else:
                cmd_w = c[:, :3]
            segments.append(
                {
                    "name": f"{log['key']}_seg{i:02d}",
                    "t": t,
                    "frame": frame,
                    "cmd": c,  # as sent, in ``frame``
                    "cmd_w": cmd_w,  # world ENU, for plotting
                    "meas": np.stack([np.interp(t, vel.t, vel[f"v{k}"]) for k in AXES], 1),
                    "yaw0": float(psi[0]),
                }
            )
    return segments


def freeze_randomization(env_cfg):
    """Fix every randomized term at its nominal value and turn the gust off."""
    for name in list(vars(env_cfg.events)):
        term = getattr(env_cfg.events, name)
        if term is None or not hasattr(term, "func"):
            continue
        func_name = getattr(term.func, "__name__", str(term.func))
        if term.mode == "startup" and "randomize_rigid_body" in func_name:
            setattr(env_cfg.events, name, None)
            continue
        params = term.params or {}
        if "scale_range" in params:
            params["scale_range"] = (1.0, 1.0)
        if "force_range" in params:
            params["force_range"] = (0.0, 0.0)
            params["torque_range"] = (0.0, 0.0)
        if "drag_area_range" in params:
            mid = 0.5 * sum(params["drag_area_range"])
            params["drag_area_range"] = (mid, mid)
    for name in list(vars(env_cfg.terminations)):
        if name != "time_out" and getattr(env_cfg.terminations, name) is not None:
            setattr(env_cfg.terminations, name, None)


def main():
    logs = [load_log(k) for k in args_cli.logs]
    payloads = {log["meta"]["payload"] for log in logs}
    if len(payloads) != 1:
        raise ValueError(f"Logs mix payloads {payloads}; run the chainsaw and no-payload logs separately.")
    task = args_cli.chainsaw_task if payloads == {"chainsaw"} else args_cli.clean_task

    env_cfg, _ = resolve_task_config(task, None, play_mode=True, overrides=hydra_args)
    dt = env_cfg.sim.dt
    segments = build_segments(logs, dt)
    n_seg = len(segments)
    n_steps = max(len(s["t"]) for s in segments)

    ctrl_cfg = env_cfg.actions.velocity_command.controller_cfg
    current = {
        "K_vel": np.mean(ctrl_cfg.K_vel_range, axis=0),
        "K_vel_int": np.mean(ctrl_cfg.K_vel_int_range, axis=0),
        "K_rot": np.mean(ctrl_cfg.K_rot_range, axis=0),
        "K_angvel": np.mean(ctrl_cfg.K_angvel_range, axis=0),
    }
    p = args_cli.previous_gains
    previous = dict(current, K_vel=np.array([p[0], p[0], p[1]]), K_vel_int=np.array([p[2], p[2], p[3]]))
    gain_sets = {"previous": previous, "current": current}
    delays = {"previous": args_cli.previous_cmd_delay, "current": args_cli.current_cmd_delay}

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = 2 * n_seg
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.decimation = 1
        env_cfg.episode_length_s = n_steps * dt + WARMUP_S + 5.0
        env_cfg.scene.env_spacing = 10.0
        freeze_randomization(env_cfg)

        env = gym.make(task, cfg=env_cfg)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        device = base_env.device
        term = base_env.action_manager.get_term("velocity_command")
        lc = term._lc
        max_v, max_yr = term.cfg.max_velocity, term.cfg.max_yaw_rate
        print(f"[INFO] {task}: {n_seg} segments x 2 gain sets, {n_steps * dt:.1f} s, root body '{robot.body_names[0]}'")
        for name, g in gain_sets.items():
            print(
                f"[INFO] {name:8s} K_vel {g['K_vel'].round(2)}  K_vel_int {g['K_vel_int'].round(2)}"
                f"  delay {delays[name]} s"
            )

        # env i < n_seg replays segment i with the previous gains, env n_seg + i with the current ones.
        set_names = ["previous"] * n_seg + ["current"] * n_seg
        seg_of_env = list(range(n_seg)) * 2

        with torch.inference_mode():
            env.reset()
            for attr, key in (
                ("K_vel_current", "K_vel"),
                ("K_vel_int_current", "K_vel_int"),
                ("K_rot_current", "K_rot"),
                ("K_angvel_current", "K_angvel"),
            ):
                getattr(lc, attr)[:] = torch.tensor(
                    np.stack([gain_sets[s][key] for s in set_names]), dtype=torch.float32, device=device
                )
            lc._integral_enabled = bool((lc.K_vel_int_current > 0).any())

            # Level, at the vehicle's yaw, payload at rest; at the vehicle's velocity once replaying.
            yaw0 = torch.tensor([segments[i]["yaw0"] for i in seg_of_env], device=device)
            zeros = torch.zeros_like(yaw0)
            root_pose = robot.data.default_root_pose.torch.clone()
            root_pose[:, :3] += base_env.scene.env_origins
            root_pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw0)
            root_vel_start = torch.zeros_like(robot.data.default_root_vel.torch)
            root_vel_start[:, :3] = torch.tensor(np.stack([segments[i]["meas"][0] for i in seg_of_env]), device=device)
            joint_pos = robot.data.default_joint_pos.torch.clone()

            def place(root_vel: torch.Tensor):
                robot.write_root_pose_to_sim_index(root_pose=root_pose)
                robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
                robot.write_joint_position_to_sim_index(position=joint_pos)
                robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos))
                lc.vel_error_integral.zero_()

            # Hover first, so the thrusters are spun up to hover thrust when the replay starts;
            # otherwise every run opens with a sag while the motors catch up.
            place(torch.zeros_like(root_vel_start))
            hover = torch.zeros(2 * n_seg, 4, device=device)
            for _ in range(int(WARMUP_S / dt)):
                env.step(hover)
            place(root_vel_start)

            # Commands per env and step, padded by holding the last sample; delays shift them in time.
            cmd = np.zeros((2 * n_seg, n_steps, 4))
            frames = []
            for e, (s, name) in enumerate(zip(seg_of_env, set_names)):
                c = segments[s]["cmd"]
                n_delay = int(round(delays[name] / dt))
                c = np.concatenate([np.repeat(c[:1], n_delay, 0), c])[: len(c)]
                cmd[e, : len(c)] = c
                cmd[e, len(c) :] = c[-1]
                frames.append(segments[s]["frame"])
            cmd = torch.tensor(cmd, dtype=torch.float32, device=device)
            world = torch.tensor([f == "world_enu" for f in frames], device=device)

            sim_vel = np.zeros((2 * n_seg, n_steps, 3))
            for k in range(n_steps):
                c = cmd[:, k].clone()
                yaw = torch.atan2(
                    2.0
                    * (
                        robot.data.root_quat_w.torch[:, 3] * robot.data.root_quat_w.torch[:, 2]
                        + robot.data.root_quat_w.torch[:, 0] * robot.data.root_quat_w.torch[:, 1]
                    ),
                    1.0 - 2.0 * (robot.data.root_quat_w.torch[:, 1] ** 2 + robot.data.root_quat_w.torch[:, 2] ** 2),
                )
                # World-frame commands: rotate into the yaw-aligned body frame the controller expects.
                cw, sw = torch.cos(yaw), torch.sin(yaw)
                bx = cw * c[:, 0] + sw * c[:, 1]
                by = -sw * c[:, 0] + cw * c[:, 1]
                c[:, 0] = torch.where(world, bx, c[:, 0])
                c[:, 1] = torch.where(world, by, c[:, 1])
                action = torch.stack([c[:, 0] / max_v, c[:, 1] / max_v, c[:, 2] / max_v, c[:, 3] / max_yr], 1)
                env.step(action)
                sim_vel[:, k] = robot.data.root_lin_vel_w.torch.cpu().numpy()
                if k % 1000 == 0:
                    print(f"[INFO] step {k}/{n_steps}")

        # Report before leaving the launch context: closing the app ends the process.
        report(segments, sim_vel, logs)
        env.close()


def report(segments: list[dict], sim_vel: np.ndarray, logs: list[dict]):
    """Write the per-segment plots, the RMSE summary plot and the summary table."""
    n_seg = len(segments)
    os.makedirs(args_cli.output_dir, exist_ok=True)
    colors = {"cmd": "0.2", "meas": "C0", "previous": "C3", "current": "C2"}
    rows = []
    for s, seg in enumerate(segments):
        n = len(seg["t"])
        t = seg["t"] - seg["t"][0]
        sims = {"previous": sim_vel[s, :n], "current": sim_vel[n_seg + s, :n]}
        rmse = {k: np.sqrt(np.mean((v - seg["meas"]) ** 2, axis=0)) for k, v in sims.items()}
        rows.append((seg["name"], t[-1], rmse["previous"], rmse["current"]))

        fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
        for a, ax in enumerate(axs):
            ax.step(t, seg["cmd_w"][:, a], where="post", color=colors["cmd"], lw=1, label="command")
            ax.plot(t, seg["meas"][:, a], color=colors["meas"], lw=2, label="measured (real)")
            for k, v in sims.items():
                ax.plot(t, v[:, a], color=colors[k], lw=1.3, label=f"sim, {k} gains (RMSE {rmse[k][a]:.3f})")
            ax.set_ylabel(f"v{AXES[a]} [m/s]")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="upper right")
        axs[-1].set_xlabel("time in segment [s]")
        fig.suptitle(f"{seg['name']} ({logs[0]['meta']['payload']} payload, command frame {seg['frame']})")
        fig.tight_layout()
        fig.savefig(args_cli.output_dir / f"{seg['name']}.png", dpi=90)
        plt.close(fig)

    lines = [f"{'segment':12s} {'len[s]':>6s}  " + "  ".join(f"{k}:{a}" for k in ("prev", "curr") for a in AXES)]
    for name, length, rp, rc in rows:
        lines.append(f"{name:12s} {length:6.1f}  " + "  ".join(f"{x:6.3f}" for x in (*rp, *rc)))
    w = np.array([r[1] for r in rows])
    rp = np.sqrt(np.sum(w[:, None] * np.array([r[2] for r in rows]) ** 2, 0) / w.sum())
    rc = np.sqrt(np.sum(w[:, None] * np.array([r[3] for r in rows]) ** 2, 0) / w.sum())
    lines.append(f"{'ALL':12s} {w.sum():6.1f}  " + "  ".join(f"{x:6.3f}" for x in (*rp, *rc)))
    text = "\n".join(lines)
    print(text)
    (args_cli.output_dir / "summary.txt").write_text(text + "\n")

    fig, axs = plt.subplots(1, 3, figsize=(15, 0.35 * n_seg + 2), sharey=True)
    y = np.arange(n_seg)
    for a, ax in enumerate(axs):
        ax.barh(y - 0.2, [r[2][a] for r in rows], 0.4, color=colors["previous"], label="previous gains")
        ax.barh(y + 0.2, [r[3][a] for r in rows], 0.4, color=colors["current"], label="current gains")
        ax.set_title(f"v{AXES[a]} RMSE vs measured [m/s]  (all: {rp[a]:.3f} -> {rc[a]:.3f})")
        ax.grid(axis="x", alpha=0.3)
    axs[0].set_yticks(y, [r[0] for r in rows])
    axs[0].invert_yaxis()
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(args_cli.output_dir / "summary.png", dpi=90)
    print(f"[INFO] Plots written to {args_cli.output_dir}")


if __name__ == "__main__":
    main()
