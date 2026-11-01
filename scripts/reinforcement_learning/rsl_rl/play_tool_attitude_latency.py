# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a chainsaw tool-tracking policy with the cutting-mouth state estimated as on the vehicle: stale attitude, fresh encoders.

Purpose
-------
On the M350 the tool-tracking (alignment) policy measures from the saw's mouth. The vehicle builds the
mouth state from the DJI attitude, which is 70-80 ms old, and the gimbal encoders, which are ~2 ms old.
When the airframe rotates, the encoders already show the rotation and the attitude does not yet, so
the sum reports mouth motion that never happened (about 0.3 m/s of horizontal mouth velocity in
flight). In sim the mouth state is read straight from the physics, so the policy was never trained
on that error. This script reproduces it, to see what the policy does with it.

The swing-suppression task measures from the airframe, where the encoder chain is empty, so the
emulation adds nothing there (a useful check: run it with that task and the phantom reads zero).

Emulation
---------
While the observations are computed, the command's :meth:`tracked_frame_state` returns the estimate
instead of the truth (rewards, terminations and the physics keep the truth). With the airframe at
position ``p``, velocity ``v``, attitude ``R`` and body rate ``w``, the encoder chain gives the mouth in
the body frame: its offset ``r_b``, orientation ``q_b`` and the velocity ``v_rel_b`` the joints add. Those
are taken fresh from the sim. The estimate is::

    pos  = p + R_old r_b
    quat = q_old * q_b
    vel  = v + R_old (w_old x r_b + v_rel_b)

where ``R_old`` / ``q_old`` and ``w_old`` are the attitude and body rate ``--attitude_delay_ms`` ago, drawn per
environment and episode and recorded every physics step (so the delay resolves to the physics step,
10 ms). With no delay this is the true mouth state. The DJI position and velocity are taken fresh,
since a delay common to every input does not build the error. ``--fresh_body_rate`` delays the attitude
only. The task's own observation noise and delay still apply on top.

The first ``--control_envs`` environments observe the truth, as a reference in the same run.

Outputs
-------
* Console: on exit, the phantom mouth velocity and position (estimate minus truth, horizontal) over the
  emulated environments, and the true mouth tracking error, action jumps and episode ends of the
  emulated and control environments.
* ``<checkpoint_dir>/tool_attitude_latency_env<N>.csv`` (or ``--csv``): one row per policy step for
  ``--log_env``: the observation as the policy saw it, the true and estimated mouth position and
  velocity, the airframe roll, pitch and body rates, the action and the done flag.

Usage::

    # watch it, recurrent policy, half the environments on the truth
    uv run python scripts/reinforcement_learning/rsl_rl/play_tool_attitude_latency.py \\
        --agent rsl_rl_rnn_cfg_entry_point --checkpoint /path/to/model_1999.pt --viz kit \\
        --num_envs 16 --control_envs 8

    # headless, 60 s, attitude delayed but the body rate fresh
    uv run python scripts/reinforcement_learning/rsl_rl/play_tool_attitude_latency.py \\
        --agent rsl_rl_rnn_cfg_entry_point --checkpoint /path/to/model_1999.pt \\
        --num_envs 64 --control_envs 32 --num_steps 600 --fresh_body_rate
"""

import argparse
import contextlib
import csv
import os
import sys
import time

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import to_dict

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _obs_csv_columns, _resolve_checkpoint
from isaaclab_rl.entrypoints.common import (
    add_common_play_args,
    apply_env_overrides,
    close_env,
    create_isaaclab_env,
    set_hydra_args,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

XYZ = ("x", "y", "z")

parser = argparse.ArgumentParser(
    description="Play a tool-tracking policy with the mouth state built from stale attitude and fresh encoders."
)
add_common_play_args(
    parser, agent_default="rsl_rl_cfg_entry_point", agent_help="Name of the RL agent configuration entry point."
)
parser.set_defaults(task="IsaacContrib-TrackTrajectory-Matrice-ChainsawTool")
parser.add_argument(
    "--attitude_delay_ms",
    type=float,
    nargs=2,
    default=(70.0, 80.0),
    metavar=("MIN", "MAX"),
    help="Age of the attitude (and body rate) in the mouth estimate, drawn uniformly per episode [ms].",
)
parser.add_argument(
    "--fresh_body_rate", action="store_true", default=False, help="Delay the attitude only, not the body rate."
)
parser.add_argument("--control_envs", type=int, default=0, help="Number of leading environments that see the truth.")
parser.add_argument("--num_steps", type=int, default=None, help="Policy steps to run; until Ctrl+C if omitted.")
parser.add_argument(
    "--log_env",
    type=int,
    default=None,
    help="Environment written to the CSV (default: the first emulated one).",
)
parser.add_argument("--csv", type=str, default=None, help="CSV path for --log_env.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser, sys.argv[1:])
set_hydra_args(remaining_args)


def main():
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        import isaaclab.utils.math as math_utils

        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = agent_cfg.seed

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
        if resume_path is None:
            return
        log_dir = os.path.dirname(resume_path)
        print(f"[INFO] Loading checkpoint: {resume_path}")

        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        base_env = env.unwrapped

        runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=base_env.device)

        device = base_env.device
        num_envs = base_env.num_envs
        robot = base_env.scene["robot"]
        command_term = base_env.command_manager.get_term("trajectory")
        all_ids = torch.arange(num_envs, device=device)
        emulated = all_ids >= args_cli.control_envs
        log_env = args_cli.log_env if args_cli.log_env is not None else min(args_cli.control_envs, num_envs - 1)

        # ------------------------------------------------------------------ #
        # Attitude history, recorded after every physics step
        # ------------------------------------------------------------------ #
        physics_dt = base_env.physics_dt
        delay_min, delay_max = (round(ms * 1e-3 / physics_dt) for ms in args_cli.attitude_delay_ms)
        history_len = delay_max + 1
        quat_hist = robot.data.root_link_quat_w.torch.clone().unsqueeze(0).repeat(history_len, 1, 1)
        rate_hist = robot.data.root_link_ang_vel_b.torch.clone().unsqueeze(0).repeat(history_len, 1, 1)
        head = 0
        delay_steps = torch.randint(delay_min, delay_max + 1, (num_envs,), device=device)

        scene_update = base_env.scene.update

        def recording_update(dt: float) -> None:
            nonlocal head
            scene_update(dt)
            head = (head + 1) % history_len
            quat_hist[head] = robot.data.root_link_quat_w.torch
            rate_hist[head] = robot.data.root_link_ang_vel_b.torch

        base_env.scene.update = recording_update

        # ------------------------------------------------------------------ #
        # Mouth estimate, active while the observations are computed
        # ------------------------------------------------------------------ #
        true_frame_state = command_term.tracked_frame_state
        computing_obs = False
        # The truth and the estimate from the last observation, for the stats and the CSV.
        last = {}

        def estimated_frame_state():
            pos_t, quat_t, vel_t = true_frame_state()
            if not computing_obs:
                return pos_t, quat_t, vel_t
            data = robot.data
            p_air = data.root_link_pos_w.torch - base_env.scene.env_origins
            v_air = data.root_link_lin_vel_w.torch
            q_now = data.root_link_quat_w.torch
            w_now_b = data.root_link_ang_vel_b.torch
            # -- encoder chain, fresh: the mouth relative to the airframe, in the body frame
            r_w = pos_t - p_air
            r_b = math_utils.quat_apply_inverse(q_now, r_w)
            q_b = math_utils.quat_mul(math_utils.quat_inv(q_now), quat_t)
            v_rel_b = math_utils.quat_apply_inverse(q_now, vel_t - v_air) - torch.cross(w_now_b, r_b, dim=-1)
            # -- DJI attitude and body rate, stale (fresh on the first step of an episode, where the
            # history still holds the previous episode)
            index = (head - delay_steps) % history_len
            q_old = quat_hist[index, all_ids]
            w_old_b = w_now_b if args_cli.fresh_body_rate else rate_hist[index, all_ids]
            first = (base_env.episode_length_buf == 0).unsqueeze(-1)
            q_old = torch.where(first, q_now, q_old)
            w_old_b = torch.where(first, w_now_b, w_old_b)

            pos_e = p_air + math_utils.quat_apply(q_old, r_b)
            quat_e = math_utils.quat_mul(q_old, q_b)
            vel_e = v_air + math_utils.quat_apply(q_old, torch.cross(w_old_b, r_b, dim=-1) + v_rel_b)

            use = emulated.unsqueeze(-1)
            pos_o, quat_o, vel_o = (
                torch.where(use, e, t) for e, t in ((pos_e, pos_t), (quat_e, quat_t), (vel_e, vel_t))
            )
            last.update(pos_t=pos_t, vel_t=vel_t, pos_e=pos_e, vel_e=vel_e)
            return pos_o, quat_o, vel_o

        command_term.tracked_frame_state = estimated_frame_state

        obs_compute = base_env.observation_manager.compute

        def compute_with_estimate(*args, **kwargs):
            nonlocal computing_obs
            computing_obs = True
            try:
                return obs_compute(*args, **kwargs)
            finally:
                computing_obs = False

        base_env.observation_manager.compute = compute_with_estimate

        print(
            f"[INFO] Mouth estimated from {args_cli.attitude_delay_ms[0]:.0f}-{args_cli.attitude_delay_ms[1]:.0f} ms old"
            f" attitude ({delay_min}-{delay_max} physics steps)"
            f"{' with a fresh body rate' if args_cli.fresh_body_rate else ' and body rate'} and fresh encoders, in"
            f" {int(emulated.sum())}/{num_envs} environments."
        )

        # ------------------------------------------------------------------ #
        # CSV
        # ------------------------------------------------------------------ #
        csv_path = os.path.abspath(args_cli.csv or os.path.join(log_dir, f"tool_attitude_latency_env{log_env}.csv"))
        csv_file = open(csv_path, "w", newline="")  # noqa: SIM115 (closed by cleanup)
        cleanup.callback(csv_file.close)
        writer = csv.writer(csv_file)
        num_actions = base_env.action_manager.total_action_dim
        writer.writerow(
            ["time", "emulated"]
            + _obs_csv_columns(base_env)
            + [f"mouth_pos_true_{a}" for a in XYZ]
            + [f"mouth_pos_est_{a}" for a in XYZ]
            + [f"mouth_vel_true_{a}" for a in XYZ]
            + [f"mouth_vel_est_{a}" for a in XYZ]
            + ["roll", "pitch"]
            + [f"true_ang_vel_b_{a}" for a in XYZ]
            + [f"action_{i}" for i in range(num_actions)]
            + ["done"]
        )

        # ------------------------------------------------------------------ #
        # Rollout
        # ------------------------------------------------------------------ #
        phantom_vel, phantom_pos = [], []
        track_err = {"emulated": [], "control": []}
        jumps = {"emulated": [], "control": []}
        ends = {"emulated": 0, "control": 0}
        prev_actions = None
        fresh = torch.ones(num_envs, dtype=torch.bool, device=device)

        # Fill the estimate for the first observation, which the env computed before the patch.
        with torch.inference_mode():
            base_env.obs_buf = base_env.observation_manager.compute()
        obs = env.get_observations()
        step = 0
        dt = base_env.step_dt
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while args_cli.num_steps is None or step < args_cli.num_steps:
                start_time = time.time()
                actions = policy(obs)

                # -- stats on the observation the action was computed from
                valid = ~fresh
                d_vel = (last["vel_e"] - last["vel_t"])[:, :2].norm(dim=-1)
                d_pos = (last["pos_e"] - last["pos_t"])[:, :2].norm(dim=-1)
                phantom_vel.append(d_vel[valid & emulated].cpu())
                phantom_pos.append(d_pos[valid & emulated].cpu())
                err = (command_term.pos_ref_w - last["pos_t"]).norm(dim=-1)
                for name, mask in (("emulated", emulated), ("control", ~emulated)):
                    track_err[name].append(err[valid & mask].cpu())
                    if prev_actions is not None:
                        jumps[name].append((actions - prev_actions).norm(dim=-1)[valid & mask].cpu())

                roll, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_quat_w.torch[log_env : log_env + 1])
                row = obs["policy"][log_env].tolist()
                row += last["pos_t"][log_env].tolist() + last["pos_e"][log_env].tolist()
                row += last["vel_t"][log_env].tolist() + last["vel_e"][log_env].tolist()
                row += [math_utils.wrap_to_pi(roll).item(), math_utils.wrap_to_pi(pitch).item()]
                row += robot.data.root_ang_vel_b.torch[log_env].tolist() + actions[log_env].tolist()
                row_head = [f"{step * dt:.4f}", int(emulated[log_env])]

                obs, _, dones, _ = env.step(actions)
                writer.writerow(row_head + [f"{v:.6g}" for v in row] + [int(dones[log_env])])

                done = dones.bool()
                ends["emulated"] += int((done & emulated).sum())
                ends["control"] += int((done & ~emulated).sum())
                n_done = int(done.sum())
                if n_done:
                    delay_steps[done] = torch.randint(delay_min, delay_max + 1, (n_done,), device=device)
                prev_actions = actions
                fresh = done
                policy.reset(dones)

                step += 1
                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

        # ------------------------------------------------------------------ #
        # Summary
        # ------------------------------------------------------------------ #
        def describe(values: list[torch.Tensor]) -> str:
            values = torch.cat(values).numpy() if values else np.zeros(0)
            if values.size == 0:
                return "      -        -        -"
            return f"{values.mean():8.3f} {np.percentile(values, 95):8.3f} {values.max():8.3f}"

        n_control = args_cli.control_envs
        lines = [
            "=" * 72,
            f"Mouth estimate from stale attitude and fresh encoders — {args_cli.task}",
            f"checkpoint: {resume_path}",
            f"{step} steps ({step * dt:.1f} s), attitude {args_cli.attitude_delay_ms[0]:.0f}-"
            f"{args_cli.attitude_delay_ms[1]:.0f} ms old, body rate"
            f" {'fresh' if args_cli.fresh_body_rate else 'as old'}, {num_envs - n_control} emulated /"
            f" {n_control} control envs",
            f"{'':34s} {'mean':>8s} {'p95':>8s} {'max':>8s}",
            f"{'phantom mouth vel, horiz. [m/s]':34s} {describe(phantom_vel)}",
            f"{'phantom mouth pos, horiz. [m]':34s} {describe(phantom_pos)}",
        ]
        for name in ("emulated", "control"):
            lines.append(f"{'true mouth error, ' + name + ' [m]':34s} {describe(track_err[name])}")
        for name in ("emulated", "control"):
            lines.append(f"{'action jump, ' + name:34s} {describe(jumps[name])}")
        lines += [
            f"episode ends (incl. time-outs): {ends['emulated']} emulated, {ends['control']} control",
            f"CSV (env {log_env}): {csv_path}",
            "=" * 72,
        ]
        print("\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
