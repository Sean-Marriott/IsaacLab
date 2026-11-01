# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a Matrice trajectory policy with the velocity setpoint reaching the controller late.

Purpose
-------
To check whether a delay between the policy and the flight controller makes the chainsaw tool
policy rock in sim, as it does on the vehicle. The policy runs at 10 Hz and the Lee velocity
controller at the 100 Hz physics rate; in sim the controller sees a new setpoint on the very first
physics step after the policy computes it.

Emulation
---------
The Lee controller's :meth:`compute` is wrapped so that, on every physics step, it is given the
velocity setpoint the action term produced ``--action_delay_ms`` earlier, instead of the current one.
The delay resolves to the physics step (10 ms), so the default 30 ms is 3 steps. On the first
physics step of an episode the history is filled with the current setpoint, so there is no carry-over
from the previous episode. Only the controller input is delayed: the command observations
(``prev_commands``) and ``last_action`` still show what the policy sent, as on the vehicle.

The first ``--control_envs`` environments run without the delay, as a reference in the same run.

Outputs
-------
* Console: on exit, the roll and pitch rate, the tilt, the action jump and the true tracking error
  over the delayed and control environments, and their episode ends.
* ``<checkpoint_dir>/action_delay_env<N>.csv`` (or ``--csv``): one row per physics step for
  ``--log_env``: the setpoint sent and the setpoint applied, roll, pitch and body rates, the tracked
  frame position error and the policy step within the episode.

Usage::

    # watch it, recurrent policy, half the environments undelayed
    uv run python scripts/reinforcement_learning/rsl_rl/play_action_delay.py \\
        --agent rsl_rl_rnn_cfg_entry_point --checkpoint /path/to/model_1999.pt --viz kit \\
        --num_envs 16 --control_envs 8

    # headless, 60 s, 50 ms of delay
    uv run python scripts/reinforcement_learning/rsl_rl/play_action_delay.py \\
        --agent rsl_rl_rnn_cfg_entry_point --checkpoint /path/to/model_1999.pt \\
        --num_envs 64 --control_envs 32 --num_steps 600 --action_delay_ms 50
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
from isaaclab_rl.entrypoints.backends.play_rsl_rl import _ObsCsvLogger, _resolve_checkpoint
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
SETPOINT = ("vx", "vy", "vz", "yaw_rate")

parser = argparse.ArgumentParser(
    description="Play a Matrice trajectory policy with the velocity setpoint delayed before the controller."
)
add_common_play_args(
    parser, agent_default="rsl_rl_cfg_entry_point", agent_help="Name of the RL agent configuration entry point."
)
parser.set_defaults(task="IsaacContrib-TrackTrajectory-Matrice-ChainsawTool")
parser.add_argument(
    "--action_delay_ms", type=float, default=30.0, help="Delay of the setpoint into the controller [ms]."
)
parser.add_argument("--action_name", type=str, default="velocity_command", help="Name of the action term.")
parser.add_argument(
    "--control_envs", type=int, default=0, help="Number of leading environments that run without the delay."
)
parser.add_argument("--num_steps", type=int, default=None, help="Policy steps to run; until Ctrl+C if omitted.")
parser.add_argument(
    "--log_env",
    type=int,
    default=None,
    help="Environment written to the CSV (default: the first delayed one).",
)
parser.add_argument("--csv", type=str, default=None, help="CSV path for --log_env.")
parser.add_argument(
    "--obs_csv",
    type=str,
    default=None,
    help="Also write the policy observations and actions of --log_env, one row per policy step (as play.py).",
)
parser.add_argument(
    "--fresh_obs",
    action="store_true",
    default=False,
    help="Remove the measurement delay (lag and held frames) from the policy observations.",
)
parser.add_argument(
    "--no_obs_noise", action="store_true", default=False, help="Remove the noise from the policy observations."
)
parser.add_argument("--no_kicks", action="store_true", default=False, help="Disable the play-mode payload swing kicks.")
parser.add_argument(
    "--command_delay_ms",
    type=float,
    default=None,
    help="Pin the velocity action's own command delay (command_delay_range_s) in every environment [ms].",
)
parser.add_argument(
    "--k_vel_xy", type=float, default=None, help="Pin the lateral K_vel of the velocity controller in every episode."
)
parser.add_argument(
    "--k_vel_z", type=float, default=None, help="Pin the vertical K_vel of the velocity controller in every episode."
)
parser.add_argument(
    "--k_vel_int_xy", type=float, default=None, help="Pin the lateral K_vel_int of the velocity controller."
)
parser.add_argument(
    "--k_vel_int_z", type=float, default=None, help="Pin the vertical K_vel_int of the velocity controller."
)
parser.add_argument(
    "--swing_term", type=str, default="swing_suppression", help="Reward term giving the payload swing amplitude."
)
parser.add_argument(
    "--nyquist_window", type=int, default=20, help="Policy steps per window of the period-2 amplitude (even)."
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser, sys.argv[1:])
set_hydra_args(remaining_args)


def apply_condition_overrides(env_cfg) -> None:
    """Apply --fresh_obs, --no_obs_noise and --no_kicks to the environment config."""
    for term in vars(env_cfg.observations.policy).values():
        if not (hasattr(term, "tensor_processors") and hasattr(term, "noise")):
            continue
        if args_cli.fresh_obs:
            term.tensor_processors = None
        if args_cli.no_obs_noise:
            term.noise = None
    if args_cli.fresh_obs:
        print("[INFO] Policy observations without measurement delay.")
    if args_cli.no_obs_noise:
        print("[INFO] Policy observations without noise.")
    action_cfg = env_cfg.actions.velocity_command
    if args_cli.command_delay_ms is not None and hasattr(action_cfg, "command_delay_range_s"):
        action_cfg.command_delay_range_s = (args_cli.command_delay_ms * 1e-3,) * 2
        print(f"[INFO] Action term command delay pinned at {args_cli.command_delay_ms:.0f} ms.")
    if args_cli.no_kicks and getattr(env_cfg.events, "kick_chainsaw_swing", None) is not None:
        env_cfg.events.kick_chainsaw_swing = None
        print("[INFO] Payload swing kicks disabled.")


def main():  # noqa: C901 (one rollout and its instrumentation, kept together)
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        import isaaclab.utils.math as math_utils

        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        apply_env_overrides(args_cli, env_cfg)
        apply_condition_overrides(env_cfg)
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
        action_term = base_env.action_manager.get_term(args_cli.action_name)
        all_ids = torch.arange(num_envs, device=device)
        delayed = all_ids >= args_cli.control_envs
        log_env = args_cli.log_env if args_cli.log_env is not None else min(args_cli.control_envs, num_envs - 1)

        # ------------------------------------------------------------------ #
        # Setpoint history, pushed on every controller call (once per physics step)
        # ------------------------------------------------------------------ #
        physics_dt = base_env.physics_dt
        delay_steps = round(args_cli.action_delay_ms * 1e-3 / physics_dt)
        history_len = delay_steps + 1
        setpoint_hist = torch.zeros(history_len, num_envs, 4, device=device)
        head = 0
        # Environments whose history must be refilled on the next controller call.
        fresh = torch.ones(num_envs, dtype=torch.bool, device=device)
        # The setpoint sent and the one applied on the last physics step, for the CSV.
        last = {}

        controller = action_term._lc
        controller_compute = controller.compute

        def delayed_compute(command: torch.Tensor) -> torch.Tensor:
            nonlocal head
            head = (head + 1) % history_len
            setpoint_hist[head] = command
            setpoint_hist[:, fresh] = command[fresh]
            fresh[:] = False
            old = setpoint_hist[(head - delay_steps) % history_len]
            applied = torch.where(delayed.unsqueeze(-1), old, command)
            last.update(sent=command.clone(), applied=applied.clone())
            return controller_compute(applied)

        controller.compute = delayed_compute

        # (attribute, axes, value) for every gain pinned on the command line
        pins = [
            (attr, axes, value)
            for attr, axes, value in (
                ("K_vel_current", slice(0, 2), args_cli.k_vel_xy),
                ("K_vel_current", slice(2, 3), args_cli.k_vel_z),
                ("K_vel_int_current", slice(0, 2), args_cli.k_vel_int_xy),
                ("K_vel_int_current", slice(2, 3), args_cli.k_vel_int_z),
            )
            if value is not None
        ]
        if pins:
            randomize_params = controller._randomize_params

            def apply_pins(env_ids):
                for attr, axes, value in pins:
                    getattr(controller, attr)[env_ids, axes] = value
                if any(attr == "K_vel_int_current" and value > 0.0 for attr, _, value in pins):
                    controller._integral_enabled = True

            def randomize_with_pinned_gains(env_ids):
                randomize_params(env_ids)
                apply_pins(env_ids)

            controller._randomize_params = randomize_with_pinned_gains
            apply_pins(slice(None))
            print(f"[INFO] Pinned gains: {[(attr, axes.start, value) for attr, axes, value in pins]}.")

        term_reset = action_term.reset

        def reset_with_history(env_ids=None):
            term_reset(env_ids)
            if env_ids is None:
                fresh[:] = True
            else:
                fresh[env_ids] = True

        action_term.reset = reset_with_history

        print(
            f"[INFO] Velocity setpoint delayed by {args_cli.action_delay_ms:.0f} ms ({delay_steps} physics steps of"
            f" {physics_dt * 1e3:.0f} ms) in {int(delayed.sum())}/{num_envs} environments."
        )
        if abs(delay_steps * physics_dt - args_cli.action_delay_ms * 1e-3) > 1e-6:
            print(f"[INFO] Delay rounded to {delay_steps * physics_dt * 1e3:.0f} ms.")

        # ------------------------------------------------------------------ #
        # CSV, one row per physics step
        # ------------------------------------------------------------------ #
        csv_path = os.path.abspath(args_cli.csv or os.path.join(log_dir, f"action_delay_env{log_env}.csv"))
        csv_file = open(csv_path, "w", newline="")  # noqa: SIM115 (closed by cleanup)
        cleanup.callback(csv_file.close)
        writer = csv.writer(csv_file)
        writer.writerow(
            ["time", "delayed"]
            + [f"sent_{a}" for a in SETPOINT]
            + [f"applied_{a}" for a in SETPOINT]
            + ["roll", "pitch"]
            + [f"ang_vel_b_{a}" for a in XYZ]
            + [f"pos_err_{a}" for a in XYZ]
            + ["episode_step"]
        )
        physics_step = 0

        # Per-physics-step rates and tilt, sampled after every scene update.
        rates = {"delayed": [], "control": []}
        tilts = {"delayed": [], "control": []}
        scene_update = base_env.scene.update

        def recording_update(dt: float) -> None:
            nonlocal physics_step
            scene_update(dt)
            if not last:
                return
            w_b = robot.data.root_link_ang_vel_b.torch
            roll, pitch, _ = math_utils.euler_xyz_from_quat(robot.data.root_link_quat_w.torch)
            roll, pitch = math_utils.wrap_to_pi(roll), math_utils.wrap_to_pi(pitch)
            rp_rate = w_b[:, :2].norm(dim=-1)
            tilt = torch.stack((roll, pitch), dim=-1).norm(dim=-1)
            for name, mask in (("delayed", delayed), ("control", ~delayed)):
                rates[name].append(rp_rate[mask].cpu())
                tilts[name].append(tilt[mask].cpu())

            pos_t, _, _ = command_term.tracked_frame_state()
            err = command_term.pos_ref_w[log_env] - pos_t[log_env]
            row = last["sent"][log_env].tolist() + last["applied"][log_env].tolist()
            row += [roll[log_env].item(), pitch[log_env].item()] + w_b[log_env].tolist() + err.tolist()
            writer.writerow(
                [f"{physics_step * physics_dt:.4f}", int(delayed[log_env])]
                + [f"{v:.6g}" for v in row]
                + [int(base_env.episode_length_buf[log_env])]
            )
            physics_step += 1

        base_env.scene.update = recording_update

        # ------------------------------------------------------------------ #
        # Rollout
        # ------------------------------------------------------------------ #
        track_err = {"delayed": [], "control": []}
        swing = {"delayed": [], "control": []}
        swing_term = None
        if args_cli.swing_term in base_env.reward_manager.active_terms:
            swing_term = base_env.reward_manager.get_term_cfg(args_cli.swing_term).func
        jumps = {"delayed": [], "control": []}
        ends = {"delayed": 0, "control": 0}
        prev_actions = None
        just_reset = torch.ones(num_envs, dtype=torch.bool, device=device)

        # Period-2 amplitude: the component that flips sign every policy step (5 Hz at a 10 Hz
        # policy), |mean((-1)^k x_k)| over the last --nyquist_window policy steps of an episode.
        # A smooth signal gives ~0; x_k = a (-1)^k gives a.
        nyq_signals = ("roll rate [deg/s]", "pitch rate [deg/s]", "action vx", "action vy", "action vz")
        nyq_window = args_cli.nyquist_window + args_cli.nyquist_window % 2
        nyq_hist = torch.zeros(nyq_window, num_envs, len(nyq_signals), device=device)
        nyq_sign = torch.tensor([(-1.0) ** k for k in range(nyq_window)], device=device).view(-1, 1, 1)
        nyq_count = torch.zeros(num_envs, dtype=torch.long, device=device)
        nyq = {"delayed": [], "control": []}

        obs_csv = None
        if args_cli.obs_csv:
            obs_csv = _ObsCsvLogger(args_cli.obs_csv, base_env, log_env)
            cleanup.callback(obs_csv.close)

        obs = env.get_observations()
        step = 0
        dt = base_env.step_dt
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while args_cli.num_steps is None or step < args_cli.num_steps:
                start_time = time.time()
                actions = policy(obs)

                valid = ~just_reset
                pos_t, _, _ = command_term.tracked_frame_state()
                err = (command_term.pos_ref_w - pos_t).norm(dim=-1)
                if swing_term is not None:
                    amplitude, _ = swing_term.swing_amplitude(base_env)
                for name, mask in (("delayed", delayed), ("control", ~delayed)):
                    track_err[name].append(err[valid & mask].cpu())
                    if swing_term is not None:
                        swing[name].append(amplitude[valid & mask].cpu())
                    if prev_actions is not None:
                        jumps[name].append((actions - prev_actions).norm(dim=-1)[valid & mask].cpu())

                # -- period-2 amplitude, on the state the action was computed from
                sample = torch.cat((robot.data.root_link_ang_vel_b.torch[:, :2] * 180.0 / np.pi, actions[:, :3]), -1)
                nyq_hist[step % nyq_window] = sample
                nyq_count[:] = torch.where(just_reset, 1, nyq_count + 1)
                full = nyq_count >= nyq_window
                amp = (nyq_hist * nyq_sign).mean(dim=0).abs()
                for name, mask in (("delayed", delayed), ("control", ~delayed)):
                    nyq[name].append(amp[full & mask].cpu())

                if obs_csv is not None:
                    obs_csv.log(actions)
                obs, _, dones, _ = env.step(actions)
                if obs_csv is not None:
                    obs_csv.after_step()

                done = dones.bool()
                ends["delayed"] += int((done & delayed).sum())
                ends["control"] += int((done & ~delayed).sum())
                prev_actions = actions
                just_reset = done
                policy.reset(dones)

                step += 1
                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

        # ------------------------------------------------------------------ #
        # Summary
        # ------------------------------------------------------------------ #
        def describe(values: list[torch.Tensor], scale: float = 1.0) -> str:
            values = torch.cat(values).numpy() * scale if values else np.zeros(0)
            if values.size == 0:
                return "      -        -        -        -"
            rms = np.sqrt(np.mean(values**2))
            return f"{values.mean():8.3f} {rms:8.3f} {np.percentile(values, 95):8.3f} {values.max():8.3f}"

        n_control = args_cli.control_envs
        rad2deg = 180.0 / np.pi
        lines = [
            "=" * 80,
            f"Velocity setpoint delayed into the controller — {args_cli.task}",
            f"checkpoint: {resume_path}",
            f"{step} steps ({step * dt:.1f} s), delay {delay_steps * physics_dt * 1e3:.0f} ms,"
            f" {num_envs - n_control} delayed / {n_control} control envs",
            f"{'':34s} {'mean':>8s} {'rms':>8s} {'p95':>8s} {'max':>8s}",
        ]
        for name in ("delayed", "control"):
            lines.append(f"{'roll/pitch rate, ' + name + ' [deg/s]':34s} {describe(rates[name], rad2deg)}")
        for name in ("delayed", "control"):
            lines.append(f"{'tilt, ' + name + ' [deg]':34s} {describe(tilts[name], rad2deg)}")
        for name in ("delayed", "control"):
            lines.append(f"{'tracked frame error, ' + name + ' [m]':34s} {describe(track_err[name])}")
        for name in ("delayed", "control"):
            if swing[name]:
                lines.append(f"{'payload swing amplitude, ' + name + ' [deg]':34s} {describe(swing[name], rad2deg)}")
        for name in ("delayed", "control"):
            lines.append(f"{'action jump, ' + name:34s} {describe(jumps[name])}")
        lines.append(f"period-2 (5 Hz) amplitude over {nyq_window} policy steps:")
        for name in ("delayed", "control"):
            amps = torch.cat(nyq[name]).numpy() if nyq[name] else np.zeros((0, len(nyq_signals)))
            for i, label in enumerate(nyq_signals):
                lines.append(f"{'  ' + label + ', ' + name:34s} {describe([torch.from_numpy(amps[:, i])])}")
        lines += [
            f"episode ends (incl. time-outs): {ends['delayed']} delayed, {ends['control']} control",
            f"CSV (env {log_env}, per physics step): {csv_path}",
            "=" * 80,
        ]
        print("\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
