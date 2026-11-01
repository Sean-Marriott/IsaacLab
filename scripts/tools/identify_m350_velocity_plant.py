# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Identify the lateral velocity plant a Matrice trajectory policy is trained on, as a low-order model.

Purpose
-------
To compare the plant behind the velocity action -- the Lee velocity controller, the attitude loop,
the motor lags and the airframe -- with a plant identified elsewhere (another simulator, the vehicle)
in the same terms.

Method
------
The task's velocity action is driven directly, without a policy, at the 100 Hz physics rate. The
command is a random staircase changed only every 10 physics steps, as a 10 Hz policy would. Every
randomized term is fixed at its nominal value, the gust is off and the gains are the centres of their
ranges. ``--envs_per_axis`` environments excite body x and as many body y, each with its own sequence.

Per lateral axis the response is fitted with::

    tilt_cmd(t) = atan(K_v (v_cmd(t - d) - v(t - d)) / g)
    tilt''      = w^2 (tilt_cmd - tilt) - 2 zeta w tilt'
    v'          = g tan(tilt)

i.e. a velocity gain commanding tilt, a second-order attitude response and a pure delay ``d`` on the
tilt command (command and feedback alike). The model is run open loop from the recorded command and
fitted to the recorded velocity and tilt; ``d`` is searched on the physics-step grid.

Usage::

    uv run python scripts/tools/identify_m350_velocity_plant.py
    uv run python scripts/tools/identify_m350_velocity_plant.py --task IsaacContrib-TrackTrajectory-Matrice
"""

import argparse
import math

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

G = 9.81

parser = argparse.ArgumentParser(description="Identify the Matrice lateral velocity plant as a low-order model.")
parser.add_argument("--task", type=str, default="IsaacContrib-TrackTrajectory-Matrice-ChainsawTool", help="Task.")
parser.add_argument("--duration", type=float, default=60.0, help="Excitation length per environment [s].")
parser.add_argument("--envs_per_axis", type=int, default=4, help="Environments per lateral axis.")
parser.add_argument("--amplitude", type=float, default=0.5, help="Peak velocity command [m/s].")
parser.add_argument("--hold", type=float, nargs=2, default=(0.1, 1.5), help="Range of command hold times [s].")
parser.add_argument("--max_delay_ms", type=float, default=100.0, help="Largest delay searched [ms].")
parser.add_argument("--body", type=str, default="body", help="Airframe body; falls back to the root if absent.")
parser.add_argument("--k_vel_xy", type=float, default=None, help="Lateral K_vel instead of the nominal.")
parser.add_argument("--seed", type=int, default=0, help="Seed of the command sequences.")
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


def freeze_randomization(env_cfg):
    """Fix every randomized term at its nominal value, turn the gust and kicks off and drop terminations."""
    for name in list(vars(env_cfg.events)):
        term = getattr(env_cfg.events, name)
        if term is None or not hasattr(term, "func"):
            continue
        func_name = getattr(term.func, "__name__", str(term.func))
        if term.mode == "startup" and "randomize_rigid_body" in func_name:
            setattr(env_cfg.events, name, None)
            continue
        if term.mode == "interval":
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


def staircase(rng: np.random.Generator, n_steps: int, dt: float) -> np.ndarray:
    """A random staircase changed only on 10 Hz boundaries."""
    out = np.zeros(n_steps)
    k = 0
    while k < n_steps:
        hold = max(10, int(round(rng.uniform(*args_cli.hold) / dt / 10)) * 10)
        out[k : k + hold] = rng.uniform(-args_cli.amplitude, args_cli.amplitude)
        k += hold
    return out


def simulate(params, delay_steps: int, u: np.ndarray, v0: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Run the low-order model open loop from command ``u``; return velocity and tilt."""
    k_v, w, zeta = params
    n = len(u)
    v = np.zeros(n)
    th = np.zeros(n)
    v[0] = v0
    th_dot = 0.0
    for k in range(n - 1):
        j = max(k - delay_steps, 0)
        th_cmd = math.atan(k_v * (u[j] - v[j]) / G)
        th_dot += dt * (w * w * (th_cmd - th[k]) - 2.0 * zeta * w * th_dot)
        th[k + 1] = th[k] + dt * th_dot
        v[k + 1] = v[k] + dt * G * math.tan(th[k + 1])
    return v, th


def fit_axis(runs: list[dict], dt: float) -> dict:
    """Fit K_v, w, zeta per delay on the grid, keeping the best."""
    from scipy.optimize import least_squares

    def residual(params, delay_steps):
        res = []
        for r in runs:
            v, th = simulate(params, delay_steps, r["u"], r["v"][0], dt)
            # Velocity in m/s and tilt in rad weighted so that 1 deg counts about like 3 cm/s.
            res.append(v - r["v"])
            res.append(2.0 * (th - r["tilt"]))
        return np.concatenate(res)

    best = None
    for d in range(0, int(round(args_cli.max_delay_ms * 1e-3 / dt)) + 1):
        sol = least_squares(residual, x0=(4.0, 10.0, 0.7), args=(d,), bounds=((0.5, 2.0, 0.1), (20.0, 60.0, 3.0)))
        cost = float(np.sqrt(np.mean(sol.fun**2)))
        if best is None or cost < best["rms"]:
            best = {"K_v": sol.x[0], "w": sol.x[1], "zeta": sol.x[2], "delay_ms": d * dt * 1e3, "rms": cost}
    v_rms, th_rms = [], []
    for r in runs:
        v, th = simulate(
            (best["K_v"], best["w"], best["zeta"]), int(round(best["delay_ms"] * 1e-3 / dt)), r["u"], r["v"][0], dt
        )
        v_rms.append(np.sqrt(np.mean((v - r["v"]) ** 2)))
        th_rms.append(np.sqrt(np.mean((th - r["tilt"]) ** 2)))
    best["v_rms"] = float(np.mean(v_rms))
    best["tilt_rms_deg"] = float(np.degrees(np.mean(th_rms)))
    best["v_std"] = float(np.mean([r["v"].std() for r in runs]))
    return best


def main():
    env_cfg, _ = resolve_task_config(args_cli.task, None, play_mode=False, overrides=hydra_args)
    dt = env_cfg.sim.dt
    n_steps = int(round(args_cli.duration / dt))
    warmup = int(round(2.0 / dt))
    num_envs = 2 * args_cli.envs_per_axis

    with launch_simulation(env_cfg, args_cli):
        import isaaclab.utils.math as math_utils

        env_cfg.scene.num_envs = num_envs
        env_cfg.scene.env_spacing = 50.0
        env_cfg.decimation = 1
        env_cfg.episode_length_s = (n_steps + warmup) * dt + 5.0
        freeze_randomization(env_cfg)
        if hasattr(env_cfg, "curriculum"):
            for name in list(vars(env_cfg.curriculum)):
                setattr(env_cfg.curriculum, name, None)

        env = gym.make(args_cli.task, cfg=env_cfg)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        device = base_env.device
        term = base_env.action_manager.get_term("velocity_command")
        lc = term._lc
        ctrl_cfg = term.cfg.controller_cfg
        nominal = {
            "K_vel_current": np.mean(ctrl_cfg.K_vel_range, axis=0),
            "K_vel_int_current": np.mean(ctrl_cfg.K_vel_int_range, axis=0),
            "K_rot_current": np.mean(ctrl_cfg.K_rot_range, axis=0),
            "K_angvel_current": np.mean(ctrl_cfg.K_angvel_range, axis=0),
        }
        if args_cli.k_vel_xy is not None:
            nominal["K_vel_current"][:2] = args_cli.k_vel_xy
        gains = ", ".join(f"{k[:-8]} {v.round(2)}" for k, v in nominal.items())
        print(f"[INFO] {args_cli.task}: nominal gains {gains}")

        rng = np.random.default_rng(args_cli.seed)
        cmd = np.zeros((num_envs, n_steps, 4))
        for e in range(num_envs):
            cmd[e, :, 0 if e < args_cli.envs_per_axis else 1] = staircase(rng, n_steps, dt)
        cmd_t = torch.tensor(cmd, dtype=torch.float32, device=device)
        scale = torch.tensor([term.cfg.max_velocity] * 3 + [term.cfg.max_yaw_rate], dtype=torch.float32, device=device)

        body_id = robot.body_names.index(args_cli.body) if args_cli.body in robot.body_names else 0
        print(f"[INFO] Measuring body '{robot.body_names[body_id]}'")

        log_v = np.zeros((num_envs, n_steps, 2))
        log_tilt = np.zeros((num_envs, n_steps, 2))
        with torch.inference_mode():
            env.reset()
            for attr, value in nominal.items():
                getattr(lc, attr)[:] = torch.tensor(value, dtype=torch.float32, device=device)
            hover = torch.zeros(num_envs, 4, device=device)
            for _ in range(warmup):
                env.step(hover)
            for k in range(n_steps):
                env.step(cmd_t[:, k] / scale)
                q = robot.data.body_link_quat_w.torch[:, body_id]
                roll, pitch, yaw = (math_utils.wrap_to_pi(angle) for angle in math_utils.euler_xyz_from_quat(q))
                v_w = robot.data.body_link_lin_vel_w.torch[:, body_id]
                cy, sy = torch.cos(yaw), torch.sin(yaw)
                # Velocity in the yaw frame, and the tilt that accelerates along each of its axes.
                log_v[:, k, 0] = (cy * v_w[:, 0] + sy * v_w[:, 1]).cpu().numpy()
                log_v[:, k, 1] = (-sy * v_w[:, 0] + cy * v_w[:, 1]).cpu().numpy()
                log_tilt[:, k, 0] = pitch.cpu().numpy()
                log_tilt[:, k, 1] = (-roll).cpu().numpy()

        lines = ["=" * 78, f"Lateral velocity plant of {args_cli.task} (nominal gains, {args_cli.duration:.0f} s/env)"]
        lines.append(
            f"{'axis':6s} {'K_v [1/s]':>10s} {'w [rad/s]':>10s} {'zeta':>6s} {'delay [ms]':>11s}"
            f" {'v rms [m/s]':>12s} {'(v std)':>8s} {'tilt rms [deg]':>15s}"
        )
        for a, name in enumerate(("x", "y")):
            envs = range(a * args_cli.envs_per_axis, (a + 1) * args_cli.envs_per_axis)
            runs = [{"u": cmd[e, :, a], "v": log_v[e, :, a], "tilt": log_tilt[e, :, a]} for e in envs]
            f = fit_axis(runs, dt)
            lines.append(
                f"{name:6s} {f['K_v']:10.2f} {f['w']:10.2f} {f['zeta']:6.2f} {f['delay_ms']:11.0f}"
                f" {f['v_rms']:12.3f} {f['v_std']:8.3f} {f['tilt_rms_deg']:15.2f}"
            )
        lines.append("=" * 78)
        print("\n" + "\n".join(lines) + "\n")
        env.close()


if __name__ == "__main__":
    main()
