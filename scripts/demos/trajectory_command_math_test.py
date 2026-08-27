# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the drone trajectory command math without launching the simulator.

The trajectory generator promises hard sup-bounds on reference speed and acceleration, which is
what makes the sampled references physically trackable by the M350. This script checks that
promise directly on a dense time grid, verifies that the analytic velocity and acceleration are
consistent with finite differences of the position, and reports the realized peak distributions so
the limits can be tuned.

Launch:
    ./isaaclab.sh -p scripts/demos/trajectory_command_math_test.py
    ./isaaclab.sh -p scripts/demos/trajectory_command_math_test.py --num_samples 20000 --device cuda
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from isaaclab_tasks.manager_based.drone_arl.mdp.commands.commands_cfg import DroneTrajectoryCommandCfg
from isaaclab_tasks.manager_based.drone_arl.mdp.commands.trajectory_command import (
    TrajectoryLimits,
    evaluate_trajectory,
    evaluate_yaw,
    sample_trajectory_params,
    sample_yaw_params,
)

parser = argparse.ArgumentParser(description="Trajectory command math validation (no simulator).")
parser.add_argument("--num_samples", type=int, default=10000, help="Number of trajectories to sample.")
parser.add_argument("--episode_length_s", type=float, default=20.0, help="Episode length to sweep over [s].")
parser.add_argument("--grid_dt", type=float, default=1e-3, help="Time step of the dense evaluation grid [s].")
parser.add_argument("--difficulty", type=float, default=1.0, help="Difficulty to sample at.")
parser.add_argument("--device", type=str, default="cpu", help="Torch device to run on.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument(
    "--output", type=str, default="trajectory_command_math_results.png", help="Path of the summary figure."
)
args_cli = parser.parse_args()


def _sweep(amplitude, omega, phase, center, times, chunk_envs: int = 256):
    """Evaluate a batch of trajectories over a dense time grid, in chunks to bound memory.

    Args:
        amplitude: Harmonic amplitudes [m], shape ``(num_envs, 3, num_harmonics)``.
        omega: Harmonic angular frequencies [rad/s], same shape as ``amplitude``.
        phase: Harmonic phase offsets [rad], same shape as ``amplitude``.
        center: Trajectory centres [m], shape ``(num_envs, 3)``.
        times: Evaluation grid [s], shape ``(num_times,)``.
        chunk_envs: Number of environments evaluated per chunk.

    Returns:
        Per-environment peak position excursion [m], speed [m/s] and acceleration [m/s^2], each of
        shape ``(num_envs,)``, plus the peak vertical speed and acceleration.
    """
    num_envs = amplitude.shape[0]
    peaks = {k: [] for k in ("pos", "vel", "acc", "vel_z", "acc_z")}
    for start in range(0, num_envs, chunk_envs):
        sl = slice(start, min(start + chunk_envs, num_envs))
        n = amplitude[sl].shape[0]
        pos, vel, acc = evaluate_trajectory(
            amplitude[sl], omega[sl], phase[sl], center[sl], times.unsqueeze(0).expand(n, -1)
        )
        peaks["pos"].append(torch.linalg.norm(pos - center[sl].unsqueeze(1), dim=-1).amax(dim=1))
        peaks["vel"].append(torch.linalg.norm(vel, dim=-1).amax(dim=1))
        peaks["acc"].append(torch.linalg.norm(acc, dim=-1).amax(dim=1))
        peaks["vel_z"].append(vel[..., 2].abs().amax(dim=1))
        peaks["acc_z"].append(acc[..., 2].abs().amax(dim=1))
    return {k: torch.cat(v) for k, v in peaks.items()}


def main():
    torch.manual_seed(args_cli.seed)
    device = args_cli.device
    n = args_cli.num_samples
    cfg = DroneTrajectoryCommandCfg()
    limits = TrajectoryLimits.from_ranges(cfg.ranges, args_cli.difficulty)

    print(f"\nSampling {n} trajectories at difficulty {args_cli.difficulty} on {device}")
    print(f"limits: {limits}\n")

    amplitude, omega, phase, center = sample_trajectory_params(
        num_envs=n,
        limits=limits,
        device=device,
        num_harmonics=cfg.num_harmonics,
        axis_weights=cfg.axis_weights,
        harmonic_multiplier_ranges=cfg.harmonic_multiplier_ranges,
        harmonic_amplitude_ranges=cfg.harmonic_amplitude_ranges,
        omega_min=cfg.omega_min,
    )

    failures = []

    # -- 1. the limit guarantee, checked on a dense grid
    times = torch.arange(0.0, args_cli.episode_length_s, args_cli.grid_dt, device=device)
    print(f"[1] Sweeping {len(times)} time steps per trajectory ...")
    peaks = _sweep(amplitude, omega, phase, center, times)
    tol = 1e-4
    for name, cap in (
        ("pos", limits.pos_cap),
        ("vel", limits.vel_cap),
        ("acc", limits.acc_cap),
        ("vel_z", limits.vel_z_cap),
        ("acc_z", limits.acc_z_cap),
    ):
        worst = peaks[name].max().item()
        ok = worst <= cap + tol
        print(f"    {'PASS' if ok else 'FAIL'}  max |{name}| = {worst:.6f}  <= cap {cap:.4f}")
        if not ok:
            failures.append(f"{name} exceeded its cap: {worst:.6f} > {cap:.4f}")

    # -- 2. analytic derivatives against central differences.
    #       Done in float64: at h = 1e-4 a float32 central difference of metre-scale positions is
    #       only accurate to ~eps/h ~ 1e-3, which says nothing about the formulas being checked.
    print("[2] Checking derivative consistency (float64) ...")
    h = 1e-4
    t0 = (torch.rand(n, device=device) * args_cli.episode_length_s).double()
    args = (amplitude.double(), omega.double(), phase.double(), center.double())
    _, vel, acc = evaluate_trajectory(*args, t0)
    pos_plus, vel_plus, _ = evaluate_trajectory(*args, t0 + h)
    pos_minus, vel_minus, _ = evaluate_trajectory(*args, t0 - h)
    vel_fd_error = ((pos_plus - pos_minus) / (2 * h) - vel).abs().max().item()
    acc_fd_error = ((vel_plus - vel_minus) / (2 * h) - acc).abs().max().item()
    for name, err in (("velocity", vel_fd_error), ("acceleration", acc_fd_error)):
        ok = err < 1e-5
        print(f"    {'PASS' if ok else 'FAIL'}  max {name} finite-difference error = {err:.3e}")
        if not ok:
            failures.append(f"{name} disagrees with finite differences: {err:.3e}")

    # -- 3. every trajectory starts at the environment origin
    print("[3] Checking p(0) == 0 ...")
    pos_0, _, _ = evaluate_trajectory(*args, torch.zeros(n, device=device, dtype=torch.float64))
    start_error = pos_0.abs().max().item()
    ok = start_error < 1e-6
    print(f"    {'PASS' if ok else 'FAIL'}  max |p(0)| = {start_error:.3e}")
    if not ok:
        failures.append(f"trajectories do not start at the origin: {start_error:.3e}")

    # -- 4. yaw rate bound of the sweep mode
    print("[4] Checking the yaw sweep rate bound ...")
    yaw_params = sample_yaw_params(n, "sweep", device, cfg.yaw_amplitude_range, cfg.yaw_omega_range, cfg.yaw_max_rate)
    yaw_rate_peak = 0.0
    for t in torch.linspace(0.0, args_cli.episode_length_s, 2001, device=device):
        _, yaw_rate = evaluate_yaw(*yaw_params, t.expand(n))
        yaw_rate_peak = max(yaw_rate_peak, yaw_rate.abs().max().item())
    ok = yaw_rate_peak <= cfg.yaw_max_rate + tol
    print(f"    {'PASS' if ok else 'FAIL'}  max |yaw rate| = {yaw_rate_peak:.6f} <= {cfg.yaw_max_rate}")
    if not ok:
        failures.append(f"yaw rate exceeded its bound: {yaw_rate_peak:.6f}")

    # -- 5. realized distributions: the sup-bounds are conservative, so report how much headroom
    #       is actually used. If the peaks cluster far below the caps, raise vel_rms_target rather
    #       than the caps themselves.
    vel_rms = torch.sqrt(0.5 * ((amplitude * omega) ** 2).sum(dim=(-1, -2)))
    print("\n[5] Realized distributions (mean / p50 / p95 / max, and % of cap at the mean):")
    for name, cap in (("pos", limits.pos_cap), ("vel", limits.vel_cap), ("acc", limits.acc_cap)):
        v = peaks[name]
        q = torch.quantile(v, torch.tensor([0.5, 0.95], device=device))
        print(
            f"    peak {name:4s}: {v.mean():6.3f} / {q[0]:6.3f} / {q[1]:6.3f} / {v.max():6.3f}"
            f"   ({100 * v.mean() / cap:.0f} % of cap {cap:.2f})"
        )
    print(f"    vel_rms  : {vel_rms.mean():6.3f} (target {limits.vel_rms_target:.2f})")

    # -- summary figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Drone trajectory command — {n} samples at difficulty {args_cli.difficulty}", fontsize=12)
    for ax, (name, cap, label) in zip(
        axes[0],
        (
            ("pos", limits.pos_cap, "peak excursion [m]"),
            ("vel", limits.vel_cap, "peak speed [m/s]"),
            ("acc", limits.acc_cap, "peak acceleration [m/s²]"),
        ),
    ):
        ax.hist(peaks[name].cpu().numpy(), bins=60, color="tab:blue", alpha=0.8)
        ax.axvline(cap, color="tab:red", linestyle="--", linewidth=1.2, label=f"cap {cap:.2f}")
        ax.set_xlabel(label)
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # sample paths, drawn from the first few trajectories
    num_paths = min(20, n)
    plot_times = torch.linspace(0.0, args_cli.episode_length_s, 1000, device=device)
    path, path_vel, _ = evaluate_trajectory(
        amplitude[:num_paths],
        omega[:num_paths],
        phase[:num_paths],
        center[:num_paths],
        plot_times.unsqueeze(0).expand(num_paths, -1),
    )
    path_np, t_np = path.cpu().numpy(), plot_times.cpu().numpy()

    axes[1, 0].set_title("xy paths", fontsize=10)
    for i in range(num_paths):
        axes[1, 0].plot(path_np[i, :, 0], path_np[i, :, 1], linewidth=0.9, alpha=0.8)
    axes[1, 0].plot(0.0, 0.0, "k+", markersize=10, label="spawn")
    axes[1, 0].set_xlabel("x [m]")
    axes[1, 0].set_ylabel("y [m]")
    axes[1, 0].set_aspect("equal", adjustable="datalim")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_title("altitude vs time", fontsize=10)
    for i in range(num_paths):
        axes[1, 1].plot(t_np, path_np[i, :, 2], linewidth=0.9, alpha=0.8)
    axes[1, 1].set_xlabel("t [s]")
    axes[1, 1].set_ylabel("z [m]")

    axes[1, 2].set_title("speed vs time", fontsize=10)
    speed_np = torch.linalg.norm(path_vel, dim=-1).cpu().numpy()
    for i in range(num_paths):
        axes[1, 2].plot(t_np, speed_np[i], linewidth=0.9, alpha=0.8)
    axes[1, 2].axhline(limits.vel_cap, color="tab:red", linestyle="--", linewidth=1.2, label="cap")
    axes[1, 2].set_xlabel("t [s]")
    axes[1, 2].set_ylabel("‖v‖ [m/s]")
    axes[1, 2].legend(fontsize=8)

    for ax in axes[1]:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args_cli.output, dpi=120)
    print(f"\nSaved figure to {args_cli.output}")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
