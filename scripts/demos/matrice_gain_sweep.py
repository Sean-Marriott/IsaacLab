# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compare the four corners + nominal of the MatriceDirectVelEnvCfg gain ranges.

Runs the same velocity step sequence for each of the five gain configurations
and overlays the results so the gain-range sanity can be assessed before training.

Configurations tested (xy gains shown; z gains scaled proportionally):
  Nominal   — K_rot=32.33, K_angvel=15.73, K_vel=3.49  (ζ = 0.85)
  Min-all   — K_rot=29.1,  K_angvel=14.2,  K_vel=3.1   (ζ ~ 0.85)
  Max-all   — K_rot=35.6,  K_angvel=17.3,  K_vel=3.8   (ζ ~ 0.85)
  Worst-ζ   — K_rot=35.6,  K_angvel=14.2,  K_vel=3.8   (ζ ~ 0.73, most oscillatory)
  Best-ζ    — K_rot=29.1,  K_angvel=17.3,  K_vel=3.1   (ζ ~ 0.99, most damped)

Actual ζ values are computed from real PhysX inertia at runtime and printed
in the legend.  The plot is saved to matrice_gain_sweep_results.png.

Launch:
    ./isaaclab.sh -p scripts/demos/matrice_gain_sweep.py
    ./isaaclab.sh -p scripts/demos/matrice_gain_sweep.py --viz kit
"""

import argparse
import copy
import math
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Matrice controller gain-range sweep.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.usd
from pxr import Gf, UsdLux

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sim import SimulationContext

from isaaclab_contrib.assets import Multirotor
from isaaclab_contrib.controllers.lee_velocity_control import LeeVelController
from isaaclab_contrib.controllers.lee_velocity_control_cfg import LeeVelControllerCfg
from isaaclab_assets.robots.arl_robot_1 import MATRICE_CFG

# ── Gain configurations ────────────────────────────────────────────────────────
# Columns: (label, K_rot_xy, K_angvel_xy, K_vel_xy, K_rot_z, K_angvel_z, K_vel_z)
# Drawn from env_cfg.py ranges: K_rot (24.9–30.5), K_angvel (13.2–16.1),
#   K_vel_xy (2.9–3.5), K_rot_z (10.0–12.2), K_angvel_z (5.3–6.4), K_vel_z (1.7–2.1)
_CONFIGS = [
    ("Nominal", 32.33, 15.73, 3.49, 12.93, 6.31, 2.10),
    ("Min-all", 29.1,  14.2,  3.1,  11.6,  5.7,  1.9 ),
    ("Max-all", 35.6,  17.3,  3.8,  14.2,  6.9,  2.3 ),
    ("Worst-ζ", 35.6,  14.2,  3.8,  14.2,  5.7,  2.3 ),   # high K_rot, low K_angvel
    ("Best-ζ",  29.1,  17.3,  3.1,  11.6,  6.9,  1.9 ),   # low K_rot, high K_angvel
]

_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

# ── Command sequence ───────────────────────────────────────────────────────────
_SEQUENCE = [
    (2.0, [0.0, 0.0, 0.0, 0.0], "Hover"),
    (4.0, [1.0, 0.0, 0.0, 0.0], "+vx"),
    (2.0, [0.0, 0.0, 0.0, 0.0], "Hover"),
    (4.0, [0.0, 1.0, 0.0, 0.0], "+vy"),
    (2.0, [0.0, 0.0, 0.0, 0.0], "Hover"),
    (3.0, [0.0, 0.0, 0.0, 0.4], "Yaw"),
    (1.0, [0.0, 0.0, 0.0, 0.0], "Hover"),
]

_DT = 0.01
_INIT_HEIGHT = 10.0


@dataclass
class RunResult:
    label: str
    zeta: float
    t: list
    vx_cmd: list
    vx_act: list
    vy_cmd: list
    vy_act: list
    pitch: list
    roll: list
    yaw_rate: list


def _build_commands(sequence: list, device: str) -> tuple[torch.Tensor, list[tuple[int, str]], int]:
    total = sum(int(dur / _DT) for dur, _, _ in sequence)
    cmds = torch.zeros((total, 4), device=device)
    boundaries = []
    idx = 0
    for dur, cmd, label in sequence:
        n = int(dur / _DT)
        cmds[idx : idx + n] = torch.tensor(cmd, device=device)
        boundaries.append((idx, label))
        idx += n
    return cmds, boundaries, total


def _reset_robot(robot: Multirotor, device: str) -> None:
    """Reset robot to hover pose without calling robot.reset() (avoids warp indexing bug)."""
    init_pose = torch.tensor([[0.0, 0.0, _INIT_HEIGHT, 0.0, 0.0, 0.0, 1.0]], device=device)
    robot.write_root_pose_to_sim_index(root_pose=init_pose, env_ids=None)
    robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=device), env_ids=None)
    default_pos = robot.data.default_joint_pos.torch.clone()
    robot.write_joint_position_to_sim_index(position=default_pos, joint_ids=None, env_ids=None)
    robot.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(default_pos), joint_ids=None, env_ids=None
    )


def _set_gains(
    controller: LeeVelController,
    K_rot_xy: float, K_angvel_xy: float, K_vel_xy: float,
    K_rot_z: float,  K_angvel_z: float,  K_vel_z: float,
    device: str,
) -> None:
    controller.K_rot_current[:]    = torch.tensor([[K_rot_xy,   K_rot_xy,   K_rot_z]],   device=device)
    controller.K_angvel_current[:] = torch.tensor([[K_angvel_xy, K_angvel_xy, K_angvel_z]], device=device)
    controller.K_vel_current[:]    = torch.tensor([[K_vel_xy,   K_vel_xy,   K_vel_z]],   device=device)


def _run_config(
    label: str,
    K_rot_xy: float, K_angvel_xy: float, K_vel_xy: float,
    K_rot_z: float,  K_angvel_z: float,  K_vel_z: float,
    I_att: float,
    robot: Multirotor,
    controller: LeeVelController,
    alloc_pinv: torch.Tensor,
    commands: torch.Tensor,
    total_steps: int,
    sim: SimulationContext,
    device: str,
) -> RunResult:
    zeta = K_angvel_xy / (2.0 * math.sqrt(K_rot_xy * I_att))
    print(f"  {label:10s}  K_rot={K_rot_xy:.1f}  K_angvel={K_angvel_xy:.1f}"
          f"  K_vel={K_vel_xy:.1f}  ζ={zeta:.3f}")

    _reset_robot(robot, device)
    _set_gains(controller, K_rot_xy, K_angvel_xy, K_vel_xy, K_rot_z, K_angvel_z, K_vel_z, device)
    controller.compute(torch.zeros((1, 4), device=device))   # warm-start
    robot.write_data_to_sim()
    sim.step()
    robot.update(_DT)

    result = RunResult(
        label=label, zeta=zeta,
        t=[], vx_cmd=[], vx_act=[], vy_cmd=[], vy_act=[],
        pitch=[], roll=[], yaw_rate=[],
    )

    for step in range(total_steps):
        cmd = commands[step : step + 1]
        wrench = controller.compute(cmd)
        thrust_cmd = torch.matmul(wrench, alloc_pinv.T).clamp(min=0.0)
        robot.set_thrust_target(thrust_cmd)
        robot.write_data_to_sim()
        sim.step()
        robot.update(_DT)

        quat_w = controller._to_torch(robot.data.root_quat_w)
        roll_rad, pitch_rad, _ = math_utils.euler_xyz_from_quat(quat_w)
        vel_b = robot.data.root_lin_vel_b.torch[0].detach().cpu()
        cmd_cpu = cmd[0].detach().cpu()

        result.t.append(step * _DT)
        result.vx_cmd.append(float(cmd_cpu[0]))
        result.vx_act.append(float(vel_b[0]))
        result.vy_cmd.append(float(cmd_cpu[1]))
        result.vy_act.append(float(vel_b[1]))
        result.pitch.append(float(pitch_rad[0]) * 180.0 / math.pi)
        result.roll.append(float(roll_rad[0])   * 180.0 / math.pi)
        result.yaw_rate.append(float(robot.data.root_ang_vel_b.torch[0, 2].detach().cpu()))

    return result


def _save_plot(results: list[RunResult], boundaries: list[tuple[int, str]]) -> None:
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 10))
    ax_vx, ax_vy, ax_att, ax_yaw = axes
    fig.suptitle(
        "Matrice controller gain-range sweep  —  ±10 % corners vs nominal",
        fontsize=11,
    )

    for ax, ylabel in zip(axes, ("vx [m/s]", "vy [m/s]", "Attitude [deg]", "Yaw rate [rad/s]")):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    ax_yaw.set_xlabel("Time [s]")

    # Command reference (same for all configs — plot once in black).
    ax_vx.plot(results[0].t, results[0].vx_cmd, "k--", linewidth=1.0, label="cmd", zorder=10)
    ax_vy.plot(results[0].t, results[0].vy_cmd, "k--", linewidth=1.0, label="cmd", zorder=10)

    for res, color in zip(results, _COLORS):
        legend_label = f"{res.label}  (ζ={res.zeta:.2f})"
        ax_vx.plot(res.t, res.vx_act,   color=color, linewidth=1.3, label=legend_label)
        ax_vy.plot(res.t, res.vy_act,   color=color, linewidth=1.3, label=legend_label)
        ax_att.plot(res.t, res.pitch,   color=color, linewidth=1.3, label=f"{res.label} pitch")
        ax_att.plot(res.t, res.roll,    color=color, linewidth=0.8, linestyle="--", alpha=0.6)
        ax_yaw.plot(res.t, res.yaw_rate, color=color, linewidth=1.3, label=legend_label)

    for ax in axes:
        ax.axhline(0.0, color="gray", linewidth=0.5, linestyle=":")

    # Phase boundary lines and labels.
    for step, label in boundaries:
        t = step * _DT
        for ax in axes:
            ax.axvline(t, color="gray", linewidth=0.7, linestyle=":")
        if label != "Hover":
            ax_vx.text(t + 0.05, 1.0, label,
                       transform=ax_vx.get_xaxis_transform(),
                       fontsize=7, color="dimgray", va="top")

    ax_vx.legend(loc="upper right", ncol=2, fontsize=7)
    ax_vy.legend(loc="upper right", ncol=2, fontsize=7)
    ax_att.legend(loc="upper right", ncol=2, fontsize=7)
    ax_yaw.legend(loc="upper right", ncol=2, fontsize=7)

    fig.tight_layout()
    out_path = "matrice_gain_sweep_results.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Plot saved to {out_path}")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=_DT)
    sim = SimulationContext(sim_cfg)
    device = str(sim.device)

    stage = omni.usd.get_context().get_stage()
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateColorAttr(Gf.Vec3f(0.53, 0.81, 0.92))
    dome.CreateIntensityAttr(1000.0)
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())

    robot_cfg = copy.deepcopy(MATRICE_CFG)
    robot_cfg.prim_path = "/World/Robot"
    robot_cfg.init_state.pos = (0.0, 0.0, _INIT_HEIGHT)
    robot_cfg.actuators["thrusters"].dt = sim_cfg.dt
    robot = Multirotor(robot_cfg)

    sim.reset()

    # Read actual PhysX inertia to compute ζ accurately.
    controller_cfg = LeeVelControllerCfg(
        K_vel_range=((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
        K_rot_range=((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
        K_angvel_range=((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
        max_inclination_angle_rad=math.radians(60.0),
        max_yaw_rate=1.0,
    )
    controller = LeeVelController(controller_cfg, robot, num_envs=1, device=device)
    I_att = max(
        controller.robot_inertia[0, 0, 0].item(),
        controller.robot_inertia[0, 1, 1].item(),
    )
    print(f"\n[INFO] PhysX I_att = {I_att:.4f} kg·m²  (used for ζ calculations)")

    alloc      = torch.tensor(robot_cfg.allocation_matrix, device=device, dtype=torch.float32)
    alloc_pinv = torch.linalg.pinv(alloc)
    commands, boundaries, total_steps = _build_commands(_SEQUENCE, device)

    print(f"\nRunning {len(_CONFIGS)} configurations × {total_steps * _DT:.0f} s each …")

    results: list[RunResult] = []
    for cfg in _CONFIGS:
        label, K_rot_xy, K_angvel_xy, K_vel_xy, K_rot_z, K_angvel_z, K_vel_z = cfg
        results.append(_run_config(
            label, K_rot_xy, K_angvel_xy, K_vel_xy, K_rot_z, K_angvel_z, K_vel_z,
            I_att, robot, controller, alloc_pinv, commands, total_steps, sim, device,
        ))

    print("\nDone. Generating plot …")
    _save_plot(results, boundaries)


if __name__ == "__main__":
    main()
    simulation_app.close()
