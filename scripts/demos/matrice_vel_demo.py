# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Matrice 350 RTK — Lee velocity controller demonstration.

Runs a structured velocity command sequence and plots tracking performance
across four panels: body-frame velocity, roll/pitch attitude, yaw rate, and
total thrust.  Controller gains are computed analytically from the actual
PhysX inertia at startup so the script adapts automatically to model changes.

Launch:
    ./isaaclab.sh -p scripts/demos/matrice_vel_demo.py --viz kit
"""

import argparse
import copy
import math

import matplotlib.pyplot as plt
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Matrice 350 Lee velocity controller demo.")
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

# ── Test sequence ─────────────────────────────────────────────────────────────
# Each entry: (duration_s, [vx, vy, vz, yaw_rate_rad_s], phase_label)
# Commands are in body frame; yaw uses the vehicle frame convention.
_TEST_SEQUENCE = [
    (2.0, [0.0,  0.0,  0.0, 0.0], "Hover"),
    (3.0, [0.0,  0.0,  0.5, 0.0], "+vz"),
    (2.0, [0.0,  0.0,  0.0, 0.0], "Hover"),
    (3.0, [0.0,  0.0, -0.5, 0.0], "-vz"),
    (2.0, [0.0,  0.0,  0.0, 0.0], "Hover"),
    (3.0, [0.5,  0.0,  0.0, 0.0], "+vx"),
    (2.0, [0.0,  0.0,  0.0, 0.0], "Hover"),
    (3.0, [0.0,  0.5,  0.0, 0.0], "+vy"),
    (2.0, [0.0,  0.0,  0.0, 0.0], "Hover"),
    (3.0, [0.0,  0.0,  0.0, 0.5], "Yaw"),
    (2.0, [0.0,  0.0,  0.0, 0.0], "Hover"),
]


def _build_command_tensor(sequence: list, dt: float, device: str) -> tuple:
    """Pre-build command tensor and phase boundary list from test sequence.

    Returns:
        commands: (total_steps, 4) tensor of velocity commands.
        boundaries: list of (start_step, label) tuples.
        total_steps: total number of simulation steps.
    """
    total_steps = sum(int(dur / dt) for dur, _, _ in sequence)
    commands = torch.zeros((total_steps, 4), device=device)
    boundaries = []
    idx = 0
    for dur, cmd, label in sequence:
        n = int(dur / dt)
        commands[idx : idx + n] = torch.tensor(cmd, device=device)
        boundaries.append((idx, label))
        idx += n
    return commands, boundaries, total_steps


def _compute_and_apply_gains(controller: LeeVelController, robot_cfg, device: str) -> None:
    """Derive Lee controller gains analytically from actual PhysX inertia and apply them.

    Design equations:
        K_rot    = min(tau_max / (0.5 * I_att), 80)   [motor-saturation-limited]
        K_angvel = 2 * 1.2 * omega_n * I_att          [zeta = 1.2, overdamped]
        K_vel    = omega_n / 3                         [3x cascade bandwidth margin]
        Yaw: K_rot_z = 0.4 * K_rot_xy, same zeta with I_zz
    """
    m = controller.mass[0].item()
    I_xx = controller.robot_inertia[0, 0, 0].item()
    I_yy = controller.robot_inertia[0, 1, 1].item()
    I_zz = controller.robot_inertia[0, 2, 2].item()
    I_att = max(I_xx, I_yy)

    thrust_max = robot_cfg.actuators["thrusters"].thrust_range[1]
    hover_thrust = m * 9.81 / 4.0
    # tau_max: (thrust headroom) × (sum of all arm contributions on the limiting axis).
    # For differential thrust each pair contributes, so the correct multiplier is
    # sum(|arm_i|) per axis — pitch arms are smaller and therefore limit K_rot.
    arm_pitch_sum = sum(abs(v) for v in robot_cfg.allocation_matrix[3])
    arm_roll_sum  = sum(abs(v) for v in robot_cfg.allocation_matrix[4])
    tau_max = (thrust_max - hover_thrust) * min(arm_pitch_sum, arm_roll_sum)

    K_rot_att = min(tau_max / (0.5 * I_att), 80.0)
    omega_n_att = (K_rot_att / I_att) ** 0.5
    K_angvel_att = 2.0 * 1.2 * omega_n_att * I_att

    K_rot_z = 0.4 * K_rot_att
    omega_n_z = (K_rot_z / I_zz) ** 0.5
    K_angvel_z = 2.0 * 1.2 * omega_n_z * I_zz

    K_vel_xy = omega_n_att / 3.0
    K_vel_z = K_vel_xy * 1.5

    k_f_lo, k_f_hi = robot_cfg.actuators["thrusters"].thrust_const_range
    hover_rps = (hover_thrust / ((k_f_lo + k_f_hi) / 2.0)) ** 0.5

    print(f"\n{'=' * 60}")
    print(f"Matrice model:  mass = {m:.4f} kg")
    print(f"  I_xx = {I_xx:.5f}  I_yy = {I_yy:.5f}  I_zz = {I_zz:.5f}  [kg·m²]")
    print(f"  hover_thrust/motor = {hover_thrust:.3f} N  (max {thrust_max:.0f} N)")
    print(f"  arm_pitch_sum = {arm_pitch_sum:.4f} m  arm_roll_sum = {arm_roll_sum:.4f} m  tau_max = {tau_max:.2f} N·m")
    print(f"Attitude inner loop  (ω_n = {omega_n_att:.3f} rad/s, ζ = 1.2):")
    print(f"  K_rot_xy   = {K_rot_att:.4f}    K_angvel_xy = {K_angvel_att:.4f}")
    print(f"  K_rot_z    = {K_rot_z:.4f}    K_angvel_z  = {K_angvel_z:.4f}")
    print(f"Velocity outer loop  (τ_vel = {1/K_vel_xy:.3f} s >> τ_att = {1/omega_n_att:.3f} s):")
    print(f"  K_vel_xy   = {K_vel_xy:.4f}    K_vel_z     = {K_vel_z:.4f}")
    print(f"Recommended MATRICE_CFG init_state rps = {hover_rps:.2f}")
    print(f"{'=' * 60}\n")

    controller.K_rot_current[:] = torch.tensor(
        [[K_rot_att, K_rot_att, K_rot_z]], device=device
    )
    controller.K_angvel_current[:] = torch.tensor(
        [[K_angvel_att, K_angvel_att, K_angvel_z]], device=device
    )
    controller.K_vel_current[:] = torch.tensor(
        [[K_vel_xy, K_vel_xy, K_vel_z]], device=device
    )


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)

    stage = omni.usd.get_context().get_stage()
    dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome.CreateColorAttr(Gf.Vec3f(0.53, 0.81, 0.92))
    dome.CreateIntensityAttr(1000.0)

    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())

    robot_cfg = copy.deepcopy(MATRICE_CFG)
    robot_cfg.prim_path = "/World/Robot"
    robot_cfg.init_state.pos = (0.0, 0.0, 10.0)
    robot_cfg.actuators["thrusters"].dt = sim_cfg.dt
    robot = Multirotor(robot_cfg)

    sim.reset()

    # Placeholder cfg — gains are overridden by _compute_and_apply_gains below.
    controller_cfg = LeeVelControllerCfg(
        K_vel_range=((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
        K_rot_range=((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
        K_angvel_range=((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
        max_inclination_angle_rad=1.0471975511965976,
        max_yaw_rate=1.0471975511965976,
    )
    controller = LeeVelController(controller_cfg, robot, num_envs=1, device=str(sim.device))
    _compute_and_apply_gains(controller, robot_cfg, str(sim.device))

    alloc = torch.tensor(robot_cfg.allocation_matrix, device=sim.device, dtype=torch.float32)
    alloc_pinv = torch.linalg.pinv(alloc)

    commands, phase_boundaries, total_steps = _build_command_tensor(
        _TEST_SEQUENCE, sim_cfg.dt, str(sim.device)
    )

    # ── Matplotlib setup ──────────────────────────────────────────────────────
    plt.ion()
    fig, (ax_vel, ax_att, ax_yaw, ax_thr) = plt.subplots(4, 1, sharex=True, figsize=(10, 9))
    fig.suptitle("Matrice 350 — Lee velocity controller (analytically tuned)", fontsize=11)

    for ax, ylabel in zip(
        (ax_vel, ax_att, ax_yaw, ax_thr),
        ("Velocity [m/s]", "Attitude [deg]", "Yaw rate [rad/s]", "Thrust [N]"),
    ):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    ax_thr.set_xlabel("Time [s]")

    _C = {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}
    vel_lines = {}
    for axis in ("x", "y", "z"):
        (vel_lines[f"cmd_{axis}"],) = ax_vel.plot([], [], "--", color=_C[axis], label=f"cmd v{axis}", linewidth=1.2)
        (vel_lines[f"act_{axis}"],) = ax_vel.plot([], [], "-",  color=_C[axis], label=f"act v{axis}", linewidth=1.5)
    ax_vel.legend(loc="upper right", ncol=3, fontsize=7)

    (line_roll,)  = ax_att.plot([], [], color="tab:red",    label="roll",  linewidth=1.5)
    (line_pitch,) = ax_att.plot([], [], color="tab:purple", label="pitch", linewidth=1.5)
    ax_att.legend(loc="upper right", fontsize=7)

    (line_cmd_yaw,) = ax_yaw.plot([], [], "--", color="tab:brown", label="cmd", linewidth=1.2)
    (line_act_yaw,) = ax_yaw.plot([], [], "-",  color="tab:cyan",  label="act", linewidth=1.5)
    ax_yaw.legend(loc="upper right", fontsize=7)

    (line_cmd_thr,) = ax_thr.plot([], [], "--", color="tab:olive", label="cmd", linewidth=1.2)
    (line_act_thr,) = ax_thr.plot([], [], "-",  color="tab:gray",  label="act", linewidth=1.5)
    ax_thr.legend(loc="upper right", fontsize=7)

    fig.tight_layout()
    plt.show(block=False)

    # ── History buffers ───────────────────────────────────────────────────────
    t_hist = []
    cmd_v = {a: [] for a in ("x", "y", "z")}
    act_v = {a: [] for a in ("x", "y", "z")}
    roll_hist, pitch_hist = [], []
    cmd_yaw_hist, act_yaw_hist = [], []
    cmd_thr_hist, act_thr_hist = [], []

    phase_step_set = {step for step, _ in phase_boundaries}
    phase_label_map = dict(phase_boundaries)

    # ── Simulation loop ───────────────────────────────────────────────────────
    print("[INFO] Running test sequence …")
    controller.compute(torch.zeros((1, 4), device=sim.device))

    sim_time = 0.0
    step_count = 0
    plot_interval = 5

    while simulation_app.is_running() and step_count < total_steps:
        vel_command = commands[step_count : step_count + 1]

        wrench = controller.compute(vel_command)
        thrust_cmd = torch.matmul(wrench, alloc_pinv.T).clamp(min=0.0)
        robot.set_thrust_target(thrust_cmd)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_cfg.dt)

        # Record state
        quat_w = controller._to_torch(robot.data.root_quat_w)
        roll_rad, pitch_rad, _ = math_utils.euler_xyz_from_quat(quat_w)
        act_vel_b = robot.data.root_lin_vel_b.torch[0].detach().cpu()
        cmd_vec = vel_command[0].detach().cpu()

        t_hist.append(sim_time)
        for i, a in enumerate(("x", "y", "z")):
            cmd_v[a].append(float(cmd_vec[i]))
            act_v[a].append(float(act_vel_b[i]))
        roll_hist.append(float(roll_rad[0]) * 180.0 / math.pi)
        pitch_hist.append(float(pitch_rad[0]) * 180.0 / math.pi)
        cmd_yaw_hist.append(float(cmd_vec[3]))
        act_yaw_hist.append(float(robot.data.root_ang_vel_b.torch[0, 2].detach().cpu()))
        cmd_thr_hist.append(float(thrust_cmd[0].sum().detach().cpu()))
        act_thr_hist.append(float(robot.actuators["thrusters"].applied_thrust[0].sum().detach().cpu()))

        # Draw phase boundary line the first time we hit it
        if step_count in phase_step_set:
            label = phase_label_map[step_count]
            for ax in (ax_vel, ax_att, ax_yaw, ax_thr):
                ax.axvline(sim_time, color="gray", linewidth=0.8, linestyle=":")
            if label != "Hover":
                ax_vel.text(
                    sim_time + 0.05, 1.0, label,
                    transform=ax_vel.get_xaxis_transform(),
                    fontsize=7, color="dimgray", va="top",
                )

        if step_count % plot_interval == 0:
            for a in ("x", "y", "z"):
                vel_lines[f"cmd_{a}"].set_data(t_hist, cmd_v[a])
                vel_lines[f"act_{a}"].set_data(t_hist, act_v[a])
            line_roll.set_data(t_hist, roll_hist)
            line_pitch.set_data(t_hist, pitch_hist)
            line_cmd_yaw.set_data(t_hist, cmd_yaw_hist)
            line_act_yaw.set_data(t_hist, act_yaw_hist)
            line_cmd_thr.set_data(t_hist, cmd_thr_hist)
            line_act_thr.set_data(t_hist, act_thr_hist)
            for ax in (ax_vel, ax_att, ax_yaw, ax_thr):
                ax.relim()
                ax.autoscale_view()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        sim_time += sim_cfg.dt
        step_count += 1

    print("[INFO] Sequence complete. Close the plot window to exit.")
    plt.ioff()
    plt.show(block=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
