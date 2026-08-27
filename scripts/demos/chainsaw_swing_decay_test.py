# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnose why the Matrice chainsaw swing decays in simulation.

Two back-to-back runs, csTubePitch initialized to a small kick angle each time:

  Experiment A — Root LOCKED (pose overwritten every step, no controller):
    Any decay here = pure PhysX TGS numerical dissipation.

  Experiment B — Root HOVERING under Lee velocity controller (zero cmd):
    Decay = numerical dissipation + attitude-controller reaction forces.

Findings (2026-05-15)
---------------------
Experiment A: swing oscillates indefinitely — no measurable decay over 60 s.
  → PhysX TGS numerical dissipation is negligible at dt=0.01 s / 4 pos iters.

Experiment B: swing decays with a half-life of ~0.4 s.
  → 100 % of the observed in-simulation damping comes from the attitude
    controller fighting the disturbance torques the swinging payload applies
    to the airframe.  Each corrective thrust pulse transmits a reaction force
    back through the articulation chain that bleeds energy out of the swing.

USD damping audit (M350-chainsaw.usd):
  csTubePitch — physxLimit:angular:damping = 0.0, no DriveAPI
  csTubeRoll  — physxLimit:angular:damping = 0.5 N·m·s/rad (at ±45° limits
                only), no DriveAPI
  All rigid-body links: physics:linearDamping = physics:angularDamping = 0.0
  (explicitly set to zero by MATRICE_CFG.spawn.rigid_props at spawn time)

Implications for RL training
-----------------------------
- The RL policy is the *only* damping mechanism in simulation.  Any swing
  suppression learned during training is an active control strategy, not a
  passive joint property.
- The real chainsaw joints have physical friction/viscosity that the sim does
  not model, so the sim is less damped than reality.  This is conservative
  (harder to suppress swing in sim than in real life) but may cause the
  policy to rely on over-sized corrective thrusts.
- Adding DriveAPI damping to csTubePitch / csTubeRoll would introduce a
  passive damping term independent of the controller.  Re-run Experiment A
  after adding it to verify the joint damping is active before retraining.

Launch:
    ./isaaclab.sh -p scripts/demos/chainsaw_swing_decay_test.py
    ./isaaclab.sh -p scripts/demos/chainsaw_swing_decay_test.py --viz kit
"""

import argparse
import copy
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Chainsaw swing decay diagnostic.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.usd
from pxr import Gf, UsdLux

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

from isaaclab_contrib.assets import Multirotor
from isaaclab_contrib.controllers.lee_velocity_control import LeeVelController
from isaaclab_contrib.controllers.lee_velocity_control_cfg import LeeVelControllerCfg

from isaaclab_assets.robots.arl_robot_1 import MATRICE_CFG

# ── Constants ─────────────────────────────────────────────────────────────────
_INIT_HEIGHT = 10.0  # m  — initial hover height
_KICK_ANGLE = 5.0  # deg — initial csTubePitch displacement
_DT = 0.01  # s  — simulation timestep
_DUR_A = 60.0  # s  — Exp A (locked root): long enough to see slow numerical decay
_DUR_B = 20.0  # s  — Exp B (hover ctrl): controller damps quickly
_STEPS_A = int(_DUR_A / _DT)
_STEPS_B = int(_DUR_B / _DT)


def _compute_lee_gains(controller: LeeVelController, robot_cfg, device: str) -> None:
    """Set Lee controller gains analytically from actual PhysX inertia."""
    m = controller.mass[0].item()
    I_xx = controller.robot_inertia[0, 0, 0].item()
    I_yy = controller.robot_inertia[0, 1, 1].item()
    I_zz = controller.robot_inertia[0, 2, 2].item()
    I_att = max(I_xx, I_yy)

    thrust_max = robot_cfg.actuators["thrusters"].thrust_range[1]
    hover_thrust = m * 9.81 / 4.0
    arm_pitch = sum(abs(v) for v in robot_cfg.allocation_matrix[3])
    arm_roll = sum(abs(v) for v in robot_cfg.allocation_matrix[4])
    tau_max = (thrust_max - hover_thrust) * min(arm_pitch, arm_roll)

    K_rot = min(tau_max / (0.5 * I_att), 200.0)
    omega_n = (K_rot / I_att) ** 0.5
    K_angvel = 2.0 * 0.85 * omega_n * I_att
    K_rot_z = 0.4 * K_rot
    omega_nz = (K_rot_z / I_zz) ** 0.5
    K_angvelz = 2.0 * 0.85 * omega_nz * I_zz
    K_vel = omega_n

    print(f"  mass={m:.3f} kg  I_att={I_att:.4f} kg·m²  tau_max={tau_max:.2f} N·m")
    print(f"  K_rot={K_rot:.2f}  K_angvel={K_angvel:.2f}  K_vel={K_vel:.2f}")

    controller.K_rot_current[:] = torch.tensor([[K_rot, K_rot, K_rot_z]], device=device)
    controller.K_angvel_current[:] = torch.tensor([[K_angvel, K_angvel, K_angvelz]], device=device)
    controller.K_vel_current[:] = torch.tensor([[K_vel, K_vel, K_vel * 0.6]], device=device)


def _reset_robot(robot: Multirotor, pitch_ids: list[int], device: str) -> None:
    """Reset robot to initial hover state with csTubePitch at _KICK_ANGLE degrees.

    Avoids robot.reset() because its warp-array indexing path is broken mid-simulation
    (wp.array does not support item indexing via _ALL_INDICES).  We replicate what it
    would do by writing directly to the sim via the public write APIs.
    """
    # Root: (0, 0, _INIT_HEIGHT), identity quaternion, zero velocity.
    init_pose = torch.tensor([[0.0, 0.0, _INIT_HEIGHT, 0.0, 0.0, 0.0, 1.0]], device=device)
    init_vel = torch.zeros((1, 6), device=device)
    robot.write_root_pose_to_sim_index(root_pose=init_pose, env_ids=None)
    robot.write_root_velocity_to_sim_index(root_velocity=init_vel, env_ids=None)

    # Reset all joints to default positions/velocities, then override csTubePitch.
    default_pos = robot.data.default_joint_pos.torch.clone()
    default_pos[0, pitch_ids[0]] = math.radians(_KICK_ANGLE)
    zero_jvel = torch.zeros_like(default_pos)
    robot.write_joint_position_to_sim_index(position=default_pos, joint_ids=None, env_ids=None)
    robot.write_joint_velocity_to_sim_index(velocity=zero_jvel, joint_ids=None, env_ids=None)


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

    # Joint indices for the two chainsaw DOFs.
    pitch_ids, pitch_names = robot.find_joints(["csTubePitch"])
    roll_ids, roll_names = robot.find_joints(["csTubeRoll"])
    print(f"[INFO] csTubePitch joint index: {pitch_ids} ({pitch_names})")
    print(f"[INFO] csTubeRoll  joint index: {roll_ids} ({roll_names})")

    # Lee controller — gains set after first sim.reset() so inertia is available.
    controller_cfg = LeeVelControllerCfg(
        K_vel_range=((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
        K_rot_range=((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
        K_angvel_range=((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
        max_inclination_angle_rad=math.radians(60.0),
        max_yaw_rate=1.0,
    )
    controller = LeeVelController(controller_cfg, robot, num_envs=1, device=device)
    print("\nLee controller gains:")
    _compute_lee_gains(controller, robot_cfg, device)

    alloc = torch.tensor(robot_cfg.allocation_matrix, device=device, dtype=torch.float32)
    alloc_pinv = torch.linalg.pinv(alloc)
    hover_cmd = torch.zeros((1, 4), device=device)

    time_axis_a: list[float] = []
    time_axis_b: list[float] = []
    pitch_locked: list[float] = []  # Experiment A
    pitch_hover: list[float] = []  # Experiment B
    locked_root_pose = torch.tensor([[0.0, 0.0, _INIT_HEIGHT, 0.0, 0.0, 0.0, 1.0]], device=device)
    locked_root_vel = torch.zeros((1, 6), device=device)

    # ── Experiment A: locked root ─────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print("Experiment A: root LOCKED, no controller")
    print(f"  csTubePitch initial = {_KICK_ANGLE:.0f}°, duration = {_DUR_A:.0f} s")
    print(f"{'=' * 55}")

    _reset_robot(robot, pitch_ids, device)
    # Zero thrusters so rotor forces don't couple into the chainsaw joints.
    num_thrusters = len(robot_cfg.actuators["thrusters"].thruster_names_expr)
    zero_thrust = torch.zeros((1, num_thrusters), device=device)
    robot.set_thrust_target(zero_thrust)
    robot.write_data_to_sim()
    sim.step()
    robot.update(_DT)

    for step in range(_STEPS_A):
        # Freeze root: overwrite pose and velocity before stepping.
        robot.write_root_pose_to_sim_index(root_pose=locked_root_pose, env_ids=None)
        robot.write_root_velocity_to_sim_index(root_velocity=locked_root_vel, env_ids=None)
        robot.set_thrust_target(zero_thrust)
        robot.write_data_to_sim()
        sim.step()
        robot.update(_DT)

        pos_deg = math.degrees(float(robot.data.joint_pos.torch[0, pitch_ids[0]]))
        pitch_locked.append(pos_deg)
        time_axis_a.append(step * _DT)

        if step % 500 == 0:
            print(f"  t={step * _DT:5.1f}s  csTubePitch={pos_deg:+7.3f}°")

    # ── Experiment B: Lee velocity hover controller ───────────────────────────
    print(f"\n{'=' * 55}")
    print("Experiment B: Lee velocity hover controller")
    print(f"  csTubePitch initial = {_KICK_ANGLE:.0f}°, duration = {_DUR_B:.0f} s")
    print(f"{'=' * 55}")

    _reset_robot(robot, pitch_ids, device)
    # Warm-start controller with one compute call so internal state is consistent.
    controller.compute(hover_cmd)
    robot.write_data_to_sim()
    sim.step()
    robot.update(_DT)

    for step in range(_STEPS_B):
        wrench = controller.compute(hover_cmd)
        thrust_cmd = torch.matmul(wrench, alloc_pinv.T).clamp(min=0.0)
        robot.set_thrust_target(thrust_cmd)
        robot.write_data_to_sim()
        sim.step()
        robot.update(_DT)

        pos_deg = math.degrees(float(robot.data.joint_pos.torch[0, pitch_ids[0]]))
        pitch_hover.append(pos_deg)
        time_axis_b.append(step * _DT)

        if step % 200 == 0:
            print(f"  t={step * _DT:5.1f}s  csTubePitch={pos_deg:+7.3f}°")

    # ── Results summary ───────────────────────────────────────────────────────
    def _half_life(series: list[float], dur: float) -> str:
        """Estimate time (s) for oscillation amplitude to halve, or '>Xs'."""
        half = abs(series[0]) / 2.0
        for i, v in enumerate(series):
            if abs(v) <= half:
                return f"{i * _DT:.1f} s"
        return f">{dur:.0f} s"

    print(f"\n{'=' * 55}")
    print("Summary")
    print(f"  Experiment A (locked root, {_DUR_A:.0f} s):  half-life ≈ {_half_life(pitch_locked, _DUR_A)}")
    print(f"  Experiment B (hover ctrl,  {_DUR_B:.0f} s):  half-life ≈ {_half_life(pitch_hover, _DUR_B)}")
    print(f"  Initial angle = {_KICK_ANGLE:.0f}°")
    print("  physxLimit:angular:damping — csTubePitch=0.0, csTubeRoll=0.5 N·m·s/rad (at limits only)")
    print(f"{'=' * 55}\n")

    # ── Plot — two subplots share y-axis so amplitudes are directly comparable ─
    fig, (ax_a, ax_b) = plt.subplots(1, 2, sharey=True, figsize=(14, 5))

    ax_a.plot(time_axis_a, pitch_locked, color="tab:blue", linewidth=1.0)
    ax_a.axhline(0.0, color="gray", linewidth=0.6, linestyle="--")
    ax_a.axhline(_KICK_ANGLE / 2, color="tab:gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax_a.axhline(-_KICK_ANGLE / 2, color="tab:gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax_a.set_xlabel("Time [s]")
    ax_a.set_ylabel("csTubePitch [deg]")
    ax_a.set_title(f"A: root LOCKED, no controller ({_DUR_A:.0f} s)\nhalf-life ≈ {_half_life(pitch_locked, _DUR_A)}")
    ax_a.grid(True, alpha=0.3)

    ax_b.plot(time_axis_b, pitch_hover, color="tab:orange", linewidth=1.0)
    ax_b.axhline(0.0, color="gray", linewidth=0.6, linestyle="--")
    ax_b.axhline(_KICK_ANGLE / 2, color="tab:gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax_b.axhline(-_KICK_ANGLE / 2, color="tab:gray", linewidth=0.8, linestyle=":", alpha=0.6)
    ax_b.set_xlabel("Time [s]")
    ax_b.set_title(f"B: hover controller ({_DUR_B:.0f} s)\nhalf-life ≈ {_half_life(pitch_hover, _DUR_B)}")
    ax_b.grid(True, alpha=0.3)

    fig.suptitle(
        f"Chainsaw swing decay — csTubePitch, initial = {_KICK_ANGLE:.0f}°  |  dotted lines = ±half-amplitude",
        fontsize=11,
    )
    fig.tight_layout()

    out_path = "chainsaw_swing_decay_results.png"
    fig.savefig(out_path, dpi=150)
    print(f"[INFO] Plot saved to {out_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
