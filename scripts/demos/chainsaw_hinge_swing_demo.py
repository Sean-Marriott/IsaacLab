# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Release the Mk3 chainsaw from a clamped M350 airframe and plot the swing as it decays.

The airframe is fixed in place, as on the hang-and-release rig, and the payload is let go from a
displaced pose: by default the one in the motion-capture take used to identify the model (tube
-19.4 deg, saw head +37.9 deg relative to the tube). A live plot shows both hinge angles and their
swing amplitude on a log scale, so the decay and the energy exchange between the tube and the saw
head can be watched alongside the viewport.

The model is ``MATRICE_MK3_CFG`` (identified mass properties and hinge damping) with the saw-head
springs and hinge friction of the chainsaw task, ``ROOT_JOINT_SPRING_STIFFNESS`` and
``ROOT_JOINT_FRICTION``.

Launch:

.. code-block:: bash

    # viewport and live plot
    ./isaaclab.sh -p scripts/demos/chainsaw_hinge_swing_demo.py --viz kit

    # without the springs, or from another pose
    ./isaaclab.sh -p scripts/demos/chainsaw_hinge_swing_demo.py --viz kit --spring_stiffness 0.0
    ./isaaclab.sh -p scripts/demos/chainsaw_hinge_swing_demo.py --viz kit --tube_deg 10 --saw_deg 0
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Chainsaw swing decay on a clamped airframe, with a live plot.")
parser.add_argument("--tube_deg", type=float, default=-19.41, help="Initial tube pitch from rest [deg].")
parser.add_argument("--saw_deg", type=float, default=37.88, help="Initial saw-head pitch relative to the tube [deg].")
parser.add_argument("--duration", type=float, default=30.0, help="Length of the release [s].")
parser.add_argument(
    "--spring_stiffness",
    type=float,
    default=None,
    help="Saw-head spring stiffness [N·m/rad]. Defaults to the chainsaw task's ROOT_JOINT_SPRING_STIFFNESS.",
)
parser.add_argument(
    "--friction",
    type=float,
    default=None,
    help="Saw-head hinge friction coefficient. Defaults to the chainsaw task's ROOT_JOINT_FRICTION.",
)
parser.add_argument("--plot_every", type=int, default=5, help="Physics steps between plot redraws.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import time

import matplotlib

if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

from isaaclab_tasks.manager_based.drone_arl.track_trajectory.config.matrice.chainsaw_env_cfg import (
    ROOT_JOINT_FRICTION,
    ROOT_JOINT_SPRING_STIFFNESS,
)

from isaaclab_assets.robots.arl_robot_1 import MATRICE_MK3_CFG

_DT = 0.01  # s, same physics step as the chainsaw task
_MOUNT_HEIGHT = 2.5  # m, airframe height so the saw clears the ground


def main():
    """Clamp the airframe, release the payload and plot the hinge angles live."""
    sim = SimulationContext(sim_utils.SimulationCfg(dt=_DT, device=args_cli.device))
    # Side-on view of the swing plane (the tube pitches about the airframe y axis).
    sim.set_camera_view(eye=(0.0, -4.5, 2.0), target=(0.0, 0.0, 1.8))
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)).func("/World/Light", sim_utils.DomeLightCfg())

    robot_cfg = MATRICE_MK3_CFG.replace(prim_path="/World/Robot")
    robot_cfg.init_state = robot_cfg.init_state.replace(pos=(0.0, 0.0, _MOUNT_HEIGHT))
    robot_cfg.spawn.articulation_props.fix_root_link = True  # the clamp
    robot_cfg.actuators["thrusters"].dt = _DT
    robot = robot_cfg.class_type(robot_cfg)
    sim.reset()

    joint_ids, _ = robot.find_joints(["csTubePitch", "root_joint"], preserve_order=True)
    pitch_id, saw_id = joint_ids
    stiffness = ROOT_JOINT_SPRING_STIFFNESS if args_cli.spring_stiffness is None else args_cli.spring_stiffness
    friction = ROOT_JOINT_FRICTION if args_cli.friction is None else args_cli.friction
    saw_ids = torch.tensor([saw_id], dtype=torch.int32, device=sim.device)
    robot.write_joint_stiffness_to_sim_index(
        stiffness=torch.full((1, 1), stiffness, device=sim.device), joint_ids=saw_ids
    )
    coeff = torch.full((1, 1), friction, device=sim.device)
    robot.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=coeff, joint_dynamic_friction_coeff=coeff, joint_ids=saw_ids
    )

    # Hold the payload at the release pose, at rest.
    joint_pos = robot.data.default_joint_pos.torch.clone()
    joint_pos[:, pitch_id] += math.radians(args_cli.tube_deg)
    joint_pos[:, saw_id] += math.radians(args_cli.saw_deg)
    robot.write_joint_position_to_sim_index(position=joint_pos)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos))
    robot.update(0.0)

    # -- live plot. The saw head can swing to about twice its release angle as energy moves into it.
    span = 2.0 * max(abs(args_cli.saw_deg), abs(args_cli.tube_deg), 5.0)
    plt.ion()
    fig, (ax_ang, ax_amp) = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    fig.suptitle(
        f"Mk3 chainsaw on a clamped airframe: swing decay"
        f" (springs {stiffness:.4f} N·m/rad, hinge friction {friction:.3f})",
        fontsize=11,
    )
    (line_tube,) = ax_ang.plot([], [], color="tab:blue", lw=1.4, label="tube pitch (csTubePitch)")
    (line_saw,) = ax_ang.plot([], [], color="tab:red", lw=1.2, label="saw head rel. tube (root_joint)")
    ax_ang.axhline(0.0, color="gray", lw=0.6, ls="--")
    ax_ang.set_ylabel("angle from rest [deg]")
    ax_ang.set_ylim(-span, span)
    ax_ang.legend(loc="upper right", fontsize=8)
    (line_tube_amp,) = ax_amp.semilogy([], [], color="tab:blue", lw=1.4, label="tube")
    (line_saw_amp,) = ax_amp.semilogy([], [], color="tab:red", lw=1.2, label="saw head")
    ax_amp.set_ylabel("swing amplitude [deg]")
    ax_amp.set_xlabel("time since release [s]")
    ax_amp.set_ylim(0.1, 1.5 * span)
    ax_amp.legend(loc="upper right", fontsize=8)
    for ax in (ax_ang, ax_amp):
        ax.set_xlim(0.0, args_cli.duration)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show(block=False)

    # Amplitude: largest excursion over the last window, a little over one half swing (~1.1 s).
    window = int(round(1.2 / _DT))
    t_hist, tube_hist, saw_hist, tube_amp, saw_amp = [], [], [], [], []
    n_steps = int(round(args_cli.duration / _DT))
    wall_start = time.perf_counter()
    for step in range(n_steps + 1):
        if not simulation_app.is_running():
            break
        t = step * _DT
        q = robot.data.joint_pos.torch[0]
        t_hist.append(t)
        tube_hist.append(math.degrees(float(q[pitch_id])))
        saw_hist.append(math.degrees(float(q[saw_id])))
        tube_amp.append(max(abs(v) for v in tube_hist[-window:]))
        saw_amp.append(max(abs(v) for v in saw_hist[-window:]))

        if step % args_cli.plot_every == 0 or step == n_steps:
            line_tube.set_data(t_hist, tube_hist)
            line_saw.set_data(t_hist, saw_hist)
            line_tube_amp.set_data(t_hist, tube_amp)
            line_saw_amp.set_data(t_hist, saw_amp)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        robot.write_data_to_sim()
        sim.step()
        robot.update(_DT)
        # Run no faster than real time, so the viewport shows the true swing speed.
        lag = (step + 1) * _DT - (time.perf_counter() - wall_start)
        if lag > 0.0:
            time.sleep(lag)

    print(
        f"[INFO] After {t_hist[-1]:.1f} s the swing amplitude is {tube_amp[-1]:.1f} deg (tube, from"
        f" {tube_amp[0]:.1f}) and {saw_amp[-1]:.1f} deg (saw head, from {saw_amp[0]:.1f})."
    )
    if matplotlib.get_backend().lower() == "agg":
        fig.savefig("chainsaw_hinge_swing_demo.png", dpi=120)
        print("[INFO] No display: plot saved to chainsaw_hinge_swing_demo.png")
    else:
        print("[INFO] Release complete. Close the viewport to exit.")
        # Keep the viewport and the plot responsive until the app is closed.
        while simulation_app.is_running():
            sim.render()
            fig.canvas.flush_events()
            time.sleep(1.0 / 30.0)
    simulation_app.close()


if __name__ == "__main__":
    main()
