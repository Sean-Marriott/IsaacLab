# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check the sign conventions of the Mk3 chainsaw payload joints in PhysX.

The M350 + Mk3 chainsaw is spawned with gravity off, so it holds whatever pose is written. Each payload
joint is set in turn to ``+angle`` (all others at zero), the simulation is stepped once, and the script
prints where the payload COM ends up relative to the pivot, in the airframe (FLU) frame:

* ``csTubePitch`` / ``csTubeRoll``: direction of the whole payload's COM from the ``top_hinge`` origin.
* ``root_joint``: direction of the saw (``roll_link``) COM from the ``base_link`` origin.

Expected, from the USD joint frames: ``+csTubePitch`` swings the payload aft (-X), ``+csTubeRoll`` swings
it left (+Y), and ``+root_joint`` rotates the saw head right-handed about +Y (+X side tips down).

.. code-block:: bash

    ./isaaclab.sh -p scripts/tools/check_chainsaw_joint_signs.py
    ./isaaclab.sh -p scripts/tools/check_chainsaw_joint_signs.py --angle 0.5
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check the sign conventions of the Mk3 chainsaw payload joints.")
parser.add_argument("--angle", type=float, default=0.3, help="Joint angle written for each test [rad].")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import sys

import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_contrib.assets import Multirotor

from isaaclab_assets.robots.arl_robot_1 import MATRICE_MK3_CFG

AIRFRAME_BODY_NAME = "body"
PIVOT_BODY_NAME = "top_hinge"
PAYLOAD_BODY_NAMES = ["main_tube", "base_link", "roll_link", "feed_arm_link"]
JOINT_NAMES = ["csTubePitch", "csTubeRoll", "root_joint"]


def describe(vec: torch.Tensor) -> str:
    """Name the dominant horizontal direction of a vector in the FLU airframe frame."""
    x, y, z = vec.tolist()
    parts = []
    if abs(x) > 1e-3:
        parts.append(f"{'forward (+X)' if x > 0 else 'aft (-X)'} {abs(x):.3f} m")
    if abs(y) > 1e-3:
        parts.append(f"{'left (+Y)' if y > 0 else 'right (-Y)'} {abs(y):.3f} m")
    tilt = math.degrees(math.atan2(math.hypot(x, y), -z))
    return f"({x:+.3f}, {y:+.3f}, {z:+.3f}) m, {tilt:5.1f} deg from straight down: " + (", ".join(parts) or "no offset")


def main():
    """Write each joint to +angle and print the resulting payload offsets."""
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, gravity=(0.0, 0.0, 0.0)))
    robot_cfg = MATRICE_MK3_CFG.replace(prim_path="/World/Robot")
    robot_cfg.actuators["thrusters"].dt = sim.get_physics_dt()
    robot = Multirotor(robot_cfg)
    sim.reset()
    dt = sim.get_physics_dt()

    airframe_id = robot.find_bodies(AIRFRAME_BODY_NAME)[0][0]
    pivot_id = robot.find_bodies(PIVOT_BODY_NAME)[0][0]
    head_id = robot.find_bodies("base_link")[0][0]
    saw_id = robot.find_bodies("roll_link")[0][0]
    payload_ids = robot.find_bodies(PAYLOAD_BODY_NAMES, preserve_order=True)[0]
    joint_ids = {name: robot.find_joints(name)[0][0] for name in JOINT_NAMES}
    print(f"[INFO] Joint order: {robot.joint_names}")

    root_pose = robot.data.default_root_pose.torch.clone()
    root_pose[:, 2] = 3.0
    root_vel = torch.zeros_like(robot.data.default_root_vel.torch)
    masses = robot.data.body_mass.torch[0, payload_ids].to(sim.device)

    cases = [("rest (all zero)", None)] + [(f"{name} = {args_cli.angle:+.2f} rad", name) for name in JOINT_NAMES]
    for label, joint in cases:
        joint_pos = torch.zeros_like(robot.data.default_joint_pos.torch)
        if joint is not None:
            joint_pos[:, joint_ids[joint]] = args_cli.angle
        robot.write_root_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
        robot.write_joint_position_to_sim_index(position=joint_pos)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint_pos))
        sim.step()
        robot.update(dt)

        quat = robot.data.body_link_quat_w.torch[0, airframe_id]
        link_pos = robot.data.body_link_pos_w.torch[0]
        com_pos = robot.data.body_com_pos_w.torch[0]
        payload_com = (masses[:, None] * com_pos[payload_ids]).sum(0) / masses.sum()
        swing = quat_apply_inverse(quat, payload_com - link_pos[pivot_id])
        saw = quat_apply_inverse(quat, com_pos[saw_id] - link_pos[head_id])
        measured = robot.data.joint_pos.torch[0, list(joint_ids.values())].tolist()

        print(f"\n{label}")
        print(f"  joint pos read back {dict(zip(JOINT_NAMES, (round(v, 3) for v in measured)))}")
        print(f"  payload COM from {PIVOT_BODY_NAME}: {describe(swing)}")
        print(f"  saw COM from base_link:       {describe(saw)}")


if __name__ == "__main__":
    main()
    # The app exits hard on close, which drops buffered output when stdout is piped.
    sys.stdout.flush()
    simulation_app.close()
