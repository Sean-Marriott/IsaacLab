# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ARL robots.

The following configuration parameters are available:

* :obj:`ARL_ROBOT_1_CFG`: The ARL_Robot_1
"""

import math
import pathlib

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

_M350_USD_PATH = str(
    pathlib.Path(__file__).parents[3] / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based" / "drone_arl" / "robots" / "M350-chainsaw.usd"
)

from isaaclab_contrib.actuators import ThrusterCfg
from isaaclab_contrib.assets import MultirotorCfg
import math
##
# Configuration - Actuators.
##

ARL_ROBOT_1_THRUSTER = ThrusterCfg(
    thrust_range=(0.1, 10.0),
    thrust_const_range=(9.26312e-06, 1.826312e-05),
    tau_inc_range=(0.05, 0.08),
    tau_dec_range=(0.005, 0.005),
    torque_to_thrust_ratio=0.07,
    thruster_names_expr=["back_left_prop", "back_right_prop", "front_left_prop", "front_right_prop"],
)

##
# Configuration - Articulation.
##

ARL_ROBOT_1_CFG = MultirotorCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NTNU/ARL-Robot-1/arl_robot_1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=MultirotorCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        rps={
            "back_left_prop": 200.0,
            "back_right_prop": 200.0,
            "front_left_prop": 200.0,
            "front_right_prop": 200.0,
        },
    ),
    actuators={"thrusters": ARL_ROBOT_1_THRUSTER},
    rotor_directions=[-1, 1, -1, 1],
    allocation_matrix=[
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [-0.13, -0.13, 0.13, 0.13],
        [-0.13, 0.13, 0.13, -0.13],
        [-0.07, 0.07, -0.07, 0.07],
    ],
)

# DJI Matrice 350 RTK.
# Thrust model: k_f = 3.35e-3 N/rps² (mean of thrust_const_range), max 55 N/motor.
# Hover RPS = sqrt(m * g / (4 * k_f_nominal)).  Run scripts/demos/arl_robot_1.py to get
# the value that matches the actual USD mass; update init_state.rps accordingly.
# Allocation matrix row conventions (motor order: BL, BR, FL, FR):
#   row 2  Fz [N]:   sum of thrusts
#   row 3  Tx [N·m]: fore-aft moment (x-arm distances)
#   row 4  Ty [N·m]: lateral moment (y-arm distances)
#   row 5  Tz [N·m]: yaw torque (torque_to_thrust_ratio × rotor_direction)
# Motor positions: back_left (-0.3182,  0.3382), back_right (-0.3182, -0.3382)
#                 front_left ( 0.3161,  0.3786), front_right ( 0.3161, -0.3786)
MATRICE_THRUSTER = ThrusterCfg(
    thrust_range=(0.5, 55.0),
    thrust_const_range=(2.8e-3, 3.9e-3),
    tau_inc_range=(0.05, 0.08),
    tau_dec_range=(0.005, 0.005),
    torque_to_thrust_ratio=0.05,
    thruster_names_expr=["back_left_prop", "back_right_prop", "front_left_prop", "front_right_prop"],
)

MATRICE_CFG = MultirotorCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_M350_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=MultirotorCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        rps={
            "back_left_prop": 69.0,
            "back_right_prop": 69.0,
            "front_left_prop": 69.0,
            "front_right_prop": 69.0,
        },
    ),
    actuators={"thrusters": MATRICE_THRUSTER},
    rotor_directions=[-1, 1, -1, 1],
    allocation_matrix=[
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [-0.3182, -0.3182, 0.3161, 0.3161],
        [-0.3382, 0.3382, 0.3786, -0.3786],
        [-0.05, 0.05, -0.05, 0.05],
    ],
)