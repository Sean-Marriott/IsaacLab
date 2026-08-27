# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ARL robots.

The following configuration parameters are available:

* :obj:`ARL_ROBOT_1_CFG`: The ARL_Robot_1
"""

import pathlib

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

_M350_ROBOTS_DIR = (
    pathlib.Path(__file__).parents[3] / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based" / "drone_arl" / "robots"
)

_M350_USD_PATH = str(_M350_ROBOTS_DIR / "M350-chainsaw.usd")
_M350_CLEAN_USD_PATH = str(_M350_ROBOTS_DIR / "M350.usd")

from isaaclab_contrib.actuators import ThrusterCfg
from isaaclab_contrib.assets import MultirotorCfg

##
# Configuration - Actuators.
##

# Note: ``tau_dec_range`` here is 10-16x shorter than ``tau_inc_range`` and shorter than the 10 ms
# physics step every drone_arl task runs at. That asymmetry biases the delivered thrust downwards --
# see the note on MATRICE_THRUSTER for the mechanism and its consequences for a deployed policy. The
# values are left as upstream defines them; the Matrice thrusters below do not inherit the problem.
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
    # The rise and fall time constants are drawn from the same range, independently per motor, and
    # they must stay comparable.
    #
    # A fall constant much shorter than the rise one makes the motor track thrust decreases faster
    # than increases, which *rectifies* a fluctuating thrust command downwards. Replaying one
    # recorded command sequence through both models: the previous (0.005, 0.005) delivered 78.8 % of
    # the commanded thrust, against 94.8 % for the symmetric pair. Reversing the two constants
    # instead delivers 114.1 %, and a non-fluctuating command shows no bias under any of them, which
    # is what identifies rectification as the mechanism.
    #
    # It is the differential (attitude) component that does the damage: per-motor command varies by
    # std/mean = 0.53 about hover while the four-motor total varies by only 0.065. Each motor
    # rectifies its own swing and the losses add in the sum.
    #
    # A policy trims the deficit out with a constant positive climb command. At 78.8 % delivery the
    # predicted trim is g*(1/0.788 - 1)/K_vel = +0.66 of full scale, and the policy trained against
    # those values learned +0.67. The trim is a simulation artifact with nothing to cancel on a real
    # vehicle, so the drone climbs away as soon as the policy is deployed.
    #
    # Sampling both constants from one range keeps the asymmetry zero-mean and randomises its sign
    # per motor, so no fixed trim is learnable. Both bounds also stay at or above the 10 ms physics
    # step, below which the first-order model is being asked to resolve dynamics faster than it is
    # integrated. A 21-inch prop is drag-limited on spin-down, so if anything the fall is the slower
    # of the two -- never an order of magnitude faster.
    #
    # A ~5 % deficit survives this fix and is expected: the lag acts on RPM while thrust scales as
    # rpm^2, so smoothing the RPM lowers the mean thrust whatever the symmetry. That part is real
    # motor physics rather than a modelling error, and it predicts a residual trim near +0.13 -- the
    # number a retrained policy should be checked against, rather than zero.
    tau_inc_range=(0.05, 0.08),
    tau_dec_range=(0.05, 0.08),
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
            "back_left_prop": 79.65,
            "back_right_prop": 79.65,
            "front_left_prop": 79.65,
            "front_right_prop": 79.65,
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

# DJI Matrice 350 RTK without the chainsaw payload.
# Same airframe as MATRICE_CFG -- identical body names, motor positions, allocation matrix and
# thruster model -- but spawned from M350.usd, which carries only base_link and the four props.
#   base_link  5.968 kg, diagonal inertia (0.7, 0.7, 0.9) kg*m^2
#   props      4 x 0.133 kg, no own inertia (they contribute via the parallel-axis term)
#   total      6.500 kg   (vs 8.665 kg with the chainsaw, jetson, hinge and tube)
# Aggregate inertia about the robot COM, as LeeControllerBase computes it from PhysX:
#   (I_xx, I_yy, I_zz) = (0.7754, 0.7560, 1.0286) kg*m^2   (vs (2.640, 2.649, 1.065) with chainsaw)
# Hover thrust per motor = 6.500 * 9.81 / 4 = 15.94 N, so with k_f = 3.35e-3 N/rps^2 the hover
# speed is sqrt(15.94 / 3.35e-3) = 68.98 rps.  Thrust-to-weight = 4 * 55 / (6.500 * 9.81) = 3.45.
# The lighter airframe has a much smaller attitude inertia than the chainsaw model, so any
# controller gains tuned for MATRICE_CFG must be re-derived -- see scripts/demos/matrice_vel_demo.py.
MATRICE_CLEAN_THRUSTER = ThrusterCfg(
    thrust_range=(0.5, 55.0),
    thrust_const_range=(2.8e-3, 3.9e-3),
    # Rise and fall drawn from one range; see the note on MATRICE_THRUSTER.
    tau_inc_range=(0.05, 0.08),
    tau_dec_range=(0.05, 0.08),
    torque_to_thrust_ratio=0.05,
    thruster_names_expr=["back_left_prop", "back_right_prop", "front_left_prop", "front_right_prop"],
)

MATRICE_CLEAN_CFG = MultirotorCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_M350_CLEAN_USD_PATH,
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
            "back_left_prop": 68.98,
            "back_right_prop": 68.98,
            "front_left_prop": 68.98,
            "front_right_prop": 68.98,
        },
    ),
    actuators={"thrusters": MATRICE_CLEAN_THRUSTER},
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
