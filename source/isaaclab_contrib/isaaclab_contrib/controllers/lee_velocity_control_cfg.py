# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass

from .lee_controller_base_cfg import LeeControllerBaseCfg
from .lee_velocity_control import LeeVelController


@configclass
class LeeVelControllerCfg(LeeControllerBaseCfg):
    """Configuration for a Lee-style geometric quadrotor velocity controller.

    Unless otherwise noted, vectors are ordered as (x, y, z) in the simulation world/body frames.
    The velocity controller gains are sampled uniformly per environment between
    their corresponding ``*_min`` and ``*_max`` bounds at reset.
    """

    class_type: type = LeeVelController
    """The class type for the velocity controller."""

    K_vel_range: tuple[tuple[float, float, float], tuple[float, float, float]] = MISSING
    """Velocity error proportional gain range about body axes [1/s].

    This is a tuple of two tuples containing the minimum and maximum gains for each axis (x, y, z).
    Format: ((min_x, min_y, min_z), (max_x, max_y, max_z))

    Example:
        ((2.5, 2.5, 1.5), (3.5, 3.5, 2.0)) for ARL Robot 1
    """

    K_vel_int_range: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    )
    """Velocity error integral gain range about world axes [1/s^2]. Zero on every axis by default,
    which disables integral action and reproduces the purely proportional controller.

    Same format as :attr:`K_vel_range`, sampled per environment at reset.

    Enable this when the vehicle the policy will be deployed on holds its velocity setpoint without
    a standing error -- that is, when commanding zero velocity makes it hold station. A purely
    proportional loop cannot do that: any steady force the controller does not model (a thrust
    shortfall, a mass error, a sustained wind) leaves a standing velocity error proportional to it.
    In a learned outer loop that error does not go unanswered, it gets absorbed into a constant
    offset on the policy's velocity command, and that offset is then wrong on a vehicle whose own
    loop would have rejected the disturbance. See :attr:`max_integral_acc` for the bound.

    Choose the gain against the proportional one: the velocity loop behaves as
    ``s^2 + K_vel s + K_vel_int``, so ``K_vel_int = K_vel^2 / 4`` is critically damped and smaller
    values are slower and safer. Integral action interacts with motor lag and with the attitude
    loop, so prefer the slow end.
    """

    max_integral_acc: float = 2.0
    """Anti-windup bound on the acceleration the integral term may contribute [m/s^2].

    The integral state is clamped so its contribution never exceeds this on any axis, which keeps a
    saturated or diverged episode from accumulating a large offset that then has to unwind. Size it
    to cover the steady disturbance being rejected with some margin, and well below gravity: a bound
    approaching 9.81 would let the integral term alone cancel the vehicle's weight.
    """
