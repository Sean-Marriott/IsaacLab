# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory following against an inner loop detuned to match the deployed PX4 controller.

The policy trained by :mod:`.env_cfg` learns against a Lee velocity controller tuned to what the
M350 airframe can actually do. The vehicle does not fly that controller: the velocity setpoint goes
to PX4, whose cascade is materially slower laterally. A policy trained on the fast loop meets a slow
one on deployment, leads it by the wrong amount, and trails the reference by roughly
``lag * reference_speed``.

This configuration closes that gap from the simulation side, by making the training plant respond
like the deployment plant instead of like the airframe's potential.

**What the deployed loop actually does.** Measured from a PX4 SITL ulog (``trajectory_setpoint``
against ``vehicle_local_position``, restricted to samples with commanded speed above 0.3 m/s, lag
taken as the shift minimising mean absolute error):

======================================  =========  ====================================
Loop                                    Lag        Note
======================================  =========  ====================================
velocity command -> achieved, lateral   0.26 s     the deficit this config reproduces
velocity command -> achieved, vertical  0.00 s     ``MPC_Z_VEL_P_ACC`` 4.0 == trained 4.0
attitude setpoint -> achieved           0.16 s     ``1 / MC_ROLL_P`` = 1/6.5 = 0.154 s
======================================  =========  ====================================

The asymmetry is the whole point. Vertical velocity comes from thrust directly and PX4's vertical
gain already equals the trained one, so it tracks with no lag and is left alone here. Lateral
velocity has to be produced by tilting, and PX4's attitude loop runs at about half the bandwidth
the Lee controller was tuned for -- that is where the 0.26 s goes.

**What is deliberately not changed.** The speed envelope (``max_velocity``, the trajectory ranges,
the observation clips) is untouched, so this config differs from :mod:`.env_cfg` in the plant alone
and the two are directly comparable. The envelope has its own mismatch with the vehicle -- the
platform clamps to 2.0 m/s against a 3.0 m/s action scale -- but that is a separate decision, and
folding it in here would confound the two effects.
"""

from isaaclab.utils import configclass

from .env_cfg import MatriceCleanTrajectoryEnvCfg, MatriceCleanTrajectoryEnvCfg_PLAY

# Target attitude natural frequency [rad/s]. PX4 reaches its attitude setpoint with a 0.16 s lag,
# and its attitude loop is a proportional one at ``MC_ROLL_P = 6.5``, giving 1/6.5 = 0.154 s. The
# two agree closely enough to take 6.5 rad/s as the bandwidth to reproduce, against the 12.84 rad/s
# the clean-airframe config is tuned to.
ATTITUDE_OMEGA_N = 6.5

# PhysX inertia of the clean airframe, as measured in :mod:`.env_cfg`. Repeated rather than imported
# because the design equations below are only meaningful next to the numbers they consume; if the
# airframe model changes, both files have to be re-derived anyway.
I_ATT = 0.7754
I_ZZ = 1.0286


def _attitude_gains(omega_n: float) -> tuple[float, float, float, float]:
    """Return ``(K_rot_xy, K_angvel_xy, K_rot_z, K_angvel_z)`` for an attitude bandwidth.

    Uses the same design equations as :mod:`.env_cfg`, so the detuned gains sit on the same damping
    curve as the tuned ones and differ only in bandwidth::

        K_rot    = I_att * omega_n**2
        K_angvel = 2 * zeta * omega_n * I_att            (zeta = 0.85)
        Yaw: K_rot_z = 0.4 * K_rot_xy, K_angvel_z from the same formula with I_zz

    Args:
        omega_n: Target attitude natural frequency [rad/s].

    Returns:
        The four nominal gains, in the units the controller config expects.
    """
    zeta = 0.85
    k_rot_xy = I_ATT * omega_n**2
    k_angvel_xy = 2.0 * zeta * omega_n * I_ATT
    k_rot_z = 0.4 * k_rot_xy
    k_angvel_z = 2.0 * zeta * (k_rot_z / I_ZZ) ** 0.5 * I_ZZ
    return k_rot_xy, k_angvel_xy, k_rot_z, k_angvel_z


def _range(nominal_xy: float, nominal_z: float, spread: float = 0.10):
    """Return a ``((min_x, min_y, min_z), (max_x, max_y, max_z))`` band around two nominals.

    Args:
        nominal_xy: Nominal value for the x and y axes.
        nominal_z: Nominal value for the z axis.
        spread: Fractional half-width of the band. Defaults to the +/-10 % used throughout.

    Returns:
        The gain range, in the nested-tuple form the controller config expects.
    """
    lo, hi = 1.0 - spread, 1.0 + spread
    return (
        (round(nominal_xy * lo, 3), round(nominal_xy * lo, 3), round(nominal_z * lo, 3)),
        (round(nominal_xy * hi, 3), round(nominal_xy * hi, 3), round(nominal_z * hi, 3)),
    )


def _apply_px4_matched_gains(cfg: MatriceCleanTrajectoryEnvCfg):
    """Detune ``cfg``'s velocity controller onto PX4's measured response.

    Args:
        cfg: The environment configuration to modify in place.
    """
    k_rot_xy, k_angvel_xy, k_rot_z, k_angvel_z = _attitude_gains(ATTITUDE_OMEGA_N)
    ctrl = cfg.actions.velocity_command.controller_cfg

    # -- attitude: the loop that carries the lateral lag.
    ctrl.K_rot_range = _range(k_rot_xy, k_rot_z)
    ctrl.K_angvel_range = _range(k_angvel_xy, k_angvel_z)

    # -- velocity: PX4's own gains, MPC_XY_VEL_P_ACC and MPC_Z_VEL_P_ACC.
    #
    # Note that the lateral gain goes *up* here, from 4.0 to 5.0. The lag is not in the velocity
    # loop -- PX4's velocity gain is already faster than the trained one -- so matching PX4 means
    # taking its faster outer gain along with its slower attitude loop. The consequence is that the
    # cascade separation falls from 3.2:1 to 6.5/5.0 = 1.3:1, which is far tighter than the ratio
    # env_cfg's design note calls sane on this airframe. That is not an oversight: it is what the
    # vehicle actually flies, and PX4 gets away with it partly through a derivative term
    # (MPC_XY_VEL_D_ACC = 2.0) that the Lee controller has no equivalent of. Verify the step
    # response before committing a long run to it -- see the module docstring of
    # ``scripts/demos/matrice_vel_demo.py``, which takes a ``--cascade_ratio``.
    ctrl.K_vel_range = _range(5.0, 4.0)

    # -- integral: PX4's MPC_XY_VEL_I_ACC and MPC_Z_VEL_I_ACC. Both are weaker than the tuned
    #    config's, the lateral one substantially so (0.4 against 1.5).
    ctrl.K_vel_int_range = _range(0.4, 2.0)

    # -- yaw rate: MPC_YAWRAUTO_MAX is 45 deg/s. The action scale still maps a unit action to
    #    pi/2 rad/s, exactly as on the vehicle, so the controller clamp is what the policy meets --
    #    which is the behaviour being reproduced, not a bug.
    ctrl.max_yaw_rate = 0.7854

    # note: max_inclination_angle_rad is left at pi/4, which already equals PX4's
    # MPC_TILTMAX_AIR of 45 degrees, so the tilt limit needs no adjustment to match.


@configclass
class MatriceDetunedTrajectoryEnvCfg(MatriceCleanTrajectoryEnvCfg):
    """Trajectory following with the inner loop detuned onto PX4's measured response."""

    def __post_init__(self):
        super().__post_init__()
        _apply_px4_matched_gains(self)


@configclass
class MatriceDetunedTrajectoryEnvCfg_PLAY(MatriceCleanTrajectoryEnvCfg_PLAY):
    """Evaluation variant of :class:`MatriceDetunedTrajectoryEnvCfg`."""

    def __post_init__(self):
        super().__post_init__()
        _apply_px4_matched_gains(self)
