# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_assets.robots.arl_robot_1 import MATRICE_CFG

from .track_trajectory_env_cfg import TrackTrajectoryEnvCfg


@configclass
class MatriceCleanTrajectoryEnvCfg(TrackTrajectoryEnvCfg):
    """Trajectory following on the DJI M350 without the chainsaw payload."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = MATRICE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["thrusters"].dt = self.sim.dt

        # Gains derived from the PhysX inertia measured on the clean airframe:
        #   mass = 6.500 kg, I_xx = 0.7754, I_yy = 0.7560, I_zz = 1.0286 kg*m^2
        #   hover_thrust/motor = 15.94 N (max 55 N, thrust-to-weight 3.45)
        #   arm_pitch_sum = 1.2686 m  ->  tau_max = 49.55 N*m
        #   nominal: K_rot_xy = 127.81, K_angvel_xy = 16.92, K_rot_z = 51.13, K_angvel_z = 12.33
        #
        # Design equations, for re-tuning after a model change:
        #   K_rot    = min(tau_max / (0.5 * I_att), 200)   tau_max = thrust headroom * min arm sum
        #   K_angvel = 2 * 0.85 * sqrt(K_rot / I_att) * I_att      (zeta = 0.85)
        #   Yaw: K_rot_z = 0.4 * K_rot_xy, K_angvel_z from the same formula with I_zz
        #
        # Velocity gains are fitted to real M350 flights: the velocity command sent to the DJI flight
        # controller against the fused ground velocity, over open-loop body-forward steps (no
        # payload) and the spans of the chainsaw swing-suppression flights where the vehicle was
        # following the offboard command (2026-10-28). The fit model was this controller's PI law
        # behind the attitude loop above, with a pure command delay.
        #
        # Laterally, delay and gain trade off: anything from 2.5 at no delay to 8.0 at 0.2 s fits
        # within 3 % of the best, and both data sets put the best at about 4. The nominal stays at
        # 4.0 (a 3.2:1 separation from the 12.84 rad/s attitude loop, 22 deg of tilt per m/s of
        # error) and the range is widened to 3.0-5.5 to cover that ridge, since the delay itself is
        # not modelled. Vertically the fit is sharp: 3.0 with a 0.1 s delay, against 4.0 with no
        # delay before, which roughly halves the tracking error on the flights.
        #
        # Ranges are resampled per environment on every reset.
        ctrl = self.actions.velocity_command.controller_cfg
        ctrl.K_rot_range = ((115.0, 115.0, 46.0), (140.6, 140.6, 56.2))
        ctrl.K_angvel_range = ((15.2, 15.2, 11.1), (18.6, 18.6, 13.6))
        ctrl.K_vel_range = ((3.0, 3.0, 2.7), (5.5, 5.5, 3.3))

        # Integral action on all three axes.
        #
        # The vehicle this policy deploys to holds station when commanded zero velocity, so its own
        # velocity loop rejects steady disturbances. A purely proportional loop here does not, and
        # the difference does not stay in the controller: whatever standing velocity error the sim
        # leaves, the policy learns to cancel with a constant offset on its velocity command, and
        # that offset becomes an uncommanded drift once the vehicle's own loop is doing the
        # rejecting. Matching the structure is what keeps the action meaning the same thing in both
        # places -- and that argument is about the loop structure, so it applies to x and y exactly
        # as it does to z.
        #
        # The lateral disturbances are not the zero-mean pair they look like. Drag opposes motion
        # and is quadratic in speed, so along a tracked reference it is a sustained, direction-
        # correlated force, not a wash; and the gust is held for 2-6 s at a time, which is long
        # relative to the 0.25 s velocity loop. Both leave a standing lateral error a proportional
        # loop cannot close.
        #
        # The gain is small, though. On the flights above the fit got steadily worse as the integral
        # grew, on every axis: over the 3-10 s spans available the vehicle shows no integral action
        # it could be told apart from noise, and the previous 1.5 lateral / 2.5 vertical fitted the
        # flights measurably worse than no integral at all. Values up to about 0.5-1.0 stay
        # within the noise, so 0.5 keeps a slow station-holding term without contradicting the data.
        # At 0.5 the velocity loop is s^2 + 4 s + 0.5 laterally and s^2 + 3 s + 0.5 vertically, both
        # heavily overdamped, so the term trims a standing error over seconds rather than shaping
        # the step response.
        #
        # Note that max_integral_acc (2.0 m/s^2, the default) is a single bound shared by all three
        # axes, and it is sized for the vertical thrust shortfall. It bounds windup rather than
        # shaping the response, so the slack laterally is tolerable.
        ctrl.K_vel_int_range = ((0.45, 0.45, 0.45), (0.55, 0.55, 0.55))

    def play_mode(self):
        """Evaluation overrides: few environments, no corruption, hardest reference."""
        super().play_mode()
        # The reference can reach 4.8 m from its centre in each direction, so a tight grid makes
        # the viewport unreadable. Inter-environment collisions are filtered, so this is cosmetic.
        self.scene.env_spacing = 18.0
        # Silence the gust so runs are repeatable, but keep the drag: it is a real aerodynamic
        # effect the vehicle will fly through, not a disturbance to be evaluated without.
        self.events.wind_and_drag.params["force_range"] = (0.0, 0.0)
        self.events.wind_and_drag.params["torque_range"] = (0.0, 0.0)
        # Evaluate at full difficulty rather than wherever a ramp happens to be.
        self.curriculum.trajectory_difficulty = None
        self.commands.trajectory.difficulty = 1.0
