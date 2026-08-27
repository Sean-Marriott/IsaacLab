# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_assets.robots.arl_robot_1 import MATRICE_CLEAN_CFG

from .track_trajectory_env_cfg import TrackTrajectoryEnvCfg


@configclass
class MatriceCleanTrajectoryEnvCfg(TrackTrajectoryEnvCfg):
    """Trajectory following on the DJI M350 without the chainsaw payload."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = MATRICE_CLEAN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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
        # K_vel is *not* set to the attitude bandwidth omega_n = 12.84 rad/s, which is the 1:1
        # cascade rule used for the chainsaw model. That rule only stays sane there because the
        # pendulum inflates I_att to 2.65; on this airframe it would command 53 deg of tilt per m/s
        # of velocity error. A 3.2:1 separation (tau_vel = 0.25 s) gives 12.84 / 3.2 = 4.0, which is
        # 22 deg per m/s and matches typical PX4 tuning. The vertical axis has no attitude loop in
        # the way, so it takes the same value rather than the usual 0.6 factor.
        #
        # Ranges are +/-10 % around nominal and are resampled per environment on every reset.
        ctrl = self.actions.velocity_command.controller_cfg
        ctrl.K_rot_range = ((115.0, 115.0, 46.0), (140.6, 140.6, 56.2))
        ctrl.K_angvel_range = ((15.2, 15.2, 11.1), (18.6, 18.6, 13.6))
        ctrl.K_vel_range = ((3.6, 3.6, 3.6), (4.4, 4.4, 4.4))

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
        # Lateral gains are lower than the vertical one despite sharing K_vel = 4.0. The vertical
        # integral acts on thrust directly, whereas x and y act through the attitude loop and pick
        # up its lag. At 1.5 the velocity loop is s^2 + 4 s + 1.5 (zeta = 1.63, w_n = 1.22 rad/s),
        # a tenth of the 12.84 rad/s attitude bandwidth -- overdamped and well separated. The
        # vertical axis keeps 2.5, against the 4.0 that would be critically damped.
        #
        # Note that max_integral_acc (2.0 m/s^2, the default) is a single bound shared by all three
        # axes, and it is sized for the vertical thrust shortfall. Laterally that is roughly 11 deg
        # of tilt, far more than the 0.23 m/s^2 gust or the 0.10 m/s^2 peak drag needs. It bounds
        # windup rather than shaping the response, so the slack is tolerable, but a lateral-only
        # bound would be tighter if the term ever proves to overshoot.
        ctrl.K_vel_int_range = ((1.35, 1.35, 2.25), (1.65, 1.65, 2.75))


@configclass
class MatriceCleanTrajectoryEnvCfg_PLAY(MatriceCleanTrajectoryEnvCfg):
    """Evaluation variant: few environments, no corruption, hardest reference."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        # The reference can reach 4.8 m from its centre in each direction, so a tight grid makes
        # the viewport unreadable. Inter-environment collisions are filtered, so this is cosmetic.
        self.scene.env_spacing = 18.0
        self.observations.policy.enable_corruption = False
        # Silence the gust so runs are repeatable, but keep the drag: it is a real aerodynamic
        # effect the vehicle will fly through, not a disturbance to be evaluated without.
        self.events.wind_and_drag.params["force_range"] = (0.0, 0.0)
        self.events.wind_and_drag.params["torque_range"] = (0.0, 0.0)
        # Evaluate at full difficulty rather than wherever a ramp happens to be.
        self.curriculum.trajectory_difficulty = None
        self.commands.trajectory.difficulty = 1.0
