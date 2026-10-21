# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Chainsaw swing suppression while aligning with a target pose.

This combines the chainsaw trajectory task with the behaviour of the direct-velocity position-tracking
task (``track_position_direct_vel``): the drone spawns away from a fixed target pose, flies to it and
lines up with it, while keeping the payload swing energy low. It is built on the trajectory task, but
trajectory following is currently disabled (see :class:`MatriceChainsawSwingEnvCfg`). The swing
starts large and is excited again during the episode, so there is always some swing to damp.

The direct-velocity task's terms do not carry over as they are:

* ``PoleEnergy`` takes the pendulum length from one body's COM in its own frame. On the Mk3 the
  saw's COM sits near its own origin, 1.1 m from the hinge, so that length would be close to zero.
  :class:`PayloadSwingEnergy` takes ``m`` and ``L`` from the whole payload instead, and measures the
  swing against gravity rather than through the hinge angles, which are relative to the airframe.
* Its ``-100`` energy penalty is replaced by a bounded kernel, like every other shaped term of the
  trajectory reward (see ``RewardsCfg`` in ``track_trajectory_env_cfg``).
* ``push_pole`` uses ``apply_external_force_torque``, which writes a *permanent* wrench, so it holds
  up to 50 N on the saw until the next push: a steady lean, not a swing. The kick here adds hinge
  velocity instead, which is an impulse.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.drone_arl.mdp as mdp
from isaaclab_tasks.manager_based.drone_arl.mdp.commands import DroneTrajectoryCommandCfg
from isaaclab_tasks.manager_based.drone_arl.mdp.events import push_joints_by_velocity
from isaaclab_tasks.manager_based.drone_arl.mdp.rewards import DroneEndPoseExp, PayloadSwingEnergyExp

from .chainsaw_env_cfg import (
    AIRFRAME_BODY_NAME,
    END_HOLD_S,
    END_SLOWDOWN_S,
    ChainsawEventCfg,
    ChainsawRewardsCfg,
    MatriceChainsawTrajectoryEnvCfg,
)
from .track_trajectory_env_cfg import COMMAND_NAME

HINGE_JOINT_NAMES = ["csTubePitch", "csTubeRoll"]
"""Two-axis gimbal at the top of the tube, outer axis first. Both are zero at the hanging rest pose."""

PAYLOAD_BODY_NAMES = ["main_tube", "base_link", "roll_link", "feed_arm_link"]
"""Every body below the gimbal: the tube, the saw-head mount, the saw and its feed arm (~2.0 kg)."""

PIVOT_BODY_NAME = "top_hinge"
"""Body between the two hinges. Its link origin is on the ``csTubePitch`` axis; ``csTubeRoll`` sits 13 mm
further along the tube, so at rest both axes lie on the vertical through the payload COM."""


STATIONARY_REFERENCE = DroneTrajectoryCommandCfg.Ranges(
    vel_rms_target=(0.0, 0.0),
    pos_cap=(0.0, 0.0),
    vel_cap=(0.0, 0.0),
    acc_cap=(0.0, 0.0),
    pos_z_cap=(0.0, 0.0),
    vel_z_cap=(0.0, 0.0),
    acc_z_cap=(0.0, 0.0),
)
"""Trajectory limits that hold the reference still at its start point.

The sampler scales every path down to fit its limits, so zero limits scale it to nothing. The yaw
reference is sampled separately and still takes a random constant heading.
"""


@configclass
class SwingSuppressionEventCfg(ChainsawEventCfg):
    """Configuration for events, with a large initial swing and the swing kicked during the episode."""

    # -- reset
    # The chainsaw task resets all three payload joints by +/-0.3 rad. Here the hinge gets its own,
    # much wider ranges per axis, so damping a large swing is part of every episode from the first
    # step, while the saw-head pitch keeps the chainsaw range: it is not part of the swing.
    reset_chainsaw_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.3, 0.3),  # ~17 deg
            "velocity_range": (-0.1, 0.1),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["root_joint"]),
        },
    )
    # Up to ~57 deg, against limits at ~+/-90 deg. At the edge this is ~10 J of swing energy.
    reset_chainsaw_swing_pitch = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.3, 0.3),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["csTubePitch"]),
        },
    )
    # Up to ~40 deg, leaving a margin to the hard limits at +/-45 deg so a reset never starts the
    # swing against a limit.
    reset_chainsaw_swing_roll = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.7, 0.7),
            "velocity_range": (-0.3, 0.3),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["csTubeRoll"]),
        },
    )

    # -- interval
    # A branch strike or a snagged chain, modelled as an impulse on the hinge. At the edge of the
    # range on both axes this puts about 0.6 J into the swing, roughly a 13 deg amplitude, which the
    # hinge damping alone takes several seconds to remove.
    kick_chainsaw_swing = EventTerm(
        func=push_joints_by_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={
            "velocity_range": (-0.5, 0.5),
            "asset_cfg": SceneEntityCfg("robot", joint_names=HINGE_JOINT_NAMES),
        },
    )


@configclass
class SwingSuppressionRewardsCfg(ChainsawRewardsCfg):
    """Reward terms of the chainsaw task, with a target pose for the airframe at the end instead of a
    target point for the cutting mouth, plus a bounded reward for a still payload."""

    # -- end pose
    # The chainsaw task rewards the cutting mouth for reaching where it would hang below the end point.
    # Here the airframe is rewarded for matching an explicit target pose: level, at the end point and
    # at the end yaw. Same windows, position kernels and weights as the chainsaw terms, so the end of
    # the episode is worth as much as before.
    tool_end_target = None
    tool_end_target_fine = None
    # The coarse term steers the approach from the start of the slowdown. Its 0.5 rad orientation
    # kernel barely constrains the tilt the vehicle needs to decelerate, but still pulls the yaw round.
    drone_end_pose = RewTerm(
        func=DroneEndPoseExp,  # type: ignore[arg-type]
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[AIRFRAME_BODY_NAME]),
            "std_pos": 0.25,
            "std_rot": 0.5,
            "window_s": END_SLOWDOWN_S + END_HOLD_S,
            "command_name": COMMAND_NAME,
            # Large frame on the target pose, small frame on the airframe.
            "debug_vis": True,
        },
    )
    # The fine term is active only during the hold, where the vehicle should be hovering still, so
    # a tilt there means it is still accelerating (or the payload is swinging it about).
    drone_end_pose_fine = RewTerm(
        func=DroneEndPoseExp,  # type: ignore[arg-type]
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[AIRFRAME_BODY_NAME]),
            "std_pos": 0.07,
            "std_rot": 0.15,
            "window_s": END_HOLD_S,
            "command_name": COMMAND_NAME,
            # The coarse term already draws the markers; a second set would reuse its prim paths.
            "debug_vis": False,
        },
    )

    # -- swing

    # Falls to 1/e at a swing of about 9.5 deg from vertical (see PayloadSwingEnergyExp). Worth up to
    # 0.5 per step, the same as the fine position kernel: enough that the policy spends tracking
    # accuracy on damping a swing, not so much that it would trade the path for a still payload.
    #
    # The swing is measured against gravity, so the lean the payload takes while the vehicle
    # accelerates is penalized too: 1 m/s^2 holds it at ~6 deg (kernel ~0.7), and the 3 m/s^2 peak of
    # the hardest reference at ~17 deg (kernel ~0.04). Raise energy_scale if that costs too much
    # tracking on the fast references.
    swing_suppression = RewTerm(
        func=PayloadSwingEnergyExp,  # type: ignore[arg-type]
        weight=0.5,
        params={
            "payload_cfg": SceneEntityCfg("robot", body_names=PAYLOAD_BODY_NAMES),
            "pivot_cfg": SceneEntityCfg("robot", body_names=[PIVOT_BODY_NAME]),
            "energy_scale": 0.3,
        },
    )


@configclass
class MatriceChainsawSwingEnvCfg(MatriceChainsawTrajectoryEnvCfg):
    """Chainsaw swing suppression on the DJI M350: fly to and align with a target pose while damping the
    swing (trajectory following disabled)."""

    events: SwingSuppressionEventCfg = SwingSuppressionEventCfg()
    rewards: SwingSuppressionRewardsCfg = SwingSuppressionRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # -- target-pose alignment instead of trajectory following
        # This reproduces the behaviour of ``track_position_direct_vel``: the drone spawns away from a
        # fixed target pose -- displaced, turned and tilted -- and has to fly to it, line up with it and
        # hold it while damping the chainsaw swing.
        #
        # Nothing of the trajectory machinery is removed. The target is the trajectory reference held
        # still: zero limits collapse the sampled path onto its start point (the environment origin),
        # the yaw reference supplies a random target heading, and the difficulty ramp has nothing left
        # to scale. The observations, terminations and markers keep reading the same command, so the
        # policy interface is unchanged.
        #
        # To follow a trajectory again, comment out this whole block.
        self.commands.trajectory.ranges = STATIONARY_REFERENCE
        self.curriculum.trajectory_difficulty = None
        # TODO: Drop lookahead_v (``self.observations.policy.lookahead_v = None``). With the reference
        # held still, every look-ahead point is the target, so the term is pos_error_v repeated five
        # times, but without its noise and delay: the policy can read a clean position error from it
        # and bypass the corruption on pos_error_v. Dropping it changes the observation size (43 -> 28),
        # so checkpoints from the trajectory task no longer load.

        # Spawn ranges of track_position_direct_vel, relative to the target pose. The yaw covers every
        # heading, so lining up can mean turning round.
        self.events.reset_base.params.update(
            {
                "pos_noise": (-2.0, 2.0),
                "vel_noise": (-0.2, 0.2),
                "yaw_noise": (-math.pi, math.pi),
                "rp_noise": (-math.pi / 6.0, math.pi / 6.0),
                "angvel_noise": (-0.2, 0.2),
            }
        )
        # The spawn can be up to 3.5 m from the target, beyond the 3.75 m tracking limit once the
        # initial velocity carries it further, so the limit is widened to keep it for real divergence.
        self.terminations.tracking_divergence.params["max_error"] = 5.0

        # The tracking kernels were sized for staying on a path, a few tens of centimetres off it. Here
        # the error starts at metres and up to pi rad, where they are flat, so they are widened to
        # give a gradient over the whole spawn range: 0.17 at 2 m, 0.05 at 3.5 m; 0.01 at pi rad.
        # The fine position term keeps its width and supplies the precision near the target.
        self.rewards.pos_tracking.params["std"] = 2.0
        self.rewards.yaw_tracking.params["std"] = 1.5
        self.rewards.yaw_tracking.weight = 0.5
        # The target pose is rewarded over the whole episode, not just at the end. The fine term is
        # halved, since it is now available for the whole episode rather than only the 3 s hold.
        self.rewards.drone_end_pose.params["window_s"] = self.episode_length_s
        self.rewards.drone_end_pose_fine.params["window_s"] = self.episode_length_s
        self.rewards.drone_end_pose_fine.weight = 2.0
        # Swing suppression is half of the task here, so it is weighted like a pose term.
        self.rewards.swing_suppression.weight = 1.0


@configclass
class MatriceChainsawSwingEnvCfg_PLAY(MatriceChainsawSwingEnvCfg):
    """Evaluation variant: few environments, no corruption, regular kicks (and the hardest reference
    when trajectory following is enabled)."""

    def __post_init__(self):
        super().__post_init__()
        # Same overrides as MatriceChainsawTrajectoryEnvCfg_PLAY.
        self.scene.num_envs = 50
        self.scene.env_spacing = 18.0
        self.observations.policy.enable_corruption = False
        self.events.wind_and_drag.params["force_range"] = (0.0, 0.0)
        self.events.wind_and_drag.params["torque_range"] = (0.0, 0.0)
        self.curriculum.trajectory_difficulty = None
        self.commands.trajectory.difficulty = 1.0
        # Keep the kicks, since they are what this task is evaluated on, but on a fixed beat so
        # every environment shows the same disturbance-and-recovery pattern.
        self.events.kick_chainsaw_swing.interval_range_s = (5.0, 5.0)
        # Spawn the payload hanging at rest, so any swing seen comes from the flight or the kicks.
        for term in (self.events.reset_chainsaw_swing_pitch, self.events.reset_chainsaw_swing_roll):
            term.params["position_range"] = (0.0, 0.0)
            term.params["velocity_range"] = (0.0, 0.0)
        # Spawn the drone at rest too, so its initial motion doesn't start the payload swinging.
        self.events.reset_base.params["vel_noise"] = (0.0, 0.0)
        self.events.reset_base.params["angvel_noise"] = (0.0, 0.0)

        # -- saw head locked at its rest pitch
        # Holds ``root_joint`` at zero (the hanging rest pose) with a stiff drive, in place of the
        # 0.0105 N·m/rad springs, so the payload swings as one rigid pendulum about the tube hinge.
        # Gravity and swing loads on the head are ~1 N·m, so 1e4 N·m/rad keeps it within ~0.1 mrad.
        # The joint stays in the observation and reads ~0, so trained checkpoints still load.
        # To let the saw head swing freely again, comment out this block.
        self.events.chainsaw_springs.params["stiffness"] = 1.0e4
        self.events.chainsaw_springs.params["scale_range"] = (1.0, 1.0)
        self.events.reset_chainsaw_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_chainsaw_joints.params["velocity_range"] = (0.0, 0.0)
