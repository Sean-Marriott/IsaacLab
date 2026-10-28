# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory following on the M350 with the Mk3 chainsaw, where the reference is for the tool, not the drone.

In the chainsaw task the airframe follows the reference and the cutting mouth is only rewarded at the
end, for hanging where it would below the end point. Here the reference position, velocity and yaw are
for a tool frame on the cutting mouth instead, over the whole episode. The airframe has to fly
whatever path puts the hanging tool on the reference, leading it through the swing. For a load
hanging from a multirotor the load position is a flat output, so that path exists for any smooth
reference.

The tool frame sits on the cutting mouth, oriented so that it lines up with the vehicle yaw frame
when the payload hangs at rest. Its yaw is therefore the vehicle heading the saw implies, and the
reference yaw keeps its meaning. The tube gimbal has no yaw axis, so the policy steers the tool yaw
directly with the vehicle yaw.

The observation layout is that of the chainsaw task, with the tracking errors taken against the tool
frame (still expressed in the vehicle frame, which is the frame of the action), so checkpoints from
the chainsaw task load as a starting point.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.drone_arl.mdp as mdp
from isaaclab_tasks.contrib.drone_arl.mdp.commands import DroneTrajectoryCommandCfg
from isaaclab_tasks.contrib.drone_arl.mdp.events import (
    RandomizePayloadHangOffset,
    push_swing_by_velocity,
    reset_swing_by_amplitude,
)
from isaaclab_tasks.contrib.drone_arl.mdp.rewards import (
    PayloadSwingEnergyExp,
    ToolEndTargetExp,
    position_tracking_exp,
    velocity_tracking_exp,
    yaw_tracking_exp,
)
from isaaclab_tasks.contrib.drone_arl.mdp.terminations import (
    PositionErrorAboveMaximumAfterCapture,
    tracked_position_error_above_maximum,
)

from .chainsaw_env_cfg import (
    AIRFRAME_BODY_NAME,
    CHAINSAW_HANG_OFFSET,
    CHAINSAW_MOUTH_OFFSET,
    END_SLOWDOWN_S,
    ChainsawEventCfg,
    MatriceChainsawTrajectoryEnvCfg,
)
from .swing_suppression_env_cfg import (
    HINGE_JOINT_NAMES,
    HOLD_RADIUS_MAX,
    PAYLOAD_BODY_NAMES,
    PIVOT_BODY_NAME,
    SAFETY_RADIUS,
)
from .track_trajectory_env_cfg import COMMAND_NAME, TerminationsCfg

TOOL_BODY_NAME = "roll_link"
"""The saw body, which carries the cutting mouth."""

TOOL_END_HOLD_S = 6.0
"""Time the reference holds its end point before the episode ends [s].

Twice the chainsaw task's 3 s. The fine end-target term is active only during the hold, and at 3 s of
a 20 s episode the policy took less than half of it after 2000 iterations, where the swing task, with
its fine pose term active over the whole episode, took nearly 80%. The end target is the end of the
reference, so a longer hold is no reason to leave the path, and the kicks that land in it give the
policy more practice bringing the tool back.
"""

SWING_NATURAL_FREQUENCY = 3.0
"""Natural frequency of the payload swing about the tube hinge [rad/s] (a ~2.1 s period on the Mk3)."""

PLAY_SPEED_SCALE = 0.3
"""Factor on the reference's speed, acceleration and excursion in play mode. At 0.3 the hardest reference
peaks at about 0.7 m/s and 0.9 m/s^2, against 2.25 m/s and 3.0 m/s^2 in training."""

HANG_OFFSET_EVENT_NAME = "randomize_hang_offset"
"""Name of the event term that holds the randomized rest pose the swing terms act about."""


CHAINSAW_TOOL_ROT = (0.5009055, 0.4942847, 0.5489637, 0.4510157)
"""Orientation of the tool frame in the ``roll_link`` body frame, as a quaternion (x, y, z, w).

It is the inverse of the ``roll_link`` orientation in the yaw-aligned vehicle frame when the payload
hangs at rest, so at rest the tool frame lines up with the vehicle yaw frame: +x forward, +z up. The
saw bar (+y of ``roll_link``) then points about 6 deg to the right of the tool +x. Measured in PhysX
with MATRICE_MK3_CFG at its default joint angles, like CHAINSAW_HANG_OFFSET; re-measure if the USD or
those change.
"""

CHAINSAW_TOOL_FRAME = DroneTrajectoryCommandCfg.OffsetCfg(pos=CHAINSAW_MOUTH_OFFSET, rot=CHAINSAW_TOOL_ROT)
"""The tracked tool frame: on the cutting mouth, lined up with the vehicle yaw frame at rest."""


@configclass
class ChainsawToolEventCfg(ChainsawEventCfg):
    """Configuration for events, with the payload's rest pose randomized and its swing kicked during the episode.

    The swing at reset and the kicks are drawn so that every scale of swing is equally likely, and a
    share of each is no swing at all (see :func:`reset_swing_by_amplitude`). Drawing each hinge axis
    uniformly instead makes small swings rare, so the policy would learn to expect a large swing at the
    start of every flight and after every disturbance, and would act on one that is not there.
    """

    # -- startup
    # The real Mk3 hangs at about 5.4 deg forward and 3 deg right, so its hinge encoders read that at
    # rest. A policy trained only on a payload that rests at zero reads it as swing. +/-0.12 rad (~7
    # deg) per hinge covers the measured offset with margin and includes no offset at all. Defined
    # after randomize_chainsaw_mass, so the COM shift uses the randomized saw mass.
    randomize_hang_offset = EventTerm(
        func=RandomizePayloadHangOffset,  # type: ignore[arg-type]
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=HINGE_JOINT_NAMES),
            "payload_cfg": SceneEntityCfg("robot", body_names=PAYLOAD_BODY_NAMES),
            "pivot_cfg": SceneEntityCfg("robot", body_names=[PIVOT_BODY_NAME]),
            "shifted_body_cfg": SceneEntityCfg("robot", body_names=[TOOL_BODY_NAME]),
            "airframe_cfg": SceneEntityCfg("robot", body_names=[AIRFRAME_BODY_NAME]),
            "offset_range": (-0.12, 0.12),
        },
    )

    # -- reset
    # A quarter of the episodes start with the payload hanging still at its rest pose. The others swing
    # from 1 to 40 deg, log-uniform (median ~7 deg; ~15% of all episodes over 20 deg), in any direction and at any
    # phase. 40 deg stays inside the +/-45 deg roll limit in every direction, as in the swing task's
    # hold evaluation. The saw head, which is locked at its rest pitch (see
    # MatriceChainsawToolTrajectoryEnvCfg), is reset to rest rather than against its stiff drive.
    reset_chainsaw_joints = EventTerm(
        func=reset_swing_by_amplitude,
        mode="reset",
        params={
            "amplitude_range": (0.02, 0.7),
            "still_prob": 0.25,
            "natural_frequency": SWING_NATURAL_FREQUENCY,
            "asset_cfg": SceneEntityCfg("robot", joint_names=HINGE_JOINT_NAMES),
            "rest_term_name": HANG_OFFSET_EVENT_NAME,
        },
    )
    reset_saw_head = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["root_joint"]),
        },
    )

    # The swing task's kick: a branch strike or a snagged chain, modelled as an impulse on the hinge.
    # A fifth of the kicks are skipped; the others add 0.05 to 0.7 rad/s, log-uniform, which is a swing
    # of about 1 to 13 deg (the top end, ~0.6 J, the same as the swing task's), knocking the mouth up
    # to ~0.25 m off the reference. They can land in the end hold too, so the policy learns to bring
    # the tool back onto the target rather than only to arrive.
    kick_chainsaw_swing = EventTerm(
        func=push_swing_by_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={
            "speed_range": (0.05, 0.7),
            "skip_prob": 0.2,
            "asset_cfg": SceneEntityCfg("robot", joint_names=HINGE_JOINT_NAMES),
        },
    )


@configclass
class ChainsawToolRewardsCfg:
    """Reward terms, with every tracking term on the tool frame.

    Every term is listed here rather than inherited, so the full reward is visible in one place. The
    tracking and regularization weights and widths match the base config.
    """

    # -- tracking (all read the tool frame from the command)
    pos_tracking = RewTerm(func=position_tracking_exp, weight=1.0, params={"std": 0.54, "command_name": COMMAND_NAME})
    pos_tracking_fine = RewTerm(
        func=position_tracking_exp, weight=0.5, params={"std": 0.15, "command_name": COMMAND_NAME}
    )
    vel_tracking = RewTerm(func=velocity_tracking_exp, weight=0.3, params={"std": 0.6, "command_name": COMMAND_NAME})
    yaw_tracking = RewTerm(func=yaw_tracking_exp, weight=0.2, params={"std": 0.35, "command_name": COMMAND_NAME})
    # The 0.35 rad kernel above is flat beyond about 45 deg (0.02 there), so a policy whose yaw has
    # drifted past that gets no signal to turn back, and the 100-iteration policy ended most episodes
    # 70-90 deg off. This wide kernel keeps a gradient over the whole range (0.76 at 45 deg, 0.33 at
    # 90 deg, 0.01 at 180 deg), like the 1.5 rad yaw kernel of the swing task. With the narrow term,
    # yaw is worth up to 0.5 per step, the same as in the swing task.
    yaw_tracking_coarse = RewTerm(func=yaw_tracking_exp, weight=0.3, params={"std": 1.5, "command_name": COMMAND_NAME})

    # -- regularization (same as the base config)
    ang_vel_penalty = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.002)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.0005)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)

    # -- end target
    # The chainsaw task's end-target pair, with the target on the end point itself rather than
    # below it, since the reference is now for the mouth. They add precision where it matters:
    # the coarse kernel from the start of the slowdown, the fine one over the hold, which here is
    # TOOL_END_HOLD_S rather than the chainsaw task's 3 s. With the longer hold the pair is worth up
    # to ~330 per episode against ~460 for tracking; it is still only earned at the end of the path.
    #
    # Both are joint kernels on the mouth position and the tool yaw, so the end target is a position
    # and a heading. On position alone, a policy could take almost all of their reward (up to 5 per
    # step during the hold, against 0.2 for yaw tracking) while facing 90 deg away from the target.
    #
    # The coarse yaw width is wide enough to pull the yaw round from a large error (0.37 at 1 rad,
    # 0.08 at 90 deg), not just to refine a small one; yaw_tracking alone is flat beyond about 45 deg.
    tool_end_target = RewTerm(
        func=ToolEndTargetExp,  # type: ignore[arg-type]
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[TOOL_BODY_NAME]),
            "std": 0.25,
            "window_s": END_SLOWDOWN_S + TOOL_END_HOLD_S,
            "tool_offset": CHAINSAW_MOUTH_OFFSET,
            "hang_offset": (0.0, 0.0, 0.0),
            # Defines the tool yaw, and draws the markers as frames: a small one on the tool frame, a
            # large one on the end target at the end yaw. Drawn only when rendering.
            "tool_rot": CHAINSAW_TOOL_ROT,
            "std_yaw": 1.0,
            "command_name": COMMAND_NAME,
            "debug_vis": True,
        },
    )
    tool_end_target_fine = RewTerm(
        func=ToolEndTargetExp,  # type: ignore[arg-type]
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[TOOL_BODY_NAME]),
            "std": 0.07,
            "window_s": TOOL_END_HOLD_S,
            "tool_offset": CHAINSAW_MOUTH_OFFSET,
            "hang_offset": (0.0, 0.0, 0.0),
            "tool_rot": CHAINSAW_TOOL_ROT,
            # 0.46 at 10 deg, 0.05 at 20 deg: the heading has to be right, not just roughly right.
            "std_yaw": 0.2,
            "command_name": COMMAND_NAME,
            # The coarse term already draws the markers; a second set would reuse its prim paths.
            "debug_vis": False,
        },
    )

    # -- swing
    # The swing task's term, with the same deadzone: swings under ~3 deg (0.03 J) score the full
    # reward, so the policy is not driven to rock the airframe chasing the last degree or two. It falls
    # to 1/e at about 10 deg.
    #
    # Active over the whole episode, in flight as well as in the hold. Leading the tool along a moving
    # reference means letting the payload lean (tan(theta) = a / g); measured against gravity, that
    # lean would count as swing and the term would fight the tracking. So the swing is measured about
    # the lean the reference acceleration calls for, and only swing on top of it costs anything.
    swing_suppression = RewTerm(
        func=PayloadSwingEnergyExp,  # type: ignore[arg-type]
        weight=0.5,
        params={
            "payload_cfg": SceneEntityCfg("robot", body_names=PAYLOAD_BODY_NAMES),
            "pivot_cfg": SceneEntityCfg("robot", body_names=[PIVOT_BODY_NAME]),
            "energy_scale": 0.3,
            "energy_deadzone": 0.03,
            "compensate_reference_accel": True,
            "command_name": COMMAND_NAME,
        },
    )


@configclass
class ChainsawToolTerminationsCfg(TerminationsCfg):
    """Termination terms, with the tracking divergence measured on the tool frame, and a safety radius
    around the end target during the hold."""

    tracking_divergence = DoneTerm(
        func=tracked_position_error_above_maximum,
        params={"max_error": 3.75, "command_name": COMMAND_NAME},
    )
    # The swing task's safety radius, on the cutting mouth and only during the hold, where the saw is
    # at the work and a tree is never far away. It arms once the mouth is within the swing task's
    # largest hold ball of the end target, so arriving late is not a termination, and the largest kick
    # (~0.25 m of mouth travel) stays well inside it.
    safety_radius = DoneTerm(
        func=PositionErrorAboveMaximumAfterCapture,  # type: ignore[arg-type]
        params={
            "max_error": SAFETY_RADIUS,
            "capture_radius": HOLD_RADIUS_MAX,
            "command_name": COMMAND_NAME,
            "asset_cfg": None,
            "window_s": TOOL_END_HOLD_S,
        },
    )


@configclass
class MatriceChainsawToolTrajectoryEnvCfg(MatriceChainsawTrajectoryEnvCfg):
    """Trajectory following on the DJI M350 with the Mk3 chainsaw, tracking the cutting mouth rather than the drone."""

    events: ChainsawToolEventCfg = ChainsawToolEventCfg()
    rewards: ChainsawToolRewardsCfg = ChainsawToolRewardsCfg()
    terminations: ChainsawToolTerminationsCfg = ChainsawToolTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Track the tool frame. The observations, the tracking rewards, the termination and the
        # command metrics all read it from the command.
        self.commands.trajectory.body_name = TOOL_BODY_NAME
        self.commands.trajectory.body_offset = CHAINSAW_TOOL_FRAME
        # The parent points the divergence check at the airframe; this one reads the command instead.
        self.terminations.tracking_divergence.params.pop("asset_cfg", None)
        # Spawn the airframe above the reference, so that the hanging mouth starts on it.
        self.events.reset_base.params["rest_offset"] = CHAINSAW_HANG_OFFSET
        # Spawn at any heading relative to the reference, as in the swing task, so the policy learns to
        # turn round onto the reference yaw rather than only hold it. The rest offset is rotated by the
        # spawn yaw, so the mouth still starts on the reference. yaw_tracking_coarse gives the gradient
        # back from large errors.
        self.events.reset_base.params["yaw_noise"] = (-math.pi, math.pi)
        # Hold the end point for TOOL_END_HOLD_S rather than the chainsaw task's 3 s.
        self.commands.trajectory.end_hold_s = TOOL_END_HOLD_S

        # -- saw head locked at its rest pitch
        # As in MatriceChainsawSwingEnvCfg.play_mode, but for training too, so the policy is trained and
        # evaluated on the same plant. Holds ``root_joint`` at zero (the hanging rest pose) with a
        # stiff drive, in place of the 0.0105 N·m/rad springs, so the payload swings as one rigid
        # pendulum about the tube hinge and the mouth is no longer on the end of a double pendulum.
        # Gravity and swing loads on the head are ~1 N·m, so 1e4 N·m/rad keeps it within ~0.1 mrad.
        # The rest pose is the one CHAINSAW_HANG_OFFSET and CHAINSAW_TOOL_ROT were measured in. (The
        # randomized hang offset tilts the payload off that pose by up to ~10 deg, so the mouth spawns
        # up to ~0.2 m off the reference; the policy has to place it.) The joint stays in the
        # observation and reads ~0, so checkpoints still load.
        # To let the saw head swing freely again, comment out this block; reset_saw_head in
        # ChainsawToolEventCfg can then take a nonzero range, as the chainsaw task gives it.
        self.events.chainsaw_springs.params["stiffness"] = 1.0e4
        self.events.chainsaw_springs.params["scale_range"] = (1.0, 1.0)

        # -- speed envelope
        # 1 m/s of velocity action, with the trajectory ranges scaled to match: the hardest reference
        # peaks at 0.75 m/s and 1.0 m/s^2. Last, so it scales whatever ranges are in place by now.
        # set_speed_envelope(self, 1.0)

    def play_mode(self):
        """Evaluation overrides: as for :class:`MatriceChainsawTrajectoryEnvCfg`, with regular kicks."""
        super().play_mode()
        # Keep the kicks, as in MatriceChainsawSwingEnvCfg, on a fixed beat and never skipped, so every
        # environment shows the same disturbance-and-recovery pattern. Their size still varies.
        self.events.kick_chainsaw_swing.interval_range_s = (5.0, 5.0)
        self.events.kick_chainsaw_swing.params["skip_prob"] = 0.0

        # -- slow reference
        # Scale the reference's speed, acceleration and excursion by PLAY_SPEED_SCALE, as
        # set_speed_envelope does, but leave max_velocity alone: it sets the action scale the policy
        # was trained with. omega_max is not scaled, so the paths keep their shape and timescale and
        # only shrink. Set PLAY_SPEED_SCALE to 1.0 for the full training envelope.
        ranges = self.commands.trajectory.ranges
        scaled = ("vel_rms_target", "pos_cap", "vel_cap", "acc_cap", "pos_z_cap", "vel_z_cap", "acc_z_cap")
        self.commands.trajectory.ranges = ranges.replace(
            **{name: tuple(v * PLAY_SPEED_SCALE for v in getattr(ranges, name)) for name in scaled}
        )
