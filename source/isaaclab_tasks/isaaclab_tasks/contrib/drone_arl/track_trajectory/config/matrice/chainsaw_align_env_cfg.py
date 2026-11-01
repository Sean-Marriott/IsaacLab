# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Aligning the Mk3 chainsaw's cutting mouth with a target pose on the M350, with no trajectory to follow.

The drone spawns away from a fixed target pose for the mouth, displaced and at any heading, and has to
fly the mouth onto it, line the saw up with the target yaw and hold it there. The policy commands drone
velocities, as in the other chainsaw tasks; the path to the target is its own.

There is no swing suppression reward. The swing is only penalized through its effect on the mouth: the
fine pose kernel and the capture's speed limit can only be met with the payload still.

It is built on the chainsaw tool task, whose command already tracks a tool frame on the mouth, with the
trajectory reference held still at the target, as in the swing task. Everything that reads the command
(the tracking errors in the observation, the rewards, the terminations and the markers) therefore reads
the mouth's error to the target. From the tool task it keeps the randomized hang offset, the swing at
reset and the kicks, the command delay, the fresh-observation fraction and the locked saw head.

The observation drops the reference velocity, acceleration and look-ahead terms: with the reference
still they are zero or a noise-free copy of the position error, which would let the policy bypass the
noise and lag on that error. It adds the airframe's own velocity and its error to where it rests with
the mouth on target, since those are what the velocity command acts on, and the mouth's position in the
target's own frame, which places it relative to the approach cone directly. The yaw error is observed as
(sin, cos). The layout differs from the trajectory tasks', and their checkpoints do not load.
"""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.contrib.drone_arl.mdp as mdp
from isaaclab_tasks.contrib.drone_arl.mdp.events import reset_root_state_on_approach
from isaaclab_tasks.contrib.drone_arl.mdp.observations import (
    CalibratedRestTargetError,
    body_lin_vel_v,
    trajectory_position_offset_r,
    trajectory_yaw_error_sin_cos,
)
from isaaclab_tasks.contrib.drone_arl.mdp.rewards import (
    ToolCapture,
    ToolCaptureProgress,
    ToolEndTargetExp,
    position_tracking_exp,
    yaw_tracking_exp,
)

from .chainsaw_env_cfg import AIRFRAME_BODY_NAME, CHAINSAW_HANG_OFFSET, CHAINSAW_MOUTH_OFFSET, ChainsawObservationsCfg
from .chainsaw_tool_env_cfg import (
    CHAINSAW_TOOL_ROT,
    HANG_OFFSET_EVENT_NAME,
    TOOL_BODY_NAME,
    MatriceChainsawToolTrajectoryEnvCfg,
)
from .swing_suppression_env_cfg import PIVOT_BODY_NAME, STATIONARY_REFERENCE
from .track_trajectory_env_cfg import _MEASUREMENT_DELAY, COMMAND_NAME, EPISODE_LENGTH_SECONDS

SPAWN_DISTANCE_RANGE = (0.6, 2.0)
"""Range of the spawn distance of the mouth from the target along the approach axis [m]. The near end is
twice the capture radius, so every episode starts with an approach to fly."""

SPAWN_LATERAL_RANGE = (-0.1, 0.1)
"""Range of the spawn offset of the mouth across the approach axis, horizontally [m]."""

SPAWN_VERTICAL_RANGE = (-0.1, 0.1)
"""Range of the spawn offset of the mouth across the approach axis, vertically [m]."""

SPAWN_YAW_NOISE = (-0.1, 0.1)
"""Range of the spawn heading relative to the target heading [rad] (about +/-6 deg)."""

SPAWN_RP_NOISE = (-0.1, 0.1)
"""Range of the spawn roll and pitch [rad]. Small, since the payload starts tilted with the airframe and a
large tilt would move the mouth off the approach line."""

CAPTURE_RADIUS = 0.3
"""Length of the capture envelope along the approach cone [m].

Twice the 0.15 m of the mouth pocket and the bar section above it, the distance a forward feed would
still take a branch from. The longer cone makes the capture reward reachable from further out, so it is
found while the policy is still learning to close in; the fine pose kernel keeps pulling the mouth onto
the target itself. Bring it back towards 0.15 m once the policy captures reliably.
"""

CAPTURE_TOLERANCE_RADIUS = 0.03
"""Radius of the ball at the target that counts as captured from any direction [m], at the scale of the
position noise, so sitting on the target is never ruled out by which side of it the noise falls."""

CAPTURE_APPROACH_AXIS = (-1.0, 0.0, 0.0)
"""Side of the target the mouth arrives from, in the target's yaw frame: behind it.

With the payload at rest the saw bar lies along the tool +x and the feed-arm hook curls under it, so the
mouth pocket opens forward and takes a branch by moving forward onto it. The mouth is therefore staged
behind the target and closes along +x. Measured from the tool frame in ``CHAINSAW_TOOL_ROT``; change it if
the cut is made another way, such as dropping the hook over the branch from above (``(0, 0, 1)``).
"""

CAPTURE_CONE_HALF_ANGLE = math.radians(15.0)
"""Half-angle of the approach cone [rad]."""

CAPTURE_TIP_RADIUS = CAPTURE_TOLERANCE_RADIUS
"""Smallest radius of the approach cone, at its tip [m]: as wide as the tolerance ball, so near the target
the cone does not narrow below it. A sharp 15 deg cone is only 2.7 cm in radius 0.1 m from the target, under
the 5 cm position noise, and a hand-written controller that settled the mouth 0.12 m from the target,
slow and square, was captured in under 5% of episodes, almost all of the misses on the cone angle. The
cone widens past this radius 0.11 m behind the target."""

CAPTURE_MAX_REL_SPEED = 0.1
"""Largest mouth speed relative to the target for a capture [m/s]. A 2 deg swing already peaks at about
0.13 m/s at the mouth, so this also asks for the swing to be damped."""

CAPTURE_MAX_YAW_ERROR = math.radians(10.0)
"""Largest tool yaw error for a capture [rad]: the bar has to cross the branch square to it, not just
reach it."""

CAPTURE_DWELL_S = 0.5
"""Time the mouth has to stay captured for an episode to count as a success [s], about a quarter of a
swing period, so a swing passing slowly through the envelope at the end of its arc does not count."""


@configclass
class ChainsawAlignObservationsCfg(ChainsawObservationsCfg):
    """Observation specifications of the chainsaw task, without the terms a still reference makes empty."""

    @configclass
    class PolicyCfg(ChainsawObservationsCfg.PolicyCfg):
        """Observations for the policy group."""

        # Zero with the reference held still.
        ref_vel_v = None
        ref_acc_v = None
        # Every look-ahead point is the target, so this is pos_error_v five times over, without its
        # noise and lag.
        lookahead_v = None

        # The tool yaw error as (sin, cos), in place of the angle. The spawn covers every heading, and the
        # wrapped angle jumps from +pi to -pi between two nearly identical states facing away from the
        # target. Same noise as the angle; no lag, as before.
        yaw_error = ObsTerm(
            func=trajectory_yaw_error_sin_cos,
            params={"command_name": COMMAND_NAME},
            noise=Gnoise(mean=0.0, std=0.01),
        )

        # -- airframe
        # The terms above are the mouth's, but the velocity command acts on the airframe. These hand the
        # policy the airframe's own state rather than leaving it to subtract the swing out of the mouth's.
        # On the vehicle both come straight from the flight controller's estimate, so they get the
        # noise and lag of the mouth position and velocity errors.
        # The airframe's target is where it sits with the saw at its calibrated rest pose and the mouth on
        # the target, not at the nominal hang: with the nominal one, zeroing this error left the mouth off
        # the target by the payload's own hang offset (up to ~0.15 m), and a controller flying on it was
        # captured in under 10% of episodes. On the vehicle the rest pose is calibrated by averaging the
        # hinge encoders in a still hover before the approach; here it is the randomized rest offset with
        # a 1 deg calibration error per hinge, redrawn each episode.
        airframe_target_error_v = ObsTerm(
            func=CalibratedRestTargetError,  # type: ignore[arg-type]
            params={
                "command_name": COMMAND_NAME,
                "hang_offset": CHAINSAW_HANG_OFFSET,
                "asset_cfg": SceneEntityCfg("robot", body_names=[AIRFRAME_BODY_NAME]),
                "pivot_cfg": SceneEntityCfg("robot", body_names=[PIVOT_BODY_NAME]),
                "rest_term_name": HANG_OFFSET_EVENT_NAME,
                "calibration_std": math.radians(1.0),
            },
            clip=(-6.0, 6.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
            tensor_processors=[_MEASUREMENT_DELAY],
        )
        airframe_lin_vel_v = ObsTerm(
            func=body_lin_vel_v,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[AIRFRAME_BODY_NAME])},
            clip=(-4.5, 4.5),
            noise=Unoise(n_min=-0.05, n_max=0.05),
            tensor_processors=[_MEASUREMENT_DELAY],
        )

        # -- mouth in the target frame
        # The mouth's position relative to the target, in the target's own yaw frame: whether it is behind
        # the target, beside it or ahead of it, and so whether it is in the approach cone (see
        # CAPTURE_APPROACH_AXIS), without rotating pos_error_v by the yaw error. It is the same
        # displacement as pos_error_v, so it gets the same noise and lag.
        mouth_offset_t = ObsTerm(
            func=trajectory_position_offset_r,
            params={"command_name": COMMAND_NAME},
            clip=(-6.0, 6.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
            tensor_processors=[_MEASUREMENT_DELAY],
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class ChainsawAlignRewardsCfg:
    """Reward terms for bringing the cutting mouth onto its target pose and holding it there.

    Every term reads the mouth, through the command's tool frame. The target is the still reference: its
    position for the mouth, its yaw for the tool yaw.
    """

    # -- approach
    # Wide kernels, so there is a gradient over the whole spawn range: 0.37 at 2 m and 0.05 at 3.5 m on
    # position, 0.01 at pi rad on yaw. They also pull the mouth back to the centre after a kick.
    pos_approach = RewTerm(func=position_tracking_exp, weight=1.0, params={"std": 2.0, "command_name": COMMAND_NAME})
    # The wide kernel is nearly flat within a metre (0.94 at 0.5 m) and the pose kernels below barely
    # reach out that far (0.02 at 0.5 m), so between about 0.25 and 0.7 m there was almost nothing to
    # pull the mouth in. This one, about the tool task's pos_tracking width, covers that gap: 0.37 at 0.5 m,
    # 0.78 at 0.25 m, with a slope of about 1.5 per metre at 0.5 m.
    pos_approach_mid = RewTerm(
        func=position_tracking_exp, weight=1.0, params={"std": 0.5, "command_name": COMMAND_NAME}
    )
    yaw_approach = RewTerm(func=yaw_tracking_exp, weight=0.5, params={"std": 1.5, "command_name": COMMAND_NAME})
    # The wide kernel above is nearly flat within 20-30 deg (0.95 of its value at 20 deg), and the yaw in
    # the pose kernels below only pays once the mouth is close. This narrow one, active at any distance,
    # squares the saw up during the approach rather than at the last moment: kernel 0.78 at 10 deg, 0.37
    # at 20 deg, 0.01 at 45 deg. Its width is that of the chainsaw tool task's yaw_tracking, at a little
    # more weight (0.3 against 0.2), since here the heading is part of the capture condition.
    yaw_align = RewTerm(func=yaw_tracking_exp, weight=0.3, params={"std": 0.35, "command_name": COMMAND_NAME})

    # -- alignment
    # The tool task's end-target pair, active over the whole episode and with the target on the
    # reference itself. Joint kernels on the mouth position and the tool yaw, so the reward is for the
    # pose, not a position with the saw facing anywhere. The coarse one carries the gradient in from
    # about half a metre; the fine one is the precision (0.13 at 0.10 m, 0.6 at 0.05 m; 0.46 at 10 deg).
    # As in the swing task, the fine weight is half the end-of-path tasks', since it is available over
    # the whole episode rather than a hold at the end. A swinging payload carries the mouth with it, so
    # the fine term can only be earned with the swing damped.
    tool_target = RewTerm(
        func=ToolEndTargetExp,  # type: ignore[arg-type]
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[TOOL_BODY_NAME]),
            "std": 0.25,
            "window_s": EPISODE_LENGTH_SECONDS,
            "tool_offset": CHAINSAW_MOUTH_OFFSET,
            "hang_offset": (0.0, 0.0, 0.0),
            "tool_rot": CHAINSAW_TOOL_ROT,
            "std_yaw": 1.0,
            "command_name": COMMAND_NAME,
            # The capture markers of tool_capture show the target and the tool heading instead; these
            # frames would cover them.
            "debug_vis": False,
        },
    )
    tool_target_fine = RewTerm(
        func=ToolEndTargetExp,  # type: ignore[arg-type]
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[TOOL_BODY_NAME]),
            "std": 0.07,
            "window_s": EPISODE_LENGTH_SECONDS,
            "tool_offset": CHAINSAW_MOUTH_OFFSET,
            "hang_offset": (0.0, 0.0, 0.0),
            "tool_rot": CHAINSAW_TOOL_ROT,
            "std_yaw": 0.2,
            "command_name": COMMAND_NAME,
            # The coarse term already draws the markers; a second set would reuse its prim paths.
            "debug_vis": False,
        },
    )

    # -- capture
    # The success condition, rewarded per step: the mouth in the capture envelope, inside the approach
    # cone, slow, and square to the target. On top of the kernels above, which pay for being near the
    # target from any side and at any speed, this pays for being where a forward feed would take the
    # branch. Logs the dwell-based success rate and time to success under Metrics/tool_capture.
    #
    # When rendering it draws the capture envelope as a translucent cone on the target, opening to the
    # approach side, a ball at its tip for the tolerance radius, and the mouth as a sphere with an arrow
    # along the saw heading, all coloured by the state: red outside the envelope, amber inside it but too
    # fast, not square or in the ball without having come through the cone, green captured, blue captured
    # for the whole dwell.
    tool_capture = RewTerm(
        func=ToolCapture,  # type: ignore[arg-type]
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[TOOL_BODY_NAME]),
            "tool_offset": CHAINSAW_MOUTH_OFFSET,
            "hang_offset": (0.0, 0.0, 0.0),
            "tool_rot": CHAINSAW_TOOL_ROT,
            "capture_radius": CAPTURE_RADIUS,
            "tolerance_radius": CAPTURE_TOLERANCE_RADIUS,
            "approach_axis": CAPTURE_APPROACH_AXIS,
            "cone_half_angle": CAPTURE_CONE_HALF_ANGLE,
            "tip_radius": CAPTURE_TIP_RADIUS,
            "max_rel_speed": CAPTURE_MAX_REL_SPEED,
            "max_yaw_error": CAPTURE_MAX_YAW_ERROR,
            # The tolerance ball counts only after the mouth has come in through the cone, so an approach
            # from in front, bar tip first into the branch, never counts. Set False to count the ball
            # however the mouth reached it.
            "require_cone_entry": True,
            "dwell_s": CAPTURE_DWELL_S,
            "window_s": EPISODE_LENGTH_SECONDS,
            "command_name": COMMAND_NAME,
        },
    )

    # -- progress along the route to capture
    # Pays for each step's progress along the route to capture: towards the staging point at the open end
    # of the cone while outside it, then down the cone to the target. With the cone entry required, a
    # mouth that arrives from in front has to back out and come round, and this scores that as progress
    # rather than as a retreat, so the policy is paid to retry after a miss. In metres per step: closing at
    # 0.5 m/s is worth 0.25 per step, and the whole approach from 2 m about 10, which the episode sum
    # telescopes to whatever the path.
    capture_progress = RewTerm(
        func=ToolCaptureProgress,  # type: ignore[arg-type]
        weight=5.0,
        params={"capture_term_name": "tool_capture", "route_aware": True},
    )

    # -- regularization (same as the base config)
    ang_vel_penalty = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.002)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.0005)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)


@configclass
class MatriceChainsawAlignEnvCfg(MatriceChainsawToolTrajectoryEnvCfg):
    """Aligning the Mk3 chainsaw's cutting mouth with a target pose on the DJI M350, with no trajectory."""

    observations: ChainsawAlignObservationsCfg = ChainsawAlignObservationsCfg()
    rewards: ChainsawAlignRewardsCfg = ChainsawAlignRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        # -- target pose instead of a trajectory
        # Zero limits collapse the sampled path onto its start point, and the yaw reference supplies a
        # random target heading. The command keeps tracking the mouth (set by the tool task), so the
        # target is a pose for the mouth. The difficulty ramp has nothing left to scale.
        self.commands.trajectory.ranges = STATIONARY_REFERENCE
        self.curriculum.trajectory_difficulty = None
        # No slowdown or hold: the target is still from the start.
        self.commands.trajectory.end_slowdown_s = 0.0
        self.commands.trajectory.end_hold_s = 0.0
        # The command's goal, path and look-ahead markers would all sit on the target, hiding the capture
        # markers of tool_capture, which draw the target instead.
        self.commands.trajectory.reference_debug_vis = False

        # -- spawn on the approach line
        # The mouth starts on the cone axis behind the target, facing it, so every episode begins with the
        # approach the capture asks for. The spreads across the line and in heading are small for now;
        # they are parameters of the event so a curriculum can widen them later (through
        # mdp.modify_term_cfg on "events.reset_base.params.<name>"), up to the swing task's spawn anywhere
        # within +/-2 m at any heading.
        self.events.reset_base = EventTerm(
            func=reset_root_state_on_approach,
            mode="reset",
            params={
                "command_name": COMMAND_NAME,
                "approach_axis": CAPTURE_APPROACH_AXIS,
                "distance_range": SPAWN_DISTANCE_RANGE,
                "lateral_range": SPAWN_LATERAL_RANGE,
                "vertical_range": SPAWN_VERTICAL_RANGE,
                "yaw_noise": SPAWN_YAW_NOISE,
                "rp_noise": SPAWN_RP_NOISE,
                "vel_noise": (-0.1, 0.1),
                "angvel_noise": (-0.1, 0.1),
                # Where the mouth sits relative to the airframe with the payload at rest, so it is the
                # mouth that starts on the line.
                "rest_offset": CHAINSAW_HANG_OFFSET,
            },
        )

        # -- terminations
        # Kept wide for the curriculum: the swing task's spawn can be 3.5 m from the target, beyond the
        # 3.75 m tracking limit once the initial velocity carries it further.
        self.terminations.tracking_divergence.params["max_error"] = 5.0
        # No safety radius. The tool task's arms once the mouth is within 0.4 m of the target and ends the
        # episode beyond 0.6 m. Over a whole episode that made approaching a gamble on the rest of the
        # episode's reward, and a policy trained with it (2026-10-29) held the mouth at 0.47-0.52 m, just
        # outside the arming radius, and never reached the target. On the vehicle the bound belongs to
        # the supervisor's abort logic.
        self.terminations.safety_radius = None

    def play_mode(self):
        """Evaluation overrides: as for :class:`MatriceChainsawToolTrajectoryEnvCfg`, starting at rest."""
        super().play_mode()
        # Spawn the drone at rest with the payload hanging still, so any swing comes from the flight or
        # the kicks.
        self.events.reset_base.params["vel_noise"] = (0.0, 0.0)
        self.events.reset_base.params["angvel_noise"] = (0.0, 0.0)
        self.events.reset_chainsaw_joints.params["still_prob"] = 1.0
