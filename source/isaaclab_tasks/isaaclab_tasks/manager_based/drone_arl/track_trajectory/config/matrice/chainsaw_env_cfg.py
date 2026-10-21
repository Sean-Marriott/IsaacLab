# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory following on the M350 with the Mk3 chainsaw and its pendulum state in the observation.

The chainsaw hangs from the airframe on a two-axis hinge (``csTubePitch``, ``csTubeRoll``) at the top
of the tube, and the saw head pitches freely on ``root_joint`` at the bottom. Without their state the
policy can only infer the swing from the disturbance it puts on the base, after the fact. This
variant appends their angles, relative to the hanging rest pose, and rates to the policy observation,
after the base terms, so the base layout is preserved as a prefix of the new one.

The reference also decelerates to rest over the last part of the episode and holds its end point.
During that hold the chainsaw cutting mouth is rewarded for sitting where it would be if the tool
hung straight down below the end point without swinging, on top of the usual tracking rewards.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise

import isaaclab_tasks.manager_based.drone_arl.mdp as mdp
from isaaclab_tasks.manager_based.drone_arl.mdp.events import (
    RandomizePassiveJointDamping,
    set_passive_joint_friction,
    set_passive_joint_stiffness,
)
from isaaclab_tasks.manager_based.drone_arl.mdp.rewards import (
    ToolEndTargetExp,
    distance_to_goal_exp,
    velocity_tracking_exp,
    yaw_tracking_exp,
)

from isaaclab_assets.robots.arl_robot_1 import MATRICE_MK3_CFG

from .env_cfg import MatriceCleanTrajectoryEnvCfg
from .track_trajectory_env_cfg import _MEASUREMENT_DELAY, COMMAND_NAME, EventCfg, ObservationsCfg

CHAINSAW_JOINT_NAMES = ["csTubePitch", "csTubeRoll", "root_joint"]
"""Passive payload joints: the tube hinge at the airframe and the saw-head pitch at the tube end."""

ROOT_JOINT_SPRING_STIFFNESS = 0.0105
"""Stiffness of the springs added on ``root_joint`` to help hold the saw head at its rest pitch [N·m/rad].

The saw head COM sits below ``root_joint``, so gravity is the main restoring torque (about 1.05 N·m/rad,
set by the mass properties in the USD). The springs add about 1% to that: with the mast clamped, the
saw head swung with nearly the same period with and without them once the larger release angle of the
spring take is allowed for. The value comes from a sim fit to three motion-capture takes (mast clamped
without and with springs, and the free hang-and-release).

The same value is authored in ``M350-chainsaw-mk3.usd``; this task writes it again at startup so it can
be randomized. Set it to 0.0 to model the saw without the springs.
"""

ROOT_JOINT_FRICTION = 0.03
"""Dry friction on ``root_joint``: the PhysX static and dynamic joint friction effort [N·m].

With the mast clamped, the real saw head lost about 4 deg of swing per half cycle whatever its
amplitude, the signature of dry friction, and the swing with and without springs could not both be
matched by viscous damping. This value, with no viscous damping on the hinge, reproduces every peak of
both clamped takes to within about 2 deg in this simulation.

The same value is authored in ``M350-chainsaw-mk3.usd``; this task writes it again at startup so it can
be randomized. Set it to 0.0 to remove it.
"""

AIRFRAME_BODY_NAME = "body"
"""Airframe body of the Mk3 USD. Its ``base_link`` is the saw-head mount, not the airframe."""

END_SLOWDOWN_S = 3.0
"""Time over which the reference decelerates to rest [s]."""
END_HOLD_S = 3.0
"""Time the reference holds its end point before the episode ends [s]."""

# Cutting mouth in the ``roll_link`` (saw) body frame: the pocket between the lower edge of the saw
# bar and the feed-arm hook, read off the Mk3 meshes. The bar runs along +y from 0.32 to 0.43 m at
# x = 0.015-0.048 m; the hook pivots at y = 0.31 m and curls round to x = -0.11 m.
CHAINSAW_MOUTH_OFFSET = (-0.03, 0.365, 0.04)
# The mouth relative to the airframe ``body`` when the payload hangs at rest below a level,
# stationary airframe, in the yaw-aligned vehicle frame. Measured in PhysX with MATRICE_MK3_CFG at
# its default joint angles, which are its hanging rest pose; re-measure if the USD or those change.
CHAINSAW_HANG_OFFSET = (0.3704, 0.0017, -1.2331)


@configclass
class ChainsawObservationsCfg(ObservationsCfg):
    """Observation specifications with the chainsaw hinge state appended."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Observations for the policy group."""

        # The hinge is read by encoders on the vehicle, so it gets the same lag as the other
        # estimated states. Angles are relative to the hanging rest pose, so zero means no swing.
        chainsaw_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CHAINSAW_JOINT_NAMES)},
            noise=Gnoise(mean=0.0, std=0.0175),  # ~1 deg
            modifiers=[_MEASUREMENT_DELAY],
        )
        chainsaw_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CHAINSAW_JOINT_NAMES)},
            noise=Gnoise(mean=0.0, std=0.035),  # ~2 deg/s
            modifiers=[_MEASUREMENT_DELAY],
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class ChainsawEventCfg(EventCfg):
    """Configuration for events, with the chainsaw dynamics randomized and its joints reset about rest."""

    # The tube-hinge damping is identified from one free hang-and-release take, and it sets how fast a
    # swing dies out, so the policy should not rely on one exact decay rate. In the USD it is authored
    # as a zero-stiffness drive, not joint friction, hence the custom term. The saw-head hinge has no
    # viscous damping (its losses are dry friction, below), so this leaves it at zero.
    randomize_chainsaw_damping = EventTerm(
        func=RandomizePassiveJointDamping,  # type: ignore[arg-type]
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=CHAINSAW_JOINT_NAMES),
            "scale_range": (0.5, 2.0),
        },
    )

    # The springs on ``root_joint``. They are a small part of the restoring torque and their rate is
    # only roughly identified, so it is randomized widely about the identified value.
    chainsaw_springs = EventTerm(
        func=set_passive_joint_stiffness,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["root_joint"]),
            "stiffness": ROOT_JOINT_SPRING_STIFFNESS,
            "scale_range": (0.5, 2.0),
        },
    )

    # Dry friction on the saw-head hinge, which dominates its losses on the rig. Identified from two
    # clamped takes, so randomized moderately about the identified value.
    chainsaw_hinge_friction = EventTerm(
        func=set_passive_joint_friction,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["root_joint"]),
            "friction": ROOT_JOINT_FRICTION,
            "scale_range": (0.7, 1.4),
        },
    )

    # The saw is ~1.9 kg of a ~2 kg payload and its mass varies with fuel, chain and battery. Only its
    # mass is randomized, not its COM: the COM offset below ``root_joint`` sets the saw-head pendulum
    # and the hanging mouth position in CHAINSAW_HANG_OFFSET, both identified for the nominal COM.
    randomize_chainsaw_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["roll_link"]),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    reset_chainsaw_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.3, 0.3),  # ~17 deg
            "velocity_range": (-0.1, 0.1),
            "asset_cfg": SceneEntityCfg("robot", joint_names=CHAINSAW_JOINT_NAMES),
        },
    )


@configclass
class ChainsawRewardsCfg:
    """Reward terms, with the chainsaw mouth rewarded for reaching its end target.

    Every term is listed here rather than inherited from the base ``RewardsCfg``, so the full reward
    is visible in one place. The tracking and regularization terms match the base values.
    """

    # -- tracking (same as the base config, but on the airframe rather than ``base_link``)
    pos_tracking = RewTerm(
        func=distance_to_goal_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[AIRFRAME_BODY_NAME]),
            "std": 0.54,
            "command_name": COMMAND_NAME,
        },
    )
    pos_tracking_fine = RewTerm(
        func=distance_to_goal_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[AIRFRAME_BODY_NAME]),
            "std": 0.15,
            "command_name": COMMAND_NAME,
        },
    )
    # These two read the frame the command tracks, which here is the airframe (the articulation root).
    vel_tracking = RewTerm(
        func=velocity_tracking_exp,
        weight=0.3,
        params={"std": 0.6, "command_name": COMMAND_NAME},
    )
    yaw_tracking = RewTerm(
        func=yaw_tracking_exp,
        weight=0.2,
        params={"std": 0.35, "command_name": COMMAND_NAME},
    )

    # -- regularization (same as the base config)
    ang_vel_penalty = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.002)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.0005)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)

    # -- chainsaw end target
    # Two kernels on the same mouth-to-target distance, like the position tracking pair. A single
    # narrow kernel would give no gradient at the errors an untrained policy starts from (a 0.05 m
    # kernel is ~1e-4 at 0.15 m), and a single wide one barely distinguishes 0.05 m from 0.15 m.
    #
    # The coarse kernel is active from the start of the slowdown, so it steers the approach while
    # the reference is still decelerating; it carries the gradient out to roughly half a metre.
    tool_end_target = RewTerm(
        func=ToolEndTargetExp,  # type: ignore[arg-type]
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["roll_link"]),
            "std": 0.25,
            "window_s": END_SLOWDOWN_S + END_HOLD_S,
            "tool_offset": CHAINSAW_MOUTH_OFFSET,
            "hang_offset": CHAINSAW_HANG_OFFSET,
            "command_name": COMMAND_NAME,
            # Orange sphere on the cutting mouth, cyan on its end target. Drawn only when rendering.
            "debug_vis": True,
        },
    )
    # The fine kernel is active only during the hold and supplies the precision gradient: about 0.13
    # at 0.10 m, 0.6 at 0.05 m. With the coarse term the hold is worth up to 5 per step against about
    # 2 for tracking, so alignment dominates there; over an episode the pair is still worth at most
    # ~180 against the ~400 of tracking, so it cannot buy its way out of following the path.
    tool_end_target_fine = RewTerm(
        func=ToolEndTargetExp,  # type: ignore[arg-type]
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["roll_link"]),
            "std": 0.07,
            "window_s": END_HOLD_S,
            "tool_offset": CHAINSAW_MOUTH_OFFSET,
            "hang_offset": CHAINSAW_HANG_OFFSET,
            "command_name": COMMAND_NAME,
            # The coarse term already draws the markers; a second set would reuse its prim paths.
            "debug_vis": False,
        },
    )


@configclass
class MatriceChainsawTrajectoryEnvCfg(MatriceCleanTrajectoryEnvCfg):
    """Trajectory following on the DJI M350 with the Mk3 chainsaw and its joint state observed."""

    observations: ChainsawObservationsCfg = ChainsawObservationsCfg()
    events: ChainsawEventCfg = ChainsawEventCfg()
    rewards: ChainsawRewardsCfg = ChainsawRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.trajectory.end_slowdown_s = END_SLOWDOWN_S
        self.commands.trajectory.end_hold_s = END_HOLD_S
        self.scene.robot = MATRICE_MK3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["thrusters"].dt = self.sim.dt

        # The shared terms track ``base_link``, which in the Mk3 USD is the saw-head mount. Point them
        # at the airframe instead. The position rewards already do so in ChainsawRewardsCfg.
        airframe = SceneEntityCfg("robot", body_names=[AIRFRAME_BODY_NAME])
        self.commands.trajectory.body_name = AIRFRAME_BODY_NAME
        for term in (
            self.terminations.tracking_divergence,
            self.events.randomize_mass,
            self.events.randomize_com,
            self.events.wind_and_drag,
        ):
            term.params["asset_cfg"] = airframe.replace()


@configclass
class MatriceChainsawTrajectoryEnvCfg_PLAY(MatriceChainsawTrajectoryEnvCfg):
    """Evaluation variant: few environments, no corruption, hardest reference."""

    def __post_init__(self):
        super().__post_init__()
        # Same overrides as MatriceCleanTrajectoryEnvCfg_PLAY.
        self.scene.num_envs = 50
        self.scene.env_spacing = 18.0
        self.observations.policy.enable_corruption = False
        self.events.wind_and_drag.params["force_range"] = (0.0, 0.0)
        self.events.wind_and_drag.params["torque_range"] = (0.0, 0.0)
        self.curriculum.trajectory_difficulty = None
        self.commands.trajectory.difficulty = 1.0
