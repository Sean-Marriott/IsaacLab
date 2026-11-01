# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Align the cutting mouth of the Mk3 chainsaw, carried by a DJI M350, with a target pose, approaching it
along a set direction.

The mouth pocket opens forward, so it takes a branch by feeding forward onto it: it has to arrive at the
target from behind, along the target's +x axis. The approach is staged. The goal is first a staging point
:data:`STAGING_DISTANCE` behind the target; once the mouth is there, slow and square, the goal moves on to
the target. The rewards are kernels on the remaining route length through the staging point, so they only
reach their maximum on the target after a staged arrival, and a cross-track kernel keeps the final feed on
the axis.

The drone spawns around the target with the payload hanging at rest. A curriculum raises the difficulty with
the success rate: first it widens the spawns from behind the target to all the way round it, then it brings
in wind gusts on the airframe. The policy commands airframe velocities and yaw rate through the Lee velocity
controller.

This is a deliberately minimal base with no domain randomization, observation delay or command delay. Add
those on top once the base task trains.
"""

import math
from dataclasses import MISSING

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_contrib.assets import MultirotorCfg
from isaaclab_contrib.controllers import LeeVelControllerCfg

from isaaclab_assets.robots.arl_robot_1 import MATRICE_MK3_CFG

import isaaclab_tasks.contrib.drone_arl.chainsaw_align.mdp as mdp

EPISODE_LENGTH_S = 20.0
COMMAND_NAME = "tool_pose"
SPAWN_EVENT_NAME = "reset_base"
WIND_EVENT_NAME = "wind"

AIRFRAME_BODY_NAME = "body"
"""Airframe body of the Mk3 USD (its ``base_link`` is the saw-head mount)."""

TOOL_BODY_NAME = "roll_link"
"""The saw body of the Mk3 USD, which carries the cutting mouth."""

PAYLOAD_JOINT_NAMES = ["csTubePitch", "csTubeRoll", "root_joint"]
"""Passive payload joints: the two-axis tube hinge at the airframe and the saw-head pitch at the tube end.
All are zero at the hanging rest pose."""

SAW_HEAD_JOINT_NAME = "root_joint"
"""The saw-head pitch joint at the bottom of the tube, locked at rest by EventCfg.lock_saw_head. It stays
in the observation (reading ~0) so it can be unlocked without changing the observation layout."""

MOUTH_POS = (-0.03, 0.365, 0.04)
"""Cutting mouth in the ``roll_link`` frame [m]: the pocket between the lower edge of the saw bar and the
feed-arm hook, read off the Mk3 meshes."""

MOUTH_ROT = (0.5009055, 0.4942847, 0.5489637, 0.4510157)
"""Orientation of the mouth frame in the ``roll_link`` frame, as a quaternion (x, y, z, w).

The inverse of the ``roll_link`` orientation in the vehicle frame with the payload hanging at rest, so at
rest the mouth frame lines up with the vehicle yaw frame (+x forward, +z up) and its yaw is the airframe
heading. Measured in PhysX with ``MATRICE_MK3_CFG`` at its default joint angles; re-measure if the USD
changes.
"""

MOUTH_HANG_OFFSET = (0.3704, 0.0017, -1.2331)
"""The mouth relative to the airframe with the payload hanging at rest below a level airframe, in the
yaw-aligned vehicle frame [m]. Measured as above."""

TARGET_POS = (0.0, 0.0, 1.5)
"""Target for the mouth, relative to the environment origin [m]. There is no ground, so the height only
places it in the viewport."""

APPROACH_AXIS = (-1.0, 0.0, 0.0)
"""Side of the target the mouth arrives from, in the target's yaw frame: behind it, so it feeds forward
along the tool +x axis, which the mouth pocket opens towards."""

STAGING_DISTANCE = 0.5
"""Distance of the staging point behind the target [m]."""


##
# Scene definition
##


@configclass
class ChainsawAlignSceneCfg(InteractiveSceneCfg):
    """An empty scene with the drone. There is no ground: the robot flies in free space."""

    robot: MultirotorCfg = MISSING

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # Staged at STAGING_DISTANCE behind the target: within 0.1 m, under 0.3 m/s and within 15 deg of the
    # target yaw. Success: within 5 cm, under 0.1 m/s, within 10 deg, having come in within 20 deg of the
    # approach axis (see the command config for the defaults).
    tool_pose = mdp.ToolPoseCommandCfg(
        asset_name="robot",
        body_name=TOOL_BODY_NAME,
        tool_pos=MOUTH_POS,
        tool_rot=MOUTH_ROT,
        target_pos=TARGET_POS,
        target_yaw=0.0,
        approach_axis=APPROACH_AXIS,
        staging_distance=STAGING_DISTANCE,
        # Longer than an episode, so the command only resets with the episode.
        resampling_time_range=(EPISODE_LENGTH_S + 1.0, EPISODE_LENGTH_S + 1.0),
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Velocity (x, y, z) in the vehicle frame and yaw rate, each in [-1, 1] and scaled by the limits.
    velocity_command = mdp.DirectVelocityActionCfg(
        asset_name="robot",
        scale=1.0,
        offset=0.0,
        preserve_order=False,
        use_default_offset=False,
        # Gains fitted to real M350 flights (see MatriceCleanTrajectoryEnvCfg for the derivation). They
        # act on the airframe; the passive payload swings below it. Resampled per environment on reset.
        controller_cfg=LeeVelControllerCfg(
            K_vel_range=((3.0, 3.0, 2.7), (5.5, 5.5, 3.3)),
            K_vel_int_range=((0.45, 0.45, 0.45), (0.55, 0.55, 0.55)),
            K_rot_range=((115.0, 115.0, 46.0), (140.6, 140.6, 56.2)),
            K_angvel_range=((15.2, 15.2, 11.1), (18.6, 18.6, 13.6)),
            max_inclination_angle_rad=math.pi / 4.0,
            max_yaw_rate=math.pi / 2.0,
        ),
        # Alignment is a slow, precise manoeuvre; a small range gives the policy finer resolution.
        max_velocity=1.0,
        max_yaw_rate=math.pi / 4.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group."""

        # -- task
        # Error to the current goal (the staging point, then the target), in the vehicle frame: what the
        # velocity command has to close.
        goal_pos_error = ObsTerm(
            func=mdp.tool_goal_error_v,
            params={"command_name": COMMAND_NAME},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        # The mouth relative to the target in the target's frame: where it is relative to the approach axis.
        tool_offset = ObsTerm(
            func=mdp.tool_offset_t,
            params={"command_name": COMMAND_NAME},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        staged = ObsTerm(func=mdp.tool_staged, params={"command_name": COMMAND_NAME})
        tool_yaw_error = ObsTerm(
            func=mdp.tool_yaw_error_sin_cos,
            params={"command_name": COMMAND_NAME},
            noise=Gnoise(mean=0.0, std=0.01),
        )
        # -- airframe
        # Error of the airframe to where it rests with the mouth on the current goal, at its current heading.
        # Closing this rather than the mouth's own error does not feed the swing back into the command: a
        # scripted controller on the mouth error drove a growing swing and never settled, while one on this
        # error succeeded in 95% of spawns all round the target.
        airframe_rest_error = ObsTerm(
            func=mdp.airframe_rest_error_v,
            params={"command_name": COMMAND_NAME, "hang_offset": MOUTH_HANG_OFFSET},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        lin_vel =ObsTerm(func=mdp.root_lin_vel_v, noise=Unoise(n_min=-0.05, n_max=0.05))
        ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        roll_pitch = ObsTerm(func=mdp.root_roll_pitch, noise=Gnoise(mean=0.0, std=0.01))
        # -- payload swing, from the hinge encoders
        payload_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=PAYLOAD_JOINT_NAMES)},
            noise=Gnoise(mean=0.0, std=0.0175),  # ~1 deg
        )
        payload_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=PAYLOAD_JOINT_NAMES)},
            noise=Gnoise(mean=0.0, std=0.035),  # ~2 deg/s
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # -- startup
    # The saw-head pivot locked at its rest pitch, as in the chainsaw trajectory tasks, so the payload
    # swings as one rigid pendulum about the tube hinge rather than as a double pendulum. Gravity and
    # swing loads on the head are ~1 N·m, so this holds it within ~0.1 mrad. MOUTH_ROT and MOUTH_HANG_OFFSET
    # were measured in this rest pose. Set to None to let the head swing on the USD's springs again.
    lock_saw_head = EventTerm(
        func=mdp.lock_passive_joints,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[SAW_HEAD_JOINT_NAME]), "stiffness": 1.0e4},
    )

    # -- reset
    # The mouth spawns 0.8-2 m from the target horizontally, at a bearing within max_azimuth of the
    # approach axis. max_azimuth and yaw_range are the curriculum's starting values; it widens them to
    # spawns all round the target at any heading (see CurriculumCfg).
    reset_base = EventTerm(
        func=mdp.reset_tool_around_target,
        mode="reset",
        params={
            "command_name": COMMAND_NAME,
            "hang_offset": MOUTH_HANG_OFFSET,
            "distance_range": (0.8, 2.0),
            "vertical_range": (-0.3, 0.3),
            "max_azimuth": math.radians(45.0),
            "yaw_range": (-0.3, 0.3),
            "rp_range": (-0.1, 0.1),
            "lin_vel_range": (-0.1, 0.1),
            "ang_vel_range": (-0.1, 0.1),
        },
    )

    # The payload starts hanging still at its rest pose.
    reset_payload = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=PAYLOAD_JOINT_NAMES),
        },
    )

    # -- interval
    # A gust on the airframe, held 2-6 s and then redrawn. It starts calm; the curriculum grows it once the
    # spawn is all round the target (see CurriculumCfg). Runs every policy step to time the holds.
    wind = EventTerm(
        func=mdp.RandomWind,  # type: ignore[arg-type]
        mode="interval",
        interval_range_s=(0.1, 0.1),
        is_global_time=True,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[AIRFRAME_BODY_NAME]),
            "force_range": (0.0, 0.0),
            "torque_range": (0.0, 0.0),
            "interval_range_s": (2.0, 6.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    The position kernels are on the remaining route length D: through the staging point until the mouth is
    staged, then straight to the target. D never increases at the switch, so reaching the staging point never
    costs reward, and the fine kernel can only be earned on the target after a staged arrival. A swinging
    payload carries the mouth with it, so the fine kernel also asks for the swing to be damped.
    """

    # -- route (wide: 0.11 at D = 3 m, the furthest spawn; mid: 0.37 at the staging point; fine: target only)
    route = RewTerm(func=mdp.tool_route_exp, weight=1.0, params={"std": 2.0, "command_name": COMMAND_NAME})
    route_mid = RewTerm(func=mdp.tool_route_exp, weight=1.0, params={"std": 0.5, "command_name": COMMAND_NAME})
    route_fine = RewTerm(func=mdp.tool_route_exp, weight=1.0, params={"std": 0.15, "command_name": COMMAND_NAME})
    # -- final feed on the approach axis (paid only once staged; 0.37 at 10 cm off the axis)
    cross_track = RewTerm(
        func=mdp.tool_cross_track_exp, weight=0.5, params={"std": 0.1, "command_name": COMMAND_NAME}
    )
    # -- holding still on the target
    # The position kernels are flat near the target. Kicking a ~5 cm swing into a mouth held on the target by a
    # scripted controller cut route_fine only from 0.97 to 0.95, but this term from 0.79 to 0.54. It pays only
    # within ~0.1 m of the target, so it cannot be earned by hovering still anywhere else.
    tool_still = RewTerm(
        func=mdp.tool_still_exp,
        weight=1.0,
        params={"std_speed": 0.1, "std_distance": 0.1, "command_name": COMMAND_NAME},
    )
    # -- saw yaw (0.33 at 90 deg, 0.01 at 180 deg for the wide kernel; 0.61 at 10 deg for the narrow)
    tool_yaw = RewTerm(func=mdp.tool_yaw_exp, weight=0.5, params={"std": 1.5, "command_name": COMMAND_NAME})
    tool_yaw_fine = RewTerm(func=mdp.tool_yaw_exp, weight=0.5, params={"std": 0.25, "command_name": COMMAND_NAME})

    # -- regularization
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.002)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    tool_too_far = DoneTerm(
        func=mdp.tool_distance_above_maximum, params={"command_name": COMMAND_NAME, "max_distance": 5.0}
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(75.0)})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # One difficulty level, up a step of 0.1 above 60% success and down below 30%, ramping in two stages:
    # - level 0-0.5: from spawns within 45 deg of the approach axis, facing within 0.3 rad of the target yaw, to
    #   spawns all round the target at any heading;
    # - level 0.5-1: wind from calm to gusts of up to 3 N per axis (0.33 m/s^2 on the 9.1 kg vehicle, ~4 N
    #   horizontally) and 0.2 N·m per axis.
    difficulty = CurrTerm(
        func=mdp.SuccessDifficultyCurriculum,  # type: ignore[arg-type]
        params={
            "command_name": COMMAND_NAME,
            "ramps": {
                f"{SPAWN_EVENT_NAME}.max_azimuth": (math.radians(45.0), math.pi, 0.0, 0.5),
                f"{SPAWN_EVENT_NAME}.yaw_range": (0.3, math.pi, 0.0, 0.5),
                f"{WIND_EVENT_NAME}.force_range": (0.0, 3.0, 0.5, 1.0),
                f"{WIND_EVENT_NAME}.torque_range": (0.0, 0.2, 0.5, 1.0),
            },
        },
    )


##
# Environment configuration
##


@configclass
class MatriceChainsawAlignEnvCfg(ManagerBasedRLEnvCfg):
    """Align the Mk3 chainsaw's cutting mouth with a target pose on the DJI M350, arriving along a set direction."""

    scene: ChainsawAlignSceneCfg = ChainsawAlignSceneCfg(num_envs=4096, env_spacing=8.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 10
        self.episode_length_s = EPISODE_LENGTH_S
        self.sim.dt = 0.01  # 100 Hz physics, 10 Hz policy
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
        # ThrusterCfg is implemented in Isaac Lab and has no Newton-native execution path.
        self.sim.use_newton_actuators = False

        self.scene.robot = MATRICE_MK3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["thrusters"].dt = self.sim.dt

    def play_mode(self):
        """Evaluation overrides: few environments, no observation noise, spawned at rest, at full difficulty."""
        super().play_mode()
        self.scene.env_spacing = 10.0
        # Every curriculum ramp at its end value, rather than wherever a training level happens to be.
        for address, (_, at_end, _, _) in self.curriculum.difficulty.params["ramps"].items():
            event_name, param_name = address.split(".")
            params = getattr(self.events, event_name).params
            params[param_name] = (-at_end, at_end) if isinstance(params[param_name], tuple) else at_end
        self.curriculum.difficulty = None
        self.events.reset_base.params["lin_vel_range"] = (0.0, 0.0)
        self.events.reset_base.params["ang_vel_range"] = (0.0, 0.0)
