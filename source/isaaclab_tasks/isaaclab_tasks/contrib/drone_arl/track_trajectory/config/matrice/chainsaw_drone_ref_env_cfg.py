# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory following on the M350 with the Mk3 chainsaw, where the reference is for the drone and the
cutting mouth is placed at the end.

On the vehicle the reference is planned for the drone, and the chainsaw tool task's reference for the
mouth is derived from it. A policy for that task turns the mouth reference back into drone motion, which
means leading the pendulum: the airframe has to run ahead of the mouth by about ``L * a / g`` while the
reference accelerates, through a velocity loop with up to 0.15 s of command delay. Here the policy tracks
the drone reference directly, as in the chainsaw task, and the mouth matters where the work is: at the end
of the path, where the hanging mouth is rewarded for reaching the point below the end point, at the end
yaw. In transit, the swing about the lean the reference acceleration calls for is penalized.

The policy also observes the end goal: the mouth's error to its end target and the time to go until the
reference reaches its end point. Without them the end reward depends on things the policy cannot see:
the end point can be beyond the 1 s look-ahead, and the reward windows open on a clock it is not given.
Both terms are appended after the chainsaw task's, so the observation is wider and checkpoints from the
other chainsaw tasks no longer load as they are.

The events, terminations, observation timing and command delay are the tool task's: the randomized hang
offset, the swing at reset and the kicks, the fresh-observation fraction, the locked saw head, the longer
end hold, the spawn at any heading and the speed envelope.
"""

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

import isaaclab_tasks.contrib.drone_arl.mdp as mdp
from isaaclab_tasks.contrib.drone_arl.mdp.observations import ToolEndTargetError, trajectory_time_to_go
from isaaclab_tasks.contrib.drone_arl.mdp.rewards import (
    PayloadSwingEnergyExp,
    ToolEndTargetExp,
    position_tracking_exp,
    velocity_tracking_exp,
    yaw_tracking_exp,
)

from .chainsaw_env_cfg import (
    AIRFRAME_BODY_NAME,
    CHAINSAW_HANG_OFFSET,
    CHAINSAW_MOUTH_OFFSET,
    END_SLOWDOWN_S,
    ChainsawObservationsCfg,
)
from .chainsaw_tool_env_cfg import (
    CHAINSAW_TOOL_ROT,
    TOOL_BODY_NAME,
    TOOL_END_HOLD_S,
    MatriceChainsawToolTrajectoryEnvCfg,
)
from .swing_suppression_env_cfg import PAYLOAD_BODY_NAMES, PIVOT_BODY_NAME
from .track_trajectory_env_cfg import _MEASUREMENT_DELAY, COMMAND_NAME

TIME_TO_GO_HORIZON_S = 6.0
"""Time to go beyond which the time-to-go observation saturates [s].

The 3 s slowdown plus about one and a half swing periods (~2.1 s each) before it, so the policy sees the
arrival coming in time to settle the swing and the heading. Further out, how far away the end is does
not matter.
"""


@configclass
class ChainsawDroneRefObservationsCfg(ChainsawObservationsCfg):
    """Observation specifications of the chainsaw task, with the end goal appended."""

    @configclass
    class PolicyCfg(ChainsawObservationsCfg.PolicyCfg):
        """Observations for the policy group."""

        # On the vehicle the mouth is placed from the estimated airframe pose and the hinge encoders, so
        # it gets the same lag as the other estimated states, and noise like that of the position error.
        # The target is at the nominal hang offset, as on the vehicle, not the randomized one.
        tool_end_target_error = ObsTerm(
            func=ToolEndTargetError,  # type: ignore[arg-type]
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[TOOL_BODY_NAME]),
                "tool_offset": CHAINSAW_MOUTH_OFFSET,
                "hang_offset": CHAINSAW_HANG_OFFSET,
                "tool_rot": CHAINSAW_TOOL_ROT,
                "command_name": COMMAND_NAME,
            },
            clip=(-6.0, 6.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
            tensor_processors=[_MEASUREMENT_DELAY],
        )
        # Known exactly from the planner's own timing: no noise and no lag.
        time_to_go = ObsTerm(
            func=trajectory_time_to_go,
            params={"command_name": COMMAND_NAME, "horizon_s": TIME_TO_GO_HORIZON_S},
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class ChainsawDroneRefRewardsCfg:
    """Reward terms, with the tracking terms on the airframe and the end target below the end point.

    Every term is listed here rather than inherited, so the full reward is visible in one place. The
    weights and widths match the tool task's.

    The fine airframe position tracking is off during the hold. Whenever the payload rests off its
    nominal hang (the randomized hang offset puts the mouth up to ~0.2 m off), the mouth on its target
    means the airframe off the end point, so the two would pull different ways. The coarse one stays on to
    anchor the airframe: at 0.2 m it gives up ~0.13 per step, against up to 5 for the end target pair, so
    it does not hold the airframe off the place that puts the mouth on target. The velocity and yaw
    tracking stay on too: the reference there is at rest and at the end yaw, which is what placing the
    mouth needs as well.
    """

    # -- tracking (all read the airframe from the command)
    # Stays on in the hold: it is the only term that still pulls the airframe back from metres away, where
    # the end target kernels are flat. Without it the airframe drifted off the end point there.
    pos_tracking = RewTerm(func=position_tracking_exp, weight=1.0, params={"std": 0.54, "command_name": COMMAND_NAME})
    pos_tracking_fine = RewTerm(
        func=position_tracking_exp,
        weight=0.5,
        params={"std": 0.15, "command_name": COMMAND_NAME, "exclude_end_s": TOOL_END_HOLD_S},
    )
    vel_tracking = RewTerm(func=velocity_tracking_exp, weight=0.3, params={"std": 0.6, "command_name": COMMAND_NAME})
    yaw_tracking = RewTerm(func=yaw_tracking_exp, weight=0.2, params={"std": 0.35, "command_name": COMMAND_NAME})
    # Keeps a gradient back from the large yaw errors of a spawn at any heading; see the tool task.
    yaw_tracking_coarse = RewTerm(func=yaw_tracking_exp, weight=0.3, params={"std": 1.5, "command_name": COMMAND_NAME})

    # -- regularization (same as the base config)
    ang_vel_penalty = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.002)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.0005)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)

    # -- end target
    # The tool task's pair, with the target where the mouth hangs at rest below the drone end point,
    # which is how the deployment derives the mouth reference from the drone's.
    tool_end_target = RewTerm(
        func=ToolEndTargetExp,  # type: ignore[arg-type]
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[TOOL_BODY_NAME]),
            "std": 0.25,
            "window_s": END_SLOWDOWN_S + TOOL_END_HOLD_S,
            "tool_offset": CHAINSAW_MOUTH_OFFSET,
            "hang_offset": CHAINSAW_HANG_OFFSET,
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
            "hang_offset": CHAINSAW_HANG_OFFSET,
            "tool_rot": CHAINSAW_TOOL_ROT,
            "std_yaw": 0.2,
            "command_name": COMMAND_NAME,
            # The coarse term already draws the markers; a second set would reuse its prim paths.
            "debug_vis": False,
        },
    )

    # -- swing
    # The tool task's term. Its lean compensation reads the reference acceleration, which is now that of
    # the pivot's own reference.
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
class MatriceChainsawDroneRefTrajectoryEnvCfg(MatriceChainsawToolTrajectoryEnvCfg):
    """Trajectory following on the DJI M350 with the Mk3 chainsaw, tracking the drone and placing the mouth at the end.

    The divergence and safety-radius terminations read the tracked frame, so they now bound the airframe
    about the reference. Placing the mouth moves the airframe off the end point by ~0.2 m at most, inside
    the safety radius's capture radius.
    """

    observations: ChainsawDroneRefObservationsCfg = ChainsawDroneRefObservationsCfg()
    rewards: ChainsawDroneRefRewardsCfg = ChainsawDroneRefRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Track the airframe, as in the chainsaw task. The observations, the tracking rewards, the
        # divergence termination and the command metrics all read it from the command.
        self.commands.trajectory.body_name = AIRFRAME_BODY_NAME
        self.commands.trajectory.body_offset = None
        # Spawn the airframe on the reference, rather than above it.
        self.events.reset_base.params["rest_offset"] = (0.0, 0.0, 0.0)
