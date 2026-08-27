# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab.utils.modifiers import DelayedObservationCfg
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_contrib.assets import MultirotorCfg
from isaaclab_contrib.controllers import LeeVelControllerCfg

import isaaclab_tasks.manager_based.drone_arl.mdp as mdp
from isaaclab_tasks.manager_based.drone_arl.mdp.commands import DroneTrajectoryCommandCfg
from isaaclab_tasks.manager_based.drone_arl.mdp.events import WindAndDrag, reset_root_state_on_trajectory
from isaaclab_tasks.manager_based.drone_arl.mdp.observations import (
    base_roll_pitch,
    trajectory_lookahead_v,
    trajectory_position_error_v,
    trajectory_ref_accel_v,
    trajectory_ref_velocity_v,
    trajectory_velocity_error_v,
    trajectory_yaw_error,
)
from isaaclab_tasks.manager_based.drone_arl.mdp.rewards import (
    distance_to_goal_exp,
    velocity_tracking_exp,
    yaw_tracking_exp,
)
from isaaclab_tasks.manager_based.drone_arl.mdp.terminations import position_error_above_maximum

EPISODE_LENGTH_SECONDS = 20.0
COMMAND_NAME = "trajectory"

# The measured observation delay is applied only to terms that a real vehicle has to estimate.
# The reference feed-forward terms are computed on-board from a known plan, so delaying them would
# model nothing. One policy step is already 100 ms, which is a realistic state-estimation lag, so
# the lag stays at a single step; ``hold_prob`` then supplies the occasional repeated frame.
# (``update_period`` must not exceed ``max_lag``, so multi-rate refresh needs a deeper lag budget.)
_MEASUREMENT_DELAY = DelayedObservationCfg(
    min_lag=0, max_lag=1, per_env=True, hold_prob=0.5, update_period=1, per_env_phase=True
)


##
# Scene definition
##


@configclass
class TrackTrajectorySceneCfg(InteractiveSceneCfg):
    """Configuration for an empty scene with a single flying robot."""

    # robots
    robot: MultirotorCfg = MISSING

    # lights
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

    trajectory = DroneTrajectoryCommandCfg(
        asset_name="robot",
        body_name="base_link",
        # Strictly longer than an episode, so accumulated float error can never trigger a
        # discontinuous mid-episode resample of the reference.
        resampling_time_range=(EPISODE_LENGTH_SECONDS + 1.0, EPISODE_LENGTH_SECONDS + 1.0),
        debug_vis=True,
        difficulty=1.0,
        yaw_mode="random_constant",
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    velocity_command = mdp.DirectVelocityActionCfg(
        asset_name="robot",
        scale=1.0,
        offset=0.0,
        preserve_order=False,
        use_default_offset=False,
        # Gains are overridden per robot; see MatriceCleanTrajectoryEnvCfg.
        controller_cfg=LeeVelControllerCfg(
            K_vel_range=((2.5, 2.5, 1.5), (3.5, 3.5, 2.0)),
            K_rot_range=((1.6, 1.6, 0.25), (1.85, 1.85, 0.4)),
            K_angvel_range=((0.4, 0.4, 0.075), (0.5, 0.5, 0.09)),
            # Tracking this envelope needs only about 17 degrees of tilt, so the clamp exists
            # purely to bound early exploration, where a 2.45 m/s velocity error would already
            # command 45 degrees. Anything looser stops being a limit at these speeds.
            max_inclination_angle_rad=math.pi / 4.0,
            max_yaw_rate=math.pi / 2.0,
        ),
        # The hardest reference tops out at 2.25 m/s and needs 0.75 m/s of standing tracking error
        # to sustain its 3.0 m/s^2 peak acceleration, which is exactly this budget. Keeping it no
        # larger than necessary matters: this constant sets the action resolution, scales the
        # exploration noise, and has to be reproduced exactly on the vehicle. Raising it means
        # scaling the trajectory ranges to match -- see the class docstring above.
        max_velocity=3.0,
        max_yaw_rate=math.pi / 2.0,  # 90 deg/s: nulls a worst-case pi yaw error in 2 s
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy group."""

        pos_error_v = ObsTerm(
            func=trajectory_position_error_v,
            params={"command_name": COMMAND_NAME},
            clip=(-6.0, 6.0),
            noise=Unoise(n_min=-0.05, n_max=0.05),
            modifiers=[_MEASUREMENT_DELAY],
        )
        vel_error_v = ObsTerm(
            func=trajectory_velocity_error_v,
            params={"command_name": COMMAND_NAME},
            clip=(-4.5, 4.5),
            noise=Unoise(n_min=-0.05, n_max=0.05),
            modifiers=[_MEASUREMENT_DELAY],
        )
        ref_vel_v = ObsTerm(func=trajectory_ref_velocity_v, params={"command_name": COMMAND_NAME})
        ref_acc_v = ObsTerm(func=trajectory_ref_accel_v, params={"command_name": COMMAND_NAME})
        yaw_error = ObsTerm(
            func=trajectory_yaw_error,
            params={"command_name": COMMAND_NAME},
            noise=Gnoise(mean=0.0, std=0.01),
        )
        # note: ``trajectory_ref_yaw_rate`` is deliberately absent. Under the default
        # ``yaw_mode="random_constant"`` the reference yaw rate is identically zero, so it carries
        # no information, and requiring it downstream forces a deployment to differentiate a yaw
        # angle it may not have. Add it back together with ``yaw_mode="sweep"``.
        lookahead_v = ObsTerm(
            func=trajectory_lookahead_v,
            params={"command_name": COMMAND_NAME},
            clip=(-7.5, 7.5),
        )
        # note: the body-frame linear velocity is deliberately absent too. It is the only quantity
        # that would need a frame other than the vehicle frame, and it is recoverable anyway:
        # ``ref_vel_v - vel_error_v`` is the vehicle-frame velocity exactly, and the body frame
        # differs from it only by the roll and pitch reported below.
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            modifiers=[_MEASUREMENT_DELAY],
        )
        base_roll_pitch = ObsTerm(
            func=base_roll_pitch,
            noise=Gnoise(mean=0.0, std=0.01),
            modifiers=[_MEASUREMENT_DELAY],
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    #
    # note: LeeControllerBase caches the mass and inertia it uses for feed-forward when the managers
    # are loaded, which happens *before* startup events run. The controller therefore keeps working
    # from the nominal mass while physics uses the randomized one. That model mismatch is exactly
    # the domain randomization wanted here, but it is a consequence of the ordering -- moving these
    # terms to "prestartup" would silently remove it.
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (0.80, 1.20),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "com_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "z": (-0.04, 0.04)},
        },
    )

    # reset
    reset_base = EventTerm(
        func=reset_root_state_on_trajectory,
        mode="reset",
        params={
            "command_name": COMMAND_NAME,
            "pos_noise": (-0.1, 0.1),
            "vel_noise": (-0.05, 0.05),
            "yaw_noise": (-0.3, 0.3),
            "rp_noise": (-0.1, 0.1),
            "angvel_noise": (-0.2, 0.2),
        },
    )

    # interval
    #
    # Runs every policy step so the drag force tracks the current velocity; the gust component is
    # held between its own slower resamples. The wrench is restricted to the base link -- without
    # an explicit body the underlying call would write an independent sample for every body of the
    # articulation, which on a multirotor means five uncorrelated wrenches rather than one gust.
    wind_and_drag = EventTerm(
        func=WindAndDrag,  # type: ignore[arg-type]
        mode="interval",
        interval_range_s=(0.1, 0.1),
        is_global_time=True,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            # 1.5 N is 0.23 m/s^2 on this airframe, about a thirteenth of the 3.0 m/s^2
            # acceleration budget. Deliberately *not* scaled with the speed envelope: gust strength
            # is a property of the weather, not of how fast the vehicle is asked to fly, so raising
            # the envelope correctly makes the disturbance relatively milder rather than keeping it
            # proportional.
            "force_range": (-1.5, 1.5),
            "torque_range": (-0.1, 0.1),
            "gust_interval_range_s": (2.0, 6.0),
            # Cd * A for an M350 sized airframe. Quadratic in speed, so at this envelope it is no
            # longer negligible: 0.62 N at the 2.25 m/s reference peak, against 0.03 N at the
            # 0.5 m/s the ranges were originally sized for.
            "drag_area_range": (0.10, 0.30),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    Every shaped term is an exponential kernel bounded to ``[0, 1]``, so the maximum per-step reward
    is about 2.0 and an undiscounted episode return is about 400. An unbounded penalty such as
    ``distance_to_goal_l2`` is deliberately avoided: early in training, when errors are metres, it
    would dominate every shaped term and make terminating the fastest way to raise the return.
    """

    # Two position kernels at different widths. The wide one carries the gradient over 0-4.5 m and
    # drives early learning; the narrow one is essentially zero until the policy is already decent
    # and then supplies the precision gradient that a single width cannot. Both widths scale with
    # the speed envelope -- see ActionsCfg -- but the fine one has a floor at the 0.05 m position
    # noise, below which it would only be rewarding noise the policy cannot observe.
    pos_tracking = RewTerm(
        func=distance_to_goal_exp,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "std": 0.54,
            "command_name": COMMAND_NAME,
        },
    )
    pos_tracking_fine = RewTerm(
        func=distance_to_goal_exp,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "std": 0.15,
            "command_name": COMMAND_NAME,
        },
    )
    vel_tracking = RewTerm(
        func=velocity_tracking_exp,
        weight=0.3,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.6, "command_name": COMMAND_NAME},
    )
    yaw_tracking = RewTerm(
        func=yaw_tracking_exp,
        weight=0.2,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.35, "command_name": COMMAND_NAME},
    )

    ang_vel_penalty = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.002)
    # A smooth 10 Hz tracking policy has sum(delta_action^2) of roughly 0.05 per step, so this
    # costs it about 1e-3 against a per-step reward near 2.0 -- small, but enough to be felt.
    # An order of magnitude less than this is indistinguishable from having no smoothness term at
    # all; an order of magnitude more and the policy stops accelerating hard enough to follow the
    # fastest references and settles for a large lag instead.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    action_magnitude = RewTerm(func=mdp.action_l2, weight=-0.0005)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)

    # TODO: Add tool frame flatness metric


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # About 1.7 s behind the reference at full speed, which is not recoverable in any useful sense.
    tracking_divergence = DoneTerm(
        func=position_error_above_maximum,
        params={
            "max_error": 3.75,
            "command_name": COMMAND_NAME,
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
        },
    )
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.309})  # 75 deg
    crash_floor = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -5.0})
    crash_ceiling = DoneTerm(func=mdp.root_height_above_maximum, params={"maximum_height": 5.0})


def ramp_trajectory_difficulty(
    env, env_ids, data: float, start_steps: int, end_steps: int, lo: float, hi: float
) -> float:
    """Ramp the trajectory difficulty linearly with the global step counter.

    Args:
        env: The manager-based RL environment instance.
        env_ids: Indices of the environments being reset. Unused: the ramp is global.
        data: Current value of the difficulty.
        start_steps: Policy-step count at which the ramp begins.
        end_steps: Policy-step count at which the ramp reaches :paramref:`hi`.
        lo: Difficulty before the ramp starts.
        hi: Difficulty after the ramp completes.

    Note:
        ``common_step_counter`` counts *policy* steps -- one per environment step, not one per
        environment per step -- so these thresholds are independent of ``num_envs`` and map
        directly onto iterations as ``steps / num_steps_per_env``.

    Returns:
        The new difficulty, or the no-change sentinel when it has not moved appreciably.
    """
    fraction = (env.common_step_counter - start_steps) / max(end_steps - start_steps, 1)
    new_value = lo + (hi - lo) * min(max(fraction, 0.0), 1.0)
    if abs(new_value - data) < 1e-4:
        return mdp.modify_term_cfg.NO_CHANGE
    return new_value


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.

    The curriculum manager runs before the command manager resamples on reset, so a difficulty
    written here already applies to the trajectory drawn in the same reset.
    """

    # With the default 32 steps per rollout, the ramp runs from about iteration 47 to about
    # iteration 470, leaving the rest of a 2000-iteration run at full difficulty.
    trajectory_difficulty = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": f"commands.{COMMAND_NAME}.difficulty",
            "modify_fn": ramp_trajectory_difficulty,
            "modify_params": {"start_steps": 1_500, "end_steps": 15_000, "lo": 0.0, "hi": 1.0},
        },
    )


##
# Environment configuration
##


@configclass
class TrackTrajectoryEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the drone trajectory-following environment."""

    # Scene settings
    scene: TrackTrajectorySceneCfg = TrackTrajectorySceneCfg(num_envs=4096, env_spacing=9.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 10
        self.episode_length_s = EPISODE_LENGTH_SECONDS
        self.sim.dt = 0.01  # 100 Hz physics, 10 Hz policy
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
        self.sim.physics = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
