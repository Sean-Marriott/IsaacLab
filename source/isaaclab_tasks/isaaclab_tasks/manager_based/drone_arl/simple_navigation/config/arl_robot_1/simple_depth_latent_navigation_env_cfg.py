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
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_camera_cfg import MultiMeshRayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_tasks.manager_based.drone_arl.mdp.events import reset_single_obstacle
from isaaclab_tasks.manager_based.drone_arl.mdp.observations import ImageLatentObservation

from isaaclab_contrib.assets import MultirotorCfg
from isaaclab_contrib.controllers import LeeVelControllerCfg
import isaaclab_tasks.manager_based.drone_arl.mdp as mdp

from isaaclab_tasks.manager_based.drone_arl.mdp.commands import DroneUniformPoseCommandCfg
from isaaclab_tasks.manager_based.drone_arl.mdp.rewards import (
    ang_vel_xyz_exp,
    distance_to_goal_exp,
    lin_vel_xyz_exp,
    yaw_aligned,
    distance_to_goal_tanh,
    distance_to_goal_l2
)

##
# Pre-defined configs
##
from .scenes.obstacle_scenes.obstacle_scene import (
    generate_obstacle_around_origin,
    OBSTACLE_SCENE_CFG
)

##
# Scene definition
##
@configclass
class ArlDepthLatentNavigationSceneCfg(InteractiveSceneCfg):
    """Scene configuration for drone navigation with simple obstacles."""
    
    # obstacles
    object_collection = generate_obstacle_around_origin()

    # robots
    robot: MultirotorCfg = MISSING

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=10,
        debug_vis=True,
    )
    
    # sensors
    depth_camera = MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link",
        mesh_prim_paths=[
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/rod"),
        ], # type: ignore
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.04), rot=(1.0, 0.0, 0.0, 0.0), convention="world"
        ),
        update_period=0.1,
        pattern_cfg=PinholeCameraPatternCfg(
            width=480, height=270, focal_length=0.193, horizontal_aperture=0.36, vertical_aperture=0.21
        ),
        data_types=["distance_to_image_plane"],
        max_distance=10.0,
        depth_clipping_behavior="max",
        debug_vis=True
    )
    
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

    target_pose = DroneUniformPoseCommandCfg(
        asset_name="robot",
        body_name="base_link",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=DroneUniformPoseCommandCfg.Ranges(
            pos_x=(-0.0, 0.0),
            pos_y=(-0.0, 0.0),
            pos_z=(-0.0, 0.0),
            roll=(-0.0, 0.0),
            pitch=(-0.0, 0.0),
            yaw=(-0.0, 0.0),
        ),
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
        controller_cfg=LeeVelControllerCfg(
            K_vel_range=((2.5, 2.5, 1.5), (3.5, 3.5, 2.0)),
            K_rot_range=((1.6, 1.6, 0.25), (1.85, 1.85, 0.4)),
            K_angvel_range=((0.4, 0.4, 0.075), (0.5, 0.5, 0.09)),
            max_inclination_angle_rad=1.0471975511965976,
            max_yaw_rate=1.0471975511965976,
        ),
        max_velocity=0.5,       # 0.5 m/s
        max_yaw_rate=0.7853982, # 45 degrees/s
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_link_position = ObsTerm(func=mdp.root_pos_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_orientation = ObsTerm(func=mdp.root_quat_w, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        last_action = ObsTerm(func=mdp.last_action, noise=Unoise(n_min=-0.0, n_max=0.0))
        depth_latent = ObsTerm(
            func=ImageLatentObservation,
            params={"sensor_cfg": SceneEntityCfg("depth_camera"), "data_type": "distance_to_image_plane"},
        )
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base_and_obstacle = EventTerm(
        func=reset_single_obstacle,
        mode="reset",
        params={
            "robot_cfg": SceneEntityCfg("robot"),
            "obstacle_cfg": SceneEntityCfg("object_collection"),
            "env_size": OBSTACLE_SCENE_CFG.env_size,
            "pose_range": {
                "x": (-2.0, 2.0),
                "y": (-2.0, 2.0),
                "z": (-2.0, 2.0),
                "yaw": (-math.pi, math.pi),
                "roll": (-math.pi / 6.0, math.pi / 6.0),
                "pitch": (-math.pi / 6.0, math.pi / 6.0),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )

    # intervals
    push_robot = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 0.2),
        params={
            "force_range": (-0.1, 0.1),
            "torque_range": (-0.05, 0.05),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    distance_to_goal_l2 = RewTerm(
        func=distance_to_goal_l2,
        weight=-5.0,  # negative weight — penalise distance
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "target_pose",
        },
    )

    distance_to_goal_exp = RewTerm(
        func=distance_to_goal_exp,
        weight=25.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 1.5,
            "command_name": "target_pose",
        }
    )
    
    distance_to_goal_tanh = RewTerm(
        func=distance_to_goal_tanh,
        weight=10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "std": 0.1,
            "command_name": "target_pose",
        },
    )

    yaw_aligned = RewTerm(
        func=yaw_aligned,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 1.0},
    )
    
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    
    lin_vel_xyz_exp = RewTerm(
        func=lin_vel_xyz_exp,
        weight=2.5,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 2.0},
    )
    ang_vel_xyz_exp = RewTerm(
        func=ang_vel_xyz_exp,
        weight=10.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 10.0},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_magnitude_l2 = RewTerm(func=mdp.action_l2, weight=-0.05)
    
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-500.0,
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    crash_floor = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -3.0})
    crash_ceiling = DoneTerm(func=mdp.root_height_above_maximum, params={"maximum_height": 3.0})
    collision = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "threshold": 1.0},
        time_out=False,
    )

##
# Environment configuration
##


@configclass
class SimpleDepthLatentNavigationEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the simple lidar drone navigation environment."""

    # Scene settings
    scene: ArlDepthLatentNavigationSceneCfg = ArlDepthLatentNavigationSceneCfg(
        num_envs=4096, env_spacing=10.0
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 10
        self.episode_length_s = 10.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        )
        self.sim.physics = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt