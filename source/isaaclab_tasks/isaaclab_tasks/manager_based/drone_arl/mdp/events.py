# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions specific to the drone ARL environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from .curriculums import get_obstacle_curriculum_term

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject, RigidObjectCollection
    from isaaclab.envs import ManagerBasedRLEnv


def reset_obstacles_with_individual_ranges(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    obstacle_configs: dict,
    wall_configs: dict,
    env_size: tuple[float, float, float],
    use_curriculum: bool = True,
    min_num_obstacles: int = 1,
    max_num_obstacles: int = 10,
    ground_offset: float = 0.1,
) -> None:
    """Reset obstacle and wall positions for specified environments without collision checking.

    This function repositions all walls and a curriculum-determined subset of obstacles
    within the specified environment bounds.

    Walls are positioned at fixed locations based on their configuration ratios. Obstacles
    are randomly placed within their designated zones, with the number of active obstacles
    determined by the curriculum difficulty level. Inactive obstacles are moved far below
    the scene (-1000m in Z) to effectively remove them from the environment.

    The curriculum scaling works as:
        num_obstacles = min + (difficulty / max_difficulty) * (max - min)

    Args:
        env: The manager-based RL environment instance.
        env_ids: Tensor of environment indices to reset.
        asset_cfg: Scene entity configuration identifying the obstacle collection.
        obstacle_configs: Dictionary mapping obstacle type names to their BoxCfg
            configurations, specifying size and placement ranges.
        wall_configs: Dictionary mapping wall names to their BoxCfg configurations.
        env_size: Tuple of (length, width, height) defining the environment bounds in meters.
        use_curriculum: If True, number of obstacles scales with curriculum difficulty.
            If False, spawns max_num_obstacles in every environment. Defaults to True.
        min_num_obstacles: Minimum number of obstacles to spawn per environment.
            Defaults to 1.
        max_num_obstacles: Maximum number of obstacles to spawn per environment.
            Defaults to 10.
        ground_offset: Z-axis offset to prevent obstacles from spawning at z=0.
            Defaults to 0.1 meters.

    Note:
        This function expects the environment to have `_obstacle_difficulty_levels` and
        `_max_obstacle_difficulty` attributes when `use_curriculum=True`. These are
        typically set by :func:`obstacle_density_curriculum`.
    """
    obstacles: RigidObjectCollection = env.scene[asset_cfg.name]

    num_objects = obstacles.num_objects
    num_envs = len(env_ids)
    object_names = obstacles.object_names

    # Get difficulty levels per environment
    if use_curriculum:
        curriculum_term = get_obstacle_curriculum_term(env)
        if curriculum_term is not None:
            # Get difficulty levels for the specific environments being reset
            difficulty_levels = curriculum_term.difficulty_levels[env_ids]
            max_difficulty = curriculum_term.max_difficulty
        else:
            # Fallback: use max obstacles if curriculum not found
            difficulty_levels = torch.ones(num_envs, device=env.device) * max_num_obstacles
            max_difficulty = max_num_obstacles
    else:
        difficulty_levels = torch.ones(num_envs, device=env.device) * max_num_obstacles
        max_difficulty = max_num_obstacles

    # Calculate active obstacles per env based on difficulty
    obstacles_per_env = (
        min_num_obstacles + (difficulty_levels / max_difficulty) * (max_num_obstacles - min_num_obstacles)
    ).long()

    # Prepare tensors
    all_poses = torch.zeros(num_envs, num_objects, 7, device=env.device)
    all_velocities = torch.zeros(num_envs, num_objects, 6, device=env.device)

    wall_names = list(wall_configs.keys())
    obstacle_types = list(obstacle_configs.values())
    env_size_t = torch.tensor(env_size, device=env.device)

    # place walls
    for wall_name, wall_cfg in wall_configs.items():
        if wall_name in object_names:
            wall_idx = object_names.index(wall_name)

            min_ratio = torch.tensor(wall_cfg.center_ratio_min, device=env.device)
            max_ratio = torch.tensor(wall_cfg.center_ratio_max, device=env.device)

            if torch.allclose(min_ratio, max_ratio):
                center_ratios = min_ratio.unsqueeze(0).repeat(num_envs, 1)
            else:
                ratios = torch.rand(num_envs, 3, device=env.device)
                center_ratios = ratios * (max_ratio - min_ratio) + min_ratio

            positions = (center_ratios - 0.5) * env_size_t
            positions[:, 2] += ground_offset
            positions += env.scene.env_origins[env_ids]

            all_poses[:, wall_idx, 0:3] = positions
            all_poses[:, wall_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(num_envs, 1)

    # Get obstacle indices
    obstacle_indices = [idx for idx, name in enumerate(object_names) if name not in wall_names]

    if len(obstacle_indices) == 0:
        obstacles.write_object_pose_to_sim(all_poses, env_ids=env_ids)
        obstacles.write_object_velocity_to_sim(all_velocities, env_ids=env_ids)
        return

    # Determine which obstacles are active per env
    active_masks = torch.zeros(num_envs, len(obstacle_indices), dtype=torch.bool, device=env.device)
    for env_idx in range(num_envs):
        num_active = obstacles_per_env[env_idx].item()
        perm = torch.randperm(len(obstacle_indices), device=env.device)[:num_active]
        active_masks[env_idx, perm] = True

    # place obstacles
    for obj_list_idx in range(len(obstacle_indices)):
        obj_idx = obstacle_indices[obj_list_idx]

        # Which envs need this obstacle?
        envs_need_obstacle = active_masks[:, obj_list_idx]

        if not envs_need_obstacle.any():
            # Move all to -1000
            all_poses[:, obj_idx, 0:3] = env.scene.env_origins[env_ids] + torch.tensor(
                [0.0, 0.0, -1000.0], device=env.device
            )
            all_poses[:, obj_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
            continue

        # Get obstacle config
        config_idx = obj_list_idx % len(obstacle_types)
        obs_cfg = obstacle_types[config_idx]

        min_ratio = torch.tensor(obs_cfg.center_ratio_min, device=env.device)
        max_ratio = torch.tensor(obs_cfg.center_ratio_max, device=env.device)

        # sample object positions
        num_active_envs = envs_need_obstacle.sum().item()
        ratios = torch.rand(int(num_active_envs), 3, device=env.device)
        positions = (ratios * (max_ratio - min_ratio) + min_ratio - 0.5) * env_size_t
        positions[:, 2] += ground_offset

        # Add env origins
        active_env_indices = torch.where(envs_need_obstacle)[0]
        positions += env.scene.env_origins[env_ids[active_env_indices]]

        # Generate quaternions
        quats = math_utils.random_orientation(num_envs, device=env.device)

        # Write poses
        all_poses[envs_need_obstacle, obj_idx, 0:3] = positions
        all_poses[envs_need_obstacle, obj_idx, 3:7] = quats[envs_need_obstacle]

        # Move inactive obstacles far away
        inactive = ~envs_need_obstacle
        all_poses[inactive, obj_idx, 0:3] = env.scene.env_origins[env_ids[inactive]] + torch.tensor(
            [0.0, 0.0, -1000.0], device=env.device
        )
        all_poses[inactive, obj_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)

    # Write to sim
    obstacles.write_object_pose_to_sim(all_poses, env_ids=env_ids)
    obstacles.write_object_velocity_to_sim(all_velocities, env_ids=env_ids)


def reset_single_obstacle(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    obstacle_cfg: SceneEntityCfg,
    env_size: tuple[float, float, float],
    pose_range: dict,
    velocity_range: dict,
    radius_min: float = 0.2,
    radius_max: float = 0.5,
    drone_offset_min: float = 1.0,
    drone_offset_max: float = 2.0,
) -> None:
    obstacles: RigidObjectCollection = env.scene[obstacle_cfg.name]
    num_objects = obstacles.num_objects
    num_envs = len(env_ids)
    object_names = obstacles.object_names

    # Prepare tensors
    all_poses = torch.zeros(num_envs, num_objects, 7, device=env.device)
    all_velocities = torch.zeros(num_envs, num_objects, 6, device=env.device)

    if "rod" not in object_names:
        raise ValueError("reset_single_obstacle requires an object named 'rod' in the scene.")

    rod_idx = object_names.index("rod")
    env_origins = env.scene.env_origins[env_ids]

    # Move all objects far below the scene by default.
    all_poses[:, :, 0:3] = env_origins.unsqueeze(1) + torch.tensor([0.0, 0.0, -1000.0], device=env.device)
    all_poses[:, :, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)

    # Sample a uniform random position in an XY ring (annulus) centered at each env origin.
    env_radius_limit = 0.5 * min(env_size[0], env_size[1])
    radius_min_m = radius_min * env_radius_limit
    radius_max_m = radius_max * env_radius_limit
    if radius_min_m < 0.0 or radius_max_m < 0.0:
        raise ValueError("radius_min and radius_max must be non-negative.")
    if radius_min_m > radius_max_m:
        raise ValueError("radius_min must be less than or equal to radius_max.")

    angles = 2.0 * torch.pi * torch.rand(num_envs, device=env.device)
    radii = torch.sqrt(torch.rand(num_envs, device=env.device) * (radius_max_m**2 - radius_min_m**2) + radius_min_m**2)

    positions = torch.zeros(num_envs, 3, device=env.device)
    positions[:, 0] = radii * torch.cos(angles)
    positions[:, 1] = radii * torch.sin(angles)
    positions += env_origins

    all_poses[:, rod_idx, 0:3] = positions
    all_poses[:, rod_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(num_envs, 1)

    # Write to sim
    obstacles.write_object_pose_to_sim(all_poses, env_ids=env_ids)
    obstacles.write_object_velocity_to_sim(all_velocities, env_ids=env_ids)

    # Get vector from origin to obstacle (in env-local frame, XY plane)
    # positions already includes env_origins, so subtract to get the local direction
    obstacle_local_xy = positions[:, 0:2] - env_origins[:, 0:2]  # (num_envs, 2)

    # Sample how far past the obstacle to place the drone
    drone_offsets = torch.rand(num_envs, device=env.device) * (drone_offset_max - drone_offset_min) + drone_offset_min

    # Normalize — radii is guaranteed > 0 if radius_min > 0; guard anyway
    directions = obstacle_local_xy / radii.unsqueeze(-1).clamp(min=1e-6)

    # Robot
    robot: RigidObject | Articulation = env.scene[robot_cfg.name]

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=robot.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=robot.device)

    # Place drone on the vector from origin past the obstacle
    drone_distance = radii + drone_offsets  # (num_envs,)
    drone_positions = torch.zeros(num_envs, 3, device=env.device)
    drone_positions[:, 0] = drone_distance * directions[:, 0]
    drone_positions[:, 1] = drone_distance * directions[:, 1]
    drone_positions[:, 2] = rand_samples[:, 2]
    drone_positions += env_origins

    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    drone_quat = torch.zeros(num_envs, 4, device=env.device)
    drone_quat[:, 3] = 1
    drone_quat = math_utils.quat_mul(drone_quat, orientations_delta)
    drone_pose = torch.cat([drone_positions, drone_quat], dim=-1)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=robot.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=robot.device)

    drone_velocity = torch.zeros(num_envs, 6, device=env.device) + rand_samples
    robot.write_root_pose_to_sim(drone_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(drone_velocity, env_ids=env_ids)


def reset_root_state_on_trajectory(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str = "trajectory",
    pos_noise: tuple[float, float] = (-0.25, 0.25),
    vel_noise: tuple[float, float] = (-0.2, 0.2),
    yaw_noise: tuple[float, float] = (-0.3, 0.3),
    rp_noise: tuple[float, float] = (-0.1, 0.1),
    angvel_noise: tuple[float, float] = (-0.2, 0.2),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the drone onto the start of a freshly sampled reference trajectory.

    Reset events run *before* the command manager resamples, so the trajectory the drone should be
    spawned on does not exist yet at this point. This function therefore drives the sampling itself:
    it asks the trajectory command term for a new trajectory, evaluates it at ``t = 0``, writes the
    robot state around that reference, and flags the term so it keeps the trajectory instead of
    immediately drawing another one.

    Noise on the yaw is applied *relative to the reference yaw* rather than uniformly. With a
    per-episode random reference yaw the absolute heading distribution is uniform either way, but
    keeping the initial yaw *error* small means an episode starts as a tracking problem rather than
    as a yaw recovery problem.

    Args:
        env: The manager-based RL environment instance.
        env_ids: Indices of the environments to reset.
        command_name: Name of the trajectory command term to sample and read the reference from.
        pos_noise: Range of the per-axis position offset from the reference [m].
        vel_noise: Range of the per-axis velocity offset from the reference [m/s].
        yaw_noise: Range of the yaw offset from the reference yaw [rad].
        rp_noise: Range of the initial roll and pitch [rad].
        angvel_noise: Range of the initial angular velocity about each axis [rad/s].
        asset_cfg: SceneEntityCfg identifying the asset to reset.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)

    # Draw the trajectory now so the spawn state can be matched to it, and tell the command term
    # not to redraw it when it resamples a moment later.
    command_term.sample_parameters(env_ids)
    command_term.evaluate_at(env_ids, torch.zeros(len(env_ids), device=asset.device))
    command_term._sampled_by_event[env_ids] = True

    def _sample(noise_range: tuple[float, float], size: tuple[int, ...]) -> torch.Tensor:
        return math_utils.sample_uniform(noise_range[0], noise_range[1], size, device=asset.device)

    num_resets = len(env_ids)
    positions = command_term.pos_ref_w[env_ids] + env.scene.env_origins[env_ids] + _sample(pos_noise, (num_resets, 3))
    roll = _sample(rp_noise, (num_resets,))
    pitch = _sample(rp_noise, (num_resets,))
    yaw = command_term.yaw_ref[env_ids] + _sample(yaw_noise, (num_resets,))
    orientations = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

    velocities = torch.cat(
        [
            command_term.vel_ref_w[env_ids] + _sample(vel_noise, (num_resets, 3)),
            _sample(angvel_noise, (num_resets, 3)),
        ],
        dim=-1,
    )

    asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)


class WindAndDrag(ManagerTermBase):
    """Apply a sustained wind gust and aerodynamic drag as a single combined wrench.

    Wind and drag are deliberately handled by one term rather than two. Both act on the same
    persistent external-wrench buffer, so two independent terms would simply overwrite each other
    and only whichever ran last would take effect.

    Drag matters here because the simulated airframe has ``linear_damping = 0.0`` and therefore no
    aerodynamic resistance at all. A policy trained without it learns to command velocities that are
    systematically too high for a real vehicle, and the underlying Lee controller has no integral
    term to absorb the difference. The drag model is the usual quadratic one,
    ``F = -0.5 * rho * Cd * A * |v| * v``, with the drag area sampled per environment at reset so
    the policy has to be robust to it rather than learning one exact value.

    This term is intended to run every policy step with ``is_global_time=True``, so the drag force
    tracks the current velocity. The gust is resampled on its own slower timer and held in between,
    which models sustained wind rather than an impulse.

    Args:
        cfg: Event term configuration.
        env: The manager-based RL environment instance.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        asset_cfg.resolve(env.scene)
        self._asset_cfg = asset_cfg
        asset: Articulation = env.scene[asset_cfg.name]

        num_envs, device = env.num_envs, asset.device
        self._gust_force = torch.zeros(num_envs, 3, device=device)
        self._gust_torque = torch.zeros(num_envs, 3, device=device)
        self._gust_time_left = torch.zeros(num_envs, device=device)
        self._drag_area = torch.zeros(num_envs, device=device)
        self._all_env_ids = torch.arange(num_envs, device=device, dtype=torch.int32)

    def reset(self, env_ids: torch.Tensor | None = None):
        """Resample the per-environment drag area and force an immediate gust resample."""
        if env_ids is None:
            env_ids = slice(None)
        drag_area_range = self.cfg.params.get("drag_area_range", (0.10, 0.30))
        shape = self._drag_area[env_ids].shape
        self._drag_area[env_ids] = math_utils.sample_uniform(*drag_area_range, shape, self._drag_area.device)
        self._gust_time_left[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        force_range: tuple[float, float] = (-6.0, 6.0),
        torque_range: tuple[float, float] = (-0.4, 0.4),
        gust_interval_range_s: tuple[float, float] = (2.0, 6.0),
        drag_area_range: tuple[float, float] = (0.10, 0.30),
        air_density: float = 1.225,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        """Write the combined gust and drag wrench for every environment.

        Args:
            env: The manager-based RL environment instance.
            env_ids: Unused. The term always writes every environment so the drag stays current.
            force_range: Range of each gust force component [N].
            torque_range: Range of each gust torque component [N*m].
            gust_interval_range_s: Range of the interval between gust resamples [s].
            drag_area_range: Range of the effective drag area ``Cd * A`` [m^2], sampled per
                environment at reset.
            air_density: Air density [kg/m^3].
            asset_cfg: SceneEntityCfg identifying the asset and the body the wrench acts on.
        """
        asset: Articulation = env.scene[asset_cfg.name]
        device = asset.device

        # -- gust: resample only the environments whose hold has expired, keep the rest
        self._gust_time_left -= env.step_dt
        expired = (self._gust_time_left <= 0.0).nonzero().flatten()
        if len(expired) > 0:
            self._gust_force[expired] = math_utils.sample_uniform(*force_range, (len(expired), 3), device)
            self._gust_torque[expired] = math_utils.sample_uniform(*torque_range, (len(expired), 3), device)
            self._gust_time_left[expired] = math_utils.sample_uniform(*gust_interval_range_s, (len(expired),), device)

        # -- drag: opposes the current world-frame velocity, quadratic in speed
        velocity = asset.data.root_lin_vel_w.torch
        speed = torch.linalg.norm(velocity, dim=-1, keepdim=True)
        drag = -0.5 * air_density * self._drag_area.unsqueeze(-1) * speed * velocity

        forces = (self._gust_force + drag).unsqueeze(1)
        torques = self._gust_torque.unsqueeze(1)
        asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces,
            torques=torques,
            body_ids=asset_cfg.body_ids,
            env_ids=self._all_env_ids,
        )
