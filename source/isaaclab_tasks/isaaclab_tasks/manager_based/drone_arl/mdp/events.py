# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event functions specific to the drone ARL environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

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
    radii = torch.sqrt(
        torch.rand(num_envs, device=env.device) * (radius_max_m**2 - radius_min_m**2) + radius_min_m**2
    )

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
    drone_offsets = (
        torch.rand(num_envs, device=env.device) * (drone_offset_max - drone_offset_min)
        + drone_offset_min
    )
    
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