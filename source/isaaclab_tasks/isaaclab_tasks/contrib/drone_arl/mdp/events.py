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
from isaaclab.utils import index_fill_

from .curriculums import get_obstacle_curriculum_term
from .utils import resolve_env_ids

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
    num_envs = len(range(env.num_envs)[env_ids]) if isinstance(env_ids, slice) else len(env_ids)
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
        index_fill_(active_masks[env_idx], perm, True)

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
        positions += env.scene.env_origins[env_ids][active_env_indices]

        # Generate quaternions
        quats = math_utils.random_orientation(num_envs, device=env.device)

        # Write poses
        all_poses[envs_need_obstacle, obj_idx, 0:3] = positions
        all_poses[envs_need_obstacle, obj_idx, 3:7] = quats[envs_need_obstacle]

        # Move inactive obstacles far away
        inactive = ~envs_need_obstacle
        all_poses[inactive, obj_idx, 0:3] = env.scene.env_origins[env_ids][inactive] + torch.tensor(
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
    env_ids = resolve_env_ids(env_ids, env.num_envs, env.device)
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
    rest_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
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
        rest_offset: Position of the tracked frame relative to the root when the vehicle is level
            and any payload hangs at rest, in the yaw-aligned vehicle frame [m]. The root is placed
            this far from the reference, so that the tracked frame starts on it. Zero when the
            command tracks the vehicle itself.
        asset_cfg: SceneEntityCfg identifying the asset to reset.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    env_ids = resolve_env_ids(env_ids, env.num_envs, asset.device)

    # Draw the trajectory now so the spawn state can be matched to it, and tell the command term
    # not to redraw it when it resamples a moment later.
    command_term.sample_parameters(env_ids)
    command_term.evaluate_at(env_ids, torch.zeros(len(env_ids), device=asset.device))
    command_term._sampled_by_event[env_ids] = True

    def _sample(noise_range: tuple[float, float], size: tuple[int, ...]) -> torch.Tensor:
        return math_utils.sample_uniform(noise_range[0], noise_range[1], size, device=asset.device)

    num_resets = len(env_ids)
    roll = _sample(rp_noise, (num_resets,))
    pitch = _sample(rp_noise, (num_resets,))
    yaw = command_term.yaw_ref[env_ids] + _sample(yaw_noise, (num_resets,))
    orientations = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

    zeros = torch.zeros_like(yaw)
    rest_offset_w = math_utils.quat_apply(
        math_utils.quat_from_euler_xyz(zeros, zeros, yaw),
        torch.tensor(rest_offset, device=asset.device).expand(num_resets, 3),
    )
    positions = (
        command_term.pos_ref_w[env_ids]
        - rest_offset_w
        + env.scene.env_origins[env_ids]
        + _sample(pos_noise, (num_resets, 3))
    )

    velocities = torch.cat(
        [
            command_term.vel_ref_w[env_ids] + _sample(vel_noise, (num_resets, 3)),
            _sample(angvel_noise, (num_resets, 3)),
        ],
        dim=-1,
    )

    asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)


def push_joints_by_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Kick the selected joints by adding a random velocity to their current velocity.

    Meant for ``mode="interval"``, to excite a passive joint such as a payload hinge mid-episode.
    :func:`~isaaclab.envs.mdp.reset_joints_by_offset` is not a substitute: it sets the state about
    the default, so it would also snap the joint back to rest. This term leaves the positions alone.

    Args:
        env: The manager-based RL environment instance.
        env_ids: Indices of the environments to push.
        velocity_range: Range of the velocity added to each selected joint [m/s or rad/s, depending
            on joint type].
        asset_cfg: SceneEntityCfg identifying the asset and the joints to push.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    iter_env_ids = env_ids[:, None] if asset_cfg.joint_ids != slice(None) else env_ids

    joint_vel = asset.data.joint_vel.torch[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)
    joint_vel_limits = asset.data.soft_joint_vel_limits.torch[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def reset_swing_graded_by_env(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    amplitude_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Release a two-axis pendulum from rest at a swing amplitude that grows with the environment index.

    Meant for evaluation: environment ``0`` gets the lower end of :paramref:`amplitude_range`, the last
    environment the upper end, and those between are spaced evenly, so one run sweeps the whole range
    and each environment sees the same amplitude in every episode. The direction of the swing is drawn
    uniformly at random on each reset.

    The amplitude is split over the two hinge axes as ``(A cos(phi), A sin(phi))``, which is the tilt
    from the rest pose to first order, and each axis is clamped to its soft limits. The joint velocities
    are zeroed, so the swing starts at its peak.

    Args:
        env: The manager-based RL environment instance.
        env_ids: Indices of the environments to reset.
        amplitude_range: Swing amplitude of the first and of the last environment [rad].
        asset_cfg: SceneEntityCfg identifying the asset and exactly two hinge joints, outer axis first.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = resolve_env_ids(env_ids, env.num_envs, asset.device)
    iter_env_ids = env_ids[:, None]

    fraction = env_ids.float() / max(env.num_envs - 1, 1)
    amplitude = amplitude_range[0] + (amplitude_range[1] - amplitude_range[0]) * fraction
    direction = math_utils.sample_uniform(0.0, 2.0 * torch.pi, (len(env_ids),), device=asset.device)

    joint_pos = asset.data.default_joint_pos.torch[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_pos[:, 0] += amplitude * torch.cos(direction)
    joint_pos[:, 1] += amplitude * torch.sin(direction)
    joint_pos_limits = asset.data.soft_joint_pos_limits.torch[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(joint_pos), joint_ids=asset_cfg.joint_ids, env_ids=env_ids
    )


def _sample_swing_magnitude(
    num: int, magnitude_range: tuple[float, float], zero_prob: float, device: str | torch.device
) -> torch.Tensor:
    """Sample a swing magnitude that is zero with probability :paramref:`zero_prob` and otherwise
    log-uniform over :paramref:`magnitude_range`.

    A log-uniform draw gives every scale the same weight, so a range spanning 1 to 40 deg is as likely
    to produce a 1-3 deg swing as a 13-40 deg one. Drawing each hinge axis uniformly instead makes the
    density of the magnitude fall to zero at zero (a swing under 10% of the range is a 1-in-100 draw on
    two axes), so a policy trained on it comes to expect a large swing.

    Args:
        num: Number of samples.
        magnitude_range: Smallest and largest non-zero magnitude. Both must be positive.
        zero_prob: Probability of a zero magnitude.
        device: Device to create the tensor on.

    Returns:
        A 1-D tensor of shape (num,) with the magnitudes.
    """
    magnitude = math_utils.sample_log_uniform(*magnitude_range, (num,), device)
    return magnitude * (torch.rand(num, device=device) >= zero_prob)


def _hinge_rest_offset(
    env: ManagerBasedRLEnv, rest_term_name: str | None, env_ids: torch.Tensor
) -> torch.Tensor | float:
    """Hinge angles at which the payload hangs at rest, relative to the default joint angles [rad].

    Read from a :class:`RandomizePayloadHangOffset` event term when :paramref:`rest_term_name` is set,
    so the swing terms act about the randomized rest pose rather than about the nominal one.
    """
    if rest_term_name is None:
        return 0.0
    return env.event_manager.get_term_cfg(rest_term_name).func.rest_offset[env_ids]


def reset_swing_by_amplitude(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    amplitude_range: tuple[float, float],
    still_prob: float,
    natural_frequency: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    rest_term_name: str | None = None,
) -> None:
    """Start a two-axis pendulum swinging with a random amplitude, direction and phase.

    The amplitude is zero (hanging still) with probability :paramref:`still_prob` and otherwise
    log-uniform over :paramref:`amplitude_range` (see :func:`_sample_swing_magnitude`), so every episode
    is not one that starts with a large swing. The swing is planar, in a direction drawn uniformly, and
    starts at a random phase of its cycle: with amplitude ``A``, phase ``psi`` and natural frequency
    ``w`` the hinge angle is ``A cos(psi)`` and its rate ``-A w sin(psi)`` along that direction, so a
    swing can start at its peak, through the bottom, or anywhere between. The swing is about the rest
    pose, the default joint angles plus the offset of the :class:`RandomizePayloadHangOffset` term
    named by :paramref:`rest_term_name`, if any. Angles are clamped to the soft joint limits.

    Args:
        env: The manager-based RL environment instance.
        env_ids: Indices of the environments to reset.
        amplitude_range: Smallest and largest non-zero swing amplitude [rad]. Both must be positive.
        still_prob: Probability of starting with no swing at all.
        natural_frequency: Natural frequency of the pendulum, ``sqrt(g / L)`` for a point mass [rad/s].
            Only sets how the swing energy is split between angle and rate at the sampled phase.
        asset_cfg: SceneEntityCfg identifying the asset and exactly two hinge joints, outer axis first.
        rest_term_name: Name of the :class:`RandomizePayloadHangOffset` event term that holds the rest
            pose. None for the default joint angles.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = resolve_env_ids(env_ids, env.num_envs, asset.device)
    iter_env_ids = env_ids[:, None]
    num = len(env_ids)

    amplitude = _sample_swing_magnitude(num, amplitude_range, still_prob, asset.device)
    direction = math_utils.sample_uniform(0.0, 2.0 * torch.pi, (num,), device=asset.device)
    phase = math_utils.sample_uniform(0.0, 2.0 * torch.pi, (num,), device=asset.device)
    axis = torch.stack([torch.cos(direction), torch.sin(direction)], dim=-1)

    joint_pos = asset.data.default_joint_pos.torch[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_pos += _hinge_rest_offset(env, rest_term_name, env_ids)
    joint_pos += (amplitude * torch.cos(phase)).unsqueeze(-1) * axis
    joint_vel = (-amplitude * natural_frequency * torch.sin(phase)).unsqueeze(-1) * axis
    joint_pos_limits = asset.data.soft_joint_pos_limits.torch[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


def push_swing_by_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    speed_range: tuple[float, float],
    skip_prob: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Kick a two-axis pendulum by adding a hinge rate of random size and direction.

    The same impulse as :func:`push_joints_by_velocity`, but with the size drawn like the swing of
    :func:`reset_swing_by_amplitude`: no kick with probability :paramref:`skip_prob`, otherwise
    log-uniform over :paramref:`speed_range`, in a direction drawn uniformly. Small knocks are then as
    common as large ones, rather than the per-axis uniform draw's rare.

    Args:
        env: The manager-based RL environment instance.
        env_ids: Indices of the environments to push.
        speed_range: Smallest and largest non-zero hinge rate added [rad/s]. Both must be positive. A
            rate ``v`` adds a swing of about ``v / w`` [rad] to a pendulum at rest, with ``w`` its
            natural frequency.
        skip_prob: Probability that a kick is skipped.
        asset_cfg: SceneEntityCfg identifying the asset and exactly two hinge joints, outer axis first.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = resolve_env_ids(env_ids, env.num_envs, asset.device)
    iter_env_ids = env_ids[:, None]
    num = len(env_ids)

    speed = _sample_swing_magnitude(num, speed_range, skip_prob, asset.device)
    direction = math_utils.sample_uniform(0.0, 2.0 * torch.pi, (num,), device=asset.device)

    joint_vel = asset.data.joint_vel.torch[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel[:, 0] += speed * torch.cos(direction)
    joint_vel[:, 1] += speed * torch.sin(direction)
    joint_vel_limits = asset.data.soft_joint_vel_limits.torch[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)


class RandomizePayloadHangOffset(ManagerTermBase):
    """Make a hanging payload rest at a random hinge offset, by moving the COM of one of its bodies.

    A real payload does not hang exactly where its hinge encoders read zero: the Mk3 chainsaw hangs at
    about 5.4 deg forward and 3 deg right. A policy trained only on a payload that rests at zero reads
    any non-zero hinge angle as swing. This term draws a rest offset per environment and moves the COM
    of :paramref:`shifted_body` sideways so that the payload COM lies straight below the pivot at that
    offset; the hinges then read the offset at rest, as the encoders would.

    The shift is first order: with the vehicle level, ``+outer`` hinge moving the payload COM aft (-x)
    and ``+inner`` moving it left (+y) of the airframe, a rest offset ``(a, b)`` needs the payload COM,
    at zero hinge angles, ``L * (a, -b)`` further along the airframe x and y, with ``L`` the
    pivot-to-COM distance. The residual tilt at the offset is at most about a tenth of a degree at the
    offsets used here. The swing terms read :attr:`rest_offset` (see :func:`reset_swing_by_amplitude`).

    It must run after any startup term that changes the payload mass, and with the vehicle at its
    spawn pose and the hinges at their default angles, which is the state at startup. The COM write
    goes through the CPU tensor API, so the term is meant for ``mode="startup"``.

    The ``params`` of the term configuration are:

    * ``asset_cfg``: SceneEntityCfg with exactly two hinge joints, outer axis first.
    * ``payload_cfg``: SceneEntityCfg with every body below the pivot as ``body_names``.
    * ``pivot_cfg``: SceneEntityCfg with the body whose link frame origin is the pivot.
    * ``shifted_body_cfg``: SceneEntityCfg with the one payload body whose COM is moved.
    * ``airframe_cfg``: SceneEntityCfg with the airframe body, whose x and y axes the signs above use.
    * ``offset_range``: Range of the rest offset drawn uniformly for each hinge [rad].
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.rest_offset = torch.zeros(env.num_envs, 2, device=env.device)
        """Hinge angles at which the payload rests, relative to the default angles [rad]. Shape (num_envs, 2)."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        payload_cfg: SceneEntityCfg,
        pivot_cfg: SceneEntityCfg,
        shifted_body_cfg: SceneEntityCfg,
        airframe_cfg: SceneEntityCfg,
        offset_range: tuple[float, float],
    ) -> None:
        """Draw the rest offsets and write the shifted COMs.

        Args:
            env: The manager-based RL environment instance.
            env_ids: Indices of the environments to randomize. All environments if None.
        """
        asset: Articulation = env.scene[asset_cfg.name]
        env_ids = resolve_env_ids(env_ids, env.num_envs, asset.device)
        data = asset.data
        payload_ids = payload_cfg.body_ids
        shifted_id: int = shifted_body_cfg.body_ids[0]  # type: ignore[index]

        offset = math_utils.sample_uniform(*offset_range, (len(env_ids), 2), device=asset.device)
        self.rest_offset[env_ids] = offset

        mass = data.body_mass.torch[env_ids[:, None], payload_ids].to(asset.device)
        com_w = torch.sum(mass.unsqueeze(-1) * data.body_com_pos_w.torch[env_ids[:, None], payload_ids], dim=1)
        com_w = com_w / mass.sum(dim=1, keepdim=True)
        length = torch.norm(com_w - data.body_link_pos_w.torch[env_ids, pivot_cfg.body_ids[0]], dim=-1)

        # Payload COM shift in the airframe frame, then the shifted body's share of it in its own frame.
        shift_a = torch.zeros(len(env_ids), 3, device=asset.device)
        shift_a[:, 0] = length * offset[:, 0]
        shift_a[:, 1] = -length * offset[:, 1]
        shift_w = math_utils.quat_apply(data.body_link_quat_w.torch[env_ids, airframe_cfg.body_ids[0]], shift_a)
        shifted_mass = data.body_mass.torch[env_ids, shifted_id].to(asset.device)
        shift_w = shift_w * (mass.sum(dim=1) / shifted_mass).unsqueeze(-1)
        shift_b = math_utils.quat_apply_inverse(data.body_link_quat_w.torch[env_ids, shifted_id], shift_w)

        coms = data.body_com_pose_b.torch[env_ids, shifted_id].clone()
        coms[:, :3] += shift_b
        asset.set_coms_index(
            coms=coms.unsqueeze(1),
            body_ids=torch.tensor([shifted_id], dtype=torch.int, device=asset.device),
            env_ids=env_ids.int(),
        )


class RandomizePassiveJointDamping(ManagerTermBase):
    """Scale the simulated damping of unactuated joints by a random factor per environment.

    Passive hinges such as a hanging payload's are often modelled as a joint drive with zero
    stiffness, so their damping lives on the drive rather than in the joint friction.
    :func:`~isaaclab.envs.mdp.randomize_joint_parameters` does not touch the drive, and
    :class:`~isaaclab.envs.mdp.randomize_actuator_gains` only reaches joints that belong to an
    actuator, which a :class:`~isaaclab_contrib.assets.Multirotor` cannot have. This term writes the
    drive damping directly.

    The factor is drawn independently for each joint and is applied to the damping the asset had
    when the term was created, so repeated calls do not compound. The write goes through the CPU
    tensor API, so the term is meant for ``mode="startup"``.

    Args:
        cfg: Event term configuration.
        env: The manager-based RL environment instance.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset_cfg.resolve(env.scene)
        asset: Articulation = env.scene[asset_cfg.name]
        self._joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int32, device=asset.device)
        self._default_damping = asset.data.joint_damping.torch[:, self._joint_ids.long()].clone()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        scale_range: tuple[float, float] = (0.5, 2.0),
    ) -> None:
        """Write the scaled damping of the selected joints into the simulation.

        Args:
            env: The manager-based RL environment instance.
            env_ids: Indices of the environments to randomize. All environments if None.
            asset_cfg: SceneEntityCfg identifying the asset and the joints to randomize.
            scale_range: Range of the factor applied to the default damping, sampled log-uniformly
                so that halving and doubling are equally likely.
        """
        asset: Articulation = env.scene[asset_cfg.name]
        env_ids = resolve_env_ids(env_ids, env.num_envs, asset.device)

        scale = math_utils.sample_log_uniform(*scale_range, (len(env_ids), len(self._joint_ids)), asset.device)
        damping = self._default_damping[env_ids] * scale
        asset.write_joint_damping_to_sim_index(damping=damping, joint_ids=self._joint_ids, env_ids=env_ids)


def set_passive_joint_stiffness(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness: float,
    scale_range: tuple[float, float] = (1.0, 1.0),
) -> None:
    """Give unactuated joints a torsional spring towards zero joint angle.

    The spring is the joint drive's stiffness, so it pulls towards the drive target, which is zero
    for a passive joint. Use it for physical springs on a hinge, such as the ones holding a balanced
    payload at its rest pose. Like :class:`RandomizePassiveJointDamping`, it writes the drive directly
    because a :class:`~isaaclab_contrib.assets.Multirotor` cannot have joint actuators, and it goes
    through the CPU tensor API, so the term is meant for ``mode="startup"``.

    Args:
        env: The manager-based RL environment instance.
        env_ids: Indices of the environments to write. All environments if None.
        asset_cfg: SceneEntityCfg identifying the asset and the joints to give a spring.
        stiffness: Spring stiffness [N·m/rad]. Zero removes the spring.
        scale_range: Range of the factor applied to :paramref:`stiffness`, sampled log-uniformly
            and independently per environment and joint.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = resolve_env_ids(env_ids, env.num_envs, asset.device)
    joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int32, device=asset.device)

    scale = math_utils.sample_log_uniform(*scale_range, (len(env_ids), len(joint_ids)), asset.device)
    asset.write_joint_stiffness_to_sim_index(stiffness=stiffness * scale, joint_ids=joint_ids, env_ids=env_ids)


def set_passive_joint_friction(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction: float,
    scale_range: tuple[float, float] = (1.0, 1.0),
) -> None:
    """Give unactuated joints dry (Coulomb) friction.

    Sets the PhysX static and dynamic joint friction to the same value and leaves the viscous
    coefficient alone. Unlike drive damping, dry friction takes off a fixed amount of swing
    per cycle regardless of amplitude, and can hold a joint still at a small angle from rest.
    :func:`~isaaclab.envs.mdp.randomize_joint_parameters` would apply the same value to the viscous
    coefficient as well, hence this term. It goes through the CPU tensor API, so it is meant for
    ``mode="startup"``.

    Args:
        env: The manager-based RL environment instance.
        env_ids: Indices of the environments to write. All environments if None.
        asset_cfg: SceneEntityCfg identifying the asset and the joints to give friction.
        friction: Friction effort [N·m or N, depending on joint type], used for both static and
            dynamic friction.
        scale_range: Range of the factor applied to :paramref:`friction`, sampled log-uniformly
            and independently per environment and joint.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    env_ids = resolve_env_ids(env_ids, env.num_envs, asset.device)
    joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int32, device=asset.device)

    scale = math_utils.sample_log_uniform(*scale_range, (len(env_ids), len(joint_ids)), asset.device)
    coeff = friction * scale
    asset.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=coeff, joint_dynamic_friction_coeff=coeff, joint_ids=joint_ids, env_ids=env_ids
    )


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
