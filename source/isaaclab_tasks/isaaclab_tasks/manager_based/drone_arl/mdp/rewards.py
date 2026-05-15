# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from .curriculums import get_obstacle_curriculum_term

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject, Articulation
    from isaaclab.envs import ManagerBasedRLEnv

"""
Drone control rewards.
"""

def distance_to_goal_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "target_pose",
) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the commanded target position
    and the body position of the asset.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
            Must have ``body_ids`` configured to select the tracked body link.
        command_name: Name of the command to read the target pose from the
            environment's command manager. The function expects the command
            tensor to contain positions in its first three columns.

    Returns:
        A 1-D tensor of shape (num_envs,) containing the per-environment distance
        errors to be combined with a penalty i.e. negative weight.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    current_position = asset.data.body_pos_w.torch[:, asset_cfg.body_ids[0]] - env.scene.env_origins # type: ignore
    
    # compute the error
    position_error = torch.norm(command[:, :3] - current_position, dim=1)
    return position_error

def distance_to_goal_tanh(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 1.0,
    command_name: str = "target_pose",
) -> torch.Tensor:
    """Reward the distance to a goal using a tanh kernel.

    The function computes the position error between the commanded target position
    and the body position of the asset and maps it with a tanh kernel.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
            Must have ``body_ids`` configured to select the tracked body link.
        std: Scaling parameter for the tanh kernel. Controls the distance at which
            the reward transitions from high to low — smaller values produce a
            sharper falloff concentrated near the goal, larger values spread the
            gradient over a wider range.
        command_name: Name of the command to read the target pose from the
            environment's command manager. The function expects the command
            tensor to contain positions in its first three columns.

    Returns:
        A 1-D tensor of shape (num_envs,) containing the per-environment reward
        values in [0, 1], with 1.0 when the position error is zero.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    current_position = asset.data.body_pos_w.torch[:, asset_cfg.body_ids[0]] - env.scene.env_origins  # type: ignore
    position_error = torch.norm(command[:, :3] - current_position, dim=1)
    return 1 - torch.tanh(position_error / std)

def distance_to_goal_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 1.0,
    command_name: str = "target_pose",
) -> torch.Tensor:
    """Reward the distance to a goal position using an exponential kernel.

    This reward computes an exponential falloff of the squared Euclidean distance
    between the commanded target position and the body position of the asset.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
            Must have ``body_ids`` configured to select the tracked body link.
        std: Standard deviation used in the exponential kernel; larger values
            produce a gentler falloff.
        command_name: Name of the command to read the target pose from the
            environment's command manager. The function expects the command
            tensor to contain positions in its first three columns.

    Returns:
        A 1-D tensor of shape (num_envs,) containing the per-environment reward
        values in [0, 1], with 1.0 when the position error is zero.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    current_position = asset.data.body_pos_w.torch[:, asset_cfg.body_ids[0]] - env.scene.env_origins  # type: ignore
    position_error_square = torch.sum(torch.square(command[:, :3] - current_position), dim=1)
    return torch.exp(-position_error_square / std**2)


def distance_to_goal_exp_curriculum(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 1.0,
    command_name: str = "target_pose",
) -> torch.Tensor:
    """Reward the distance to a goal position using an exponential kernel with curriculum-based scaling.

    This reward extends :func:`distance_to_goal_exp` by applying a scaling factor that
    increases with the obstacle difficulty level. As the curriculum progresses and obstacle
    density increases, the reward weight grows to compensate for the added difficulty.

    The scaling weight is computed as: ``1.0 + (difficulty_level / max_difficulty)``, meaning
    the reward scales from 1.0x (at minimum difficulty) to 2.0x (at maximum difficulty).

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
            Must have ``body_ids`` configured to select the tracked body link.
        std: Standard deviation used in the exponential kernel; larger values
            produce a gentler falloff. Defaults to 1.0.
        command_name: Name of the command to read the target pose from the
            environment's command manager. The function expects the command
            tensor to contain positions in its first three columns.

    Returns:
        A 1-D tensor of shape (num_envs,) containing the per-environment weighted
        reward values. Values are in [0, weight], where weight varies based on the
        current curriculum difficulty level.

    Note:
        If no curriculum is active (i.e., ObstacleDensityCurriculum is not found),
        the function behaves identically to :func:`distance_to_goal_exp` with weight=1.0.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    current_position = asset.data.body_pos_w.torch[:, asset_cfg.body_ids[0]] - env.scene.env_origins  # type: ignore
    position_error_square = torch.sum(torch.square(command[:, :3] - current_position), dim=1)

    # Get curriculum term and compute weight
    curriculum_term = get_obstacle_curriculum_term(env)
    if curriculum_term is not None:
        weight = 1.0 + curriculum_term.difficulty_levels.float() / float(curriculum_term.max_difficulty)
    else:
        weight = 1.0

    return weight * torch.exp(-position_error_square / std**2)


def ang_vel_xyz_exp(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 1.0
) -> torch.Tensor:
    """Penalize angular velocity magnitude with an exponential kernel.

    This reward computes exp(-||omega||^2 / std^2) where omega is the body-frame
    angular velocity of the asset. It is useful for encouraging low rotational
    rates.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
        std: Standard deviation used in the exponential kernel; controls
            sensitivity to angular velocity magnitude.

    Returns:
        A 1-D tensor of shape (num_envs,) with values in (0, 1], where 1 indicates
        zero angular velocity.
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # compute squared magnitude of angular velocity (all axes)
    ang_vel_squared = torch.sum(torch.square(asset.data.root_ang_vel_b.torch), dim=1)

    return torch.exp(-ang_vel_squared / std**2)


def velocity_to_goal_reward_curriculum(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), command_name: str = "target_pose"
) -> torch.Tensor:
    """Reward velocity alignment toward the goal with curriculum-based scaling.

    This reward encourages the agent to move in the direction of the goal by computing
    the dot product between the asset's velocity vector and the normalized direction
    vector to the goal. A curriculum-based scaling factor is applied that increases
    with obstacle difficulty.

    The reward is positive when moving toward the goal, negative when moving away,
    and zero when moving perpendicular to the goal direction. The magnitude scales
    linearly with speed in the goal direction.

    The scaling weight is computed as: 1.0 + (difficulty_level / max_difficulty),
    allowing the reward to scale from 1.0x to 2.0x as difficulty increases.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
        command_name: Name of the command to read the target pose from the
            environment's command manager. The function expects the command
            tensor to contain positions in its first three columns.

    Returns:
        A 1-D tensor of shape (num_envs,) containing the per-environment weighted
        reward values. Values can be positive (moving toward goal), negative
        (moving away), or zero (perpendicular motion), scaled by the curriculum weight.

    Note:
        If no curriculum is active (i.e., ObstacleDensityCurriculum is not found),
        the function uses weight=1.0 without curriculum scaling.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # get the center of the environment
    command = env.command_manager.get_command(command_name)

    current_position = wp.to_torch(asset.data.root_pos_w) - env.scene.env_origins
    direction_to_goal = command[:, :3] - current_position
    direction_to_goal = direction_to_goal / (torch.norm(direction_to_goal, dim=1, keepdim=True) + 1e-8)
    # compute the reward as the dot product between the velocity and the direction to the goal
    velocity_towards_goal = torch.sum(wp.to_torch(asset.data.root_lin_vel_w) * direction_to_goal, dim=1)

    # Get curriculum term and compute weight
    curriculum_term = get_obstacle_curriculum_term(env)
    if curriculum_term is not None:
        weight = 1.0 + curriculum_term.difficulty_levels.float() / float(curriculum_term.max_difficulty)
    else:
        weight = 1.0

    return weight * velocity_towards_goal


def lin_vel_xyz_exp(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 1.0
) -> torch.Tensor:
    """Penalize linear velocity magnitude with an exponential kernel.

    Computes exp(-||v||^2 / std^2) where v is the asset's linear velocity in
    world frame. Useful for encouraging the agent to reduce translational speed.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
        std: Standard deviation used in the exponential kernel.

    Returns:
        A 1-D tensor of shape (num_envs,) with values in (0, 1], where 1 indicates
        zero linear velocity.
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # compute squared magnitude of linear velocity (all axes)
    lin_vel_squared = torch.sum(torch.square(asset.data.root_lin_vel_w.torch), dim=1)

    return torch.exp(-lin_vel_squared / std**2)


def yaw_aligned(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 0.5
) -> torch.Tensor:
    """Reward alignment of the vehicle's yaw to zero using an exponential kernel.

    The function extracts the yaw (rotation about Z) from the world-frame root
    quaternion and computes exp(-yaw^2 / std^2). This encourages heading
    alignment to a zero-yaw reference.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
        std: Standard deviation used in the exponential kernel; smaller values
            make the reward more sensitive to yaw deviations.

    Returns:
        A 1-D tensor of shape (num_envs,) with values in (0, 1], where 1 indicates
        perfect yaw alignment (yaw == 0).
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # extract yaw from current orientation
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w.torch)

    # normalize yaw to [-pi, pi] (target is 0)
    yaw = math_utils.wrap_to_pi(yaw)

    # return exponential reward (1 when yaw=0, approaching 0 when rotated)
    return torch.exp(-(yaw**2) / std**2)

def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg  = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint position deviation from a target value using a circular L2-norm.

    Joint positions are read as absolute angles in the USD joint coordinate frame.
    The difference from the target is wrapped to ``(-π, π]`` before squaring, which
    keeps the penalty continuous everywhere — including when the target is near ±π,
    where wrapping the raw position first would create a large discontinuity.

    Args:
        env: The manager-based RL environment instance.
        asset_cfg: SceneEntityCfg identifying the asset (defaults to "robot").
            Configure ``joint_ids`` to select specific joints.
        target: Target joint angle [rad], in the same absolute coordinate frame
            as ``asset.data.joint_pos``.

    Returns:
        A 1-D tensor of shape (num_envs,) containing the per-environment sum of
        squared angular errors, intended for use with a negative reward weight.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = asset.data.joint_pos.torch
    if asset_cfg.joint_ids is not None:
        joint_pos = joint_pos[:, asset_cfg.joint_ids]

    return torch.sum(torch.square(math_utils.wrap_to_pi(joint_pos - target)), dim=1)

def upright_while_moving(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg  = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    # pole error (squared deviation from upright)
    pitch = asset.data.joint_pos[:, asset_cfg.joint_ids[0]]
    roll = asset.data.joint_pos[:, asset_cfg.joint_ids[1]]
    pole_error_sq = pitch**2 + roll**2

    # base speed (horizontal plane)
    base_speed = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=-1)

    # reward is "upright bonus weighted by how much you're moving"
    # the +0.3 floor keeps stationary upright still slightly rewarded
    weight = base_speed + 0.3
    return -pole_error_sq * weight


class PoleEnergy(ManagerTermBase):
    """Penalize pole swing energy: E = ½·I·ω² + m·g·L·(1 − cos θ).

    Physical constants (mass, pendulum length, gravity) are read from the
    simulation at initialization so nothing needs to be hardcoded in the config.
    Effective inertia uses the point-mass-at-COM approximation ``I = m·L²``,
    which is the dominant term when the COM offset L is large relative to the
    body's own radius of gyration.

    The combined kinetic + potential energy is zero only when the pole is
    perfectly upright and motionless. Damping alone cannot drive energy to zero
    quickly — the policy must apply active counter-torques, which is exactly
    the behaviour we want to incentivise. Use with a negative weight.

    Args:
        cfg: Reward term configuration. Must contain ``asset_cfg`` (with
            ``joint_ids`` for [pitch, roll] joints) and ``pole_body_cfg``
            (with ``body_names`` for the hanging pole link) in ``params``.
        env: The manager-based RL environment instance.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        pole_body_cfg: SceneEntityCfg = cfg.params["pole_body_cfg"]

        # Resolve both configs so that joint_ids and body_ids are int lists.
        asset_cfg.resolve(env.scene)
        pole_body_cfg.resolve(env.scene)

        joint_ids = asset_cfg.joint_ids
        body_ids = pole_body_cfg.body_ids
        assert isinstance(joint_ids, list) and len(joint_ids) >= 2, (
            "asset_cfg must specify exactly 2 joints by name: [pitch, roll]"
        )
        assert isinstance(body_ids, list) and len(body_ids) >= 1, (
            "pole_body_cfg must specify the pole body by name"
        )

        # Cache integer indices so __call__ avoids re-indexing a list|slice union.
        self._pitch_id: int = joint_ids[0]
        self._roll_id: int = joint_ids[1]

        asset: Articulation = env.scene[asset_cfg.name]
        pole_body_id: int = body_ids[0]

        # Mass per env [kg], shape (num_envs,). Cloned so the cached tensor is
        # independent of the underlying PhysX buffer (re-fetched each access).
        mass = asset.data.body_mass.torch[:, pole_body_id].clone()

        # COM offset in the body link frame [m], shape (num_envs, 3).
        # The link frame origin is at the joint, so ‖offset‖ is pendulum length L.
        com_all = asset.data.body_com_pose_b.torch  # (num_envs, num_bodies, 7)
        length = torch.norm(com_all[:, pole_body_id, :3], dim=-1).clone()  # (num_envs,)

        # Gravity magnitude [m/s²] from sim config (z-component magnitude).
        gravity = abs(env.sim.cfg.gravity[2])

        # Pre-compute per-env coefficients to avoid repeated multiplications.
        # Using spherical pendulum energy model
        # kinetic:   ½ · m · L² · (ω_pitch² + ω_roll²)
        # potential: m · g · L  · (1 − cos √(θ_pitch² + θ_roll²))
        self._kinetic_coeff = 0.5 * mass * length**2    # (num_envs,)
        self._potential_coeff = mass * gravity * length  # (num_envs,)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        pole_body_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]

        joint_pos = asset.data.joint_pos.torch  # (num_envs, num_joints)
        joint_vel = asset.data.joint_vel.torch  # (num_envs, num_joints)
        
        pitch     = joint_pos[:, self._pitch_id]
        roll      = joint_pos[:, self._roll_id]
        pitch_vel = joint_vel[:, self._pitch_id]
        roll_vel  = joint_vel[:, self._roll_id]

        vel_sq  = pitch_vel**2 + roll_vel**2
        tilt_sq = pitch**2 + roll**2

        kinetic   = self._kinetic_coeff * vel_sq
        potential = self._potential_coeff * (1.0 - torch.cos(torch.sqrt(tilt_sq + 1e-8)))

        return kinetic + potential