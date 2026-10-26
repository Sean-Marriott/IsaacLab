# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the drone environments."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg

from .curriculums import get_obstacle_curriculum_term

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
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

    current_position = asset.data.body_pos_w.torch[:, asset_cfg.body_ids[0]] - env.scene.env_origins  # type: ignore

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

    # compute the error
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

    current_position = asset.data.root_pos_w.torch - env.scene.env_origins
    direction_to_goal = command[:, :3] - current_position
    direction_to_goal = direction_to_goal / (torch.norm(direction_to_goal, dim=1, keepdim=True) + 1e-8)
    # compute the reward as the dot product between the velocity and the direction to the goal
    velocity_towards_goal = torch.sum(asset.data.root_lin_vel_w.torch * direction_to_goal, dim=1)

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


def joint_pos_target_l2(
    env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
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


def upright_while_moving(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    # pole error (squared deviation from upright)
    pitch = asset.data.joint_pos.torch[:, asset_cfg.joint_ids[0]]
    roll = asset.data.joint_pos.torch[:, asset_cfg.joint_ids[1]]
    pole_error_sq = pitch**2 + roll**2

    # base speed (horizontal plane)
    base_speed = torch.norm(asset.data.root_lin_vel_w.torch[:, :2], dim=-1)

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
        assert isinstance(body_ids, list) and len(body_ids) >= 1, "pole_body_cfg must specify the pole body by name"

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
        self._kinetic_coeff = 0.5 * mass * length**2  # (num_envs,)
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

        pitch = joint_pos[:, self._pitch_id]
        roll = joint_pos[:, self._roll_id]
        pitch_vel = joint_vel[:, self._pitch_id]
        roll_vel = joint_vel[:, self._roll_id]

        vel_sq = pitch_vel**2 + roll_vel**2
        tilt_sq = pitch**2 + roll**2

        kinetic = self._kinetic_coeff * vel_sq
        potential = self._potential_coeff * (1.0 - torch.cos(torch.sqrt(tilt_sq + 1e-8)))

        return kinetic + potential


class PayloadSwingEnergy(ManagerTermBase):
    """Swing energy of a hanging payload, measured in the world frame [J].

    The payload is treated as a point mass ``m`` at its centre of mass, a distance ``L`` from the
    pivot it hangs from. With ``theta`` the angle between the pivot-to-COM vector and gravity, and
    ``v_rel`` the velocity of the COM relative to the pivot point,

    ``E = 1/2 * m * |v_rel|^2 + m * g * L * (1 - cos(theta))``,

    which is zero only when the payload hangs straight down and is not moving relative to the pivot.
    Nothing here depends on the joints, so it holds for any suspension, and ``m`` and ``L`` come from
    every body of the payload rather than from one body's own COM offset as in :class:`PoleEnergy`.

    Because the reference is gravity, not the airframe, the quasi-static lean of the payload while
    the vehicle accelerates counts as swing too: a sustained horizontal acceleration ``a`` holds the
    payload at ``tan(theta) = a / g``, which costs ``m * g * L * (1 - cos(theta))`` for as long as it
    lasts. This term therefore penalizes hard acceleration as well as free swinging.

    The payload mass is read on the first call rather than at construction, so that it includes the
    startup randomization, which runs after the managers are created.

    The ``params`` of the term configuration are:

    * ``payload_cfg``: SceneEntityCfg with every body below the pivot as ``body_names``.
    * ``pivot_cfg``: SceneEntityCfg with the body whose link frame origin is the pivot.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        payload_cfg: SceneEntityCfg = cfg.params["payload_cfg"]
        pivot_cfg: SceneEntityCfg = cfg.params["pivot_cfg"]
        for entity_cfg in (payload_cfg, pivot_cfg):
            entity_cfg.resolve(env.scene)

        self._asset: Articulation = env.scene[payload_cfg.name]
        self._payload_ids = payload_cfg.body_ids
        self._pivot_id: int = pivot_cfg.body_ids[0]  # type: ignore[index]
        gravity = torch.tensor(env.sim.cfg.gravity, device=env.device)
        self._gravity = torch.norm(gravity).item()
        self._down_w = gravity / self._gravity
        self._mass: torch.Tensor | None = None  # per-body payload mass [kg], shape (num_envs, num_payload_bodies)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        payload_cfg: SceneEntityCfg,
        pivot_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the swing energy.

        Returns:
            A 1-D tensor of shape (num_envs,) with the swing energy [J].
        """
        energy, _, _ = self.swing_state(env)
        return energy

    def swing_state(self, env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the swing energy together with the pendulum quantities it is measured against.

        Returns:
            A tuple ``(energy, total_mass, length)``: the swing energy [J], the payload mass [kg] and
            the pivot-to-COM distance [m], each of shape (num_envs,).
        """
        data = self._asset.data
        if self._mass is None:
            self._mass = data.body_mass.torch[:, self._payload_ids].to(env.device)
        weights = (self._mass / self._mass.sum(dim=1, keepdim=True)).unsqueeze(-1)
        total_mass = self._mass.sum(dim=1)

        # Payload COM state relative to the pivot point, in the world frame.
        com_w = torch.sum(weights * data.body_com_pos_w.torch[:, self._payload_ids], dim=1)
        com_vel_w = torch.sum(weights * data.body_com_lin_vel_w.torch[:, self._payload_ids], dim=1)
        offset = com_w - data.body_link_pos_w.torch[:, self._pivot_id]
        vel_rel = com_vel_w - data.body_link_lin_vel_w.torch[:, self._pivot_id]

        length = torch.norm(offset, dim=-1)
        cos_tilt = torch.sum(offset * self._down_w, dim=-1) / length.clamp_min(1e-6)

        kinetic = 0.5 * total_mass * torch.sum(vel_rel**2, dim=-1)
        potential = total_mass * self._gravity * length * (1.0 - cos_tilt)
        return kinetic + potential, total_mass, length

    def swing_amplitude(self, env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
        """Amplitude of the swing, as the angle from vertical at which the payload would hold all of
        its swing energy as potential energy.

        Unlike the instantaneous angle, this is conserved over a swing cycle, so it does not pulse
        twice per period: ``theta_amp = acos(1 - E / (m * g * L))``.

        Returns:
            A tuple ``(theta_amp, length)``: the amplitude [rad], in [0, pi], and the pivot-to-COM
            distance [m], each of shape (num_envs,).
        """
        energy, total_mass, length = self.swing_state(env)
        potential_scale = (total_mass * self._gravity * length).clamp_min(1e-6)
        return torch.acos((1.0 - energy / potential_scale).clamp(-1.0, 1.0)), length


class PayloadSwingEnergyExp(PayloadSwingEnergy):
    """Exponential kernel on :class:`PayloadSwingEnergy`: ``exp(-E / energy_scale)``, in ``[0, 1]``.

    The swing energy grows with the square of the swing amplitude, so this is a Gaussian in the
    amplitude: on a payload with ``m * g * L`` of about 22.4 J (the Mk3 chainsaw), an ``energy_scale`` of 0.3 J falls to
    ``1/e`` at a swing of roughly 9.5 deg.

    An optional deadzone makes the kernel ``exp(-max(E - energy_deadzone, 0) / energy_scale)``: any swing
    with less energy than the deadzone scores the full reward, so the policy gains nothing by chasing
    a small residual swing, which it otherwise does by rocking the airframe back and forth. Since the
    swing energy is conserved over a cycle, the deadzone is a bound on the amplitude: on the Mk3
    chainsaw, 0.03 J is a swing of about 3 deg. Outside the deadzone the kernel is the same curve,
    shifted by the deadzone, so the reward is continuous at its edge.

    Takes the same ``params`` as :class:`PayloadSwingEnergy`, plus:

    * ``energy_scale``: Swing energy at which the kernel falls to ``1/e``, measured from the edge
      of the deadzone [J].
    * ``energy_deadzone``: Swing energy below which the reward is 1 [J]. Defaults to 0, no deadzone.
    * ``window_s``: If given, the reward is zero except during the last ``window_s`` seconds of the
      trajectory reference, as in :class:`ToolEndTargetExp` [s]. If ``None``, the default, it is
      active over the whole episode.
    * ``command_name``: Name of the trajectory command term whose clock sets the window. Only used
      when ``window_s`` is given.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        payload_cfg: SceneEntityCfg,
        pivot_cfg: SceneEntityCfg,
        energy_scale: float = 0.3,
        energy_deadzone: float = 0.0,
        window_s: float | None = None,
        command_name: str = "trajectory",
    ) -> torch.Tensor:
        """Compute the reward.

        Returns:
            A 1-D tensor of shape (num_envs,) with values in [0, 1].
        """
        energy = super().__call__(env, payload_cfg, pivot_cfg)
        reward = torch.exp(-(energy - energy_deadzone).clamp_min(0.0) / energy_scale)
        if window_s is not None:
            cmd = env.command_manager.get_term(command_name)
            reward = reward * (cmd.ref_time >= cmd.episode_length_s - window_s)
        return reward


"""
Trajectory tracking rewards.
"""


def position_tracking_exp(env: ManagerBasedRLEnv, std: float = 1.0, command_name: str = "trajectory") -> torch.Tensor:
    """Reward matching the reference position using an exponential kernel.

    Unlike :func:`distance_to_goal_exp`, which compares a named body with any position command,
    this compares the frame tracked by a :class:`DroneTrajectoryCommand` (see its
    ``tracked_frame_state``), so it follows a tool frame when the command tracks one.

    Args:
        env: The manager-based RL environment instance.
        std: Standard deviation used in the exponential kernel [m]; larger values produce a
            gentler falloff.
        command_name: Name of the trajectory command to read the reference position from.

    Returns:
        A 1-D tensor of shape (num_envs,) with values in (0, 1], where 1 indicates an exact match.
    """
    command_term = env.command_manager.get_term(command_name)
    pos_w, _, _ = command_term.tracked_frame_state()

    position_error_square = torch.sum(torch.square(command_term.command[:, :3] - pos_w), dim=1)
    return torch.exp(-position_error_square / std**2)


def velocity_tracking_exp(env: ManagerBasedRLEnv, std: float = 1.0, command_name: str = "trajectory") -> torch.Tensor:
    """Reward matching the reference velocity using an exponential kernel.

    Position tracking alone is not enough on a closed reference path: sitting near the centroid and
    letting the reference orbit around keeps the mean position error small. Rewarding the reference
    velocity makes that solution strictly worse, so the drone actually flies the path.

    The velocity is that of the frame tracked by the :class:`DroneTrajectoryCommand`.

    Args:
        env: The manager-based RL environment instance.
        std: Standard deviation used in the exponential kernel [m/s]; larger values produce a
            gentler falloff.
        command_name: Name of the trajectory command to read the reference velocity from.

    Returns:
        A 1-D tensor of shape (num_envs,) with values in (0, 1], where 1 indicates an exact match.
    """
    command_term = env.command_manager.get_term(command_name)
    _, _, lin_vel_w = command_term.tracked_frame_state()

    velocity_error_square = torch.sum(torch.square(command_term.command[:, 3:6] - lin_vel_w), dim=1)
    return torch.exp(-velocity_error_square / std**2)


def yaw_tracking_exp(env: ManagerBasedRLEnv, std: float = 0.5, command_name: str = "trajectory") -> torch.Tensor:
    """Reward alignment of the tracked yaw with the reference yaw using an exponential kernel.

    The yaw is that of the frame tracked by the :class:`DroneTrajectoryCommand`. This differs from
    :func:`yaw_aligned`, which always targets a yaw of zero.

    Args:
        env: The manager-based RL environment instance.
        std: Standard deviation used in the exponential kernel [rad]; smaller values make the
            reward more sensitive to yaw deviations.
        command_name: Name of the trajectory command to read the reference yaw from.

    Returns:
        A 1-D tensor of shape (num_envs,) with values in (0, 1], where 1 indicates perfect
        alignment.
    """
    command_term = env.command_manager.get_term(command_name)
    _, quat_w, _ = command_term.tracked_frame_state()

    _, _, yaw = math_utils.euler_xyz_from_quat(quat_w)
    yaw_error = math_utils.wrap_to_pi(command_term.command[:, 9] - yaw)
    return torch.exp(-(yaw_error**2) / std**2)


class ToolEndTargetExp(ManagerTermBase):
    """Reward a hanging tool for reaching its target at the end of the trajectory.

    The target is where the tool point would be if the vehicle sat level at the trajectory end point
    with the payload hanging at rest: the end position plus ``hang_offset`` rotated by the end yaw
    only. The reward is an exponential kernel on the distance
    from the tool point to the target, and is zero except during the last ``window_s`` seconds of
    the episode.

    With ``std_yaw`` set, the tool yaw has to match the end yaw as well, through a joint kernel:

    ``exp(-|p_err|^2 / std^2 - yaw_err^2 / std_yaw^2)``

    Only the yaw is compared, not the full orientation: the tilt of a hanging tool is set by gravity
    and the swing, so the vehicle cannot hold it, whereas the yaw follows the vehicle heading.

    The ``params`` of the term configuration are:

    When the simulation is rendering and ``debug_vis`` is set, the tool and its target are drawn on
    every rendered frame, so their alignment can be inspected over the whole episode. Without
    ``tool_rot`` they are drawn as spheres, the tool point orange and the target cyan. With it they
    are drawn as coordinate frames (x red, y green, z blue): the larger frame is the target, at the
    end yaw, and the smaller one the tool.

    * ``asset_cfg``: SceneEntityCfg identifying the tool body. Must have ``body_names`` configured.
    * ``std``: Standard deviation used in the exponential kernel [m].
    * ``window_s``: Length of the window before the episode end in which the reward is active [s].
    * ``tool_offset``: Tool point in the tool body frame [m].
    * ``hang_offset``: Tool point relative to the tracked body when the payload hangs at rest below a
      level vehicle, in the yaw-aligned vehicle frame [m].
    * ``tool_rot``: Orientation of the tool frame in the tool body frame, as a quaternion
      (x, y, z, w), chosen so that the tool frame lines up with the vehicle yaw frame at rest. It
      sets the tool yaw compared by ``std_yaw`` and draws the markers as frames. Defaults to None
      (spheres).
    * ``std_yaw``: Standard deviation of the kernel on the tool yaw error [rad]. Requires
      ``tool_rot``. Defaults to None, which rewards the position only.
    * ``command_name``: Name of the :class:`DroneTrajectoryCommand` to read the end pose from.
    * ``debug_vis``: Whether to draw the tool and target markers. Defaults to True.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset_cfg.resolve(env.scene)
        self._asset: RigidObject = env.scene[asset_cfg.name]
        self._body_id: int = asset_cfg.body_ids[0]  # type: ignore[index]
        self._command_term = env.command_manager.get_term(cfg.params.get("command_name", "trajectory"))

        tool_offset = torch.tensor(cfg.params["tool_offset"], device=env.device)
        hang_offset = torch.tensor(cfg.params["hang_offset"], device=env.device)
        self._tool_b = tool_offset.expand(env.num_envs, 3)
        self._target_v = hang_offset.expand(env.num_envs, 3)
        tool_rot = cfg.params.get("tool_rot")
        self._tool_rot_b = (
            None if tool_rot is None else torch.tensor(tool_rot, device=env.device).expand(env.num_envs, 4)
        )
        if cfg.params.get("std_yaw") is not None and tool_rot is None:
            raise ValueError("ToolEndTargetExp: 'std_yaw' needs 'tool_rot' to define the tool yaw.")

        self._tool_visualizer = None
        self._target_visualizer = None
        self._debug_vis_handle = None
        if cfg.params.get("debug_vis", True) and env.sim.is_rendering:
            # note: imported here because the marker class pulls in USD, and this module is imported
            # by task configs that are parsed before the simulation app is launched.
            from isaaclab.markers import VisualizationMarkers

            if self._tool_rot_b is None:
                tool_marker_cfg = _sphere_marker_cfg("/Visuals/Reward/tool_point", radius=0.02, color=(1.0, 0.35, 0.0))
                target_marker_cfg = _sphere_marker_cfg(
                    "/Visuals/Reward/tool_target", radius=0.02, color=(0.0, 0.9, 0.9)
                )
            else:
                from isaaclab.markers.config import FRAME_MARKER_CFG

                # Same sizes as the DroneEndPoseExp frames.
                tool_marker_cfg = _frame_marker_cfg(FRAME_MARKER_CFG, "/Visuals/Reward/tool_pose", scale=0.3)
                target_marker_cfg = _frame_marker_cfg(FRAME_MARKER_CFG, "/Visuals/Reward/tool_target_pose", scale=0.6)
            self._tool_visualizer = VisualizationMarkers(tool_marker_cfg)
            self._target_visualizer = VisualizationMarkers(target_marker_cfg)
            # Drawn from the post-update event, like the command debug markers, rather than from
            # :meth:`__call__`: rewards are computed after the step has been rendered, so markers set
            # there would trail the drawn robot by a full policy step.
            import omni.kit.app

            self._debug_vis_handle = (
                omni.kit.app.get_app_interface()
                .get_post_update_event_stream()
                .create_subscription_to_pop(lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event))
            )

    def __del__(self):
        if self._debug_vis_handle is not None:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    def _tool_and_target(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tool pose and its end target pose, in the environment-local world frame.

        Returns:
            A tuple ``(tool_pos, tool_quat, target_pos, target_quat)``. Positions are in [m], shape
            (num_envs, 3); orientations are quaternions (x, y, z, w), shape (num_envs, 4). The tool
            orientation is that of the tool body, rotated by ``tool_rot`` when it is set; the target
            orientation is the end yaw.
        """
        body_pos = self._asset.data.body_pos_w.torch[:, self._body_id] - self._env.scene.env_origins
        body_quat = self._asset.data.body_quat_w.torch[:, self._body_id]
        tool_pos = body_pos + math_utils.quat_apply(body_quat, self._tool_b)
        tool_quat = body_quat if self._tool_rot_b is None else math_utils.quat_mul(body_quat, self._tool_rot_b)

        cmd = self._command_term
        zeros = torch.zeros_like(cmd.yaw_end)
        end_yaw_quat = math_utils.quat_from_euler_xyz(zeros, zeros, cmd.yaw_end)
        target = cmd.pos_end_w + math_utils.quat_apply(end_yaw_quat, self._target_v)
        return tool_pos, tool_quat, target, end_yaw_quat

    def _debug_vis_callback(self, event):
        # note: needed in case the robot is de-initialized, since its data cannot be accessed then
        if not self._asset.is_initialized:
            return
        tool_pos, tool_quat, target, target_quat = self._tool_and_target()
        origins = self._env.scene.env_origins
        if self._tool_rot_b is None:
            self._tool_visualizer.visualize(tool_pos + origins)
            self._target_visualizer.visualize(target + origins)
        else:
            self._tool_visualizer.visualize(tool_pos + origins, tool_quat)
            self._target_visualizer.visualize(target + origins, target_quat)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        std: float,
        window_s: float,
        tool_offset: tuple[float, float, float],
        hang_offset: tuple[float, float, float],
        tool_rot: tuple[float, float, float, float] | None = None,
        std_yaw: float | None = None,
        command_name: str = "trajectory",
        debug_vis: bool = True,
    ) -> torch.Tensor:
        """Compute the reward.

        Returns:
            A 1-D tensor of shape (num_envs,) with values in [0, 1].
        """
        tool_pos, tool_quat, target, _ = self._tool_and_target()
        cmd = self._command_term
        exponent = torch.sum(torch.square(tool_pos - target), dim=1) / std**2
        if std_yaw is not None:
            _, _, tool_yaw = math_utils.euler_xyz_from_quat(tool_quat)
            yaw_error = math_utils.wrap_to_pi(tool_yaw - cmd.yaw_end)
            exponent = exponent + torch.square(yaw_error) / std_yaw**2
        active = cmd.ref_time >= cmd.episode_length_s - window_s
        return torch.exp(-exponent) * active


class DroneEndPoseExp(ManagerTermBase):
    """Reward the vehicle for matching a target pose at the end of the trajectory.

    The target pose is the trajectory end position with the airframe level and at the end yaw, i.e.
    hovering at rest at the end point. The reward is a joint exponential kernel on the position
    error and the orientation error (the angle of the rotation between the two poses), so both
    have to match for it to be high:

    ``exp(-|p_err|^2 / std_pos^2 - angle_err^2 / std_rot^2)``

    It is zero except during the last ``window_s`` seconds of the episode.

    With ``radius`` set, the position error is measured from the surface of a ball of that radius
    around the target rather than from its centre, ``|p_err| -> max(|p_err| - radius, 0)``, so the
    vehicle is free to move anywhere inside the ball. With ``yaw_only`` set, the orientation error
    is the yaw error alone: roll and pitch are how a multirotor accelerates, so a term that has to
    allow the vehicle to manoeuvre about the target must not penalize them.

    When the simulation is rendering and ``debug_vis`` is set, the target pose and the current pose
    of the body are drawn as coordinate frames (x red, y green, z blue) on every rendered frame: the
    larger frame is the target, the smaller one the vehicle. The target is drawn for the whole
    episode, so the approach to it can be followed.

    The ``params`` of the term configuration are:

    * ``asset_cfg``: SceneEntityCfg identifying the vehicle body. Must have ``body_names`` configured.
    * ``std_pos``: Standard deviation of the kernel on the position error [m].
    * ``std_rot``: Standard deviation of the kernel on the orientation error [rad].
    * ``window_s``: Length of the window before the episode end in which the reward is active [s].
    * ``command_name``: Name of the :class:`DroneTrajectoryCommand` to read the end pose from.
    * ``radius``: Radius of the ball around the target inside which the position error is zero [m].
      Defaults to 0, the target point itself.
    * ``yaw_only``: Whether to compare the yaw alone rather than the full orientation. Defaults to
      False.
    * ``debug_vis``: Whether to draw the pose markers. Defaults to True.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset_cfg.resolve(env.scene)
        self._asset: RigidObject = env.scene[asset_cfg.name]
        self._body_id: int = asset_cfg.body_ids[0]  # type: ignore[index]
        self._command_term = env.command_manager.get_term(cfg.params.get("command_name", "trajectory"))

        self._target_visualizer = None
        self._current_visualizer = None
        self._debug_vis_handle = None
        if cfg.params.get("debug_vis", True) and env.sim.is_rendering:
            # note: imported here for the same reason as in ToolEndTargetExp.
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import FRAME_MARKER_CFG

            self._target_visualizer = VisualizationMarkers(
                _frame_marker_cfg(FRAME_MARKER_CFG, "/Visuals/Reward/drone_end_pose", scale=0.6)
            )
            self._current_visualizer = VisualizationMarkers(
                _frame_marker_cfg(FRAME_MARKER_CFG, "/Visuals/Reward/drone_pose", scale=0.3)
            )
            import omni.kit.app

            self._debug_vis_handle = (
                omni.kit.app.get_app_interface()
                .get_post_update_event_stream()
                .create_subscription_to_pop(lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event))
            )

    def __del__(self):
        if self._debug_vis_handle is not None:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    def _target_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Target position [m] in the environment-local world frame and orientation (x, y, z, w).

        Shapes are (num_envs, 3) and (num_envs, 4).
        """
        cmd = self._command_term
        zeros = torch.zeros_like(cmd.yaw_end)
        return cmd.pos_end_w, math_utils.quat_from_euler_xyz(zeros, zeros, cmd.yaw_end)

    def _debug_vis_callback(self, event):
        # note: needed in case the robot is de-initialized, since its data cannot be accessed then
        if not self._asset.is_initialized:
            return
        target_pos, target_quat = self._target_pose()
        self._target_visualizer.visualize(target_pos + self._env.scene.env_origins, target_quat)
        self._current_visualizer.visualize(
            self._asset.data.body_pos_w.torch[:, self._body_id], self._asset.data.body_quat_w.torch[:, self._body_id]
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        std_pos: float,
        std_rot: float,
        window_s: float,
        command_name: str = "trajectory",
        radius: float = 0.0,
        yaw_only: bool = False,
        debug_vis: bool = True,
    ) -> torch.Tensor:
        """Compute the reward.

        Returns:
            A 1-D tensor of shape (num_envs,) with values in [0, 1].
        """
        return self._reward(env, std_pos, std_rot, window_s, radius, yaw_only)

    def _reward(
        self,
        env: ManagerBasedRLEnv,
        std_pos: float,
        std_rot: float,
        window_s: float,
        radius: float | torch.Tensor,
        yaw_only: bool,
    ) -> torch.Tensor:
        """Kernel shared by this term and its subclasses, with ``radius`` either a constant [m] or a
        per-environment tensor of shape (num_envs,) [m]."""
        target_pos, target_quat = self._target_pose()
        body_pos = self._asset.data.body_pos_w.torch[:, self._body_id] - env.scene.env_origins
        body_quat = self._asset.data.body_quat_w.torch[:, self._body_id]

        pos_error = torch.norm(body_pos - target_pos, dim=1)
        pos_error_sq = torch.square((pos_error - radius).clamp_min(0.0))
        if yaw_only:
            _, _, yaw = math_utils.euler_xyz_from_quat(body_quat)
            rot_error = math_utils.wrap_to_pi(yaw - self._command_term.yaw_end)
        else:
            rot_error = math_utils.quat_error_magnitude(body_quat, target_quat)

        cmd = self._command_term
        active = cmd.ref_time >= cmd.episode_length_s - window_s
        return torch.exp(-pos_error_sq / std_pos**2 - torch.square(rot_error) / std_rot**2) * active


class SwingRadiusEndPoseExp(DroneEndPoseExp):
    """:class:`DroneEndPoseExp` with a hold radius that grows with the payload swing.

    Damping a hanging payload means moving its pivot: the vehicle has to accelerate in the direction
    the payload swings, which takes it some way off the target and tilts it. A term that rewards the
    exact target point forbids exactly that, so the vehicle is given a ball around the target to
    manoeuvre in, sized by how much swing there is to remove:

    ``radius = clamp(radius_min + radius_gain * L * theta_amp, max=radius_max)``

    where ``L * theta_amp`` is the arc of the swing (see :meth:`PayloadSwingEnergy.swing_amplitude`).
    Once the swing has been damped the ball shrinks back to ``radius_min``, so the vehicle still has
    to hold the target closely while the payload hangs still, rather than settle anywhere in a fixed
    ball. The pivot excursion that damps a swing of amplitude ``theta`` in a time ``tau`` is about
    ``2 * L * theta / (omega * tau)``; on the Mk3 chainsaw (``omega`` ~3 rad/s) a ``radius_gain`` of
    0.75 leaves room to damp it in about a second.

    The orientation is compared on the yaw alone by default, since the tilt is how the vehicle
    accelerates.

    Takes the same ``params`` as :class:`DroneEndPoseExp`, except ``radius``, plus those of
    :class:`PayloadSwingEnergy`:

    * ``payload_cfg``: SceneEntityCfg with every body below the pivot as ``body_names``.
    * ``pivot_cfg``: SceneEntityCfg with the body whose link frame origin is the pivot.
    * ``radius_min``: Hold radius with the payload hanging still [m].
    * ``radius_gain``: Hold radius per metre of swing arc [m/m].
    * ``radius_max``: Upper bound on the hold radius [m].
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._swing = PayloadSwingEnergy(cfg, env)

    def hold_radius(
        self, env: ManagerBasedRLEnv, radius_min: float, radius_gain: float, radius_max: float
    ) -> torch.Tensor:
        """Hold radius for the current swing [m], shape (num_envs,)."""
        theta_amp, length = self._swing.swing_amplitude(env)
        return (radius_min + radius_gain * length * theta_amp).clamp(max=radius_max)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        payload_cfg: SceneEntityCfg,
        pivot_cfg: SceneEntityCfg,
        std_pos: float,
        std_rot: float,
        window_s: float,
        radius_min: float = 0.05,
        radius_gain: float = 0.75,
        radius_max: float = 0.4,
        command_name: str = "trajectory",
        yaw_only: bool = True,
        debug_vis: bool = True,
    ) -> torch.Tensor:
        """Compute the reward.

        Returns:
            A 1-D tensor of shape (num_envs,) with values in [0, 1].
        """
        radius = self.hold_radius(env, radius_min, radius_gain, radius_max)
        return self._reward(env, std_pos, std_rot, window_s, radius, yaw_only)


def _frame_marker_cfg(base_cfg: VisualizationMarkersCfg, prim_path: str, scale: float) -> VisualizationMarkersCfg:
    """Build a coordinate-frame marker configuration at the given scale, without the connecting line."""
    # note: ``copy`` is deep, so changing the scale does not alias the shared FRAME_MARKER_CFG.
    cfg = base_cfg.copy()
    cfg.prim_path = prim_path
    cfg.markers = {"frame": cfg.markers["frame"]}
    cfg.markers["frame"].scale = (scale, scale, scale)
    return cfg


def _sphere_marker_cfg(prim_path: str, radius: float, color: tuple[float, float, float]) -> VisualizationMarkersCfg:
    """Build a standalone single-sphere marker configuration."""
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)
            )
        },
    )
