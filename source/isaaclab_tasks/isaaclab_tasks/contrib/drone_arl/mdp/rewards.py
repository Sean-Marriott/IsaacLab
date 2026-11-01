# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the drone environments."""

from __future__ import annotations

import math
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg

from .curriculums import get_obstacle_curriculum_term
from .utils import hanging_tool_and_end_target

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
    lasts. This term therefore penalizes hard acceleration as well as free swinging, unless the swing is
    measured against a known acceleration (see :meth:`swing_state`).

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

    def swing_state(
        self, env: ManagerBasedRLEnv, accel_w: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the swing energy together with the pendulum quantities it is measured against.

        Args:
            env: The manager-based RL environment instance.
            accel_w: Acceleration of the pivot to measure the swing against, in the world frame
                [m/s^2], shape (num_envs, 3). The payload then hangs at rest along the effective gravity
                ``g - a`` rather than along gravity, so the lean that this acceleration holds it at is
                not counted as swing; the potential term uses ``|g - a|`` in place of ``g``. None
                measures against gravity alone.

        Returns:
            A tuple ``(energy, total_mass, length)``: the swing energy [J], the payload mass [kg] and
            the pivot-to-COM distance [m], each of shape (num_envs,). With :paramref:`accel_w` set,
            the energy is that of the swing about the leaning rest pose.
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
        if accel_w is None:
            gravity, down_w = self._gravity, self._down_w
        else:
            gravity_eff_w = self._gravity * self._down_w - accel_w
            gravity = torch.norm(gravity_eff_w, dim=-1).clamp_min(1e-6)
            down_w = gravity_eff_w / gravity.unsqueeze(-1)
        cos_tilt = torch.sum(offset * down_w, dim=-1) / length.clamp_min(1e-6)

        kinetic = 0.5 * total_mass * torch.sum(vel_rel**2, dim=-1)
        potential = total_mass * gravity * length * (1.0 - cos_tilt)
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
    * ``compensate_reference_accel``: If True, the swing is measured about the lean that the
      trajectory reference's acceleration calls for, ``tan(theta) = a / g``, rather than about the
      vertical (see :meth:`PayloadSwingEnergy.swing_state`). Following a moving reference then costs
      nothing for the lean it needs, only for swing on top of it, so the term can stay active while
      the reference moves. The reference acceleration is used rather than the measured one, so the
      policy cannot hide a swing by accelerating the airframe along it. Defaults to False.
    * ``command_name``: Name of the trajectory command term whose clock sets the window and whose
      reference acceleration is compensated. Only used when ``window_s`` or
      ``compensate_reference_accel`` is set.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        payload_cfg: SceneEntityCfg,
        pivot_cfg: SceneEntityCfg,
        energy_scale: float = 0.3,
        energy_deadzone: float = 0.0,
        window_s: float | None = None,
        compensate_reference_accel: bool = False,
        command_name: str = "trajectory",
    ) -> torch.Tensor:
        """Compute the reward.

        Returns:
            A 1-D tensor of shape (num_envs,) with values in [0, 1].
        """
        cmd = env.command_manager.get_term(command_name) if window_s is not None or compensate_reference_accel else None
        energy, _, _ = self.swing_state(env, cmd.acc_ref_w if compensate_reference_accel else None)
        reward = torch.exp(-(energy - energy_deadzone).clamp_min(0.0) / energy_scale)
        if window_s is not None:
            reward = reward * (cmd.ref_time >= cmd.episode_length_s - window_s)
        return reward


"""
Trajectory tracking rewards.
"""


def position_tracking_exp(
    env: ManagerBasedRLEnv, std: float = 1.0, command_name: str = "trajectory", exclude_end_s: float | None = None
) -> torch.Tensor:
    """Reward matching the reference position using an exponential kernel.

    Unlike :func:`distance_to_goal_exp`, which compares a named body with any position command,
    this compares the frame tracked by a :class:`DroneTrajectoryCommand` (see its
    ``tracked_frame_state``), so it follows a tool frame when the command tracks one.

    Args:
        env: The manager-based RL environment instance.
        std: Standard deviation used in the exponential kernel [m]; larger values produce a
            gentler falloff.
        command_name: Name of the trajectory command to read the reference position from.
        exclude_end_s: If given, the reward is zero during the last ``exclude_end_s`` seconds of the
            reference, such as the hold at its end point, so that another term can place the vehicle
            there [s]. Defaults to None, active over the whole episode.

    Returns:
        A 1-D tensor of shape (num_envs,) with values in [0, 1], where 1 indicates an exact match.
    """
    command_term = env.command_manager.get_term(command_name)
    pos_w, _, _ = command_term.tracked_frame_state()

    position_error_square = torch.sum(torch.square(command_term.command[:, :3] - pos_w), dim=1)
    reward = torch.exp(-position_error_square / std**2)
    if exclude_end_s is not None:
        reward = reward * (command_term.ref_time < command_term.episode_length_s - exclude_end_s)
    return reward


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


class ActionSecondDifferenceL2(ManagerTermBase):
    """Penalize the second difference of the action: ``||a_t - 2 a_{t-1} + a_{t-2}||^2``.

    It is zero for a held or steadily ramping command and largest for one that flips sign every
    policy step: an action alternating with amplitude ``a`` costs ``16 a^2`` per axis, against ``4 a^2``
    from :func:`~isaaclab.envs.mdp.action_rate_l2`, which also charges smooth changes the same. It
    therefore targets the period-2 (half the policy rate) oscillation without resisting tracking.
    The first two steps of an episode, before there is a history, score zero. Use with a negative weight.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        dim = env.action_manager.total_action_dim
        self._prev = torch.zeros(env.num_envs, dim, device=env.device)
        self._prev_prev = torch.zeros(env.num_envs, dim, device=env.device)
        self._history = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._history[env_ids] = 0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        action = env.action_manager.action
        penalty = torch.sum(torch.square(action - 2.0 * self._prev + self._prev_prev), dim=1)
        penalty = torch.where(self._history >= 2, penalty, torch.zeros_like(penalty))
        self._prev_prev[:] = self._prev
        self._prev[:] = action
        self._history += 1
        return penalty


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
        return hanging_tool_and_end_target(
            self._asset,
            self._body_id,
            self._tool_b,
            self._tool_rot_b,
            self._target_v,
            self._command_term,
            self._env.scene.env_origins,
        )

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


class ToolCaptureState(NamedTuple):
    """The capture conditions of :class:`ToolCapture` at one instant. Boolean fields have shape (num_envs,)."""

    in_envelope: torch.Tensor
    """Whether the tool point is in the cone or the ball, however it got there."""
    in_region: torch.Tensor
    """Whether it is in the capture region as ``require_cone_entry`` defines it."""
    captured: torch.Tensor
    """Whether it is captured: in the capture region, slow and square."""
    approached: torch.Tensor
    """The cone-entry flag for this instant: set in the cone, kept in the envelope, cleared outside it."""
    offset_t: torch.Tensor
    """Tool point minus target, in the target's yaw frame [m]. Shape (num_envs, 3)."""
    tool_pos: torch.Tensor
    """Tool point in the environment-local world frame [m]. Shape (num_envs, 3)."""
    tool_quat: torch.Tensor
    """Tool frame orientation, quaternion (x, y, z, w). Shape (num_envs, 4)."""
    target_pos: torch.Tensor
    """Target in the environment-local world frame [m]. Shape (num_envs, 3)."""
    target_quat: torch.Tensor
    """Target yaw orientation, quaternion (x, y, z, w). Shape (num_envs, 4)."""


class ToolCapture(ManagerTermBase):
    """Reward a hanging tool for being captured at its end target: in the envelope, in the approach cone, slow.

    The target is that of :class:`ToolEndTargetExp`: where the tool point would be if the vehicle sat
    level at the trajectory end point with the payload hanging at rest. With ``d`` the tool point
    relative to the target, in the target's yaw frame, the tool is captured when all of these hold:

    * **Envelope and cone**: ``|d| <= tolerance_radius`` (the ball, where any direction counts), or ``d``
      in the cone on the approach side: with ``a`` its distance along ``approach_axis`` and ``l`` its
      distance from that axis, ``0 <= a <= capture_radius`` and ``l <= max(tip_radius, a * tan(cone_half_angle))``.
      With ``tip_radius`` the cone has a blunt tip: near the target it is a cylinder of that radius, as
      wide as the pocket the tool closes on, rather than narrowing to a point that position noise alone
      would leave. It widens into the cone beyond ``a = tip_radius / tan(cone_half_angle)``.
      With ``require_cone_entry``, the ball counts only once the tool has come in through the cone: a
      per-environment flag is set while the tool is in the cone and cleared when it leaves the envelope
      altogether, so a tool that reaches the ball from in front or from the side is not captured, while
      one that came in through the cone can settle anywhere in the ball.
    * **Low relative velocity**: the speed of the tool point relative to the target, which is at rest at
      the end point, is at most ``max_rel_speed``.
    * **Heading** (if ``max_yaw_error`` is set): the tool yaw is within ``max_yaw_error`` of the end yaw.

    The reward is 1 while captured and 0 otherwise, during the last ``window_s`` seconds of the episode.
    A momentary capture can be a swing passing through the target at the end of its arc, so the term also
    logs, per episode, whether the tool stayed captured for ``dwell_s`` without a break
    (``Metrics/tool_capture/success``) and the mean time at which that first happened in the episodes
    that did (``Metrics/tool_capture/time_to_success``) [s].

    The ``params`` of the term configuration are:

    * ``asset_cfg``: SceneEntityCfg identifying the tool body. Must have ``body_names`` configured.
    * ``tool_offset``: Tool point in the tool body frame [m].
    * ``hang_offset``: Tool point relative to the tracked body when the payload hangs at rest below a
      level vehicle, in the yaw-aligned vehicle frame [m].
    * ``tool_rot``: Orientation of the tool frame in the tool body frame, as a quaternion (x, y, z, w),
      chosen so that the tool frame lines up with the vehicle yaw frame at rest. Sets the tool yaw.
    * ``capture_radius``: Length of the envelope along the cone [m].
    * ``tolerance_radius``: Radius of the ball at the target that counts from any direction [m].
    * ``approach_axis``: Direction from the target to the side the tool must arrive from, in the
      target's yaw frame. Normalized internally.
    * ``cone_half_angle``: Half-angle of the approach cone [rad].
    * ``tip_radius``: Smallest radius of the cone, at its tip [m]. Defaults to 0.0, a sharp tip.
    * ``max_rel_speed``: Largest speed of the tool point relative to the target [m/s].
    * ``max_yaw_error``: Largest tool yaw error to the end yaw [rad], or None to ignore the yaw.
    * ``require_cone_entry``: Whether the ball counts only after an approach through the cone. Defaults
      to False, in which case the ball counts however the tool reached it.
    * ``dwell_s``: Time the tool has to stay captured for the episode to count as a success [s].
    * ``window_s``: Length of the window before the episode end in which the reward is active [s].
    * ``command_name``: Name of the :class:`DroneTrajectoryCommand` to read the end pose from.
    * ``debug_vis``: Whether to draw the capture envelope and the tool point, coloured by the capture
      state, when the simulation is rendering. Defaults to True.
    """

    _HEADING_POINTER_LENGTH = 0.1
    """Length of the tool heading pointer drawn by the debug visualization [m]."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset_cfg.resolve(env.scene)
        self._asset: RigidObject = env.scene[asset_cfg.name]
        self._body_id: int = asset_cfg.body_ids[0]  # type: ignore[index]
        self._command_term = env.command_manager.get_term(cfg.params.get("command_name", "trajectory"))

        def _expand(value, size):
            return torch.tensor(value, device=env.device, dtype=torch.float32).expand(env.num_envs, size)

        self._tool_b = _expand(cfg.params["tool_offset"], 3)
        self._hang_v = _expand(cfg.params["hang_offset"], 3)
        self._tool_rot_b = _expand(cfg.params["tool_rot"], 4)
        self._axis_t = torch.nn.functional.normalize(_expand(cfg.params["approach_axis"], 3), dim=-1)

        self._dwell_steps = max(1, round(cfg.params["dwell_s"] / env.step_dt))
        self._streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        # Whether the tool came into the envelope through the cone (see ``require_cone_entry``).
        self._approached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._success_time = torch.zeros(env.num_envs, device=env.device)

        self._debug_vis_handle = None
        if cfg.params.get("debug_vis", True) and env.sim.is_rendering:
            self._init_debug_vis(cfg)

    def _init_debug_vis(self, cfg: RewardTermCfg):
        """Create the capture markers, coloured by the capture state on every rendered frame.

        * A translucent cone with its apex on the target, opening to the approach side, and, with
          ``tip_radius``, a translucent cylinder of that radius along its axis: together the capture
          cone, blunt tip included.
        * A translucent ball at the target: the tolerance ball, captured from any direction.
        * A small sphere on the tool point, with a pointer along the tool heading. The envelope cone's
          axis is the target heading, so the tool is square to the target when the pointer runs along
          the cone from its open end to its tip, the way the tool closes on the target.

        All of them share one colour for the state: red outside the envelope, amber inside it but not
        captured (too fast, not square, or in the ball without having come through the cone), green
        captured, blue captured for the whole dwell.
        """
        # note: imported here because the marker class pulls in USD, and this module is imported by task
        # configs that are parsed before the simulation app is launched.
        import omni.kit.app

        from isaaclab.markers import VisualizationMarkers

        params = cfg.params
        colors = ((0.9, 0.15, 0.1), (1.0, 0.65, 0.0), (0.1, 0.85, 0.2), (0.0, 0.55, 1.0))

        def _markers(prim_path: str, make_shape, opacity: float) -> VisualizationMarkers:
            # One prototype per state, so the state picks the colour through ``marker_indices``.
            markers = {
                f"state_{i}": make_shape(sim_utils.PreviewSurfaceCfg(diffuse_color=c, opacity=opacity))
                for i, c in enumerate(colors)
            }
            return VisualizationMarkers(VisualizationMarkersCfg(prim_path=prim_path, markers=markers))

        # A USD cone is centred on its origin with the apex towards +Z. Point the apex at the target, so it
        # opens along the approach axis, and centre it half its height out along that axis.
        height = params["capture_radius"]
        radius = height * math.tan(params["cone_half_angle"])
        self._cone_visualizer = _markers(
            "/Visuals/Reward/capture_cone",
            lambda mat: sim_utils.ConeCfg(radius=radius, height=height, axis="Z", visual_material=mat),
            opacity=0.25,
        )
        # The blunt tip: the cylinder the cone is at least as wide as, over the same length.
        tip_radius = params.get("tip_radius", 0.0)
        self._tip_visualizer = None
        if tip_radius > 0.0:
            self._tip_visualizer = _markers(
                "/Visuals/Reward/capture_cone_tip",
                lambda mat: sim_utils.CylinderCfg(radius=tip_radius, height=height, axis="Z", visual_material=mat),
                opacity=0.25,
            )
        self._ball_visualizer = _markers(
            "/Visuals/Reward/capture_tolerance",
            lambda mat: sim_utils.SphereCfg(radius=params["tolerance_radius"], visual_material=mat),
            opacity=0.35,
        )
        self._tool_visualizer = _markers(
            "/Visuals/Reward/capture_tool_point",
            lambda mat: sim_utils.SphereCfg(radius=0.02, visual_material=mat),
            opacity=1.0,
        )
        # A slim cone just ahead of the tool point, its tip along the tool heading.
        self._heading_visualizer = _markers(
            "/Visuals/Reward/capture_tool_heading",
            lambda mat: sim_utils.ConeCfg(
                radius=0.015, height=self._HEADING_POINTER_LENGTH, axis="X", visual_material=mat
            ),
            opacity=1.0,
        )
        apex_dir = -self._axis_t[:1]
        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=apex_dir.device)
        rot_axis = torch.cross(z_axis, apex_dir, dim=-1)
        angle = torch.acos(torch.sum(z_axis * apex_dir, dim=-1).clamp(-1.0, 1.0))
        if torch.norm(rot_axis) < 1e-6:
            # The apex direction is along +Z or -Z: any horizontal axis turns it.
            rot_axis = torch.tensor([[1.0, 0.0, 0.0]], device=apex_dir.device)
        rot = math_utils.quat_from_angle_axis(angle, torch.nn.functional.normalize(rot_axis, dim=-1))
        num_envs = self._axis_t.shape[0]
        self._cone_rot_t = rot.expand(num_envs, 4)
        self._cone_center_t = self._axis_t * (height / 2.0)

        # Drawn from the post-update event, like the command debug markers, rather than from
        # :meth:`__call__`: rewards are computed after the step has been rendered, so markers set there
        # would trail the drawn robot by a full policy step.
        self._debug_vis_handle = (
            omni.kit.app.get_app_interface()
            .get_post_update_event_stream()
            .create_subscription_to_pop(lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event))
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        # Called by the reward manager after the environment has cleared its log for this reset, so the
        # episode outcome can be added to it.
        success = self._success[env_ids]
        if success.numel() > 0:
            log = self._env.extras.setdefault("log", {})
            log["Metrics/tool_capture/success"] = success.float().mean().item()
            if success.any():
                log["Metrics/tool_capture/time_to_success"] = self._success_time[env_ids][success].mean().item()
        self._streak[env_ids] = 0
        self._approached[env_ids] = False
        self._success[env_ids] = False
        self._success_time[env_ids] = 0.0

    def __del__(self):
        if self._debug_vis_handle is not None:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    def captured(self) -> torch.Tensor:
        """Whether the tool is captured now, regardless of the reward window.

        Returns:
            A boolean tensor of shape (num_envs,).
        """
        return self.capture_state().captured

    def capture_state(self) -> ToolCaptureState:
        """Evaluate the capture conditions.

        This has no side effects, so the debug visualization and other terms can call it between steps:
        the cone-entry flag it returns for this instant is stored by :meth:`__call__`, once per step.

        Returns:
            The capture state.
        """
        params = self.cfg.params
        tool_pos, tool_quat, target_pos, target_quat = hanging_tool_and_end_target(
            self._asset,
            self._body_id,
            self._tool_b,
            self._tool_rot_b,
            self._hang_v,
            self._command_term,
            self._env.scene.env_origins,
        )
        # -- envelope and cone, in the target's yaw frame
        offset_t = math_utils.quat_apply_inverse(target_quat, tool_pos - target_pos)
        distance = torch.norm(offset_t, dim=-1)
        along = torch.sum(offset_t * self._axis_t, dim=-1)
        lateral = torch.norm(offset_t - along.unsqueeze(-1) * self._axis_t, dim=-1)
        cone_radius = torch.clamp(along * math.tan(params["cone_half_angle"]), min=params.get("tip_radius", 0.0))
        in_cone = (along >= 0.0) & (along <= params["capture_radius"]) & (lateral <= cone_radius)
        in_ball = distance <= params["tolerance_radius"]
        in_envelope = in_cone | in_ball
        # Set while in the cone, kept while still in the envelope, cleared on leaving it.
        approached = (self._approached | in_cone) & in_envelope
        if params.get("require_cone_entry", False):
            in_capture_region = in_cone | (in_ball & approached)
        else:
            in_capture_region = in_envelope

        # -- relative velocity: the target is at rest at the end point, so it is the tool point's own
        body_quat = self._asset.data.body_link_quat_w.torch[:, self._body_id]
        lever_w = math_utils.quat_apply(body_quat, self._tool_b)
        ang_vel_w = self._asset.data.body_link_ang_vel_w.torch[:, self._body_id]
        tool_vel_w = self._asset.data.body_link_lin_vel_w.torch[:, self._body_id] + torch.cross(
            ang_vel_w, lever_w, dim=-1
        )
        slow = torch.norm(tool_vel_w, dim=-1) <= params["max_rel_speed"]

        captured = in_capture_region & slow
        if params.get("max_yaw_error") is not None:
            _, _, tool_yaw = math_utils.euler_xyz_from_quat(tool_quat)
            yaw_error = math_utils.wrap_to_pi(tool_yaw - self._command_term.yaw_end)
            captured &= torch.abs(yaw_error) <= params["max_yaw_error"]
        return ToolCaptureState(
            in_envelope=in_envelope,
            in_region=in_capture_region,
            captured=captured,
            approached=approached,
            offset_t=offset_t,
            tool_pos=tool_pos,
            tool_quat=tool_quat,
            target_pos=target_pos,
            target_quat=target_quat,
        )

    def _debug_vis_callback(self, event):
        # note: needed in case the robot is de-initialized, since its data cannot be accessed then
        if not self._asset.is_initialized:
            return
        cs = self.capture_state()
        in_envelope, captured, tool_pos, tool_quat = cs.in_envelope, cs.captured, cs.tool_pos, cs.tool_quat
        target_pos, target_quat = cs.target_pos, cs.target_quat
        # 0 outside the envelope, 1 inside it but not captured (too fast, not square, or in the ball without
        # having come through the cone), 2 captured, 3 captured for the whole dwell (a success).
        state = in_envelope.long() + captured.long() + (captured & (self._streak >= self._dwell_steps)).long()
        origins = self._env.scene.env_origins
        cone_quat = math_utils.quat_mul(target_quat, self._cone_rot_t)
        cone_pos = target_pos + math_utils.quat_apply(target_quat, self._cone_center_t)
        self._cone_visualizer.visualize(cone_pos + origins, cone_quat, marker_indices=state)
        if self._tip_visualizer is not None:
            # Symmetric about its axis, so the cone's orientation and centre serve.
            self._tip_visualizer.visualize(cone_pos + origins, cone_quat, marker_indices=state)
        self._ball_visualizer.visualize(target_pos + origins, marker_indices=state)
        self._tool_visualizer.visualize(tool_pos + origins, marker_indices=state)
        # The heading pointer shows the tool yaw only, which is what the capture compares. It starts at the
        # tool point, so it is centred half its length ahead.
        heading_quat = math_utils.yaw_quat(tool_quat)
        pointer_offset = torch.zeros_like(tool_pos)
        pointer_offset[:, 0] = self._HEADING_POINTER_LENGTH / 2.0 + 0.02
        pointer_pos = tool_pos + math_utils.quat_apply(heading_quat, pointer_offset)
        self._heading_visualizer.visualize(pointer_pos + origins, heading_quat, marker_indices=state)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        tool_offset: tuple[float, float, float],
        hang_offset: tuple[float, float, float],
        tool_rot: tuple[float, float, float, float],
        capture_radius: float,
        tolerance_radius: float,
        approach_axis: tuple[float, float, float],
        cone_half_angle: float,
        max_rel_speed: float,
        max_yaw_error: float | None,
        dwell_s: float,
        window_s: float,
        require_cone_entry: bool = False,
        tip_radius: float = 0.0,
        command_name: str = "trajectory",
        debug_vis: bool = True,
    ) -> torch.Tensor:
        """Compute the reward.

        Returns:
            A 1-D tensor of shape (num_envs,) with values in {0, 1}.
        """
        cmd = self._command_term
        active = cmd.ref_time >= cmd.episode_length_s - window_s
        cs = self.capture_state()
        self._approached = cs.approached
        captured = cs.captured & active

        self._streak = torch.where(captured, self._streak + 1, torch.zeros_like(self._streak))
        newly = (self._streak >= self._dwell_steps) & ~self._success
        self._success_time = torch.where(newly, cmd.ref_time, self._success_time)
        self._success |= newly
        return captured.float()


class ToolCaptureProgress(ManagerTermBase):
    """Reward the tool for making progress towards capture by a :class:`ToolCapture` term.

    The reward is the change in a potential per step, ``phi(s_t) - phi(s_{t-1})``, so it pays for moving
    towards capture from wherever the tool is, including after a miss, and sums over an episode to the
    potential at its end minus that at its start. The first step after a reset scores zero.

    With ``route_aware``, the potential is the negative length of the route through the approach cone:

    * in the capture region (the cone, and the ball as the capture term's ``require_cone_entry``
      defines it): ``-|d|``, the distance to the target;
    * elsewhere: ``-(|d - s| + staging_distance)``, the distance to a staging point ``s`` on the cone
      axis ``staging_distance`` behind the target, plus the way from there down the cone.

    So outside the cone, progress means heading for the staging point, not the target: after arriving
    from the wrong side, backing out and coming round scores as progress rather than as a retreat. The
    two pieces agree at the staging point, so a clean approach through it is smooth; entering the region
    elsewhere scores a one-off gain, and leaving it the matching loss. Without ``route_aware``, the
    potential is ``-|d|`` everywhere.

    The route is a straight line to the staging point, which from in front passes through the target: fine
    for a point target, not for a physical one, which would need a route around it.

    The ``params`` of the term configuration are:

    * ``capture_term_name``: Name of the :class:`ToolCapture` reward term whose target, cone and capture
      region this follows.
    * ``route_aware``: Whether the potential follows the route through the cone. Defaults to True.
    * ``staging_distance``: Distance of the staging point behind the target along the cone axis [m], or
      None for the capture term's ``capture_radius``, the open end of the cone. Defaults to None.
    * ``debug_vis``: Whether to draw the staging point and the route, coloured by the last step's
      progress, when the simulation is rendering (see :meth:`_init_debug_vis`). Defaults to True.
    """

    _PROGRESS_DEADBAND = 0.005
    """Per-step progress below which the debug route is drawn as no progress [m]."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._capture: ToolCapture | None = None
        self._prev_phi = torch.zeros(env.num_envs, device=env.device)
        self._has_prev = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._last_progress = torch.zeros(env.num_envs, device=env.device)

        self._debug_vis_handle = None
        if cfg.params.get("debug_vis", True) and env.sim.is_rendering:
            self._init_debug_vis()

    def __del__(self):
        if self._debug_vis_handle is not None:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._has_prev[env_ids] = False
        self._last_progress[env_ids] = 0.0

    def _capture_term(self) -> ToolCapture:
        # Looked up on first use rather than in __init__: the reward manager builds the terms in order, so
        # the capture term may not exist yet when this one is built.
        if self._capture is None:
            func = self._env.reward_manager.get_term_cfg(self.cfg.params["capture_term_name"]).func
            if isinstance(func, type):
                # Not built yet: the manager is still constructing its terms.
                return func  # type: ignore[return-value]
            self._capture = func  # type: ignore[assignment]
        return self._capture

    def _staging_distance(self) -> float:
        staging_distance = self.cfg.params.get("staging_distance")
        if staging_distance is None:
            staging_distance = self._capture_term().cfg.params["capture_radius"]
        return staging_distance

    def _potential(self, cs: ToolCaptureState) -> torch.Tensor:
        distance = torch.norm(cs.offset_t, dim=-1)
        if not self.cfg.params.get("route_aware", True):
            return -distance
        staging_distance = self._staging_distance()
        staging_t = self._capture_term()._axis_t * staging_distance
        route = torch.norm(cs.offset_t - staging_t, dim=-1) + staging_distance
        return -torch.where(cs.in_region, distance, route)

    def _init_debug_vis(self):
        """Create the route markers, drawn on every rendered frame.

        * The staging point: a small sphere on the cone axis, grey until the tool has come in through the
          cone (the capture term's cone-entry flag), cyan once it has.
        * The route the potential measures: a thin rod from the tool to the staging point and another down
          the cone to the target; inside the capture region, a single rod from the tool to the target.
          Without ``route_aware``, always that single rod.

        The rods are coloured by the last step's progress: green for progress along the route, red for
        lost ground, grey for under :attr:`_PROGRESS_DEADBAND`.
        """
        # note: imported here because the marker class pulls in USD, and this module is imported by task
        # configs that are parsed before the simulation app is launched.
        import omni.kit.app

        from isaaclab.markers import VisualizationMarkers

        def _material(color):
            return sim_utils.PreviewSurfaceCfg(diffuse_color=color)

        rod = {
            f"progress_{name}": sim_utils.CylinderCfg(radius=0.006, height=1.0, axis="Z", visual_material=_material(c))
            for name, c in (("gain", (0.1, 0.85, 0.2)), ("loss", (0.9, 0.15, 0.1)), ("none", (0.6, 0.6, 0.6)))
        }
        self._route_visualizer = VisualizationMarkers(
            VisualizationMarkersCfg(prim_path="/Visuals/Reward/capture_route", markers=rod)
        )
        staging = {
            f"staging_{name}": sim_utils.SphereCfg(radius=0.02, visual_material=_material(c))
            for name, c in (("not_approached", (0.6, 0.6, 0.6)), ("approached", (0.0, 0.8, 0.9)))
        }
        self._staging_visualizer = VisualizationMarkers(
            VisualizationMarkersCfg(prim_path="/Visuals/Reward/capture_staging", markers=staging)
        )
        # Drawn from the post-update event, like the capture markers, so they keep up with the drawn robot.
        self._debug_vis_handle = (
            omni.kit.app.get_app_interface()
            .get_post_update_event_stream()
            .create_subscription_to_pop(lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event))
        )

    def _debug_vis_callback(self, event):
        capture = self._capture_term()
        # note: needed in case the robot is de-initialized, since its data cannot be accessed then
        if isinstance(capture, type) or not capture._asset.is_initialized:
            return
        cs = capture.capture_state()
        origins = self._env.scene.env_origins
        staging_w = cs.target_pos + math_utils.quat_apply(cs.target_quat, capture._axis_t * self._staging_distance())
        direct = cs.in_region | (not self.cfg.params.get("route_aware", True))
        # Two rods per environment: tool -> staging point -> target outside the capture region; inside it,
        # tool -> target and a hidden second rod. A constant count keeps the point instancer from resizing.
        first_end = torch.where(direct.unsqueeze(-1), cs.target_pos, staging_w)
        starts = torch.stack([cs.tool_pos, staging_w], dim=1)
        ends = torch.stack([first_end, cs.target_pos], dim=1)
        visible = torch.stack([torch.ones_like(direct), ~direct], dim=1)
        progress = self._last_progress
        state = torch.where(
            progress > self._PROGRESS_DEADBAND, 0, torch.where(progress < -self._PROGRESS_DEADBAND, 1, 2)
        )
        self._draw_rods(
            starts.reshape(-1, 3),
            ends.reshape(-1, 3),
            visible.reshape(-1),
            state.repeat_interleave(2),
            origins.repeat_interleave(2, dim=0),
        )
        self._staging_visualizer.visualize(staging_w + origins, marker_indices=cs.approached.long())

    def _draw_rods(self, starts, ends, visible, marker_indices, origins):
        """Draw unit-height Z cylinders stretched from ``starts`` to ``ends``; hidden ones get zero scale."""
        segment = ends - starts
        length = torch.norm(segment, dim=-1)
        direction = segment / length.clamp(min=1e-6).unsqueeze(-1)
        z_axis = torch.zeros_like(direction)
        z_axis[:, 2] = 1.0
        rot_axis = torch.cross(z_axis, direction, dim=-1)
        rot_norm = torch.norm(rot_axis, dim=-1)
        angle = torch.atan2(rot_norm, torch.sum(z_axis * direction, dim=-1))
        # Parallel or antiparallel to Z: any horizontal axis turns it.
        rot_axis = torch.where(
            (rot_norm < 1e-6).unsqueeze(-1), torch.tensor([1.0, 0.0, 0.0], device=starts.device), rot_axis
        )
        quat = math_utils.quat_from_angle_axis(angle, torch.nn.functional.normalize(rot_axis, dim=-1))
        thickness = visible.float()
        scales = torch.stack([thickness, thickness, length * thickness], dim=-1)
        self._route_visualizer.visualize(
            translations=(starts + ends) / 2.0 + origins,
            orientations=quat,
            scales=scales,
            marker_indices=marker_indices,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        capture_term_name: str,
        route_aware: bool = True,
        staging_distance: float | None = None,
        debug_vis: bool = True,
    ) -> torch.Tensor:
        """Compute the reward.

        Returns:
            A 1-D tensor of shape (num_envs,): the change in the potential this step [m].
        """
        phi = self._potential(self._capture_term().capture_state())
        reward = torch.where(self._has_prev, phi - self._prev_phi, torch.zeros_like(phi))
        self._prev_phi = phi
        self._has_prev[:] = True
        self._last_progress = reward
        return reward


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
