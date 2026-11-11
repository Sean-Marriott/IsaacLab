# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for the chainsaw alignment environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from .observations import rolled_hang_offset

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObjectCollection
    from isaaclab.envs import ManagerBasedRLEnv

    from .tool_pose_command import ToolPoseCommand


def lock_passive_joints(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg, stiffness: float
) -> None:
    """Hold unactuated joints at zero angle with a stiff drive.

    The drive target of a passive joint is zero, so a large drive stiffness locks it there. The drive is
    written directly because a :class:`~isaaclab_contrib.assets.Multirotor` cannot have joint actuators.
    It goes through the CPU tensor API, so the term is meant for ``mode="startup"``.

    Args:
        env: The manager-based RL environment instance.
        env_ids: Indices of the environments to write. All environments if None.
        asset_cfg: SceneEntityCfg identifying the asset and the joints to lock.
        stiffness: Drive stiffness [N·m/rad].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=asset.device)
    joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int32, device=asset.device)
    stiffness_values = torch.full((len(env_ids), len(joint_ids)), stiffness, device=asset.device)
    asset.write_joint_stiffness_to_sim_index(stiffness=stiffness_values, joint_ids=joint_ids, env_ids=env_ids)


class RandomWind(ManagerTermBase):
    """Apply a random wind gust to a body as a constant force and torque, resampled at random intervals.

    Each environment holds its gust for a random interval, then draws a new one, which models sustained wind
    rather than an impulse. Every gust component is drawn uniformly from its range, so zero ranges mean no
    wind. The ranges are read on every call, so a curriculum can grow them during training.

    On top of that, with a nonzero ``burst_force_range``, each environment occasionally takes a strong horizontal
    burst: a force of a magnitude drawn from that range, in a random horizontal direction, held for
    ``burst_duration_range_s`` and then gone, with ``burst_interval_range_s`` of quiet between bursts. Strong enough,
    it throws the vehicle off course, which is how an evaluation can show the recovery. The first burst of an
    episode comes a quiet interval after the reset.

    Run it every policy step (``mode="interval"`` with ``interval_range_s`` equal to the policy step and
    ``is_global_time=True``) so the hold intervals are timed per step. The wrench goes into the asset's
    permanent wrench buffer and stays applied between calls.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        device = env.device
        self._force = torch.zeros(env.num_envs, 3, device=device)
        self._torque = torch.zeros(env.num_envs, 3, device=device)
        self._time_left = torch.zeros(env.num_envs, device=device)
        self._burst_force = torch.zeros(env.num_envs, 3, device=device)
        self._burst_time_left = torch.zeros(env.num_envs, device=device)
        # time to the next burst; negative means one is still to be scheduled
        self._next_burst = torch.full((env.num_envs,), -1.0, device=device)
        self._all_env_ids = torch.arange(env.num_envs, device=device, dtype=torch.int32)

    @property
    def burst_active(self) -> torch.Tensor:
        """Whether each environment is taking a burst. Shape is (num_envs,)."""
        return self._burst_time_left > 0.0

    def reset(self, env_ids: torch.Tensor | None = None):
        """Draw a new gust, and schedule a new burst, for the reset environments on the next call."""
        ids = slice(None) if env_ids is None else env_ids
        self._time_left[ids] = 0.0
        self._burst_time_left[ids] = 0.0
        self._next_burst[ids] = -1.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        force_range: tuple[float, float],
        torque_range: tuple[float, float],
        interval_range_s: tuple[float, float] = (2.0, 6.0),
        burst_force_range: tuple[float, float] = (0.0, 0.0),
        burst_duration_range_s: tuple[float, float] = (1.0, 2.0),
        burst_interval_range_s: tuple[float, float] = (6.0, 12.0),
    ) -> None:
        """Resample the gusts that have run out, start and end bursts, and write the wind wrench for every environment.

        Args:
            env: The manager-based RL environment instance.
            env_ids: Unused. The term always writes every environment.
            asset_cfg: SceneEntityCfg identifying the asset and the body the wind acts on.
            force_range: Range of each gust force component, in the world frame [N].
            torque_range: Range of each gust torque component, in the world frame [N·m].
            interval_range_s: Range of the time each gust is held for [s].
            burst_force_range: Range of the magnitude of a burst's horizontal force [N]. Zero for no bursts.
            burst_duration_range_s: Range of the time a burst lasts [s].
            burst_interval_range_s: Range of the quiet time before each burst [s].
        """
        asset: Articulation = env.scene[asset_cfg.name]
        device = env.device
        self._time_left -= env.step_dt
        expired = (self._time_left <= 0.0).nonzero().flatten()
        if len(expired) > 0:
            n = len(expired)
            self._force[expired] = math_utils.sample_uniform(*force_range, (n, 3), device)
            self._torque[expired] = math_utils.sample_uniform(*torque_range, (n, 3), device)
            self._time_left[expired] = math_utils.sample_uniform(*interval_range_s, (n,), device)

        force = self._force
        if burst_force_range[1] > 0.0:
            self._burst_time_left -= env.step_dt
            unscheduled = (self._next_burst < 0.0).nonzero().flatten()
            if len(unscheduled) > 0:
                self._next_burst[unscheduled] = math_utils.sample_uniform(
                    *burst_interval_range_s, (len(unscheduled),), device
                )
            self._next_burst -= env.step_dt
            starting = ((self._next_burst <= 0.0) & ~self.burst_active).nonzero().flatten()
            if len(starting) > 0:
                n = len(starting)
                magnitude = math_utils.sample_uniform(*burst_force_range, (n,), device)
                heading = math_utils.sample_uniform(-torch.pi, torch.pi, (n,), device)
                self._burst_force[starting] = torch.stack(
                    (magnitude * torch.cos(heading), magnitude * torch.sin(heading), torch.zeros_like(heading)), dim=-1
                )
                duration = math_utils.sample_uniform(*burst_duration_range_s, (n,), device)
                self._burst_time_left[starting] = duration
                self._next_burst[starting] = duration + math_utils.sample_uniform(*burst_interval_range_s, (n,), device)
            force = force + self._burst_force * self.burst_active.unsqueeze(-1)

        asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=force.unsqueeze(1),
            torques=self._torque.unsqueeze(1),
            body_ids=asset_cfg.body_ids,
            env_ids=self._all_env_ids,
            is_global=True,
        )


def reset_tool_around_target(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    command_name: str,
    hang_offset: tuple[float, float, float],
    distance_range: tuple[float, float],
    vertical_range: tuple[float, float],
    max_azimuth: float,
    yaw_range: tuple[float, float],
    rp_range: tuple[float, float] = (0.0, 0.0),
    lin_vel_range: tuple[float, float] = (0.0, 0.0),
    ang_vel_range: tuple[float, float] = (0.0, 0.0),
    keep_out_half_extents: tuple[float, float, float] | None = None,
    max_resamples: int = 10,
    roll_cfg: SceneEntityCfg | None = None,
    roll_hang_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the robot so that its tool starts at a sampled bearing and distance from the target.

    The tool is placed ``distance`` from the target horizontally, at an azimuth within ``max_azimuth`` of the
    approach axis (0 is straight behind the target on the axis, pi is anywhere around it), and ``vertical``
    above or below it. The robot faces the target yaw turned by ``yaw``, and is placed so that the tool is
    there with the payload hanging at rest. The approach axis is taken to be horizontal.

    With ``keep_out_half_extents``, tool positions inside a box centred on the target and aligned with its yaw
    frame are drawn again, up to ``max_resamples`` times, so that the payload does not spawn inside an obstacle
    there. Any still inside after that are kept.

    If the tool can be rolled, the hang offset allows for the roll (see
    :func:`~.observations.rolled_hang_offset`), which is read from the joint: run any event that sets the roll
    before this one.

    The target is read from the command term, which does not change it on reset, so it is valid here even though
    the command is reset after the events. Run any event that turns the target before this one.

    Args:
        env: The manager-based RL environment instance.
        env_ids: Indices of the environments to reset.
        command_name: Name of the :class:`ToolPoseCommand` term.
        hang_offset: Tool position relative to the root with the robot level, the tool unrolled and the payload
            hanging at rest, in the yaw-aligned vehicle frame [m].
        distance_range: Range of the horizontal distance of the tool from the target [m].
        vertical_range: Range of the height of the tool relative to the target [m].
        max_azimuth: Largest angle of the tool's bearing from the approach axis, either side [rad].
        yaw_range: Range of the robot yaw relative to the target yaw [rad].
        rp_range: Range of the initial roll and pitch [rad].
        lin_vel_range: Range of the initial linear velocity on each axis [m/s].
        ang_vel_range: Range of the initial angular velocity about each axis [rad/s].
        keep_out_half_extents: Half extents of the box the tool must not start in, in the target's yaw frame
            [m]. None for no keep-out box.
        max_resamples: Number of times a tool position inside the keep-out box is drawn again.
        roll_cfg: The tool-roll joint, if the tool can be rolled (see :func:`~.observations.rolled_hang_offset`).
        roll_hang_shift: Shift of the hang offset with the roll [m] (see
            :func:`~.observations.rolled_hang_offset`).
        asset_cfg: SceneEntityCfg identifying the asset to reset.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    term: ToolPoseCommand = env.command_manager.get_term(command_name)
    # A full reset passes a slice, which cannot be counted or used to gather.
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=asset.device)[env_ids if env_ids is not None else slice(None)]
    n = len(env_ids)

    def _sample(value_range: tuple[float, float], size: tuple[int, ...]) -> torch.Tensor:
        return math_utils.sample_uniform(value_range[0], value_range[1], size, device=asset.device)

    # -- tool position around the target
    axis_azimuth = torch.atan2(term.approach_axis_w[env_ids, 1], term.approach_axis_w[env_ids, 0])

    def _sample_offset(count: int, rows: torch.Tensor | slice) -> torch.Tensor:
        azimuth = axis_azimuth[rows] + _sample((-max_azimuth, max_azimuth), (count,))
        distance = _sample(distance_range, (count,))
        return torch.stack(
            (distance * torch.cos(azimuth), distance * torch.sin(azimuth), _sample(vertical_range, (count,))), dim=-1
        )

    offset_w = _sample_offset(n, slice(None))
    if keep_out_half_extents is not None:
        half_extents = torch.tensor(keep_out_half_extents, device=asset.device)
        for _ in range(max_resamples):
            offset_t = math_utils.quat_apply_inverse(term.target_quat_w[env_ids], offset_w)
            inside = torch.all(torch.abs(offset_t) < half_extents, dim=-1).nonzero().flatten()
            if len(inside) == 0:
                break
            offset_w[inside] = _sample_offset(len(inside), inside)
    tool_pos_w = term.target_pos_w[env_ids] + offset_w

    # -- robot placed so the tool is there with the payload at rest
    zeros = torch.zeros(n, device=asset.device)
    yaw = term.target_yaw[env_ids] + _sample(yaw_range, (n,))
    hang_offset_w = math_utils.quat_apply(
        math_utils.quat_from_euler_xyz(zeros, zeros, yaw),
        rolled_hang_offset(asset, hang_offset, env_ids, roll_cfg, roll_hang_shift),
    )
    positions = tool_pos_w - hang_offset_w
    orientations = math_utils.quat_from_euler_xyz(_sample(rp_range, (n,)), _sample(rp_range, (n,)), yaw)
    velocities = torch.cat((_sample(lin_vel_range, (n, 3)), _sample(ang_vel_range, (n, 3))), dim=-1)

    asset.write_root_pose_to_sim_index(root_pose=torch.cat((positions, orientations), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)


class RandomBranch(ManagerTermBase):
    """Place a branch at the target on every reset, of random diameter, direction and slope, and turn the target
    and the tool's roll to match.

    Each branch is a body of a :class:`~isaaclab.assets.RigidObjectCollection`, so every environment holds all of
    them and the branch can change between episodes without per-environment geometry. On each reset one is chosen
    and the others are parked out of reach. The chosen branch lies across the approach: along the target's y axis,
    turned by the target yaw and tilted by the slope about the target's x axis (the approach axis), so its far
    (+y) end rises.

    The target yaw is drawn from ``yaw_range`` and set on the command term (see
    :meth:`ToolPoseCommand.set_target_yaw`), so the approach follows the branch. Run this term before any reset event
    that reads the target, such as the spawn. The slope is drawn from 0 to ``max_tilt``: one sign is enough, since
    seen from the other side a branch slopes the other way, and with the yaw drawn over the full circle every branch
    line is covered.

    The tool has to be rolled to the slope to take the branch. ``roll_joint_name`` names a joint of the robot that
    rolls the tool about its forward axis, positive with the slope, such as the Mk3's saw-roll joint. It is held at
    the slope by its drive, as the vehicle's own saw-roll controller would, and set there on reset so the tool spawns
    already rolled. The drive target is written straight to the physics view, because a
    :class:`~isaaclab_contrib.assets.Multirotor` has no joint actuators to do it.

    The branch is placed at its seat offset from the target, rolled with the slope, so a seated branch puts the tool
    on the target. A steep branch would rise past the vehicle carrying the tool, so with ``max_rise`` the branch is
    slid down its own axis until it rises no more than that above the target: it ends above the cut, as a stem does
    near its top. A shallow branch is left centred on the target. The diameter and slope in use are kept in
    :attr:`diameter` and :attr:`tilt`.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        branches: RigidObjectCollection = env.scene[cfg.params["asset_cfg"].name]
        seats: dict[str, tuple[float, float, float]] = cfg.params["seats"]
        names = list(seats)
        self._body_ids = torch.tensor(
            [branches.find_bodies(f"^{name}$")[0][0] for name in names], dtype=torch.long, device=env.device
        )
        self._seats = torch.tensor([seats[name][:2] for name in names], device=env.device)
        self._diameters = torch.tensor([seats[name][2] for name in names], device=env.device)
        self.diameter = torch.zeros(env.num_envs, device=env.device)
        """Diameter of the branch in use in each environment [m]. Shape is (num_envs,)."""
        self.tilt = torch.zeros(env.num_envs, device=env.device)
        """Slope of the branch in use, and roll of the tool, in each environment [rad]. Shape is (num_envs,)."""

        # The roll joint, and the drive targets of every joint of the robot as authored, in the physics view's order.
        roll_joint_name: str | None = cfg.params.get("roll_joint_name")
        if roll_joint_name is not None:
            robot: Articulation = env.scene[cfg.params.get("robot_cfg", SceneEntityCfg("robot")).name]
            self._robot = robot
            self._roll_joint_id = robot.find_joints(f"^{roll_joint_name}$")[0][0]
            view = robot.root_view
            self._roll_dof = list(view.shared_metatype.dof_names).index(roll_joint_name)
            self._dof_targets = wp.to_torch(view.get_dof_position_targets()).clone()
            self._view_indices = torch.arange(env.num_envs, dtype=torch.int32, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        seats: dict[str, tuple[float, float, float]],
        yaw_range: tuple[float, float] = (0.0, 0.0),
        max_tilt: float = 0.0,
        roll_joint_name: str | None = None,
        branch_length: float | None = None,
        max_rise: float | None = None,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        park_offset: tuple[float, float, float] = (0.0, 0.0, -50.0),
    ) -> None:
        """Choose a branch for each reset environment, turn the target and the tool to it, and place it.

        Args:
            env: The manager-based RL environment instance.
            env_ids: Indices of the environments to reset. All environments if None.
            asset_cfg: SceneEntityCfg identifying the branch collection.
            command_name: Name of the :class:`ToolPoseCommand` term whose target the branch is placed at.
            seats: For each branch body, its seat ``(x, z, diameter)``: the branch axis relative to the tool when
                seated, along the tool's +x and +z axes before it is rolled [m], and the branch diameter [m]. Read
                at construction.
            yaw_range: Range of the target yaw [rad].
            max_tilt: Largest slope of the branch, and roll of the tool [rad]. Read on every call, so a curriculum
                can grow it.
            roll_joint_name: Joint that rolls the tool to the slope. None to leave the tool unrolled, in which case
                ``max_tilt`` must be zero for the branch to seat. Read at construction.
            branch_length: Length of the branches [m]. Needed with ``max_rise``.
            max_rise: Largest height of the branch's upper end above the target [m]. None to keep every branch
                centred on the target.
            robot_cfg: SceneEntityCfg identifying the robot with the roll joint.
            park_offset: Position the unused branches are parked at, relative to the environment origin [m].
        """
        branches: RigidObjectCollection = env.scene[asset_cfg.name]
        term: ToolPoseCommand = env.command_manager.get_term(command_name)
        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)
        n, num_branches = len(env_ids), len(self._body_ids)

        # -- the branch, and the target turned to it
        choice = torch.randint(num_branches, (n,), device=env.device)
        tilt = math_utils.sample_uniform(0.0, max_tilt, (n,), device=env.device)
        self.diameter[env_ids] = self._diameters[choice]
        self.tilt[env_ids] = tilt
        term.set_target_yaw(env_ids, math_utils.sample_uniform(*yaw_range, (n,), device=env.device))
        zeros = torch.zeros_like(tilt)
        branch_quat_w = math_utils.quat_mul(
            term.target_quat_w[env_ids], math_utils.quat_from_euler_xyz(tilt, zeros, zeros)
        )

        # -- all branches parked, then the chosen one at its seat, rolled with the slope
        positions = (env.scene.env_origins[env_ids] + torch.tensor(park_offset, device=env.device)).unsqueeze(1)
        positions = positions.repeat(1, num_branches, 1)
        seat = self._seats[choice]
        seat_t = torch.stack((seat[:, 0], torch.zeros_like(seat[:, 0]), seat[:, 1]), dim=-1)
        rows = torch.arange(n, device=env.device)
        positions[rows, choice] = term.target_pos_w[env_ids] + math_utils.quat_apply(branch_quat_w, seat_t)
        if max_rise is not None:
            # length above the target, and the slide down the axis that leaves it
            half = 0.5 * branch_length
            rise = torch.clamp(max_rise / torch.sin(tilt).clamp(min=1e-6), max=half)
            axis_w = math_utils.quat_apply(branch_quat_w, torch.tensor([0.0, 1.0, 0.0], device=env.device).expand(n, 3))
            positions[rows, choice] -= (half - rise).unsqueeze(-1) * axis_w
        orientations = branch_quat_w.unsqueeze(1).repeat(1, num_branches, 1)
        branches.write_body_pose_to_sim_index(
            body_poses=torch.cat((positions, orientations), dim=-1), body_ids=self._body_ids, env_ids=env_ids
        )

        # -- the tool rolled to the slope: set there, and held there by the joint's drive
        if hasattr(self, "_robot"):
            joint_ids = [self._roll_joint_id]
            self._robot.write_joint_position_to_sim_index(
                position=tilt.unsqueeze(-1), joint_ids=joint_ids, env_ids=env_ids
            )
            self._robot.write_joint_velocity_to_sim_index(
                velocity=zeros.unsqueeze(-1), joint_ids=joint_ids, env_ids=env_ids
            )
            self._dof_targets[env_ids, self._roll_dof] = tilt
            self._robot.root_view.set_dof_position_targets(
                wp.from_torch(self._dof_targets), wp.from_torch(self._view_indices)
            )
