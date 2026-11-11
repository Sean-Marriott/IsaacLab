# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Staged approach of a tool frame carried by the robot to a target pose."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import ToolPoseCommandCfg


class ToolPoseCommand(CommandTerm):
    """A staged approach of a tool frame, fixed to a body of the robot, to a target pose.

    Each episode starts in the staging phase, with the goal at a staging point on the approach axis behind
    the target. Once the tool is there, slow and square to the target, the goal switches to the target. If the
    staged tool then leaves the approach corridor (too far off the axis, turned too far, past the target, or
    backed out beyond the staging point), the approach is aborted: the goal returns to the staging point, and
    the tool has to stage again before the target counts. The term also owns the tool frame, so every
    observation, reward and termination reads the tool state and its errors from one place.

    The target position is fixed relative to the environment origin, and so is the target yaw unless a reset event
    sets it with :meth:`set_target_yaw`. The command does not change the target when it resamples, so the events
    can place the robot relative to it before the command is reset. The policy observes only relative quantities,
    so a fixed target loses no generality.

    Metrics, taken at the end of the episode: the distance and yaw error to the target, whether the tool was
    staged, the arrival angle (see :attr:`ToolPoseCommandCfg.success_max_arrival_angle`) in degrees, and
    whether the episode succeeded: the tool at some point on the target, slow, square, and having arrived
    within the arrival angle, however it got there. The hold fraction is the share of the steps since the tool
    first came within the success radius that it spent there under the success speed (0 if it never arrived),
    which tells a held arrival from a swing passing through. The retries are the number of aborted approaches.
    """

    cfg: ToolPoseCommandCfg

    def __init__(self, cfg: ToolPoseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self._tool_pos_b = torch.tensor(cfg.tool_pos, device=self.device).repeat(self.num_envs, 1)
        self._tool_quat_b = torch.tensor(cfg.tool_rot, device=self.device).repeat(self.num_envs, 1)

        # -- target and staging point, in the simulation world frame
        self.target_pos_w = torch.tensor(cfg.target_pos, device=self.device) + self._env.scene.env_origins
        """Target position [m]. Shape is (num_envs, 3)."""
        self.target_yaw = torch.full((self.num_envs,), cfg.target_yaw, device=self.device)
        """Target yaw [rad]. Shape is (num_envs,)."""
        self.target_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        """Target orientation (yaw only), as a quaternion (x, y, z, w). Shape is (num_envs, 4)."""
        self.approach_axis_w = torch.zeros(self.num_envs, 3, device=self.device)
        """Unit approach axis, from the target towards the side the tool arrives from. Shape is (num_envs, 3)."""
        self.staging_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        """Staging position [m]. Shape is (num_envs, 3)."""
        self._approach_axis_t = torch.nn.functional.normalize(
            torch.tensor(cfg.approach_axis, device=self.device), dim=0
        )
        self.set_target_yaw(torch.arange(self.num_envs, device=self.device), self.target_yaw.clone())

        # -- per-episode state
        self.staged = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """Whether the tool has been staged this episode, so the goal is the target. Shape is (num_envs,)."""
        self.succeeded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        """Whether the episode has succeeded so far. Shape is (num_envs,)."""
        self._arrival_dir = torch.zeros(self.num_envs, 3, device=self.device)
        self._arrived = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._steps_since_arrival = torch.zeros(self.num_envs, device=self.device)
        self._hold_steps = torch.zeros(self.num_envs, device=self.device)
        self._retries = torch.zeros(self.num_envs, device=self.device)
        self._time_since_abort = torch.full((self.num_envs,), torch.inf, device=self.device)

        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["yaw_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["staged"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["arrival_angle_deg"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["hold_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["retries"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "ToolPoseCommand:\n"
        msg += f"\tTool body: {self.cfg.body_name}\n"
        msg += f"\tStaging distance: {self.cfg.staging_distance} m\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The current goal position relative to the environment origin and the target yaw, as (x, y, z, yaw).
        Shape is (num_envs, 4)."""
        return torch.cat((self.goal_pos_w - self._env.scene.env_origins, self.target_yaw.unsqueeze(-1)), dim=-1)

    @property
    def goal_pos_w(self) -> torch.Tensor:
        """The current goal: the staging point until the tool is staged, then the target [m]. Shape is (num_envs, 3)."""
        return torch.where(self.staged.unsqueeze(-1), self.target_pos_w, self.staging_pos_w)

    @property
    def tool_pos_w(self) -> torch.Tensor:
        """Tool position in the simulation world frame [m]. Shape is (num_envs, 3)."""
        body_pos_w = self.robot.data.body_link_pos_w.torch[:, self.body_idx]
        body_quat_w = self.robot.data.body_link_quat_w.torch[:, self.body_idx]
        return body_pos_w + math_utils.quat_apply(body_quat_w, self._tool_pos_b)

    @property
    def tool_quat_w(self) -> torch.Tensor:
        """Tool orientation in the simulation world frame, as a quaternion (x, y, z, w). Shape is (num_envs, 4)."""
        return math_utils.quat_mul(self.robot.data.body_link_quat_w.torch[:, self.body_idx], self._tool_quat_b)

    @property
    def tool_lin_vel_w(self) -> torch.Tensor:
        """Tool linear velocity in the simulation world frame [m/s]. Shape is (num_envs, 3)."""
        body_quat_w = self.robot.data.body_link_quat_w.torch[:, self.body_idx]
        lever_w = math_utils.quat_apply(body_quat_w, self._tool_pos_b)
        lin_vel_w = self.robot.data.body_link_lin_vel_w.torch[:, self.body_idx]
        ang_vel_w = self.robot.data.body_link_ang_vel_w.torch[:, self.body_idx]
        return lin_vel_w + torch.linalg.cross(ang_vel_w, lever_w)

    @property
    def tool_yaw(self) -> torch.Tensor:
        """Heading of the tool +x axis in the horizontal plane [rad]. Shape is (num_envs,)."""
        x_axis = torch.zeros_like(self._tool_pos_b)
        x_axis[:, 0] = 1.0
        x_axis_w = math_utils.quat_apply(self.tool_quat_w, x_axis)
        return torch.atan2(x_axis_w[:, 1], x_axis_w[:, 0])

    """
    Operations
    """

    def set_target_yaw(self, env_ids: torch.Tensor, yaw: torch.Tensor):
        """Turn the target to a new yaw, with its approach axis and staging point.

        Call it from a reset event, before any event that reads the target, such as the spawn.

        Args:
            env_ids: Indices of the environments whose target to turn. Shape is (N,).
            yaw: New target yaw [rad]. Shape is (N,).
        """
        self.target_yaw[env_ids] = yaw
        zeros = torch.zeros_like(yaw)
        self.target_quat_w[env_ids] = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
        self.approach_axis_w[env_ids] = math_utils.quat_apply(
            self.target_quat_w[env_ids], self._approach_axis_t.expand(len(env_ids), 3)
        )
        self.staging_pos_w[env_ids] = (
            self.target_pos_w[env_ids] + self.cfg.staging_distance * self.approach_axis_w[env_ids]
        )

    """
    Errors
    """

    def target_error_w(self) -> torch.Tensor:
        """Target position minus tool position, in the simulation world frame [m]. Shape is (num_envs, 3)."""
        return self.target_pos_w - self.tool_pos_w

    def goal_error_w(self) -> torch.Tensor:
        """Current goal position minus tool position, in the simulation world frame [m]. Shape is (num_envs, 3)."""
        return self.goal_pos_w - self.tool_pos_w

    def tool_offset_t(self) -> torch.Tensor:
        """Tool position relative to the target, in the target's yaw frame [m]. Shape is (num_envs, 3)."""
        return math_utils.quat_apply_inverse(self.target_quat_w, -self.target_error_w())

    def yaw_error(self) -> torch.Tensor:
        """Target yaw minus tool yaw, wrapped to [-pi, pi] [rad]. Shape is (num_envs,)."""
        return math_utils.wrap_to_pi(self.target_yaw - self.tool_yaw)

    def route_distance(self) -> torch.Tensor:
        """Length of the remaining route to the target [m]. Shape is (num_envs,).

        Before staging, the distance to the staging point plus the staging distance; after, the distance to
        the target. By the triangle inequality it never increases at the switch (it drops by at most twice the
        staging radius), so rewards on it do not drop when the goal moves on. It does jump up when an approach
        is aborted, which is what makes losing the alignment cost reward.
        """
        to_staging = torch.linalg.norm(self.staging_pos_w - self.tool_pos_w, dim=-1) + self.cfg.staging_distance
        to_target = torch.linalg.norm(self.target_error_w(), dim=-1)
        return torch.where(self.staged, to_target, to_staging)

    def cross_track_error(self) -> torch.Tensor:
        """Distance of the tool from the approach axis line through the target [m]. Shape is (num_envs,)."""
        offset_w = -self.target_error_w()
        along = torch.sum(offset_w * self.approach_axis_w, dim=-1, keepdim=True)
        return torch.linalg.norm(offset_w - along * self.approach_axis_w, dim=-1)

    """
    Implementation specific functions.
    """

    def _arrival_angle(self) -> torch.Tensor:
        cos_angle = torch.sum(self._arrival_dir * self.approach_axis_w, dim=-1)
        return torch.acos(cos_angle.clamp(-1.0, 1.0))

    def _update_metrics(self):
        offset_w = -self.target_error_w()
        distance = torch.linalg.norm(offset_w, dim=-1)
        yaw_error = torch.abs(self.yaw_error())
        speed = torch.linalg.norm(self.tool_lin_vel_w, dim=-1)

        # The direction the tool came in from: recorded while it is still outside the arrival distance.
        outside = distance >= self.cfg.arrival_distance
        self._arrival_dir[outside] = offset_w[outside] / distance[outside].unsqueeze(-1)
        arrival_angle = self._arrival_angle()

        on_target = distance < self.cfg.success_radius
        holding = on_target & (speed < self.cfg.success_max_speed)
        self.succeeded |= (
            holding
            & (yaw_error < self.cfg.success_max_yaw_error)
            & (arrival_angle < self.cfg.success_max_arrival_angle)
        )

        # Hold fraction: of the steps since the tool first reached the target, the share it spent holding there.
        self._arrived |= on_target
        self._steps_since_arrival += self._arrived.float()
        self._hold_steps += holding.float()

        self.metrics["position_error"] = distance
        self.metrics["yaw_error"] = yaw_error
        self.metrics["staged"] = self.staged.float()
        self.metrics["arrival_angle_deg"] = torch.rad2deg(arrival_angle)
        self.metrics["success"] = self.succeeded.float()
        self.metrics["hold_fraction"] = self._hold_steps / self._steps_since_arrival.clamp(min=1.0)
        self.metrics["retries"] = self._retries

    def _resample_command(self, env_ids: Sequence[int]):
        # The target is fixed; a new episode starts in the staging phase.
        self.staged[env_ids] = False
        self.succeeded[env_ids] = False
        self._arrived[env_ids] = False
        self._steps_since_arrival[env_ids] = 0.0
        self._hold_steps[env_ids] = 0.0
        self._retries[env_ids] = 0.0
        self._time_since_abort[env_ids] = torch.inf
        # Until the tool has been outside the arrival distance, count it as arriving along the axis.
        self._arrival_dir[env_ids] = self.approach_axis_w[env_ids]

    def _update_command(self):
        yaw_error = torch.abs(self.yaw_error())

        # -- abort a staged approach that has left the corridor
        # The abort and staging regions do not overlap, so an approach cannot be aborted and staged in one step.
        along = torch.sum(-self.target_error_w() * self.approach_axis_w, dim=-1)
        off_corridor = (
            (self.cross_track_error() > self.cfg.abort_max_cross_track)
            | (yaw_error > self.cfg.abort_max_yaw_error)
            | (along < -self.cfg.abort_max_overshoot)
        )
        if self.cfg.abort_on_back_out:
            off_corridor |= along > self.cfg.staging_distance + self.cfg.staging_radius
        aborted = self.staged & off_corridor
        self.staged &= ~aborted
        self._retries += aborted.float()
        self._time_since_abort += self._env.step_dt
        self._time_since_abort[aborted] = 0.0

        # -- stage
        at_staging = torch.linalg.norm(self.staging_pos_w - self.tool_pos_w, dim=-1) < self.cfg.staging_radius
        slow = torch.linalg.norm(self.tool_lin_vel_w, dim=-1) < self.cfg.staging_max_speed
        square = yaw_error < self.cfg.staging_max_yaw_error
        self.staged |= at_staging & slow & square

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)
                self.staging_visualizer = VisualizationMarkers(self.cfg.staging_visualizer_cfg)
                self.approach_visualizer = VisualizationMarkers(self.cfg.approach_visualizer_cfg)
                self.tool_visualizer = VisualizationMarkers(self.cfg.tool_visualizer_cfg)
                # An infinite limit has no surface to draw.
                if math.isfinite(self.cfg.abort_max_cross_track) and math.isfinite(self.cfg.abort_max_overshoot):
                    self.corridor_visualizer = VisualizationMarkers(self.cfg.corridor_visualizer_cfg)
            self.target_visualizer.set_visibility(True)
            self.staging_visualizer.set_visibility(True)
            self.approach_visualizer.set_visibility(True)
            self.tool_visualizer.set_visibility(True)
            if hasattr(self, "corridor_visualizer"):
                self.corridor_visualizer.set_visibility(True)
        elif hasattr(self, "target_visualizer"):
            self.target_visualizer.set_visibility(False)
            self.staging_visualizer.set_visibility(False)
            self.approach_visualizer.set_visibility(False)
            self.tool_visualizer.set_visibility(False)
            if hasattr(self, "corridor_visualizer"):
                self.corridor_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # the robot may be de-initialized, in which case its data cannot be read
        if not self.robot.is_initialized:
            return
        self.target_visualizer.visualize(self.target_pos_w, self.target_quat_w)
        self.tool_visualizer.visualize(self.tool_pos_w, self.tool_quat_w)

        # The geometry follows the target, which reset events may turn (see set_target_yaw), so it is rebuilt on
        # every call. Rendering runs at the policy rate or slower, so the cost is negligible.
        # The rail: a unit cylinder along its z axis, turned onto the approach axis, stretched to the staging
        # distance and centred between the staging point and the target.
        z_axis = torch.zeros_like(self.approach_axis_w)
        z_axis[:, 2] = 1.0
        rot_axis = torch.linalg.cross(z_axis, self.approach_axis_w)
        # An approach along z needs no turn (or a half turn about x); keep the rotation axis well defined.
        rot_axis[torch.linalg.norm(rot_axis, dim=-1) < 1e-6] = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        angle = torch.acos(torch.sum(z_axis * self.approach_axis_w, dim=-1).clamp(-1.0, 1.0))
        rail_quat = math_utils.quat_from_angle_axis(angle, torch.nn.functional.normalize(rot_axis, dim=-1))
        rail_scale = torch.ones(self.num_envs, 3, device=self.device)
        rail_scale[:, 2] = self.cfg.staging_distance
        self.approach_visualizer.visualize(0.5 * (self.staging_pos_w + self.target_pos_w), rail_quat, scales=rail_scale)

        # The staging region, amber (marker 0) until staged, then green (marker 1).
        staging_scale = torch.full((self.num_envs, 3), self.cfg.staging_radius, device=self.device)
        self.staging_visualizer.visualize(
            self.staging_pos_w, self.target_quat_w, scales=staging_scale, marker_indices=self.staged.int()
        )

        if hasattr(self, "corridor_visualizer"):
            # The corridor: four rods parallel to the rail, at the cross-track limit to either side of it and above
            # and below it, running from the overshoot limit past the target to the back of the staging region.
            # Each environment's four rods are consecutive marker instances.
            front = -self.cfg.abort_max_overshoot
            back = self.cfg.staging_distance + self.cfg.staging_radius
            centre = self.target_pos_w + 0.5 * (front + back) * self.approach_axis_w
            side = torch.nn.functional.normalize(torch.linalg.cross(z_axis, self.approach_axis_w), dim=-1)
            # A vertical approach axis has no side; any horizontal direction will do.
            side[torch.linalg.norm(side, dim=-1) < 1e-6] = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            up = torch.linalg.cross(self.approach_axis_w, side)
            offsets = self.cfg.abort_max_cross_track * torch.stack((side, -side, up, -up), dim=1)
            corridor_scale = torch.ones(4 * self.num_envs, 3, device=self.device)
            corridor_scale[:, 2] = back - front
            # Grey (0) until staged, blue (1) while staged, red (2) just after an abort.
            just_aborted = self._time_since_abort < self.cfg.abort_vis_hold_s
            state = torch.where(just_aborted, 2, self.staged.int())
            self.corridor_visualizer.visualize(
                (centre.unsqueeze(1) + offsets).reshape(-1, 3),
                rail_quat.repeat_interleave(4, dim=0),
                scales=corridor_scale,
                marker_indices=state.repeat_interleave(4),
            )
