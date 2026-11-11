# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass, replace

if TYPE_CHECKING:
    from .tool_pose_command import ToolPoseCommand


@configclass
class ToolPoseCommandCfg(CommandTermCfg):
    """Configuration for a staged approach of a tool frame, fixed to a body of the robot, to a target pose.

    The tool first has to reach a staging point ``staging_distance`` along ``approach_axis`` from the target,
    slow and square to it. Only then does the goal switch to the target itself, so the tool arrives along the
    approach axis. If the tool then leaves the approach corridor (see the ``abort_*`` parameters), the approach
    is aborted and the goal returns to the staging point for another try.
    """

    class_type: type["ToolPoseCommand"] | str = "{DIR}.tool_pose_command:ToolPoseCommand"

    asset_name: str = MISSING
    """Name of the asset that carries the tool."""

    body_name: str = MISSING
    """Name of the body the tool frame is fixed to."""

    tool_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Origin of the tool frame in the body's link frame [m]."""

    tool_rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    """Orientation of the tool frame in the body's link frame, as a quaternion (x, y, z, w).

    The tool yaw is the heading of the tool frame's +x axis, projected onto the horizontal plane.
    """

    target_pos: tuple[float, float, float] = MISSING
    """Target position relative to the environment origin [m]."""

    target_yaw: float = 0.0
    """Target yaw [rad]."""

    approach_axis: tuple[float, float, float] = (-1.0, 0.0, 0.0)
    """Direction from the target to the side the tool arrives from, in the target's yaw frame. Normalized
    internally."""

    staging_distance: float = 0.5
    """Distance of the staging point from the target along the approach axis [m]."""

    staging_radius: float = 0.1
    """Distance from the staging point within which the tool counts as staged [m]."""

    staging_max_speed: float = 0.3
    """Largest tool speed at which the tool counts as staged [m/s]."""

    staging_max_yaw_error: float = math.radians(15.0)
    """Largest tool yaw error at which the tool counts as staged [rad]."""

    abort_max_cross_track: float = 0.2
    """Largest distance of a staged tool from the approach axis before the approach is aborted [m].

    Each abort threshold is wider than the matching staging one, so the staged flag cannot flicker at the edge.
    Set an abort threshold to ``math.inf`` to disable it.
    """

    abort_max_yaw_error: float = math.radians(30.0)
    """Largest yaw error of a staged tool before the approach is aborted [rad]."""

    abort_max_overshoot: float = 0.1
    """Largest distance a staged tool may pass beyond the target along the approach axis before the approach is
    aborted [m]."""

    abort_on_back_out: bool = True
    """Whether a staged tool that backs out along the approach axis beyond the staging region (more than
    :attr:`staging_distance` plus :attr:`staging_radius` behind the target) aborts the approach, so the policy
    can retry by backing straight out."""

    success_radius: float = 0.05
    """Distance from the target within which the tool can count as a success [m]."""

    success_max_speed: float = 0.1
    """Largest tool speed for a success [m/s]."""

    success_max_yaw_error: float = math.radians(10.0)
    """Largest tool yaw error for a success [rad]."""

    success_max_arrival_angle: float = math.radians(20.0)
    """Largest arrival angle for a success [rad]: the angle between the approach axis and the direction of
    the tool from the target when it was last :attr:`arrival_distance` away."""

    arrival_distance: float = 0.3
    """Distance from the target at which the arrival direction is recorded [m]."""

    target_visualizer_cfg: VisualizationMarkersCfg = replace(FRAME_MARKER_CFG, prim_path="/Visuals/Command/tool_target")
    """Marker for the target pose."""

    staging_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/tool_staging",
        markers={
            "pending": sim_utils.SphereCfg(
                radius=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.0), opacity=0.35),
            ),
            "staged": sim_utils.SphereCfg(
                radius=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2), opacity=0.35),
            ),
        },
    )
    """Marker for the staging region: a translucent sphere of :attr:`staging_radius` at the staging point, amber
    until the tool has been staged, then green. Unit sized; scaled at runtime."""

    approach_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/tool_approach",
        markers={
            "rail": sim_utils.CylinderCfg(
                radius=0.01,
                height=1.0,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.6, 0.0)),
            ),
        },
    )
    """Marker for the approach: a thin rail along the approach axis from the staging point to the target. Unit
    length; scaled at runtime to :attr:`staging_distance`."""

    corridor_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/tool_corridor",
        markers={
            "idle": sim_utils.CylinderCfg(
                radius=0.005,
                height=1.0,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
            ),
            "armed": sim_utils.CylinderCfg(
                radius=0.005,
                height=1.0,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 1.0)),
            ),
            "aborted": sim_utils.CylinderCfg(
                radius=0.008,
                height=1.0,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
            ),
        },
    )
    """Marker for the abort corridor: four thin rods parallel to the approach axis, left, right, above and below it
    at :attr:`abort_max_cross_track`, from :attr:`abort_max_overshoot` past the target to the back of the staging
    region. Grey while the tool is not staged, blue while it is (the corridor is armed), and red for
    :attr:`abort_vis_hold_s` after an abort. Rods rather than a translucent tube, because the viewport renders
    marker opacity as solid, which would hide the other markers. The yaw limit is not drawn. Not drawn if the
    cross-track or overshoot limit is infinite. Unit length; scaled at runtime."""

    abort_vis_hold_s: float = 1.0
    """Time the corridor marker stays red after an abort [s]."""

    tool_visualizer_cfg: VisualizationMarkersCfg = replace(FRAME_MARKER_CFG, prim_path="/Visuals/Command/tool_pose")
    """Marker for the current tool pose."""

    target_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    tool_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
