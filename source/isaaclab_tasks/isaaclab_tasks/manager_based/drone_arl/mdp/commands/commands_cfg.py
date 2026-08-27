# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .drone_pose_command import DroneUniformPoseCommand
    from .trajectory_command import DroneTrajectoryCommand


def _sphere_marker_cfg(prim_path: str, radius: float, color: tuple[float, float, float]) -> VisualizationMarkersCfg:
    """Build a standalone single-sphere marker configuration.

    Args:
        prim_path: Prim path the markers are spawned under.
        radius: Sphere radius [m].
        color: Diffuse colour as an RGB triplet in ``[0, 1]``.

    Returns:
        A marker configuration that shares no state with any other.
    """
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color)
            )
        },
    )


def _flash_marker_cfg(
    prim_path: str, radius: float, colors: dict[str, tuple[float, float, float]], opacity: float
) -> VisualizationMarkersCfg:
    """Build a marker configuration holding one translucent sphere prototype per named cause.

    The insertion order of :paramref:`colors` is the prototype order, which is what
    ``marker_indices`` selects on, so the keys double as the mapping from a termination term name to
    its colour.

    Args:
        prim_path: Prim path the markers are spawned under.
        radius: Sphere radius [m].
        colors: Mapping from termination term name to a diffuse RGB triplet in ``[0, 1]``.
        opacity: Surface opacity in ``[0, 1]``.

    Returns:
        A marker configuration that shares no state with any other.
    """
    return VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            name: sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, emissive_color=color, opacity=opacity),
            )
            for name, color in colors.items()
        },
    )


@configclass
class DroneUniformPoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for uniform drone pose command generator."""

    class_type: type["DroneUniformPoseCommand"] | str = "{DIR}.drone_pose_command:DroneUniformPoseCommand"


@configclass
class DroneTrajectoryCommandCfg(CommandTermCfg):
    """Configuration for the sum-of-sinusoids drone trajectory command generator.

    The trajectory limits are given as ``(easy, hard)`` pairs and interpolated by
    :attr:`difficulty`, which a curriculum term can ramp over training.
    """

    class_type: type["DroneTrajectoryCommand"] | str = "{DIR}.trajectory_command:DroneTrajectoryCommand"

    asset_name: str = MISSING
    """Name of the asset in the environment for which the command is generated."""

    body_name: str = MISSING
    """Name of the body in the asset whose position is tracked against the reference."""

    difficulty: float = 1.0
    """Interpolation factor between the easy and hard end of every range in :attr:`ranges`."""

    lookahead_times_s: tuple[float, ...] = (0.1, 0.2, 0.4, 0.7, 1.0)
    """Times ahead of the current one at which the reference position is also reported [s].

    The total lag from action to velocity is roughly 0.4 s (velocity loop, attitude loop and one
    policy step), so the policy needs at least that much preview to cancel the phase lag. The
    geometric spacing keeps near-term resolution fine where it drives the action, while still
    reaching far enough to anticipate curvature.
    """

    num_harmonics: int = 3
    """Number of sinusoids summed per axis."""

    axis_weights: tuple[float, float, float] = (1.0, 1.0, 0.35)
    """Per-axis amplitude weights applied before normalization."""

    harmonic_multiplier_ranges: tuple[tuple[float, float], ...] = ((0.8, 1.2), (1.7, 2.4), (2.8, 3.6))
    """Sampling range of the frequency multiplier of each harmonic, relative to the fundamental.

    The values are deliberately non-integer, so the reference is quasi-periodic and does not close
    into a short loop that a policy could memorize.
    """

    harmonic_amplitude_ranges: tuple[tuple[float, float], ...] = ((0.6, 1.0), (0.1, 0.45), (0.0, 0.2))
    """Sampling range of the raw amplitude of each harmonic, before normalization."""

    omega_min: float = 0.15
    """Lower end of the fundamental angular frequency range [rad/s]."""

    yaw_mode: Literal["fixed", "random_constant", "sweep"] = "random_constant"
    """How the yaw reference is generated.

    ``"random_constant"`` is the sensible default: velocity actions are interpreted in the yaw-only
    vehicle frame, so a per-episode random yaw already forces the policy to learn that frame mapping
    without also having to chase a moving yaw target.
    """

    yaw_amplitude_range: tuple[float, float] = (0.3, 1.2)
    """Sampling range of the yaw sweep amplitude [rad]. Only used when :attr:`yaw_mode` is ``"sweep"``."""

    yaw_omega_range: tuple[float, float] = (0.1, 0.4)
    """Sampling range of the yaw sweep angular frequency [rad/s]. Only used for ``"sweep"``."""

    yaw_max_rate: float = 0.6
    """Upper bound on the yaw rate of the sweep [rad/s]. Only used for ``"sweep"``."""

    max_vis_envs: int = 32
    """Maximum number of environments to draw debug markers for.

    Drawing the full path for every environment of a large scene creates hundreds of thousands of
    marker instances and makes the viewport unusable.
    """

    num_path_vis_points: int = 60
    """Number of points used to draw the reference path of one environment."""

    @configclass
    class Ranges:
        """Trajectory limits at the easy and hard end of the difficulty range.

        The defaults are sized for the clean DJI M350 (6.5 kg, thrust-to-weight 3.45) flown through
        a Lee velocity controller with ``K_vel = 4.0``, within a 3 m/s action range.

        The speed budget is what ties these together. Sustaining the peak acceleration needs a
        standing velocity error of ``acc_cap / K_vel``, so the action range has to cover
        ``vel_cap + acc_cap / K_vel``. At the hard end that is ``2.25 + 3.0 / 4.0 = 3.0 m/s``
        exactly. The acceleration is still modest for this airframe -- 3 m/s^2 is 17.0 degrees of
        tilt and 16.7 N per motor against a 15.94 N hover and a 55 N limit -- so the envelope is
        limited by the action range, not by the vehicle.

        To change the envelope, scale ``vel_rms_target``, ``vel_cap``, ``acc_cap`` and all three
        ``pos`` limits by the same factor and raise ``max_velocity`` to match; leaving ``omega_max``
        alone keeps the shape and timescale of the paths and changes only their size. Several terms
        outside this class are coupled to the same factor -- see
        :class:`~isaaclab_tasks.manager_based.drone_arl.track_trajectory.config.matrice.track_trajectory_env_cfg.ActionsCfg`
        for the full procedure.
        """

        vel_rms_target: tuple[float, float] = (0.675, 1.125)
        """Target root-mean-square speed of the reference [m/s]."""
        pos_cap: tuple[float, float] = (1.5, 4.8)
        """Maximum excursion from the trajectory centre [m]."""
        vel_cap: tuple[float, float] = (0.6, 2.25)
        """Maximum reference speed [m/s]."""
        acc_cap: tuple[float, float] = (0.6, 3.0)
        """Maximum reference acceleration [m/s^2]."""
        pos_z_cap: tuple[float, float] = (0.39, 1.5)
        """Maximum vertical excursion from the trajectory centre [m]."""
        vel_z_cap: tuple[float, float] = (0.3, 1.14)
        """Maximum vertical reference speed [m/s]."""
        acc_z_cap: tuple[float, float] = (0.39, 1.5)
        """Maximum vertical reference acceleration [m/s^2]."""
        omega_max: tuple[float, float] = (0.40, 0.85)
        """Upper end of the fundamental angular frequency range [rad/s]."""

    ranges: Ranges = Ranges()
    """Trajectory limits, interpolated by :attr:`difficulty`."""

    # note: each visualizer gets its own marker config rather than a ``.replace()`` of the shared
    # SPHERE_MARKER_CFG, because ``replace`` is shallow and the three would end up aliasing one
    # marker dict -- and mutating the module-level default along with it.
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = _sphere_marker_cfg(
        "/Visuals/Command/trajectory_goal", radius=0.15, color=(0.0, 1.0, 0.0)
    )
    """Marker drawn at the reference position, oriented by the reference yaw."""

    path_visualizer_cfg: VisualizationMarkersCfg = _sphere_marker_cfg(
        "/Visuals/Command/trajectory_path", radius=0.04, color=(0.2, 0.4, 1.0)
    )
    """Markers drawn along the whole reference path of the episode."""

    lookahead_visualizer_cfg: VisualizationMarkersCfg = _sphere_marker_cfg(
        "/Visuals/Command/trajectory_lookahead", radius=0.07, color=(1.0, 0.85, 0.0)
    )
    """Markers drawn at the look-ahead reference positions."""

    termination_flash_steps: int = 5
    """Number of policy steps an early-termination flash stays on screen. Zero disables it.

    A terminating environment is reset on the same step, so the flash is latched at the pose the
    drone failed at and held afterwards; without the hold it would be a single frame and, at 10 Hz,
    essentially invisible.
    """

    termination_flash_visualizer_cfg: VisualizationMarkersCfg = _flash_marker_cfg(
        "/Visuals/Command/termination_flash",
        radius=0.6,
        colors={
            "tracking_divergence": (1.0, 0.0, 0.0),
            "bad_orientation": (1.0, 0.45, 0.0),
            "crash_floor": (0.75, 0.0, 0.9),
            "crash_ceiling": (0.0, 0.7, 1.0),
        },
        opacity=0.45,
    )
    """Translucent spheres marking where an environment terminated early.

    Only terminations that are *not* time-outs are drawn, so the marker separates a genuine failure
    from an episode that simply ran to :attr:`~ManagerBasedRLEnvCfg.episode_length_s`. Each key
    names a termination term and gives it its own colour; a term that is active but unlisted falls
    back to the first prototype. Keys that name no active term are simply never selected, so the
    default palette is safe to leave in place for a task with a different termination set.
    """
