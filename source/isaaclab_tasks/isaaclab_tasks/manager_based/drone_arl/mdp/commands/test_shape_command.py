# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic geometric test references (square, circle, line, zigzag) for a multirotor.

These are the simulator-side equivalents of the shapes flown on the vehicle by
``dji_tap_and_go/generate_test_trajectory.py``, so a policy can be scored on the same figures in
both places. They replace the randomly sampled sum-of-sinusoids reference of
:class:`~isaaclab_tasks.manager_based.drone_arl.mdp.commands.DroneTrajectoryCommand`; everything
else about the command -- tensor layout, look-ahead, metrics, debug visualization -- is inherited
unchanged, so the existing observation and reward terms work against it as-is.

**How the shapes are built.** Each figure is defined as a closed polyline traversed at constant
speed, resampled uniformly in arc length, and then projected onto a truncated Fourier series::

    p_k(t) = c_k + sum_i A_ki * sin(w_ki * t + phi_ki),   w_ki = i * 2*pi / period

That is the same representation the parent class evaluates, which is why the closed-form velocity
and acceleration come for free and no new evaluation code is needed.

The truncation is deliberate, not a compromise. An ideal polyline has infinite acceleration at its
corners, which is not a reference any vehicle can track and not what the deployment flies either:
on the vehicle these waypoints go through a polynomial (min-snap) generator, which likewise rounds
the corners. :attr:`~DroneTestShapeCommandCfg.num_harmonics` is the roundedness knob -- one
harmonic gives the smoothest figure the shape admits (a circle becomes exact, a square becomes its
inscribed ellipse), and more harmonics sharpen the corners toward the ideal polyline.

A consequence worth knowing: only the *mean* speed over a lap equals
:attr:`~DroneTestShapeCommandCfg.speed`. A truncated square slows into its corners and speeds up
along its edges, exactly as a real flown figure does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .trajectory_command import DroneTrajectoryCommand, sample_yaw_params

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import DroneTestShapeCommandCfg


SHAPES = ("square", "circle", "line", "zigzag")
"""The shapes :func:`shape_polyline` knows how to build."""


"""
Shape math (simulator-free).
"""


def shape_polyline(shape: str, size: float, num_points: int = 8) -> torch.Tensor:
    """Return the vertices of one closed lap of a test shape, centred on the origin.

    The vertices mirror ``generate_test_trajectory.generate_offsets`` in the ROS package, with two
    deliberate differences. The figure is centred on the origin rather than hung off the drone's
    current position -- in simulation the drone is spawned onto the reference, so the absolute
    placement carries no information and centring keeps the viewport tidy. And the open shapes
    (``line``, ``zigzag``) are closed by retracing, since the reference here is periodic and has to
    return to its start.

    Args:
        shape: One of :data:`SHAPES`.
        size: Characteristic size [m] -- side length, diameter, or length along x.
        num_points: Number of segments in the ``zigzag``. Unused by the other shapes.

    Returns:
        Vertices [m], shape ``(num_vertices, 3)``, with the last vertex equal to the first.

    Raises:
        ValueError: If :paramref:`shape` is not one of :data:`SHAPES`.
    """
    half = size / 2.0
    if shape == "square":
        corners = [(half, half), (half, -half), (-half, -half), (-half, half), (half, half)]
        points = [(x, y, 0.0) for x, y in corners]
    elif shape == "circle":
        # A polygon fine enough that its own discretization is far below the Fourier truncation.
        n = 256
        points = [
            (half * math.cos(2.0 * math.pi * i / n), half * math.sin(2.0 * math.pi * i / n), 0.0) for i in range(n + 1)
        ]
    elif shape == "line":
        points = [(-half, 0.0, 0.0), (half, 0.0, 0.0), (-half, 0.0, 0.0)]
    elif shape == "zigzag":
        # Out along the zigzag and back along it, so the lap closes. The amplitude is size/4 and the
        # advance per segment size/num_points, matching the ROS shape.
        amplitude = size / 4.0
        out = [(-half + size * i / num_points, amplitude if i % 2 else -amplitude, 0.0) for i in range(num_points + 1)]
        points = out + out[-2::-1]
    else:
        raise ValueError(f"Unknown shape '{shape}'. Expected one of {SHAPES}.")
    return torch.tensor(points, dtype=torch.float64)


def resample_uniform_arc_length(vertices: torch.Tensor, num_samples: int) -> tuple[torch.Tensor, float]:
    """Resample a closed polyline at equal arc-length spacing.

    Args:
        vertices: Polyline vertices [m], shape ``(num_vertices, 3)``, first vertex repeated last.
        num_samples: Number of samples to draw over one lap.

    Returns:
        A tuple ``(samples, perimeter)``: the resampled points [m] with shape
        ``(num_samples, 3)``, and the total arc length of one lap [m].
    """
    segment = vertices[1:] - vertices[:-1]
    lengths = torch.linalg.norm(segment, dim=-1)
    cumulative = torch.cat([torch.zeros(1, dtype=lengths.dtype), torch.cumsum(lengths, dim=0)])
    perimeter = float(cumulative[-1])

    # Exclude the endpoint: sample i sits at i/num_samples of the lap, so the lap closes without
    # duplicating its first point, which is what the FFT below expects.
    targets = torch.arange(num_samples, dtype=vertices.dtype) * (perimeter / num_samples)
    index = torch.searchsorted(cumulative, targets, right=True).clamp(1, len(lengths)) - 1
    alpha = ((targets - cumulative[index]) / lengths[index].clamp(min=1e-12)).unsqueeze(-1)
    return vertices[index] + alpha * segment[index], perimeter


def fourier_coefficients(samples: torch.Tensor, num_harmonics: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project one lap of a path onto a truncated sine series.

    The parent class evaluates ``c + sum_i A_i sin(w_i t + phi_i)``, so the real FFT coefficients
    are converted into that form: a term ``2/N * Re(X_k e^{i k theta})`` is
    ``2|X_k|/N * cos(k theta + arg X_k)``, and a cosine is a sine advanced by ``pi/2``.

    Args:
        samples: One lap sampled uniformly in arc length [m], shape ``(num_samples, 3)``.
        num_harmonics: Number of harmonics to keep.

    Returns:
        A tuple ``(amplitude, phase, center)`` with shapes ``(3, num_harmonics)``,
        ``(3, num_harmonics)`` and ``(3,)``, in [m], [rad] and [m].
    """
    num_samples = samples.shape[0]
    spectrum = torch.fft.rfft(samples, dim=0)  # (num_samples // 2 + 1, 3)
    center = (spectrum[0].real / num_samples).to(torch.float64)

    keep = spectrum[1 : num_harmonics + 1]  # (num_harmonics, 3)
    amplitude = (2.0 * keep.abs() / num_samples).T.contiguous()
    phase = (torch.angle(keep) + math.pi / 2.0).T.contiguous()
    return amplitude.to(torch.float64), phase.to(torch.float64), center


def required_harmonics(shape: str, num_points: int = 8) -> int:
    """Return the number of harmonics a shape needs before it stops being that shape.

    Truncating below this does not merely round the corners, it destroys the figure. The zigzag is
    the case that forces this: its ``num_points`` alternations put its content near harmonic
    ``num_points``, so the default three harmonics of the parent class reconstruct it as a plain
    line with no zigzag left in it at all.

    Args:
        shape: One of :data:`SHAPES`.
        num_points: Number of segments in the ``zigzag``. Unused by the other shapes.

    Returns:
        The minimum useful harmonic count for the shape.
    """
    # A circle is a single harmonic exactly. A line and a square are recognisable from their first
    # harmonic (an oscillation and an ellipse); harmonics beyond that sharpen the corners, which is
    # what ``num_harmonics`` is for. Only the zigzag has a floor set by its own geometry.
    return num_points if shape == "zigzag" else 1


def shape_harmonics(
    shape: str, size: float, speed: float, num_harmonics: int, num_points: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Build the harmonic representation of one test shape.

    The truncated series is rescaled per axis so the reconstructed figure spans the same extent as
    the ideal polyline, and the lap period is then set from the length of the *reconstructed* path.
    Without both corrections the truncation would quietly shrink the figure and slow it down -- a
    one-harmonic line reaches only 81 % of its length, which would make ``size`` and ``speed`` mean
    something other than what they say.

    Args:
        shape: One of :data:`SHAPES`.
        size: Characteristic size [m].
        speed: Target mean speed along the path [m/s].
        num_harmonics: Number of harmonics to keep. Raised to :func:`required_harmonics` when the
            shape needs more than this to survive truncation.
        num_points: Number of segments in the ``zigzag``. Unused by the other shapes.

    Returns:
        A tuple ``(amplitude, omega, phase, center, period)`` with shapes ``(3, num_kept)``
        for the first three, ``(3,)`` for the centre, in [m], [rad/s], [rad], [m] and [s].
    """
    num_kept = max(num_harmonics, required_harmonics(shape, num_points))
    vertices = shape_polyline(shape, size, num_points)
    # Enough samples that the projection is exact to float precision for any sane harmonic count.
    samples, _ = resample_uniform_arc_length(vertices, num_samples=4096)
    amplitude, phase, center = fourier_coefficients(samples, num_kept)

    # -- restore the figure's extent, which truncation shrinks.
    harmonic = torch.arange(1, num_kept + 1, dtype=torch.float64)
    theta = torch.linspace(0.0, 2.0 * math.pi, 2049, dtype=torch.float64)[:-1]
    angle = harmonic * theta.reshape(-1, 1, 1) + phase  # (num_theta, 3, num_kept)
    path = center + (amplitude * torch.sin(angle)).sum(dim=-1)

    target_extent = samples.max(dim=0).values - samples.min(dim=0).values
    actual_extent = path.max(dim=0).values - path.min(dim=0).values
    # A flat axis (z everywhere, y on the line) has nothing to rescale and must be left alone.
    scale = torch.where(actual_extent > 1e-9, target_extent / actual_extent.clamp(min=1e-9), torch.ones(()))
    amplitude = amplitude * scale.unsqueeze(-1)

    # -- set the period from the rescaled path's own length, so the mean speed is exactly ``speed``.
    path = center + (amplitude * torch.sin(angle)).sum(dim=-1)
    lap_length = float(torch.linalg.norm(path.roll(-1, dims=0) - path, dim=-1).sum())
    period = lap_length / speed
    omega = (2.0 * math.pi / period) * harmonic.unsqueeze(0).expand(3, -1).contiguous()
    return amplitude, omega, phase, center, period


class DroneTestShapeCommand(DroneTrajectoryCommand):
    """Command generator producing a deterministic geometric test figure per environment.

    Shapes are dealt out over the environments by index -- environment ``i`` flies
    ``cfg.shapes[i % len(cfg.shapes)]`` -- so a single run exercises every figure at once and each
    one lands in the same environment on every run, which is what makes two runs comparable.

    The command tensor layout is inherited verbatim from
    :class:`~isaaclab_tasks.manager_based.drone_arl.mdp.commands.DroneTrajectoryCommand`.
    """

    cfg: DroneTestShapeCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: DroneTestShapeCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.

        Raises:
            ValueError: If :attr:`~DroneTestShapeCommandCfg.shapes` is empty or names an unknown
                shape, or if :attr:`~DroneTestShapeCommandCfg.speed` is not positive.
        """
        if not cfg.shapes:
            raise ValueError("shapes must name at least one shape")
        unknown = [s for s in cfg.shapes if s not in SHAPES]
        if unknown:
            raise ValueError(f"Unknown shape(s) {unknown}. Expected values from {SHAPES}.")
        if cfg.speed <= 0.0:
            raise ValueError("speed must be positive")

        # Build every shape's harmonics first: they are pure geometry and need no simulator state,
        # and their widths decide how wide the parent's buffers have to be.
        built = [shape_harmonics(shape, cfg.size, cfg.speed, cfg.num_harmonics, cfg.num_points) for shape in cfg.shapes]
        width = max(amplitude.shape[-1] for amplitude, *_ in built)
        # The parent sizes its ``(num_envs, 3, num_harmonics)`` buffers from this, so a shape that
        # needed more harmonics than asked for (see :func:`required_harmonics`) has to widen it.
        cfg.num_harmonics = width

        super().__init__(cfg, env)

        # Index the harmonics by environment. The shapes are fixed for the lifetime of the run, so
        # this happens once rather than on every reset.
        table_amplitude, table_omega, table_phase, table_center, periods = [], [], [], [], []
        for amplitude, omega, phase, center, period in built:
            pad = width - amplitude.shape[-1]
            if pad:
                # Zero amplitude contributes nothing, so a narrower shape simply ignores the tail.
                amplitude = torch.nn.functional.pad(amplitude, (0, pad))
                omega = torch.nn.functional.pad(omega, (0, pad))
                phase = torch.nn.functional.pad(phase, (0, pad))
            table_amplitude.append(amplitude)
            table_omega.append(omega)
            table_phase.append(phase)
            table_center.append(center)
            periods.append(period)

        def stack(values: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(values).to(device=self.device, dtype=torch.float32)

        self._table_amplitude = stack(table_amplitude)
        self._table_omega = stack(table_omega)
        self._table_phase = stack(table_phase)
        self._table_center = stack(table_center)
        self._periods = torch.tensor(periods, device=self.device, dtype=torch.float32)

        # Environment -> shape index. Fixed, so a given environment flies the same figure all run.
        self._shape_idx = torch.arange(self.num_envs, device=self.device) % len(cfg.shapes)
        self.sample_parameters(slice(None))

    def __str__(self) -> str:
        msg = "DroneTestShapeCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tSize: {self.cfg.size:.2f} m at {self.cfg.speed:.2f} m/s\n"
        msg += f"\tHarmonics per axis: {self.cfg.num_harmonics}\n"
        msg += f"\tLook-ahead times: {self.cfg.lookahead_times_s} s\n"
        msg += f"\tYaw mode: {self.cfg.yaw_mode}\n"
        for i, shape in enumerate(self.cfg.shapes):
            count = int((self._shape_idx == i).sum())
            msg += f"\t{shape:>8}: {self._periods[i]:6.2f} s per lap, {count} envs\n"
        return msg

    """
    Operations.
    """

    def sample_parameters(self, env_ids: Sequence[int] | torch.Tensor | slice):
        """Assign each environment the harmonics of its shape.

        Overrides the random sum-of-sinusoids draw of the parent. Nothing is actually sampled: the
        figure an environment flies is fixed by its index, which is the point of a test reference.
        The yaw reference is still drawn through the parent's sampler, so ``yaw_mode`` behaves the
        same way it does in training.

        Args:
            env_ids: Indices of the environments to assign, or ``slice(None)`` for all.
        """
        idx = self._shape_idx[env_ids]
        self._amplitude[env_ids] = self._table_amplitude[idx]
        self._omega[env_ids] = self._table_omega[idx]
        self._phase[env_ids] = self._table_phase[idx]
        self._center[env_ids] = self._table_center[idx]

        num = idx.shape[0]
        yaw_0, yaw_amplitude, yaw_omega, yaw_phase = sample_yaw_params(
            num_envs=num,
            yaw_mode=self.cfg.yaw_mode,
            device=self.device,
            amplitude_range=self.cfg.yaw_amplitude_range,
            omega_range=self.cfg.yaw_omega_range,
            max_yaw_rate=self.cfg.yaw_max_rate,
        )
        self._yaw_0[env_ids] = yaw_0
        self._yaw_amplitude[env_ids] = yaw_amplitude
        self._yaw_omega[env_ids] = yaw_omega
        self._yaw_phase[env_ids] = yaw_phase

        self._difficulty[env_ids] = 1.0
        self._path_vis_dirty = True

    """
    Debug visualization.
    """

    def _path_vis_time_grid(self, n: int) -> torch.Tensor:
        """Draw exactly one lap per environment, rather than however many fit in an episode."""
        fraction = torch.linspace(0.0, 1.0, self.cfg.num_path_vis_points, device=self.device)
        return self._periods[self._shape_idx[:n]].unsqueeze(-1) * fraction.unsqueeze(0)
