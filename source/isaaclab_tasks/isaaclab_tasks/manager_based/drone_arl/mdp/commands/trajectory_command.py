# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing a time-parameterized trajectory command generator for multirotors.

The reference is a per-axis sum of sinusoids, which gives closed-form position, velocity and
acceleration at any time without integrating anything::

    p_k(t) = c_k + sum_i A_ki * sin(w_ki * t + phi_ki)
    v_k(t) =       sum_i A_ki * w_ki    * cos(w_ki * t + phi_ki)
    a_k(t) =     - sum_i A_ki * w_ki**2 * sin(w_ki * t + phi_ki)

The sampling math is exposed as free functions (:func:`sample_trajectory_params`,
:func:`evaluate_trajectory`) that depend only on :mod:`torch`, so the trajectory generator can be
validated without launching the simulator -- see ``scripts/demos/trajectory_command_math_test.py``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands_cfg import DroneTrajectoryCommandCfg


"""
Trajectory math (simulator-free).
"""


@dataclass(frozen=True)
class TrajectoryLimits:
    """Resolved sup-bounds on a sampled reference trajectory.

    Every field is an absolute upper bound that the sampler enforces exactly (in the sup-norm over
    all time), except :attr:`vel_rms_target`, which shapes the typical speed rather than capping it.
    """

    vel_rms_target: float
    """Target root-mean-square speed of the reference [m/s].

    Amplitudes are scaled to hit this exactly, unless doing so would violate one of the caps below.
    It is therefore the primary knob on how aggressive the reference is; the caps only bound the
    worst case.
    """
    pos_cap: float
    """Maximum excursion from the trajectory centre [m]."""
    vel_cap: float
    """Maximum reference speed [m/s]."""
    acc_cap: float
    """Maximum reference acceleration [m/s^2]."""
    pos_z_cap: float
    """Maximum vertical excursion from the trajectory centre [m]."""
    vel_z_cap: float
    """Maximum vertical reference speed [m/s]."""
    acc_z_cap: float
    """Maximum vertical reference acceleration [m/s^2]."""
    omega_max: float
    """Upper end of the fundamental angular frequency range [rad/s]."""

    @classmethod
    def from_ranges(cls, ranges, difficulty: float) -> TrajectoryLimits:
        """Linearly interpolate every limit between its easy and hard end.

        Args:
            ranges: Object exposing one ``(easy, hard)`` tuple per field of this class.
            difficulty: Interpolation factor, clamped to ``[0, 1]``.

        Returns:
            The resolved limits at the requested difficulty.
        """
        d = min(max(difficulty, 0.0), 1.0)
        return cls(**{f: (v := getattr(ranges, f))[0] + d * (v[1] - v[0]) for f in cls.__dataclass_fields__})


def sample_trajectory_params(
    num_envs: int,
    limits: TrajectoryLimits,
    device: str | torch.device,
    num_harmonics: int = 3,
    axis_weights: tuple[float, float, float] = (1.0, 1.0, 0.35),
    harmonic_multiplier_ranges: Sequence[tuple[float, float]] = ((0.8, 1.2), (1.7, 2.4), (2.8, 3.6)),
    harmonic_amplitude_ranges: Sequence[tuple[float, float]] = ((0.6, 1.0), (0.1, 0.45), (0.0, 0.2)),
    omega_min: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample sum-of-sinusoid trajectory parameters that provably respect ``limits``.

    A single fundamental frequency is drawn per environment and shared by all three axes, then
    multiplied by per-axis, per-harmonic factors. Keeping one tempo makes the three axes
    rhythmically coherent, so the path reads as a flight path rather than as noise, while the
    non-integer multipliers keep it quasi-periodic (there is no short loop for a policy to memorize).

    Amplitudes are then rescaled by a single closed-form factor: first up or down so the trajectory
    hits :attr:`~TrajectoryLimits.vel_rms_target`, then down again if that would breach a cap.
    Because ``|sin| <= 1``, the per-axis sums ``sum_i A_ki``, ``sum_i A_ki*w_ki`` and
    ``sum_i A_ki*w_ki**2`` are exact sup-bounds on position, velocity and acceleration, and all
    three are linear in the amplitude, so one scale factor enforces every cap simultaneously with no
    rejection sampling.

    Args:
        num_envs: Number of environments to sample for.
        limits: Sup-bounds to enforce.
        device: Device to allocate the parameter tensors on.
        num_harmonics: Number of sinusoids summed per axis.
        axis_weights: Per-axis amplitude weights applied before normalization. The vertical axis is
            weighted down so the reference stays mostly horizontal.
        harmonic_multiplier_ranges: Sampling range of the frequency multiplier of each harmonic.
        harmonic_amplitude_ranges: Sampling range of the raw amplitude of each harmonic.
        omega_min: Lower end of the fundamental angular frequency range [rad/s].

    Returns:
        A tuple ``(amplitude, omega, phase, center)`` with shapes ``(num_envs, 3, num_harmonics)``
        for the first three and ``(num_envs, 3)`` for the centre. Amplitudes are in [m], frequencies
        in [rad/s], phases in [rad] and the centre in [m]. The centre is chosen so ``p(0) == 0``.
    """
    if len(harmonic_multiplier_ranges) < num_harmonics or len(harmonic_amplitude_ranges) < num_harmonics:
        raise ValueError(
            f"Need at least {num_harmonics} multiplier and amplitude ranges, got"
            f" {len(harmonic_multiplier_ranges)} and {len(harmonic_amplitude_ranges)}."
        )

    shape = (num_envs, 3, num_harmonics)

    def _uniform(lo: float, hi: float, size: tuple[int, ...]) -> torch.Tensor:
        return torch.rand(size, device=device) * (hi - lo) + lo

    # -- fundamental frequency, shared across axes within an environment
    omega_0 = _uniform(omega_min, limits.omega_max, (num_envs, 1, 1))
    multipliers = torch.cat(
        [_uniform(*harmonic_multiplier_ranges[i], (num_envs, 3, 1)) for i in range(num_harmonics)], dim=-1
    )
    omega = omega_0 * multipliers

    # -- raw amplitudes, weighted per axis
    amplitude = torch.cat(
        [_uniform(*harmonic_amplitude_ranges[i], (num_envs, 3, 1)) for i in range(num_harmonics)], dim=-1
    )
    amplitude = amplitude * torch.tensor(axis_weights, device=device).view(1, 3, 1)

    phase = _uniform(-math.pi, math.pi, shape)

    # -- enforce the isotropic limits, then the tighter vertical ones
    amplitude = amplitude * _limit_scale(
        amplitude, omega, limits.pos_cap, limits.vel_cap, limits.acc_cap, limits.vel_rms_target
    ).view(num_envs, 1, 1)
    amplitude[:, 2:3, :] = amplitude[:, 2:3, :] * _limit_scale(
        amplitude[:, 2:3, :], omega[:, 2:3, :], limits.pos_z_cap, limits.vel_z_cap, limits.acc_z_cap
    ).view(num_envs, 1, 1)

    # -- centre the trajectory so that it starts at the origin of the environment
    center = -(amplitude * torch.sin(phase)).sum(dim=-1)

    return amplitude, omega, phase, center


def _limit_scale(
    amplitude: torch.Tensor,
    omega: torch.Tensor,
    pos_cap: float,
    vel_cap: float,
    acc_cap: float,
    vel_rms_target: float | None = None,
) -> torch.Tensor:
    """Compute the per-environment amplitude scale that enforces every sup-bound at once.

    Args:
        amplitude: Harmonic amplitudes [m], shape ``(num_envs, num_axes, num_harmonics)``.
        omega: Harmonic angular frequencies [rad/s], same shape as :paramref:`amplitude`.
        pos_cap: Maximum excursion from the centre [m].
        vel_cap: Maximum speed [m/s].
        acc_cap: Maximum acceleration [m/s^2].
        vel_rms_target: Optional target root-mean-square speed [m/s]. When given, the trajectory is
            scaled *to* this value -- up as well as down -- and only then clipped by the caps, so it
            is the knob that actually sets how aggressive the reference is. When omitted the scale
            can only shrink the trajectory, which is what the vertical pass needs.

    Returns:
        Scale factors, shape ``(num_envs,)``. Bounded above by 1 when :paramref:`vel_rms_target` is
        omitted.
    """
    eps = 1e-6
    pos_bound = torch.linalg.norm(amplitude.sum(dim=-1), dim=-1)
    vel_bound = torch.linalg.norm((amplitude * omega).sum(dim=-1), dim=-1)
    acc_bound = torch.linalg.norm((amplitude * omega**2).sum(dim=-1), dim=-1)

    cap_scale = torch.minimum(
        torch.minimum(pos_cap / (pos_bound + eps), vel_cap / (vel_bound + eps)),
        acc_cap / (acc_bound + eps),
    )
    if vel_rms_target is None:
        return cap_scale.clamp(max=1.0)

    # For independent uniform phases the exact RMS speed is sqrt(0.5 * sum_ki (A_ki*w_ki)^2).
    # The raw amplitudes are arbitrary, so scale to the target first and let the caps clip it.
    vel_rms = torch.sqrt(0.5 * ((amplitude * omega) ** 2).sum(dim=(-1, -2)))
    return torch.minimum(vel_rms_target / (vel_rms + eps), cap_scale)


def evaluate_trajectory(
    amplitude: torch.Tensor,
    omega: torch.Tensor,
    phase: torch.Tensor,
    center: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate reference position, velocity and acceleration at the given times.

    Args:
        amplitude: Harmonic amplitudes [m], shape ``(num_envs, 3, num_harmonics)``.
        omega: Harmonic angular frequencies [rad/s], same shape as :paramref:`amplitude`.
        phase: Harmonic phase offsets [rad], same shape as :paramref:`amplitude`.
        center: Trajectory centre [m], shape ``(num_envs, 3)``.
        time: Evaluation time [s], shape ``(num_envs,)`` for a single time per environment or
            ``(num_envs, num_times)`` to evaluate several times per environment at once.

    Returns:
        A tuple ``(position, velocity, acceleration)`` in [m], [m/s] and [m/s^2]. Shapes are
        ``(num_envs, 3)`` for a 1-D :paramref:`time` and ``(num_envs, num_times, 3)`` for a 2-D one.
    """
    squeeze = time.dim() == 1
    time = time.unsqueeze(-1) if squeeze else time
    # (num_envs, num_times, 3, num_harmonics)
    angle = omega.unsqueeze(1) * time[..., None, None] + phase.unsqueeze(1)
    sin, cos = torch.sin(angle), torch.cos(angle)

    amplitude, omega = amplitude.unsqueeze(1), omega.unsqueeze(1)
    position = center.unsqueeze(1) + (amplitude * sin).sum(dim=-1)
    velocity = (amplitude * omega * cos).sum(dim=-1)
    acceleration = -(amplitude * omega**2 * sin).sum(dim=-1)

    if squeeze:
        return position.squeeze(1), velocity.squeeze(1), acceleration.squeeze(1)
    return position, velocity, acceleration


def sample_yaw_params(
    num_envs: int,
    yaw_mode: str,
    device: str | torch.device,
    amplitude_range: tuple[float, float] = (0.3, 1.2),
    omega_range: tuple[float, float] = (0.1, 0.4),
    max_yaw_rate: float = 0.6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample the yaw reference parameters ``psi(t) = psi_0 + A*sin(w*t + phi)``.

    Args:
        num_envs: Number of environments to sample for.
        yaw_mode: One of ``"fixed"`` (always zero), ``"random_constant"`` (uniform per episode,
            zero rate) or ``"sweep"`` (sinusoidal with a bounded rate).
        device: Device to allocate the parameter tensors on.
        amplitude_range: Sampling range of the sweep amplitude [rad].
        omega_range: Sampling range of the sweep angular frequency [rad/s].
        max_yaw_rate: Upper bound on the yaw rate of the sweep [rad/s]; the amplitude is scaled down
            to respect it.

    Returns:
        A tuple ``(yaw_0, amplitude, omega, phase)``, each of shape ``(num_envs,)``, in [rad],
        [rad], [rad/s] and [rad].
    """
    zeros = torch.zeros(num_envs, device=device)
    if yaw_mode == "fixed":
        return zeros, zeros.clone(), zeros.clone(), zeros.clone()

    yaw_0 = torch.rand(num_envs, device=device) * (2.0 * math.pi) - math.pi
    if yaw_mode == "random_constant":
        return yaw_0, zeros, zeros.clone(), zeros.clone()
    if yaw_mode != "sweep":
        raise ValueError(f"Unknown yaw mode '{yaw_mode}'. Expected 'fixed', 'random_constant' or 'sweep'.")

    amplitude = torch.rand(num_envs, device=device) * (amplitude_range[1] - amplitude_range[0]) + amplitude_range[0]
    omega = torch.rand(num_envs, device=device) * (omega_range[1] - omega_range[0]) + omega_range[0]
    # |psi_dot| <= A*w, so scale the amplitude down until the rate bound holds.
    amplitude = amplitude * (max_yaw_rate / (amplitude * omega + 1e-6)).clamp(max=1.0)
    phase = torch.rand(num_envs, device=device) * (2.0 * math.pi) - math.pi
    return yaw_0, amplitude, omega, phase


def evaluate_yaw(
    yaw_0: torch.Tensor,
    amplitude: torch.Tensor,
    omega: torch.Tensor,
    phase: torch.Tensor,
    time: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the yaw reference and its rate at the given times.

    Args:
        yaw_0: Yaw offset [rad], shape ``(num_envs,)``.
        amplitude: Sweep amplitude [rad], shape ``(num_envs,)``.
        omega: Sweep angular frequency [rad/s], shape ``(num_envs,)``.
        phase: Sweep phase offset [rad], shape ``(num_envs,)``.
        time: Evaluation time [s], shape ``(num_envs,)``.

    Returns:
        A tuple ``(yaw, yaw_rate)`` in [rad] and [rad/s], each of shape ``(num_envs,)``. The yaw is
        wrapped to ``(-pi, pi]``.
    """
    angle = omega * time + phase
    yaw = math_utils.wrap_to_pi(yaw_0 + amplitude * torch.sin(angle))
    yaw_rate = amplitude * omega * torch.cos(angle)
    return yaw, yaw_rate


"""
Command term.
"""


class DroneTrajectoryCommand(CommandTerm):
    """Command generator producing a smooth, time-varying position reference for a multirotor.

    Each environment gets its own randomly sampled sum-of-sinusoid trajectory (see
    :func:`sample_trajectory_params`), sampled once per episode and then swept out in time. The
    command exposes the reference at the current time together with a short look-ahead, so a policy
    can lead the reference instead of chasing it.

    The command tensor is laid out as ``(num_envs, 11 + 3 * num_lookahead)``:

    ==============================  ==========================================================
    Slice                           Contents
    ==============================  ==========================================================
    ``[:, 0:3]``                    reference position [m], in the environment-local world frame
    ``[:, 3:6]``                    reference velocity [m/s], world frame
    ``[:, 6:9]``                    reference acceleration [m/s^2], world frame
    ``[:, 9]``                      reference yaw [rad]
    ``[:, 10]``                     reference yaw rate [rad/s]
    ``[:, 11:]``                    look-ahead positions [m], environment-local world frame
    ==============================  ==========================================================

    Positions are environment-local (that is, ``scene.env_origins`` is *not* added), which matches
    the convention of the ``distance_to_goal_*`` reward terms so those work against this command
    unchanged. Policy observations are expected to consume the dedicated observation terms rather
    than this tensor directly, since those transform the errors into the vehicle frame.
    """

    cfg: DroneTrajectoryCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: DroneTrajectoryCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        super().__init__(cfg, env)

        self.robot = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        num_envs, device = self.num_envs, self.device
        harmonics = cfg.num_harmonics
        self._lookahead_times = torch.tensor(cfg.lookahead_times_s, device=device)
        self.num_lookahead = len(cfg.lookahead_times_s)

        # -- trajectory parameters
        self._amplitude = torch.zeros(num_envs, 3, harmonics, device=device)
        self._omega = torch.zeros_like(self._amplitude)
        self._phase = torch.zeros_like(self._amplitude)
        self._center = torch.zeros(num_envs, 3, device=device)
        self._yaw_0 = torch.zeros(num_envs, device=device)
        self._yaw_amplitude = torch.zeros_like(self._yaw_0)
        self._yaw_omega = torch.zeros_like(self._yaw_0)
        self._yaw_phase = torch.zeros_like(self._yaw_0)

        # -- episode phase clock. ``episode_length_buf`` cannot be used: it is zeroed *after*
        #    the command manager is reset, so it is stale exactly when the trajectory is sampled.
        self._time = torch.zeros(num_envs, device=device)
        self._difficulty = torch.zeros(num_envs, device=device)
        # Set by :func:`~isaaclab_tasks.manager_based.drone_arl.mdp.events.reset_root_state_on_trajectory`
        # to hand over sampling, so the drone can be spawned on a reference that does not exist yet
        # when the reset events run.
        self._sampled_by_event = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # -- evaluated reference (public: read by observations, rewards and evaluation scripts)
        self.pos_ref_w = torch.zeros(num_envs, 3, device=device)
        self.vel_ref_w = torch.zeros_like(self.pos_ref_w)
        self.acc_ref_w = torch.zeros_like(self.pos_ref_w)
        self.yaw_ref = torch.zeros(num_envs, device=device)
        self.yaw_rate_ref = torch.zeros_like(self.yaw_ref)
        self.lookahead_w = torch.zeros(num_envs, self.num_lookahead, 3, device=device)
        self._command = torch.zeros(num_envs, 11 + 3 * self.num_lookahead, device=device)

        # -- metrics. ``CommandTerm.reset`` logs the value *at reset*, which for a moving reference
        #    is just wherever the trajectory happened to end. The running means are the useful ones.
        self.metrics["position_error"] = torch.zeros(num_envs, device=device)
        self.metrics["position_error_mean"] = torch.zeros(num_envs, device=device)
        self.metrics["velocity_error_mean"] = torch.zeros(num_envs, device=device)
        self.metrics["yaw_error_mean"] = torch.zeros(num_envs, device=device)
        self.metrics["ref_speed_mean"] = torch.zeros(num_envs, device=device)
        self.metrics["difficulty"] = torch.zeros(num_envs, device=device)
        self._error_sum = torch.zeros(num_envs, device=device)
        self._vel_error_sum = torch.zeros(num_envs, device=device)
        self._yaw_error_sum = torch.zeros(num_envs, device=device)
        self._ref_speed_sum = torch.zeros(num_envs, device=device)
        self._step_count = torch.zeros(num_envs, device=device)

        # -- debug visualization
        self._path_vis_times = torch.linspace(0.0, env.cfg.episode_length_s, cfg.num_path_vis_points, device=device)
        self._num_vis_envs = min(cfg.max_vis_envs, num_envs)
        self._path_vis_dirty = True

    def __str__(self) -> str:
        msg = "DroneTrajectoryCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tHarmonics per axis: {self.cfg.num_harmonics}\n"
        msg += f"\tLook-ahead times: {self.cfg.lookahead_times_s} s\n"
        msg += f"\tYaw mode: {self.cfg.yaw_mode}\n"
        msg += f"\tDifficulty: {self.cfg.difficulty}\n"
        return msg

    """
    Properties.
    """

    @property
    def command(self) -> torch.Tensor:
        """The reference trajectory sample. Shape is (num_envs, 11 + 3 * num_lookahead)."""
        return self._command

    """
    Operations.
    """

    def sample_parameters(self, env_ids: Sequence[int] | torch.Tensor):
        """Draw a new trajectory for the given environments.

        This is public so that a reset event can sample the trajectory before writing the robot
        state, which lets the drone be spawned on the reference. The event must then set
        :attr:`_sampled_by_event` so the command term does not immediately re-sample.

        Args:
            env_ids: Indices of the environments to sample for.
        """
        limits = TrajectoryLimits.from_ranges(self.cfg.ranges, self.cfg.difficulty)
        amplitude, omega, phase, center = sample_trajectory_params(
            num_envs=len(env_ids),
            limits=limits,
            device=self.device,
            num_harmonics=self.cfg.num_harmonics,
            axis_weights=self.cfg.axis_weights,
            harmonic_multiplier_ranges=self.cfg.harmonic_multiplier_ranges,
            harmonic_amplitude_ranges=self.cfg.harmonic_amplitude_ranges,
            omega_min=self.cfg.omega_min,
        )
        self._amplitude[env_ids] = amplitude
        self._omega[env_ids] = omega
        self._phase[env_ids] = phase
        self._center[env_ids] = center

        yaw_0, yaw_amplitude, yaw_omega, yaw_phase = sample_yaw_params(
            num_envs=len(env_ids),
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

        self._difficulty[env_ids] = self.cfg.difficulty
        self._path_vis_dirty = True

    def _evaluate_position(
        self, env_ids: Sequence[int] | torch.Tensor | slice, time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate the positional reference for the given environments at the given times.

        This is the single place the path geometry is defined, so a subclass can swap the shape of
        the reference without reimplementing the buffer bookkeeping in :meth:`evaluate_at`, the
        look-ahead, or the debug visualization -- all of which route through here.

        Args:
            env_ids: Indices of the environments to evaluate, or ``slice(None)`` for all.
            time: Evaluation time [s], shape ``(len(env_ids),)`` or ``(len(env_ids), num_times)``.

        Returns:
            A tuple ``(position, velocity, acceleration)`` in [m], [m/s] and [m/s^2], with a
            trailing time axis when :paramref:`time` has one.
        """
        return evaluate_trajectory(
            self._amplitude[env_ids], self._omega[env_ids], self._phase[env_ids], self._center[env_ids], time
        )

    def evaluate_at(self, env_ids: Sequence[int] | torch.Tensor | slice, time: torch.Tensor):
        """Evaluate the reference at ``time`` and refresh the public buffers and command tensor.

        Args:
            env_ids: Indices of the environments to evaluate, or ``slice(None)`` for all.
            time: Evaluation time [s], shape ``(len(env_ids),)``.
        """
        pos, vel, acc = self._evaluate_position(env_ids, time)
        yaw, yaw_rate = evaluate_yaw(
            self._yaw_0[env_ids],
            self._yaw_amplitude[env_ids],
            self._yaw_omega[env_ids],
            self._yaw_phase[env_ids],
            time,
        )
        lookahead, _, _ = self._evaluate_position(env_ids, time.unsqueeze(-1) + self._lookahead_times)

        self.pos_ref_w[env_ids] = pos
        self.vel_ref_w[env_ids] = vel
        self.acc_ref_w[env_ids] = acc
        self.yaw_ref[env_ids] = yaw
        self.yaw_rate_ref[env_ids] = yaw_rate
        self.lookahead_w[env_ids] = lookahead

        self._command[env_ids] = torch.cat(
            [pos, vel, acc, yaw.unsqueeze(-1), yaw_rate.unsqueeze(-1), lookahead.flatten(start_dim=-2)], dim=-1
        )

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        # Environments whose trajectory was already drawn by the reset event keep it, so that the
        # robot state written by that event still matches the reference.
        pending = env_ids[~self._sampled_by_event[env_ids]]
        if len(pending) > 0:
            self.sample_parameters(pending)
        self._sampled_by_event[env_ids] = False

        self._time[env_ids] = 0.0
        self._error_sum[env_ids] = 0.0
        self._vel_error_sum[env_ids] = 0.0
        self._yaw_error_sum[env_ids] = 0.0
        self._ref_speed_sum[env_ids] = 0.0
        self._step_count[env_ids] = 0.0
        self.evaluate_at(env_ids, self._time[env_ids])

    def _update_command(self):
        # Evaluate at the current time *before* advancing the clock: the trajectory must be sampled
        # at t=0 on the step where the robot is still at its reset pose. Advancing first would offset
        # the reference by one policy step, which is 0.3 m at 3 m/s.
        self.evaluate_at(slice(None), self._time)
        self._time += self._env.step_dt

    def _update_metrics(self):
        pos_w = self.robot.data.body_pos_w.torch[:, self.body_idx] - self._env.scene.env_origins
        position_error = torch.linalg.norm(self.pos_ref_w - pos_w, dim=-1)
        velocity_error = torch.linalg.norm(self.vel_ref_w - self.robot.data.root_lin_vel_w.torch, dim=-1)
        _, _, yaw = math_utils.euler_xyz_from_quat(self.robot.data.root_quat_w.torch)
        yaw_error = torch.abs(math_utils.wrap_to_pi(self.yaw_ref - yaw))

        self._error_sum += position_error
        self._vel_error_sum += velocity_error
        self._yaw_error_sum += yaw_error
        self._ref_speed_sum += torch.linalg.norm(self.vel_ref_w, dim=-1)
        self._step_count += 1.0
        steps = self._step_count.clamp(min=1.0)

        # note: write in place rather than rebinding. ``CommandTerm.reset`` zeroes each metric
        # tensor in place after logging it, so rebinding a metric to a state buffer -- as the
        # difficulty is -- would have reset silently wipe that state as well.
        self.metrics["position_error"][:] = position_error
        self.metrics["position_error_mean"][:] = self._error_sum / steps
        self.metrics["velocity_error_mean"][:] = self._vel_error_sum / steps
        self.metrics["yaw_error_mean"][:] = self._yaw_error_sum / steps
        self.metrics["ref_speed_mean"][:] = self._ref_speed_sum / steps
        self.metrics["difficulty"][:] = self._difficulty

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.path_visualizer = VisualizationMarkers(self.cfg.path_visualizer_cfg)
                self.lookahead_visualizer = VisualizationMarkers(self.cfg.lookahead_visualizer_cfg)
                self.termination_flash_visualizer = VisualizationMarkers(self.cfg.termination_flash_visualizer_cfg)
                self._init_termination_flash()
            for visualizer in self._visualizers:
                visualizer.set_visibility(True)
        elif hasattr(self, "goal_pose_visualizer"):
            for visualizer in self._visualizers:
                visualizer.set_visibility(False)

    @property
    def _visualizers(self) -> tuple[VisualizationMarkers, ...]:
        return (
            self.goal_pose_visualizer,
            self.path_visualizer,
            self.lookahead_visualizer,
            self.termination_flash_visualizer,
        )

    def _path_vis_time_grid(self, n: int) -> torch.Tensor:
        """Return the times at which the drawn path is sampled, shape ``(n, num_path_vis_points)``.

        Sampling the whole episode is right for a reference that never repeats. A subclass whose
        reference is periodic should override this to cover exactly one period instead, so the
        drawn path is the shape rather than however many laps fit in an episode.

        Args:
            n: Number of environments being visualized.
        """
        return self._path_vis_times.unsqueeze(0).expand(n, len(self._path_vis_times))

    def _init_termination_flash(self):
        """Allocate the latch buffers backing the early-termination flash."""
        # Prototype order is the marker insertion order, so the keys are the term -> colour mapping.
        self._flash_term_names = list(self.cfg.termination_flash_visualizer_cfg.markers.keys())
        self._flash_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._flash_proto = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Latched failure pose, and the pose of the previous rendered frame it is latched from.
        self._flash_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self._flash_last_step = -1

    def _update_termination_flash(self, n: int, origins: torch.Tensor):
        """Latch, age and draw the early-termination markers.

        The environment is reset in the same step it terminates, so by the time this callback runs
        the drone has already been teleported back onto the start of a fresh reference. The pose
        recorded on the previous rendered frame is therefore the last one before the failure, and it
        is that pose the marker is pinned to.

        Args:
            n: Number of environments being visualized.
            origins: Environment origins of those environments [m], shape ``(n, 3)``.
        """
        pos = self.robot.data.body_pos_w.torch[:, self.body_idx] - self._env.scene.env_origins

        # Age the latch once per environment step, not once per rendered frame.
        step = self._env.common_step_counter
        if step != self._flash_last_step:
            self._flash_last_step = step
            # Age first, then latch, so a flash latched this frame is drawn for the full
            # ``termination_flash_steps`` frames rather than one fewer.
            self._flash_timer[self._flash_timer > 0] -= 1
            terminated = self._env.termination_manager.terminated
            if torch.any(terminated):
                idx = terminated.nonzero().flatten()
                self._flash_pos[idx] = self._prev_pos[idx]
                self._flash_timer[idx] = self.cfg.termination_flash_steps
                # Colour by cause. Later terms win, which only matters when two fire on one step.
                self._flash_proto[idx] = 0
                for i, name in enumerate(self._flash_term_names):
                    if name in self._env.termination_manager.active_terms:
                        hit = self._env.termination_manager.get_term(name) & terminated
                        self._flash_proto[hit] = i
            self._prev_pos[:] = pos

        # Constant instance count, hidden by a zero scale, so the point instancer never resizes.
        scales = (self._flash_timer[:n] > 0).float().unsqueeze(-1).expand(-1, 3)
        self.termination_flash_visualizer.visualize(
            translations=self._flash_pos[:n] + origins,
            scales=scales,
            marker_indices=self._flash_proto[:n],
        )

    def _debug_vis_callback(self, event):
        # note: this is needed in case the robot is de-initialized, since we can't access the data
        if not self.robot.is_initialized:
            return
        n = self._num_vis_envs
        origins = self._env.scene.env_origins[:n]

        # -- moving target, oriented by the yaw reference so that it is visible too
        zeros = torch.zeros(n, device=self.device)
        self.goal_pose_visualizer.visualize(
            self.pos_ref_w[:n] + origins, math_utils.quat_from_euler_xyz(zeros, zeros, self.yaw_ref[:n])
        )

        # -- look-ahead chain
        self.lookahead_visualizer.visualize((self.lookahead_w[:n] + origins.unsqueeze(1)).reshape(-1, 3))

        # -- the whole path, which only changes when a trajectory is resampled
        if self._path_vis_dirty:
            path, _, _ = self._evaluate_position(slice(n), self._path_vis_time_grid(n))
            self._path_vis_points = (path + origins.unsqueeze(1)).reshape(-1, 3)
            self._path_vis_dirty = False
        self.path_visualizer.visualize(self._path_vis_points)

        # -- where an environment terminated early, held for a few steps so it is visible
        if self.cfg.termination_flash_steps > 0:
            self._update_termination_flash(n, origins)
