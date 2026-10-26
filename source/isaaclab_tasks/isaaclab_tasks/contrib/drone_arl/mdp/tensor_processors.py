# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tensor processors used by the drone ARL observations."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass
from isaaclab.utils.buffers import DelayBuffer
from isaaclab.utils.tensor_processors import TensorProcessor, TensorProcessorCfg


class DelayedObservation(TensorProcessor):
    r"""A processor that returns a stochastically delayed (stale) version of an observation term.

    This can also be used to model multi-rate observations for non-sensor terms, e.g., pure MDP terms or
    proprioceptive terms.

    Each new batched observation is pushed into a :class:`~isaaclab.utils.buffers.DelayBuffer` and an older
    sample is returned according to a per-environment integer time-lag. Lags are drawn uniformly from
    ``[min_lag, max_lag]``, with an optional probability to *hold* the previous lag (to mimic repeated frames).
    With ``update_period > 0`` (multi-rate), new lags are applied only on refresh ticks, which occur every
    ``update_period`` policy steps. Between refreshes the realised lag can increase at most by +1 (frame hold).
    This process is causal: the lag for each environment can only increase by 1 each step, ensuring that the
    returned observation is never older than the previous step's lagged observation.

    Unlike :attr:`~isaaclab.managers.ObservationTermCfg.delay_max_lag`, this processor runs before noise,
    clipping and scaling, and supports the multi-rate refresh and causal lag clamp described above.

    Shapes are preserved: the returned tensor has the exact shape of the wrapped term
    (``[num_envs, *obs_shape]``).

    Example:

    .. code-block:: python

        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos,
            noise=Gnoise(mean=0.0, std=0.0175),
            tensor_processors=[
                DelayedObservationCfg(
                    min_lag=0, max_lag=3, per_env=True, hold_prob=0.66, update_period=3, per_env_phase=True
                )
            ],
        )
    """

    def __init__(self, cfg: DelayedObservationCfg, data_dim: tuple[int, ...], device: str):
        """Initialize the delayed observation processor.

        Args:
            cfg: Configuration parameters.
            data_dim: The dimensions of the observation. The first element is the number of environments.
            device: The device to run the processor on.
        """
        super().__init__(cfg, data_dim, device)
        if cfg.min_lag < 0 or cfg.max_lag < cfg.min_lag:
            raise ValueError("DelayedObservation: require 0 <= min_lag <= max_lag.")
        if cfg.hold_prob < 0.0 or cfg.hold_prob > 1.0:
            raise ValueError("DelayedObservation: hold_prob must be in [0, 1].")
        if cfg.update_period < 0:
            raise ValueError("DelayedObservation: update_period must be non-negative.")
        if cfg.update_period > 0 and cfg.update_period > cfg.max_lag:
            raise ValueError("DelayedObservation: update_period must be <= max_lag.")

        self._batch_size = data_dim[0]
        self._expand_1d = len(data_dim) == 1  # e.g., input shape (N,)
        # ensure at least one feature dim for DelayBuffer resets
        self._feature_shape = (1,) if self._expand_1d else data_dim[1:]

        # state
        self._buf = DelayBuffer(history_length=cfg.max_lag + 1, batch_size=self._batch_size, device=device)
        self._prev_realized_lags: torch.Tensor | None = None  # [N]
        self._phases: torch.Tensor | None = None  # [N] if multi-rate
        self._step: int = 0

        # prefill buffer with zeros so early delays are valid
        self._prefill()

    def _prefill(self):
        """Fill the delay buffer with zeros so every lag up to ``max_lag`` has a sample."""
        zeros = torch.zeros((self._batch_size, *self._feature_shape), device=self._device)
        for _ in range(self._cfg.max_lag + 1):
            self._buf.compute(zeros)

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the delay buffer and internal state.

        When ``env_ids`` is given, only the previous lags of those environments are reset to zero, forcing the
        latest observation to be returned on the next call so that no observation from before the reset is
        returned.

        Args:
            env_ids: The environment ids. Defaults to None, in which case all environments are considered.
        """
        if env_ids is None:
            self._buf.reset()
            self._prev_realized_lags = None
            self._phases = None
            self._step = 0
            self._prefill()
        elif self._prev_realized_lags is not None:
            self._prev_realized_lags[env_ids] = 0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Add the current data to the delay buffer and return a stale sample according to the current lag.

        Args:
            data: The data to apply delay to.

        Returns:
            Delayed data. Shape is the same as data.
        """
        cfg = self._cfg
        num_envs = self._data_dim[0]
        self._step += 1

        data_in = data.unsqueeze(-1) if (self._expand_1d and data.dim() == 1) else data

        # initialize phases for multi-rate on first use
        if cfg.update_period > 0 and self._phases is None:
            if cfg.per_env_phase:
                self._phases = torch.randint(0, cfg.update_period, (num_envs,), device=self._device)
            else:
                self._phases = torch.zeros(num_envs, dtype=torch.long, device=self._device)

        # sample desired lags in [min_lag, max_lag]
        if cfg.min_lag == cfg.max_lag:
            desired_lags = torch.full((num_envs,), cfg.max_lag, dtype=torch.long, device=self._device)
        else:
            desired_lags = torch.randint(cfg.min_lag, cfg.max_lag + 1, (num_envs,), device=self._device)

        if not cfg.per_env:
            desired_lags = torch.full_like(desired_lags, desired_lags[0])

        # optional: hold previous realized lag
        if cfg.hold_prob > 0.0 and self._prev_realized_lags is not None:
            hold_mask = torch.rand((num_envs,), device=self._device) < cfg.hold_prob
            desired_lags = torch.where(hold_mask, self._prev_realized_lags, desired_lags)

        if self._prev_realized_lags is None:
            realized_lags = desired_lags
        elif cfg.update_period > 0:
            # multi-rate: between refreshes, the lag can only increase by +1 (clamped)
            refresh_mask = ((self._step - self._phases) % cfg.update_period) == 0
            hold_realized_lags = (self._prev_realized_lags + 1).clamp(max=cfg.max_lag)
            realized_lags = torch.where(refresh_mask, desired_lags, hold_realized_lags)
        else:
            # every step: causal clamp (at most +1 step older)
            realized_lags = torch.minimum(desired_lags, self._prev_realized_lags + 1)

        realized_lags = realized_lags.clamp(min=cfg.min_lag, max=cfg.max_lag)
        self._prev_realized_lags = realized_lags

        # return stale sample
        self._buf.set_time_lag(realized_lags)
        out = self._buf.compute(data_in)

        if self._expand_1d and out.dim() == 2 and data.dim() == 1:
            out = out.squeeze(-1)
        return out


@configclass
class DelayedObservationCfg(TensorProcessorCfg):
    """Configuration parameters for a delayed observation processor.

    For more information, please check the :class:`DelayedObservation` class.
    """

    func: type[DelayedObservation] = DelayedObservation
    """The delayed observation processor class."""

    min_lag: int = 0
    """The minimum lag (in number of policy steps) to be applied to the observations. Defaults to 0."""

    max_lag: int = 3
    """The maximum lag (in number of policy steps) to be applied to the observations.

    This value must be greater than or equal to :attr:`min_lag`.
    """

    per_env: bool = True
    """Whether to use a separate lag for each environment."""

    hold_prob: float = 0.0
    """The probability of holding the previous lag when updating the lag."""

    update_period: int = 1
    """The period (in number of policy steps) at which the lag is updated.

    If set to 0, the lag is updated every step, subject only to the causal clamp. If set to a positive
    integer, new lags are applied every ``update_period`` policy steps. Defaults to 1.

    This value must be less than or equal to :attr:`max_lag` if it is greater than 0.
    """

    per_env_phase: bool = True
    """Whether to use a separate phase for each environment when updating the lag.

    Only relevant if :attr:`update_period` is greater than 0. Defaults to True.
    """
