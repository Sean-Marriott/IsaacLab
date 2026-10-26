# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers shared by the drone ARL MDP terms."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def resolve_env_ids(
    env_ids: Sequence[int] | torch.Tensor | slice | None, num_envs: int, device: str | torch.device
) -> torch.Tensor:
    """Convert environment indices into an index tensor.

    The managers pass ``None`` or a :class:`slice` when all environments are selected, which cannot be
    counted with :func:`len` or used to gather per-environment values.

    Args:
        env_ids: Environment indices, a slice over all environments, or None for all environments.
        num_envs: Number of environments.
        device: Device of the returned tensor.

    Returns:
        The selected environment indices, shape ``(num_selected,)``, dtype ``torch.long``.
    """
    if env_ids is None:
        env_ids = slice(None)
    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device)[env_ids]
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)
