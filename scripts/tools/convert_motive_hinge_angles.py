# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert an OptiTrack Motive rigid-body export into hinge angles over time.

Motive exports each tracked rigid body as a world-frame quaternion and position. A passive hinge
shows up as the rotation of one body relative to another (or to the world, when the parent is
clamped and not tracked). For every requested joint this script:

1. forms the child orientation relative to its parent,
2. measures it from a reference pose (by default, the mean over the last seconds of the take),
3. finds the hinge axis as the principal direction of that rotation over the take, and
4. writes the angle about that axis, plus the angle about the next principal axis, which picks up
   a second gimbal axis or tracking error.

The output is the CSV format read by ``analyze_hinge_decay.py``: a ``time`` column [s] and one
column per angle [deg].

Usage:

.. code-block:: bash

    # tube swinging from a clamped (untracked) airframe, saw head hinged to the tube
    ./isaaclab.sh -p scripts/tools/convert_motive_hinge_angles.py take.csv angles.csv \\
        --joint tube=Chainsaw2 --joint saw_head=Bottom:Chainsaw2
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

parser = argparse.ArgumentParser(description="Convert a Motive rigid-body CSV export into hinge angles.")
parser.add_argument("input", help="Motive CSV export with rigid-body quaternions.")
parser.add_argument("output", help="Output CSV of hinge angles.")
parser.add_argument(
    "--joint",
    action="append",
    required=True,
    metavar="NAME=CHILD[:PARENT]",
    help="Joint to extract: output name, child rigid body, and optional parent body (world if omitted). Repeatable.",
)
parser.add_argument(
    "--ref_window",
    type=float,
    nargs=2,
    default=None,
    metavar=("START", "END"),
    help="Time window [s] whose mean pose is the zero angle. Defaults to the last 2 s of the take.",
)
args_cli = parser.parse_args()


# -- quaternion helpers, (x, y, z, w) order as exported by Motive


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quaternions)."""
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b``, broadcasting over leading dimensions."""
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Rotation vector [rad] of the shortest rotation represented by each quaternion."""
    q = np.where(q[..., 3:4] < 0.0, -q, q)
    xyz, w = q[..., :3], np.clip(q[..., 3], -1.0, 1.0)
    s = np.linalg.norm(xyz, axis=-1)
    angle = 2.0 * np.arctan2(s, w)
    scale = np.where(s > 1e-12, angle / np.maximum(s, 1e-12), 2.0)
    return xyz * scale[..., None]


def quat_mean(q: np.ndarray) -> np.ndarray:
    """Mean orientation: principal eigenvector of the quaternion outer-product sum (sign-invariant)."""
    _, vecs = np.linalg.eigh(q.T @ q)
    return vecs[:, -1]


# -- Motive export parsing


def load_motive(path: str) -> tuple[dict[str, str], np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Read a Motive CSV export.

    Returns:
        The take metadata, the frame indices, the time [s] of each frame, and the quaternion of
        every rigid body, shape (num_frames, 4) in (x, y, z, w) order, NaN where untracked.
    """
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    meta = dict(zip(rows[0][0::2], rows[0][1::2]))
    if meta.get("Rotation Type", "Quaternion") != "Quaternion":
        sys.exit(f"Re-export the take with 'Rotation Type' set to Quaternion (got '{meta['Rotation Type']}').")
    header = next(i for i, r in enumerate(rows) if r and r[0] == "Frame")
    types = next(r for r in rows[:header] if len(r) > 1 and r[1] == "Type")
    names = next(r for r in rows[:header] if len(r) > 1 and r[1] == "Name")
    kinds, axes = rows[header - 1], rows[header]

    body = rows[header + 1 :]
    width = len(axes)
    table = np.full((len(body), width), np.nan)
    for i, r in enumerate(body):
        for j, v in enumerate(r[:width]):
            if v != "":
                table[i, j] = float(v)

    quats: dict[str, np.ndarray] = {}
    for name in dict.fromkeys(n for n, t in zip(names, types) if t == "Rigid Body"):
        idx = [
            next(
                j
                for j in range(width)
                if names[j] == name and types[j] == "Rigid Body" and kinds[j] == "Rotation" and axes[j] == a
            )
            for a in "XYZW"
        ]
        quats[name] = table[:, idx]

    frames = table[:, 0]
    rate = float(meta.get("Export Frame Rate", "nan"))
    # The exported time column is rounded to the millisecond; the frame rate is exact.
    time = (frames - frames[0]) / rate if np.isfinite(rate) else table[:, 1]
    return meta, frames, time, quats


def main():
    """Extract the requested hinge angles and write them as one CSV."""
    meta, frames, time, quats = load_motive(args_cli.input)
    print(
        f"Take '{meta.get('Take Name', '?')}': {len(time)} frames at {meta.get('Export Frame Rate', '?')} Hz, "
        f"rigid bodies {list(quats)}"
    )

    joints = []
    for spec in args_cli.joint:
        name, _, bodies = spec.partition("=")
        child, _, parent = bodies.partition(":")
        for b in (child, parent):
            if b and b not in quats:
                sys.exit(f"Rigid body '{b}' not in the take; available: {list(quats)}")
        joints.append((name, child, parent or None))

    # Frames where any body we need is untracked are dropped.
    used = {b for _, c, p in joints for b in (c, p) if b}
    valid = np.all([np.all(np.isfinite(quats[b]), axis=1) for b in used], axis=0)
    if not valid.all():
        gaps = np.diff(frames[valid])
        print(f"Dropped {np.sum(~valid)} untracked frames; longest gap {int(gaps.max()) - 1} frames.")
    t = time[valid]

    if args_cli.ref_window is None:
        ref = t >= t[-1] - 2.0
    else:
        ref = (t >= args_cli.ref_window[0]) & (t <= args_cli.ref_window[1])
    if ref.sum() < 10:
        sys.exit("The reference window holds fewer than 10 frames.")

    columns = {"time": t}
    for name, child, parent in joints:
        qc = quats[child][valid]
        qc /= np.linalg.norm(qc, axis=1, keepdims=True)
        if parent is None:
            # Clamped parent: the hinge axis is fixed in the world, so measure the rotation there.
            q_rel = qc
            q_ref = quat_mean(q_rel[ref])
            delta = quat_mul(q_rel, quat_conj(q_ref))
            frame = "world"
        else:
            qp = quats[parent][valid]
            qp /= np.linalg.norm(qp, axis=1, keepdims=True)
            # The hinge axis is fixed in the parent body, so measure the rotation in its frame.
            q_rel = quat_mul(quat_conj(qp), qc)
            q_ref = quat_mean(q_rel[ref])
            delta = quat_mul(q_rel, quat_conj(q_ref))
            frame = f"'{parent}' body"
        rotvec = np.degrees(quat_to_rotvec(delta))

        # Principal axes of the rotation over the take: the first is the hinge.
        centred = rotvec - rotvec.mean(axis=0)
        _, sv, vt = np.linalg.svd(centred, full_matrices=False)
        share = sv**2 / np.sum(sv**2)
        axes = vt * np.sign(vt[np.arange(3), np.argmax(np.abs(vt), axis=1)])[:, None]
        columns[name] = rotvec @ axes[0]
        columns[f"{name}_secondary"] = rotvec @ axes[1]
        print(f"\n{name}: '{child}' relative to {frame}")
        print(f"  hinge axis     {np.round(axes[0], 3)}  ({100 * share[0]:.1f}% of the motion)")
        print(f"  secondary axis {np.round(axes[1], 3)}  ({100 * share[1]:.1f}%)")
        print(f"  angle range    {columns[name].min():.1f} to {columns[name].max():.1f} deg")

    with open(args_cli.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(columns))
        for row in zip(*columns.values()):
            writer.writerow([f"{v:.6f}" for v in row])
    print(f"\nWrote {len(t)} rows to {args_cli.output}")


if __name__ == "__main__":
    main()
