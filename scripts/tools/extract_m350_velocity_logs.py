# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Extract velocity commands and the measured response from real M350 flight bags.

Reads the ROS 2 (mcap) bags under ``m350_velocity_logs/<log>/bag`` and writes, per log, the small
CSV files that :mod:`compare_m350_velocity_response` replays in simulation:

* ``cmd.csv``  -- ``t, vx, vy, vz, yaw_rate``: the velocity command as sent to the DJI flight
  controller [m/s, rad/s]. Body FLU for the open-loop step log, world ENU for the others.
* ``vel.csv``  -- ``t, vE, vN, vU``: the fused ground velocity in world ENU [m/s].
* ``pose.csv`` -- ``t, qx, qy, qz, qw``: the airframe attitude in world ENU.
* ``meta.json`` -- the command frame, the payload, and the segments to compare on.

Segments are the parts of each log where the vehicle is actually following the offboard command.
On the chainsaw flights the pilot sometimes takes authority back while the offboard controller keeps
publishing, and nothing in the bag records it. Those spans show up as a sustained gap between the
command and the response, and are dropped: the response is compared with a deliberately loose
reference (0.2 s delay, 0.5 s first-order lag) and any span where the 3-D gap stays above
``--threshold`` for 0.5 s is removed, with margins, on all axes.

``rosbags`` is not a project dependency, so run this with it added for the one call::

    UV_CACHE_DIR=/local/uv-cache uv run --no-project --with rosbags --with pandas \\
        python scripts/tools/extract_m350_velocity_logs.py --logs_dir m350_velocity_logs
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

parser = argparse.ArgumentParser(description="Extract M350 velocity commands and response from flight bags.")
parser.add_argument("--logs_dir", type=Path, default=Path("m350_velocity_logs"), help="Directory of log folders.")
parser.add_argument("--out_dir", type=Path, default=None, help="Output directory. Defaults to <logs_dir>/extracted.")
parser.add_argument("--threshold", type=float, default=0.15, help="Command/response gap marking an override [m/s].")
args = parser.parse_args()

CMD_TOPICS = {
    "/wrapper/psdk_ros2/flight_control_setpoint_FLUvelocity_yawrate": "body_flu",
    "/wrapper/psdk_ros2/flight_control_setpoint_ENUvelocity_yawrate": "world_enu",
}
VEL_TOPIC = "/wrapper/psdk_ros2/velocity_ground_fused"
POSE_TOPIC = "/aerial_platform/drone_pose"

DT = 0.01
REF_DELAY_S, REF_TAU_S = 0.2, 0.5
MIN_BAD_S, PAD_BEFORE_S, PAD_AFTER_S, MIN_KEEP_S, SKIP_START_S = 0.5, 1.0, 2.0, 3.0, 1.0


def read_bag(bag_dir: Path) -> tuple[pd.DataFrame, str, pd.DataFrame, pd.DataFrame]:
    """Read the command, fused velocity and pose topics, with times relative to the bag start."""
    cmd, vel, pose, frame = [], [], [], None
    with AnyReader([bag_dir], default_typestore=get_typestore(Stores.ROS2_HUMBLE)) as reader:
        t0 = reader.start_time * 1e-9
        topics = set(CMD_TOPICS) | {VEL_TOPIC, POSE_TOPIC}
        conns = [c for c in reader.connections if c.topic in topics and c.msgcount > 0]
        for conn, t_ns, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            t = t_ns * 1e-9 - t0
            if conn.topic in CMD_TOPICS:
                frame = CMD_TOPICS[conn.topic]
                cmd.append([t, *list(msg.axes)[:4]])
            elif conn.topic == VEL_TOPIC:
                vel.append([t, msg.vector.x, msg.vector.y, msg.vector.z])
            else:
                q = msg.pose.orientation
                pose.append([t, q.x, q.y, q.z, q.w])
    return (
        pd.DataFrame(cmd, columns=["t", "vx", "vy", "vz", "yaw_rate"]),
        frame,
        pd.DataFrame(vel, columns=["t", "vE", "vN", "vU"]),
        pd.DataFrame(pose, columns=["t", "qx", "qy", "qz", "qw"]),
    )


def yaw_from_quat(q: np.ndarray) -> np.ndarray:
    """Yaw [rad] of (x, y, z, w) quaternions."""
    x, y, z, w = q.T
    return np.unwrap(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def command_world(cmd: pd.DataFrame, frame: str, pose: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    """Zero-order hold of the command on ``t``, rotated into world ENU when it is body-frame."""
    idx = np.clip(np.searchsorted(cmd.t.values, t, side="right") - 1, 0, None)
    c = cmd[["vx", "vy", "vz"]].values[idx]
    if frame == "body_flu":
        psi = np.interp(t, pose.t, yaw_from_quat(pose[["qx", "qy", "qz", "qw"]].values))
        c = np.stack(
            [np.cos(psi) * c[:, 0] - np.sin(psi) * c[:, 1], np.sin(psi) * c[:, 0] + np.cos(psi) * c[:, 1], c[:, 2]], 1
        )
    return c


def loose_reference(c: np.ndarray) -> np.ndarray:
    """Delay and low-pass a command so it resembles a plausible response without assuming gains."""
    n_delay = int(round(REF_DELAY_S / DT))
    c = np.concatenate([np.repeat(c[:1], n_delay, 0), c])[: len(c)]
    out, y = np.empty_like(c), c[0].copy()
    for k in range(len(c)):
        y += DT / REF_TAU_S * (c[k] - y)
        out[k] = y
    return out


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start and end (exclusive) indices of the True runs in ``mask``."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def following_segments(windows, cmd, frame, vel, pose, threshold) -> list[list[float]]:
    """Sub-spans of the commanded windows where the response follows the command."""
    segments = []
    for a, b in windows:
        inside = cmd[(cmd.t >= a) & (cmd.t <= b)]
        if len(inside) < 20:
            continue
        t = np.arange(inside.t.iloc[0], inside.t.iloc[-1], DT)
        c = command_world(inside, frame, pose, t)
        v = np.stack([np.interp(t, vel.t, vel[k]) for k in ("vE", "vN", "vU")], 1)
        gap = np.linalg.norm(v - loose_reference(c), axis=1)
        drop = np.zeros(len(t), dtype=bool)
        for i, j in runs(gap > threshold):
            if (j - i) * DT >= MIN_BAD_S:
                drop[max(0, i - int(PAD_BEFORE_S / DT)) : j + int(PAD_AFTER_S / DT)] = True
        # The controller has only just taken over at the start of a window.
        drop[: int(SKIP_START_S / DT)] = True
        for i, j in runs(~drop):
            if (j - i) * DT >= MIN_KEEP_S:
                segments.append([round(float(t[i]), 3), round(float(t[j - 1]), 3)])
    return segments


def main():
    out_root = args.out_dir or args.logs_dir / "extracted"
    for log_dir in sorted(p for p in args.logs_dir.iterdir() if (p / "bag").is_dir()):
        desc = json.loads((log_dir / "description.json").read_text())
        cmd, frame, vel, pose = read_bag(log_dir / "bag")
        # Open-loop step logs have no commanded windows listed: the whole log is commanded.
        windows = desc.get("commanded_windows_s") or [[float(cmd.t.iloc[0]), float(cmd.t.iloc[-1])]]
        if frame == "body_flu":
            # Open-loop steps: no pilot in the loop, so keep the whole commanded span.
            segments = [[round(float(cmd.t.iloc[0]), 3), round(float(cmd.t.iloc[-1]), 3)]]
        else:
            segments = following_segments(windows, cmd, frame, vel, pose, args.threshold)
        payload = "chainsaw" if str(desc.get("payload", "")).startswith("yes") else "none"
        key = log_dir.name[:2]
        out = out_root / key
        out.mkdir(parents=True, exist_ok=True)
        cmd.to_csv(out / "cmd.csv", index=False)
        vel.to_csv(out / "vel.csv", index=False)
        pose.to_csv(out / "pose.csv", index=False)
        meta = {"name": log_dir.name, "frame": frame, "payload": payload, "segments": segments}
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        kept = sum(b - a for a, b in segments)
        print(f"{key}: {frame:9s} payload={payload:8s} {len(segments):2d} segments, {kept:5.1f} s -> {out}")


if __name__ == "__main__":
    main()
