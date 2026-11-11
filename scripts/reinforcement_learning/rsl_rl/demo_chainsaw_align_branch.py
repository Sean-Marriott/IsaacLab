# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Side-by-side demonstration of seating a branch in the chainsaw mouth, policy against a scripted approach,
made for a presentation slide.

Purpose
-------
Two M350 + Mk3 chainsaw vehicles start from the same pose relative to the same branch, and each has to feed
the cutting mouth forward onto it (``IsaacContrib-ChainsawAlignBranch-Matrice``):

* **Scripted approach** -- a proportional controller: the velocity command is ``--baseline_gain`` times a
  position error, capped at ``--baseline_speed``, and likewise for the yaw. With ``--baseline airframe``
  the error is that of the airframe to where it rests with the mouth on the goal, which does not feed the
  swing back into the command; with ``--baseline mouth`` it is the error of the mouth itself. It follows
  the same staged route as the policy (the staging point behind the branch, then the branch), and knows
  nothing about the payload's swing or the branch's push.
* **RL policy** -- the trained recurrent policy commands the airframe.

Both send velocity commands to the same Lee velocity controller in one physics session. To make the two
runs differ only in who commands the airframe, everything the task randomizes per environment is fixed:
the branch (diameter, heading, level), the spawn, the controller gains, the command delay, the hinge
damping and the saw mass; the wind is off. The debug markers are hidden unless ``--markers`` is passed.

Outputs
-------
Written to ``--output_dir`` (default ``<checkpoint_dir>/align_branch_demo``):

* ``align_branch_demo.mp4``      -- the slide video: the rendered scene with a label over each vehicle and
  the distance of both mouths from the seat plotted underneath, drawn as the video plays.
* ``align_branch_demo_raw.mp4``  -- the rendered scene alone, for adding your own captions.
* ``align_branch_demo.png``      -- the distance plot alone, as a static figure.
* ``align_branch_demo_still.png`` -- one rendered frame, taken ``--still_s`` after the start.
* ``align_branch_demo.npz``      -- the logged time series.

Pass ``--no_video`` to skip rendering and write only the plot and the data.

Usage::

    uv run python scripts/reinforcement_learning/rsl_rl/demo_chainsaw_align_branch.py \\
        --checkpoint logs/rsl_rl/matrice_chainsaw_align_branch_rnn/<run>/model_1499.pt

    # a thin branch, starting on the far side of it, against the scripted approach on the mouth error
    uv run python scripts/reinforcement_learning/rsl_rl/demo_chainsaw_align_branch.py \\
        --checkpoint /path/to/model_1499.pt --branch branch_10mm --spawn_offset 1.5 0.0 0.4 --baseline mouth
"""

import argparse
import contextlib
import math
import os
import warnings
from typing import TYPE_CHECKING

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl as cli_args
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

parser = argparse.ArgumentParser(description="Side-by-side chainsaw branch seating demo: policy vs. scripted approach.")
parser.add_argument(
    "--task",
    type=str,
    default="IsaacContrib-ChainsawAlignBranch-Matrice",
    help="Gym task ID of the chainsaw branch alignment task (play-mode overrides are applied).",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_rnn_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--branch", type=str, default="branch_30mm", help="Name of the branch to seat (see BRANCH_SEATS).")
parser.add_argument(
    "--spawn_offset",
    type=float,
    nargs=3,
    default=[-1.6, 0.5, 0.3],
    help="Position the mouth starts at, relative to the seat in its frame [m]: x is along the feed direction"
    " (negative is behind the branch), y along the branch, z up.",
)
parser.add_argument("--spawn_yaw_deg", type=float, default=30.0, help="Starting yaw relative to the feed direction.")
parser.add_argument("--duration_s", type=float, default=15.0, help="Length of the demo [s].")
parser.add_argument(
    "--baseline",
    type=str,
    default="airframe",
    choices=["airframe", "mouth"],
    help="Position error the scripted approach closes: that of the airframe to its rest position over the goal,"
    " or that of the mouth itself.",
)
parser.add_argument("--baseline_gain", type=float, default=1.0, help="Gain of the scripted approach [1/s].")
parser.add_argument("--baseline_speed", type=float, default=1.0, help="Largest speed the scripted approach asks [m/s].")
parser.add_argument("--spacing", type=float, default=4.5, help="Distance between the two branches [m].")
parser.add_argument("--fps", type=int, default=25, choices=[10, 20, 25, 50], help="Frame rate of the videos.")
parser.add_argument("--resolution", type=int, nargs=2, default=[1920, 720], help="Size of the rendered scene [px].")
parser.add_argument("--camera_distance", type=float, default=9.0, help="Distance of the camera from the scene [m].")
parser.add_argument(
    "--camera_azimuth_deg",
    type=float,
    default=20.0,
    help="Angle of the camera from side-on, about the vertical: zero looks along the branches.",
)
parser.add_argument("--camera_height", type=float, default=1.2, help="Height of the camera above the scene centre [m].")
parser.add_argument("--hold_s", type=float, default=1.0, help="Time the first frame is held before the start [s].")
parser.add_argument("--still_s", type=float, default=0.0, help="Time after the start of the still frame [s].")
parser.add_argument("--markers", action="store_true", help="Draw the task's debug markers (target, staging, corridor).")
parser.add_argument("--no_video", action="store_true", help="Skip rendering; write only the plot and the data.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
# Anything not parsed here is passed on to Hydra as config overrides.
args_cli, hydra_args = parser.parse_known_args()

OUTPUT_NAME = "align_branch_demo"

# Environment 0 flies the policy, environment 1 the scripted approach: with the camera on the -y side, the
# scripted vehicle (at -x) is then on the left, and both feed from left to right.
ON, OFF = 0, 1
LABELS = {OFF: "Scripted approach", ON: "RL policy"}
# Slots 1 and 2 of the dataviz categorical palette (blue, orange), with its light chart surface and inks.
COLORS = {ON: "#2a78d6", OFF: "#eb6834"}
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a63", "#e4e3dd"


def reset_tool_fixed(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    command_name: str,
    hang_offset: tuple[float, float, float],
    offset_t: tuple[float, float, float],
    yaw_offset: float,
    roll_cfg: "SceneEntityCfg | None" = None,
    roll_hang_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Reset the robot level and at rest, with its tool at the same place relative to the target in every
    environment.

    Args:
        env: The manager-based RL environment instance.
        env_ids: Indices of the environments to reset.
        command_name: Name of the tool pose command term.
        hang_offset: Tool position relative to the root with the payload hanging at rest [m].
        offset_t: Tool position relative to the target, in the target's yaw frame [m].
        yaw_offset: Robot yaw relative to the target yaw [rad].
        roll_cfg: The tool-roll joint, if the tool can be rolled.
        roll_hang_shift: Shift of the hang offset with the roll [m].
    """
    import isaaclab.utils.math as math_utils

    from isaaclab_tasks.contrib.drone_arl.chainsaw_align.mdp import rolled_hang_offset
    from isaaclab_tasks.contrib.drone_arl.mdp.utils import resolve_env_ids

    term = env.command_manager.get_term(command_name)
    asset = term.robot
    env_ids = resolve_env_ids(env_ids, env.num_envs, asset.device)
    n = len(env_ids)
    offset = torch.tensor(offset_t, device=asset.device).expand(n, 3)
    tool_pos_w = term.target_pos_w[env_ids] + math_utils.quat_apply(term.target_quat_w[env_ids], offset)
    zeros = torch.zeros(n, device=asset.device)
    orientation = math_utils.quat_from_euler_xyz(zeros, zeros, term.target_yaw[env_ids] + yaw_offset)
    position = tool_pos_w - math_utils.quat_apply(
        orientation, rolled_hang_offset(asset, hang_offset, env_ids, roll_cfg, roll_hang_shift)
    )
    asset.write_root_pose_to_sim_index(root_pose=torch.cat((position, orientation), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=asset.device), env_ids=env_ids)


def midpoint(value_range: tuple) -> tuple:
    """Collapse a ``(low, high)`` range of scalars or vectors onto its midpoint."""
    low, high = np.asarray(value_range[0], dtype=float), np.asarray(value_range[1], dtype=float)
    mid = 0.5 * (low + high)
    mid = float(mid) if mid.ndim == 0 else tuple(mid.tolist())
    return (mid, mid)


def first_time(time_axis: np.ndarray, flag: np.ndarray) -> float:
    """Time at which the flag is first set [s]; NaN if it never is."""
    hits = np.where(flag)[0]
    return float(time_axis[hits[0]]) if len(hits) > 0 else float("nan")


def style_axes(ax: plt.Axes) -> None:
    """Recessive grid and axes, so the two series carry the figure."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, length=0, labelsize=13)
    ax.grid(True, axis="y", color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)


def draw_distance_plot(
    ax: plt.Axes, time_axis: np.ndarray, distance: np.ndarray, seated: dict[int, float], radius: float
) -> dict:
    """Draw the distance of both mouths from the seat and return the line artists by environment."""
    style_axes(ax)
    lines = {}
    for env_i in (OFF, ON):
        (lines[env_i],) = ax.plot(
            time_axis, distance[:, env_i], color=COLORS[env_i], linewidth=2.5, label=LABELS[env_i]
        )
    # The seat: a log axis keeps the last centimetres readable next to the metres of the approach.
    ax.axhline(radius, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 4)))
    ax.text(
        0.01 * time_axis[-1], radius * 0.9, f"seated: within {100 * radius:.0f} cm", color=INK_MUTED, fontsize=12,
        ha="left", va="top",
    )  # fmt: skip
    ax.set_yscale("log")
    ax.set_xlim(0.0, time_axis[-1])
    ax.set_ylim(0.007, max(3.0, 1.2 * float(distance.max())))
    ax.set_yticks([0.01, 0.1, 1.0], labels=["1 cm", "10 cm", "1 m"])
    ax.minorticks_off()
    ax.set_xlabel("Time [s]", color=INK_MUTED, fontsize=14)
    ax.set_ylabel("Mouth to branch seat", color=INK_MUTED, fontsize=14)
    legend_labels = [
        f"{LABELS[env_i]}  —  "
        + (
            f"seated in {seated[env_i]:.1f} s"
            if not math.isnan(seated[env_i])
            else f"not seated after {time_axis[-1]:.0f} s"
        )
        for env_i in (OFF, ON)
    ]
    ax.legend(
        [lines[OFF], lines[ON]],
        legend_labels,
        loc="upper right",
        frameon=False,
        fontsize=14,
        labelcolor=INK,
        handlelength=1.6,
    )
    return lines


def main():  # noqa: C901
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=True, overrides=hydra_args)

    record = not args_cli.no_video
    width, height = args_cli.resolution
    if record:
        from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
        from isaaclab.visualizers import VisualizerCfg, WindowCfg

        env_cfg.sim.visualizer_cfgs = [
            VisualizerCfg(window=WindowCfg(size=(width, height)), enable_markers=args_cli.markers, focal_length=24.0)
        ]
        # Declared so the launch sets up offscreen rendering; it never starts a clip of its own, since the
        # frames are captured below at the video frame rate rather than once per policy step.
        env_cfg.video_recorders = [VideoRecorderCfg(source="viz:kit", step_offset=10**9)]

    with launch_simulation(env_cfg, args_cli):
        # USD-dependent modules can only be imported once the simulator is up.
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.managers import EventTermCfg as EventTerm

        from isaaclab_tasks.contrib.drone_arl.chainsaw_align import mdp

        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

        env_cfg.scene.num_envs = 2
        env_cfg.scene.env_spacing = args_cli.spacing
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.seed = agent_cfg.seed
        command_name = env_cfg.events.reset_base.params["command_name"]
        command_cfg = getattr(env_cfg.commands, command_name)
        if not args_cli.markers:
            command_cfg.debug_vis = False

        # -- the same branch for both: one diameter, level, with the feed direction along +x
        branch_params = env_cfg.events.place_branch.params
        if args_cli.branch not in branch_params["seats"]:
            raise ValueError(f"Unknown branch '{args_cli.branch}'; choose from {list(branch_params['seats'])}.")
        branch_params["seats"] = {args_cli.branch: branch_params["seats"][args_cli.branch]}
        branch_params["yaw_range"] = (0.0, 0.0)
        branch_params["max_tilt"] = 0.0
        # -- the same start for both
        spawn_params = env_cfg.events.reset_base.params
        env_cfg.events.reset_base = EventTerm(
            func=reset_tool_fixed,
            mode="reset",
            params={
                "command_name": command_name,
                "hang_offset": spawn_params["hang_offset"],
                "offset_t": tuple(args_cli.spawn_offset),
                "yaw_offset": math.radians(args_cli.spawn_yaw_deg),
                "roll_cfg": spawn_params.get("roll_cfg"),
                "roll_hang_shift": spawn_params.get("roll_hang_shift", (0.0, 0.0, 0.0)),
            },
        )
        # -- the same vehicle for both: no per-environment randomization, and no wind
        env_cfg.events.randomize_hinge_damping = None
        env_cfg.events.randomize_saw_mass = None
        wind_params = env_cfg.events.wind.params
        wind_params["force_range"] = wind_params["torque_range"] = wind_params["burst_force_range"] = (0.0, 0.0)
        action_cfg = env_cfg.actions.velocity_command
        action_cfg.command_delay_range_s = midpoint(action_cfg.command_delay_range_s)
        controller_cfg = action_cfg.controller_cfg
        for name in ("K_vel_range", "K_vel_int_range", "K_rot_range", "K_angvel_range"):
            setattr(controller_cfg, name, midpoint(getattr(controller_cfg, name)))
        # The episode must outlast the demo, or the reset would show up in the video.
        env_cfg.episode_length_s = max(env_cfg.episode_length_s, args_cli.duration_s + 1.0)

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        output_dir = os.path.abspath(args_cli.output_dir or os.path.join(os.path.dirname(resume_path), OUTPUT_NAME))
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Loading checkpoint: {resume_path}")

        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        base_env: ManagerBasedRLEnv = env.unwrapped  # type: ignore[assignment]

        from rsl_rl.runners import OnPolicyRunner

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=base_env.device)

        device = base_env.device
        command_term = base_env.command_manager.get_term(command_name)
        saw_contact = base_env.scene.sensors["saw_contact"]
        # The manager resolves its own copy of the term's parameters, so the roll joint is resolved again here.
        rest_params = dict(base_env.cfg.observations.policy.airframe_rest_error.params)
        if rest_params.get("roll_cfg") is not None:
            rest_params["roll_cfg"].resolve(base_env.scene)
        success_radius = command_term.cfg.success_radius
        num_steps = round(args_cli.duration_s / base_env.step_dt)

        # ------------------------------------------------------------------ #
        # Logging and frame capture, once per physics step
        # ------------------------------------------------------------------ #
        # The environment logs nothing and renders at most once per policy step (10 Hz), which makes a
        # jagged plot and a jerky video. ``scene.update`` runs after every physics step, with the asset
        # data refreshed, so the mouth is logged there (100 Hz) and a frame captured every few steps.
        num_samples = num_steps * base_env.cfg.decimation + 1
        logs = {
            "distance": np.zeros((num_samples, 2), dtype=np.float32),
            "offset": np.zeros((num_samples, 2, 3), dtype=np.float32),
            "yaw_error": np.zeros((num_samples, 2), dtype=np.float32),
            "contact_force": np.zeros((num_samples, 2), dtype=np.float32),
            "staged": np.zeros((num_samples, 2), dtype=bool),
            "seated": np.zeros((num_samples, 2), dtype=bool),
        }
        frames: list[np.ndarray] = []
        steps_per_frame = round(1.0 / (args_cli.fps * base_env.physics_dt))
        scene_update = base_env.scene.update
        sample = 0
        viz = None
        if record:
            viz = next(v for v in base_env.sim.visualizers if v.cfg.backend.visualizer_type == "kit")

        def log_and_capture() -> None:
            nonlocal sample
            offset = command_term.tool_offset_t()
            logs["offset"][sample] = offset.cpu().numpy()
            logs["distance"][sample] = offset.norm(dim=-1).cpu().numpy()
            logs["yaw_error"][sample] = np.degrees(command_term.yaw_error().cpu().numpy())
            forces = saw_contact.data.net_forces_w.torch
            logs["contact_force"][sample] = forces.norm(dim=-1).max(dim=-1)[0].cpu().numpy()
            logs["staged"][sample] = command_term.staged.cpu().numpy()
            logs["seated"][sample] = command_term.succeeded.cpu().numpy()
            if viz is not None and sample % steps_per_frame == 0:
                frames.append(np.array(viz.render_rgb_array(), copy=True))
            sample += 1

        def update_and_log(*args, **kwargs) -> None:
            scene_update(*args, **kwargs)
            if sample < num_samples:
                log_and_capture()

        def scripted_approach() -> torch.Tensor:
            """Normalized velocity action of a proportional approach to the current goal, shape (num_envs, 4)."""
            if args_cli.baseline == "airframe":
                error = mdp.airframe_rest_error_v(base_env, **rest_params)
            else:
                error = mdp.tool_goal_error_v(base_env, command_name)
            velocity = args_cli.baseline_gain * error
            speed = velocity.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            velocity = velocity * (speed.clamp(max=args_cli.baseline_speed) / speed)
            yaw_rate = args_cli.baseline_gain * command_term.yaw_error().unsqueeze(-1)
            cfg = base_env.cfg.actions.velocity_command
            action = torch.cat([velocity / cfg.max_velocity, yaw_rate / cfg.max_yaw_rate], dim=-1)
            return action.clamp(-1.0, 1.0)

        ended = {OFF: None, ON: None}
        with torch.inference_mode():
            base_env.reset()
            obs = env.get_observations()
            if hasattr(policy, "reset"):
                with contextlib.suppress(Exception):
                    policy.reset(torch.ones(2, dtype=torch.bool, device=device))
            targets = command_term.target_pos_w.cpu().numpy()
            print(f"[INFO] Branch seats (scripted, policy): {targets[OFF].round(2)}, {targets[ON].round(2)}")
            if record:
                # Look at the middle of the scene (between the seats, shifted back and up to take in the
                # starts and the airframes), from the -y side, turned off side-on to show the branches' length.
                spawn = np.array(args_cli.spawn_offset)
                # The mouths range from the start to the staging point; the airframes fly above them.
                staging = -command_term.cfg.staging_distance
                centre_x = 0.5 * (min(spawn[0], staging) + max(spawn[0], 0.0))
                centre = targets.mean(axis=0) + np.array([centre_x, 0.0, 0.45 + 0.5 * max(spawn[2], 0.0)])
                azimuth = math.radians(args_cli.camera_azimuth_deg)
                view_dir = np.array([math.sin(azimuth), math.cos(azimuth), 0.0])
                eye = centre - args_cli.camera_distance * view_dir + np.array([0.0, 0.0, args_cli.camera_height])
                across = targets[ON] - targets[OFF]
                # Image-right is view_dir x up, so the policy vehicle is on the left when this is negative.
                on_is_left = float(np.cross(view_dir, [0.0, 0.0, 1.0]) @ across) < 0.0
                base_env.sim.set_camera_view(eye=tuple(eye), target=tuple(centre))
                # The first frames after start-up are blank while the renderer warms up.
                for _ in range(20):
                    viz.render_rgb_array()
            log_and_capture()
            base_env.scene.update = update_and_log

            for t in range(num_steps):
                actions = policy(obs)
                actions[OFF] = scripted_approach()[OFF]
                obs, _, dones, _ = env.step(actions)
                for env_i in (OFF, ON):
                    if dones[env_i] and ended[env_i] is None:
                        reasons = [
                            name
                            for name in base_env.termination_manager.active_terms
                            if base_env.termination_manager.get_term(name)[env_i]
                        ]
                        ended[env_i] = ((t + 1) * base_env.step_dt, ", ".join(reasons))
                        warnings.warn(
                            f"{LABELS[env_i]}: episode ended at {ended[env_i][0]:.1f} s ({reasons})", stacklevel=1
                        )
                if hasattr(policy, "reset"):
                    with contextlib.suppress(Exception):
                        policy.reset(dones)

        base_env.scene.update = scene_update
        dt = base_env.physics_dt
        env.close()

        # ------------------------------------------------------------------ #
        # Summary
        # ------------------------------------------------------------------ #
        time_axis = np.arange(num_samples) * dt
        # A vehicle whose episode ended was respawned: nothing after that counts.
        for env_i in (OFF, ON):
            if ended[env_i] is not None:
                logs["seated"][time_axis >= ended[env_i][0], env_i] = False
        seated = {env_i: first_time(time_axis, logs["seated"][:, env_i]) for env_i in (OFF, ON)}
        print(
            f"\n--- Seating {args_cli.branch} from {args_cli.spawn_offset} m,"
            f" scripted approach on the {args_cli.baseline} ---"
        )
        for env_i in (OFF, ON):
            tail = time_axis >= time_axis[-1] - 2.0
            print(
                f"  {LABELS[env_i]:18s} seated at={seated[env_i]:5.1f} s"
                f"  distance over the last 2 s: mean={100 * logs['distance'][tail, env_i].mean():5.1f} cm"
                f" max={100 * logs['distance'][tail, env_i].max():5.1f} cm"
                f"  peak contact force={logs['contact_force'][:, env_i].max():5.1f} N"
                + (f"  EPISODE ENDED at {ended[env_i][0]:.1f} s: {ended[env_i][1]}" if ended[env_i] else "")
            )
        np.savez(os.path.join(output_dir, f"{OUTPUT_NAME}.npz"), time=time_axis, **logs)

        # ------------------------------------------------------------------ #
        # Static plot
        # ------------------------------------------------------------------ #
        fig, ax = plt.subplots(figsize=(12.8, 4.8), dpi=150, facecolor=SURFACE)
        draw_distance_plot(ax, time_axis, logs["distance"], seated, success_radius)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{OUTPUT_NAME}.png"))
        plt.close(fig)
        print(f"[INFO] Plot saved: {os.path.join(output_dir, f'{OUTPUT_NAME}.png')}")
        if not record:
            return

        # ------------------------------------------------------------------ #
        # Videos
        # ------------------------------------------------------------------ #
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        from PIL import Image

        frame_time = np.arange(len(frames)) / args_cli.fps
        still = frames[int(np.argmin(np.abs(frame_time - args_cli.still_s)))]
        Image.fromarray(still).save(os.path.join(output_dir, f"{OUTPUT_NAME}_still.png"))

        hold = [0] * round(args_cli.hold_s * args_cli.fps)
        order = hold + list(range(len(frames)))
        raw_path = os.path.join(output_dir, f"{OUTPUT_NAME}_raw.mp4")
        ImageSequenceClip([frames[i] for i in order], fps=args_cli.fps).write_videofile(
            raw_path, codec="libx264", audio=False, logger=None
        )
        print(f"[INFO] Raw video saved: {raw_path}")

        # Scene on top, plot underneath, in one 16:9 figure that is redrawn for every frame.
        plot_height = round(width * 9 / 16) - height
        if plot_height < 200:
            warnings.warn(
                "The scene leaves no room for the plot in a 16:9 frame; skipping the slide video.", stacklevel=1
            )
            return
        dpi = 100
        fig = plt.figure(figsize=(width / dpi, (height + plot_height) / dpi), dpi=dpi, facecolor=SURFACE)
        scene_ax = fig.add_axes((0.0, plot_height / (height + plot_height), 1.0, height / (height + plot_height)))
        scene_ax.axis("off")
        image = scene_ax.imshow(frames[0], aspect="auto")
        left = ON if on_is_left else OFF
        readouts = {}
        for env_i in (OFF, ON):
            x = 0.25 if env_i == left else 0.75
            # The outline carries the series color; the text stays in ink.
            box = dict(boxstyle="round,pad=0.45", facecolor=SURFACE, edgecolor=COLORS[env_i], linewidth=3)
            scene_ax.text(
                x, 0.91, LABELS[env_i], transform=scene_ax.transAxes, ha="center", va="center",
                fontsize=24, fontweight="bold", color=INK, bbox=box,
            )  # fmt: skip
            box = dict(boxstyle="round,pad=0.45", facecolor=SURFACE, edgecolor="none", alpha=0.9)
            readouts[env_i] = scene_ax.text(
                x, 0.07, "", transform=scene_ax.transAxes, ha="center", va="center", fontsize=18, color=INK, bbox=box
            )
        plot_ax = fig.add_axes((0.08, 0.065, 0.90, plot_height / (height + plot_height) - 0.09))
        lines = draw_distance_plot(plot_ax, time_axis, logs["distance"], seated, success_radius)
        dots = {
            env_i: plot_ax.plot([], [], "o", color=COLORS[env_i], markersize=10, mec=SURFACE, mew=2)[0]
            for env_i in (OFF, ON)
        }

        composed = []
        for i in order:
            now = frame_time[i]
            image.set_data(frames[i])
            shown = time_axis <= now + 1e-6
            k = int(np.searchsorted(time_axis, now + 1e-6)) - 1
            for env_i in (OFF, ON):
                distance_now = float(np.interp(now, time_axis, logs["distance"][:, env_i]))
                lines[env_i].set_data(
                    np.append(time_axis[shown], now), np.append(logs["distance"][shown, env_i], distance_now)
                )
                dots[env_i].set_data([now], [distance_now])
                if ended[env_i] is not None and now >= ended[env_i][0]:
                    state = "failed, restarted"
                elif logs["seated"][: k + 1, env_i].any() and distance_now < success_radius:
                    state = "branch seated"
                elif logs["staged"][k, env_i]:
                    state = "feeding onto the branch"
                else:
                    state = "lining up behind the branch"
                readouts[env_i].set_text(f"{state}  ·  {100 * distance_now:4.0f} cm from the seat")
            fig.canvas.draw()
            composed.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
        slide_path = os.path.join(output_dir, f"{OUTPUT_NAME}.mp4")
        ImageSequenceClip(composed, fps=args_cli.fps).write_videofile(
            slide_path, codec="libx264", audio=False, logger=None
        )
        print(f"[INFO] Slide video saved: {slide_path}")


if __name__ == "__main__":
    main()
