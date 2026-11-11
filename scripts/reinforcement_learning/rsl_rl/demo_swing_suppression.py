# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Side-by-side demonstration of chainsaw swing suppression on and off, made for a presentation slide.

Purpose
-------
Two M350 + Mk3 chainsaw vehicles hover next to each other on their target poses. Both chainsaws are
released from rest at the same swing angle, at the same instant:

* **Suppression off** -- a plain position hold: the velocity command is ``--hold_gain`` times the
  position error (and likewise for the yaw), as a conventional autopilot would hold a waypoint. It
  knows nothing about the payload, so the swing decays through the hinge damping alone.
* **Suppression on**  -- the trained swing-suppression policy
  (``IsaacContrib-TrackTrajectory-Matrice-ChainsawSwing``) commands the airframe.

Both send velocity commands to the same Lee velocity controller in one physics session, so the only
difference between them is who commands the airframe. The baseline is not zero commands, because a
velocity hold does not hold position: the swinging payload carries the airframe off the target.
Gusts and swing kicks are off (play mode), the training-time safety radius is lifted so that neither
vehicle is reset mid-video, and the debug markers are hidden unless ``--markers`` is passed.

Outputs
-------
Written to ``--output_dir`` (default ``<checkpoint_dir>/swing_demo``):

* ``swing_demo.mp4``      -- the slide video: the rendered scene with a label over each vehicle and the
  swing angle of both plotted underneath, drawn as the video plays.
* ``swing_demo_raw.mp4``  -- the rendered scene alone, for adding your own captions.
* ``swing_demo.png``      -- the swing-angle plot alone, as a static figure.
* ``swing_demo_still.png`` -- one rendered frame, taken ``--still_s`` after the release.
* ``swing_demo.npz``      -- the logged time series.

Pass ``--no_video`` to skip rendering and write only the plot and the data.

Usage::

    uv run python scripts/reinforcement_learning/rsl_rl/demo_swing_suppression.py \\
        --checkpoint logs/rsl_rl/matrice_chainsaw_swing/<run>/model_1400.pt

    # recurrent policy, a 40 deg fore-aft swing, watched live in the Kit viewport
    uv run python scripts/reinforcement_learning/rsl_rl/demo_swing_suppression.py \\
        --agent rsl_rl_rnn_cfg_entry_point --checkpoint /path/to/model_1999.pt \\
        --swing_deg 40 --swing_heading_deg 0 --viz kit
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

parser = argparse.ArgumentParser(description="Side-by-side chainsaw swing suppression demo: policy on vs. off.")
parser.add_argument(
    "--task",
    type=str,
    default="IsaacContrib-TrackTrajectory-Matrice-ChainsawSwingHold",
    help="Gym task ID of a chainsaw swing-suppression task (play-mode overrides are applied).",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--swing_deg", type=float, default=35.0, help="Swing angle the chainsaws are released from [deg].")
parser.add_argument(
    "--swing_heading_deg",
    type=float,
    default=0.0,
    help="Direction of the swing: 0 is fore-aft (csTubePitch), across the camera view; 90 is sideways"
    " (csTubeRoll, limited to 45 deg), towards the camera.",
)
parser.add_argument("--duration_s", type=float, default=12.0, help="Length of the demo after the release [s].")
parser.add_argument("--hold_gain", type=float, default=1.0, help="Gain of the baseline position hold [1/s].")
parser.add_argument("--spacing", type=float, default=3.0, help="Distance between the two vehicles [m].")
parser.add_argument("--settle_deg", type=float, default=3.0, help="Swing amplitude counted as settled [deg].")
parser.add_argument("--fps", type=int, default=25, choices=[10, 20, 25, 50], help="Frame rate of the videos.")
parser.add_argument("--resolution", type=int, nargs=2, default=[1920, 720], help="Size of the rendered scene [px].")
parser.add_argument("--camera_distance", type=float, default=7.0, help="Distance of the camera from the vehicles [m].")
parser.add_argument("--hold_s", type=float, default=1.0, help="Time the first frame is held before the release [s].")
parser.add_argument("--still_s", type=float, default=0.0, help="Time after the release of the still frame [s].")
parser.add_argument("--markers", action="store_true", help="Draw the task's debug markers (target pose frames).")
parser.add_argument("--no_video", action="store_true", help="Skip rendering; write only the plot and the data.")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
# Anything not parsed here is passed on to Hydra as config overrides.
args_cli, hydra_args = parser.parse_known_args()

HINGE_JOINT_NAMES = ["csTubePitch", "csTubeRoll"]
SWING_TERM_NAME = "swing_suppression"

# Environment 0 flies the baseline position hold, environment 1 the policy.
OFF, ON = 0, 1
LABELS = {OFF: "Swing suppression OFF", ON: "Swing suppression ON"}
SUBLABELS = {OFF: "position hold only", ON: "RL policy"}
# Slots 1 and 2 of the dataviz categorical palette (blue, orange), with its light chart surface and inks.
COLORS = {ON: "#2a78d6", OFF: "#eb6834"}
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a63", "#e4e3dd"


def reset_swing_fixed(
    env: "ManagerBasedRLEnv",
    env_ids: torch.Tensor,
    amplitude: float,
    heading: float,
    asset_cfg: "SceneEntityCfg",
) -> None:
    """Release a two-axis pendulum from rest at the same swing angle and direction in every environment.

    Args:
        env: The manager-based RL environment instance.
        env_ids: Indices of the environments to reset.
        amplitude: Swing angle at the release [rad].
        heading: Direction of the swing in the plane of the two hinge axes [rad]; zero is the outer axis.
        asset_cfg: SceneEntityCfg identifying the asset and exactly two hinge joints, outer axis first.
    """
    from isaaclab_tasks.contrib.drone_arl.mdp.utils import resolve_env_ids

    asset = env.scene[asset_cfg.name]
    env_ids = resolve_env_ids(env_ids, env.num_envs, asset.device)
    joint_pos = asset.data.default_joint_pos.torch[env_ids[:, None], asset_cfg.joint_ids].clone()
    joint_pos[:, 0] += amplitude * math.cos(heading)
    joint_pos[:, 1] += amplitude * math.sin(heading)
    limits = asset.data.soft_joint_pos_limits.torch[env_ids[:, None], asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])
    asset.write_joint_position_to_sim_index(position=joint_pos, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(
        velocity=torch.zeros_like(joint_pos), joint_ids=asset_cfg.joint_ids, env_ids=env_ids
    )


def settle_time(time_axis: np.ndarray, swing: np.ndarray, threshold: float) -> float:
    """Time after which the swing amplitude stays below the threshold [s]; NaN if it never does."""
    above = np.where(swing >= threshold)[0]
    if len(above) == 0:
        return 0.0
    return float(time_axis[above[-1] + 1]) if above[-1] < len(time_axis) - 1 else float("nan")


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


def draw_swing_plot(ax: plt.Axes, time_axis: np.ndarray, angle: np.ndarray, settle: dict[int, float]) -> dict:
    """Draw the signed swing angle of both vehicles and return the line artists by environment."""
    style_axes(ax)
    limit = 5.0 * math.ceil(np.abs(angle).max() / 5.0 + 0.2)
    ax.axhline(0.0, color=GRID, linewidth=1.5)
    lines = {}
    for env_i in (OFF, ON):
        (lines[env_i],) = ax.plot(time_axis, angle[:, env_i], color=COLORS[env_i], linewidth=2.5, label=LABELS[env_i])
    ax.set_xlim(0.0, time_axis[-1])
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("Time since release [s]", color=INK_MUTED, fontsize=14)
    ax.set_ylabel("Chainsaw swing angle [deg]", color=INK_MUTED, fontsize=14)
    legend_labels = [
        f"{LABELS[env_i]}  —  "
        + (
            f"settled in {settle[env_i]:.1f} s"
            if not math.isnan(settle[env_i])
            else f"still swinging after {time_axis[-1]:.0f} s"
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


def main():
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
        from isaaclab.managers import SceneEntityCfg

        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

        env_cfg.scene.num_envs = 2
        env_cfg.scene.env_spacing = args_cli.spacing
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.seed = agent_cfg.seed
        # Same target heading for both vehicles (zero yaw), so they face the same way.
        env_cfg.commands.trajectory.yaw_mode = "fixed"
        if not args_cli.markers:
            env_cfg.commands.trajectory.debug_vis = False
            env_cfg.rewards.drone_end_pose.params["debug_vis"] = False
        # The same release in both environments, in place of the task's own swing reset.
        for name in ("reset_chainsaw_swing_pitch", "reset_chainsaw_swing_roll", "reset_chainsaw_swing_graded"):
            if hasattr(env_cfg.events, name):
                setattr(env_cfg.events, name, None)
        env_cfg.events.reset_chainsaw_swing_fixed = EventTerm(
            func=reset_swing_fixed,
            mode="reset",
            params={
                "amplitude": math.radians(args_cli.swing_deg),
                "heading": math.radians(args_cli.swing_heading_deg),
                "asset_cfg": SceneEntityCfg("robot", joint_names=HINGE_JOINT_NAMES),
            },
        )
        # A training bound on the hold; here it would reset a vehicle in the middle of the video.
        env_cfg.terminations.safety_radius = None
        # The episode must outlast the demo, or the reset would show up in the video.
        env_cfg.episode_length_s = max(env_cfg.episode_length_s, args_cli.duration_s + 1.0)

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        if args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        output_dir = os.path.abspath(args_cli.output_dir or os.path.join(os.path.dirname(resume_path), "swing_demo"))
        os.makedirs(output_dir, exist_ok=True)
        print(f"[INFO] Loading checkpoint: {resume_path}")

        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        base_env: ManagerBasedRLEnv = env.unwrapped  # type: ignore[assignment]

        from rsl_rl.runners import OnPolicyRunner

        from isaaclab_tasks.contrib.drone_arl.mdp.observations import trajectory_position_error_v, trajectory_yaw_error

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=base_env.device)

        device = base_env.device
        robot = base_env.scene["robot"]
        command_term = base_env.command_manager.get_term("trajectory")
        action_cfg = base_env.cfg.actions.velocity_command
        swing_term = base_env.reward_manager.get_term_cfg(SWING_TERM_NAME).func
        hinge_ids = [robot.find_joints(name)[0][0] for name in HINGE_JOINT_NAMES]
        num_steps = round(args_cli.duration_s / base_env.step_dt)
        heading = math.radians(args_cli.swing_heading_deg)

        # ------------------------------------------------------------------ #
        # Logging and frame capture, once per physics step
        # ------------------------------------------------------------------ #
        # The environment logs nothing and renders at most once per policy step (10 Hz), which makes a
        # jagged plot and a jerky video. ``scene.update`` runs after every physics step, with the asset
        # data refreshed, so the swing is logged there (100 Hz) and a frame captured every few steps.
        num_samples = num_steps * base_env.cfg.decimation + 1
        logs = {
            "swing": np.zeros((num_samples, 2), dtype=np.float32),
            "angle": np.zeros((num_samples, 2), dtype=np.float32),
            "hinge": np.zeros((num_samples, 2, 2), dtype=np.float32),
            "pos_error": np.zeros((num_samples, 2), dtype=np.float32),
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
            amplitude, _ = swing_term.swing_amplitude(base_env)
            hinge = robot.data.joint_pos.torch[:, hinge_ids]
            logs["swing"][sample] = np.degrees(amplitude.cpu().numpy())
            logs["hinge"][sample] = np.degrees(hinge.cpu().numpy())
            # Hinge angle along the direction of the release, which is what the eye follows in the video.
            logs["angle"][sample] = np.degrees(
                (hinge[:, 0] * math.cos(heading) + hinge[:, 1] * math.sin(heading)).cpu().numpy()
            )
            pos = robot.data.root_pos_w.torch - base_env.scene.env_origins
            logs["pos_error"][sample] = (command_term.pos_ref_w - pos).norm(dim=-1).cpu().numpy()
            if viz is not None and sample % steps_per_frame == 0:
                frames.append(np.array(viz.render_rgb_array(), copy=True))
            sample += 1

        def update_and_log(*args, **kwargs) -> None:
            scene_update(*args, **kwargs)
            if sample < num_samples:
                log_and_capture()

        def position_hold() -> torch.Tensor:
            """Normalized velocity action of a proportional position and yaw hold, shape (num_envs, 4)."""
            velocity = args_cli.hold_gain * trajectory_position_error_v(base_env, "trajectory")
            yaw_rate = args_cli.hold_gain * trajectory_yaw_error(base_env, "trajectory").reshape(-1, 1)
            action = torch.cat([velocity / action_cfg.max_velocity, yaw_rate / action_cfg.max_yaw_rate], dim=-1)
            return action.clamp(-1.0, 1.0)

        with torch.inference_mode():
            base_env.reset()
            obs = env.get_observations()
            if hasattr(policy, "reset"):
                with contextlib.suppress(Exception):
                    policy.reset(torch.ones(2, dtype=torch.bool, device=device))
            root_pos = robot.data.root_pos_w.torch.cpu().numpy()
            print(f"[INFO] Vehicle positions (off, on): {root_pos[OFF].round(2)}, {root_pos[ON].round(2)}")
            if record:
                # Look at the point midway between the vehicles, level with the middle of the chainsaw,
                # side-on to the line between them and from slightly above.
                centre = root_pos.mean(axis=0) + np.array([0.0, 0.0, -0.55])
                across = root_pos[ON] - root_pos[OFF]
                view_dir = np.array([-across[1], across[0], 0.0]) / np.linalg.norm(across[:2])
                eye = centre - args_cli.camera_distance * view_dir + np.array([0.0, 0.0, 0.4])
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
                actions[OFF] = position_hold()[OFF]
                obs, _, dones, _ = env.step(actions)
                if dones.any():
                    warnings.warn(f"Episode ended at step {t + 1}: {dones.cpu().numpy()}", stacklevel=1)
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
        settle = {env_i: settle_time(time_axis, logs["swing"][:, env_i], args_cli.settle_deg) for env_i in (OFF, ON)}
        print(
            f"\n--- Swing released from {args_cli.swing_deg:.0f} deg, settled below {args_cli.settle_deg:.0f} deg ---"
        )
        for env_i in (OFF, ON):
            print(
                f"  {LABELS[env_i]:24s} settle={settle[env_i]:5.1f} s"
                f"  swing at end={logs['swing'][-1, env_i]:5.1f} deg"
                f"  max position error={logs['pos_error'][:, env_i].max():.2f} m"
            )
        np.savez(os.path.join(output_dir, "swing_demo.npz"), time=time_axis, **logs)

        # ------------------------------------------------------------------ #
        # Static plot
        # ------------------------------------------------------------------ #
        fig, ax = plt.subplots(figsize=(12.8, 4.8), dpi=150, facecolor=SURFACE)
        draw_swing_plot(ax, time_axis, logs["angle"], settle)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "swing_demo.png"))
        plt.close(fig)
        print(f"[INFO] Plot saved: {os.path.join(output_dir, 'swing_demo.png')}")
        if not record:
            return

        # ------------------------------------------------------------------ #
        # Videos
        # ------------------------------------------------------------------ #
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
        from PIL import Image

        frame_time = np.arange(len(frames)) / args_cli.fps
        still = frames[int(np.argmin(np.abs(frame_time - args_cli.still_s)))]
        Image.fromarray(still).save(os.path.join(output_dir, "swing_demo_still.png"))

        hold = [0] * round(args_cli.hold_s * args_cli.fps)
        order = hold + list(range(len(frames)))
        raw_path = os.path.join(output_dir, "swing_demo_raw.mp4")
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
        plot_ax = fig.add_axes((0.06, 0.065, 0.92, plot_height / (height + plot_height) - 0.09))
        lines = draw_swing_plot(plot_ax, time_axis, logs["angle"], settle)
        dots = {
            env_i: plot_ax.plot([], [], "o", color=COLORS[env_i], markersize=10, mec=SURFACE, mew=2)[0]
            for env_i in (OFF, ON)
        }

        composed = []
        for i in order:
            now = frame_time[i]
            image.set_data(frames[i])
            shown = time_axis <= now + 1e-6
            for env_i in (OFF, ON):
                angle_now = float(np.interp(now, time_axis, logs["angle"][:, env_i]))
                swing_now = float(np.interp(now, time_axis, logs["swing"][:, env_i]))
                lines[env_i].set_data(
                    np.append(time_axis[shown], now), np.append(logs["angle"][shown, env_i], angle_now)
                )
                dots[env_i].set_data([now], [angle_now])
                readouts[env_i].set_text(f"{SUBLABELS[env_i]}  ·  swing amplitude {swing_now:4.1f}°")
            fig.canvas.draw()
            composed.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
        slide_path = os.path.join(output_dir, "swing_demo.mp4")
        ImageSequenceClip(composed, fps=args_cli.fps).write_videofile(
            slide_path, codec="libx264", audio=False, logger=None
        )
        print(f"[INFO] Slide video saved: {slide_path}")


if __name__ == "__main__":
    main()
