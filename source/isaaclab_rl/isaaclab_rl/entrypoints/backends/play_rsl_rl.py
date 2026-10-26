# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL playback backend of the unified reinforcement learning entrypoint."""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import math
import os

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.utils import to_dict
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.seed import configure_seed
from isaaclab.utils.string import list_intersection

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli

from ...rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from ...utils.wandb import is_wandb_checkpoint, resolve_wandb_checkpoint
from ..common import (
    CHECKPOINT_SELECTORS,
    add_common_play_args,
    apply_env_overrides,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    enable_cameras_for_video,
    normalize_task_name,
    pre_launch_video_config,
    resolve_checkpoint_selector,
    resolve_published_checkpoint,
    run_playback,
    set_hydra_args,
    show_run_summary,
    startup_screen,
)
from . import cli_args_rsl_rl as cli_args

logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# Column names for the --obs_csv log, per observation term. These follow the log format of the drone
# deployment stack, so a sim run can be compared with a flight column by column. Terms not listed
# here are written as ``<term>_<i>``.
_XYZ = ("x", "y", "z")
_DRONE_ACTION_LABELS = ("vx", "vy", "vz", "yaw_rate")
_OBS_CSV_LABELS = {
    "pos_error_v": [f"pos_err_v_{a}" for a in _XYZ],
    "vel_error_v": [f"vel_err_v_{a}" for a in _XYZ],
    "ref_vel_v": [f"ref_vel_v_{a}" for a in _XYZ],
    "ref_acc_v": [f"ref_acc_v_{a}" for a in _XYZ],
    "yaw_error": ["yaw_err"],
    "base_ang_vel": [f"ang_vel_b_{a}" for a in _XYZ],
    "base_roll_pitch": ["roll", "pitch"],
    "last_action": [f"last_action_{a}" for a in _DRONE_ACTION_LABELS],
}


def _obs_csv_columns(env) -> list[str]:
    """Names of the policy observation elements, in the order of the concatenated observation."""
    obs_manager = env.observation_manager
    columns = []
    for name, cfg, dims in zip(
        obs_manager.active_terms["policy"],
        obs_manager._group_obs_term_cfgs["policy"],
        obs_manager.group_obs_term_dim["policy"],
    ):
        dim = math.prod(dims)
        labels = _OBS_CSV_LABELS.get(name)
        asset_cfg = cfg.params.get("asset_cfg")
        if name == "lookahead_v":
            # one point per look-ahead time of the trajectory command
            times = env.command_manager.get_term(cfg.params["command_name"]).cfg.lookahead_times_s
            labels = [f"lookahead_{t:.1f}s_{a}" for t in times for a in _XYZ]
        elif name.endswith(("joint_pos", "joint_vel")) and asset_cfg is not None:
            # named after the resolved joints, which are in articulation order
            joint_names = env.scene[asset_cfg.name].joint_names
            ids = asset_cfg.joint_ids
            selected = joint_names[ids] if isinstance(ids, slice) else [joint_names[i] for i in ids]
            labels = [f"joint_{name[-3:]}_{j}" for j in selected]
        if labels is None or len(labels) != dim:
            labels = [f"{name}_{i}" for i in range(dim)]
        columns += labels
    return columns


class _ObsCsvLogger:
    """Write the policy observations and actions of one environment to a CSV, one row per policy step."""

    def __init__(self, path: str, env, env_index: int):
        self._env_index = env_index
        self._dt = env.step_dt
        self._obs_buf = lambda: env.obs_buf["policy"]
        self._step = 0
        obs_columns = _obs_csv_columns(env)
        action_columns = [f"action_{a}" for a in _DRONE_ACTION_LABELS]
        if len(action_columns) != env.action_manager.total_action_dim:
            action_columns = [f"action_{i}" for i in range(env.action_manager.total_action_dim)]
        # The trajectory command's clock, if the task has one. The command advances it right after
        # evaluating the reference, so after a step the observation is from one step earlier.
        self._ref_command = None
        if "trajectory" in env.command_manager.active_terms:
            self._ref_command = env.command_manager.get_term("trajectory")
        self._ref_t = self._ref_command._time[env_index].item() if self._ref_command is not None else float("nan")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._file = open(path, "w", newline="")  # noqa: SIM115 (closed by close())
        self._writer = csv.writer(self._file)
        # pose_time is the timestamp of the pose estimate on the vehicle; in sim it is the sim time
        self._writer.writerow(["time", "pose_time", "ref_t"] + obs_columns + action_columns)
        logger.info(f"Writing observations of environment {env_index} to: {os.path.abspath(path)}")

    def log(self, actions) -> None:
        """Record the current observation and the action computed from it."""
        sim_time = self._step * self._dt
        policy_obs = self._obs_buf()[self._env_index].tolist()
        self._writer.writerow(
            [f"{sim_time:.6f}", f"{sim_time:.6f}", f"{self._ref_t:.4f}"]
            + [f"{v:.6g}" for v in policy_obs + actions[self._env_index].tolist()]
        )
        self._step += 1

    def after_step(self) -> None:
        """Advance the reference clock after the environment has stepped."""
        if self._ref_command is not None:
            self._ref_t = self._ref_command._time[self._env_index].item() - self._dt

    def close(self) -> None:
        self._file.close()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse RSL-RL playback arguments."""
    parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RSL-RL.")
    add_common_play_args(
        parser,
        agent_default="rsl_rl_cfg_entry_point",
        agent_help="Name of the RL agent configuration entry point.",
    )
    parser.add_argument(
        "--external_callback", default=None, help="Fully qualified path to an externally defined callback."
    )
    parser.add_argument(
        "--obs_csv",
        type=str,
        default=None,
        help="Write the policy observations and actions of one environment to a CSV.",
    )
    parser.add_argument("--obs_csv_env", type=int, default=0, help="Index of the environment written by --obs_csv.")
    cli_args.add_rsl_rl_args(parser)
    add_launcher_args(parser)
    remaining_args_env_registration = cli_args.register_external_tasks(argv)
    args_cli, remaining_args = setup_preset_cli(parser, argv)
    if remaining_args_env_registration is not None:
        _, remaining_args_env_registration = setup_preset_cli(parser, remaining_args_env_registration)
    enable_cameras_for_video(args_cli)
    set_hydra_args(list_intersection(remaining_args, remaining_args_env_registration))
    return args_cli


def _resolve_checkpoint(
    args_cli: argparse.Namespace, agent_cfg: RslRlBaseRunnerCfg, env_cfg: object, log_root_path: str
) -> str | None:
    """Resolve the checkpoint to play, or None when no published checkpoint exists."""
    if args_cli.checkpoint and is_wandb_checkpoint(args_cli.checkpoint):
        return resolve_wandb_checkpoint(args_cli.checkpoint)
    if args_cli.checkpoint == "pretrained":
        return resolve_published_checkpoint("rsl_rl", args_cli.task, env_cfg)
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="rsl_rl",
            task=normalize_task_name(args_cli.task),
            checkpoint_pattern=r"model_.*\.pt",
            metadata={"agent": args_cli.agent},
        )
    if args_cli.checkpoint and os.path.isdir(args_cli.checkpoint):
        return get_checkpoint_path(
            os.path.dirname(args_cli.checkpoint), os.path.basename(args_cli.checkpoint), agent_cfg.load_checkpoint
        )
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def run(argv: list[str]) -> None:
    """Play a checkpoint of an RSL-RL agent."""
    args_cli = _parse_args(argv)
    with startup_screen(args_cli, num_stages=3) as screen:
        env_cfg, agent_cfg = resolve_task_config(args_cli.task, args_cli.agent, play_mode=not args_cli.train_env_cfg)
        pre_launch_video_config(env_cfg, args_cli)
        screen.stage("Launching simulation")
        with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
            show_run_summary(screen, args_cli, env_cfg, library="rsl_rl", action="play")
            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
            apply_env_overrides(args_cli, env_cfg)
            # certain randomizations occur in the environment initialization so we set the seed here
            env_cfg.seed = agent_cfg.seed

            log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
            logger.info(f"Loading experiment from directory: {log_root_path}")
            resume_path = _resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root_path)
            if resume_path is None:
                return
            log_dir = os.path.dirname(resume_path)
            env_cfg.log_dir = log_dir
            apply_video_recording(env_cfg, log_dir, args_cli, subdir="play", checkpoint_path=resume_path)

            screen.stage("Creating environment")
            env = create_isaaclab_env(
                args_cli.task,
                env_cfg,
                args_cli,
                convert_marl_to_single_agent=isinstance(env_cfg, DirectMARLEnvCfg),
            )
            cleanup.callback(lambda: close_env(env))

            screen.stage("Loading policy")
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            logger.info(f"Loading model checkpoint from: {resume_path}")
            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(env, to_dict(agent_cfg), log_dir=None, device=agent_cfg.device)
            else:
                raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
            # configure_seed must run after runner construction so torch determinism does not disturb its initialization
            if args_cli.deterministic:
                configure_seed(env_cfg.seed, torch_deterministic=True)
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=env.unwrapped.device)

            export_model_dir = os.path.join(log_dir, "exported")
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")

            obs = env.get_observations()

            obs_csv = None
            if args_cli.obs_csv:
                obs_csv = _ObsCsvLogger(args_cli.obs_csv, env.unwrapped, args_cli.obs_csv_env)
                cleanup.callback(obs_csv.close)

            def step() -> None:
                nonlocal obs
                actions = policy(obs)
                if obs_csv is not None:
                    obs_csv.log(actions)
                obs, _, dones, _ = env.step(actions)
                if obs_csv is not None:
                    obs_csv.after_step()
                # reset recurrent states for episodes that have terminated
                policy.reset(dones)

            screen.close()
            run_playback(step, dt=env.unwrapped.step_dt, args_cli=args_cli, env_cfg=env_cfg)
