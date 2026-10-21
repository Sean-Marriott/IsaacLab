# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

import argparse
import contextlib
import csv
import importlib.metadata as metadata
import math
import os
import sys
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
parser.add_argument(
    "--obs_csv", type=str, default=None, help="Write the policy observations and actions of one environment to a CSV."
)
parser.add_argument("--obs_csv_env", type=int, default=0, help="Index of the environment written by --obs_csv.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True


# Call an external callback if requested. This gives opportunity to external code to register the environments
# The function is expected to return a list of arguments that were not consumed by the callback.
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# clear out sys.argv for Hydra
# The remaining arguments are the arguments that were not consumed by both this scripts
# argparser and (optionally) the external callback function.
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

# Check for installed RSL-RL version
installed_version = metadata.version("rsl-rl-lib")

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


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        # grab task name for checkpoint path
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        # handle deprecated configurations
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # export the trained policy to JIT and ONNX formats
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

        if version.parse(installed_version) >= version.parse("4.0.0"):
            # use the new export functions for rsl-rl >= 4.0.0
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            policy_nn = None  # Not needed for rsl-rl >= 4.0.0
        else:
            # extract the neural network for rsl-rl < 4.0.0
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic

            # extract the normalizer
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            # export to JIT and ONNX
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        dt = env.unwrapped.step_dt

        # reset environment
        obs = env.get_observations()
        timestep = 0

        # observation log: one row per policy step for one environment
        csv_file = csv_writer = None
        if args_cli.obs_csv:
            unwrapped = env.unwrapped
            csv_env = args_cli.obs_csv_env
            obs_columns = _obs_csv_columns(unwrapped)
            action_columns = [f"action_{a}" for a in _DRONE_ACTION_LABELS]
            if len(action_columns) != unwrapped.action_manager.total_action_dim:
                action_columns = [f"action_{i}" for i in range(unwrapped.action_manager.total_action_dim)]
            # The trajectory command's clock, if the task has one. The command advances it right after
            # evaluating the reference, so after a step the observation is from one step earlier.
            ref_command = None
            if "trajectory" in unwrapped.command_manager.active_terms:
                ref_command = unwrapped.command_manager.get_term("trajectory")
            ref_t = ref_command._time[csv_env].item() if ref_command is not None else float("nan")
            os.makedirs(os.path.dirname(os.path.abspath(args_cli.obs_csv)), exist_ok=True)
            csv_file = open(args_cli.obs_csv, "w", newline="")  # noqa: SIM115 (closed after the loop)
            csv_writer = csv.writer(csv_file)
            # pose_time is the timestamp of the pose estimate on the vehicle; in sim it is the sim time
            csv_writer.writerow(["time", "pose_time", "ref_t"] + obs_columns + action_columns)
            print(f"[INFO] Writing observations of environment {csv_env} to: {os.path.abspath(args_cli.obs_csv)}")

        # simulate environment
        try:
            while True:
                start_time = time.time()
                # run everything in inference mode
                with torch.inference_mode():
                    # agent stepping
                    actions = policy(obs)
                    if csv_writer is not None:
                        sim_time = timestep * dt
                        policy_obs = unwrapped.obs_buf["policy"][csv_env].tolist()
                        csv_writer.writerow(
                            [f"{sim_time:.6f}", f"{sim_time:.6f}", f"{ref_t:.4f}"]
                            + [f"{v:.6g}" for v in policy_obs + actions[csv_env].tolist()]
                        )
                    # env stepping
                    obs, _, dones, _ = env.step(actions)
                    if csv_writer is not None and ref_command is not None:
                        ref_t = ref_command._time[csv_env].item() - dt
                    # reset recurrent states for episodes that have terminated
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    else:
                        policy_nn.reset(dones)
                timestep += 1
                if args_cli.video and timestep == args_cli.video_length:
                    break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            # close the simulator
            env.close()
        except KeyboardInterrupt:
            pass
        finally:
            if csv_file is not None:
                csv_file.close()


if __name__ == "__main__":
    main()
