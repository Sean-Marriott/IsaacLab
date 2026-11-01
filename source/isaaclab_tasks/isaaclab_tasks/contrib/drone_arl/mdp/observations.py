# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create drone observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.images import normalize_depth

from .utils import hanging_tool_and_end_target

# ``source/isaaclab_tasks/data``, which ships the drone ARL VAE weights
_ISAACLAB_TASKS_DATA_DIR = os.path.join(os.path.dirname(__file__), *([os.pardir] * 4), "data")

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors.camera.camera import Camera
    from isaaclab.sensors.camera.tiled_camera import TiledCamera
    from isaaclab.sensors.ray_caster.multi_mesh_ray_caster import MultiMeshRayCaster
    from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_camera import MultiMeshRayCasterCamera
    from isaaclab.sensors.ray_caster.ray_caster_camera import RayCasterCamera

    from isaaclab_contrib.assets import Multirotor

    from .commands import DroneTrajectoryCommand

from isaaclab.envs.utils.io_descriptors import generic_io_descriptor, record_shape

"""
State.
"""


@generic_io_descriptor(dtype=torch.float32, observation_type="RootState", on_inspect=[record_shape])
def base_roll_pitch(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the base roll and pitch in the simulation world frame.

    Parameters:
        env: Manager-based environment providing the scene and tensors.
        asset_cfg: Scene entity config pointing to the target robot (default: "robot").

    Returns:
        torch.Tensor: Shape (num_envs, 2). Column 0 is roll, column 1 is pitch.
        Values are radians normalized to [-pi, pi], expressed in the world frame.

    Notes:
        - Euler angles are computed from asset.data.root_quat_w using XYZ convention.
        - Only roll and pitch are returned; yaw is omitted.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # extract euler angles (in world frame)
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w.torch)
    # normalize angle to [-pi, pi]
    roll = math_utils.wrap_to_pi(roll)
    pitch = math_utils.wrap_to_pi(pitch)

    return torch.cat((roll.unsqueeze(-1), pitch.unsqueeze(-1)), dim=-1)


"""
Sensors
"""


def lidar_scan(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    normalize: bool = True,
) -> torch.Tensor:
    """Lidar ranges from the given sensor.

    This returns per-ray distances [m] from the sensor origin to hit points in world frame.
    Rays without a valid hit are set to the sensor maximum distance.

    Args:
        env: The environment.
        sensor_cfg: Scene entity configuration for a :class:`MultiMeshRayCaster` sensor.
        normalize: If True, scale ranges by ``sensor.cfg.max_distance`` to [0, 1].

    Returns:
        Per-ray lidar ranges. Shape is (num_envs, num_rays).
    """
    # extract the used quantities (to enable type-hinting)
    sensor: MultiMeshRayCaster = env.scene.sensors[sensor_cfg.name]  # type: ignore[assignment]

    # Compute Euclidean range to each hit point.
    ray_vectors_w = sensor.data.ray_hits_w.torch - sensor.data.pos_w.torch.unsqueeze(1)
    ranges = torch.linalg.norm(ray_vectors_w, dim=-1)

    # Replace invalid hits (inf/nan) with max distance.
    max_distance = float(sensor.cfg.max_distance)
    ranges = torch.nan_to_num(ranges, nan=max_distance, posinf=max_distance, neginf=0.0)
    ranges = torch.clamp(ranges, min=0.0, max=max_distance)

    if normalize:
        return ranges / max_distance
    return ranges


def lidar_min_distance(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    normalize: bool = True,
) -> torch.Tensor:
    """Minimum lidar range (closest obstacle) from the given sensor.

    Args:
        env: The environment.
        sensor_cfg: Scene entity configuration for a :class:`MultiMeshRayCaster` sensor.
        normalize: If True, scale distance by ``sensor.cfg.max_distance`` to [0, 1].

    Returns:
        Minimum range per environment. Shape is (num_envs, 1).
    """
    # Reuse range computation from lidar_scan to keep behavior consistent.
    ranges = lidar_scan(env=env, sensor_cfg=sensor_cfg, normalize=False)
    min_range = torch.min(ranges, dim=1, keepdim=True).values

    if normalize:
        sensor: MultiMeshRayCaster = env.scene.sensors[sensor_cfg.name]  # type: ignore[assignment]
        max_distance = float(sensor.cfg.max_distance)
        return min_range / max_distance
    return min_range


def lidar_sector_min_distances(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    num_sectors: int = 8,
    normalize: bool = True,
) -> torch.Tensor:
    """Directional lidar feature using minimum range per azimuth sector.

    Rays are grouped by their azimuth angle in the sensor frame. For each sector,
    this returns the minimum hit distance [m]. This preserves directional obstacle
    information with a compact observation size.

    Args:
        env: The environment.
        sensor_cfg: Scene entity configuration for a :class:`MultiMeshRayCaster` sensor.
        num_sectors: Number of uniform azimuth sectors spanning [-pi, pi].
        normalize: If True, scale ranges by ``sensor.cfg.max_distance`` to [0, 1].

    Returns:
        Sector-wise minimum distances. Shape is (num_envs, num_sectors).
    """
    if num_sectors <= 0:
        raise ValueError("num_sectors must be positive.")

    sensor: MultiMeshRayCaster = env.scene.sensors[sensor_cfg.name]  # type: ignore[assignment]
    ranges = lidar_scan(env=env, sensor_cfg=sensor_cfg, normalize=False)

    # Ray directions are defined in the sensor frame and repeated across envs.
    ray_directions = sensor.ray_directions[0]
    azimuth = torch.atan2(ray_directions[:, 1], ray_directions[:, 0])

    max_distance = float(sensor.cfg.max_distance)
    sector_mins = torch.full((env.num_envs, num_sectors), max_distance, device=env.device)

    # Uniform azimuth partition in [-pi, pi].
    bin_edges = torch.linspace(-torch.pi, torch.pi, num_sectors + 1, device=env.device)
    for sector_idx in range(num_sectors):
        start_angle = bin_edges[sector_idx]
        end_angle = bin_edges[sector_idx + 1]
        if sector_idx == num_sectors - 1:
            in_sector = (azimuth >= start_angle) & (azimuth <= end_angle)
        else:
            in_sector = (azimuth >= start_angle) & (azimuth < end_angle)

        if in_sector.any():
            sector_mins[:, sector_idx] = ranges[:, in_sector].amin(dim=1)

    if normalize:
        return sector_mins / max_distance
    return sector_mins


class ImageLatentObservation(ManagerTermBase):
    """Callable observation term that returns VAE latents from camera images.

    This observation term extracts images from a configured camera sensor, normalizes them
    based on the data type, and passes them through a pre-trained VAE model to obtain
    latent representations. The VAE model is loaded once and cached on the class to avoid
    repeated disk loads across all instances.

    The term is designed to work with the Isaac Lab observation manager and integrates
    seamlessly with other observation terms in multi-modal observation spaces.

    Attributes:
        camera_sensor: The camera sensor to extract images from (TiledCamera, Camera,
            RayCasterCamera, or MultiMeshRayCasterCamera).
        data_type: Type of data to extract from the sensor (e.g., "distance_to_image_plane").
        convert_perspective_to_orthogonal: Whether to convert perspective depth to orthogonal.
        normalize: Whether to normalize images before passing to VAE.

    Example:
        To use this in an environment configuration:

        .. code-block:: python

            depth_latent = ObsTerm(
                func=mdp.ImageLatentObservation,
                params={
                    "sensor_cfg": SceneEntityCfg("depth_camera"),
                    "data_type": "distance_to_image_plane",
                    "normalize": True,
                },
            )
    """

    _model: torch.jit.ScriptModule | None = None

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the image latent observation term.

        Extracts configuration from cfg.params and caches a reference to the camera sensor
        for efficient repeated access during observation collection.

        Args:
            cfg: Configuration object containing the observation term configuration,
                including params dict with:
                - sensor_cfg (SceneEntityCfg): Scene entity config for the camera sensor.
                - data_type (str): Data type to extract from the sensor.
                - convert_perspective_to_orthogonal (bool, optional): Whether to convert
                  perspective to orthogonal depth. Defaults to False.
                - normalize (bool, optional): Whether to normalize images. Defaults to True.
            env: The manager-based RL environment instance.

        Raises:
            KeyError: If required params ("sensor_cfg", "data_type") are missing.
            RuntimeError: If the specified camera sensor is not found in the scene.
        """
        super().__init__(cfg, env)
        self.camera_sensor: TiledCamera | Camera | RayCasterCamera | MultiMeshRayCasterCamera = env.scene.sensors[
            cfg.params["sensor_cfg"].name
        ]  # type: ignore
        self.data_type: str = cfg.params["data_type"]  # type: ignore
        self.convert_perspective_to_orthogonal = bool(cfg.params.get("convert_perspective_to_orthogonal", False))
        self.normalize = bool(cfg.params.get("normalize", True))

    @classmethod
    def _get_model(cls, device):
        """Load or retrieve the cached VAE model.

        The model is loaded from disk only once per process and cached on the class.
        Subsequent calls return the cached instance, avoiding repeated I/O and model
        initialization overhead.

        Args:
            device: PyTorch device to load the model onto (e.g., "cpu", "cuda:0").

        Returns:
            Loaded VAE model as a TorchScript ScriptModule, set to evaluation mode.

        Raises:
            FileNotFoundError: If the VAE model file cannot be found at the expected path.
            RuntimeError: If the model cannot be loaded (e.g., corrupted file).
        """
        if cls._model is None:
            model_path = os.path.join(_ISAACLAB_TASKS_DATA_DIR, "drone_arl", "vae_model.pt")
            cls._model = torch.jit.load(model_path, map_location=device)
            cls._model.eval()
        return cls._model

    def __call__(self, env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, data_type: str) -> torch.Tensor:
        """Compute VAE latents for the current camera frame.

        Extracts images from the camera sensor, applies normalization if configured,
        and passes them through the VAE model to obtain latent representations.

        Args:
            env: The manager-based environment providing scene and device information.
            sensor_cfg: Scene entity config for the camera sensor (unused, already set in __init__).
            data_type: Data type to extract from the sensor (unused, already set in __init__).
            convert_perspective_to_orthogonal: Whether to convert perspective to orthogonal depth
                (unused, already set in __init__).
            normalize: Whether to normalize images (unused, already set in __init__).

        Returns:
            torch.Tensor: Latent representations from the VAE model.
                Shape is determined by the VAE architecture (typically (num_envs, latent_dim)).

        Raises:
            ValueError: If data_type is "distance_to_image_plane" but normalize is False,
                or if an unsupported data_type is encountered with normalize=True.
            RuntimeError: If the VAE model inference fails or tensors have incompatible shapes.

        Notes:
            - Images are converted to float16 before passing to the VAE for efficiency.
            - Invalid (NaN or infinite) depth values are set to 10.0 during normalization.
            - Very small depth values (< 0.02) are set to -1.0 to indicate invalid regions.
            - The parameters (sensor_cfg, data_type, etc.) are ignored here as they are
            already stored during initialization. They are included in the signature only
            to satisfy the observation manager's parameter validation.
        """
        images = self.camera_sensor.data.output[self.data_type].torch

        if (self.data_type == "distance_to_camera") and self.convert_perspective_to_orthogonal:
            images = math_utils.orthogonalize_perspective_depth(images, self.camera_sensor.data.intrinsic_matrices)

        if self.normalize:
            if self.data_type != "distance_to_image_plane":
                raise ValueError(f"Image data type: {self.data_type} not supported")
            # the VAE was trained on depth scaled over 10 m, with pixels nearer than 0.2 m marked invalid
            images = normalize_depth(images, invalid_value=10.0, max_depth=10.0)
            images[images < 0.02] = -1.0

        vae_model = self._get_model(env.device)
        with torch.no_grad():
            latents = vae_model(images.squeeze(-1).half())

        return latents


"""
Actions.
"""


@generic_io_descriptor(dtype=torch.float32, observation_type="Action", on_inspect=[record_shape])
def last_action_navigation(env: ManagerBasedEnv, action_name: str = "velocity_commands") -> torch.Tensor:
    """The last processed position/velocity/acceleration commands from the navigation action term.

    This function accesses the position/velocity/acceleration commands (vx, vy, vz, yaw_rate) that
    were computed by the NavigationAction term. This avoids duplicating the
    action processing logic.

    Args:
        env: Manager-based environment providing the action manager.
        action_name: Name of the navigation action term. Defaults to "velocity_commands".

    Returns:
        torch.Tensor: Shape (num_envs, 4) containing position/velocity/acceleration commands.
    """
    action_term = env.action_manager.get_term(action_name)
    # Access the velocity_commands property from NavigationAction
    return action_term.prev_commands


"""
Commands.
"""


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def generated_drone_commands(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Generate a body-frame direction and distance to the commanded position.

    This observation reads a command from env.command_manager identified by command_name,
    interprets its first three components as a target position in the world frame, and
    returns:
        [dir_x, dir_y, dir_z, distance]
    where dir_* is the unit vector from the current body origin to the target, expressed
    in the multirotor body (root link) frame, and distance is the Euclidean separation.

    Parameters:
        env: Manager-based RL environment providing scene and command manager.
        command_name: Name of the command term to query from the command manager.
        asset_cfg: Scene entity config for the multirotor asset (default: "robot").

    Returns:
        torch.Tensor: Shape (num_envs, 4) with body-frame unit direction (3) and distance (1).

    Frame conventions:
        - Current position is asset.data.root_pos_w relative to env.scene.env_origins (world frame).
        - Body orientation uses asset.data.root_link_quat_w to rotate world vectors into the body frame.

    Assumptions:
        - env.command_manager.get_command(command_name) returns at least three values
          representing a world-frame target position per environment.
        - A small epsilon (1e-8) is used to guard against zero-length direction vectors.
    """
    asset: Multirotor = env.scene[asset_cfg.name]
    current_position_w = asset.data.root_pos_w.torch - env.scene.env_origins
    command = env.command_manager.get_command(command_name)
    current_position_b = math_utils.quat_apply_inverse(
        asset.data.root_link_quat_w.torch, command[:, :3] - current_position_w
    )
    current_position_b_dir = current_position_b / (torch.linalg.norm(current_position_b, dim=-1, keepdim=True) + 1e-8)
    current_position_b_mag = torch.linalg.norm(current_position_b, dim=-1, keepdim=True)
    return torch.cat((current_position_b_dir, current_position_b_mag), dim=-1)


"""
Trajectory tracking.

All of these express the reference in the *yaw-only vehicle frame* rather than the full body frame.
That matches how :class:`~isaaclab_contrib.controllers.LeeVelController` interprets the velocity
action: it builds its setpoint frame from yaw alone. Using the full body frame would mix roll and
pitch into the observation-to-action mapping and force the network to undo a rotation it does not
command. Roll and pitch still reach the policy through :func:`base_roll_pitch` and the body-frame
linear and angular velocities.

The errors are taken against the frame the command tracks (see
:meth:`~isaaclab_tasks.contrib.drone_arl.mdp.commands.DroneTrajectoryCommand.tracked_frame_state`),
which is the vehicle unless the command is set to track a payload frame such as a tool tip. They are
still expressed in the vehicle frame, since that is the frame the action is in.
"""


def _vehicle_frame_quat(asset: Multirotor) -> torch.Tensor:
    """Return the yaw-only orientation of the asset. Shape is (num_envs, 4)."""
    return math_utils.yaw_quat(asset.data.root_link_quat_w.torch)


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def trajectory_position_error_v(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Position error to the reference, in the vehicle frame.

    Args:
        env: The manager-based RL environment instance.
        command_name: Name of the trajectory command term to read the reference from.
        asset_cfg: SceneEntityCfg identifying the asset.

    Returns:
        The reference position minus the position of the tracked frame [m], expressed in the
        yaw-only vehicle frame. Shape is (num_envs, 3).
    """
    asset: Multirotor = env.scene[asset_cfg.name]
    command_term: DroneTrajectoryCommand = env.command_manager.get_term(command_name)
    pos_w, _, _ = command_term.tracked_frame_state()
    error_w = command_term.command[:, :3] - pos_w
    return math_utils.quat_apply_inverse(_vehicle_frame_quat(asset), error_w)


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def trajectory_velocity_error_v(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Velocity error to the reference, in the vehicle frame.

    Args:
        env: The manager-based RL environment instance.
        command_name: Name of the trajectory command term to read the reference from.
        asset_cfg: SceneEntityCfg identifying the asset.

    Returns:
        The reference velocity minus the velocity of the tracked frame [m/s], expressed in the
        yaw-only vehicle frame. Shape is (num_envs, 3).
    """
    asset: Multirotor = env.scene[asset_cfg.name]
    command_term: DroneTrajectoryCommand = env.command_manager.get_term(command_name)
    _, _, lin_vel_w = command_term.tracked_frame_state()
    error_w = command_term.command[:, 3:6] - lin_vel_w
    return math_utils.quat_apply_inverse(_vehicle_frame_quat(asset), error_w)


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def trajectory_ref_velocity_v(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reference velocity feed-forward, in the vehicle frame.

    The optimal action is roughly ``v_ref + Kp * position_error``, so handing the policy ``v_ref``
    directly leaves it only the correction to learn.

    Args:
        env: The manager-based RL environment instance.
        command_name: Name of the trajectory command term to read the reference from.
        asset_cfg: SceneEntityCfg identifying the asset.

    Returns:
        The reference velocity [m/s] in the yaw-only vehicle frame. Shape is (num_envs, 3).
    """
    asset: Multirotor = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    return math_utils.quat_apply_inverse(_vehicle_frame_quat(asset), command[:, 3:6])


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def trajectory_ref_accel_v(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reference acceleration feed-forward, in the vehicle frame.

    This is what lets the policy lead the first-order velocity loop instead of lagging it.

    Args:
        env: The manager-based RL environment instance.
        command_name: Name of the trajectory command term to read the reference from.
        asset_cfg: SceneEntityCfg identifying the asset.

    Returns:
        The reference acceleration [m/s^2] in the yaw-only vehicle frame. Shape is (num_envs, 3).
    """
    asset: Multirotor = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    return math_utils.quat_apply_inverse(_vehicle_frame_quat(asset), command[:, 6:9])


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def trajectory_yaw_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Yaw error of the tracked frame to the reference, wrapped to (-pi, pi].

    The Lee velocity controller commands yaw *rate* and never closes a loop on yaw angle, so the
    policy has to close it and therefore needs this term.

    Args:
        env: The manager-based RL environment instance.
        command_name: Name of the trajectory command term to read the reference from.
        asset_cfg: SceneEntityCfg identifying the asset. Unused: the yaw is that of the tracked frame.

    Returns:
        The wrapped yaw error [rad]. Shape is (num_envs, 1).
    """
    command_term: DroneTrajectoryCommand = env.command_manager.get_term(command_name)
    _, quat_w, _ = command_term.tracked_frame_state()
    _, _, yaw = math_utils.euler_xyz_from_quat(quat_w)
    return math_utils.wrap_to_pi(command_term.command[:, 9] - yaw).unsqueeze(-1)


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def trajectory_yaw_error_sin_cos(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Sine and cosine of the yaw error of the tracked frame to the reference.

    The same error as :func:`trajectory_yaw_error`, encoded without its jump at +/-pi: two headings
    either side of pointing directly away from the reference give nearly the same observation instead of
    opposite ones.

    Args:
        env: The manager-based RL environment instance.
        command_name: Name of the trajectory command term to read the reference from.

    Returns:
        The sine and cosine of the yaw error. Shape is (num_envs, 2).
    """
    command_term: DroneTrajectoryCommand = env.command_manager.get_term(command_name)
    _, quat_w, _ = command_term.tracked_frame_state()
    _, _, yaw = math_utils.euler_xyz_from_quat(quat_w)
    error = command_term.command[:, 9] - yaw
    return torch.stack([torch.sin(error), torch.cos(error)], dim=-1)


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def trajectory_position_offset_r(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Position of the tracked frame relative to the reference, in the reference's yaw frame.

    :func:`trajectory_position_error_v` gives the same displacement in the vehicle frame, which is the
    frame of the action. This one is in the frame of the target pose itself, so where the tracked frame
    sits relative to the target (ahead of it, behind it, to one side) can be read directly, without
    rotating by the yaw error.

    Args:
        env: The manager-based RL environment instance.
        command_name: Name of the trajectory command term to read the reference from.

    Returns:
        The tracked frame's position minus the reference position [m], expressed in the frame rotated by
        the reference yaw. Shape is (num_envs, 3).
    """
    command_term: DroneTrajectoryCommand = env.command_manager.get_term(command_name)
    pos_w, _, _ = command_term.tracked_frame_state()
    command = command_term.command
    zeros = torch.zeros_like(command[:, 9])
    ref_quat = math_utils.quat_from_euler_xyz(zeros, zeros, command[:, 9])
    return math_utils.quat_apply_inverse(ref_quat, pos_w - command[:, :3])


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def trajectory_ref_yaw_rate(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Reference yaw rate feed-forward.

    Args:
        env: The manager-based RL environment instance.
        command_name: Name of the trajectory command term to read the reference from.

    Returns:
        The reference yaw rate [rad/s]. Shape is (num_envs, 1).
    """
    return env.command_manager.get_command(command_name)[:, 10:11]


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def trajectory_lookahead_v(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Future reference positions relative to the current one, in the vehicle frame.

    Each look-ahead point is reported as an offset from the *current* position of the tracked frame
    rather than as an absolute one, so the observation stays bounded and centred on zero regardless
    of where in the environment the drone is.

    Args:
        env: The manager-based RL environment instance.
        command_name: Name of the trajectory command term to read the reference from.
        asset_cfg: SceneEntityCfg identifying the asset.

    Returns:
        The look-ahead offsets [m] in the yaw-only vehicle frame, flattened.
        Shape is (num_envs, 3 * num_lookahead).
    """
    asset: Multirotor = env.scene[asset_cfg.name]
    command_term: DroneTrajectoryCommand = env.command_manager.get_term(command_name)
    position_w, _, _ = command_term.tracked_frame_state()
    lookahead_w = command_term.command[:, 11:].reshape(env.num_envs, -1, 3) - position_w.unsqueeze(1)
    vehicle_quat = _vehicle_frame_quat(asset).unsqueeze(1).expand(-1, lookahead_w.shape[1], -1)
    return math_utils.quat_apply_inverse(vehicle_quat, lookahead_w).flatten(start_dim=1)


@generic_io_descriptor(dtype=torch.float32, observation_type="RootState", on_inspect=[record_shape])
def body_lin_vel_v(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Linear velocity of a body, in the vehicle frame.

    When the command tracks a payload frame, the velocity errors are the payload's. This gives the policy
    the velocity of the airframe itself, which is what its velocity command acts on.

    Args:
        env: The environment instance.
        asset_cfg: SceneEntityCfg identifying the asset, with ``body_names`` selecting the body.

    Returns:
        The body's linear velocity [m/s] in the yaw-only vehicle frame. Shape is (num_envs, 3).
    """
    asset: Multirotor = env.scene[asset_cfg.name]
    lin_vel_w = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids[0]]  # type: ignore[index]
    return math_utils.quat_apply_inverse(_vehicle_frame_quat(asset), lin_vel_w)


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def body_rest_target_error_v(
    env: ManagerBasedRLEnv, command_name: str, hang_offset: tuple[float, float, float], asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Error of a body to where it rests when the hanging tool it carries is on the reference, in the vehicle frame.

    When the command tracks a hanging tool, the body's rest target is the reference position minus
    ``hang_offset`` rotated by the reference yaw: where the body would sit, level, with the payload
    hanging still and the tool on the reference. The difference from the tool's own error is how far the
    swing and the hang carry the tool from the body.

    Args:
        env: The manager-based RL environment instance.
        command_name: Name of the trajectory command term to read the reference from.
        hang_offset: Tool point relative to the body when the payload hangs at rest below a level
            vehicle, in the yaw-aligned vehicle frame [m].
        asset_cfg: SceneEntityCfg identifying the asset, with ``body_names`` selecting the body.

    Returns:
        The rest target minus the body position [m], in the yaw-only vehicle frame. Shape is (num_envs, 3).
    """
    asset: Multirotor = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    zeros = torch.zeros_like(command[:, 9])
    ref_yaw_quat = math_utils.quat_from_euler_xyz(zeros, zeros, command[:, 9])
    hang_offset_v = torch.tensor(hang_offset, device=env.device).expand(env.num_envs, 3)
    rest_target_w = command[:, :3] - math_utils.quat_apply(ref_yaw_quat, hang_offset_v)
    body_pos_w = asset.data.body_pos_w.torch[:, asset_cfg.body_ids[0]] - env.scene.env_origins  # type: ignore[index]
    return math_utils.quat_apply_inverse(_vehicle_frame_quat(asset), rest_target_w - body_pos_w)


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def trajectory_time_to_go(env: ManagerBasedRLEnv, command_name: str, horizon_s: float) -> torch.Tensor:
    """Time left until the reference reaches its end point, clipped to a horizon and normalized.

    It reads 1 while the end point is more than ``horizon_s`` away, ramps down to 0 on arrival and stays
    at 0 for the whole hold (see
    :meth:`~isaaclab_tasks.contrib.drone_arl.mdp.commands.DroneTrajectoryCommand.time_to_end_point`), so
    the policy can prepare the arrival without learning when the episode times out. A planner on the
    vehicle knows its own timing, so it can supply the same signal.

    Args:
        env: The manager-based RL environment instance.
        command_name: Name of the trajectory command term to read the reference from.
        horizon_s: Time to go beyond which the observation saturates at 1 [s].

    Returns:
        The normalized time to go, in [0, 1]. Shape is (num_envs, 1).
    """
    command_term: DroneTrajectoryCommand = env.command_manager.get_term(command_name)
    return torch.clamp(command_term.time_to_end_point() / horizon_s, max=1.0).unsqueeze(-1)


class CalibratedRestTargetError(ManagerTermBase):
    """Error of a body to where it rests when the hanging tool it carries is on the reference, using the
    payload's calibrated rest pose rather than its nominal one, in the vehicle frame.

    :func:`body_rest_target_error_v` places the tool with a fixed ``hang_offset``. A real payload rests at a
    hinge offset of its own (see :class:`~isaaclab_tasks.contrib.drone_arl.mdp.events.RandomizePayloadHangOffset`),
    so zeroing that error leaves the tool off the reference by the difference, up to ~0.15 m on the Mk3
    chainsaw. This term uses the rest offset of the payload instead: the nominal offset swung about the pivot
    by the payload's rest hinge angles, outer hinge first. Zeroing it puts the tool on the reference once any
    swing has died out, and since the rest pose does not move with the swing, the error stays as smooth as
    the nominal one.

    On the vehicle the rest angles come from a calibration: the hinge encoders averaged while hovering still
    before the approach. In simulation they are the randomized rest offset plus a calibration error drawn
    per episode, so the policy still needs the tool's own error for the last centimetres.

    The hinges are taken as rotations about the vehicle's y axis (outer, ``+`` swings the payload aft) and x
    axis (inner, ``+`` swings it left), as in ``RandomizePayloadHangOffset``, both through the pivot.

    The ``params`` of the term configuration are:

    * ``command_name``: Name of the trajectory command term to read the reference from.
    * ``hang_offset``: Nominal tool point relative to the body with the payload hanging at its default hinge
      angles below a level vehicle, in the yaw-aligned vehicle frame [m].
    * ``asset_cfg``: SceneEntityCfg identifying the asset, with ``body_names`` selecting the body.
    * ``pivot_cfg``: SceneEntityCfg with the body whose link frame origin is the pivot.
    * ``rest_term_name``: Name of the ``RandomizePayloadHangOffset`` event term holding the rest angles.
    * ``calibration_std``: Standard deviation of the calibration error on each rest angle [rad]. Defaults
      to 0.0175 (1 deg).

    Returns a tensor of shape (num_envs, 3): the rest target minus the body position [m], in the yaw-only
    vehicle frame.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        pivot_cfg: SceneEntityCfg = cfg.params["pivot_cfg"]
        asset_cfg.resolve(env.scene)
        pivot_cfg.resolve(env.scene)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._body_id: int = asset_cfg.body_ids[0]  # type: ignore[index]
        self._pivot_id: int = pivot_cfg.body_ids[0]  # type: ignore[index]
        self._hang_v = torch.tensor(cfg.params["hang_offset"], device=env.device).expand(env.num_envs, 3)
        self._rest_term_name: str = cfg.params["rest_term_name"]
        self._calibration_std: float = cfg.params.get("calibration_std", 0.0175)
        self._calibration_error = torch.zeros(env.num_envs, 2, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        error = self._calibration_error[env_ids]
        self._calibration_error[env_ids] = torch.randn_like(error) * self._calibration_std

    def rest_hang_offset(self) -> torch.Tensor:
        """Tool point relative to the body with the payload at its calibrated rest pose below a level vehicle.

        Returns:
            The offset in the yaw-aligned vehicle frame [m]. Shape is (num_envs, 3).
        """
        rest = self._env.event_manager.get_term_cfg(self._rest_term_name).func.rest_offset + self._calibration_error
        data = self._asset.data
        body_quat = data.body_link_quat_w.torch[:, self._body_id]
        # The pivot is fixed to the body: its offset in the body frame, which is the vehicle frame when level.
        pivot_b = math_utils.quat_apply_inverse(
            body_quat, data.body_link_pos_w.torch[:, self._pivot_id] - data.body_link_pos_w.torch[:, self._body_id]
        )
        zeros = torch.zeros_like(rest[:, 0])
        hinge_quat = math_utils.quat_mul(
            math_utils.quat_from_euler_xyz(zeros, rest[:, 0], zeros),
            math_utils.quat_from_euler_xyz(rest[:, 1], zeros, zeros),
        )
        return pivot_b + math_utils.quat_apply(hinge_quat, self._hang_v - pivot_b)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        hang_offset: tuple[float, float, float],
        asset_cfg: SceneEntityCfg,
        pivot_cfg: SceneEntityCfg,
        rest_term_name: str,
        calibration_std: float = 0.0175,
    ) -> torch.Tensor:
        """Compute the observation.

        Returns:
            The rest target minus the body position [m], in the yaw-only vehicle frame. Shape is (num_envs, 3).
        """
        command = env.command_manager.get_command(command_name)
        zeros = torch.zeros_like(command[:, 9])
        ref_yaw_quat = math_utils.quat_from_euler_xyz(zeros, zeros, command[:, 9])
        rest_target_w = command[:, :3] - math_utils.quat_apply(ref_yaw_quat, self.rest_hang_offset())
        body_pos_w = self._asset.data.body_pos_w.torch[:, self._body_id] - env.scene.env_origins
        return math_utils.quat_apply_inverse(_vehicle_frame_quat(self._asset), rest_target_w - body_pos_w)


class ToolEndTargetError(ManagerTermBase):
    """Error of a hanging tool to its target at the trajectory end point, in the vehicle frame.

    The target is the one rewarded by :class:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.ToolEndTargetExp`:
    where the tool would be if the vehicle sat level at the end point with the payload hanging at rest.
    It is observed over the whole episode, not only when that reward is active, so the policy can see
    where the tool has to go before it gets there.

    The ``params`` of the term configuration are:

    * ``asset_cfg``: SceneEntityCfg identifying the tool body. Must have ``body_names`` configured.
    * ``tool_offset``: Tool point in the tool body frame [m].
    * ``hang_offset``: Tool point relative to the tracked body when the payload hangs at rest below a
      level vehicle, in the yaw-aligned vehicle frame [m].
    * ``tool_rot``: Orientation of the tool frame in the tool body frame, as a quaternion (x, y, z, w),
      chosen so that the tool frame lines up with the vehicle yaw frame at rest. It sets the tool yaw.
    * ``command_name``: Name of the trajectory command term to read the end pose from.

    Returns a tensor of shape (num_envs, 4): the target minus the tool position [m] in the yaw-only
    vehicle frame, and the end yaw minus the tool yaw [rad], wrapped to (-pi, pi].
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset_cfg.resolve(env.scene)
        self._asset: Articulation = env.scene[asset_cfg.name]
        self._body_id: int = asset_cfg.body_ids[0]  # type: ignore[index]
        self._command_term: DroneTrajectoryCommand = env.command_manager.get_term(cfg.params["command_name"])
        self._tool_b = torch.tensor(cfg.params["tool_offset"], device=env.device).expand(env.num_envs, 3)
        self._hang_v = torch.tensor(cfg.params["hang_offset"], device=env.device).expand(env.num_envs, 3)
        self._tool_rot_b = torch.tensor(cfg.params["tool_rot"], device=env.device).expand(env.num_envs, 4)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        tool_offset: tuple[float, float, float],
        hang_offset: tuple[float, float, float],
        tool_rot: tuple[float, float, float, float],
        command_name: str,
    ) -> torch.Tensor:
        """Compute the observation.

        Returns:
            The position and yaw error of the tool to its end target. Shape is (num_envs, 4).
        """
        tool_pos, tool_quat, target_pos, _ = hanging_tool_and_end_target(
            self._asset,
            self._body_id,
            self._tool_b,
            self._tool_rot_b,
            self._hang_v,
            self._command_term,
            env.scene.env_origins,
        )
        pos_error_v = math_utils.quat_apply_inverse(_vehicle_frame_quat(self._asset), target_pos - tool_pos)
        _, _, tool_yaw = math_utils.euler_xyz_from_quat(tool_quat)
        yaw_error = math_utils.wrap_to_pi(self._command_term.yaw_end - tool_yaw)
        return torch.cat([pos_error_v, yaw_error.unsqueeze(-1)], dim=-1)
