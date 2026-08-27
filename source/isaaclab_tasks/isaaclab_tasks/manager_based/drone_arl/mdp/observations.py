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
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg
    from isaaclab.sensors.camera.camera import Camera
    from isaaclab.sensors.camera.tiled_camera import TiledCamera
    from isaaclab.sensors.ray_caster.multi_mesh_ray_caster_camera import MultiMeshRayCasterCamera
    from isaaclab.sensors.ray_caster.ray_caster_camera import RayCasterCamera

    from isaaclab_contrib.assets import Multirotor

from isaaclab.envs.utils.io_descriptors import generic_io_descriptor, record_shape

from isaaclab_tasks import ISAACLAB_TASKS_EXT_DIR

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
    ray_vectors_w = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
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
            model_path = os.path.join(ISAACLAB_TASKS_EXT_DIR, "data", "drone_arl", "vae_model.pt")
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
            - Infinity values in depth images are clamped to 10.0 during normalization.
            - Very small depth values (< 0.02) are set to -1.0 to indicate invalid regions.
            - The parameters (sensor_cfg, data_type, etc.) are ignored here as they are
            already stored during initialization. They are included in the signature only
            to satisfy the observation manager's parameter validation.
        """
        images = self.camera_sensor.data.output[self.data_type].clone()

        if (self.data_type == "distance_to_camera") and self.convert_perspective_to_orthogonal:
            images = math_utils.orthogonalize_perspective_depth(images, self.camera_sensor.data.intrinsic_matrices)

        if self.normalize:
            if self.data_type == "distance_to_image_plane":
                images[images == float("inf")] = 10.0
                images[images == -float("inf")] = 10.0
                images[images > 10.0] = 10.0
                images = images / 10.0
                images[images < 0.02] = -1.0
            else:
                raise ValueError(f"Image data type: {self.data_type} not supported")

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
        The reference position minus the current position [m], expressed in the yaw-only vehicle
        frame. Shape is (num_envs, 3).
    """
    asset: Multirotor = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error_w = command[:, :3] - (asset.data.root_pos_w.torch - env.scene.env_origins)
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
        The reference velocity minus the current velocity [m/s], expressed in the yaw-only vehicle
        frame. Shape is (num_envs, 3).
    """
    asset: Multirotor = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    error_w = command[:, 3:6] - asset.data.root_lin_vel_w.torch
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
    """Yaw error to the reference, wrapped to (-pi, pi].

    The Lee velocity controller commands yaw *rate* and never closes a loop on yaw angle, so the
    policy has to close it and therefore needs this term.

    Args:
        env: The manager-based RL environment instance.
        command_name: Name of the trajectory command term to read the reference from.
        asset_cfg: SceneEntityCfg identifying the asset.

    Returns:
        The wrapped yaw error [rad]. Shape is (num_envs, 1).
    """
    asset: Multirotor = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w.torch)
    return math_utils.wrap_to_pi(command[:, 9] - yaw).unsqueeze(-1)


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

    Each look-ahead point is reported as an offset from the *current* position rather than as an
    absolute one, so the observation stays bounded and centred on zero regardless of where in the
    environment the drone is.

    Args:
        env: The manager-based RL environment instance.
        command_name: Name of the trajectory command term to read the reference from.
        asset_cfg: SceneEntityCfg identifying the asset.

    Returns:
        The look-ahead offsets [m] in the yaw-only vehicle frame, flattened.
        Shape is (num_envs, 3 * num_lookahead).
    """
    asset: Multirotor = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    position_w = asset.data.root_pos_w.torch - env.scene.env_origins
    lookahead_w = command[:, 11:].reshape(env.num_envs, -1, 3) - position_w.unsqueeze(1)
    vehicle_quat = _vehicle_frame_quat(asset).unsqueeze(1).expand(-1, lookahead_w.shape[1], -1)
    return math_utils.quat_apply_inverse(vehicle_quat, lookahead_w).flatten(start_dim=1)
