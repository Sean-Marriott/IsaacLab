# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@AGENTS.md

## Repository overview

IsaacLab is a GPU-accelerated robot learning framework built on NVIDIA Isaac Sim / Omniverse. Simulations run in parallel batches on GPU; all environments, sensors, and physics are vectorized.

## Common commands

```bash
# Run a specific test
./isaaclab.sh -p -m pytest source/isaaclab/test/path/to/test_file.py

# Run a specific test method
./isaaclab.sh -p -m pytest source/isaaclab/test/path/to/test_file.py::TestClass::test_method

# Lint/format all files (run BEFORE committing)
./isaaclab.sh -f

# Run inline Python
./isaaclab.sh -p -c "import isaaclab; print(isaaclab.__version__)"

# Build documentation
./isaaclab.sh -d

# Install a specific extension
./isaaclab.sh -i [isaaclab|isaaclab_tasks|isaaclab_rl|all]
```

## Rendering / headless mode

Scripts run **headless by default**. To enable the GUI pass `--viz <renderer>`, e.g.:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-TrackPositionDirectVel-Matrice-v0 \
  --viz kit
```

Do **not** pass `--headless` — it is no longer needed and may not be a valid flag.

## Package structure

`source/` contains independent extensions, each with its own `config/extension.toml`, `setup.py`, and changelog:

| Package | Purpose |
|---|---|
| `isaaclab` | Core framework: managers, assets, sensors, controllers, sim context |
| `isaaclab_assets` | Robot and object USD asset definitions |
| `isaaclab_tasks` | Task suite: manager-based and direct workflow environments |
| `isaaclab_rl` | RL library wrappers (SKRL, RSL-RL, RL Games, SB3) |
| `isaaclab_contrib` | Community robots, controllers |
| `isaaclab_mimic` | Motion imitation (requires cuRobo) |
| `isaaclab_teleop` | Teleoperation |
| `isaaclab_physx` | PhysX backend |
| `isaaclab_newton` | Newton physics backend |

## Architecture: two environment workflows

**Manager-based** (`isaaclab_tasks/manager_based/`) — the standard approach for RL:
- Environment config inherits `ManagerBasedRLEnvCfg`
- Defines an `InteractiveSceneCfg` (robots, sensors, obstacles) and 8 composable managers:
  - `ActionManager`, `ObservationManager`, `RewardManager`, `TerminationManager`
  - `CommandManager`, `EventManager` (domain randomization/resets), `CurriculumManager`, `RecorderManager`
- Each manager term is a Python dataclass (`*_cfg.py`), enabling enable/disable, weight tuning, and composition from a library of pre-built terms without subclassing the env.

**Direct workflow** (`isaaclab_tasks/direct/`) — single-file envs with full manual control, no manager overhead.

## Core subsystems

**Asset system** (`isaaclab/assets/`): `Articulation` (robots/articulated objects), `RigidObject`, `DeformableObject`, `RigidObjectCollection`. All batched across GPU environments.

**Sensor system** (`isaaclab/sensors/`): `Camera` (RTX RGB/depth/segmentation), `RayCaster` (GPU LIDAR), `ContactSensor`, `IMU`, `FrameTransformer`.

**Controller system** (`isaaclab/controllers/`): `DifferentialIKController`, `OperationalSpaceController`, `JointImpedanceController`.

**Scene system** (`isaaclab/scene/`): `InteractiveScene` clones environments in a grid for parallel GPU training.

**Sim context** (`isaaclab/sim/`): `SimulationContext` wraps Isaac Sim; `spawners/` instantiate USD assets from configs.

## Environment registration

`isaaclab_tasks/__init__.py` calls `import_packages()` on first import, which walks subdirectories and calls `gym.register()` for each environment config it finds. New tasks must follow the same directory convention (`__init__.py` with `gym.register` calls) to be auto-discovered.

## RL training

`isaaclab_rl` provides thin wrappers adapting `ManagerBasedRLEnv` / `DirectRLEnv` to each library's interface. Agent configs live under `config/<robot>/agents/` alongside the environment config (e.g., `skrl_ppo_cfg.yaml`, `rsl_rl_ppo_cfg.py`).

## Drone tasks (current branch)

Custom drone environments are under `source/isaaclab_tasks/isaaclab_tasks/manager_based/drone_arl/`. Shared MDP components (observations, rewards, events, curriculums, commands) live in `drone_arl/mdp/`. Task subdirectories, all configured for `arl_robot_1`:

| Subdirectory | Description |
|---|---|
| `track_position_state_based/` | State-based position tracking with PPO (SKRL, RSL-RL, RL Games) |
| `track_position_direct_vel/` | Direct velocity command tracking |
| `navigation/` | Navigation with floating obstacles |
| `simple_navigation/` | Lightweight navigation; variants: lidar (`simple_lidar_navigation_env_cfg.py`) and depth-latent (`simple_depth_latent_navigation_env_cfg.py`) |

### Robot configurations

Both robot configs are defined in `source/isaaclab_assets/isaaclab_assets/robots/arl_robot_1.py`:

| Config | Robot | USD source | Notes |
|---|---|---|---|
| `ARL_ROBOT_1_CFG` | ARL Robot 1 | Isaac Nucleus (`NTNU/ARL-Robot-1/arl_robot_1.usd`) | Hover ~200 RPS, thrust 0.1–10 N/motor |
| `MATRICE_CFG` | DJI Matrice 350 RTK | Local path (`drone_arl/robots/M350.usd`) | 6.5 kg airframe, hover ~69 RPS, thrust 0.5–55 N/motor |

`MATRICE_CFG` is currently active in `track_position_direct_vel`. The local USD path means it won't resolve on machines without that file.
