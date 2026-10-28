* Added :class:`~isaaclab_tasks.contrib.drone_arl.mdp.events.RandomizePassiveJointDamping`,
  :func:`~isaaclab_tasks.contrib.drone_arl.mdp.events.set_passive_joint_stiffness` and
  :func:`~isaaclab_tasks.contrib.drone_arl.mdp.events.set_passive_joint_friction` to set the drive
  damping, stiffness and dry friction of unactuated joints, which a multirotor cannot give actuators.
* Added the saw-head hinge springs and dry friction to the ``IsaacContrib-TrackTrajectory-Matrice-Chainsaw``
  task, set by ``ROOT_JOINT_SPRING_STIFFNESS`` and ``ROOT_JOINT_FRICTION`` in its config (0.0 removes
  them), with values identified from motion capture of the real payload, and randomized the saw mass,
  the hinge damping, the spring rate and the hinge friction at startup.
* Added the ``IsaacContrib-TrackTrajectory-Matrice-ChainsawSwing`` tasks, built on the chainsaw trajectory task, in which the M350 spawns away from a target pose and has to fly to
  it and align with it while damping the swing of the Mk3 chainsaw, as in the direct-velocity position-tracking
  task. The chainsaw starts with a large swing and is kicked during the episode, and a reward is given for low
  payload swing energy. Trajectory following is disabled by holding the reference still, and can be re-enabled
  in the task configuration.
* Added the :class:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.PayloadSwingEnergy` and
  :class:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.PayloadSwingEnergyExp` rewards for the swing energy
  of a hanging payload, measured against gravity in the world frame.
* Added the :class:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.DroneEndPoseExp` reward, with pose
  markers, for the vehicle matching a level target pose at the trajectory end point.
* Added the :func:`~isaaclab_tasks.contrib.drone_arl.mdp.events.push_joints_by_velocity` event to kick
  selected joints mid-episode without resetting their positions.
* Added ``IsaacContrib-TrackTrajectory-Matrice-ChainsawTool``, in which the trajectory reference is
  tracked by the chainsaw cutting mouth rather than by the drone.
* Added :attr:`~isaaclab_tasks.contrib.drone_arl.mdp.commands.DroneTrajectoryCommandCfg.body_offset`
  and :meth:`~isaaclab_tasks.contrib.drone_arl.mdp.commands.DroneTrajectoryCommand.tracked_frame_state`
  to track a frame fixed to a body, such as a tool tip, against the trajectory reference.
* Added :func:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.position_tracking_exp` and
  :func:`~isaaclab_tasks.contrib.drone_arl.mdp.terminations.tracked_position_error_above_maximum`,
  and the ``rest_offset`` parameter of
  :func:`~isaaclab_tasks.contrib.drone_arl.mdp.events.reset_root_state_on_trajectory`.
* Added the ``tool_rot`` and ``std_yaw`` parameters of
  :class:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.ToolEndTargetExp`, which draw the
  tool and its end target as pose markers and reward the tool yaw at the end target.
* Added the ``window_s`` parameter of
  :class:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.PayloadSwingEnergyExp`, which limits
  the reward to the end of the trajectory.
* Added a PPO runner configuration for ``IsaacContrib-TrackTrajectory-Matrice-ChainsawTool`` with a
  larger entropy bonus, to keep the action noise from collapsing.
* Locked the chainsaw saw-head pitch joint at its rest pose in
  ``IsaacContrib-TrackTrajectory-Matrice-ChainsawTool``, including in play mode, so the payload swings as a
  single rigid pendulum.
* Added the ``IsaacContrib-TrackTrajectory-Matrice-Chainsaw`` tasks, which fly the M350 with the Mk3 chainsaw, add its payload joint angles and rates to the trajectory-tracking policy observation,
  randomize the hinge state on reset, and reward the chainsaw cutting mouth for reaching its hanging target
  at the end of the trajectory.
* Added :attr:`~isaaclab_tasks.contrib.drone_arl.mdp.commands.DroneTrajectoryCommandCfg.end_slowdown_s` and
  :attr:`~isaaclab_tasks.contrib.drone_arl.mdp.commands.DroneTrajectoryCommandCfg.end_hold_s` to bring the
  trajectory reference smoothly to rest and hold it at the end of an episode.
* Added the :class:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.ToolEndTargetExp` reward, with debug markers, for a hanging tool reaching its target at the trajectory end point.
* Added the ``IsaacContrib-TrackTrajectory-Matrice-TestShapes`` evaluation environment and
  :class:`~isaaclab_tasks.contrib.drone_arl.mdp.commands.DroneTestShapeCommand`, which fly the
  square, circle, line and zigzag figures of the vehicle-side ``generate_test_trajectory`` tool at
  the same size and cruise speed, so a policy can be scored on identical shapes in simulation and
  on hardware. Each figure is a truncated Fourier series through the constant-speed polyline, so
  velocity and acceleration stay closed-form and the corners are rounded rather than infinite.
* Added the ``IsaacContrib-TrackTrajectory-Matrice`` environments, in which a DJI M350 follows a randomly sampled smooth 3D reference trajectory by
  commanding velocity setpoints into a Lee velocity controller.
* Added the ``IsaacContrib-TrackTrajectory-Matrice-Detuned`` variant, which run the same task against an
  inner loop detuned onto the measured response of the PX4 controller the policy is deployed
  against.
* Added :class:`~isaaclab_tasks.contrib.drone_arl.mdp.commands.DroneTrajectoryCommand`, a
  time-parameterized sum-of-sinusoids trajectory command with closed-form position, velocity and
  acceleration, a configurable look-ahead, and hard sup-bounds on reference speed and acceleration
  that scale with a ``difficulty`` factor suitable for a curriculum.
* Added trajectory-tracking observation terms that express the reference in the yaw-only vehicle
  frame, matching how the velocity action is interpreted.
* Added the ``velocity_tracking_exp`` and ``yaw_tracking_exp`` reward terms, the
  ``position_error_above_maximum`` termination term, and the ``reset_root_state_on_trajectory``
  event, which spawns the drone on the start of the reference with matched velocity.
* Enabled integral action on all three axes of the trajectory task's velocity controller, so that
  commanding zero velocity holds station as it does on the vehicle this policy targets. Without it
  the controller's standing velocity error was absorbed into a constant offset on the policy's own
  velocity command, which became an uncommanded drift once deployed. The lateral gains are set
  below the vertical one because they act through the attitude loop rather than on thrust directly.
* Added :class:`~isaaclab_tasks.contrib.drone_arl.mdp.tensor_processors.DelayedObservation`, a stochastic,
  causal observation delay with optional multi-rate refresh, used by the drone tasks.
* Added :func:`~isaaclab_tasks.contrib.drone_arl.mdp.terminations.root_height_above_maximum`.
* Added the ``IsaacContrib-TrackPositionDirectVel-ARL-Robot-1``, ``IsaacContrib-TrackPositionDirectVel-Matrice``,
  ``IsaacContrib-SimpleLidarNavigation-ARL-Robot-1`` and ``IsaacContrib-SimpleLatentDepthNavigation-ARL-Robot-1``
  tasks.
* Added the :func:`~isaaclab_tasks.contrib.drone_arl.mdp.events.reset_swing_by_amplitude` and
  :func:`~isaaclab_tasks.contrib.drone_arl.mdp.events.push_swing_by_velocity` events, which start and
  kick a payload swing with a log-uniform size and a share of no swing at all, and the
  :class:`~isaaclab_tasks.contrib.drone_arl.mdp.events.RandomizePayloadHangOffset` event, which makes
  a hanging payload rest at a random hinge offset.
* Added the ``compensate_reference_accel`` parameter of
  :class:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.PayloadSwingEnergyExp`, which measures the swing
  about the lean that the trajectory reference's acceleration calls for.
* Added the ``window_s`` parameter, and tracked-frame support through ``asset_cfg=None``, to
  :class:`~isaaclab_tasks.contrib.drone_arl.mdp.terminations.PositionErrorAboveMaximumAfterCapture`.
