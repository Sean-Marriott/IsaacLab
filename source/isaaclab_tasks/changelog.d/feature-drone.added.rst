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
* Added a recurrent PPO runner configuration for ``IsaacContrib-TrackTrajectory-Matrice-ChainsawTool``.
* Added the ``fresh_env_fraction`` parameter of
  :class:`~isaaclab_tasks.contrib.drone_arl.mdp.tensor_processors.DelayedObservationCfg`, which gives a
  fraction of the environments the minimum lag with no held frames, the same ones for every term.
* Added the :class:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.ActionSecondDifferenceL2` reward, which
  penalizes an action that flips sign every policy step without resisting held or ramping commands.
* Added ``IsaacContrib-TrackTrajectory-Matrice-ChainsawDroneRef``, the chainsaw tool task with the
  trajectory reference tracked by the drone, as it is planned on the vehicle, and the cutting mouth
  rewarded for reaching its hanging target below the end point, with PPO and recurrent PPO runners.
  Its policy also observes the mouth's error to that target and the time to go until the end point,
  and the fine airframe position tracking is off during the end hold.
* Added the :class:`~isaaclab_tasks.contrib.drone_arl.mdp.observations.ToolEndTargetError` and
  :func:`~isaaclab_tasks.contrib.drone_arl.mdp.observations.trajectory_time_to_go` observations,
  :meth:`~isaaclab_tasks.contrib.drone_arl.mdp.commands.DroneTrajectoryCommand.time_to_end_point`, and
  the ``exclude_end_s`` parameter of :func:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.position_tracking_exp`,
  which turns the reward off for the end of the reference.
* Added ``IsaacContrib-TrackTrajectory-Matrice-ChainsawAlign``, in which the M350 spawns away from a
  target pose for the Mk3 chainsaw's cutting mouth and flies the mouth onto it and holds it there while
  damping the swing, with no trajectory to follow, with PPO and recurrent PPO runners.
  Its policy observes the airframe's velocity and its error to where it rests with the mouth on
  target, from the new :func:`~isaaclab_tasks.contrib.drone_arl.mdp.observations.body_lin_vel_v` and
  :func:`~isaaclab_tasks.contrib.drone_arl.mdp.observations.body_rest_target_error_v` terms.
* Added the :class:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.ToolCapture` reward, which scores a hanging
  tool for being captured at its end target, inside an approach cone, slow and square to it, and logs a
  dwell-based success rate, and used it in ``IsaacContrib-TrackTrajectory-Matrice-ChainsawAlign``.
* Added capture markers to :class:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.ToolCapture`: the capture
  envelope as a translucent cone, the tolerance ball, and the tool point with a heading pointer, all
  coloured by the capture state. Added
  :attr:`~isaaclab_tasks.contrib.drone_arl.mdp.commands.DroneTrajectoryCommandCfg.reference_debug_vis`
  to hide the reference markers of a trajectory command while keeping its termination flash.
* Added the :func:`~isaaclab_tasks.contrib.drone_arl.mdp.observations.trajectory_yaw_error_sin_cos` and
  :func:`~isaaclab_tasks.contrib.drone_arl.mdp.observations.trajectory_position_offset_r` observations, and
  used them in ``IsaacContrib-TrackTrajectory-Matrice-ChainsawAlign`` for the yaw error and the mouth's
  position in the target frame.
* Added the ``require_cone_entry`` parameter of :class:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.ToolCapture`,
  which counts the tolerance ball only after the tool has come in through the approach cone.
* Added the :class:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.ToolCaptureProgress` reward, which pays per-step
  progress along the route to capture through the approach cone, and the
  :func:`~isaaclab_tasks.contrib.drone_arl.mdp.events.reset_root_state_on_approach` event, which spawns the tracked
  frame on the approach line to a still target with configurable spreads. ``ToolCapture`` now returns its state as a
  :class:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.ToolCaptureState`. In
  ``IsaacContrib-TrackTrajectory-Matrice-ChainsawAlign`` the mouth now spawns on the approach line, the cone
  entry is required and the progress reward is used.
* Added route markers to :class:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.ToolCaptureProgress`: the staging
  point, coloured by the cone-entry flag, and the route the progress is measured along, coloured by the last step's
  progress.
* Added the ``tip_radius`` parameter of :class:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.ToolCapture`, a blunt
  tip that keeps the approach cone at least that wide near the target, drawn as a cylinder along the cone. The cone
  is now bounded along its axis at ``capture_radius`` rather than by distance, and ``IsaacContrib-TrackTrajectory-Matrice-ChainsawAlign``
  uses a tip as wide as its tolerance ball.
* Added ``IsaacContrib-ChainsawAlign-Matrice``, a standalone task in which the M350 flies the Mk3 chainsaw's
  cutting mouth onto a target position and yaw through the Lee velocity controller, arriving from behind
  along the target's approach axis, with no trajectory reference. It has PPO and recurrent PPO runners.
  Its MDP terms live in ``isaaclab_tasks.contrib.drone_arl.chainsaw_align.mdp``:
  :class:`~isaaclab_tasks.contrib.drone_arl.chainsaw_align.mdp.ToolPoseCommand`, a staged approach of a tool
  frame through a staging point to its target that logs the arrival angle and a direction-aware success rate;
  route-length and cross-track rewards; the
  :func:`~isaaclab_tasks.contrib.drone_arl.chainsaw_align.mdp.reset_tool_around_target` spawn event; the
  :class:`~isaaclab_tasks.contrib.drone_arl.chainsaw_align.mdp.SuccessDifficultyCurriculum`, which raises a
  difficulty level with the success rate and ramps event parameters with it, widening the spawn around the
  target and then bringing in the :class:`~isaaclab_tasks.contrib.drone_arl.chainsaw_align.mdp.RandomWind`
  gusts; and
  :func:`~isaaclab_tasks.contrib.drone_arl.chainsaw_align.mdp.lock_passive_joints`, which locks the saw-head
  pivot at its rest pose.
* Added sim-to-real terms to ``IsaacContrib-ChainsawAlign-Matrice``: a 0-0.15 s velocity command delay, up to
  one policy step of lag on the estimated observations (always fresh in a quarter of the environments), and
  startup randomization of the tube-hinge damping and the saw mass.
* Added approach aborts to :class:`~isaaclab_tasks.contrib.drone_arl.chainsaw_align.mdp.ToolPoseCommand`: a staged
  tool that leaves the approach corridor (``abort_max_cross_track``, ``abort_max_yaw_error``,
  ``abort_max_overshoot``, ``abort_on_back_out``) is unstaged, so the goal returns to the staging point for
  another try, and the number of retries is logged. The corridor is drawn as four rods along the approach axis,
  grey while unstaged, blue while staged and red just after an abort.
* Added ``IsaacContrib-ChainsawAlignBranch-Matrice``, the chainsaw alignment task with the target made a rigid,
  fixed branch of 10-50 mm diameter, drawn per episode and placed so that, seated in the Mk3's mouth pocket, it puts
  the mouth on the target pose. Hard contact between the saw and the branch is penalized, contact between the
  branch and the rest of the vehicle ends the episode, and the policy observations are unchanged, so it feels the
  branch only through the payload and airframe. Added
  :class:`~isaaclab_tasks.contrib.drone_arl.chainsaw_align.mdp.RandomBranch` to place the branch, and a keep-out
  box to :func:`~isaaclab_tasks.contrib.drone_arl.chainsaw_align.mdp.reset_tool_around_target`. The branch is
  turned to any heading, with the target turned to match, and slopes across the approach by up to 90 deg, ramped
  in by the curriculum, with the saw head rolled to the slope at spawn and held there. A steep branch ends 0.8 m
  above the cut, so it stays clear of the rotors. Added
  :func:`~isaaclab_tasks.contrib.drone_arl.chainsaw_align.mdp.rolled_hang_offset`, and ``roll_cfg`` and
  ``roll_hang_shift`` parameters on ``airframe_rest_error_v`` and ``reset_tool_around_target``, so the
  airframe-rest observation and the spawn follow how the saw roll moves the hanging mouth. Added
  :meth:`~isaaclab_tasks.contrib.drone_arl.chainsaw_align.mdp.ToolPoseCommand.set_target_yaw`, so a reset event
  can turn the target.
* Added occasional strong horizontal wind bursts to
  :class:`~isaaclab_tasks.contrib.drone_arl.chainsaw_align.mdp.RandomWind` (``burst_force_range``,
  ``burst_duration_range_s``, ``burst_interval_range_s``), off by default. The chainsaw alignment tasks turn them on
  in play mode, at 20-30 N for 1-2 s about once per episode, so an evaluation shows the policy thrown out of the
  approach corridor and staging again.
