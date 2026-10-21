Added
^^^^^

* Added ``Isaac-TrackTrajectory-Matrice-ChainsawTool-v0``, in which the trajectory reference is
  tracked by the chainsaw cutting mouth rather than by the drone.
* Added :attr:`~isaaclab_tasks.manager_based.drone_arl.mdp.commands.DroneTrajectoryCommandCfg.body_offset`
  and :meth:`~isaaclab_tasks.manager_based.drone_arl.mdp.commands.DroneTrajectoryCommand.tracked_frame_state`
  to track a frame fixed to a body, such as a tool tip, against the trajectory reference.
* Added :func:`~isaaclab_tasks.manager_based.drone_arl.mdp.rewards.position_tracking_exp` and
  :func:`~isaaclab_tasks.manager_based.drone_arl.mdp.terminations.tracked_position_error_above_maximum`,
  and the ``rest_offset`` parameter of
  :func:`~isaaclab_tasks.manager_based.drone_arl.mdp.events.reset_root_state_on_trajectory`.
* Added the ``tool_rot`` and ``std_yaw`` parameters of
  :class:`~isaaclab_tasks.manager_based.drone_arl.mdp.rewards.ToolEndTargetExp`, which draw the
  tool and its end target as pose markers and reward the tool yaw at the end target.
* Added the ``window_s`` parameter of
  :class:`~isaaclab_tasks.manager_based.drone_arl.mdp.rewards.PayloadSwingEnergyExp`, which limits
  the reward to the end of the trajectory.
* Added a PPO runner configuration for ``Isaac-TrackTrajectory-Matrice-ChainsawTool-v0`` with a
  larger entropy bonus, to keep the action noise from collapsing.
* Locked the chainsaw saw-head pitch joint at its rest pose in
  ``Isaac-TrackTrajectory-Matrice-ChainsawTool-v0`` and its play variant, so the payload swings as a
  single rigid pendulum.

Changed
^^^^^^^

* **Breaking:** :func:`~isaaclab_tasks.manager_based.drone_arl.mdp.rewards.velocity_tracking_exp` and
  :func:`~isaaclab_tasks.manager_based.drone_arl.mdp.rewards.yaw_tracking_exp` no longer take
  ``asset_cfg``; they read the frame tracked by the trajectory command. Remove ``asset_cfg`` from
  their ``params``.
* Changed the trajectory observation terms to take their errors against the frame tracked by the
  trajectory command. On the clean M350 this moves the position error from the articulation root
  (``jetson``) to ``base_link``, which the rewards already used, a shift of about 1.5 cm.
