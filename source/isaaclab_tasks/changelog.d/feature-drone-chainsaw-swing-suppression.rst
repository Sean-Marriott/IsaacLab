Added
^^^^^

* Added the ``Isaac-TrackTrajectory-Matrice-ChainsawSwing-v0`` and ``Isaac-TrackTrajectory-Matrice-ChainsawSwing-Play-v0``
  tasks, built on the chainsaw trajectory task, in which the M350 spawns away from a target pose and has to fly to
  it and align with it while damping the swing of the Mk3 chainsaw, as in the direct-velocity position-tracking
  task. The chainsaw starts with a large swing and is kicked during the episode, and a reward is given for low
  payload swing energy. Trajectory following is disabled by holding the reference still, and can be re-enabled
  in the task configuration.
* Added the :class:`~isaaclab_tasks.manager_based.drone_arl.mdp.rewards.PayloadSwingEnergy` and
  :class:`~isaaclab_tasks.manager_based.drone_arl.mdp.rewards.PayloadSwingEnergyExp` rewards for the swing energy
  of a hanging payload, measured against gravity in the world frame.
* Added the :class:`~isaaclab_tasks.manager_based.drone_arl.mdp.rewards.DroneEndPoseExp` reward, with pose
  markers, for the vehicle matching a level target pose at the trajectory end point.
* Added the :func:`~isaaclab_tasks.manager_based.drone_arl.mdp.events.push_joints_by_velocity` event to kick
  selected joints mid-episode without resetting their positions.
