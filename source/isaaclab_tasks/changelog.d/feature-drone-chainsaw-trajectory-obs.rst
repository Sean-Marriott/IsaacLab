Added
^^^^^

* Added the ``Isaac-TrackTrajectory-Matrice-Chainsaw-v0`` and ``Isaac-TrackTrajectory-Matrice-Chainsaw-Play-v0``
  tasks, which fly the M350 with the Mk3 chainsaw, add its payload joint angles and rates to the trajectory-tracking policy observation,
  randomize the hinge state on reset, and reward the chainsaw cutting mouth for reaching its hanging target
  at the end of the trajectory.
* Added :attr:`~isaaclab_tasks.manager_based.drone_arl.mdp.commands.DroneTrajectoryCommandCfg.end_slowdown_s` and
  :attr:`~isaaclab_tasks.manager_based.drone_arl.mdp.commands.DroneTrajectoryCommandCfg.end_hold_s` to bring the
  trajectory reference smoothly to rest and hold it at the end of an episode.
* Added the :class:`~isaaclab_tasks.manager_based.drone_arl.mdp.rewards.ToolEndTargetExp` reward, with debug markers, for a hanging tool reaching its target at the trajectory end point.
