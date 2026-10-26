* Fixed ``IsaacContrib-TrackPositionNoObstacles-ARL-Robot-1`` failing on the first step since
  :func:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.distance_to_goal_exp` requires a body; it now
  tracks ``base_link``, the body its command targets.
