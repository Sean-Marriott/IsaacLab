* **Breaking:** :func:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.velocity_tracking_exp` and
  :func:`~isaaclab_tasks.contrib.drone_arl.mdp.rewards.yaw_tracking_exp` no longer take
  ``asset_cfg``; they read the frame tracked by the trajectory command. Remove ``asset_cfg`` from
  their ``params``.
* Changed the trajectory observation terms to take their errors against the frame tracked by the
  trajectory command. On the clean M350 this moves the position error from the articulation root
  (``jetson``) to ``base_link``, which the rewards already used, a shift of about 1.5 cm.
