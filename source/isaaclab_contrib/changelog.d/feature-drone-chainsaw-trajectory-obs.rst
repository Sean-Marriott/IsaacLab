Fixed
^^^^^

* Fixed :class:`~isaaclab_contrib.assets.Multirotor` ignoring ``init_state.joint_pos`` and
  ``init_state.joint_vel``, which left the default joint state of passive payload joints at zero.
* Fixed :class:`~isaaclab_contrib.assets.Multirotor` leaving the soft joint position and velocity
  limits at zero, which made joint reset events such as
  :func:`~isaaclab.envs.mdp.reset_joints_by_offset` pin every joint at zero position and velocity.
