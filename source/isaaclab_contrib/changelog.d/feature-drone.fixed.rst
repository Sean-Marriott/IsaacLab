* Fixed :class:`~isaaclab_contrib.assets.Multirotor` ignoring ``init_state.joint_pos`` and
  ``init_state.joint_vel``, which left the default joint state of passive payload joints at zero.
* Fixed :class:`~isaaclab_contrib.assets.Multirotor` leaving the soft joint velocity limits unset,
  which made joint reset events such as :func:`~isaaclab.envs.mdp.reset_joints_by_offset` fail on
  multirotors with passive payload joints.
* Fixed :class:`~isaaclab_contrib.controllers.LeeVelController` ignoring
  ``max_inclination_angle_rad``. The commanded lateral acceleration is now clamped to the tilt that
  the configured angle allows. Previously a large velocity error produced an arbitrarily large
  commanded tilt, bounded only by motor saturation, which made the action-to-acceleration map
  discontinuous once the vehicle started to flip.
* Fixed :class:`~isaaclab_contrib.assets.Multirotor` leaving the soft joint position limits at zero,
  which made joint reset events that clamp to them, such as
  :func:`~isaaclab.envs.mdp.reset_joints_by_offset`, write every passive payload joint at zero.
