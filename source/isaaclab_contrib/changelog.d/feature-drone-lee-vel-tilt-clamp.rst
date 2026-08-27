Fixed
^^^^^

* Fixed :class:`~isaaclab_contrib.controllers.LeeVelController` ignoring
  ``max_inclination_angle_rad``. The commanded lateral acceleration is now clamped to the tilt that
  the configured angle allows. Previously a large velocity error produced an arbitrarily large
  commanded tilt, bounded only by motor saturation, which made the action-to-acceleration map
  discontinuous once the vehicle started to flip.
