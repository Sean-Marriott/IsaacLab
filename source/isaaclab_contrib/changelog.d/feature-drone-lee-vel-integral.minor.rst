Added
^^^^^

* Added optional integral action to :class:`~isaaclab_contrib.controllers.LeeVelController`, via
  ``LeeVelControllerCfg.K_vel_int_range`` and ``LeeVelControllerCfg.max_integral_acc``. The gain is
  zero on every axis by default, so the controller stays purely proportional unless it is enabled.

  Enable it when the vehicle a policy will be deployed on holds its velocity setpoint without a
  standing error. A purely proportional loop leaves a standing velocity error proportional to any
  unmodelled steady force, and a learned outer loop absorbs that error into a constant offset on its
  velocity command -- an offset that is then wrong on a vehicle whose own loop would have rejected
  the disturbance. ``max_integral_acc`` bounds the integral state for anti-windup, and the state is
  cleared per environment on reset.
