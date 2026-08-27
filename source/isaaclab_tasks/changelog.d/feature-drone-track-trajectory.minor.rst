Added
^^^^^

* Added the ``Isaac-TrackTrajectory-Matrice-v0`` and ``Isaac-TrackTrajectory-Matrice-Play-v0``
  environments, in which a DJI M350 follows a randomly sampled smooth 3D reference trajectory by
  commanding velocity setpoints into a Lee velocity controller.
* Added the ``Isaac-TrackTrajectory-Matrice-Detuned-v0`` and
  ``Isaac-TrackTrajectory-Matrice-Detuned-Play-v0`` variants, which run the same task against an
  inner loop detuned onto the measured response of the PX4 controller the policy is deployed
  against.
* Added :class:`~isaaclab_tasks.manager_based.drone_arl.mdp.commands.DroneTrajectoryCommand`, a
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
