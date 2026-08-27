Added
^^^^^

* Added the ``Isaac-TrackTrajectory-Matrice-TestShapes-Play-v0`` evaluation environment and
  :class:`~isaaclab_tasks.manager_based.drone_arl.mdp.commands.DroneTestShapeCommand`, which fly the
  square, circle, line and zigzag figures of the vehicle-side ``generate_test_trajectory`` tool at
  the same size and cruise speed, so a policy can be scored on identical shapes in simulation and
  on hardware. Each figure is a truncated Fourier series through the constant-speed polyline, so
  velocity and acceleration stay closed-form and the corners are rounded rather than infinite.
