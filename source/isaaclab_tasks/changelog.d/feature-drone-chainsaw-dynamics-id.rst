Added
^^^^^

* Added :class:`~isaaclab_tasks.manager_based.drone_arl.mdp.events.RandomizePassiveJointDamping`,
  :func:`~isaaclab_tasks.manager_based.drone_arl.mdp.events.set_passive_joint_stiffness` and
  :func:`~isaaclab_tasks.manager_based.drone_arl.mdp.events.set_passive_joint_friction` to set the drive
  damping, stiffness and dry friction of unactuated joints, which a multirotor cannot give actuators.
* Added the saw-head hinge springs and dry friction to the ``Isaac-TrackTrajectory-Matrice-Chainsaw-v0``
  task, set by ``ROOT_JOINT_SPRING_STIFFNESS`` and ``ROOT_JOINT_FRICTION`` in its config (0.0 removes
  them), with values identified from motion capture of the real payload, and randomized the saw mass,
  the hinge damping, the spring rate and the hinge friction at startup.
