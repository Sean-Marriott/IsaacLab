Fixed
^^^^^

* Fixed :class:`~isaaclab_physx.assets.Articulation` and :class:`~isaaclab_physx.assets.RigidObject`
  property writes on GPU sometimes sending stale values to PhysX. Joint stiffness, damping, limits,
  armature and friction, and body masses, centers of mass and inertias are staged through pinned CPU
  buffers, and the device-to-host copy into them was not waited on, so PhysX could read the previous
  contents (often zeros). This also affected the randomization events built on these writes, such as
  :func:`~isaaclab.envs.mdp.randomize_rigid_body_mass`.
