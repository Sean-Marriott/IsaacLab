Fixed
^^^^^

* Fixed ``ModuleNotFoundError: No module named 'omni.physics.tensors.api'`` when constructing
  :class:`~isaaclab_physx.assets.RigidObjectCollection`,
  :class:`~isaaclab_physx.assets.DeformableObject`, or
  :class:`~isaaclab_physx.sensors.ContactSensor` on Isaac Sim builds that still ship the PhysX
  tensors API as ``omni.physics.tensors.impl.api``. The module is only referenced in type
  annotations, so it is now imported under ``TYPE_CHECKING`` to match
  :class:`~isaaclab_physx.assets.RigidObject` and :class:`~isaaclab_physx.assets.Articulation`.
