Fixed
^^^^^

* Fixed :meth:`~isaaclab.managers.ObservationManager.get_IO_descriptors` producing output that
  could not be exported to YAML when any observation term used a modifier. Modifier configurations
  are now reduced to their class name and scalar fields. Previously
  ``scripts/environments/export_IODescriptors.py`` raised a representer error and wrote an empty
  file for any environment using, for example, ``DelayedObservationCfg``.
