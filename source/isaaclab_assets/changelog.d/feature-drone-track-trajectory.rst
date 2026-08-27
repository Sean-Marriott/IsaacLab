Added
^^^^^

* Added ``MATRICE_CLEAN_CFG``, a DJI M350 configuration without the chainsaw payload (6.500 kg
  instead of 8.665 kg, thrust-to-weight 3.45). It shares the airframe, body names and allocation
  matrix of ``MATRICE_CFG`` but has a much lower attitude inertia, so controller gains tuned for
  ``MATRICE_CFG`` must be re-derived before use.
