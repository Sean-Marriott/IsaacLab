Fixed
^^^^^

* Fixed the thruster fall time constant in ``MATRICE_THRUSTER``, which was 10-16x shorter than the
  rise constant (5 ms against 50-80 ms) and below the physics step. The asymmetry made the motor
  track thrust decreases far faster than increases, rectifying a fluctuating thrust command
  downwards so that only 78.8 % of the commanded thrust was delivered, against 94.8 % once the
  constants match. Policies compensated by learning a large constant positive climb command, which
  is a simulation artifact that produces an uncommanded climb on a real vehicle. ``tau_dec_range``
  now matches ``tau_inc_range``. **Policies trained against the previous values carry this trim and
  must be retrained.**
