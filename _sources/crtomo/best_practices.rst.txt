Configuration checklist
^^^^^^^^^^^^^^^^^^^^^^^

**Before** using any ``crtomo.cfg`` file check the following settings:

* Robust/non-robust
* Anisotropic filtering?
* Homogeneous starting model (use F if you invert surface configurations, this
  will use an average of the data values as the starting model)
* Magnitude errors (abs. error of 1e-4 stabilzes)
* Phase errors
* FPI?
* Start FPI with homogenous starting model (only if phases get over fitted in
  complex inversion)?
* Error ellipsis enabled/disabled?

.. todo::

    An explanation what the different settings are about and how to change
    them would be helpfull.
.. warning::

    The following "standard" inversion scheme are really old and should be
    reviewed and modified to adapt modern inversion practices!

Standard inversion schemes
^^^^^^^^^^^^^^^^^^^^^^^^^^

The following inversion schemes can be used for initial screenings of the data
if no error analysis was made. They can also be used as a starting point for
further error parameter adaption.

.. todo::

    How to make an error analysis? Is there a section, then link it here.
    If there is no such section, then create one.

Each scheme denotes important parameters to be changed in the ``crtomo.cfg``
file.

Normal
""""""

* 5 % Magnitude error + :math:`1e^{-4}\Omega`
* 2 mrad phase error
* robust inversion
* no homogeneous starting model for FPI

Optimistic
""""""""""


* 5 % Magnitude error + :math:`1e^{-4}\Omega`
* 1 mrad phase error
* non-robust inversion
* no homogeneous starting model for FPI


Pessimistic
"""""""""""

* 10 % Magnitude error + :math:`1e^{-4}\Omega`
* 4 mrad phase error
* robust inversion
* no homogeneous starting model for FPI
