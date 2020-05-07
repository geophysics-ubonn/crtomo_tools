#!/usr/bin/env python
r"""
Data error handling in CRTomo
=============================

CRTomo implements a last-square inversion with regularization based on the
Thikonov appraoch. Herey a cost function is minimized, consisting of a data
misfit :math:`\Psi_d` and a model misfit :math:`\Psi_m`. Both terms are
weighted against each other using a regularization parameter $\lambda$.

.. math::

        \Psi &= \Psi_d + \lambda \cdot \Psi_m \\
        \Psi_d &= \sqrt{\sum_i \left(\frac{d_i -
        f_i(m)}{\epsilon_i}\right)^2}\\
        \Psi_m &= \left| \left| W_m m \right| \right|^2_2

hereby :math:`W_m` is the regularization matrix, which by default is a
first-order approximation of the spatial gradient.

The data misfit term weights the difference between data and response of the
complex resistivity model, :math:`f(m)` against the data estimate
:math:`\epsilon`.
A data point is thus "fitted" if the forward response can reproduce it within
the bounds of the error estimate.
Numbers below 1 indicate that the model describes the data better than can be
expected by the data error estimate, possibly indicating an overfitting in
which noise components are treated as data, most often resulting in artifacts
(artificial structures that are introduced by the inversion in order to explain
the noise).

In principle an individual data error can be assigned to each data point.
However, since data errors are not easily determined in geoelectrical
applications, it is common to employ various data error models that exploit
empirically observed relationships between the data errors and the measurements
themselves.
"""
###############################################################################
# Setup:
import crtomo

mesh = crtomo.crt_grid.create_surface_grid(nr_electrodes=10, spacing=1)
tdm = crtomo.tdMan(grid=mesh)

tdm.add_homogeneous_model(100, 0)
tdm.configs.gen_dipole_dipole(skipc=0)
tdm.measurements()
###############################################################################
# Magnitude error model
# ^^^^^^^^^^^^^^^^^^^^^
#
# One common data error model for the resistivity/resistance parameters was
# formulated by LaBrecque et al 1996 and assumes a linear relationship between
# data error and the measured resistance:
#
# :math:`\Delta R = a \cdot R + b`
#
# Hereby a is a relative component (in CRTomo this is a percentage value,
# :math:`a_{crtomo} = a / 100`) and b and absolute value (in :math:`\Omega`).

# set absolute error to 0.01 Ohm
tdm.crtomo_cfg['mag_abs'] = 0.01

# set relative error to 5 percent
tdm.crtomo_cfg['mag_rel'] = 5

###############################################################################
# Phase error model
# ^^^^^^^^^^^^^^^^^
#
# Phase errors are implemented in CRTomo as:
#
# :math:`\delta \phi = A1*abs(R)^B1 + A2*abs(pha) + p0`
#
# In practise, however, only the relative and absolute parameters A2 and p0 are
# used:
# 5% relative phase error
tdm.crtomo_cfg['pha_rel'] = 5

# 1 mrad absolute phase error
tdm.crtomo_cfg['pha_abs'] = 1

###############################################################################
