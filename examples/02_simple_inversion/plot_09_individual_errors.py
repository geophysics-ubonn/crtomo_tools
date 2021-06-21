#!/usr/bin/env python
r"""
Using individual data errors in CRTomo
======================================

In principle an individual data error can be assigned to each data point.
However, since data errors are not easily determined in geoelectrical
applications, it is common to employ various data error models that exploit
empirically observed relationships between the data errors and the measurements
themselves.
"""
###############################################################################
# Setup:
import numpy as np
import crtomo

mesh = crtomo.crt_grid.create_surface_grid(nr_electrodes=10, spacing=1)
tdm = crtomo.tdMan(grid=mesh)

tdm.add_homogeneous_model(100, 0)
tdm.configs.gen_dipole_dipole(skipc=0)
measurements = tdm.measurements()

###############################################################################
# Determine individual errors in some meaningful way
mag_errors = np.arange(0, measurements.shape[0]) + 1
pha_errors = np.arange(0, measurements.shape[0]) + 1000

###############################################################################
# Register them
tdm.register_data_errors(
    tdm.configs.add_measurements(mag_errors),
    tdm.configs.add_measurements(pha_errors),
    norm_mag=1,
    norm_pha=1,
)

###############################################################################
# Note that the volt.dat will look slightly different now
tdm.save_to_tomodir('td_test')
