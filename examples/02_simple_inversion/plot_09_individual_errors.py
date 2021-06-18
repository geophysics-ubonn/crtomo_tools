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
import crtomo

mesh = crtomo.crt_grid.create_surface_grid(nr_electrodes=10, spacing=1)
tdm = crtomo.tdMan(grid=mesh)

tdm.add_homogeneous_model(100, 0)
tdm.configs.gen_dipole_dipole(skipc=0)
measurements = tdm.measurements()

import numpy as np
mag_errors = np.arange(0, measurements.shape[0]) + 1
pha_errors = np.arange(0, measurements.shape[0]) + 1000

tdm.register_data_errors(
    tdm.configs.add_measurements(mag_errors),
    tdm.configs.add_measurements(pha_errors),
)

tdm.save_to_tomodir('td_test')
