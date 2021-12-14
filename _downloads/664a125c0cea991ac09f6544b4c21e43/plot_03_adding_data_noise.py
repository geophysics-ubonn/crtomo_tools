#!/usr/bin/env python
r"""
Adding data noise to measurements
=================================

Especially for synthetic modeling/inversion studies it is important to add data
noise to the data.

While CRTomo contains built-in functionality to add data noise, we now
recommend to add noise manually for better control.

Note that error parameters in the inversion context usually refer to standard
deviations in the statistical sense. This implies that the actual realization
of a given data noise component can be smaller or larger than the specified
noise level. As such it **may** be possible/useful to sometimes reduce a given
noise estimate below the actual noise level added to synthetic data.

For further reading, see:

https://en.wikipedia.org/wiki/Pseudorandom_number_generator
"""
###############################################################################
# Imports
import numpy as np
import crtomo

###############################################################################
# Setup: Generate some synthetic data
mesh = crtomo.crt_grid.create_surface_grid(nr_electrodes=10, spacing=1)
tdm = crtomo.tdMan(grid=mesh)

tdm.add_homogeneous_model(100, 0)
tdm.configs.gen_dipole_dipole(skipc=0)
rmag = tdm.measurements()[:, 0]
rpha = tdm.measurements()[:, 1]

###############################################################################
# Generate data noise
# -------------------
# For synthetic studies a good starting point is that the structure of the
# actual noise should be the same as used in the inversion error model. As such
# we use a linear model for the magnitude noise components, and an absolute
# standard deviation for the phase values.

# Important: ALWAYS initialize the random number generator using a seed!
np.random.seed(2048)

# absolute component in [Ohm ]
noise_level_rmag_absolute = 0.01
# relative component [0, 1]
noise_level_rmag_relative = 0.05

noise_rmag = np.random.normal(
    loc=0,
    scale=rmag * noise_level_rmag_relative + noise_level_rmag_absolute
)

rmag_with_noise = rmag + noise_rmag

# 0.5 mrad absolute noise level
noise_level_phases = 0.5

noise_rpha = np.random.normal(
    loc=0,
    scale=noise_level_phases
)
rpha_with_noise = rpha + noise_rpha

# register the noise-added data as new measurements and mark them for use in a
# subsequent inversion
tdm.register_measurements(rmag_with_noise, rpha_with_noise)

###############################################################################
# Remove physically implausible negative magnitude values
indices = np.where(rmag_with_noise <= 0)[0]
tdm.configs.delete_data_points(indices)
