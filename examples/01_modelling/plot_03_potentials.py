#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
Plot a potential distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CRMod can return the potential distribution associated with the current
electrode locations of each registered quadrupole. We can retrieve those
potentials using the tdMan object and plot them. The current distribution can
then be computed
TODO

"""
###############################################################################
# Imports
import numpy as np
import pylab as plt

import crtomo

###############################################################################
# create a tomodir object from an existing FE mesh
grid = crtomo.crt_grid(
    'grid_surface/g2_large_boundary/elem.dat',
    'grid_surface/g2_large_boundary/elec.dat'
)
td = crtomo.tdMan(grid=grid)

###############################################################################
# define configurations
td.configs.add_to_configs(
    np.array((
        (2, 8, 5, 7),
        (1, 3, 10, 7),
    ))
)

###############################################################################
# add a forward model with a conductive region
pid_mag, pid_pha = td.add_homogeneous_model(100, -50)
td.parman.modify_area(
    pid_mag,
    -3, 11,
    -5, -2,
    1,
)

###############################################################################
# compute FEM solution using CRMod
td.model(potentials=True)

###############################################################################
# retrieve the potential distribution of the first quadrupole
# the potential distribution is returned as real and imaginary part
pot_re, pot_im = td.get_potential(0)
pot_mag = np.sqrt(
    pot_re ** 2 + pot_im ** 2
)

# add node data to the parameter manager
# for visualization purposes, we also store a transformed potential
# distribution
nid = td.nodeman.add_data(pot_re)
nid_asinh = td.nodeman.add_data(
    crtomo.plotManager.converter_asinh(pot_re)
)

# Create a nice plot
fig, ax = plt.subplots(1, 1, figsize=(16 / 2.54, 8 / 2.54), sharex=True)
ax.set_title('Current and potential lines', loc='left')

# plot the resistivity distribution of the forward model
td.plot.plot_elements_to_ax(
    pid_mag,
    ax,
    plot_colorbar=True,
    cmap='jet',
    cmap_name='autumn',
    cblabel=r'$\rho~[\Omega m]$',
)

# plot current lines
td.plot.plot_nodes_current_streamlines_to_ax(
    ax,
    # nid_asinh,
    nid,
    pid_mag,
    density=0.6,
)

# plot potential lines (contour lines)
# Note that we use the asinh-transformed potentials here. This ensures nicer
# looking potential lines
td.plot.plot_nodes_contour_to_ax(
    ax,
    nid_asinh,
    # nid,
    plot_colorbar=False,
    fill_contours=False,
    cblevels=21,
    alpha=0.6,
)

ax.set_aspect('equal')

fig.tight_layout()
fig.savefig('out_02_potential_distribution.jpg', dpi=300)
