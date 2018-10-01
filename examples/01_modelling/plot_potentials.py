#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
Plot a potential distribution, computed with CRMod
==================================================


"""
###############################################################################
# create a tomodir object
import crtomo
grid = crtomo.crt_grid('grid_surface/elem.dat', 'grid_surface/elec.dat')
td = crtomo.tdMan(grid=grid)

###############################################################################
# define configurations
import numpy as np
td.configs.add_to_configs(
    np.array((
        (1, 10, 5, 7),
        (1, 3, 10, 7),
    ))
)

###############################################################################
# add a homogeneous forward model
td.add_homogeneous_model(100, 0)

###############################################################################
# compute FEM solution using CRMod
td.model(potentials=True)


###############################################################################
# plot first quadrupole
pot_mag, pot_pha = td.get_potential(0)

# add node data to the parameter manager
nid = td.nodeman.add_data(pot_mag)
# TODO: is this the phase, or imaginary part of the potential?
nid_pha = td.nodeman.add_data(pot_pha)

# plot
import pylab as plt
fig, axes = plt.subplots(2, 1, figsize=(16 / 2.54, 8 / 2.54))

ax = axes[0]
td.plot.plot_nodes_pcolor_to_ax(
    ax,
    nid,
    plot_colorbar=True,
)
# td.plot.plot_nodes_contour_to_ax(
#     ax, nid,
#     plot_colorbar=True,

# )
ax.set_aspect('equal')

ax = axes[1]
td.plot.plot_nodes_pcolor_to_ax(
    ax,
    nid_pha,
    plot_colorbar=True,
)
ax.set_aspect('equal')
fig.savefig('test.pdf', bbox_inches='tight')
