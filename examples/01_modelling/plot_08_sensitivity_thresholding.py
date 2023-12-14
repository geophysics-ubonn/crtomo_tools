#!/usr/bin/env python
"""
Segmenting sensitivity distributions
====================================

"""
###############################################################################
# Imports
import numpy as np
import crtomo
import pylab as plt

###############################################################################
# create and save a FEM-grid
grid = crtomo.crt_grid.create_surface_grid(
    nr_electrodes=40,
    spacing=0.25,
    depth=10,
    char_lengths=[0.1, 0.5, 0.5, 0.5],
    left=5,
    right=5,
)
grid.plot_grid()

grid.save_elem_file('elem.dat')
grid.save_elec_file('elec.dat')

###############################################################################
# create the measurement configuration
configs = np.array((
    (1, 20, 2, 18),
))


###############################################################################
# for different background, plot the sensitivities
bg = 100
td = crtomo.tdMan(grid=grid)
td.configs.add_to_configs(configs)
td.add_homogeneous_model(bg, 0)
td.model(sensitivities=True)

fig, ax = plt.subplots(1, 1)
s_abs = np.abs(td.parman.parsets[2])

threshold = np.quantile(s_abs, 0.75)
s_abs[s_abs <= threshold] = 0
s_abs[s_abs > threshold] = 1

td.plot.plot_elements_to_ax(
    s_abs,
    ax=ax,
    plot_colorbar=True,
    cmap_name='binary',
)
fig.tight_layout()
fig.show()
fig.savefig('test_sens.jpg', dpi=300)
# sphinx_gallery_thumbnail_number = -1
