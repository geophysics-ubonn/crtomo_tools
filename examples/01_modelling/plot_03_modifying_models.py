#!/usr/bin/env python
# *-* coding: utf-8 *-*
"""
Modify a subsurface model
^^^^^^^^^^^^^^^^^^^^^^^^^

"""
###############################################################################
import numpy as np

import crtomo
import matplotlib.pylab as plt
###############################################################################
# load a mesh file that we want to create a model for
grid = crtomo.crt_grid('grid_surface/elem.dat', 'grid_surface/elec.dat')
# create a parameter manager
parman = crtomo.ParMan(grid)

###############################################################################
# we need a plot manager to plot our mesh/model parameters
# notice that we link the parameter manager to the plot manager
plotman = crtomo.pltMan(grid=grid, pm=parman)
###############################################################################
# create an empty parameter set
pid = parman.add_empty_dataset(value=100)

###############################################################################


###############################################################################
fig, ax = plt.subplots()
plotman.plot_elements_to_ax(
    pid,
    ax=ax,
    plot_colorbar=True,
)
fig.savefig('out_03_model.jpg', dpi=300)

