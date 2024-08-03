#!/usr/bin/env python
# *-* coding: utf-8 *-*
"""
Modify a subsurface model
^^^^^^^^^^^^^^^^^^^^^^^^^

A subsurface model is basically an array of the same length as the number of
model cells in the finite-element mesh.
We refer to such an array as a parameter set (or **parset**), and the numeric
(integer) id that refers to such as parset in the parameter manager as a *pid*.
For DC forward modeling you only need one resistivity model (with M resistivity
values for the M model cells), while for complex resistivity modeling you need
a resistivity and a phase array.

Parameter sets are usually manager using the parameter manger class
:py:class:`crtomo.ParManager.ParMan.modify_polygon`, which also has an alias to
`crtomo.ParMan`.

If you are using a single-frequency tomodir object `tdm`
(:py:class:'crtomo.tdMan`), one parameter manager is already initialized as
`tdm.parman`.

There are various ways to modify such an array

* modify by index
* modify by polygon
* add Gaussian anomalies (link to other example)


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
pid_mag = parman.add_empty_dataset(value=100)

###############################################################################
# Modify by polygon
# -----------------
# :py:meth:`crtomo.ParMan.modify_polygon`
from shapely.geometry import Polygon # noqa:402
poly = Polygon([
    [0, 0],
    [1, 0],
    [1, -1],
    [0, -2],
])
parman.modify_polygon(pid_mag, poly, 66)

###############################################################################
fig, ax = plt.subplots()
plotman.plot_elements_to_ax(
    pid_mag,
    ax=ax,
    plot_colorbar=True,
)
# lets draw the original polygon
from shapely.plotting import plot_polygon # noqa: 402
plot_polygon(poly, ax=ax, color='k')

fig.savefig('out_03_model.jpg', dpi=300)
