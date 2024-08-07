#!/usr/bin/env python
# *-* coding: utf-8 *-*
"""
Creating and handling meshes
============================

Grids for CRTomo can be created in multiple ways. We now support two ways:

* using the :py:meth:`crtomo.crt_grid.create_surface_grid` function
* using the command line command :py:mod:`cr_trig_create`

This example only shows the usage of the former approach.

.. note::

    The CRTomo documentation is inconsistent in the use of the terms `mesh` and
    `grid`. Usually, a grid refers to a mesh with regularly spaced node
    locations in x and z direction.
    CRTomo also supports triangular meshes, which are still referred to as
    *grids* throughout the code and documentation.

"""
###############################################################################
# The top level crtomo import suffices for most tasks
import crtomo

###############################################################################
# Create a simple surface grid with this wrapper
grid = crtomo.crt_grid.create_surface_grid(
   nr_electrodes=10,
   spacing=1.5,
)
grid.plot_grid()

# number the electrodes (useful for numerical studies)
grid.plot_grid(plot_electrode_numbers=True)

# save this grid to disc
grid.save_elem_file('elem.dat')
grid.save_elec_file('elec.dat')

###############################################################################
# The mesh can be read from disk:
grid1 = crtomo.crt_grid('elem.dat', 'elec.dat')
print(grid1)
grid1.plot_grid()

###############################################################################
# Create a grid with layering
grid = crtomo.crt_grid.create_surface_grid(
    nr_electrodes=10,
    spacing=1.5,
    lines=[0.5, 1],
)
print(grid)
grid.plot_grid()
