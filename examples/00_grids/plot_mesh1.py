#!/usr/bin/env python
# *-* coding: utf-8 *-*
"""
Creating and handling meshes 1
==============================

Unstructured meshes (with triangular cells) for CRTomo can be created in
various ways. We now support two ways:

* using the :py:meth:`crtomo.crt_grid.create_surface_grid` function
* using the command line command :py:mod:`cr_trig_create`
* there is a new (2026) interface to  :py:mod:`cr_trig_create`,
  :py:class:`crtomo.mesh_interface.CRTomoGMSHMeshGenerator`.
  Please refer to the corresponding example for more information:
  :ref:`sphx_glr__examples_00_grids_plot_mesh_generation.py`

This example only shows the usage of the first approach.

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
# Create a simple surface mesh using the following wrapper
mesh0 = crtomo.crt_grid.create_surface_grid(
   nr_electrodes=10,
   spacing=1.5,
)
mesh0.plot()

# number the electrodes (useful for numerical studies)
mesh0.plot_grid(plot_electrode_numbers=True)

# save this grid to disc
mesh0.save_elem_file('elem.dat')
mesh0.save_elec_file('elec.dat')

###############################################################################
# The mesh can be read from disk:
mesh1 = crtomo.crt_grid('elem.dat', 'elec.dat')
print(mesh1)
mesh1.plot()

###############################################################################
# Create a grid with layering
mesh = crtomo.crt_grid.create_surface_grid(
    nr_electrodes=10,
    spacing=1.5,
    lines=[0.5, 1],
)
print(mesh)
mesh.plot()
###############################################################################
# Some notes on characteristic lengths
# ------------------------------------
# You have some control over the final mesh cell size during mesh generation
# using the **char_lengths** parameter.
# This parameter is either one float, or a list/tuple of 4 floats.
# If four values are provided, they determine the requested cell size at:
#
# * the electrodes
# * the boundaries
# * extra lines
# * extra nodes
# .. warning::
#
#   Always make sure to at least put one node between adjacent electrodes.
#   Better, two.

mesh = crtomo.crt_grid.create_surface_grid(
    nr_electrodes=10,
    spacing=1.5,
    char_lengths=[1.7, 1.7, 1.7, 1.7],
)
mesh.plot(
    title='Same char. length everywhere',
    figsize=(8 / 2.54, 4 / 2.54), dpi=150
)

mesh = crtomo.crt_grid.create_surface_grid(
    nr_electrodes=10,
    spacing=1.5,
    char_lengths=2,
)
mesh.plot(
    title='Same char. length everywhere, simplified',
    figsize=(8 / 2.54, 4 / 2.54), dpi=150
)

mesh = crtomo.crt_grid.create_surface_grid(
    nr_electrodes=10,
    spacing=1.5,
    char_lengths=[0.25, 1, 1, 1],
)
mesh.plot(
    title='refinement at electrodes',
    figsize=(8 / 2.54, 4 / 2.54), dpi=150
)

mesh = crtomo.crt_grid.create_surface_grid(
    nr_electrodes=10,
    spacing=1.5,
    lines=[2],
    char_lengths=[1, 1, 0.25, 1],
)
print(mesh)
mesh.plot(
    title="refinement at extra line",
    figsize=(8 / 2.54, 4 / 2.54), dpi=150
)
