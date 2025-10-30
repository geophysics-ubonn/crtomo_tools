#!/usr/bin/env python3
# *-* coding: utf-8 *-*

"""
Forward Modeling with CRMod
^^^^^^^^^^^^^^^^^^^^^^^^^^^


"""
###############################################################################
# Imports
import crtomo

###############################################################################
# create a tomodir object from an existing FE mesh
mesh = crtomo.crt_grid(
    'grid_surface/g2_large_boundary/elem.dat',
    'grid_surface/g2_large_boundary/elec.dat'
)
###############################################################################
# create a tomodir-manager object
# this object will hold everything required for modeling and inversion runs
tdm = crtomo.tdMan(grid=mesh)

###############################################################################
# generate measurement configurations
tdm.configs.gen_dipole_dipole(skipc=0)

###############################################################################
# add a forward model with a conductive region
pid_mag, pid_pha = tdm.add_homogeneous_model(100, -50)
tdm.parman.modify_area(
    pid_mag,
    -3, 11,
    -5, -2,
    1,
)

###############################################################################
# plot the forward models
fig, ax = tdm.plot_forward_models(mag_only=True)

###############################################################################
fig, axes = tdm.plot_forward_models(mag_only=False)

###############################################################################
# Two lists are returned, each containing either two figures, or two axes
figs, axes = tdm.plot_forward_models(mag_only=False, separate_figures=True)
figs[0].suptitle('Magnitude model')
figs[1].suptitle('Phase model')

###############################################################################
# Compute FEM solution using CRMod:
tdm.model(silent=True)

###############################################################################
# measurements can now be retrieved
# First column: resistances, second column: phase values [mrad]
measurements = tdm.measurements()
print(measurements)

###############################################################################

# Let's plot a pseudo-section of the modelled data
ert = tdm.get_fwd_reda_container()
print(ert)
print(ert.data)
# compute geometric factors
ert.compute_K_analytical(spacing=1)
# type 2 pseudosection
fig, ax, cb = ert.pseudosection_type2('rho_a', interpolate=False, spacing=1)
# type 2 pseudosection, interpolated
fig, ax, cb = ert.pseudosection_type2('rho_a', interpolate=True, spacing=1)

# sphinx_gallery_multi_image = "single"
