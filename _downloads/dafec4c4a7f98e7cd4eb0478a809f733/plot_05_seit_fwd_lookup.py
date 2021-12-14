#!/usr/bin/env python
"""
Generating sEIT forward models using a mask and a lookup table
==============================================================

"""
import numpy as np

import reda
# used to generate a Cole-Cole response
import sip_models
import crtomo

###############################################################################
# set up plotting facilities - this will often generate better fitting plots
import crtomo.mpl
crtomo.mpl.setup()

###############################################################################
grid = crtomo.crt_grid.create_surface_grid(nr_electrodes=15, spacing=1)

fig, ax = grid.plot_grid()
fig.show()
###############################################################################

# create single-frequency tomodir (i.e, one inversion container)
tdm = crtomo.tdMan(grid=grid)
pid_mask = tdm.parman.add_empty_dataset(value=-1)

fig, ax = tdm.show_parset(pid_mask, plot_colorbar=True, title='Empty Mask')
fig.show()
###############################################################################

for index, (x, z) in enumerate(tdm.grid.get_element_centroids()):
    # print(index, x, z)
    # depending on location assign a different pixel. This should be an integer
    # that is later used to assign spectra to all pixels
    tdm.parman.parsets[pid_mask][index] = int((np.abs(z) * 3))
fig, ax = tdm.show_parset(pid_mask, plot_colorbar=True, title='Mask Indices')
fig.show()
###############################################################################

np.savetxt('mask.dat', tdm.parman.parsets[pid_mask], '%i')

mask = np.loadtxt('mask.dat').astype(int)

# generate the lookup table
lookup_table = {}

frequencies = np.logspace(-3, 3, 6)
colecole = sip_models.res.cc.cc(frequencies=frequencies)
spectrum = colecole.response([100, 0.1, 0.04, 0.6])

for index in np.unique(mask.astype(int)):
    lookup_table[index] = spectrum

eit = crtomo.eitMan(grid=grid, frequencies=frequencies)
eit.assign_sip_signatures_using_mask(mask, lookup_table)
# %%
# sphinx_gallery_defer_figures
r = eit.plot_forward_models(
    maglim=[80, 110],
    phalim=[-30, 0],
)
r['rmag']['fig'].show()
r['rpha']['fig'].show()

fig, _ = spectrum._plot()
fig.show()

configs = reda.ConfigManager()
configs.nr_electrodes = 10
configs.gen_dipole_dipole(skipc=0)
eit.add_to_configs(configs.configs)
r = eit.measurements()
# sphinx_gallery_thumbnail_number = -1
