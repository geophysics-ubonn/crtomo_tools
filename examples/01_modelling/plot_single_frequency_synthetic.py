#!/usr/bin/env python
# *-* coding: utf-8 *-*
"""
Single frequency synthetic study
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
###################################################################
import crtomo

grid = crtomo.crt_grid.create_surface_grid(
    nr_electrodes=30, spacing=1, char_lengths=[0.3, 1, 1, 1]
)
fig, ax = grid.plot_grid()
fig.savefig('grid.jpg')

man = crtomo.tdMan(grid=grid)
pid_mag, pid_pha = man.add_homogeneous_model(
    magnitude=100, phase=-5
)

man.parman.modify_area(
    pid_mag,
    xmin=1, xmax=5,
    zmin=-3, zmax=-2,
    value=10
)

man.parman.modify_area(
    pid_pha,
    xmin=1, xmax=5,
    zmin=-3, zmax=-2,
    value=-30
)

fig, ax = man.show_parset(pid_mag)
fig.savefig('model_magnitude.jpg')
fig, ax = man.show_parset(pid_pha)
fig.savefig('model_phase.jpg')

man.configs.gen_dipole_dipole(skipc=1)

import pylab as plt
# conduct forward modeling
rmag_rpha_mod = man.measurements()

fig, axes = plt.subplots(2, 1)
ax = axes[0]
ax.hist(rmag_rpha_mod[:, 0], 100)
ax.set_xlabel('magnitudes')
ax = axes[1]
ax.hist(rmag_rpha_mod[:, 1], 100)
ax.set_xlabel('phases')
fig.savefig('modeled_data.jpg')

# now add syntehtic noise
# TODO

man.save_measurements('volt.dat')
tdman = crtomo.tdMan(grid=grid)
tdman.read_voltages('volt.dat')
tdman.invert()

print(tdman.a)
fig, ax = tdman.show_parset(tdman.a['inversion']['rmag'][-1])
fig.savefig('inversion_result.jpg')
