#!/USr/bin/env python
# *-* coding: utf-8 *-*
"""
Single frequency synthetic study
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
###############################################################################
# imports required for the study
import crtomo
import shapely.geometry
import pylab as plt

###############################################################################
# Generate a simple Finite-Element mesh for forward and inverse modeling
grid = crtomo.crt_grid.create_surface_grid(
    nr_electrodes=30, spacing=1, char_lengths=[0.3, 1, 1, 1]
)
fig, ax = grid.plot_grid()
fig.savefig('grid.jpg')

###############################################################################
man = crtomo.tdMan(grid=grid)

###############################################################################
pid_mag, pid_pha = man.add_homogeneous_model(
    magnitude=100, phase=-5
)
# import IPython
# IPython.embed()

man.parman.modify_area(
    pid_mag,
    xmin=1, xmax=5,
    zmin=-5, zmax=-1,
    value=10
)

man.parman.modify_area(
    pid_pha,
    xmin=1, xmax=5,
    zmin=-3, zmax=-2,
    value=-30
)

polygon = shapely.geometry.Polygon((
    (2, 0), (4, -1), (2, -1)
))
man.parman.modify_polygon(pid_mag, polygon, 3)

fig, ax = man.show_parset(pid_mag)
fig.savefig('model_magnitude.jpg')
fig, ax = man.show_parset(pid_pha)
fig.savefig('model_phase.jpg')

###############################################################################
man.configs.gen_dipole_dipole(skipc=0)
man.configs.gen_dipole_dipole(skipc=1)
man.configs.gen_dipole_dipole(skipc=2)

###############################################################################
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

# now add synthetic noise
# TODO

man.save_measurements('volt.dat')

###############################################################################
# prepare the inversion
# note that we create another tdMan class instance for the inversion. This is
# not necessarily required, but serves nicely to separate the different steps
# of the inversion
tdman = crtomo.tdMan(grid=grid)

# inversion settings can be changed here:
print(tdman.crtomo_cfg)

# for example, let's change the relative magnitude error estimate to 7 %
tdman.crtomo_cfg['mag_rel'] = 7

tdman.read_voltages('volt.dat')

# this command actually calls CRTomo and conducts the actual inversion
tdman.invert()
###############################################################################
# The convergence behavior of the inversion is stored in a pandas.DataFrame:
print(tdman.inv_stats)

# save the RMS values of the final iteration
rms_all, rms_mag, rms_pha = tdman.inv_stats.iloc[-1][
    ['dataRMS', 'magRMS', 'phaRMS']
].values
###############################################################################
# As a short reminder, this dictionary contains all information on where to
# find data/results in the tdMan instance
print(tdman.a)

###############################################################################
# We can now visualize the inversion results
fig, ax = tdman.show_parset(
    tdman.a['inversion']['rmag'][-1],
    cmap_name='viridis',
    plot_colorbar=True,
    cblabel=r'$\rho~[\Omega m]$',
    xmin=-2,
    xmax=32,
    zmin=-8,
    title='Magnitude inversion (rms-all: {}, rms-mag: {}'.format(
        rms_all, rms_mag
    ),

)
fig.savefig('inversion_result_rmag.jpg')

fig, ax = tdman.show_parset(
    tdman.a['inversion']['rpha'][-1],
    cmap_name='turbo',
    plot_colorbar=True,
    cblabel=r'$\phi~[mrad]$',
    xmin=-2,
    xmax=32,
    zmin=-8,
    title='Phase inversion result (rms-all: {}, rms-pha: {}'.format(
        rms_all, rms_pha
    ),
)
fig.savefig('inversion_result_rpha.jpg')
