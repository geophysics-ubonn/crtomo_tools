#!/USr/bin/env python
# *-* coding: utf-8 *-*
r"""
Resolution matrix and other measures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example aims to provide a full discussion of a single-frequency complex
resistivity simulation study.
A simulation study usually consists of the following steps:

1) Generate a Finite-Element mesh for the forward modeling step. This
   includes defining the bounding geometry of the subsurface to investigate
   (the *boundary*), electrode positions on the boundary and embedded in
   the subsurface, as well as internal geometry.

   Internal geometry reflects the geophysical scenario that is being
   simulated and is usually driven by the *scientific research context* of
   the study.
2) Create a complex-resistivity subsurface model
3) Generate measurement configurations
4) Conduct the forward modeling
5) Add normally-distributed noise to the synthetic data. Remove all data
   points that have a resulting transfer impedance below 0 (CRMod
   automatically outputs measurement configurations with positive geometric
   factors. Therefore, all measurements :math:`<= 0 \Omega` are deemed as
   physically unacceptable.
6) Conduct an inversion using the synthetic data contaminated with
   synthetic noise. Usually error estimates are chosen equal to the noise
   distributions previously added. Depending on the scope of the synthetic
   study, additional noise levels can be estimated to account for other
   error components.
7) Analyse inversion results.
"""
###############################################################################
# Note that we use a context manager from reda to place output files in a
# separate output directory
import reda
# imports required for the study
import crtomo
import numpy as np
import shapely.geometry
import pylab as plt

###############################################################################
# 1. Generate forward mesh
# ------------------------
# Generate a simple Finite-Element mesh for forward and inverse modeling.
# For more information on mesh creation, please refer to
# :ref:`grid_creation:irregular grids`.
# :doc:`grid_creation`.

grid = crtomo.crt_grid.create_surface_grid(
    nr_electrodes=30,
    spacing=1,
    # these lengths determine the size of mesh cells
    char_lengths=[0.3, 1, 1, 1]
)

fig, ax = grid.plot_grid()
with reda.CreateEnterDirectory('output_plot_11'):
    fig.savefig('grid.jpg', dpi=300, bbox_inches='tight')

###############################################################################
# Create a tdManager instance used for single-frequency forward and inverse
# modeling
man = crtomo.tdMan(grid=grid)

###############################################################################
# 2. Create complex-resistivity subsurface model
# ----------------------------------------------

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

with reda.CreateEnterDirectory('output_plot_11'):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    fig, ax = man.show_parset(
        pid_mag,
        ax=axes[0],
        plot_colorbar=True,
        cblabel=r'$\rho~[\Omega m]$',
        cmap_name='viridis',
        title='Magnitude forward model',
    )
    fig.savefig('model_magnitude.jpg', dpi=300)
    fig, ax = man.show_parset(
        pid_pha,
        ax=axes[1],
        plot_colorbar=True,
        cblabel=r'$\varphi~[mrad]$',
        cmap_name='turbo',
        title='Phase forward model',
    )
    fig.savefig('model_phase.jpg', dpi=300)

###############################################################################
# 3. Generate Measurement Configurations
# --------------------------------------
man.configs.gen_dipole_dipole(skipc=0)
man.configs.gen_dipole_dipole(skipc=1)
man.configs.gen_dipole_dipole(skipc=2)

###############################################################################
# 4. Generate Measurement Configurations
# --------------------------------------
rmag_rpha_mod = man.measurements()
rmag = man.measurements()[:, 0]
rpha = man.measurements()[:, 1]

###############################################################################
# Let's plot the results
with reda.CreateEnterDirectory('output_plot_11'):
    fig, axes = plt.subplots(3, 1)

    ax = axes[0]
    ax.set_title('Measured transfer resistances', loc='left', fontsize=8)
    ax.hist(rmag_rpha_mod[:, 0], 100)
    ax.set_xlabel(r'$R~[\Omega]$')
    ax.set_ylabel('Count [-]')

    ax = axes[1]
    ax.set_title(
        'Measured transfer resistances (log10)', loc='left', fontsize=8
    )
    ax.hist(rmag_rpha_mod[:, 0], 100)
    ax.set_xlabel(r'$R~[\Omega]$')
    ax.set_xscale('log')
    ax.set_ylabel('Count [-]')

    ax = axes[2]
    ax.set_title('Phase measurements', loc='left', fontsize=8)
    ax.hist(rmag_rpha_mod[:, 1], 100)
    ax.set_xlabel('Phases [mrad]')
    ax.set_ylabel('Count [-]')

    fig.tight_layout()
    fig.savefig('modeled_data.jpg', dpi=300)

###############################################################################
# 5. Add noise to synthetic data
# ------------------------------
# Important: ALWAYS initialize the random number generator using a seed!
np.random.seed(2048)

# absolute component in [Ohm ]
noise_level_rmag_absolute = 0.01
# relative component [0, 1]
noise_level_rmag_relative = 0.05

noise_rmag = np.random.normal(
    loc=0,
    scale=rmag * noise_level_rmag_relative + noise_level_rmag_absolute
)

rmag_with_noise = rmag + noise_rmag

# 0.5 mrad absolute noise level
noise_level_phases = 0.5

noise_rpha = np.random.normal(
    loc=0,
    scale=noise_level_phases
)
rpha_with_noise = rpha + noise_rpha

# register the noise-added data as new measurements and mark them for use in a
# subsequent inversion
man.register_measurements(rmag_with_noise, rpha_with_noise)

# Remove physically implausible negative magnitude values
indices = np.where(rmag_with_noise <= 0)[0]
man.configs.delete_data_points(indices)

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
tdman.crtomo_cfg['mag_rel'] = 5
tdman.crtomo_cfg['mag_abs'] = 0.01
tdman.crtomo_cfg['pha_abs'] = 0.5

# + 2: MCM_1 = A^T C_d^-1 A + \lambda C_m^{-1}
# + 4: write out main diagonals of Resolution matrix
tdman.crtomo_cfg['mswitch'] = 4 + 2

tdman.read_voltages('volt.dat')

###############################################################################
# 6. Conduct the actual inversion
# -------------------------------
# this command actually calls CRTomo and conducts the actual inversion
tdman.invert()

###############################################################################
# 7. Analyse inversion results
# ----------------------------

# The convergence behavior of the inversion is stored in a pandas.DataFrame:
print(tdman.inv_stats)

# save the RMS values of the final iteration
final_rms_all, final_rms_mag, final_rms_pha = tdman.inv_stats.iloc[-1][
    ['dataRMS', 'magRMS', 'phaRMS']
].values
print('Final RMS mag+pha:', final_rms_all)
print('Final RMS mag:', final_rms_mag)
print('Final RMS pha:', final_rms_pha)

###############################################################################
# The primary convergence goal of the inversion is to reach an RMS near 1.
# This would indicate that the data is described by the subsurface model within
# their data bounds.
# We do not care about update loops of the inversion. Therefore, filter the
# convergence information and plot it.
inv_stats_main = tdman.inv_stats.query('type=="main"').reset_index()

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(
    inv_stats_main['dataRMS'], '.-', label=r'$\mathbf{RMS}_{\mathrm{all}}$')
ax.plot(
    inv_stats_main['magRMS'], '.-', label=r'$\mathbf{RMS}_{\mathrm{mag}}$')
ax.plot(
    inv_stats_main['phaRMS'], '.-', label=r'$\mathbf{RMS}_{\mathrm{pha}}$')
ax.axhline(y=1.0, color='black', linestyle='dotted', label='target RMS')
ax.legend()
ax.set_xlabel('Iteration number')
ax.set_ylabel('RMS~[-]')
fig.tight_layout()

with reda.CreateEnterDirectory('output_plot_11'):
    fig.savefig(
        'inversion_convergence_evolution.jpg', dpi=300, bbox_inches='tight'
    )

###############################################################################
# As a short reminder, this dictionary contains all information on where to
# find data/results in the tdMan instance
print(tdman.a)

###############################################################################
# We can now visualize the inversion results.
# First, plot the L1-Coverage to get an idea on the information distribution in
# the subsurface.
# We will later use the coverage to add transparency ('alpha') to our plots, or
# to apply a filter mask to the results.
# The intention here is to visually remove all pixels that definitively do not
# hold useful subsurface information. Be careful: High sensitivities do not
# always imply good recovery of the subsurface!
fig, ax = tdman.show_parset(
    tdman.a['inversion']['l1_dw_log10_norm'],
    plot_colorbar=True,
    cmap_name='magma',
    xmin=-2,
    xmax=32,
    zmin=-8,
    title='L1-Coverage (normalized)',
    cblabel=r'$log_{10}(Cov_{L1}~[-])$',
)
with reda.CreateEnterDirectory('output_plot_11'):
    fig.savefig('inversion_coverage_l1.jpg', dpi=300, bbox_inches='tight')

log10_threshold = -2
mask = (tdman.parman.parsets[
    tdman.a['inversion']['l1_dw_log10_norm']
] > log10_threshold).astype(int)

###############################################################################
# diagonal entries of resolution matrix
fig, ax = tdman.show_parset(
    tdman.a['inversion']['resm'],
    cmap_name='turbo',
    plot_colorbar=True,
    cblabel=r'DIAG RESM',
    xmin=-2,
    xmax=32,
    zmin=-8,
)

with reda.CreateEnterDirectory('output_plot_11'):
    fig.savefig('inversion_diag_resm.jpg', dpi=300, bbox_inches='tight')

###############################################################################
# Plot Magnitude results:
fig, axes = plt.subplots(3, 1, figsize=(8, 11), sharex=True)

fig, ax = tdman.show_parset(
    tdman.a['inversion']['rmag'][-1],
    ax=axes[0],
    cmap_name='viridis',
    plot_colorbar=True,
    cblabel=r'$\rho~[\Omega m]$',
    xmin=-2,
    xmax=32,
    zmin=-8,
    title='Magnitude inversion (rms-all: {}, rms-mag: {})'.format(
        final_rms_all,
        final_rms_mag
    ),
)
ax.set_title('No mask', loc='left', fontsize=8)

fig, ax = tdman.show_parset(
    tdman.a['inversion']['rmag'][-1],
    ax=axes[1],
    cmap_name='viridis',
    plot_colorbar=True,
    cblabel=r'$\rho~[\Omega m]$',
    xmin=-2,
    xmax=32,
    zmin=-8,
    title='Magnitude inversion (rms-all: {}, rms-mag: {})'.format(
        final_rms_all,
        final_rms_mag
    ),
    cid_alpha=tdman.a['inversion']['l1_dw_log10_norm'],
    # default is: 3
    alpha_sens_threshold=2,
)

fig, ax = tdman.show_parset(
    tdman.a['inversion']['rmag'][-1],
    ax=axes[2],
    cmap_name='viridis',
    plot_colorbar=True,
    cblabel=r'$\rho~[\Omega m]$',
    xmin=-2,
    xmax=32,
    zmin=-8,
    title='Magnitude inversion (rms-all: {}, rms-mag: {})'.format(
        final_rms_all,
        final_rms_mag
    ),
    cid_alpha=mask,
)
with reda.CreateEnterDirectory('output_plot_11'):
    fig.savefig('inversion_result_rmag_cov_mask.jpg')

###############################################################################
# Phase inversion results:

fig, ax = tdman.show_parset(
    tdman.a['inversion']['rpha'][-1],
    cmap_name='turbo',
    plot_colorbar=True,
    cblabel=r'$\phi~[mrad]$',
    xmin=-2,
    xmax=32,
    zmin=-8,
    title='Phase inversion result (rms-all: {}, rms-pha: {})'.format(
        final_rms_all,
        final_rms_mag
    ),
    cid_alpha=mask,
)
with reda.CreateEnterDirectory('output_plot_11'):
    fig.savefig('inversion_result_rpha.jpg')

# sphinx_gallery_thumbnail_number = -1
