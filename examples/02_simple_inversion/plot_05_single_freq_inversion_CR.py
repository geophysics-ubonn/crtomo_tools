#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A single-frequency inversion (CR container)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Full processing of one frequency of one timestep of the data from Weigand and
Kemna 2017 (Biogeosciences).

This example uses the CR container to load single-frequency data from a
CRTomo-style volt.dat file.
"""
###############################################################################
# imports
import os

import numpy as np

import reda
import reda.utils.geometric_factors as geom_facs
from reda.utils.fix_sign_with_K import fix_sign_with_K
import reda.importers.eit_fzj as eit_fzj
###############################################################################
# define an output directory for all files
output_directory = 'output_single_freq_inversion_CR'

###############################################################################
cr = reda.CR()
cr.import_crtomo_data('data/volt.dat')
# this is a container measurement, we need to compute geometric factors using

# numerical modeling
# Note that this only work if CRMod is available
settings = {
    'rho': 100,
    'elem': 'data/elem.dat',
    'elec': 'data/elec.dat',
    'sink_node': '6467',
    '2D': True,
}
k = geom_facs.compute_K_numerical(cr.data, settings)
cr.data = geom_facs.apply_K(cr.data, k)
fix_sign_with_K(cr.data)
print(cr.data.iloc[0:5])
###############################################################################
# apply correction factors, as described in Weigand and Kemna, 2017 BG
corr_facs_nor = np.loadtxt('data/corr_fac_avg_nor.dat')
corr_facs_rec = np.loadtxt('data/corr_fac_avg_rec.dat')
corr_facs = np.vstack((corr_facs_nor, corr_facs_rec))
cr.data, cfacs = eit_fzj.apply_correction_factors(cr.data, corr_facs)

###############################################################################
# apply some filters
# import IPython
# IPython.embed()
cr.filter('r < 0')
cr.filter('rho_a < 15 or rho_a > 35')
cr.filter('rpha < - 40 or rpha > 3')
cr.filter('rphadiff < -5 or rphadiff > 5')
cr.filter('k > 400')
cr.filter('rho_a < 0')
cr.filter('a == 12 or b == 12 or m == 12 or n == 12')
cr.filter('a == 13 or b == 13 or m == 13 or n == 13')

# NOTE: this is also a single-frequency filtering,

cr.print_data_journal()
cr.print_log()
###############################################################################
# export to volt.dat file
# note that this is not required for the subsequent code
with reda.CreateEnterDirectory(output_directory):
    cr.export_crtomo('volt.dat', 'nor')

###############################################################################
# alternatively: directly create a tdman object that represents a
# single-frequency inversion with CRTomo
import crtomo
grid = crtomo.crt_grid('data/elem.dat', 'data/elec.dat')
tdman = cr.export_to_crtomo_td_manager(grid, norrec='nor')

# set inversion settings
tdman.crtomo_cfg['robust_inv'] = 'F'
tdman.crtomo_cfg['mag_abs'] = 0.012
tdman.crtomo_cfg['mag_rel'] = 0.5
tdman.crtomo_cfg['hom_bg'] = 'T'
tdman.crtomo_cfg['d2_5'] = 0
tdman.crtomo_cfg['fic_sink'] = 'T'
tdman.crtomo_cfg['fic_sink_node'] = 6467

# run the inversion, use the given output directory to store the CRTomo
# directory structure for later use
# only invert if the output directory does not exists
outdir = '{}/tomodir_inversion'.format(output_directory)
if not os.path.isdir(outdir):
    tdman.invert(
        catch_output=False,
        output_directory=outdir,
    )
else:
    tdman.read_inversion_results(outdir)
    print('Statistics of last iteration:')
    print(tdman.inv_stats.iloc[-1])

###############################################################################
# evolution of the inversion
fig = tdman.plot_inversion_evolution()
with reda.CreateEnterDirectory(output_directory):
    fig.savefig('inversion_evolution.png')

# the statistics are stored in a data frame
print(tdman.inv_stats)
print(tdman.inv_stats.columns)

###############################################################################
# evolution of data misfits
fig = tdman.plot_eps_data()
fig = tdman.plot_eps_data_hist()

# eps data is found here:
tdman.eps_data

###############################################################################
r = tdman.plot.plot_elements_to_ax(
    tdman.a['inversion']['rmag'][-1],
    plot_colorbar=True,
    cmap_name='jet_r',
)
with reda.CreateEnterDirectory(output_directory):
    r[0].savefig('rmag.png', bbox_inches='tight')

r = tdman.plot.plot_elements_to_ax(
    tdman.a['inversion']['rpha'][-1],
    plot_colorbar=True,
    cmap_name='jet_r',
)
with reda.CreateEnterDirectory(output_directory):
    r[0].savefig('rpha_last_it.png', bbox_inches='tight')
