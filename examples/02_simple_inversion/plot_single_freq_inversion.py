#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A single-frequency inversion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Full processing of one frequency of one timestep of the data from Weigand and
Kemna 2017 (Biogeosciences).
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
# import the sEIT data set
seit = reda.sEIT()
seit.import_eit_fzj(
    'data/bnk_raps_20130408_1715_03_einzel.mat',
    'data/configs.dat'
)

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
k = geom_facs.compute_K_numerical(seit.data, settings)
seit.data = geom_facs.apply_K(seit.data, k)
fix_sign_with_K(seit.data)
###############################################################################
# apply correction factors, as described in Weigand and Kemna, 2017 BG
corr_facs_nor = np.loadtxt('data/corr_fac_avg_nor.dat')
corr_facs_rec = np.loadtxt('data/corr_fac_avg_rec.dat')
corr_facs = np.vstack((corr_facs_nor, corr_facs_rec))
seit.data, cfacs = eit_fzj.apply_correction_factors(seit.data, corr_facs)

###############################################################################
# apply some filters
# import IPython
# IPython.embed()
seit.filter('r < 0')
seit.filter('rho_a < 15 or rho_a > 35')
seit.filter('rpha < - 40 or rpha > 3')
seit.filter('rphadiff < -5 or rphadiff > 5')
seit.filter('k > 400')
seit.filter('rho_a < 0')
seit.filter('a == 12 or b == 12 or m == 12 or n == 12')
seit.filter('a == 13 or b == 13 or m == 13 or n == 13')

seit.filter_incomplete_spectra(flimit=300, percAccept=85)
seit.print_data_journal()
seit.print_log()
###############################################################################
# export to volt.dat file
# note that this is not required for the following code
with reda.CreateEnterDirectory('output_single_freq_inversion'):
    seit.export_to_crtomo_one_frequency('volt.dat', 70.0, 'nor')

###############################################################################
# alternatively: directly create a tdman object that represents a
# single-frequency inversion with CRTomo
import crtomo
grid = crtomo.crt_grid('data/elem.dat', 'data/elec.dat')
tdman = seit.export_to_crtomo_td_manager(grid, frequency=70.0)

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
outdir = 'output_single_freq_inversion/tomodir_inversion'
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
with reda.CreateEnterDirectory('output_single_freq_inversion'):
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
with reda.CreateEnterDirectory('output_single_freq_inversion'):
    r[0].savefig('rmag.png', bbox_inches='tight')

r = tdman.plot.plot_elements_to_ax(
    tdman.a['inversion']['rpha'][-1],
    plot_colorbar=True,
    cmap_name='jet_r',
)
with reda.CreateEnterDirectory('output_single_freq_inversion'):
    r[0].savefig('rpha_last_it.png', bbox_inches='tight')
