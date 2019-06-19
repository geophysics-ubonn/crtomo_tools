#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full sEIT data processing example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Full processing of one timestep of the data from Weigand and Kemna 2017
(Biogeosciences).

"""
###############################################################################
# imports
import os
import subprocess

import reda
from reda.utils.fix_sign_with_K import fix_sign_with_K
import reda.utils.geometric_factors as geom_facs
import numpy as np
import reda.importers.eit_fzj as eit_fzj
###############################################################################
# data import
seit = reda.sEIT()
seit.import_eit_fzj(
    'data/bnk_raps_20130408_1715_03_einzel.mat',
    'data/configs.dat'
)
print(seit.data[['a', 'b', 'm', 'n']].iloc[0:10])
###############################################################################
# compute geometric factors and correct for signs/phase shifts by pi
settings = {
    'rho': 100,
    'elem': 'data/elem.dat',
    'elec': 'data/elec.dat',
    'sink_node': '6467',
    '2D': True,
}
k = geom_facs.compute_K_numerical(seit.data, settings)
seit.data = geom_facs.apply_K(seit.data, k)

# input('nr 2, press enter to continue')
fix_sign_with_K(seit.data)
###############################################################################
# apply correction factors for 2D rhizotron tank
corr_facs_nor = np.loadtxt('data/corr_fac_avg_nor.dat')
corr_facs_rec = np.loadtxt('data/corr_fac_avg_rec.dat')
corr_facs = np.vstack((corr_facs_nor, corr_facs_rec))
seit.data, cfacs = eit_fzj.apply_correction_factors(seit.data, corr_facs)

###############################################################################
# apply data filters
seit.filter('r < 0')
seit.filter('rho_a < 15 or rho_a > 35')
seit.filter('rpha < - 40 or rpha > 3')
seit.filter('rphadiff < -5 or rphadiff > 5')
seit.filter('k > 400')
seit.filter('rho_a < 0')
seit.filter('a == 12 or b == 12 or m == 12 or n == 12')
seit.filter('a == 13 or b == 13 or m == 13 or n == 13')

# import IPython
# IPython.embed()
seit.print_data_journal()
seit.filter_incomplete_spectra(flimit=300, percAccept=85)
seit.print_data_journal()
seit.print_log()
###############################################################################

import crtomo
grid = crtomo.crt_grid('data/elem.dat', 'data/elec.dat')

seitinv = seit.export_to_crtomo_seit_manager(grid, norrec='nor')
seitinv.crtomo_cfg['robust_inv'] = 'F'
seitinv.crtomo_cfg['mag_abs'] = 0.012
seitinv.crtomo_cfg['mag_rel'] = 0.5
seitinv.crtomo_cfg['hom_bg'] = 'T'
seitinv.crtomo_cfg['d2_5'] = 0
seitinv.crtomo_cfg['fic_sink'] = 'T'
seitinv.crtomo_cfg['fic_sink_node'] = 6467
seitinv.apply_crtomo_cfg()


###############################################################################
# now run the inversion
# we do this the "old" style using the command td_run_all_local, which is also
# included in the crtomo_tools

# only invert if the sipdir does not already exist
if not os.path.isdir('sipdir'):
    # save to a sip-directory
    seitinv.save_to_eitdir('sipdir')

    os.chdir('sipdir')
    subprocess.call('td_run_all_local -t 1 -n 2', shell=True)
    os.chdir('..')

###############################################################################
# Now plot the results
# at this point all plot scripts for sEIT results are located in old,
# deprecated, packages, and thus we need to write a new plotting tools.
# In the mean time you can enter each tomodir in the sipdir/invmod subdirectory
# and plot it using the single-frequency plot command "td_plot"

# TODO
###############################################################################
import crtomo
import numpy as np
sinv = crtomo.eitMan(seitdir='sipdir/')
# sinv.extract_points('rpha', np.atleast_2d(np.array((-0.5, 13))))
sinv.extract_points('rpha', np.atleast_2d(np.array((20, -5))))
