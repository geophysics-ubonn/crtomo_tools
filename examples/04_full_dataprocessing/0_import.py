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
import reda
###############################################################################
seit = reda.sEIT()
seit.import_eit_fzj(
    'data/bnk_raps_20130408_1715_03_einzel.mat',
    'data/configs.dat'
)
print(seit.data[['a', 'b', 'm', 'n']].iloc[0:10])
# input('nr 1, press enter to continue')
import reda.utils.geometric_factors as geom_facs
settings = {
    'rho': 100,
    'elem': 'data/elem.dat',
    'elec': 'data/elec.dat',
    'sink_node': '6467',
    '2D': True,
}
k = geom_facs.compute_K_numerical(seit.data, settings)
seit.data = geom_facs.apply_K(seit.data, k)
print(seit.data[['a', 'b', 'm', 'n']].iloc[0:10])
# input('nr 2, press enter to continue')
from reda.utils.fix_sign_with_K import fix_sign_with_K
fix_sign_with_K(seit.data)
print(seit.data.columns)
seit.data.query(
    'a == 1 and b == 27 and m == 32 and n == 28'
)[['a', 'b', 'm', 'n', 'k']]

print(seit.data.groupby('frequency')['a'].describe())
import numpy as np
import reda.importers.eit_fzj as eit_fzj

corr_facs_nor = np.loadtxt('data/corr_fac_avg_nor.dat')
corr_facs_rec = np.loadtxt('data/corr_fac_avg_rec.dat')
corr_facs = np.vstack((corr_facs_nor, corr_facs_rec))
# print('before')
# print(seit.data.loc[seit.data.index[0:10], ['a', 'b', 'm', 'n', 'r']])
df, cfacs = eit_fzj.apply_correction_factors(seit.data, corr_facs)
seit.data = df

# import IPython
# IPython.embed()
# print('after')
# print(df.loc[df.index[0:10], ['a', 'b', 'm', 'n', 'r']])
seit.filter('r < 0')
seit.filter('rho_a < 15 or rho_a > 35')
seit.filter('rpha < - 40 or rpha > 3')
seit.filter('rphadiff < -5 or rphadiff > 5')
seit.filter('k > 400')
seit.filter('rho_a < 0')
seit.filter('a == 12 or b == 12 or m == 12 or n == 12')
seit.filter('a == 13 or b == 13 or m == 13 or n == 13')

import IPython
IPython.embed()
seit.print_data_journal()
seit.filter_incomplete_spectra(flimit=300, percAccept=85)
seit.print_data_journal()
seit.print_log()

import crtomo
grid = crtomo.crt_grid('data/elem.dat', 'data/elec.dat')

seitinv = seit.export_to_crtomo_seit_manager(grid)
seitinv.crtomo_cfg['robust_inv'] = 'F'
seitinv.crtomo_cfg['mag_abs'] = 0.012
seitinv.crtomo_cfg['mag_rel'] = 0.5
seitinv.crtomo_cfg['hom_bg'] = 'T'
seitinv.crtomo_cfg['d2_5'] = 0
seitinv.crtomo_cfg['fic_sink'] = 'T'
seitinv.crtomo_cfg['fic_sink_node'] = 6467
seitinv.apply_crtomo_cfg()

# save to a sip-directory
seitinv.save_to_eitdir('sipdir', norrec='nor')

###############################################################################
# now run the inversion



###############################################################################
# Now plot the results
