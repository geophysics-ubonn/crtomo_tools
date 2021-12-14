#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculating 2D correction factors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Calculation factors used to correct rhizotron tank measurements for analysis
with the 2D CRTomo branch, as described in Weigand and Kemna 2017
(Biogeosciences).

* create your measurement configurations abmn
* conduct a full measurements on water with known conductivity
* calculate synthetic forward results using 2D-CRMod using the true water
  conductivity

"""
###############################################################################
# imports
import pandas as pd
import crtomo
import reda
###############################################################################
# compute correction factors using data from this frequency
target_frequency = 70
###############################################################################
# dataset 1
seit1 = reda.sEIT()
seit1.import_eit_fzj(
    'data/CalibrationData/bnk_water_blubber_20130428_1810_34_einzel.mat',
    'data/configs.dat'
)
configurations = seit1.data.query(
    'frequency == {}'.format(target_frequency)
)[['a', 'b', 'm', 'n']].values
# extract configurations for one frequency
###############################################################################
# synthetic forward modeling
grid = crtomo.crt_grid('data/elem.dat', 'data/elec.dat')
tdman = crtomo.tdMan(grid=grid)

# add configuration added from the measurement data
tdman.configs.add_to_configs(configurations)

# true water conductivity 37.5 mS/m
tdman.add_homogeneous_model(1 / (37.5 / 1000), 0)

tdman.crmod_cfg['2D'] = '0'
tdman.crmod_cfg['fictitious_sink'] = 'T'
tdman.crmod_cfg['sink_node'] = '6467'
R_mod = tdman.measurements()

df_mod = pd.concat(
    (
        pd.DataFrame(tdman.configs.configs),
        pd.DataFrame(R_mod[:, 0])
    ),
    axis=1,
)

df_mod.columns = ('a', 'b', 'm', 'n', 'r_mod')

###############################################################################
R_meas_70 = seit1.data.query('frequency == 70')[['a', 'b', 'm', 'n', 'r']]
all_data = pd.merge(df_mod, R_meas_70, on=['a', 'b', 'm', 'n'])
all_data['correction_factor'] = all_data['r_mod'] / all_data['r']
all_data['ab'] = all_data['a'] * 1e4 + all_data['b']
all_data['mn'] = all_data['m'] * 1e4 + all_data['n']

print(all_data)
# save to file
import numpy as np
np.savetxt(
    'correction_factors.dat',
    all_data[['ab', 'mn', 'correction_factor']].values,
    fmt='%i %i %f'
)
