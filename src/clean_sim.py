#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Clean a simulation directory of all modeling/inversion files
"""
import numpy as np
import os
import glob


def main():
    rm_list = []

    required_files_inversion = (
        'exe/crtomo.cfg',
        'grid/elem.dat',
        'grid/elec.dat',
        'mod/volt.dat')
    clean_inv = np.all([os.path.isfile(x) for x in required_files_inversion])
    if clean_inv:
        rm_list += glob.glob('inv/*')
        rm_list += [
            'exe/error.dat',
            'exe/crtomo.pid',
            'exe/variogram.gnu',
            'exe/inv.elecpositions',
            'exe/inv.gstat',
            'exe/inv.lastmod',
            'exe/inv.lastmod_rho',
            'exe/inv.mynoise_pha',
            'exe/inv.mynoise_rho',
            'exe/inv.mynoise_voltages',
            'exe/tmp.kfak',
            'overview.png',
        ]

    required_files_modelling = (
        'exe/crmod.cfg',
        'grid/elem.dat',
        'grid/elec.dat',
        'config/config.dat',
        'rho/rho.dat'
    )
    clean_mod = np.all([os.path.isfile(x) for x in required_files_modelling])
    if clean_mod:
        rm_list += glob.glob('mod/sens/*')
        rm_list += glob.glob('mod/pot/*')
        rm_list += ['mod/volt.dat', ]
        rm_list += ['exe/crmod.pid', ]

    for filename in rm_list:
        if os.path.isfile(filename):
            # print('Removing file {0}'.format(filename))
            os.remove(filename)


if __name__ == '__main__':
    main()
