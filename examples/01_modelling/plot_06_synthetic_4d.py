#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generating a 4D synthetic data set with noise.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A 2D space, time and frequency data set is generated for testing purposes in
reda.
"""
###############################################################################
# imports
import os
from glob import glob

import numpy as np

import crtomo
import reda

###############################################################################
# Generate the forward models
frequencies = np.logspace(-3, 3, 5)
grid = crtomo.crt_grid(
    'data_synthetic_4d/elem.dat', 'data_synthetic_4d/elec.dat'
)

# this context manager makes sure that all output is relative to the given
# directory
with reda.CreateEnterDirectory('output_synthetic_4d'):
    for nr, anomaly_z_pos in enumerate(range(0, -10, -3)):
        outdir = 'modV_{:02}'.format(nr)
        if os.path.isdir(outdir):
            continue
        sinv = crtomo.eitMan(grid=grid, frequencies=frequencies)
        sinv.add_homogeneous_model(100, 0)
        sinv.set_area_to_single_colecole(
            18, 22, anomaly_z_pos -2.0, anomaly_z_pos,
            [100, 0.1, 0.04, 0.6]
        )
        r = sinv.plot_forward_models()
        r['rmag']['fig'].savefig('forward_rmag_{:02}.pdf'.format(nr))
        r['rpha']['fig'].savefig('forward_rpha_{:02}.pdf'.format(nr))
        for f, td in sinv.tds.items():
            td.configs.gen_dipole_dipole(skipc=0, nr_voltage_dipoles=40)
            td.configs.gen_reciprocals(append=True)
        r = sinv.measurements()

        sinv.save_measurements_to_directory(outdir)

    # plot pseudosections
    Vdirs = sorted(glob('modV*'))
    for nr, Vdir in enumerate(Vdirs):
        seit = reda.sEIT()
        seit.import_crtomo(Vdir)
        seit.compute_K_analytical(spacing=1)
        seit.plot_pseudosections(
            'r', return_fig=True
        ).savefig('ps_r_{:02}.jpg'.format(nr), dpi=300)
        seit.plot_pseudosections(
            'rho_a', return_fig=True
        ).savefig('ps_rho_a_{:02}.jpg'.format(nr), dpi=300)
        seit.plot_pseudosections(
            'rpha', return_fig=True
        ).savefig('ps_rpha_{:02}.jpg'.format(nr), dpi=300)


###############################################################################
# now generate noisy data

# this context manager makes sure that all output is relative to the given
# directory
with reda.CreateEnterDirectory('output_synthetic_4d'):
    Vdirs = sorted(glob('modV*'))
    for nr, Vdir in enumerate(Vdirs):
        seit = reda.sEIT()
        seit.import_crtomo(Vdir)
        seit.compute_K_analytical(spacing=1)
        # use different seeds for different time steps
        np.random.seed(34 + nr)
        noise = np.random.normal(loc=0, scale=1, size=seit.data.shape[0])
        r_save = seit.data['r'].values.copy()
        seit.data['r'] = r_save + noise * r_save / 8000.0 * np.log(seit.data['k'])
        seit.data['rho_a'] = seit.data['r'] * seit.data['k']
        seit.plot_pseudosections(
            'rho_a', return_fig=True
        ).savefig('noisy_ps_rho_a_{:02}.jpg'.format(nr), dpi=300)
        rpha_save = seit.data['rpha'].values.copy()
        noise_rpha = np.random.normal(loc=0, scale=1, size=seit.data.shape[0])
        seit.data['rpha'] = rpha_save + noise_rpha * rpha_save / 10.0
        seit.plot_pseudosections(
            'rpha', return_fig=True
        ).savefig('ps_rpha_{:02}.jpg'.format(nr), dpi=300)
        seit.export_to_crtomo_multi_frequency(Vdir + '_noisy')
