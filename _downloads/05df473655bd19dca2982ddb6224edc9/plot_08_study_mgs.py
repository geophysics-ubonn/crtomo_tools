#!/usr/bin/env python
# *-* coding: utf-8 *-*
"""
Using alternative regularization approaches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example shortly shows how to use the MGS regularization

"""
###################################################################
# imports
import os
import matplotlib.pylab as plt
import matplotlib.cm
import numpy as np

import reda
import crtomo
###################################################################

grid = crtomo.crt_grid.create_surface_grid(
    nr_electrodes=30, spacing=1, char_lengths=[0.3, 1, 1, 1],
    lines=[-5, -3, -1],
    internal_lines=[
        [1, -5, 1, -1],
        [5, -5, 5, -1],
        [20, -5, 20, -1],
        [24, -5, 24, -1],
    ],
)
fig, ax = grid.plot_grid()
fig.savefig('grid.jpg')

man = crtomo.tdMan(grid=grid)
# import IPython
# IPython.embed()

# xmin xmax zmin zmax
geometry_anomaly_mag = [10, 14, -5, -1]
geometry_anomaly_pha = [20, 24, -5, -1]
val_anomaly_mag = 10
val_anomaly_pha = -30
val_background_mag = 100
val_background_pha = -5

pid_mag, pid_pha = man.add_homogeneous_model(
    magnitude=val_background_mag, phase=val_background_pha
)

man.parman.modify_area(
    pid_mag,
    *geometry_anomaly_mag, val_anomaly_mag
    # xmin=10, xmax=14,
    # zmin=-5, zmax=-1,
    # value=10
)

man.parman.modify_area(
    pid_pha,
    xmin=20, xmax=24,
    zmin=-5, zmax=-1,
    value=-30
)
true_slice_horizontal_rmag = man.parman.extract_along_line(
    pid_mag, [0, -3], [30, -3], 50
)
true_slice_horizontal_rpha = man.parman.extract_along_line(
    pid_pha, [0, -3], [30, -3], 50
)

fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 5 / 2.54))
man.show_parset(
    pid_mag, ax=axes[0], plot_colorbar=True, cblabel=r'$|\rho|~[\Omega m]$',
    cmap_name='turbo',
)
man.show_parset(
    pid_pha, ax=axes[1], plot_colorbar=True,
    cmap_name='jet_r',
    cblabel=r'$\phi$ [mrad]',
)
fig.tight_layout()
fig.savefig('forward_model.jpg', dpi=300)

man.configs.gen_dipole_dipole(skipc=0)
man.configs.gen_dipole_dipole(skipc=1)
man.configs.gen_dipole_dipole(skipc=2)

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


################
# Inversion

grid_inv = crtomo.crt_grid.create_surface_grid(
    nr_electrodes=30, spacing=1, char_lengths=[0.2, 1, 1, 1],
)

with reda.CreateEnterDirectory('output_08_mgs'):
    # save data from forward modeling
    man.save_measurements('volt.dat')

    fig_all_slices, axes_all_slices = plt.subplots(
        2, 1, figsize=(16 / 2.54, 12 / 2.54), sharex=True)

    # betas = (1e-9, 0.005, 0.009, 0.01, 0.02, 0.1)
    betas = np.sort(
        np.hstack((
            # 1e-9,
            # np.linspace(0.01, 0.2, 10),
            np.linspace(0.015, 0.01522, 10),
            # 0.01,
            # 0.011,
            # 0.015,
            # 0.017,
            # 0.02,
            # 0.025,
            # 0.027
        ))
    )

    colors_rmag = matplotlib.cm.get_cmap('jet')(np.linspace(0, 1, len(betas)))
    colors_rpha = matplotlib.cm.get_cmap('jet')(np.linspace(0, 1, len(betas)))
    for nr, beta in enumerate(betas):
        outdir = 'tomodir_{:02}_beta_{}'.format(nr, beta)
        if os.path.isdir(outdir):
            tdm = crtomo.tdMan(tomodir=outdir)
        else:
            print('Skipping because not existing yet')
            continue
            tdm = crtomo.tdMan(grid=grid_inv)
            tdm.read_voltages('volt.dat')

            tdm.crtomo_cfg['robust_inv'] = 'F'
            # activate MGS
            tdm.crtomo_cfg['mswitch2'] = 5
            # set inversion settings
            # this is actually the beta value of the MGS here
            tdm.crtomo_cfg['lambda'] = beta
            # tdm.invert(cores=4)
            tdm.save_to_tomodir(outdir)

        print(tdm.a)
        if tdm.a['inversion'] is None or len(tdm.a['inversion']['rmag']) == 0:
            print('Skipping because inversion not finished yet')
            continue
        fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 5 / 2.54))
        tdm.show_parset(
            tdm.a['inversion']['rmag'][-1],
            ax=axes[0],
            cblabel=r'$|\rho|~[\Omega m]$',
            plot_colorbar=True,
            cmap_name='turbo',
        )
        tdm.show_parset(
            tdm.a['inversion']['rpha'][-1],
            ax=axes[1],
            cblabel=r'$\phi$ [mrad]',
            plot_colorbar=True,
            cmap_name='jet_r',
        )
        fig.tight_layout()
        fig.savefig('inversion_results_{:02}_beta_{}.jpg'.format(
            nr, beta), dpi=300, bbox_inches='tight')
        plt.close(fig)

        # slices
        pid_rmag = tdm.a['inversion']['rmag'][-1]
        pid_rpha = tdm.a['inversion']['rpha'][-1]

        # x y value
        slice_horizontal_rmag = tdm.parman.extract_along_line(
            pid_rmag, [0, -3], [30, -3], 50
        )
        slice_horizontal_rpha = tdm.parman.extract_along_line(
            pid_rpha, [0, -3], [30, -3], 50
        )

        fig, ax = plt.subplots(1, 1, figsize=(16 / 2.54, 8 / 2.54))
        ax.plot(
            slice_horizontal_rmag[:, 0],
            slice_horizontal_rmag[:, 2],
            '.',
            ms=5,
        )
        ax.set_xlabel(r'$|\rho|~[\Omega m]$')
        ax2 = ax.twinx()
        ax2.plot(
            slice_horizontal_rpha[:, 0],
            slice_horizontal_rpha[:, 2],
            label=r'$-\phi$ [mrad]',
            linestyle='dashed',
            color='gray',
        )
        ax2.grid(None)
        ax2.set_ylabel(r'$-\phi [mrad]$')
        ax.set_xlabel('x [m]')
        ax.set_title(r'$\beta = ${:.4f}'.format(beta))
        fig.tight_layout()
        fig.savefig('inversion_slice_hor_{:02}_beta_{}.jpg'.format(
            nr, beta), dpi=300, bbox_inches='tight')
        plt.close(fig)

        # once again for the all-in-one plot
        ax = axes_all_slices[0]
        ax.plot(
            slice_horizontal_rmag[:, 0],
            slice_horizontal_rmag[:, 2],
            '.-',
            # ms=5,
            label=r'$\beta$ = {:.5f}'.format(beta),
            color=colors_rmag[nr],
        )

        ax = axes_all_slices[1]
        ax.plot(
            slice_horizontal_rpha[:, 0],
            slice_horizontal_rpha[:, 2],
            '.-',
            label=r'$-\phi$ [mrad]',
            linestyle='dashed',
            # color='k',
            color=colors_rpha[nr],
        )
    # plot true model on top
    ax = axes_all_slices[0]
    ax.plot(
        true_slice_horizontal_rmag[:, 0],
        true_slice_horizontal_rmag[:, 2],
        '-',
        color='k',
        label='true model',
    )
    ax = axes_all_slices[1]
    ax.plot(
        true_slice_horizontal_rpha[:, 0],
        true_slice_horizontal_rpha[:, 2],
        '-',
        color='k',
    )

    axes_all_slices[0].set_ylabel(r'$|\rho|~[\Omega m]$')
    axes_all_slices[0].legend(fontsize=6, ncol=3)
    axes_all_slices[1].set_ylabel(r'$-\phi [mrad]$')
    axes_all_slices[1].set_xlabel('x [m]')

    fig_all_slices.tight_layout()
    fig_all_slices.savefig(
        'inversion_all_slices_hor.jpg', dpi=300, bbox_inches='tight')
    plt.close(fig_all_slices)
