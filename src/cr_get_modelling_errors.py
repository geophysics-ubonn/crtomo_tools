#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot deviation between CRMod-derived apparent resistivities and analytical
resistivity vs K-factors. Also save the relative deviations to a .dat file.

dev = (R_mod * K - rho0) / rho0

* R_mod - modelled resistance value (using CRMod)
* K - geometric factor over a homogeneous half-space, computed using the
  analytical formula
* rho0: resistivity of homogeneous half-space

Output files
------------

OUTPUT.png: plot K vs. rel. modelling errors

OUTPUT.dat: Nx2 array, first column: analytical K factors,
                       second column: rel. modelling errors

Usage example
-------------

cr_get_modelling_errors.py --elem grid1/elem.dat --elec grid1/elec.dat\
        --config grid1/config.dat -o grid1_modelling_error.png

"""
from optparse import OptionParser

import numpy as np

from crtomo.mpl_setup import *
import crtomo.grid as CRGrid
import crtomo.tdManager as tdManager


def handle_cmd_options():
    parser = OptionParser()
    parser.add_option(
        "-e", "--elem",
        dest="elem_file",
        type="string",
        help="elem.dat file (default: elem.dat)",
        default="elem.dat"
    )
    parser.add_option(
        "-t", "--elec",
        dest="elec_file",
        type="string",
        help="elec.dat file (default: elec.dat)",
        default="elec.dat"
    )
    parser.add_option(
        "--config", dest="config_file", type="string",
        help="config.dat file (default: config.dat)",
        default="config.dat"
    )

    parser.add_option(
        "-o", "--output",
        dest="output_file",
        type="string",
        help="Output file (plot) (default: modelling_error.png)",
        default='modelling_error.png'
    )

    (options, args) = parser.parse_args()
    return options


def compute_K_factors(options):
    grid = CRGrid.crt_grid()
    grid.load_grid(options.elem_file, options.elec_file)

    configs = np.loadtxt(options.config_file, skiprows=1)
    A = np.round(configs[:, 0] / 1e4).astype(int) - 1
    B = np.mod(configs[:, 0], 1e4).astype(int) - 1

    M = np.round(configs[:, 1] / 1e4).astype(int) - 1
    N = np.mod(configs[:, 1], 1e4).astype(int) - 1

    # we assume that electrodes are positioned on the surface
    # therefore we only need X coordinates
    Exz = grid.get_electrode_positions()
    Ex = Exz[:, 0]

    # make sure Ez are all the same
    if np.any(Exz[:, 1] - Exz[0, 1] != 0):
        print('Are you sure that the grid approximates a halfspace?')
        exit()

    # get coordinates
    Xa = Ex[A]
    Xb = Ex[B]
    Xm = Ex[M]
    Xn = Ex[N]

    r_am = np.abs(Xa - Xm)
    r_bm = np.abs(Xb - Xm)
    r_an = np.abs(Xa - Xn)
    r_bn = np.abs(Xb - Xn)

    geom = (1 / r_am - 1 / r_bm - 1 / r_an + 1 / r_bn)
    K = (2 * np.pi) * (geom) ** (-1)

    return K, np.vstack((A + 1, B + 1, M + 1, N + 1)).T


def get_R_mod(options, rho0):
    """Compute synthetic measurements over a homogeneous half-space
    """
    tomodir = tdManager.tdMan(
        elem_file=options.elem_file,
        elec_file=options.elec_file,
        config_file=options.config_file,
    )

    # set model
    tomodir.add_homogeneous_model(magnitude=rho0)

    # only interested in magnitudes
    Z = tomodir.measurements()[:, 0]

    return Z


def plot_and_save_deviations(rho0, rho_mod, Kfactors, filename, configs):
    print('plotting')
    # multiply by 100 to get percentage
    deviation = np.abs((rho_mod - rho0) / rho0) * 100

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.loglog(np.abs(Kfactors), deviation, '.')
    ax.set_xlabel('K [m]')
    ax.set_ylabel(r'$\frac{K \cdot R^{mod} - \rho_0}{\rho_0}~[\%]$')
    fig.tight_layout()
    fig.savefig(filename, dpi=300)

    # save
    filename_plot = filename[:-3] + 'dat'

    Kfactors = Kfactors[:, np.newaxis]
    deviation = deviation[:, np.newaxis]

    K_dev = np.hstack((
        configs,
        Kfactors,
        deviation
    ))
    with open(filename_plot, 'wb') as fid:
        fid.write(bytes('#A   B   M   N    K       Dev[%]\n', 'utf-8'))
        np.savetxt(
            fid,
            K_dev,
            fmt='%.3i %.3i %.3i %.3i %.4f %.6f'
        )

    indices_sorted = np.argsort(deviation[:, 0])

    K_dev_sorted = np.hstack(
        (
            configs[indices_sorted, :],
            Kfactors[indices_sorted, :],
            deviation[indices_sorted, :]
        )
    )
    with open(filename_plot[:-4] + '_sorted.dat', 'wb') as fid:
        fid.write(bytes('#A   B   M   N    K       Dev[%]\n', 'utf-8'))
        np.savetxt(
            fid,
            K_dev_sorted,
            fmt='%.3i %.3i %.3i %.3i %.4f %.6f'
        )


def main():
    # homogeneous background resistivity [Ohm m]
    rho0 = 100
    options = handle_cmd_options()
    Kfactors, configs = compute_K_factors(options)
    Rmod = get_R_mod(options, rho0)
    rho_mod = np.abs(Kfactors) * Rmod
    plot_and_save_deviations(
        rho0,
        rho_mod,
        Kfactors,
        options.output_file,
        configs
    )


if __name__ == '__main__':
    main()
