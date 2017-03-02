#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compute analytical solutions of the homogeneous half-space corresponding to
electrode positions as recovered from a CRTomo FE-grid. Also, export potentials
at each node for each current injection.

TODO
----

* properly check/fix and describe output files
"""
import sys
import IPython.core.ultratb as ultratb
sys.excepthook = ultratb.VerboseTB(
    call_pdb=True,
)
import os
from optparse import OptionParser
import numpy as np
import crtomo.grid as CRGrid

import crtomo.analytical_solution as am


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
        "--config", dest="config_file",
        type="string",
        help="config.dat file (default: config.dat)",
        default="config.dat"
    )
    parser.add_option(
        "--rho", dest="rho",
        type="float",
        help="Resistivity (default: 100 Ohm m)",
        default=100.0
    )
    parser.add_option(
        "-o", "--output",
        dest="output_dir",
        type="string",
        help="Output directory (default: mod_analytical)",
        default='mod_analytical.png'
    )
    parser.add_option(
        "-p", "--potential",
        action='store_true',
        dest="compute_potentials",
        help="compute potentials (default: True)",
        default=True
    )
    parser.add_option(
        "-v", "--voltages",
        action='store_true',
        dest="compute_voltages",
        help="compute voltages (default: True)",
        default=False
    )

    (options, args) = parser.parse_args()
    return options


def load_grid(options):
    grid = CRGrid.crt_grid()
    grid.load_grid(options.elem_file, options.elec_file)
    return grid


def load_configs(options):
    configs_raw = np.loadtxt(options.config_file, skiprows=1)
    configs = np.vstack((
        np.round(configs_raw[:, 0] / 1e4),
        configs_raw[:, 0] % 1e4,
        np.round(configs_raw[:, 1] / 1e4),
        configs_raw[:, 1] % 1e4)
    ).astype(int).T
    return configs


def save_voltages(grid, voltages):
    outdir = 'mod'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    pwd = os.getcwd()
    os.chdir(outdir)
    np.savetxt('volt.dat', voltages)
    os.chdir(pwd)


def save_potentials(grid, potentials_raw):
    # potentials = [x[grid.nodes['rev_cutmck_index']] for x in potentials_raw]
    # xy = grid.nodes['sorted'][grid.nodes['rev_cutmck_index'], 1:]
    potentials = [x for x in potentials_raw]
    xy = grid.nodes['sorted'][:, 1:]

    outdir = 'pot'
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    pwd = os.getcwd()
    os.chdir(outdir)
    for nr, pot in enumerate(potentials):
        xyp = np.hstack((xy, pot[:, np.newaxis]))
        np.savetxt('pot{0}.dat'.format(nr + 1), xyp, fmt='%.8f %.8f %.8f')
    os.chdir(pwd)


def main():
    options = handle_cmd_options()
    grid = load_grid(options)
    configs = load_configs(options)
    potentials_raw = am.compute_potentials_analytical_hs(
        grid,
        configs,
        options.rho
    )
    voltages = am.compute_voltages(grid, configs, potentials_raw)
    if not os.path.isdir(options.output_dir):
        os.makedirs(options.output_dir)
    pwd = os.getcwd()
    os.chdir(options.output_dir)
    save_voltages(grid, voltages)
    save_potentials(grid, potentials_raw)
    os.chdir(pwd)


if __name__ == '__main__':
    main()
