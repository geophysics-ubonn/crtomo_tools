#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot a nice histogram of element angles to plot_element_angles.jpg. The
script expects an elem.dat file in the present working directory.

Examples
--------

    >>> ls -1
    elec.dat
    elem.dat
    >>> grid_plot_elem_angles
    >>> ls -1
    elec.dat
    elem.dat
    plot_element_angles.jpg


"""
import os

import crtomo.grid as CRGrid


def main():
    if not os.path.isfile('elem.dat'):
        raise Exception('elem.dat not found!')

    grid = CRGrid.crt_grid()
    grid.load_elem_file('elem.dat')
    fig, ax = grid.analyze_internal_angles(return_plot=True)
    fig.savefig('plot_element_angles.jpg', dpi=300)
