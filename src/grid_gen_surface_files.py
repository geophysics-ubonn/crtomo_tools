#!/usr/bin/env python
"""
Create electrodes.dat and boundaries.dat for a surface grid

Parameters
----------

nr_electrodes : int
    Number of electrodes to use
spacing : float
    Spacing between electrodes
grid_depth : float
    Depth of the grid
"""
from optparse import OptionParser

import numpy as np


def handle_cmd_options():
    parser = OptionParser()
    parser.add_option(
        '-n', "--nr_electrodes", dest="nr_electrodes",
        type="int", help="Number of electrodes to use"
    )
    parser.add_option(
        '-s', "--spacing", dest="spacing",
        type="float", help="electrode spacing",
    )
    parser.add_option(
        '-d', "--depth", dest="depth",
        type="float", help="grid depth",
    )
    parser.add_option(
        '-m', "--margin", dest="margin",
        type="float", help="margin on left/right",
        default=2.0,
    )
    (options, args) = parser.parse_args()
    return options


def main():
    options = handle_cmd_options()

    electrodes = np.array(
        [(x, 0.0) for x in np.arange(0.0, options.nr_electrodes)]
    )
    electrodes[:, 0] = electrodes[:, 0] * options.spacing

    minx = electrodes[:, 0].min()
    maxx = electrodes[:, 0].max()

    margin = options.margin
    # min/max coordinates of final grid
    minimum_x = minx - margin
    maximum_x = maxx + margin
    minimum_z = -options.depth
    maximum_z = 0

    boundary_noflow = 11
    boundary_mixed = 12

    surface_electrodes = np.hstack((
        electrodes, boundary_noflow * np.ones((electrodes.shape[0], 1))
    ))
    boundaries = np.vstack((
        (minimum_x, 0, boundary_noflow),
        surface_electrodes,
        (maximum_x, maximum_z, boundary_mixed),
    ))

    boundaries = np.vstack((
        boundaries,
        (maximum_x, minimum_z, boundary_mixed),
        (minimum_x, minimum_z, boundary_mixed),
    ))
    np.savetxt('electrodes.dat', electrodes)
    np.savetxt('boundaries.dat', boundaries, fmt='%.4f %.4f %i')


if __name__ == '__main__':
    main()
