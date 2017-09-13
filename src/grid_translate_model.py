#!/usr/bin/env python
# *-* coding: utf-8 *-*
"""Transfer complex resistivity models between two FE grids.
"""
import os
# import shapely
from optparse import OptionParser
from shapely.geometry import Polygon
import crtomo.grid as CRGrid
import numpy as np


def handle_cmd_options():
    parser = OptionParser()
    parser.add_option("--old", dest="old_dir", type="string",
                      help="directory containing the old grid (elem.dat and " +
                      "elec.dat files), (default: old/)",
                      default="old")
    parser.add_option("--new", dest="new_dir", type="string",
                      help="directory containing the new grid (elem.dat and " +
                      "elec.dat AND rho.dat files), (default: new/)",
                      default="new")

    parser.add_option("-o", "--output", dest="output",
                      help="Output rho file (default: rho_new.dat)",
                      metavar="FILE", default="rho_new.dat")

    (options, args) = parser.parse_args()
    return options


def _load_grid(directory):
    if not os.path.isdir(directory):
        raise IOError('directory not found: {0}'.format(directory))

    for filename in (directory + os.sep + 'elem.dat',
                     directory + os.sep + 'elec.dat',):
        if not os.path.isfile(filename):
            raise IOError('filename not found: {0}'.format(filename))

    cells = []
    grid = CRGrid.crt_grid()
    grid.load_elem_file(directory + os.sep + 'elem.dat')
    grid.load_elec_file(directory + os.sep + 'elec.dat')
    gx = grid.grid['x']
    gz = grid.grid['z']
    for x, z in zip(gx, gz):
        # find all cell that touch this cell
        coords = [(xi, zi) for xi, zi in zip(x, z)]
        p2 = Polygon(coords)
        cells.append(p2)

    rhofile = directory + os.sep + 'rho.dat'
    if os.path.isfile(rhofile):
        rho = np.loadtxt(rhofile, skiprows=1)
    else:
        rho = None

    return gx, gz, cells, rho


def _almost_equal(a, b):
    """Check if the two numbers are almost equal
    """
    # arbitrary small number!!!
    threshold = 1e-9
    diff = np.abs(a - b)
    return (diff < threshold)


def main():
    options = handle_cmd_options()

    gx_old, gz_old, cells_old, rho_old = _load_grid(options.old_dir)
    gx_new, gz_new, cells_new, _ = _load_grid(options.new_dir)
    print('finished preparing grids')

    rho_new_raw = []
    for new_cell_id, cell_new in enumerate(cells_new):
        print('working on cell {0} from {1}'.format(
            new_cell_id, len(cells_new)))

        # find all cells in the old grid that touch the new one
        # check using the area
        area = cell_new.area
        touching_cells = {}
        for index, cell_old in enumerate(cells_old):
            diff = cell_new.intersection(cell_old)
            if diff.area > 0:
                # store area fraction
                touching_cells[index] = diff.area / area

            area_found = np.sum([x for x in touching_cells.items()])
            if _almost_equal(area_found, 1.0):
                # we found all cells toching the new cell
                break

        # set the new rho values for this cell

        # strategy 1: largest wins
        if touching_cells:
            largest_id = max(touching_cells, key=touching_cells.get)
            rho_new_raw.append(rho_old[largest_id, :])
        else:
            rho_new_raw.append([np.nan, np.nan])

    rho_new = np.array(rho_new_raw)
    with open(options.output, 'wb') as fid:
        fid.write(
            bytes(
                '{0}\n'.format(rho_new.shape[0]),
                'utf-8',
            )
        )
        np.savetxt(fid, rho_new)


if __name__ == '__main__':
    main()
