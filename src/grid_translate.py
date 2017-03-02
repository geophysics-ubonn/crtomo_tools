#!/usr/bin/env python
# *-* coding:utf-8 *-*
"""Translate a given grid using the user-supplied offsets dx, dz

Examples
--------

    grid_translate.py -e original/elem.dat -z 600 -o elem.dat

"""
from optparse import OptionParser
import numpy as np
import crtomo.grid as CRGrid


def handle_cmd_options():
    parser = OptionParser()
    parser.add_option('-e', "--elem", dest="elem_file", type="string",
                      help="elem.dat file (default: elem.dat)",
                      default="elem.dat")

    # parser.add_option("-x", "--center_x", dest="center_x", type="float",
    #                   help="Center around which to rotate (X-coordiante)",
    #                   default=0.0)
    # parser.add_option("-y", "--center_y", dest="center_y", type="float",
    #                   help="Center around which to rotate (Y-coordiante)",
    #                   default=0.0)

    parser.add_option("-x", "--dx", dest="dx", type="float",
                      help="Offset on x-axis (default: 0)",
                      default=0.0)
    parser.add_option("-z", "--dz", dest="dz", type="float",
                      help="Offset on z-axis (default: 0)",
                      default=0.0)
    parser.add_option("-o", "--output", dest="output",
                      help="Output file (default: elem_trans.dat)",
                      metavar="FILE", default="elem_rot.dat")

    (options, args) = parser.parse_args()
    return options


def translate_nodes(xy, dx, dz):
    offset = np.array((dx, dz))
    trans_xy = []

    for vector in xy:
        trans_xy.append(vector + offset)
    trans_xy_array = np.array(trans_xy)
    return trans_xy_array


def main():
    options = handle_cmd_options()
    # put in dummy center coordinates
    options.center_x = 0.0
    options.center_y = 0.0

    grid = CRGrid.crt_grid()
    grid.load_elem_file(options.elem_file)
    rotated_nodes = translate_nodes(
        grid.nodes['raw'][:, 1:3],
        options.dx,
        options.dz
    )
    grid.nodes['raw'][:, 1:3] = rotated_nodes
    grid.save_elem_file(options.output)


if __name__ == '__main__':
    main()
