#!/usr/bin/env python
# *-* coding: utf-8 *-*
"""Rotate a given grid clockwise

Note that, at the moment, the grid is always rotate around the origin (0,0).

END DOCUMENTATION
"""
from optparse import OptionParser
import numpy as np
from crlab_py.mpl import *
from crlab_py import elem2
import math


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

    parser.add_option("-a", "--angle", dest="angle_deg", type="float",
                      help="Rotation angle (in degrees, default: 90)",
                      default=90.0)
    parser.add_option("-o", "--output", dest="output",
                      help="Output file (default: elem_rot.dat)",
                      metavar="FILE", default="elem_rot.dat")

    (options, args) = parser.parse_args()
    return options


def rotmat(alpha):
    """Rotate around z-axis
    """
    R = np.array(((np.cos(alpha), -np.sin(alpha)),
                 (np.sin(alpha), np.cos(alpha))))

    return R


def rotate_nodes(xy, cx, cy, angle_rad):
    rot_xy = []
    R = rotmat(angle_rad)

    for vector in xy:
        rot_xy.append(R.dot(vector))
    rot_xy_array = np.array(rot_xy)
    return rot_xy_array


if __name__ == '__main__':
    options = handle_cmd_options()
    # put in dummy center coordinates
    options.center_x = 0.0
    options.center_y = 0.0

    grid = elem2.crt_grid()
    grid.load_elem_file(options.elem_file)
    rotated_nodes = rotate_nodes(grid.nodes['raw'][:, 1:3],
                                 options.center_x,
                                 options.center_y,
                                 math.radians(options.angle_deg))
    grid.nodes['raw'][:, 1:3] = rotated_nodes
    grid.save_elem_file(options.output)
