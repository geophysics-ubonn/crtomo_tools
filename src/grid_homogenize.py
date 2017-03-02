#!/usr/bin/env python
"""
Create homogenized version of a CRTomo grid.

Examples
--------


"""
import os
import shutil
from optparse import OptionParser
import subprocess
import numpy as np

from crtomo.mpl_setup import *


class grid_container():

    def __init__(self, electrodes_file=None, boundaries_file=None,
                 char_length_file=None):

        if char_length_file is not None and os.path.isfile(char_length_file):
            self.char_length_file = char_length_file
        else:
            self.char_length_file = None

        if electrodes_file is not None:
            self.electrodes = np.loadtxt(electrodes_file)
        else:
            self.electrodes = None

        if boundaries_file is not None:
            self.boundaries = np.loadtxt(boundaries_file)
        else:
            self.boundaries = None

        self.script = None

    def save_to_dir(self, directory):

        if not os.path.isdir(directory):
            os.makedirs(directory)
        pwd = os.getcwd()
        os.chdir(directory)

        np.savetxt('electrodes.dat', self.electrodes)
        np.savetxt('boundaries.dat', self.boundaries)
        if self.char_length_file is not None:
            shutil.copy(self.char_length_file, './')

        if self.script is not None:
            with open('create_map_grid.sh', 'w') as fid:
                fid.write(self.script)
            print('Calling create_map_grid.sh')
            subprocess.call('sh create_map_grid.sh', shell=True)
        os.chdir(pwd)


def handle_cmd_options():
    parser = OptionParser()
    parser.add_option(
        '-d', "--data_dir",
        dest="data_dir",
        type="string",
        help="data dir containing the file electrodes.dat " +
        "and boundaries.dat' (default: data/)",
        default="data",
    )
    parser.add_option(
        "-o", "--output", dest="output",
        help="Output directory (default: grid_hom)",
        metavar="DIR",
        default="grid_hom",
    )
    parser.add_option(
        "--dx", dest="dx",
        help="dx",
        type="float",
        metavar="FLOAT",
        default=50,
    )
    parser.add_option(
        "--dy", dest="dy",
        help="Depth of grid below electrodes (default: 100 m)",
        type="float",
        metavar="FLOAT",
        default=100,
    )

    (options, args) = parser.parse_args()
    return options


def rotate_point(xorigin, yorigin, x, y, angle):
    """Rotate the given point by angle
    """
    rotx = (x - xorigin) * np.cos(angle) - (y - yorigin) * np.sin(angle)
    roty = (x - yorigin) * np.sin(angle) + (y - yorigin) * np.cos(angle)
    return rotx, roty


def homogenize_grid(grid_old, dx, dy):
    """
    1) fit line through electrodes
    2) rotate electrodes so that line lies in the horizontal plane
    3) translate z-coordinates so that all z-coordinates are negative
    """

    # 1 line fit
    x = grid_old.electrodes[:, 0]
    y = grid_old.electrodes[:, 1]
    sort_indices = np.argsort(x)
    x_sort = x[sort_indices]
    y_sort = y[sort_indices]
    p = np.polyfit(x_sort, y_sort, 1)

    # 2. rotate around first electrode
    offsetx = x_sort[0]
    offsety = y_sort[0]

    alpha = -np.arctan2(p[0], 1.0)  # * 180 / np.pi

    xn = []
    yn = []
    for xc, yc in zip(x, y):
        rotx, roty = rotate_point(offsetx, offsety, xc, yc, alpha)
        xn.append(rotx + offsetx)
        yn.append(roty + offsety)

    new_coordinates = np.vstack((xn, yn)).T

    # move vertically

    # this line is a horizontal line
    p_rot = np.polyfit(
        new_coordinates[:, 0],
        new_coordinates[:, 1],
        1
    )
    y_rot = np.polyval(
        p_rot,
        new_coordinates[:, 0],
    )
    ymax = y_rot[0]

    new_coordinates_trans = np.copy(new_coordinates)
    new_coordinates_trans[:, 1] -= ymax

    #
    fig, ax = plt.subplots(1, 1)
    ax.scatter(x, y, color='r', label='original')

    ax.plot(
        x,
        np.polyval(p, x),
        '-',
        label='fit',
        color='r',
    )

    ax.scatter(
        xn, yn,
        color='c',
        label='rotated',
    )

    ax.plot(
        xn,
        y_rot,
        '-',
        label='fit',
        color='c',
    )

    ax.scatter(
        new_coordinates_trans[:, 0],
        new_coordinates_trans[:, 1],
        label='homog',
    )

    # plot the line through the new coordintes
    pnew = np.polyfit(
        new_coordinates_trans[:, 0],
        new_coordinates_trans[:, 1],
        1
    )
    ax.plot(
        new_coordinates_trans[:, 0],
        np.polyval(pnew, new_coordinates_trans[:, 0]),
        '-',
        label='fit homogenized',
    )

    ax.legend(loc='best')

    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    fig.tight_layout()
    fig.savefig('output_electrodes.png', dpi=300)

    # electrodes = np.hstack((
    # boundaries
    bx = new_coordinates_trans[sort_indices, 0]
    by = new_coordinates_trans[sort_indices, 1]
    btype = [12 for i in bx]

    # add boundary
    # get deepest boundary coordinate
    y1 = by[-1] - dy
    y2 = by[0] - dy

    ymin = min(y1, y2)

    bx = np.hstack(
        (bx[0] - dx,
         bx,
         [bx[-1] + dx, bx[-1] + dx, bx[0] - dx])
    )

    by = np.hstack(
        (by[0],
         by,
         [by[-1], ymin, ymin]
         )
    )

    btype = np.hstack((12,
                       btype,
                       [11, 11, 11]))

    boundaries = np.vstack((bx, by, btype)).T

    # fig, ax = plt.subplots(1, 1)
    # ax.scatter(bx, by)
    # ax.set_aspect('equal')
    # fig.tight_layout()
    # fig.savefig('boundaries.png', dpi=300)

    grid_new = grid_container(None, None, grid_old.char_length_file)
    grid_new.boundaries = np.copy(boundaries)
    grid_new.electrodes = np.copy(new_coordinates_trans)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(bx, by, color='g', label='boundaries')
    ax.scatter(new_coordinates[:, 0], new_coordinates[:, 1], color='g',
               label='electrodes')
    ax.set_aspect('equal')
    ax.legend()
    fig.tight_layout()
    fig.savefig('output_boundaries.png', dpi=300)

    def transform_back(data):
        new_data = np.copy(data)
        new_data[:, 1] += ymax
        new_data[:, 0] -= offsetx
        new_data[:, 1] -= offsety

        tmpx = (new_data[:, 0]) * np.cos(-alpha) - (new_data[:, 1]) * np.sin(
            -alpha)
        tmpy = (new_data[:, 0]) * np.sin(-alpha) + (new_data[:, 1]) * np.cos(
            -alpha)
        tmpx += offsetx
        tmpy += offsety
        return tmpx, tmpy

    shell_script = ''
    shell_script += '#!/bin/bash\n'
    shell_script += 'cr_trig_create grid\n'

    cmd1 = ''.join((
        'grid_translate -e grid/elem.dat ',
        '--dx {0} --dz {1} -o elem_trans1.dat'.format(
            offsetx, ymax - offsety)
    ))
    cmd2 = ''.join((
        'grid_rotate -e elem_trans1.dat ',
        '-a {0} -o elem_trans1_rot1.dat'.format(-alpha * 180 / np.pi)
    ))
    cmd3 = ''.join((
        'grid_translate -e elem_trans1_rot1.dat ',
        '--dx {0} --dz {1} -o elem_trans1_rot1_trans2.dat'.format(
            offsetx, offsety)
    ))
    shell_script += cmd1 + '\n'
    shell_script += cmd2 + '\n'
    shell_script += cmd3 + '\n'

    shell_script += ''.join((
        'grid_plot_wireframe --fancy -t grid/elec.dat ',
        '-e elem_trans1_rot1_trans2.dat -o trans1_rot1_trans2.png'
    ))

    grid_new.script = shell_script

    tmpx, tmpy = transform_back(grid_new.electrodes)
    bx, by = transform_back(grid_new.boundaries[:, 0:2])

    grid_map = grid_container(char_length_file=grid_old.char_length_file)
    grid_map.electrodes = np.vstack((tmpx, tmpy)).T
    grid_map.boundaries = np.vstack((bx, by, grid_new.boundaries[:, 2])).T

    fig, ax = plt.subplots(1, 1)
    ax.scatter(tmpx, tmpy, color='r', label='new')
    ax.scatter(x, y, color='b', label='old')
    ax.scatter(bx, by, color='g', label='boundaries')
    ax.set_aspect('equal')
    ax.legend()
    fig.tight_layout()
    fig.savefig('output_map.png', dpi=300)

    return grid_new, grid_map


def main():
    options = handle_cmd_options()
    electrodes_file = options.data_dir + os.sep + 'electrodes.dat'
    boundaries_file = options.data_dir + os.sep + 'boundaries.dat'
    char_length_file = os.path.abspath(os.path.normpath(
        options.data_dir + os.sep + 'char_length.dat'
    ))

    for filename in (electrodes_file, boundaries_file):
        if not os.path.isfile(filename):
            raise Exception('Could find file: {0}'.format(filename))

    grid_old = grid_container(
        electrodes_file,
        boundaries_file,
        char_length_file,
    )
    grid_new, grid_map = homogenize_grid(grid_old, options.dx, options.dy)

    grid_new.save_to_dir(options.output)


if __name__ == '__main__':
    main()
