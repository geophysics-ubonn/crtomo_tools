#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Date:         May/2015
Author:     Jannik Faßmer
Aim:           Calculate the angles in each element polygon and
                  its area.
                  Writes calculated values to file. With active --show
                  it produces two  histograms of both - angle and area - data.

END DOCUMENTATION
'''

import crlab_py.elem2 as elem2
import crlab_py.polygon as polygon
from optparse import OptionParser
import matplotlib.pyplot as plt


def handle_cmd_options():
    '''
    Get  the options from the command line.
    '''
    parser = OptionParser()
    parser.add_option("-e", "--elem", dest="elem",
                      help="Path to elem.dat file", type="string")
    parser.add_option("-f", "--out", dest="output",
                      help="Path to Output-file", type="string")
    parser.add_option("-s", "--show", action="store_true", dest="show",
                      help="plot results", default=False)
    (options, args) = parser.parse_args()
    return options, args


def get_area_angles(grid):
    '''
    Main Function.
    '''
    shape = grid.grid['x'].shape[1]
    areas = []
    angles = []
    for x, z in zip(grid.grid['x'], grid.grid['z']):
        corners = [(x[i], z[i]) for i in xrange(shape)]
        area = polygon.Polygon(corners).get_area()
        areas.append(area)
        angles_i = polygon.Polygon(corners).get_internal_angles()
        angles.append(angles_i)
    return areas, angles


def write2file(grid, areas, angles, name):
    '''
    Write calc. data to file
    '''
    with open(name, 'w+') as fid:
        fid.write('Element_nr \t Angles \t Area \n')
        c = 1
        for i, j in zip(angles, areas):
            fid.write('{0} \t {1} \t {2} \n'.format(c, i, j))
            c += 1


def plot_hist(values, x_label, name):
    '''
    Plot histogram of values
    '''
    plt.figure()
    plt.rcParams["font.size"] = 12
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Computer Modern Roman'
    plt.rcParams['text.usetex'] = True
    plt.hist(values, bins=100)
    plt.ylabel('Frequency')
    plt.xlabel(x_label)
    plt.tight_layout()
    plt.savefig(name, dpi=300)


if (__name__ == '__main__'):
    options, args = handle_cmd_options()
    grid = elem2.crt_grid()
    grid.load_elem_file(options.elem)
    areas, angles = get_area_angles(grid)
    write2file(grid, areas, angles, options.output)
    if (options.show is True):
        plot_hist(areas, 'Element area size', 'area_hist.png')
        angles2plot = []
        for i in angles:
            for j in i:
                angles2plot.append(j)
        plot_hist(angles2plot, r'Angle ' '[' '$^{\circ}$' ']',
                  'angles_hist.png')
