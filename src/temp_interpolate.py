#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 08:25:32 2018

@author: budler
"""
import numpy as np
from optparse import OptionParser
import crtomo.grid as CRGrid


def handle_options():
    parser = OptionParser()
    parser.add_option('-d',
                      '--dimension',
                      dest='dimension',
                      type='int',
                      help='dimension of given temeprature data, default=1d',
                      default=1,
                      )
    parser.add_option('-i',
                      '--input',
                      dest='input',
                      help='input file with temperature information',
                      default='temp/tdata.dat',
                      )
    parser.add_option('-o',
                      '--out',
                      dest='out',
                      help='output file',
                      default='temp/tprofile.mag',
                      )
    parser.add_option('--elem',
                      dest='elem',
                      help='path to elem.dat',
                      default='grid/elem.dat',
                      )
    parser.add_option('--elec',
                      dest='elec',
                      help='path to elec.dat',
                      default='grid/elec.dat',
                      )
    (options, args) = parser.parse_args()
    return options


def read_data(filename):
    try:
        data = np.loadtxt(filename, usecols=([0, 1, 2]))
        dimension = 3
    except:
        try:
            data = np.loadtxt(filename, usecols=([0, 1]))
            dimension = 2
        except:
            data = np.loadtxt(filename, usecols=([0]))
            dimension = 1
    return dimension, data


def interpolate1d(data, grid):
    profile = []
    profile.extend([data] * len(grid.elements))
    return np.array(profile)


def interpolate2d():
    print('Not implemented yet!')
    exit()


def interpolate3d():
    print('Not implemented yet!')
    exit()


def save_tprofile(data, filename, grid):
    coords = grid.get_element_centroids()
    content = np.column_stack((coords, data))
    with open(filename, 'w') as fid:
        fid.write('{0}\n'.format(data.shape[0]))
    with open(filename, 'ab') as fid:
        np.savetxt(fid, np.array(content), fmt='%f')


def main():
    options = handle_options()
    dimension, data = read_data(options.input)
    grid = CRGrid.crt_grid(options.elem,
                           options.elec)
    if dimension == 1:
        temp = interpolate1d(data, grid)
    elif dimension == 2:
        temp = interpolate2d(data, grid)
    elif dimension == 3:
        temp = interpolate3d(data, grid)
    save_tprofile(temp,
                  options.out,
                  grid)


if __name__ == '__main__':
    main()
