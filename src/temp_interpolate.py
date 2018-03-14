#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 08:25:32 2018

@author: budler
"""
import numpy as np
from optparse import OptionParser
import crtomo.grid as CRGrid
from scipy.interpolate import griddata
import crtomo.plotManager as CRPlot
import matplotlib.pyplot as plt


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


def interpolate2d(data, grid):
    xdim = grid.get_minmax()[0]
    xmin = []
    xmax = []
    xmin.extend([xdim[0]] * len(data))
    xmax.extend([xdim[1]] * len(data))
    coordinate = np.column_stack((xmin, data[:, 0]))
    coordinate = np.append(coordinate, np.column_stack((xmax, data[:, 0])))
    coordinate = np.reshape(coordinate, (-1, 2))
    temp = np.hstack((data[:, 1], data[:, 1]))
    centroids = grid.get_element_centroids()
    profile = griddata(coordinate,
                       temp,
                       centroids,
                       method='linear',
                       )
    return profile


def interpolate3d(data, grid):
    print('Not implemented yet!')
    exit()


def save_tprofile(data, filename, grid):
    coords = grid.get_element_centroids()
    content = np.column_stack((coords, data))
    with open(filename, 'w') as fid:
        fid.write('{0}\n'.format(data.shape[0]))
    with open(filename, 'ab') as fid:
        np.savetxt(fid, np.array(content), fmt='%f')


def plot(data, grid):
    f, ax = plt.subplots(1)
    print(data)
    plotman = CRPlot.plotManager(grid=grid)
    cid = plotman.parman.add_data(data)
    plotman.plot_elements_to_ax(
            cid=cid,
            ax=ax,
            cblabel='Temperature',
            plot_colorbar=True,
            cmap_name='jet',
            )
    f.tight_layout()
    f.savefig('temp/tprofile.png', dpi=300)


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
    if np.any(np.isnan(temp)):
        print('Not enough temperature information to interpolate to the whole'
              + ' grid. Please add information to cover the grid dimension.')
    save_tprofile(temp, options.out, grid)
    plot(temp, grid)


if __name__ == '__main__':
    main()
