#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Script to interpolate temperature data to a grid. Output is needed for a temp-
erature correction.

Options
-------
* -d: dimension of given temprature data (1d, 2d, 3d)
* -i: input file with temperature information (1d, 2d, 3d)
    0d: only one temperature, give only one number
    1d: temperature only depth dependent, give z coordinate and temperature
        (2 column-file)
    2d: temperature informationd ependent on x- and z-coordinate (3 columns)
* -o: output file name
* --elem: path to elem.dat of the grid
* --elec: path to elec.dat of the grid
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
    '''Read in the data from the given file (filename), check its dimension and
    for at lest two different depth values (if not 0d).
    '''
    try:
        data = np.loadtxt(filename, usecols=([0, 1, 2]))
        dimension = 2
    except:
        try:
            data = np.loadtxt(filename, usecols=([0, 1]))
            dimension = 1
        except:
            data = np.loadtxt(filename, usecols=([0]))
            dimension = 0
    if not dimension == 0:
        try:
            length = data.shape[1]
        except:
            print('Give at least 2 different depth values')
            exit()

    return dimension, data


def interpolate0d(data, grid):
    '''Interpolate the given data value on the given grid.
    '''
    profile = []
    profile.extend([data] * len(grid.elements))
    return np.array(profile)


def interpolate1d(data, grid):
    '''Interpolate the given temperature-depth data to the given grid. It is
    expected, that the surface in the grid is horizontal.
    '''
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


def interpolate2d(data, grid):
    '''Interpolate the 2d given temperature information on the given grid.
    Not implemented yet.
    '''
    print('Not implemented yet!')
    exit()


def save_tprofile(data, filename, grid):
    '''Save the interpolated temperature information to file. Grid is needed
    for the coordinate-information.
    '''
    coords = grid.get_element_centroids()
    content = np.column_stack((coords, data))
    with open(filename, 'w') as fid:
        fid.write('{0}\n'.format(data.shape[0]))
    with open(filename, 'ab') as fid:
        np.savetxt(fid, np.array(content), fmt='%f')


def plot(data, grid):
    '''Plot the interpolated tempreature data on the grid to temp/tprofile.png.
    '''
    f, ax = plt.subplots(1)
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
    '''Function to interpolate temperature data from file to a grid, read from
    file, and write and plot the interpolated data to file.
    '''
    options = handle_options()
    dimension, data = read_data(options.input)
    grid = CRGrid.crt_grid(options.elem,
                           options.elec)
    if dimension == 0:
        print('Using 0d temperature information.')
        temp = interpolate0d(data[0], grid)
    elif dimension == 1:
        print('Using 1d temperature information.')
        temp = interpolate1d(data, grid)
    elif dimension == 2:
        print('Using 2d temperature information.')
        temp = interpolate2d(data, grid)
    save_tprofile(temp, options.out, grid)
    if np.any(np.isnan(temp)):
        print('Not enough temperature information to interpolate to the whole'
              + ' grid. Please add information to cover the grid dimension.')
        exit()
    plot(temp, grid)


if __name__ == '__main__':
    main()
