#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Has to be run in tomodir
END DOCUMENTATION
'''
import numpy as np
import os
from optparse import OptionParser
import scipy as sc
import crtomo.plotManager as CRPlot
import crtomo.grid as CRGrid
import pylab as plt
import matplotlib


def handle_base_options():
    '''Handle options, which are the same for all subplots or for the overview.
    '''
    parser = OptionParser()
    parser.add_option('-x',
                      '--xmin',
                      dest='xmin',
                      help='Minium X range',
                      type='float',
                      )
    parser.add_option('-X',
                      '--xmax',
                      dest='xmax',
                      help='Maximum X range',
                      type='float',
                      )
    parser.add_option('-z',
                      '--zmin',
                      dest='zmin',
                      help='Minium Z range',
                      type='float',
                      )
    parser.add_option('-Z',
                      '--zmax',
                      dest='zmax',
                      help='Maximum Z range',
                      type='float',
                      )
    parser.add_option('-u',
                      '--unit',
                      dest='unit',
                      help='Unit of length scale, typically meters (m) ' +
                      'or centimeters (cm)',
                      metavar='UNIT',
                      type='str',
                      default='m',
                      )
    parser.add_option('--no_elecs',
                      action='store_true',
                      dest='no_elecs',
                      help='Plot no electrodes (default: false)',
                      default=False,
                      )
    parser.add_option('--title',
                      dest='title_override',
                      type='string',
                      help='Global override for title',
                      default=None,
                      )
    parser.add_option('--x_nr',
                      dest='x_nr',
                      help='Number of X-axis tiks',
                      type=int,
                      metavar='INT',
                      default=None,
                      )
    parser.add_option('--y_nr',
                      dest='y_nr',
                      help='Number of Y-axis tiks',
                      type=int, metavar='INT',
                      default=None,
                      )
    parser.add_option('--aspect',
                      dest='aspect',
                      help='Aspect ratio of plot (default: 1)',
                      type=float,
                      metavar='float',
                      default=1.0,
                      )
    (options, args) = parser.parse_args()
    return options


# handle options for mag
    # linear, ctiks, cmin, cmax, cname
# handle options for phase
    # linear, ctiks, cmin, cmax, cname
# handle oprions for real
    # linear, ctiks, cmin, cmax, cname
# handle options for imag
    # linear, ctiks, cmin, cmax, cname


def read_iter(use_fpi):
    '''Return the path to the final .mag file either for the complex or the fpi
    inversion.
    '''
    filename_rhosuffix = 'exe/inv.lastmod_rho'
    filename = 'exe/inv.lastmod'
    # filename HAS to exist. Otherwise the inversion was not finished
    if(not os.path.isfile(filename)):
        print('Inversion was not finished! No last iteration found.')

    if(use_fpi is True):
        if(os.path.isfile(filename_rhosuffix)):
            filename = filename_rhosuffix

    linestring = open(filename, 'r').readline().strip()
    linestring = linestring.replace('\n', '')
    linestring = linestring.replace('../', '')
    return linestring


def list_datafiles():
    '''Get the type of the tomodir and the highest iteration to list all files,
    which will be plotted.
    '''
    # get type of the tomodir
    cfg = np.genfromtxt('exe/crtomo.cfg',
                        skip_header=15,
                        dtype='str',
                        usecols=([0]))
    is_complex = False
    if cfg[0] == 'F':
        is_complex = True
    is_fpi = False
    if cfg[2] == 'T':
        is_fpi = True
    # get the highest iteration
    it_rho = read_iter(is_fpi)
    it_phase = read_iter(False)
    # list the files
    files = ['inv/coverage.mag']
    dtype = ['cov']
    files.append(it_rho)
    dtype.append('mag')
    if is_complex:
        files.append(it_rho.replace('mag', 'pha'))
        dtype.append('pha')
    if is_fpi:
        files.append(it_phase.replace('mag', 'pha'))
        dtype.append('pha_fpi')

    return files, dtype


# read in coverage
# read in mag, pha

# calculate real
# calculate imag

# plot content corresponding to identifiers, use plots below, return fig, ax
# plot mag
# plot phase
# plot cov
# plot real
# plot imag
# plot overview
    # load grid
    # load plotmanager
    # plot content


def main():
    b_options = handle_base_options()
    [datafiles, filetype] = list_datafiles()
    # read files
    # plot content
    # plot overview


if __name__ == '__main__':
    main()
