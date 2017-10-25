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


# characterize tomodir
# list files


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
    # list files
    # read files
    # plot content
    # plot overview


if __name__ == '__main__':
    main()
