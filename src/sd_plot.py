#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
END DOCUMENTATION
'''
import numpy as np
import os
from optparse import OptionParser
import crtomo.plotManager as CRPlot
import crtomo.grid as CRGrid
import matplotlib.pyplot as plt
import matplotlib
import math
import edf.main.units as units
import crtomo.mpl as mpl_style
import crtomo.td_plot as td

def handle_options():
    '''Handle options.
    '''
    parser = OptionParser()
    parser.set_defaults(cmaglin=False)
    parser.set_defaults(single=False)
    parser.set_defaults(alpha_cov=False)
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
    parser.add_option('-c',
                      '--column',
                      dest='column',
                      help='column to plot of input file',
                      type='int',
                      default=2,
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
                      dest='title',
                      type='string',
                      help='Global override for title',
                      default=None,
                      )
    parser.add_option("--alpha_cov",
                      action="store_true",
                      dest="alpha_cov",
                      help="use coverage for transparency",
                      )
#    parser.add_option('--x_nr',
#                      dest='x_nr',
#                      help='Number of X-axis tiks',
#                      type=int,
#                      metavar='INT',
#                      default=None,
#                      )
#    parser.add_option('--y_nr',
#                      dest='y_nr',
#                      help='Number of Y-axis tiks',
#                      type=int, metavar='INT',
#                      default=None,
#                      )
    # options for subplots
    parser.add_option('--cov_cbtiks',
                      dest='cov_cbtiks',
                      help="Number of CB tiks for coverage",
                      type=int,
                      metavar="INT",
                      default=3,
                      )
    parser.add_option('--mag_cbtiks',
                      dest='mag_cbtiks',
                      help="Number of CB tiks for magnitude",
                      type=int,
                      metavar="INT",
                      default=3,
                      )
    parser.add_option("--cmaglin",
                      action="store_true",
                      dest="cmaglin",
                      help="linear colorbar for magnitude",
                      )
    parser.add_option("--single",
                      action="store_true",
                      dest="single",
                      help="plot only magnitude",
                      )
    parser.add_option('--pha_cbtiks',
                      dest='pha_cbtiks',
                      help="Number of CB tiks for phase",
                      type=int,
                      metavar="INT",
                      default=3,
                      )
    parser.add_option('--real_cbtiks',
                      dest='real_cbtiks',
                      help="Number of CB tiks for real part",
                      type=int,
                      metavar="INT",
                      default=3,
                      )
    parser.add_option('--imag_cbtiks',
                      dest='imag_cbtiks',
                      help="Number of CB tiks for imag part",
                      type=int,
                      metavar="INT",
                      default=3,
                      )

    (options, args) = parser.parse_args()
    return options


def list_plottypes(invmod):
    plots = ['mag', 'cov']
    dirs = os.listdir(invmod)
    cfg = np.genfromtxt(invmod + dirs[0] + '/exe/crtomo.cfg',
                        skip_header=15,
                        dtype='str',
                        usecols=([0]))
    if cfg[0] == 'F':
        plots.append('pha')
        plots.append('real')
        plots.append('imag')
    if cfg[2] == 'T':
        plots.append('pha_FPI')
        plots.append('real_FPI')
        plots.append('imag_FPI')
    return plots


def list_frequencies(invmod):
    dirlist = os.listdir(invmod)
    dirlist.sort()
    frequencies = [dir[3:] for dir in dirlist]
    return dirlist, frequencies


def get_plotdetails(datatype):
    if datatype == 'mag':
        plt_routine = 'plot_mag'
        title = 'Magnitude'
    if datatype == 'cov':
        plt_routine = 'plot_cov'
        title = 'Coverage'
    if datatype == 'pha':
        plt_routine = 'plot_pha'
        title = 'Phase'
    if datatype == 'real':
        plt_routine = 'plot_real'
        title = 'Real Part'
    if datatype == 'imag':
        plt_routine = 'plot_imag'
        title = 'Imaginary Part'
    if datatype == 'pha_FPI':
        plt_routine = 'plot_pha'
        title = 'FPI Phase'
    if datatype == 'real_FPI':
        plt_routine = 'plot_real'
        title = 'FPI Real Part'
    if datatype == 'imag_FPI':
        plt_routine = 'plot_imag'
        title = 'FPI Imaginary Part'
    is_fpi = False
    if datatype[-3:] == 'FPI':
        is_fpi = True
    return plt_routine, is_fpi, title


def main():
    options = handle_options()
    matplotlib.style.use('default')
    mpl_style.general_settings()
    plotlist = list_plottypes('./')
    dirs, freqs = list_frequencies('./')# load grid
    grid = CRGrid.crt_grid(dirs[0] + '/grid/elem.dat',
                           dirs[0] + '/grid/elec.dat')
    plotman = CRPlot.plotManager(grid=grid)
    for plot in plotlist:
        routine, fpi, title = get_plotdetails(plot)
        # create figure
        f, ax = plt.subplots(len(dirs)//4, 4, figsize=(14, 2 * len(dirs)//4))
        if options.title is not None:
            plt.suptitle(options.title, fontsize=18)
        else:
            plt.suptitle(title, fontsize=18)
        
        for direc, frequ in zip(dirs, freqs):
            os.chdir(direc)
            it = td.read_iter(fpi)
            print(it)
            ###
            # td.read_datafiles()
            # cid = plotman.parman.add_data(mag)
            os.chdir('..')
#            td.plot...
#            add to summary plot
    
    


if __name__ == '__main__':
    main()