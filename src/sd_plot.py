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
    parser.add_option('--cbtiks',
                      dest='cbtiks',
                      help="Number of CB tiks",
                      type=int,
                      metavar="INT",
                      default=3,
                      )
    parser.add_option("--cmaglin",
                      action="store_true",
                      dest="cmaglin",
                      help="linear colorbar for magnitude",
                      )
    parser.add_option('-t',
                      '--type',
                      dest='type',
                      help='what type of data should be plotted',
                      type='str',
                      default='mag',
                      )

    (options, args) = parser.parse_args()
    return options


def get_plotoptions(typ, cmaglin):
    options = []
    if typ == 'mag':
        options.append('inv.lastmod')  # file with last iteration
        options.append('mag')  # file ending of datafile
        options.append('Magnitude')  # title of plot
        if cmaglin:
            options.append('rho')  # key for cb-unit
        else:
            options.append('log_rho')
        options.append('viridis')  # cb
    elif typ == 'pha':
        options.append('inv.lastmod')
        options.append('pha')
        options.append('Phase')
        options.append('phi')
        options.append('plasma')
#    elif typ == 'real':
#        options.append('inv.lastmod')
#        options.append('rho00.mag')
#        options.append('Real Part')
#        options.append('log_real')  # insert option cmaglin rho
#        options.append('viridis_r')
#    elif typ == 'imag':
#        options.append('inv.lastmod')
#        options.append('rho00.mag')
#        options.append('Imaginary Part')
#        options.append('log_imag')  # insert option cmaglin rho
#        options.append('plasma_r')
    else:
        print("This data format isn't specified. Please select 'mag' or 'pha'")
        exit()
    return options


def load_grid(td, alpha_cov):
    '''Load grid and calculate alpha values from the coverage/2.5.
    '''
    grid = CRGrid.crt_grid(td + '/grid/elem.dat',
                           td + '/grid/elec.dat')
    plotman = CRPlot.plotManager(grid=grid)

    name = td + '/inv/coverage.mag'
    content = np.genfromtxt(name, skip_header=1, skip_footer=1, usecols=([2]))
    abscov = np.abs(content)
    if alpha_cov:
        normcov = np.divide(abscov, 2.5)
        normcov[np.where(normcov > 1)] = 1
        mask = np.subtract(1, normcov)
        alpha = plotman.parman.add_data(mask)
    else:
        alpha = plotman.parman.add_data(np.ones(len(abscov)))
    return alpha, plotman


def get_data(direc, options, column, plotman):
    os.chdir(direc)
    # get iteration
    linestring = open('exe/' + options[0], 'r').readline().strip()
    linestring = linestring.replace('\n', '')
    linestring = linestring.replace('../', '')
    linestring = linestring.replace('mag', '')
    # open data file
    name = linestring + options[1]

    if options[1] == 'mag':
        try:
            content = np.loadtxt(name, skiprows=1, usecols=([column]))
        except:
            raise ValueError('Given column to open does not exist.')
    if options[1] == 'pha':
        try:
            content = np.loadtxt(name, skiprows=1, usecols=([2]))
        except:
            raise ValueError('No phase data to open.')
    # add data to plotman
    if options[3] == 'logrho':
        cid = plotman.parman.add_data(np.power(10, content))
    else:
        cid = plotman.parman.add_data(content)
    os.chdir('..')

    return cid


def plot_data(plotman, ax, cid, alpha, options, xunit, title,
              xmin, xmax, zmin, zmax, cbtiks):
    # handle options
    cblabel = units.get_label(options[3])
    zlabel = 'z [' + xunit + ']'
    xlabel = 'x [' + xunit + ']'
    cm = options[4]
    # plot
    fig, ax, cnorm, cmap, cb = plotman.plot_elements_to_ax(cid=cid,
                                                           cid_alpha=alpha,
                                                           ax=ax,
                                                           xmin=xmin,
                                                           xmax=xmax,
                                                           zmin=zmin,
                                                           zmax=zmax,
                                                           cblabel=cblabel,
                                                           cbnrticks=cbtiks,
                                                           title=title,
                                                           zlabel=zlabel,
                                                           xlabel=xlabel,
                                                           plot_colorbar=True,
                                                           cmap_name=cm,
                                                           )
    return fig, ax, cnorm, cmap, cb


def main():
    # options
    options = handle_options()
    matplotlib.style.use('default')
    mpl_style.general_settings()
    opt = get_plotoptions(options.type, options.cmaglin)

    # directories to plot
    os.chdir('invmod')
    freq_dirs = os.listdir('.')
    freq_dirs.sort()

    # create figure
    fig, axs = plt.subplots(math.ceil(len(freq_dirs)/4),
                            ncols=4,
                            figsize=(15, 1.8 * math.ceil(len(freq_dirs)/4)))
    plt.suptitle(opt[2], fontsize=18)
    plt.subplots_adjust(wspace=1, top=2.8)
    i = 0
    j = 0

    # plot each subplot
    for subplot in np.arange(4 * math.ceil(len(freq_dirs)/4)):
        try:
            #for direc in freq_dirs:
            alpha, plotman = load_grid(freq_dirs[subplot],
                                       options.alpha_cov)
            cid = get_data(freq_dirs[subplot],
                           opt,
                           options.column,
                           plotman)
            plot_data(plotman, axs[i//4, j], cid, alpha, opt, options.unit,
                      freq_dirs[subplot][3:] + ' Hz',
                      options.xmin, options.xmax, options.zmin, options.zmax,
                      options.cbtiks)
        except:
            # no subplot needed
            axs[i//4, j].axis('off')
        i = i + 1
        j = j + 1
        if j == 4:
            j = 0

    os.chdir('..')
    fig.tight_layout()
    fig.savefig('sd_' + opt[1] + '.png', dpi=300)


if __name__ == '__main__':
    main()
