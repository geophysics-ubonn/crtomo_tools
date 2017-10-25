#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Has to be run in tomodir
END DOCUMENTATION
'''
import numpy as np
import os
from optparse import OptionParser
import crtomo.plotManager as CRPlot
import crtomo.grid as CRGrid
import matplotlib.pyplot as plt
import math


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


def handle_cov_options():
    '''Handle options, which are the same for all subplots or for the overview.
    '''
    parser = OptionParser()
    parser.add_option('--cov_cbtiks',
                      dest='cbtiks',
                      help="Number of CB tiks for coverage",
                      type=int,
                      metavar="INT",
                      default=3,
                      )

    (options, args) = parser.parse_args()
    return options


def handle_mag_options():
    '''Handle options, which are the same for all subplots or for the overview.
    '''
    parser = OptionParser()
    parser.add_option('--mag_cbtiks',
                      dest='cbtiks',
                      help="Number of CB tiks for magnitude",
                      type=int,
                      metavar="INT",
                      default=3,
                      )

    (options, args) = parser.parse_args()
    return options


def handle_pha_options():
    '''Handle options, which are the same for all subplots or for the overview.
    '''
    parser = OptionParser()
    parser.add_option('--pha_cbtiks',
                      dest='cbtiks',
                      help="Number of CB tiks for phase",
                      type=int,
                      metavar="INT",
                      default=3,
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


def read_datafiles(files, dtype):
    pha = []
    pha_fpi = []
    for filename, filetype in zip(files, dtype):
        if filetype == 'cov':
            cov = load_cov(filename)
        elif filetype == 'mag':
            mag = load_rho(filename)
        elif filetype == 'pha':
            pha = load_rho(filename)
        elif filetype == 'pha_fpi':
            pha_fpi = load_rho(filename)

    return cov, mag, pha, pha_fpi


def load_cov(name):
    content = np.genfromtxt(name, skip_header=1, skip_footer=1, usecols=([2]))

    return content


def load_rho(name):
    content = np.loadtxt(name, skiprows=1, usecols=([2]))

    return content


def calc_complex(mag, pha):
    complx = [10 ** m * math.e ** (1j * p / 1e3) for m, p in zip(mag, pha)]
    real = [math.log10((1 / c).real) for c in complx]
    imag = [((1 / c).imag) for c in complx]  # ##############log entfernt
    return real, imag


def plot_real(cid, ax, plotman, title):
    # load options
    options = handle_mag_options()
    # handle options
    xmin = b_options.xmin
    xmax = b_options.xmax
    zmin = b_options.zmin
    zmax = b_options.zmax
    cblabel = r"$\log_{10}(\sigma'$ [S/m]"
    zlabel = 'z [' + b_options.unit + ']'
    xlabel = 'x [' + b_options.unit + ']'
    cbtiks = options.cbtiks
    # plot
    fig, ax, cnorm, cmap, cb = plotman.plot_elements_to_ax(cid=cid,
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
                                                           )


def plot_imag(cid, ax, plotman, title):
    # load options
    options = handle_mag_options()
    # handle options
    xmin = b_options.xmin
    xmax = b_options.xmax
    zmin = b_options.zmin
    zmax = b_options.zmax
    cblabel = r"$\log_{10}(\sigma''$ [S/m]"
    zlabel = 'z [' + b_options.unit + ']'
    xlabel = 'x [' + b_options.unit + ']'
    cbtiks = options.cbtiks
    # plot
    fig, ax, cnorm, cmap, cb = plotman.plot_elements_to_ax(cid=cid,
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
                                                           )


def plot_mag(cid, ax, plotman, title):
    # load options
    options = handle_mag_options()
    # handle options
    xmin = b_options.xmin
    xmax = b_options.xmax
    zmin = b_options.zmin
    zmax = b_options.zmax
    cblabel = r'$|\rho|\,[\Omega\mbox{m}]$'
    zlabel = 'z [' + b_options.unit + ']'
    xlabel = 'x [' + b_options.unit + ']'
    cbtiks = options.cbtiks
    # plot
    fig, ax, cnorm, cmap, cb = plotman.plot_elements_to_ax(cid=cid,
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
                                                           )


def plot_pha(cid, ax, plotman, title):
    # load options
    options = handle_pha_options()
    # handle options
    xmin = b_options.xmin
    xmax = b_options.xmax
    zmin = b_options.zmin
    zmax = b_options.zmax
    cblabel = r'$\phi$ [mrad]'
    zlabel = 'z [' + b_options.unit + ']'
    xlabel = 'x [' + b_options.unit + ']'
    cbtiks = options.cbtiks
    # plot
    fig, ax, cnorm, cmap, cb = plotman.plot_elements_to_ax(cid=cid,
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
                                                           )


def plot_cov(cid, ax, plotman, title):
    # load options
    options = handle_cov_options()
    # handle options
    xmin = b_options.xmin
    xmax = b_options.xmax
    zmin = b_options.zmin
    zmax = b_options.zmax
    cblabel = 'L1 Coverage'
    zlabel = 'z [' + b_options.unit + ']'
    xlabel = 'x [' + b_options.unit + ']'
    cbtiks = options.cbtiks
    # plot
    fig, ax, cnorm, cmap, cb = plotman.plot_elements_to_ax(cid=cid,
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
                                                           )


def plot_tomodir(cov, mag, pha, pha_fpi):
    # load grid
    grid = CRGrid.crt_grid('grid/elem.dat',
                           'grid/elec.dat')
    plotman = CRPlot.plotManager(grid=grid)
    # create figure
    f, ax = plt.subplots(2, 4, figsize=(14, 4))
    # plot coverage
    cid = plotman.parman.add_data(cov)
    plot_cov(cid, ax[1, 0], plotman, 'Coverage')
    # plot magnitue
    cid = plotman.parman.add_data(mag)
    plot_mag(cid, ax[0, 0], plotman, 'Magnitude')
    # plot phase, real, imag
    if pha != []:
        cid = plotman.parman.add_data(pha)
        plot_pha(cid, ax[0, 1], plotman, 'Phase')
        [real, imag] = calc_complex(mag, pha)
        cid_re = plotman.parman.add_data(real)
        cid_im = plotman.parman.add_data(imag)
        plot_real(cid_re, ax[0, 2], plotman, 'Real Part')
        plot_imag(cid_im, ax[0, 3], plotman, 'Imaginary Part')
    # plot fpi phase, real, imag
    if pha_fpi != []:
        cid = plotman.parman.add_data(pha_fpi)
        plot_pha(cid, ax[1, 1], plotman, 'FPI Phase')
        [real, imag] = calc_complex(mag, pha_fpi)
        cid_fre = plotman.parman.add_data(real)
        cid_fim = plotman.parman.add_data(imag)
        plot_real(cid_fre, ax[1, 2], plotman, 'FPI Real Part')
        plot_imag(cid_fim, ax[1, 3], plotman, 'FPI Imaginary Part')
    f.tight_layout()
    f.savefig('td_overview.png', dpi=300)


def main():
    global b_options
    b_options = handle_base_options()
    [datafiles, filetype] = list_datafiles()
    [cov, mag, pha, pha_fpi] = read_datafiles(datafiles, filetype)
    plot_tomodir(cov, mag, pha, pha_fpi)


if __name__ == '__main__':
    main()
