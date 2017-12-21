#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Tool to plot inversion results of tomodir. Included data is
* magnitude
* coverage
* phase
* phase of FPI
* real part
* real part of FPI
* imaginary part
* imaginary part of FPI

But it is possible to only plot the magnitude (--single).
The script has to be run in a tomodir. Output file will be saved in tomodir.
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


def td_type():
    '''get type of the tomodir
    '''
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

    return is_complex, is_fpi


def list_datafiles():
    '''Get the type of the tomodir and the highest iteration to list all files,
    which will be plotted.
    '''
    is_cplx, is_fpi = td_type()
    # get the highest iteration
    it_rho = read_iter(is_fpi)
    it_phase = read_iter(False)
    # list the files
    files = ['inv/coverage.mag']
    dtype = ['cov']
    files.append(it_rho)
    dtype.append('mag')
    if is_cplx:
        files.append(it_rho.replace('mag', 'pha'))
        dtype.append('pha')
    if is_fpi:
        files.append(it_phase.replace('mag', 'pha'))
        dtype.append('pha_fpi')

    return files, dtype


def read_datafiles(files, dtype, column):
    '''Load the datafiles and return cov, mag, phase and fpi phase values.
    '''
    pha = []
    pha_fpi = []
    for filename, filetype in zip(files, dtype):
        if filetype == 'cov':
            cov = load_cov(filename)
        elif filetype == 'mag':
            mag = load_rho(filename, column)
        elif filetype == 'pha':
            pha = load_rho(filename, 2)
        elif filetype == 'pha_fpi':
            pha_fpi = load_rho(filename, 2)

    return cov, mag, pha, pha_fpi


def load_cov(name):
    '''Load a datafile with coverage file structure.
    '''
    content = np.genfromtxt(name, skip_header=1, skip_footer=1, usecols=([2]))

    return content


def load_rho(name, column):
    '''Load a datafile with rho structure like mag and phase
    '''
    try:
        content = np.loadtxt(name, skiprows=1, usecols=([column]))
    except:
        raise ValueError('Given column to open does not exist.')

    return content


def calc_complex(mag, pha):
    ''' Calculate real and imaginary part of the complex conductivity from
    magnitude and phase in log10.
    '''
    complx = [10 ** m * math.e ** (1j * p / 1e3) for m, p in zip(mag, pha)]
    real = [math.log10((1 / c).real) for c in complx]
    imag = []
    for c in complx:
        if ((1 / c).imag) == 0:
            imag.append(math.nan)
        else:
            i = math.log10(abs((1 / c).imag))
            imag.append(i)
    return real, imag


def plot_real(cid, ax, plotman, title, alpha,
              xmin, xmax, zmin, zmax, xunit, cbtiks, elecs):
    '''Plot real parts of the complex conductivity using the real_options.
    '''
    # handle options
    cblabel = units.get_label('log_real')
    zlabel = 'z [' + xunit + ']'
    xlabel = 'x [' + xunit + ']'
    cm = 'viridis_r'
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
                                                           no_elecs=elecs,
                                                           )
    return fig, ax, cnorm, cmap, cb


def plot_imag(cid, ax, plotman, title, alpha,
              xmin, xmax, zmin, zmax, xunit, cbtiks, elecs):
    '''Plot imag parts of the complex conductivity using the imag_options.
    '''
    # handle options
    cblabel = units.get_label('log_imag')
    zlabel = 'z [' + xunit + ']'
    xlabel = 'x [' + xunit + ']'
    cm = 'plasma_r'
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
                                                           no_elecs=elecs,
                                                           )
    return fig, ax, cnorm, cmap, cb


def plot_mag(cid, ax, plotman, title, unit, alpha,
             xmin, xmax, zmin, zmax, xunit, cbtiks, elecs):
    '''Plot magnitude of the complex resistivity using the mag_options.
    '''
    # handle options
    cblabel = units.get_label(unit)
    zlabel = 'z [' + xunit + ']'
    xlabel = 'x [' + xunit + ']'
    # plot
    fig, ax, cnorm, cmap, cb = plotman.plot_elements_to_ax(cid=cid,
                                                           ax=ax,
                                                           cid_alpha=alpha,
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
                                                           no_elecs=elecs,
                                                           )
    return fig, ax, cnorm, cmap, cb


def plot_pha(cid, ax, plotman, title, alpha,
             xmin, xmax, zmin, zmax, xunit, cbtiks, elecs):
    '''Plot phase of the complex resistivity using the pha_options.
    '''
    # handle options
    cblabel = units.get_label('phi')
    zlabel = 'z [' + xunit + ']'
    xlabel = 'x [' + xunit + ']'
    cm = 'plasma'
    # plot
    fig, ax, cnorm, cmap, cb = plotman.plot_elements_to_ax(cid=cid,
                                                           ax=ax,
                                                           cid_alpha=alpha,
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
                                                           no_elecs=elecs,
                                                           )
    return fig, ax, cnorm, cmap, cb


def plot_cov(cid, ax, plotman, title,
             xmin, xmax, zmin, zmax, xunit, cbtiks, elecs):
    '''Plot coverage of the complex resistivity using the cov_options.
    '''
    # handle options
    cblabel = 'L1 Coverage'
    zlabel = 'z [' + xunit + ']'
    xlabel = 'x [' + xunit + ']'
    cm = 'GnBu'
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
                                                           cmap_name=cm,
                                                           no_elecs=elecs,
                                                           )
    return fig, ax, cnorm, cmap, cb


def plot_single(plotman, filename, mag, alpha, options):
    '''Plot only the magnitude of the last iteration in a single plot.
    '''
    f, ax = plt.subplots(1, figsize=(3.5, 2))
    if options.title is None:
        options.title = 'Magnitude'
    if options.cmaglin:
        cid = plotman.parman.add_data(np.power(10, mag))
        loglin = 'rho'
    else:
        cid = plotman.parman.add_data(mag)
        loglin = 'log_rho'
    plot_mag(cid, ax, plotman, options.title, loglin, alpha,
             options.xmin, options.xmax, options.zmin, options.zmax,
             options.unit, options.mag_cbtiks, options.no_elecs,)
    f.tight_layout()
    f.savefig(filename[4:] + '.png', dpi=300)


def alpha_from_cov(plotman, alpha_cov):
    '''Calculate alpha values from the coverage/2.5.
    '''
    abscov = np.abs(load_cov('inv/coverage.mag'))
    if alpha_cov:
        normcov = np.divide(abscov, 2.5)
        normcov[np.where(normcov > 1)] = 1
        mask = np.subtract(1, normcov)
        alpha = plotman.parman.add_data(mask)
    else:
        alpha = plotman.parman.add_data(np.ones(len(abscov)))
    return alpha, plotman


def plot_tomodir(plotman, cov, mag, pha, pha_fpi, alpha, options):
    '''Plot the data of the tomodir in one overview plot.
    '''
    # create figure
    f, ax = plt.subplots(2, 4, figsize=(14, 4))
    if options.title is not None:
        plt.suptitle(options.title, fontsize=18)
        plt.subplots_adjust(wspace=1, top=0.8)
    # plot magnitue
    if options.cmaglin:
        cid = plotman.parman.add_data(np.power(10, mag))
        loglin = 'rho'
    else:
        cid = plotman.parman.add_data(mag)
        loglin = 'log_rho'
    plot_mag(cid, ax[0, 0], plotman, 'Magnitude', loglin, alpha,
             options.xmin, options.xmax, options.zmin, options.zmax,
             options.unit, options.mag_cbtiks, options.no_elecs,
             )
    # plot coverage
    cid = plotman.parman.add_data(cov)
    plot_cov(cid, ax[1, 0], plotman, 'Coverage',
             options.xmin, options.xmax, options.zmin, options.zmax,
             options.unit, options.cov_cbtiks, options.no_elecs,
             )
    # plot phase, real, imag
    if pha != []:
        cid = plotman.parman.add_data(pha)
        plot_pha(cid, ax[0, 1], plotman, 'Phase', alpha,
                 options.xmin, options.xmax, options.zmin, options.zmax,
                 options.unit, options.pha_cbtiks, options.no_elecs,
                 )
        [real, imag] = calc_complex(mag, pha)
        cid_re = plotman.parman.add_data(real)
        cid_im = plotman.parman.add_data(imag)
        plot_real(cid_re, ax[0, 2], plotman, 'Real Part', alpha,
                  options.xmin, options.xmax, options.zmin, options.zmax,
                  options.unit, options.real_cbtiks, options.no_elecs,
                  )
        plot_imag(cid_im, ax[0, 3], plotman, 'Imaginary Part', alpha,
                  options.xmin, options.xmax, options.zmin, options.zmax,
                  options.unit, options.imag_cbtiks, options.no_elecs,
                  )
    else:
        ax[0, 1].axis('off')
        ax[0, 2].axis('off')
        ax[0, 3].axis('off')
    # plot fpi phase, real, imag
    if pha_fpi != []:
        cid = plotman.parman.add_data(pha_fpi)
        plot_pha(cid, ax[1, 1], plotman, 'FPI Phase', alpha,
                 options.xmin, options.xmax, options.zmin, options.zmax,
                 options.unit, options.pha_cbtiks, options.no_elecs,
                 )
        [real, imag] = calc_complex(mag, pha_fpi)
        cid_fre = plotman.parman.add_data(real)
        cid_fim = plotman.parman.add_data(imag)
        plot_real(cid_fre, ax[1, 2], plotman, 'FPI Real Part', alpha,
                  options.xmin, options.xmax, options.zmin, options.zmax,
                  options.unit, options.real_cbtiks, options.no_elecs,
                  )
        plot_imag(cid_fim, ax[1, 3], plotman, 'FPI Imaginary Part', alpha,
                  options.xmin, options.xmax, options.zmin, options.zmax,
                  options.unit, options.imag_cbtiks, options.no_elecs,
                  )
    else:
        ax[1, 1].axis('off')
        ax[1, 2].axis('off')
        ax[1, 3].axis('off')
    f.tight_layout()
    f.savefig('td_overview.png', dpi=300)
    return f, ax


def main():
    options = handle_options()
    matplotlib.style.use('default')
    mpl_style.general_settings()
    # load grid
    grid = CRGrid.crt_grid('grid/elem.dat',
                           'grid/elec.dat')
    plotman = CRPlot.plotManager(grid=grid)
    # get alpha
    alpha, plotman = alpha_from_cov(plotman, options.alpha_cov)
    if not options.single:
        [datafiles, filetype] = list_datafiles()
        [cov, mag, pha, pha_fpi] = read_datafiles(datafiles,
                                                  filetype,
                                                  options.column)
        plot_tomodir(plotman, cov, mag, pha, pha_fpi, alpha, options)
    else:
        filename = read_iter(False)
        mag = load_rho(filename, options.column)
        plot_single(plotman, filename, mag, alpha, options)


if __name__ == '__main__':
    main()
