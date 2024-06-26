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

The three main options are to plot everything in one figure, to plot individual
figures (--single) or to plot anisotropic results of magnitude and phase
(--aniso).
The script has to be run in a tomodir. Output file will be saved in tomodir.

'''
import os
from optparse import OptionParser
import numpy as np

import crtomo.mpl
import crtomo.mpl as mpl_style
import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib as mpl

import crtomo.plotManager as CRPlot
import crtomo.grid as CRGrid
import reda.main.units as units
from crtomo.mpl import get_mpl_version
mpl_version = get_mpl_version()
plt, mpl = crtomo.mpl.setup()


def handle_options():
    '''Handle options.
    '''
    parser = OptionParser()
    parser.set_defaults(cmaglin=False)
    parser.set_defaults(single=False)
    parser.set_defaults(aniso=False)
    parser.set_defaults(hlam=False)
    parser.set_defaults(alpha_cov=False)
    # general options
    parser.add_option("--single",
                      action="store_true",
                      dest="single",
                      help="plot each value into a separate file",
                      )
    parser.add_option("--aniso",
                      action="store_true",
                      dest="aniso",
                      help="plot anisotropic xyz",
                      )
    parser.add_option("--hlam",
                      action="store_true",
                      dest="hlam",
                      help="plot anisotropic hor/ver",
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
    parser.add_option('-c',
                      '--column',
                      dest='column',
                      help='column to plot of input file',
                      type='int',
                      default=2,
                      )
    parser.add_option("--cmaglin",
                      action="store_true",
                      dest="cmaglin",
                      help="linear colorbar for magnitude",
                      )
    parser.add_option("--crholin",
                      action="store_true",
                      dest="crholin",
                      help="linear colorbar for fwd model magnitude",
                      )
    # geometric options
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
                      help='Unit of length scale, typically meters (m) '
                      'or centimeters (cm)',
                      metavar='UNIT',
                      type='str',
                      default='m',
                      )
    # options for colorbars
    parser.add_option('--cov_cbtiks',
                      dest='cov_cbtiks',
                      help="Number of CB tiks for coverage",
                      type=int,
                      metavar="INT",
                      default=3,
                      )
    parser.add_option('--cov_vmin',
                      dest='cov_vmin',
                      help='Minium of colorbar',
                      type='float',
                      )
    parser.add_option('--cov_vmax',
                      dest='cov_vmax',
                      help='Maximum of colorbar',
                      type='float',
                      )
    parser.add_option('--mag_cbtiks',
                      dest='mag_cbtiks',
                      help="Number of CB tiks for magnitude",
                      type=int,
                      metavar="INT",
                      default=3,
                      )
    parser.add_option('--mag_vmin',
                      dest='mag_vmin',
                      help='Minium of colorbar',
                      type='float',
                      )
    parser.add_option('--mag_vmax',
                      dest='mag_vmax',
                      help='Maximum of colorbar',
                      type='float',
                      )
    parser.add_option('--pha_cbtiks',
                      dest='pha_cbtiks',
                      help="Number of CB tiks for phase",
                      type=int,
                      metavar="INT",
                      default=3,
                      )
    parser.add_option('--pha_vmin',
                      dest='pha_vmin',
                      help='Minium of colorbar',
                      type='float',
                      )
    parser.add_option('--pha_vmax',
                      dest='pha_vmax',
                      help='Maximum of colorbar',
                      type='float',
                      )
    parser.add_option('--real_cbtiks',
                      dest='real_cbtiks',
                      help="Number of CB tiks for real part",
                      type=int,
                      metavar="INT",
                      default=3,
                      )
    parser.add_option('--real_vmin',
                      dest='real_vmin',
                      help='Minium of colorbar',
                      type='float',
                      )
    parser.add_option('--real_vmax',
                      dest='real_vmax',
                      help='Maximum of colorbar',
                      type='float',
                      )
    parser.add_option('--imag_cbtiks',
                      dest='imag_cbtiks',
                      help="Number of CB tiks for imag part",
                      type=int,
                      metavar="INT",
                      default=3,
                      )
    parser.add_option('--imag_vmin',
                      dest='imag_vmin',
                      help='Minium of colorbar',
                      type='float',
                      )
    parser.add_option('--imag_vmax',
                      dest='imag_vmax',
                      help='Maximum of colorbar',
                      type='float',
                      )
    parser.add_option('--rat_vmin',
                      dest='rat_vmin',
                      help='Minium of colorbar',
                      type='float',
                      )
    parser.add_option('--rat_vmax',
                      dest='rat_vmax',
                      help='Maximum of colorbar',
                      type='float',
                      )

    parser.add_option(
        "--clog",
        action="store_true",
        dest="c_in_log",
        help="Plot real and imaginary part of conductivity in log10",
        # default=True,
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
    if (not os.path.isfile(filename)):
        print('Inversion was not finished! No last iteration found.')
        return 0

    if (use_fpi is True):
        if (os.path.isfile(filename_rhosuffix)):
            filename = filename_rhosuffix

    linestring = open(filename, 'r').readline().strip()
    linestring = linestring.replace('\n', '')
    linestring = linestring.replace('../', '')
    return linestring


def td_type():
    '''get type of the tomodir (complex or dc and whether fpi)
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
    if isinstance(it_rho, str) or it_rho > 0:
        files.append(it_rho)
        dtype.append('mag')

    if (isinstance(it_rho, str) or it_rho > 0) and is_cplx:
        files.append(it_rho.replace('mag', 'pha'))
        dtype.append('pha')
    if (isinstance(it_phase, str) or it_phase > 0) and is_fpi:
        files.append(it_phase.replace('mag', 'pha'))
        dtype.append('pha_fpi')

    return files, dtype


def read_datafiles(files, dtype, column):
    '''Load the datafiles and return cov, mag, phase and fpi phase values.
    '''
    mag = None
    pha = None
    pha_fpi = None
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
    if not os.path.isfile(name):
        return None
    # we need to support an older file format, therefore use the number of
    # columns in the header to detect the format
    header = np.loadtxt(name, max_rows=1)
    if header.size == 2:
        content = np.genfromtxt(
            name, skip_header=1, usecols=([2])
        )
    else:
        content = np.genfromtxt(
            name, skip_header=1, skip_footer=1, usecols=([2])
        )

    return content


def load_rho(name, column):
    '''Load a datafile with rho structure like mag and phase
    '''
    try:
        content = np.loadtxt(name, skiprows=1, usecols=([column]))
    except Exception:
        raise ValueError('Given column to open does not exist.')

    return content


def calc_complex(rmag, rpha):
    ''' Calculate real and imaginary part of the complex conductivity from
    magnitude and phase in log10.
    '''
    crho = 10 ** rmag * np.exp(1j * rpha / 1000.0)
    csigma = 1 / crho
    return csigma.real, csigma.imag


def plot_real(cid, ax, plotman, title, alpha, vmin, vmax,
              xmin, xmax, zmin, zmax, xunit, cbtiks, elecs):
    '''Plot real parts of the complex conductivity using the real_options.
    '''
    # handle options
    cblabel = units.get_label('log_real')
    zlabel = 'z [' + xunit + ']'
    xlabel = 'x [' + xunit + ']'
    cm = 'jet_r'
    xmin, xmax, zmin, zmax, vmin, vmax = check_minmax(
        plotman,
        cid,
        xmin, xmax,
        zmin, zmax,
        vmin, vmax,
    )
    # plot
    fig, ax, cnorm, cmap, cb, scalarMap = plotman.plot_elements_to_ax(
        cid=cid,
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
        cbmin=vmin,
        cbmax=vmax,
    )
    return fig, ax, cnorm, cmap, cb


def plot_imag(cid, ax, plotman, title, alpha, vmin, vmax,
              xmin, xmax, zmin, zmax, xunit, cbtiks, elecs):
    '''Plot imag parts of the complex conductivity using the imag_options.
    '''
    # handle options
    cblabel = units.get_label('log_imag')
    zlabel = 'z [' + xunit + ']'
    xlabel = 'x [' + xunit + ']'
    cm = 'plasma_r'
    xmin, xmax, zmin, zmax, vmin, vmax = check_minmax(
        plotman,
        cid,
        xmin, xmax,
        zmin, zmax,
        vmin, vmax,
    )
    print('IMAG vmin/vmax', vmin, vmax)
    # plot
    fig, ax, cnorm, cmap, cb, scalarMap = plotman.plot_elements_to_ax(
        cid=cid,
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
        cbmin=vmin,
        cbmax=vmax,
    )
    return fig, ax, cnorm, cmap, cb


def plot_mag(cid, ax, plotman, title, unit, alpha, vmin, vmax,
             xmin, xmax, zmin, zmax, xunit, cbtiks, elecs):
    '''Plot magnitude of the complex resistivity using the mag_options.
    '''
    # handle options
    cblabel = units.get_label(unit)
    zlabel = 'z [' + xunit + ']'
    xlabel = 'x [' + xunit + ']'
    xmin, xmax, zmin, zmax, vmin, vmax = check_minmax(
        plotman,
        cid,
        xmin, xmax,
        zmin, zmax,
        vmin, vmax,
    )
    # plot
    fig, ax, cnorm, cmap, cb, scalarMap = plotman.plot_elements_to_ax(
        cid=cid,
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
        cbmin=vmin,
        cbmax=vmax,
        cmap_name='jet_r',
    )
    return fig, ax, cnorm, cmap, cb


def plot_pha(cid, ax, plotman, title, alpha, vmin, vmax,
             xmin, xmax, zmin, zmax, xunit, cbtiks, elecs):
    '''Plot phase of the complex resistivity using the pha_options.
    '''
    # handle options
    cblabel = units.get_label('phi')
    zlabel = 'z [' + xunit + ']'
    xlabel = 'x [' + xunit + ']'
    cm = 'plasma'
    cm = 'jet_r'
    xmin, xmax, zmin, zmax, vmin, vmax = check_minmax(
        plotman,
        cid,
        xmin, xmax,
        zmin, zmax,
        vmin, vmax,
    )
    # plot
    fig, ax, cnorm, cmap, cb, scalarMap = plotman.plot_elements_to_ax(
        cid=cid,
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
        cbmin=vmin,
        cbmax=vmax,
    )
    return fig, ax, cnorm, cmap, cb


def plot_cov(cid, ax, plotman, title, vmin, vmax,
             xmin, xmax, zmin, zmax, xunit, cbtiks, elecs):
    '''Plot coverage of the complex resistivity using the cov_options.
    '''
    # handle options
    cblabel = units.get_label('cov')
    zlabel = 'z [' + xunit + ']'
    xlabel = 'x [' + xunit + ']'
    cm = 'GnBu'
    xmin, xmax, zmin, zmax, vmin, vmax = check_minmax(
        plotman,
        cid,
        xmin, xmax,
        zmin, zmax,
        vmin, vmax,
    )
    # plot
    fig, ax, cnorm, cmap, cb, scalarMap = plotman.plot_elements_to_ax(
        cid=cid,
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
        cbmin=vmin,
        cbmax=vmax,
    )
    return fig, ax, cnorm, cmap, cb


def plot_ratio(cid, ax, plotman, title, alpha, vmin, vmax,
               xmin, xmax, zmin, zmax, xunit, cbtiks, elecs):
    '''Plot ratio of two conductivity directions.
    '''
    # handle options
    cblabel = 'anisotropy ratio'
    zlabel = 'z [' + xunit + ']'
    xlabel = 'x [' + xunit + ']'
    # cm = 'brg'
    cm = 'RdYlGn'
    xmin, xmax, zmin, zmax, vmin, vmax = check_minmax(
        plotman,
        cid,
        xmin, xmax,
        zmin, zmax,
        vmin, vmax,
    )
    # plot
    fig, ax, cnorm, cmap, cb, scalarMap = plotman.plot_elements_to_ax(
        cid=cid,
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
        cbmin=vmin,
        cbmax=vmax,
    )
    return fig, ax, cnorm, cmap, cb


def alpha_from_cov(plotman, alpha_cov):
    '''Calculate alpha values from the coverage/2.5.
    '''
    cov_file = 'inv/coverage.mag'
    if not os.path.isfile(cov_file):
        return None, plotman

    abscov = np.abs(load_cov(cov_file))

    if alpha_cov:
        normcov = np.divide(abscov, 3)
        normcov[np.where(normcov > 1)] = 1
        mask = np.subtract(1, normcov)
        alpha = plotman.parman.add_data(mask)
    else:
        alpha = plotman.parman.add_data(np.ones(len(abscov)))
    return alpha, plotman


def check_minmax(plotman, cid, xmin, xmax, zmin, zmax, vmin, vmax):
    """
    Get min and max values for axes and colorbar if not given
    """
    if xmin is None:
        xmin = plotman.grid.grid['x'].min()
    if xmax is None:
        xmax = plotman.grid.grid['x'].max()
    if zmin is None:
        zmin = plotman.grid.grid['z'].min()
    if zmax is None:
        zmax = plotman.grid.grid['z'].max()
    if isinstance(cid, int):
        subdata = plotman.parman.parsets[cid]
    else:
        subdata = cid
    if vmin is None:
        vmin = np.nanmin(subdata)
    if vmax is None:
        vmax = np.nanmax(subdata)

    return xmin, xmax, zmin, zmax, vmin, vmax


def getfigsize(plotman):
    '''calculate appropriate sizes for the subfigures
    '''
    xmin = plotman.grid.grid['x'].min()
    xmax = plotman.grid.grid['x'].max()
    zmin = plotman.grid.grid['z'].min()
    zmax = plotman.grid.grid['z'].max()
    if np.abs(zmax - zmin) < np.abs(xmax - xmin):
        sizex = 10 / 2.54
        sizez = 1.2 * sizex * (np.abs(zmax - zmin) / np.abs(xmax - xmin))
    else:
        sizez = 10 / 2.54
        sizex = sizez * (np.abs(xmax - xmin) / np.abs(zmax - zmin))
    # add 1 inch to accommodate colorbar
    sizex += 1.3
    sizez += 1
    return sizex, sizez


def create_non_dcplots(plotman, ax, mag, pha, options, alpha):
    if pha != []:
        cid = plotman.parman.add_data(pha)
        plot_pha(
            cid, ax[0, 1], plotman, 'Phase', alpha,
            options.pha_vmin, options.pha_vmax,
            options.xmin, options.xmax, options.zmin, options.zmax,
            options.unit, options.pha_cbtiks, options.no_elecs,
        )
        [real, imag] = calc_complex(mag, pha)
        cid_re = plotman.parman.add_data(real)
        cid_im = plotman.parman.add_data(np.log10(imag))
        plot_real(
            cid_re, ax[0, 2], plotman, 'Real Part', alpha,
            options.real_vmin, options.real_vmax,
            options.xmin, options.xmax, options.zmin, options.zmax,
            options.unit, options.real_cbtiks, options.no_elecs,
        )
        plot_imag(
            cid_im, ax[0, 3], plotman, 'Imaginary Part', alpha,
            options.imag_vmin, options.imag_vmax,
            options.xmin, options.xmax, options.zmin, options.zmax,
            options.unit, options.imag_cbtiks, options.no_elecs,
        )
    else:
        ax[0, 1].axis('off')
        ax[0, 2].axis('off')
        ax[0, 3].axis('off')


def create_fpiplots(plotman, ax, mag, pha_fpi, options, alpha):
    if pha_fpi != []:
        cid = plotman.parman.add_data(pha_fpi)
        plot_pha(cid, ax[1, 1], plotman, 'FPI Phase', alpha,
                 options.pha_vmin, options.pha_vmax,
                 options.xmin, options.xmax, options.zmin, options.zmax,
                 options.unit, options.pha_cbtiks, options.no_elecs,
                 )
        [real, imag] = calc_complex(mag, pha_fpi)
        cid_fre = plotman.parman.add_data(real)
        cid_fim = plotman.parman.add_data(imag)
        plot_real(cid_fre, ax[1, 2], plotman, 'FPI Real Part', alpha,
                  options.real_vmin, options.real_vmax,
                  options.xmin, options.xmax, options.zmin, options.zmax,
                  options.unit, options.real_cbtiks, options.no_elecs,
                  )
        plot_imag(cid_fim, ax[1, 3], plotman, 'FPI Imaginary Part',
                  alpha, options.imag_vmin, options.imag_vmax,
                  options.xmin, options.xmax, options.zmin, options.zmax,
                  options.unit, options.imag_cbtiks, options.no_elecs,
                  )
    else:
        ax[1, 1].axis('off')
        ax[1, 2].axis('off')
        ax[1, 3].axis('off')


def create_tdplot(plotman, cov, mag, pha, pha_fpi, alpha, options):
    '''Plot the data of the tomodir in one overview plot.
    '''
    sizex, sizez = getfigsize(plotman)
    # create figure
    f, ax = plt.subplots(2, 4, figsize=(4 * sizex, 2 * sizez))
    if options.title is not None:
        plt.suptitle(options.title, fontsize=18)
    # plot magnitue
    if options.cmaglin:
        cid = plotman.parman.add_data(np.power(10, mag))
        loglin = 'rho'
    else:
        cid = plotman.parman.add_data(mag)
        loglin = 'log_rho'
    plot_mag(cid, ax[0, 0], plotman, 'Magnitude', loglin, alpha,
             options.mag_vmin, options.mag_vmax,
             options.xmin, options.xmax, options.zmin, options.zmax,
             options.unit, options.mag_cbtiks, options.no_elecs,
             )
    # plot coverage
    cid = plotman.parman.add_data(cov)
    plot_cov(cid, ax[1, 0], plotman, 'Coverage',
             options.cov_vmin, options.cov_vmax,
             options.xmin, options.xmax, options.zmin, options.zmax,
             options.unit, options.cov_cbtiks, options.no_elecs,
             )
    # plot phase, real, imag
    create_non_dcplots(plotman, ax, mag, pha, options, alpha)
    # plot fpi phase, real, imag
    create_fpiplots(plotman, ax, mag, pha_fpi, options, alpha)
    f.tight_layout()
    f.savefig('td_overview.png', dpi=300)
    return f, ax


def create_singleplots(plotman, cov, mag, pha, pha_fpi, alpha, options):
    '''Plot the data of the tomodir in individual plots.
    '''
    if len(mag) == 0:
        mag = np.ones(plotman.grid.nr_of_elements) * np.nan
    if cov is None:
        cov = np.ones_like(mag) * np.nan

    magunit = 'log_rho'
    if pha is not None:
        [real, imag] = calc_complex(mag, pha)
        if options.c_in_log:
            real = np.log10(real)
            with np.errstate(divide='ignore'):
                imag = np.log10(imag)
            imag[np.isinf(imag)] = np.nan

        if pha_fpi is not None:
            [real_fpi, imag_fpi] = calc_complex(mag, pha_fpi)
            if options.cmaglin:
                mag = np.power(10, mag)
                magunit = 'rho'
            if options.c_in_log:
                real_fpi = np.log10(real_fpi)
                with np.errstate(divide='ignore'):
                    imag_fpi = np.log10(imag_fpi)
                imag_fpi[np.isinf(imag_fpi)] = np.nan
            # import IPython
            # IPython.embed()
            # print(imag_fpi, np.nanmin(imag_fpi), np.nanmax(imag_fpi))
            # print(options.imag_vmin, options.imag_vmax)
            # exit()

            data = np.column_stack((
                mag, cov, pha, real, imag,
                pha_fpi, real_fpi, imag_fpi
            ))
            titles = [
                'Magnitude', 'Coverage',
                'Phase', 'Real Part', 'Imaginary Part',
                'FPI Phase', 'FPI Real Part', 'FPI Imaginary Part'
            ]
            unites = [
                magunit, 'cov',
                'phi', 'log_real', 'log_imag',
                'phi', 'log_real', 'log_imag'
            ]
            vmins = [
                options.mag_vmin, options.cov_vmin,
                options.pha_vmin, options.real_vmin, options.imag_vmin,
                options.pha_vmin, options.real_vmin, options.imag_vmin
            ]
            vmaxs = [options.mag_vmax, options.cov_vmax,
                     options.pha_vmax, options.real_vmax, options.imag_vmax,
                     options.pha_vmax, options.real_vmax, options.imag_vmax]
            cmaps = ['jet', 'GnBu',
                     'jet_r', 'jet_r', 'plasma_r',
                     'jet_r', 'jet_r', 'plasma_r']
            saves = ['rho', 'cov',
                     'phi', 'real', 'imag',
                     'fpi_phi', 'fpi_real', 'fpi_imag']
        else:
            if options.cmaglin:
                print('Cmaglin')
                print(mag, mag.min(), mag.max())
                mag = np.power(10, mag)
                print('after', mag.min(), mag.max())
                magunit = 'rho'
            data = np.column_stack((mag, cov, pha, real, imag))
            titles = ['Magnitude', 'Coverage',
                      'Phase', 'Real Part', 'Imaginary Part']
            unites = [magunit, 'cov',
                      'phi', 'log_real', 'log_imag']
            vmins = [options.mag_vmin, options.cov_vmin,
                     options.pha_vmin, options.real_vmin, options.imag_vmin]
            vmaxs = [options.mag_vmax, options.cov_vmax,
                     options.pha_vmax, options.real_vmax, options.imag_vmax]
            cmaps = ['jet', 'GnBu',
                     'jet_r', 'jet_r', 'plasma_r']
            saves = ['rho', 'cov',
                     'phi', 'real', 'imag']
    else:
        data = np.column_stack((mag, cov))
        titles = ['Magnitude', 'Coverage']
        unites = [magunit, 'cov']
        vmins = [options.mag_vmin, options.cov_vmin]
        vmaxs = [options.mag_vmax, options.cov_vmax]
        cmaps = ['jet', 'GnBu']
        saves = ['rho', 'cov']
    try:
        mod_rho = np.genfromtxt('rho/rho.dat', skip_header=1, usecols=([0]))
        if not options.crholin:
            mod_rho = np.log10(mod_rho)
        mod_pha = np.genfromtxt('rho/rho.dat', skip_header=1, usecols=([1]))
        if data.size == 0:
            data = np.column_stack((mod_rho, mod_pha))
        else:
            data = np.column_stack((data, mod_rho, mod_pha))
        titles.append('Model')
        titles.append('Model')
        if not options.crholin:
            unites.append('log_rho')
        else:
            unites.append('rho')
        unites.append('phi')
        vmins.append(options.mag_vmin)
        vmins.append(options.pha_vmin)
        vmaxs.append(options.mag_vmax)
        vmaxs.append(options.pha_vmax)
        cmaps.append('jet')
        cmaps.append('plasma')
        saves.append('rhomod')
        saves.append('phamod')
    except Exception as e:
        print(e)
        print('BAD ERROR')
        pass

    for datum, title, unit, vmin, vmax, cm, save in zip(
            np.transpose(data), titles, unites, vmins, vmaxs, cmaps, saves):
        if len(datum) == 0:
            continue
        if np.all(np.isnan(datum)):
            continue
        # print(save)
        # if save == 'fpi_imag':
        #     import IPython
        #     IPython.embed()
        # if save == 'rho' and options.cmaglin:
        #     datum = np.power(10, datum)
        #     unit = 'rho'

        sizex, sizez = getfigsize(plotman)
        f, ax = plt.subplots(1, figsize=(sizex, sizez))
        cid = plotman.parman.add_data(datum)
        # handle options
        cblabel = units.get_label(unit)
        if options.title is not None:
            title = options.title
        zlabel = 'z [' + options.unit + ']'
        xlabel = 'x [' + options.unit + ']'
        xmin, xmax, zmin, zmax, vmin, vmax = check_minmax(
            plotman,
            cid,
            options.xmin, options.xmax,
            options.zmin, options.zmax,
            vmin, vmax
        )
        # plot
        # https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.9.0.html#top-level-cmap-registration-and-access-functions-in-mpl-cm
        if mpl_version[0] <= 3 and mpl_version[1] < 9:
            cmap = mpl.cm.get_cmap(cm)
        else:
            cmap = mpl.colormaps[cm]

        fig, ax, cnorm, cmap, cb, scalarMap = plotman.plot_elements_to_ax(
                cid=cid,
                cid_alpha=alpha,
                ax=ax,
                xmin=xmin,
                xmax=xmax,
                zmin=zmin,
                zmax=zmax,
                cblabel=cblabel,
                title=title,
                zlabel=zlabel,
                xlabel=xlabel,
                plot_colorbar=True,
                cmap_name=cm,
                over=cmap(1.0),
                under=cmap(0.0),
                no_elecs=options.no_elecs,
                cbmin=vmin,
                cbmax=vmax,
                )
        f.tight_layout()
        f.savefig(save + '.png', dpi=300)


def create_anisomagplot(plotman, x, y, z, alpha, options):
    '''Plot the data of the tomodir in one overview plot.
    '''
    sizex, sizez = getfigsize(plotman)
    # create figure
    f, ax = plt.subplots(2, 3, figsize=(3 * sizex, 2 * sizez))
    if options.title is not None:
        plt.suptitle(options.title, fontsize=18)
        plt.subplots_adjust(wspace=1.5, top=2)
    # plot magnitue
    if options.cmaglin:
        cidx = plotman.parman.add_data(np.power(10, x))
        cidy = plotman.parman.add_data(np.power(10, y))
        cidz = plotman.parman.add_data(np.power(10, z))
        loglin = 'rho'
    else:
        cidx = plotman.parman.add_data(x)
        cidy = plotman.parman.add_data(y)
        cidz = plotman.parman.add_data(z)
        loglin = 'log_rho'
    cidxy = plotman.parman.add_data(np.divide(x, y))
    cidyz = plotman.parman.add_data(np.divide(y, z))
    cidzx = plotman.parman.add_data(np.divide(z, x))
    plot_mag(cidx, ax[0, 0], plotman, 'x', loglin, alpha,
             options.mag_vmin, options.mag_vmax,
             options.xmin, options.xmax, options.zmin, options.zmax,
             options.unit, options.mag_cbtiks, options.no_elecs,
             )
    plot_mag(cidy, ax[0, 1], plotman, 'y', loglin, alpha,
             options.mag_vmin, options.mag_vmax,
             options.xmin, options.xmax, options.zmin, options.zmax,
             options.unit, options.mag_cbtiks, options.no_elecs,
             )
    plot_mag(cidz, ax[0, 2], plotman, 'z', loglin, alpha,
             options.mag_vmin, options.mag_vmax,
             options.xmin, options.xmax, options.zmin, options.zmax,
             options.unit, options.mag_cbtiks, options.no_elecs,
             )
    plot_ratio(cidxy, ax[1, 0], plotman, 'x/y', alpha,
               options.rat_vmin, options.rat_vmax,
               options.xmin, options.xmax, options.zmin, options.zmax,
               options.unit, options.mag_cbtiks, options.no_elecs,
               )
    plot_ratio(cidyz, ax[1, 1], plotman, 'y/z', alpha,
               options.rat_vmin, options.rat_vmax,
               options.xmin, options.xmax, options.zmin, options.zmax,
               options.unit, options.mag_cbtiks, options.no_elecs,
               )
    plot_ratio(cidzx, ax[1, 2], plotman, 'z/x', alpha,
               options.rat_vmin, options.rat_vmax,
               options.xmin, options.xmax, options.zmin, options.zmax,
               options.unit, options.mag_cbtiks, options.no_elecs,
               )
    f.tight_layout()
    f.savefig('mag_aniso.png', dpi=300)
    return f, ax


def create_anisophaplot(plotman, x, y, z, alpha, options):
    '''Plot the data of the tomodir in one overview plot.
    '''
    sizex, sizez = getfigsize(plotman)
    # create figure
    f, ax = plt.subplots(2, 3, figsize=(3 * sizex, 2 * sizez))
    if options.title is not None:
        plt.suptitle(options.title, fontsize=18)
        plt.subplots_adjust(wspace=1, top=0.8)
    # plot phase
    cidx = plotman.parman.add_data(x)
    cidy = plotman.parman.add_data(y)
    cidz = plotman.parman.add_data(z)
    cidxy = plotman.parman.add_data(np.subtract(x, y))
    cidyz = plotman.parman.add_data(np.subtract(y, z))
    cidzx = plotman.parman.add_data(np.subtract(z, x))
    plot_pha(cidx, ax[0, 0], plotman, 'x', alpha,
             options.pha_vmin, options.pha_vmax,
             options.xmin, options.xmax, options.zmin, options.zmax,
             options.unit, options.pha_cbtiks, options.no_elecs,
             )
    plot_pha(cidy, ax[0, 1], plotman, 'y', alpha,
             options.pha_vmin, options.pha_vmax,
             options.xmin, options.xmax, options.zmin, options.zmax,
             options.unit, options.pha_cbtiks, options.no_elecs,
             )
    plot_pha(cidz, ax[0, 2], plotman, 'z', alpha,
             options.pha_vmin, options.pha_vmax,
             options.xmin, options.xmax, options.zmin, options.zmax,
             options.unit, options.pha_cbtiks, options.no_elecs,
             )
    plot_ratio(cidxy, ax[1, 0], plotman, 'x-y', alpha,
               options.rat_vmin, options.rat_vmax,
               options.xmin, options.xmax, options.zmin, options.zmax,
               options.unit, options.mag_cbtiks, options.no_elecs,
               )
    plot_ratio(cidyz, ax[1, 1], plotman, 'y-z', alpha,
               options.rat_vmin, options.rat_vmax,
               options.xmin, options.xmax, options.zmin, options.zmax,
               options.unit, options.mag_cbtiks, options.no_elecs,
               )
    plot_ratio(cidzx, ax[1, 2], plotman, 'z-x', alpha,
               options.rat_vmin, options.rat_vmax,
               options.xmin, options.xmax, options.zmin, options.zmax,
               options.unit, options.mag_cbtiks, options.no_elecs,
               )
    f.tight_layout()
    f.savefig('pha_aniso.png', dpi=300)
    return f, ax


def create_hlammagplot(plotman, h, ratio, alpha, options):
    '''Plot the data of the tomodir in one overview plot.
    '''
    sizex, sizez = getfigsize(plotman)
    # create figure
    f, ax = plt.subplots(1, 3, figsize=(3 * sizex, sizez))
    if options.title is not None:
        plt.suptitle(options.title, fontsize=18)
        plt.subplots_adjust(wspace=1, top=0.8)
    # plot magnitue
    if options.cmaglin:
        cidh = plotman.parman.add_data(np.power(10, h))
        cidv = plotman.parman.add_data(
                np.divide(np.power(10, h), np.power(10, ratio)))
        loglin = 'rho'
    else:
        cidh = plotman.parman.add_data(h)
        cidv = plotman.parman.add_data(
                np.log10(np.divide(np.power(10, h), np.power(10, ratio))))
        loglin = 'log_rho'

    cidr = plotman.parman.add_data(np.power(10, ratio))
    plot_mag(cidh, ax[0], plotman, 'horizontal', loglin, alpha,
             options.mag_vmin, options.mag_vmax,
             options.xmin, options.xmax, options.zmin, options.zmax,
             options.unit, options.mag_cbtiks, options.no_elecs,
             )
    plot_mag(cidv, ax[1], plotman, 'vertical', loglin, alpha,
             options.mag_vmin, options.mag_vmax,
             options.xmin, options.xmax, options.zmin, options.zmax,
             options.unit, options.mag_cbtiks, options.no_elecs,
             )
    plot_ratio(cidr, ax[2], plotman, 'hor/ver', alpha,
               options.rat_vmin, options.rat_vmax,
               options.xmin, options.xmax, options.zmin, options.zmax,
               options.unit, options.mag_cbtiks, options.no_elecs,
               )
    f.tight_layout()
    f.savefig('mag_hlam.png', dpi=300)
    return f, ax


def create_hlamphaplot(plotman, h, v, alpha, options):
    '''Plot the data of the tomodir in one overview plot.
    '''
    sizex, sizez = getfigsize(plotman)
    # create figure
    f, ax = plt.subplots(1, 3, figsize=(3 * sizex, sizez))
    if options.title is not None:
        plt.suptitle(options.title, fontsize=18)
        plt.subplots_adjust(wspace=1, top=0.8)
    cidh = plotman.parman.add_data(h)
    cidv = plotman.parman.add_data(v)

    cidr = plotman.parman.add_data(np.subtract(h, v))
    plot_pha(cidh, ax[0], plotman, 'horizontal', alpha,
             options.pha_vmin, options.pha_vmax,
             options.xmin, options.xmax, options.zmin, options.zmax,
             options.unit, options.pha_cbtiks, options.no_elecs,
             )
    plot_pha(cidv, ax[1], plotman, 'vertical', alpha,
             options.pha_vmin, options.pha_vmax,
             options.xmin, options.xmax, options.zmin, options.zmax,
             options.unit, options.pha_cbtiks, options.no_elecs,
             )
    plot_ratio(cidr, ax[2], plotman, 'hor - ver', alpha,
               options.rat_vmin, options.rat_vmax,
               options.xmin, options.xmax, options.zmin, options.zmax,
               options.unit, options.pha_cbtiks, options.no_elecs,
               )
    f.tight_layout()
    f.savefig('pha_hlam.png', dpi=300)
    return f, ax


def main():
    if os.path.basename(os.getcwd()) == 'exe':
        os.chdir('..')

    options = handle_options()
    matplotlib.style.use('default')
    mpl_style.general_settings()
    # load grid
    grid = CRGrid.crt_grid('grid/elem.dat',
                           'grid/elec.dat')
    plotman = CRPlot.plotManager(grid=grid)
    # get alpha
    alpha, plotman = alpha_from_cov(plotman, options.alpha_cov)
    # make tomodir overview plot
    if not options.single and not options.aniso and not options.hlam:
        [datafiles, filetype] = list_datafiles()
        [cov, mag, pha, pha_fpi] = read_datafiles(
                datafiles,
                filetype,
                options.column)
        create_tdplot(plotman, cov, mag, pha, pha_fpi, alpha, options)
    # make individual plots
    elif options.single and not options.aniso and not options.hlam:
        [datafiles, filetype] = list_datafiles()
        [cov, mag, pha, pha_fpi] = read_datafiles(
                datafiles,
                filetype,
                options.column)
        create_singleplots(plotman, cov, mag, pha, pha_fpi, alpha, options)
    # make plots of anisotropic results
    elif options.aniso and not options.single and not options.hlam:
        filename = read_iter(False)
        x = load_rho(filename, 2)
        y = load_rho(filename, 3)
        z = load_rho(filename, 4)
        create_anisomagplot(plotman, x, y, z, alpha, options)
        x = load_rho(filename[:-3] + 'pha', 2)
        y = load_rho(filename[:-3] + 'pha', 3)
        z = load_rho(filename[:-3] + 'pha', 4)
        create_anisophaplot(plotman, x, y, z, alpha, options)
    elif options.hlam and not options.single and not options.aniso:
        filename = read_iter(False)
        hor = load_rho(filename, 2)
        lam = load_rho(filename, 3)
        create_hlammagplot(plotman, hor, lam, alpha, options)
        hor = load_rho(filename[:-3] + 'pha', 2)
        ver = load_rho(filename[:-3] + 'pha', 4)
        create_hlamphaplot(plotman, hor, ver, alpha, options)
    else:
        print(
            'You can only use one option out of these: '
            '"single", "hlam" or "aniso", not two at the same time.'
        )
        exit()


if __name__ == '__main__':
    main()
