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
                      dest='xunit',
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
    parser.add_option('-v',
                      '--vmin',
                      dest='vmin',
                      help='Minium of colorbar',
                      type='float',
                      )
    parser.add_option('-V',
                      '--vmax',
                      dest='vmax',
                      help='Maximum of colorbar',
                      type='float',
                      )

    (options, args) = parser.parse_args()
    return options


def check_minmax(plotman, cid, xmin, xmax, zmin, zmax, vmin, vmax):
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
        vmin = subdata.min()
    if vmax is None:
        vmax = subdata.max()

    return xmin, xmax, zmin, zmax, vmin, vmax


class overview_plot():

    def __init__(self, title, liste, alpha, unit):
        self.title = title
        self.dirs = liste
        N = len(self.dirs)
        self.rows = math.ceil(N/4)
        self.columns = 4
        self.cbunit = units.get_label(unit)

        self.create_figure()
        self.load_grid(alpha)
        self.cm(unit)

    def cm(self, unit):
        if unit == 'log_rho':
            self.cm = 'viridis'
        elif unit == 'phi':
            self.cm = 'plasma'
        elif unit == 'log_real':
            self.cm = 'viridis_r'
        elif unit == 'log_imag':
            self.cm = 'plasma_r'
        else:
            print('No colorbar defined')
            exit()

    def create_figure(self):
        self.fig, self.axs = plt.subplots(self.rows,
                                          ncols=4,
                                          figsize=(15, 1.8 * self.rows))
        plt.suptitle(self.title, fontsize=18)
        plt.subplots_adjust(wspace=1, top=2.8)

    def save(self):
        self.fig.tight_layout()
        self.fig.savefig('sd_' + self.title + '.png', dpi=300)

    def load_grid(self, alpha):
        '''Load grid and calculate alpha values from the coverage/2.5.
        '''
        grid = CRGrid.crt_grid(self.dirs[0] + '/grid/elem.dat',
                               self.dirs[0] + '/grid/elec.dat')
        self.plotman = CRPlot.plotManager(grid=grid)

        name = self.dirs[0] + '/inv/coverage.mag'
        content = np.genfromtxt(name, skip_header=1,
                                skip_footer=1, usecols=([2]))
        abscov = np.abs(content)
        if alpha:
            normcov = np.divide(abscov, 2.5)
            normcov[np.where(normcov > 1)] = 1
            mask = np.subtract(1, normcov)
            self.alpha = self.plotman.parman.add_data(mask)
        else:
            self.alpha = self.plotman.parman.add_data(np.ones(len(abscov)))


class subplot():

    def __init__(self, ov_plot, title, typ='cmplx'):
        self.title = title
        self.type = typ
        self.plotman = ov_plot.plotman

    def load_data(self, opt):
        os.chdir(self.title)
        # get iteration
        linestring = open('exe/inv.lastmod', 'r').readline().strip()
        linestring = linestring.replace('\n', '')
        linestring = linestring.replace('../', '')
        linestring = linestring.replace('mag', '')
        # open data file
        name = linestring + self.type

        if self.type == 'mag':
            try:
                self.data = np.loadtxt(name, skiprows=1,
                                       usecols=([opt.column]))
            except:
                raise ValueError('Given column to open does not exist.')
        if self.type == 'pha':
            try:
                self.data = np.loadtxt(name, skiprows=1, usecols=([2]))
            except:
                raise ValueError('No phase data to open.')
        os.chdir('..')

    def plot_data(self, ov_plot, opt, i, j):
        self.ax = ov_plot.axs[i, j]
        # add data to plotman
        if opt.cmaglin and self.type == 'mag':
            self.cid = self.plotman.parman.add_data(np.power(10, self.data))
            ov_plot.cbunit = 'rho'
        else:
            self.cid = self.plotman.parman.add_data(self.data)
        # handle options
        cblabel = units.get_label(ov_plot.cbunit)
        zlabel = 'z [' + opt.xunit + ']'
        xlabel = 'x [' + opt.xunit + ']'
        xmin, xmax, zmin, zmax, vmin, vmax = check_minmax(ov_plot.plotman,
                                                          self.cid,
                                                          opt.xmin, opt.xmax,
                                                          opt.zmin, opt.zmax,
                                                          opt.vmin, opt.vmax,
                                                          )
        # plot
        fig, ax, cnorm, cmap, cb = ov_plot.plotman.plot_elements_to_ax(
               cid=self.cid,
               cid_alpha=ov_plot.alpha,
               ax=self.ax,
               xmin=xmin,
               xmax=xmax,
               zmin=zmin,
               zmax=zmax,
               cblabel=cblabel,
               cbnrticks=opt.cbtiks,
               title=self.title[3:],
               zlabel=zlabel,
               xlabel=xlabel,
               plot_colorbar=True,
               cmap_name=ov_plot.cm,
               cbmin=vmin,
               cbmax=vmax,
               )


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


def main():
    # options
    options = handle_options()
    matplotlib.style.use('default')
    mpl_style.general_settings()

    # directories to plot
    os.chdir('invmod')
    freq_dirs = os.listdir('.')
    freq_dirs.sort()

    # init overview plots
    ov_mag = overview_plot(title='Magnitude',
                           liste=freq_dirs,
                           alpha=options.alpha_cov,
                           unit='log_rho',
                           )
    ov_pha = overview_plot(title='Phase',
                           liste=freq_dirs,
                           alpha=options.alpha_cov,
                           unit='phi',
                           )
    ov_real = overview_plot(title='Real Part',
                            liste=freq_dirs,
                            alpha=options.alpha_cov,
                            unit='log_real',
                            )
    ov_imag = overview_plot(title='Imaginary Part',
                            liste=freq_dirs,
                            alpha=options.alpha_cov,
                            unit='log_imag',
                            )
    # create figure

    i = 0
    j = 0

    # plot each subplot
    for sub in np.arange(ov_mag.rows * ov_mag.columns):
        try:
            mag = subplot(ov_plot=ov_mag,
                          title=freq_dirs[sub],
                          typ='mag')
            pha = subplot(ov_plot=ov_pha,
                          title=freq_dirs[sub],
                          typ='pha')
            imag = subplot(ov_plot=ov_imag,
                           title=freq_dirs[sub],
                           )
            real = subplot(ov_plot=ov_real,
                           title=freq_dirs[sub],
                           )

            mag.load_data(options)
            pha.load_data(options)
            real.data, imag.data = calc_complex(mag.data, pha.data)

            mag.plot_data(ov_mag, options, i//4, j)
            pha.plot_data(ov_pha, options, i//4, j)
            imag.plot_data(ov_imag, options, i//4, j)
            real.plot_data(ov_real, options, i//4, j)
        except:
            # no subplot needed
            ov_mag.axs[i//4, j].axis('off')
            ov_pha.axs[i//4, j].axis('off')
            ov_imag.axs[i//4, j].axis('off')
            ov_real.axs[i//4, j].axis('off')
        i = i + 1
        j = j + 1
        if j == 4:
            j = 0

    os.chdir('..')
    # save plots
    ov_mag.save()
    ov_pha.save()
    ov_real.save()
    ov_imag.save()


if __name__ == '__main__':
    main()
