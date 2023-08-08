#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
# from mpl_toolkits.axes_grid1 import AxesGrid


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Parameters
    ----------
    cmap:
        The matplotlib colormap to be altered
    start:
        Offset from lowest point in the colormap's range.  Defaults to 0.0 (no
        lower ofset). Should be between 0.0 and `midpoint`.
    midpoint:
        The new center of the colormap. Defaults to 0.5 (no shift). Should be
        between 0.0 and 1.0. In general, this should be  1 - vmax/(vmax +
        abs(vmin)) For example if your data range from -15.0 to +5.0 and you
        want the center of the colormap at 0.0, `midpoint` should be set to  1
        - 5/(5 + 15)) or 0.75
    stop:
        Offset from highets point in the colormap's range.  Defaults to 1.0 (no
        upper ofset). Should be between `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)

    # plt.register_cmap(cmap=newcmap)

    return newcmap


def read_lastmodfile(directory):
    """
    Return the number of the final inversion result.
    """
    filename = '{0}/exe/inv.lastmod'.format(directory)
    # filename HAS to exist. Otherwise the inversion was not finished
    if not os.path.isfile(filename):
        return None

    linestring = open(filename, 'r').readline().strip()
    linestring = linestring.replace("\n", '')
    linestring = linestring.replace(".mag", '')
    linestring = linestring.replace("../inv/rho", '')
    return linestring


def open_data():
    content = np.loadtxt('mod/volt.dat', skiprows=1)
    return content  # , array


def open_inv():
    num = read_lastmodfile('.')
    content = np.loadtxt('inv/volt' + num + '.dat', skiprows=1)
    return content  # , array


def main():
    data = open_data()
    inv = open_inv()
    # import IPython
    # IPython.embed()
    ab_are_equal = (data[:, 0] - inv[:, 0] == 0).all()
    mn_are_equal = (data[:, 1] - inv[:, 1] == 0).all()
    if ab_are_equal and mn_are_equal:
        print('Dipoles are equal')
        rdata = pd.DataFrame((data[:, 0] / 1e4).astype(int), columns=['a', ])
        rdata['b'] = (data[:, 0] % 1e4).astype(int)
        rdata['m'] = (data[:, 1] / 1e4).astype(int)
        rdata['n'] = (data[:, 1] % 1e4).astype(int)
        rdata['ab_pos'] = (rdata['b'] - rdata['a']) / 2 + rdata['a']
        rdata['mn_pos'] = (rdata['n'] - rdata['m']) / 2 + rdata['m']
        rdata['diff_res'] = data[:, 2] - inv[:, 2]
        rdata['diff_pha'] = data[:, 3] - inv[:, 3]
        rdata['diff_res_abs'] = np.abs(rdata['diff_res'])
        rdata['diff_pha_abs'] = np.abs(rdata['diff_pha'])

        # import IPython
        # IPython.embed()

        vmin = min(rdata['diff_res'])
        vmax = max(rdata['diff_res'])
        shrunk_cmap = shiftedColorMap(
            mpl.cm.coolwarm,
            midpoint=1 - vmax / (vmax + abs(vmin)),
            name='map1',
        )
        fig, ax = plt.subplots()
        sc = ax.scatter(
            rdata['ab_pos'],
            rdata['mn_pos'],
            c=rdata['diff_res'],
            # s=7,
            edgecolor='k',
            linewidth=.1,
            cmap=shrunk_cmap
        )
        plt.colorbar(sc)
        ax.set_xlabel('source position [m]')
        ax.set_ylabel('receiver position [m]')
        fig.tight_layout()
        fig.savefig('residuals_mag.png', dpi=300)

        vmin = min(rdata['diff_pha'])
        vmax = max(rdata['diff_pha'])
        shrunk_cmap = shiftedColorMap(
            mpl.cm.coolwarm, midpoint=1 - vmax / (vmax + abs(vmin)),
            name='map2',
        )
        fig, ax = plt.subplots()
        sc = ax.scatter(
            rdata['ab_pos'],
            rdata['mn_pos'],
            c=rdata['diff_pha'],
            # s=7,
            edgecolor='k',
            linewidth=.1,
            cmap=shrunk_cmap
        )
        plt.colorbar(sc)
        ax.set_xlabel('source position [m]')
        ax.set_ylabel('receiver position [m]')
        fig.tight_layout()
        fig.savefig('residuals_pha.png', dpi=300)

        # save residuals to file
        rdata.sort_values('diff_res_abs', ascending=False).to_csv(
            'residuals.csv',
            sep='\t',
            float_format='%.3f',
        )


if __name__ == '__main__':
    main()
