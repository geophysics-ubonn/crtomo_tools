#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
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
    plt.register_cmap(cmap=newcmap)

    return newcmap


def read_lastmodfile(directory):
    """
    Return the number of the final inversion result.
    """
    filename = '{0}/exe/inv.lastmod'.format(directory)
    # filename HAS to exist. Otherwise the inversion was not finished
    if(not os.path.isfile(filename)):
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
    if(
            (data[:, 0].all() - inv[:, 0].all() == 0) and
            (data[:, 1].all() - inv[:, 1].all() == 0)):
        print('Dipoles are equal')
        source = []
        receiver = []
        for AB, MN, rho, pha in data:
            s = ((AB % 1e4) - (round(AB / 1e4))) / 2 + round(AB / 1e4)
            source.append(s)
            r = ((round(MN / 1e4)) - (MN % 1e4)) / 2 + (MN % 1e4)
            receiver.append(r)
        mag_res = np.column_stack(
            (source, receiver, data[:, 2] - inv[:, 2], )
        )

        vmin = min(mag_res[:, 2])
        vmax = max(mag_res[:, 2])
        shrunk_cmap = shiftedColorMap(
            mpl.cm.coolwarm, midpoint=1 - vmax / (vmax + abs(vmin))
        )
        fig, ax = plt.subplots()
        sc = ax.scatter(mag_res[:, 0],
                        mag_res[:, 1],
                        c=mag_res[:, 2],
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
        pha_res = np.column_stack(
            (source, receiver, data[:, 3] - inv[:, 3], )
        )

        vmin = min(pha_res[:, 2])
        vmax = max(pha_res[:, 2])
        shrunk_cmap = shiftedColorMap(
            mpl.cm.coolwarm, midpoint=1 - vmax / (vmax + abs(vmin))
        )
        fig, ax = plt.subplots()
        sc = ax.scatter(
            pha_res[:, 0],
            pha_res[:, 1],
            c=pha_res[:, 2],
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


if __name__ == '__main__':
    main()
