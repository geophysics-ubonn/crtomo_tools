#!/usr/bin/env python
"""
Plot a histogram of a given volt.dat file (mod/volt.dat)
END DOCUMENTATION
"""
import os
import numpy as np

from crtomo.mpl_setup import *


def main():
    voltfiles = (
        'mod/volt.dat',
        'volt.dat'
    )

    for filename in voltfiles:
        if not os.path.isfile(filename):
            continue

        print('loading file: {0}'.format(filename))
        volt_data = np.loadtxt(filename, skiprows=1)
        break

    for i in range(10, 100, 10):
        print('Magnitude percentile {0}%: {1:0.6} (log10: {2})'.format(
            i,
            np.percentile(volt_data[:, 2], i),
            np.log10(np.percentile(volt_data[:, 2], i)),
        ))

    for i in range(10, 100, 10):
        print('Phase percentile {0}%: {1}'.format(
            i,
            np.percentile(volt_data[:, 3], i),
        ))

    fig, axes = plt.subplots(2, 2, figsize=(7, 5))

    ax = axes[0, 0]
    ax.hist(volt_data[:, 2], 100)
    ax.set_xlabel(r'$|Z|~[\Omega]$')
    ax.set_ylabel('$\#$')

    ax = axes[0, 1]
    ax.hist(volt_data[:, 3], 100)
    ax.set_xlabel(r'$\phi~[mrad]$')
    ax.set_ylabel('$\#$')

    ax = axes[1, 0]
    ax.hist(np.log10(volt_data[:, 2]), 100)
    ax.set_xlabel(r'$log_{10}(|Z|~[\Omega])$')
    ax.set_ylabel('$\#$')
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

    fig.tight_layout()

    axes[1, 1].set_visible(False)
    fig.savefig('volt_histogram.png', dpi=300)


if __name__ == '__main__':
    main()
