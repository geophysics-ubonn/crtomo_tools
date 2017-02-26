#!/usr/bin/env python
"""
Plot a histogram of a given volt.dat file (mod/volt.dat).

Examples
--------

>>> volt_histogram
loading file: mod/volt.dat
minimum/maximum magnitude: 0.000164396115 / 9.16193485 Ohm
minimum/maximum phase: 0.0 / 0.0 mrad
Magnitude percentile 10%: 0.00116352 (log10: -2.9342247593548754)
Magnitude percentile 20%: 0.00292693 (log10: -2.533587969324661)
Magnitude percentile 30%: 0.00668467 (log10: -2.1749200553584274)
Magnitude percentile 40%: 0.0135107 (log10: -1.869322193615539)
Magnitude percentile 50%: 0.0261565 (log10: -1.5824204939114044)
Magnitude percentile 60%: 0.048195 (log10: -1.3169977466790868)
Magnitude percentile 70%: 0.0959452 (log10: -1.0179765941996923)
Magnitude percentile 80%: 0.209225 (log10: -0.679386765549746)
Magnitude percentile 90%: 0.592489 (log10: -0.22731998487816202)
Phase percentile 10%: 0.0
Phase percentile 20%: 0.0
Phase percentile 30%: 0.0
Phase percentile 40%: 0.0
Phase percentile 50%: 0.0
Phase percentile 60%: 0.0
Phase percentile 70%: 0.0
Phase percentile 80%: 0.0
Phase percentile 90%: 0.0

"""
import os
import numpy as np

from crtomo.mpl_setup import *


def main():

    # we check for these files
    voltfiles = (
        'mod/volt.dat',
        'volt.dat',
        '../mod/volt.dat',
    )

    for filename in voltfiles:
        if not os.path.isfile(filename):
            continue

        print('loading file: {0}'.format(filename))
        volt_data = np.loadtxt(filename, skiprows=1)
        break

    print('minimum/maximum magnitude: {0} / {1} Ohm'.format(
        np.min(volt_data[:, 2]),
        np.max(volt_data[:, 2]),
    ))
    print('minimum/maximum phase: {0} / {1} mrad'.format(
        np.min(volt_data[:, 3]),
        np.max(volt_data[:, 3]),
    ))

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
