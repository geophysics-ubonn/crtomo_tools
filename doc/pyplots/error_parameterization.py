#!/usr/bin/env python
# *-* coding: utf-8 *-*
# generate a plot which visualizes certain properties of the error model for
# resistances.
from crtomo.mpl_setup import *
import numpy as np


def dR(a, b, R):
    """The error parameterization: a * R + b
    """
    return a * R + b


def dlogR(a, b, R):
    """logarithmic error parameterization
    """
    dlogR = np.abs(1 / R * dR(a, b, R))
    return dlogR


R = np.logspace(-4, 2, 100)

# generate the plot
fig, axes = plt.subplots(1, 2, figsize=(15 / 2.54, 7 / 2.54))

ax = axes[0]
print(
    dR(0.05, 0.01, R) / R
)
ax.plot(R, dR(0.05, 0.01, R) / R, '-')

# show absolute error
ax.plot(R, 0.01 / R)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$R [\Omega m]$')
ax.set_ylabel(r'$\Delta R / R [-]$')

# log. parameterization
ax = axes[1]

for a in (0.01, ):
    for b in (1e-3, 1e-2, 1e-1):
        ax.plot(
            np.log(R),
            dlogR(0.05, 0.01, R) / np.abs(np.log(R)),
            '-',
            label='{0} R + {1}'.format(a, b),
        )

        ax.plot(
            np.log(R),
            np.abs(1 / R * b) / np.abs(np.log(R)),
            label=r'{0} $\Omega$ const'.format(b),
            linestyle='dashed',
        )
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'$R [\Omega m]$')
ax.set_ylabel(r'$\Delta log(R) / log(R) [-]$')

ax.legend(
    loc="lower center",
    ncol=4,
    bbox_to_anchor=(0, 0, 1, 1),
    bbox_transform=fig.transFigure,
    fontsize=6.0
)

fig.tight_layout()

fig.subplots_adjust(bottom=0.5)
fig.savefig('out.png', dpi=150)
