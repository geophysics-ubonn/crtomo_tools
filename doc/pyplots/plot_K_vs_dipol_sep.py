#!/usr/bin/env python
# plot the geometric factor K vs. the dipole separation of a dipole-dipole
# configuration
import crtomo.configManager as CRc
import numpy as np
from crtomo.mpl_setup import *


config = CRc.ConfigManager(nr_of_electrodes=40)
# generate configs, by hand
a = 1
b = 2
quads = []
for m in range(3, 40):
    quads.append((a, b, m, m + 1))

config.add_to_configs(quads)
config.compute_K_factors(spacing=1)

fig, ax = plt.subplots(figsize=(15 / 2.54, 10 / 2.54))

for spacing in np.arange(0.5, 4, 0.5):
    dipole_separation = (config.configs[:, 2] - config.configs[:, 1]) * spacing
    print(spacing, dipole_separation)
    K = config.compute_K_factors(spacing=spacing)
    ax.plot(dipole_separation, np.abs(K), '.-', label='spacing {0}m'.format(
        spacing
    ))

ax.set_xlabel('dipole separation [m]')
ax.set_ylabel('K [m]')
ax.set_title(
    'geometric factor for different electrode distances ' +
    '(dipole-dipole skip-0)',
    fontsize=10.0,
)

ax.axhline(y=5000, color='k', linestyle='dashed')
ax.annotate(
    'K = 5000',
    xy=(0, 6000),
)

ax.legend(loc='best')

fig.tight_layout()
fig.savefig('K_vs_dippol_sep.png')
ax.set_xlim(None, 40)
ax.set_ylim(-5000, 1e5)
fig.savefig('K_vs_dippol_sep_zoom.png')
