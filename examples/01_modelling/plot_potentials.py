#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
This is my example script
=========================

This example doesn't do much, it just makes a simple plot
"""
import crtomo
grid = crtomo.crt_grid('grid_rhizotron/elem.dat', 'grid_rhizotron/elec.dat')

td = crtomo.tdMan(grid=grid)
td.crmod_cfg['write_pots'] = 'T'
td.crmod_cfg['2D'] = '0'
td.crmod_cfg['fictitious_sink'] = 'T'

import numpy as np
td.configs.add_to_configs(
    np.array((
        (6, 26, 1, 2),
        (1, 3, 7, 5),
    ))
)

# import IPython; IPython.embed()
# exit()
# td.add_homogeneous_model(100, 0)
r = td.load_rho_file('grid_rhizotron/rho_root.dat')
td.register_forward_model(*r)

td.model(potentials=True)

pot_mag, pot_pha = td.get_potential(0)

nid = td.nodeman.add_data(pot_mag)
nid_pha = td.nodeman.add_data(pot_pha)

import pylab as plt
fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 8 / 2.54))

ax = axes[0]
td.plot.plot_nodes_pcolor_to_ax(
    ax, nid,
    plot_colorbar=True,
)
# td.plot.plot_nodes_contour_to_ax(
#     ax, nid,
#     plot_colorbar=True,

# )
ax.set_aspect('equal')
ax = axes[1]
td.plot.plot_nodes_pcolor_to_ax(
    ax, nid_pha,
    plot_colorbar=True,
)
ax.set_aspect('equal')
fig.savefig('test.pdf')
