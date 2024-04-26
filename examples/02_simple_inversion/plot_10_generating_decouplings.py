#!/usr/bin/env python
r"""
Generating regularization decouplins
====================================

"""
###############################################################################
# Imports
import crtomo
import numpy as np
import matplotlib.pylab as plt
# this import is only required to plot directly into the output directory
import reda

###############################################################################
mesh = crtomo.crt_grid.create_surface_grid(20, spacing=1, debug=True)
tdm = crtomo.tdMan(grid=mesh)

###############################################################################
path_info1 = tdm.grid.determine_path_along_nodes(
    [0, 0],
    [20, -2],
)
decouplings1 = np.hstack(
    (path_info1[:, 2:4], np.ones(path_info1.shape[0])[:, np.newaxis])
)

path_info2 = tdm.grid.determine_path_along_nodes(
    [-6, -4],
    [20, -8],
)
decouplings2 = np.hstack(
    (path_info2[:, 2:4], np.ones(path_info2.shape[0])[:, np.newaxis] * 0.5)
)
tdm.add_to_decouplings(decouplings1)
tdm.add_to_decouplings(decouplings2)
###############################################################################
# Decouplings can now be accessed as a numpy array:
print(tdm.decouplings.shape)

###############################################################################
fig, ax = plt.subplots()
mesh.plot_grid_to_ax(ax)
tdm.plot_decouplings_to_ax(ax, True)
ax.plot([0, 20], [0, -2], color='yellow', label='input polygon 1')
ax.plot([-6, 20], [-4, -8], color='yellow', label='input polygon 2')
ax.set_title(
    'Generating decouplings from polygons',
    loc='left',
)
ax.legend(
    facecolor='white',
    framealpha=1,
)
with reda.CreateEnterDirectory('out_10'):
    fig.savefig('decouplings_from_polygon.jpg', dpi=300, bbox_inches='tight')
