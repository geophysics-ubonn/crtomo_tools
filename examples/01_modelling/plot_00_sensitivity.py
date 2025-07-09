#!/usr/bin/env python
# *-* coding: utf-8 *-*
"""
Plotting sensitivities
^^^^^^^^^^^^^^^^^^^^^^

Sensitivity distributions can be easily plotted using the tdMan class:
"""
###############################################################################
# imports
import numpy as np
import crtomo
# is only used for reda.CreateEnterDirectory
import reda
# sphinx_gallery_start_ignore
# Set global figure parameters to get unified output.
# Doesn't seem to work for plots created in the for-loop.
plt, mpl = crtomo.mpl.setup()
plt.rcParams["figure.figsize"] = (4, 1.75)
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.autolayout"] = True  # This does not seem to work?!
# sphinx_gallery_end_ignore

###############################################################################
# create and save a FEM-grid
grid = crtomo.crt_grid.create_surface_grid(
    nr_electrodes=10,
    spacing=1,
    depth=4,
    char_lengths=0.05,
    left=5,
    right=5,
)

fig, ax = grid.plot_grid()
# sphinx_gallery_start_ignore
# Explicitly state the expression "fig", to make it caught by sphinx_gallery.
# fig.set_size_inches(4, 1.75)
# fig.set_dpi(300)
fig
# sphinx_gallery_end_ignore

with reda.CreateEnterDirectory('output_plot_00_sensitivity'):
    grid.save_elem_file('elem.dat')
    grid.save_elec_file('elec.dat')

###############################################################################
# create the measurement configuration
configs = np.array((
    (1, 4, 10, 7),
))


###############################################################################
# for different background, plot the sensitivities
for bg in (1, 10, 100, 1000):
    td = crtomo.tdMan(grid=grid)
    td.configs.add_to_configs(configs)
    pid_mag, pid_pha = td.add_homogeneous_model(bg, 0)

    from shapely.geometry import Polygon # noqa:402
    poly = Polygon([
        [4, -1],
        [6, -1],
        [6, -2],
        [4, -2],
    ])
    td.parman.modify_polygon(pid_mag, poly, 50)

    poly = Polygon([
        [0, -1],
        [3, -1],
        [3, -2],
        [0, -2],
    ])
    td.parman.modify_polygon(pid_pha, poly, -150)

    td.model(sensitivities=True, silent=True)
    fig, ax = td.plot_sensitivity(0)
    # sphinx_gallery_start_ignore
    axes = fig.get_axes()
    axes[0].set_title("Magnitude Sensitivity", fontsize=8, loc='left')
    axes[1].set_title("Phase Sensitivity", fontsize=8, loc='left')
    for cb in [a._colorbar for a in axes[2:]]:
        cb.set_label("S [V/Ohm m]", fontsize="x-small")
    fig.set_size_inches(6, 2)
    fig.set_dpi(300)
    fig
    # sphinx_gallery_end_ignore
    with reda.CreateEnterDirectory('output_plot_00_sensitivity'):
        fig.savefig(
            'sensitivity_bg_{}.jpg'.format(bg),
            dpi=300,
            bbox_inches='tight'
        )

    fig, ax = td.plot_sensitivity(0, mag_only=True)
    # sphinx_gallery_start_ignore
    ax[0].set_title("Phase")
    fig.get_axes()[1]._colorbar.set_label("S [V/Ohm m]", fontsize="x-small")
    fig.set_size_inches(6, 2)
    fig.set_dpi(300)
    fig
    # sphinx_gallery_end_ignore
    with reda.CreateEnterDirectory('output_plot_00_sensitivity'):
        fig.savefig(
            'sensitivity_magonly_bg_{}.jpg'.format(bg),
            dpi=300,
            bbox_inches='tight'
        )
    # sphinx_gallery_start_ignore
    # sphinx_gallery_multi_image_block = "single"
    # sphinx_gallery_end_ignore
