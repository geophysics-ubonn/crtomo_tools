#!/usr/bin/env python
# *-* coding: utf-8 *-*
"""Plot a wireframe of a given grid
"""
import os
from optparse import OptionParser
import numpy as np
from crtomo.mpl_setup import *
import crtomo.grid as CRGrid

# # environment variables
#
node_mark_size = float(os.environ.get('MARK_SIZE', 1.0))
# line width
cell_mark_size = float(os.environ.get('MARK_WIDTH', 1.0))
cell_line_width = float(os.environ.get('CELL_WIDTH', 1.0))
# electrode size
elec_size = float(os.environ.get('ELEC_SIZE', 30.0))
dpi = int(os.environ.get('DPI', 300))


def handle_cmd_options():
    parser = OptionParser()
    parser.add_option(
        '-e', "--elem",
        dest="elem_file",
        type="string",
        help="elem.dat file (default: elem.dat)",
        default="elem.dat",
    )

    parser.add_option(
        '-t', "--elec",
        dest="elec_file",
        type="string",
        help="elec.dat file (default: elec.dat)",
        default="elec.dat",
    )

    parser.add_option(
        '-d', "--decouplings",
        dest="decoupling_file",
        type="string",
        help='decouplings file (default: ../exe/decouplings.dat)',
        default="../exe/decouplings.dat",
    )

    parser.add_option(
        "-o", "--output",
        dest="output",
        help="Output file (default: grid.png)",
        metavar="FILE",
        default="grid.png",
    )

    parser.add_option(
        "-m", "--mark_node",
        dest="mark_node",
        help="Mark one node (index starts with 0)",
        type="int",
        metavar="NR",
        default=None,
    )

    parser.add_option(
        "-c", "--mark_cell",
        dest="mark_cell",
        help="Mark one cell (index starts with 0)",
        type="int",
        metavar="NR",
        default=None,
    )
    parser.add_option(
        "--fancy",
        action="store_true",
        dest="plot_fancy",
        help="Create a fancy plot (default: True)",
        default=True,
    )
    parser.add_option(
        '-n',
        "--plot_elec_nr",
        action="store_true",
        dest="plot_elec_nr",
        help="Plot electrode numbers next to the electrodes",
        default=False,
    )
    (options, args) = parser.parse_args()
    return options


def plot_wireframe(options):
    grid = CRGrid.crt_grid()
    grid.load_elem_file(options.elem_file)
    grid.load_elec_file(options.elec_file)

    xmin = grid.grid['x'].min()
    xmax = grid.grid['x'].max()
    zmin = grid.grid['z'].min()
    zmax = grid.grid['z'].max()

    fig, ax = plt.subplots(1, 1, frameon=False)
    all_xz = []
    for x, z in zip(grid.grid['x'], grid.grid['z']):
        tmp = np.vstack((x, z)).T
        all_xz.append(tmp)
    collection = mpl.collections.PolyCollection(
        all_xz,
        edgecolor='k',
        facecolor='none',
        linewidth=cell_line_width,
    )
    ax.add_collection(collection)

    # plot electrodes
    ax.scatter(
        grid.electrodes[:, 1],
        grid.electrodes[:, 2],
        edgecolors='none',
        clip_on=False,
        label='electrodes',
        s=elec_size
    )

    if options.plot_elec_nr:
        for nr, xy in enumerate(grid.electrodes[:, 1:3]):
            ax.text(
                xy[0], xy[1],
                '{}'.format(nr + 1),
                bbox=dict(boxstyle='circle', facecolor='red', alpha=0.8)
            )

    # mark nodes
    if options.mark_node is not None:
        xy = grid.nodes['sorted'][options.mark_node]
        ax.scatter(
            xy[1],
            xy[2],
            s=node_mark_size,
            color='r',
            edgecolors='none',
            label='marked node',
        )

    if options.mark_cell is not None:
        index = options.mark_cell
        x = grid.grid['x'][index]
        z = grid.grid['z'][index]
        label = 'marked cell'
        for i in range(0, x.size):
            i1 = (i + 1) % x.size
            ax.plot(
                x[[i, i1]],
                z[[i, i1]],
                color='r',
                linewidth=cell_mark_size,
                label=label,
            )
            label = ''

        polygon = mpl.patches.Polygon(
            [(a, b) for a, b in zip(x, z)],
            closed=True,
            color='r',
            alpha=1.0,
        )
        ax.add_patch(polygon)

    if os.path.isfile(options.decoupling_file):
        decouplings = np.loadtxt(options.decoupling_file, skiprows=1)
        # plot decouplings
        label = 'decoupling line'
        for (el1, el2, coef) in decouplings:
            n1 = grid.elements[int(el1) - 1]
            n2 = grid.elements[int(el2) - 1]

            ints = np.intersect1d(n1, n2)

            x = grid.nodes['presort'][ints, 1]
            z = grid.nodes['presort'][ints, 2]

            ax.plot(
                x,
                z,
                '.-',
                color='b',
                linestyle='dashed',
                label=label,
            )
            label = ''

    ax.autoscale_view()
    ax.set_aspect('equal')
    if options.plot_fancy:
        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        ax.legend(
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0, 0, 1, 1),
            bbox_transform=fig.transFigure
        )

    else:
        ax.axis('off')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)

    fig.savefig(options.output, dpi=dpi, bbox_inches='tight')


def main():
    options = handle_cmd_options()
    plot_wireframe(options)
