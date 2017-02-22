#!/usr/bin/python
# *-* coding: utf-8 *-*
"""
Plot a wireframe of a given grid
END DOCUMENTATION
"""
import os
from optparse import OptionParser
import numpy as np
from crlab_py.mpl import *
from crlab_py import elem2

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
    parser.add_option('-e', "--elem", dest="elem_file", type="string",
                      help="elem.dat file (default: elem.dat)",
                      default="elem.dat")

    parser.add_option('-t', "--elec", dest="elec_file", type="string",
                      help="elec.dat file (default: elec.dat)",
                      default="elec.dat")

    parser.add_option("-o", "--output", dest="output",
                      help="Output file (default: grid.png)",
                      metavar="FILE", default="grid.png")

    parser.add_option("-m", "--mark_node", dest="mark_node",
                      help="Mark one node (index starts with 0)",
                      type="int",
                      metavar="NR", default=None)

    parser.add_option("-c", "--mark_cell", dest="mark_cell",
                      help="Mark one cell (index starts with 0)",
                      type="int",
                      metavar="NR", default=None)
    parser.add_option("--fancy", action="store_true", dest="plot_fancy",
                      help="Create a fancy plot (default:false)",
                      default=False)
    (options, args) = parser.parse_args()
    return options


def plot_wireframe(options):
    grid = elem2.crt_grid()
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
    collection = mpl.collections.PolyCollection(all_xz,
                                                edgecolor='k',
                                                facecolor='none',
                                                linewidth=cell_line_width,
                                                )
    ax.add_collection(collection)

    # plot electrodes
    ax.scatter(grid.electrodes[:, 1], grid.electrodes[:, 2],
               edgecolors='none', clip_on=False,
               s=elec_size)

    # mark nodes
    if options.mark_node is not None:
        xy = grid.nodes['sorted'][options.mark_node]
        ax.scatter(xy[1], xy[2], s=node_mark_size,
                   color='r', edgecolors='none')

    if options.mark_cell is not None:
        index = options.mark_cell
        x = grid.grid['x'][index]
        z = grid.grid['z'][index]
        for i in xrange(0, x.size):
            i1 = (i + 1) % x.size
            ax.plot(x[[i, i1]],
                    z[[i, i1]],
                    color='r',
                    linewidth=cell_mark_size)

    ax.autoscale_view()
    ax.set_aspect('equal')
    if options.plot_fancy:
        pass
    else:
        ax.axis('off')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)

    fig.savefig(options.output, dpi=dpi, bbox_inches='tight')


if __name__ == '__main__':
    options = handle_cmd_options()
    plot_wireframe(options)
