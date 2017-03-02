#!/usr/bin/env python
"""Create depth-dependent grid decouplings.

WORK IN PROGRESS!
"""
from optparse import OptionParser

import numpy as np
from scipy.interpolate import UnivariateSpline

from crtomo.mpl_setup import *
import crtomo.grid as CRGrid


def handle_cmd_options():
    parser = OptionParser()
    parser.add_option('-e', "--elem", dest="elem_file", type="string",
                      help="elem.dat file (default: elem.dat)",
                      default="elem.dat")
    parser.add_option('-d', "--decfile", dest="dec_file", type="string",
                      help="elem.dat file (default: decfile.dat)",
                      default="decfile.dat")
    # parser.add_option("--eps", dest="eps", type="float",
    #                   help="User override for distance eps",
    #                   default=None)
    # parser.add_option('-l', "--linefile", dest="linefile",
    #                   help="Line file (default: extra_lines.dat)",
    #                   default='extra_lines.dat')
    parser.add_option("-o", "--output", dest="output",
                      help="Output file (default: decouplings.dat)",
                      metavar="FILE", default="decouplings.dat")

    (options, args) = parser.parse_args()
    return options


def load_decfile(decfile):
    decdata_raw = np.loadtxt(decfile)
    indices = np.argsort(decdata_raw[:, 0])
    decdata = decdata_raw[indices, :]

    return decdata


def get_univariate_spline(zn, dn):
    spl = UnivariateSpline(xn, dn)
    spl.set_smoothing_factor(0.1)
    return spl


def get_linear_fit(zn, dn):

    def func(z):
        if z <= zn.min():
            index = np.argmin(zn)
            return dn[index]
        if z >= zn.max():
            index = np.argmax(zn)
            return dn[index]
        if z in zn:
            index = np.where(z == zn)[0]
            return dn[index]

        diff = zn - z
        p = np.where(diff > 0)[0]
        pos_nr = np.argmin(zn[p])
        max_index = p[pos_nr]

        p = np.where(diff < 0)[0]
        neg_nr = np.argmax(zn[p])
        min_index = p[neg_nr]

        x = [zn[min_index], zn[max_index]]
        y = [dn[min_index], dn[max_index]]

        from scipy.interpolate import interp1d
        func = interp1d(x, y)

        return func(z)

    def array_func(z_array):
        result = [func(z) for z in z_array]
        return result

    return array_func


def get_func_decoupling_at_z(decfile, filename=None):
    decdata = load_decfile(decfile)

    # interpolate between the nearest z values
    # Use a spline interpolation to get a smooth curve
    zn = decdata[:, 0]
    dn = decdata[:, 1]

    # interp_func = get_univariate_spline(zn, dn)
    interp_func = get_linear_fit(zn, dn)

    # interp_func(-1.1)

    zs = np.linspace(zn.min(), zn.max(), 40)
    smooth_decouplings = interp_func(zs)

    if filename is not None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.plot(decdata[:, 1],
                decdata[:, 0],
                '.-',
                color='k')

        ax.plot(smooth_decouplings,
                zs,
                '.-',
                color='r')

        ax.set_xlabel('decoupling')
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylabel('depth z')
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
    return interp_func


def find_neighbors(grid, element_id):
    # find neighboring elements
    neighbors = []

    for index, elm1 in enumerate(grid.elements):
        indices = np.intersect1d(grid.elements[element_id], elm1)
        if len(indices) == 2:
            # append element index and the two adjoining nodes
            neighbors.append((
                index,
                indices[0],
                indices[1]
            ))
        # skip rest of the elements if we reached the maximum nr of possible
        # neighbors
        if len(neighbors) == 4:
            break
    return neighbors


def main():
    options = handle_cmd_options()
    decfunc = get_func_decoupling_at_z(
        options.dec_file,
        options.output + '.png'
    )
    grid = CRGrid.crt_grid()
    grid.load_elem_file(options.elem_file)

    nx = grid.nodes['sorted'][:, 1]
    ny = grid.nodes['sorted'][:, 2]
    nxy = np.vstack((nx, ny)).T

    decoupling_items = []
    for elmnr, element in enumerate(grid.elements):
        neighbors = find_neighbors(grid, elmnr)
        for neighbor in neighbors:
            xz = np.array([nxy[i] for i in neighbor[1:]])
            # we are only interested in z-coordinates
            z = xz[:, 1]

            if z.min() == z.max():
                # add decoupling
                decoupling = decfunc((z[0], ))[0]
                # print('decoupling', decoupling)
                decoupling_items.append((elmnr + 1,
                                         neighbor[0] + 1,
                                         decoupling))
            else:
                # print('vertical side')
                continue

    final_decouplings = np.array(decoupling_items)
    with open(options.output, 'w') as fid:
        fid.write('{0}\n'.format(final_decouplings.shape[0]))
        np.savetxt(fid, final_decouplings, fmt='%i %i %f')


if __name__ == '__main__':
    main()
