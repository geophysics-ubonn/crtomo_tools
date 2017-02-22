# *-* coding: utf-8 *-*
"""Manage node and element plots

TODO
----

* we need the nodeManager

"""
import numpy as np
import scipy.interpolate
import matplotlib as mpl

import crtomo.grid as CRGrid
import crtomo.parManager as pM
import crtomo.nodeManager as nM


class plotManager(object):
    """The :class:`plotManager` produces plots for a given grid. It uses
    :class:`crtomo.grid.crt_grid` to manage the grid, and
    :class:`crtomo.parManager` to manage element parameter values.
    :class:`crtomo.nodeManager` is used to manage node data.
    """

    # configuration options
    rcParams = {

    }

    def __init__(self, **kwargs):
        """initialize the plot manager. Multiple combinations are possible for
        the kwargs.

        Parameters
        ----------
        grid: crtomo.grid.crt_grid, optional
            a fully initialized grid object. If not provided, elem and elec
            files must be provided!
        elem_file: file path
            file path to an elem.dat FE grid file. If no grid is provided, a
            new crt_grid oject is initialized
        elec_file: file path
            file path to an elec.dat FE electrode file
        nm: crtomo.nodeManager.nodeMan instance, optional
            node manager for node data. If none is provided, an empty one is
            initialized
        pm: crtomo.parManager.parMan instance, optional
            parameter manager for element data. If none is provided, an empty
            one is initialized
        """
        # initialize grid
        grid = kwargs.get('grid', None)
        if grid is None:
            elem_file = kwargs.get('elem_file', None)
            if elem_file is None:
                raise Exception('No grid object and no elem path provided!')
            elec_file = kwargs.get('elec_file', None)
            grid = CRGrid.crt_grid()
            grid.load_grid(elem_file, elec_file)
        self.grid = grid

        # node manager
        nodeman = kwargs.get('nm', None)
        if nodeman is None:
            nodeman = nM.NodeMan(self.grid)
        self.nodeman = nodeman

        # par manager
        parman = kwargs.get('pm', None)
        if parman is None:
            parman = pM.ParMan(self.grid)
        self.parman = parman

    def plot_nodes_pcolor_to_ax(self, ax, cid, config):
        """

        """
        x = self.grid.nodes['presort'][:, 1]
        z = self.grid.nodes['presort'][:, 2]
        ax.scatter(x, z)
        xz = np.vstack((x, z)).T

        # generate grid
        X, Z = np.meshgrid(
            np.linspace(x.min(), x.max(), 100),
            np.linspace(z.min(), z.max(), 100),
        )

        values = np.array(self.nodeman.nodevals[cid])
        # linear
        # cubic
        cint = scipy.interpolate.griddata(
            xz,
            values,
            (X, Z),
            method='linear',
            # method='linear',
            # method='nearest',
            fill_value=np.nan,
        )
        cint_ma = np.ma.masked_invalid(cint)

        pc = ax.pcolormesh(
            X, Z, cint_ma,
            vmin=-40,
            vmax=40,
        )
        # cb = fig.colorbar(pc)
        return pc

    def plot_nodes_contour_to_ax(self, ax, cid, config):
        """

        """
        x = self.grid.nodes['presort'][:, 1]
        z = self.grid.nodes['presort'][:, 2]
        ax.scatter(x, z)
        xz = np.vstack((x, z)).T

        # generate grid
        X, Z = np.meshgrid(
            np.linspace(x.min(), x.max(), 100),
            np.linspace(z.min(), z.max(), 100),
        )

        values = np.array(self.nodeman.nodevals[cid])
        # linear
        # cubic
        cint = scipy.interpolate.griddata(
            xz,
            values,
            (X, Z),
            method='linear',
            # method='linear',
            # method='nearest',
            fill_value=np.nan,
        )
        cint_ma = np.ma.masked_invalid(cint)

        pc = ax.contourf(
            X, Z, cint_ma,
            vmin=-40,
            vmax=40,
        )
        # pc = ax.pcolormesh(
        #     X, Z, cint_ma,
        #     vmin=-40,
        #     vmax=40,
        # )
        # cb = fig.colorbar(pc)
        return pc

    def plot_nodes_streamlines_to_ax(self, ax, cid, config):
        """

        """
        x = self.grid.nodes['presort'][:, 1]
        z = self.grid.nodes['presort'][:, 2]
        # ax.scatter(x, z)
        xz = np.vstack((x, z)).T

        # generate grid
        X, Z = np.meshgrid(
            np.linspace(x.min(), x.max(), 100),
            np.linspace(z.min(), z.max(), 100),
        )

        values = np.array(self.nodeman.nodevals[cid])

        # linear
        # cubic
        cint = scipy.interpolate.griddata(
            xz,
            values,
            (X, Z),
            method='linear',
            # method='linear',
            # method='nearest',
            fill_value=np.nan,
        )
        cint_ma = np.ma.masked_invalid(cint)

        print(cint_ma.shape)
        U, V = np.gradient(cint_ma)

        current = np.sqrt(U ** 2 + V ** 2)
        start_points = np.array((
            (1.0, 0.0),
            (11.0, 0.0),
        ))
        print(start_points.shape)
        ax.streamplot(
            X,
            Z,
            V,
            U,
            density=2.0,
            minlength=0.5,
            # start_points=start_points,
        )

        ax.contour(
            X,
            Z,
            current,
        )
        ax.contour(
            X,
            Z,
            cint,
            N=10,
        )
        # pc = ax.pcolormesh(
        #     X, Z, current,
        #     # vmin=-40,
        #     # vmax=40,
        # )
        # pc
        # Q = ax.quiver(X, Z, U, V, units='width')
        # qk = ax.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
        #                   coordinates='figure')
        # Q, qk


#         pc = ax.pcolormesh(
#             X, Z, cint_ma,
#             vmin=-40,
#             vmax=40,
#         )
#         # cb = fig.colorbar(pc)
#         return pc

    def plot_elements_to_ax(self, ax, cid, config):
        # sl2 = 10 ** sl2_raw[:, 2]

        # alpha = np.ones(sl2.size)
        # alpha[np.where(sl2 < 1e-3)] = 0.0

        # convert data
        subdata = self.parman.parsets[cid]
        cmap = mpl.cm.get_cmap('jet')
        cnorm = mpl.colors.Normalize(vmin=subdata.min(), vmax=subdata.max())
        scalarMap = mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap)

        fcolors = scalarMap.to_rgba(subdata)
        # fcolors[:, 3] = alpha

        all_xz = []
        for x, z in zip(self.grid.grid['x'], self.grid.grid['z']):
            tmp = np.vstack((x, z)).T
            all_xz.append(tmp)

        # ecolors = np.zeros((len(all_xz), 4))
        # ecolors[:, 0] = 0.5
        # ecolors[:, 3] = 1.0

        # fcolors = np.zeros((len(all_xz), 4))
        # fcolors[:, 0] = 0.5
        # fcolors[:, 3] = 1.0

        # fcolors[-100:, 3] = 0.5

        collection = mpl.collections.PolyCollection(
            all_xz,
            edgecolor=fcolors,
            facecolor=fcolors,
            linewidth=0.4,
        )
        ax.add_collection(collection)
        if self.grid.electrodes is not None:
            ax.scatter(
                self.grid.electrodes[:, 1],
                self.grid.electrodes[:, 2],
                color=self.grid.props['electrode_color'],
                clip_on=False,
            )
        ax.set_xlim(self.grid.grid['x'].min(), self.grid.grid['x'].max())
        ax.set_ylim(self.grid.grid['z'].min(), self.grid.grid['z'].max())
        # ax.autoscale_view()
        ax.set_aspect('equal')
