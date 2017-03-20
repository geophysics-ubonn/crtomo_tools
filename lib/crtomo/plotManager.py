# *-* coding: utf-8 *-*
"""Manage node and element plots

"""
import numpy as np
import scipy.interpolate

from crtomo.mpl_setup import *
import crtomo.grid as CRGrid
import crtomo.parManager as pM
import crtomo.nodeManager as nM


class plotManager(object):
    """The :class:`plotManager` produces plots for a given grid. It uses
    :class:`crtomo.grid.crt_grid` to manage the grid, and
    :class:`crtomo.parManager` to manage element parameter values.
    :class:`crtomo.nodeManager` is used to manage node data.
    """

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
        self.parman = kwargs.get('pm', pM.ParMan(self.grid))

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

    def plot_elements_to_ax(self, cid, ax=None, **kwargs):
        """Plot element data (parameter sets).

        Parameters
        ----------
        cid
        ax
        alpha_cid
        xmin
        xmax
        zmin
        zmax
        cmap_name
        cblabel
        plot_colorbar: bool

        Returns
        -------
        fig:

        ax:

        cnorm:

        cmap:

        cb: colorbar instance, optional
            only of plot_colorbar is True
        """

        xmin = kwargs.get('xmin', self.grid.grid['x'].min())
        xmax = kwargs.get('xmax', self.grid.grid['x'].max())
        zmin = kwargs.get('zmin', self.grid.grid['z'].min())
        zmax = kwargs.get('zmax', self.grid.grid['z'].max())

        # try to create a suitable default figure size
        if ax is None:
            # 15 cm
            sizex = 15 / 2.54
            sizez = sizex * (np.abs(zmax - zmin) / np.abs(xmax - xmin) * 1.1)
            fig, ax = plt.subplots(figsize=(sizex, sizez))
        else:
            fig = ax.get_figure()

        # get data
        subdata = self.parman.parsets[cid]
        if 'converter' in kwargs:
            subdata = kwargs['converter'](subdata)

        # color map
        cmap_name = kwargs.get('cmap_name', 'jet')
        cmap = mpl.cm.get_cmap(cmap_name)

        # normalize data
        cnorm = mpl.colors.Normalize(vmin=subdata.min(), vmax=subdata.max())
        scalarMap = mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap)
        fcolors = scalarMap.to_rgba(subdata)

        # if applicable, apply alpha values
        alpha_cid = kwargs.get('cid_alpha', None)
        print('alpha_cid', alpha_cid)
        if isinstance(alpha_cid, int):
            print('applying alpha')
            alpha = self.parman.parsets[alpha_cid]
            print(alpha)
            # make sure this data set is normalized between 0 and 1
            if np.nanmin(alpha) < 0 or np.nanmax(alpha) > 1:
                raise Exception(
                    'alpha data set must be normalized between 0 and 1'
                )
            fcolors[:, 3] = alpha

        all_xz = []
        for x, z in zip(self.grid.grid['x'], self.grid.grid['z']):
            tmp = np.vstack((x, z)).T
            all_xz.append(tmp)

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

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(zmin, zmax)
        ax.set_xlabel(kwargs.get('xlabel', 'x'))
        ax.set_ylabel(kwargs.get('zlabel', 'z'))
        ax.set_aspect('equal')

        if kwargs.get('plot_colorbar', False):
            print('colorbar')
            cb_boundaries = mpl_get_cb_bound_below_plot(ax)
            cax = fig.add_axes(cb_boundaries, frame_on=True)
            cb = self._return_colorbar(
                cax,
                cnorm,
                cmap,
                label=kwargs.get('cblabel', ''),
            )
            return fig, ax, cnorm, cmap, cb

        return fig, ax, cnorm, cmap

    def _return_colorbar(self, cax, norm, cmap, label=''):
        """plot a colorbar to the provided axis
        """
        cb = mpl.colorbar.ColorbarBase(
            ax=cax,
            cmap=cmap,
            norm=norm,
            orientation='horizontal',
        )
        cb.set_label(label)
        return cb


def converter_pm_log10(data):
    """Convert the given data to:

        log10(subdata) for subdata > 0
        log10(-subdata') for subdata' < 0
        0 for subdata'' == 0

    Parameters
    ----------
    data: array
        input data

    Returns
    -------
    array_converted: array
        converted data

    """
    # indices_zero = np.where(data == 0)
    indices_gt_zero = np.where(data > 0)
    indices_lt_zero = np.where(data < 0)

    data_converted = np.zeros(data.shape)
    data_converted[indices_gt_zero] = np.log10(data[indices_gt_zero])
    data_converted[indices_lt_zero] = np.log10(-data[indices_lt_zero])
    return data_converted
