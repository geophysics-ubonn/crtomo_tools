# *-* coding: utf-8 *-*
"""Manage node and element plots

"""
import numpy as np
import scipy.interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

    def plot_nodes_pcolor_to_ax(self, ax, nid, **kwargs):
        """Plot node data to an axes object

        Parameters
        ----------
        ax : axes object
            axes to plot to
        nid : int
            node id pointing to the respective data set
        cmap : string, optional
            color map to use. Default: jet
        vmin : float, optional
            Minimum colorbar value
        vmax : float, optional
            Maximum colorbar value

        Returns
        -------

        """
        fig = ax.get_figure()
        x = self.grid.nodes['presort'][:, 1]
        z = self.grid.nodes['presort'][:, 2]
        ax.scatter(x, z)
        xz = np.vstack((x, z)).T

        # generate grid
        X, Z = np.meshgrid(
            np.linspace(x.min(), x.max(), 100),
            np.linspace(z.min(), z.max(), 100),
        )

        values = np.array(self.nodeman.nodevals[nid])
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
            X, Z,
            cint_ma,
            cmap=kwargs.get('cmap', 'jet'),
            vmin=kwargs.get('vmin', None),
            vmax=kwargs.get('vmax', None),
        )
        if kwargs.get('plot_colorbar', False):
            divider = make_axes_locatable(ax)
            cbposition = kwargs.get('cbposition', 'vertical')
            if cbposition == 'horizontal':
                ax_cb = divider.new_vertical(
                    size=0.1, pad=0.4, pack_start=True
                )
            elif cbposition == 'vertical':
                ax_cb = divider.new_horizontal(
                    size=0.1, pad=0.4,
                )
            else:
                raise Exception('cbposition not recognized')

            ax.get_figure().add_axes(ax_cb)

            cb = fig.colorbar(
                pc,
                cax=ax_cb,
                orientation=cbposition,
                label=kwargs.get('cblabel', ''),
                ticks=mpl.ticker.MaxNLocator(kwargs.get('cbnrticks', 3)),
                format=kwargs.get('cbformat', None),
                extend='both',
            )

        no_elecs = kwargs.get('no_elecs', False)
        if self.grid.electrodes is not None and no_elecs is not True:
            ax.scatter(
                self.grid.electrodes[:, 1],
                self.grid.electrodes[:, 2],
                color=self.grid.props['electrode_color'],
                # clip_on=False,
            )

            return fig, ax, pc, cb
        return fig, ax, pc

    def plot_nodes_contour_to_ax(self, ax, nid, **kwargs):
        """Plot node data to an axes object

        Parameters
        ----------
        ax : axes object
            axes to plot to
        nid : int
            node id pointing to the respective data set
        cmap : string, optional
            color map to use. Default: jet
        vmin : float, optional
            Minimum colorbar value
        vmax : float, optional
            Maximum colorbar value

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

        values = np.array(self.nodeman.nodevals[nid])
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
            cmap=kwargs.get('cmap', 'jet'),
            vmin=kwargs.get('vmin', None),
            vmax=kwargs.get('vmax', None),
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

        If the parameter *ax* is not set, then a new figure will be created
        with a corresponding axes.


        Parameters
        ----------
        cid : int or :py:class:`numpy.ndarray`
            if *cid* is an int, then treat it as the id of the parameter set
            stored in self.parman. Otherwise, expect it to be the data to plot.
            At the moment no checks are made that the data fits the grid.
        ax : matplotlib.Axes, optional
            plot to this axes object, if provided
        alpha_cid : int, optional
            if given, use the corresponding dataset in self.parman as the alpha
            channel. No checks are made if all values of this data set lie
            between 0 and 1 (0 being fully transparent, and 1 being opaque).
        xmin : float, optional
            minimal x limit to plot
        xmax : float, optional
            maximal x limit to plot
        zmin : float, optional
            minimal z limit to plot
        zmax : float, optional
            maximial z limit to plot
        converter : function, optional
            if given, then use this function to convert the data into another
            representation. The given function must work with a numpy array.
            Default: None
        norm : norm object, optional
            the norm object for matplotlib plotting can be provided here
        cmap_name : string, optional
            name of the colorbar to use. Default is "viridis". To reverse
            colors, use the _r version "viridis_r"
        cbposition : ?
            ?
        cblabel : string, optional
            colorbar label
        cbsegments : int, optional
            ?
        cbnrticks : int, optional
            ?
        over : color, optional
            color to use for values above the current cb-limit. Default: ?
        under :
            color to use for values below the current cb-limit. Default: ?
        bad :
            color to use for nan-values. Default: ?
        plot_colorbar : bool, optional
            if true, plot a colorbar next to the plot
        title : string, optional
            plot title string
        xlabel : string, optional
            Set xlabel of the resulting plot
        ylabel : string, optional
            Set ylabel of the resulting plot
        no_elecs : bool, optional
            If True, plot no electrodes
        rasterize: bool, optional
            if True, rasterize the plot. Default: False

        Returns
        -------
        fig:

        ax:

        cnorm:

        cmap:

        cb: colorbar instance, optional
            only of plot_colorbar is True
        scalarMap:
            use to create custom colorbars

        """

        rasterize = kwargs.get('rasterize', False)
        xmin = kwargs.get('xmin', self.grid.grid['x'].min())
        xmax = kwargs.get('xmax', self.grid.grid['x'].max())
        zmin = kwargs.get('zmin', self.grid.grid['z'].min())
        zmax = kwargs.get('zmax', self.grid.grid['z'].max())

        # try to create a suitable default figure size
        if ax is None:
            # 15 cm
            sizex = 15 / 2.54
            sizez = sizex * (np.abs(zmax - zmin) / np.abs(xmax - xmin) * 1.1)
            # add 1 inch to accommodate colorbar
            sizez += 1.3
            fig, ax = plt.subplots(figsize=(sizex, sizez))
        else:
            fig = ax.get_figure()
            sizex, sizez = fig.get_size_inches()

        # get data
        if isinstance(cid, int):
            subdata = self.parman.parsets[cid]
        else:
            subdata = cid

        if 'converter' in kwargs:
            subdata = kwargs['converter'](subdata)

        # color map
        cmap_name = kwargs.get('cmap_name', 'viridis')
        cmap = mpl.cm.get_cmap(
            cmap_name,
            kwargs.get('cbsegments', None)
        )
        over = kwargs.get('over', 'orange')
        under = kwargs.get('under', 'mediumblue')
        bad = kwargs.get('bad', 'white')
        cmap.set_over(over)
        cmap.set_under(under)
        cmap.set_bad(bad)

        # normalize data
        data_min = kwargs.get('cbmin', subdata.min())
        data_max = kwargs.get('cbmax', subdata.max())
        if(data_min is not None and data_max is not None and
           data_min == data_max):
            data_min -= 1
            data_max += 1
        cnorm = mpl.colors.Normalize(vmin=data_min, vmax=data_max)
        scalarMap = mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap)
        fcolors = scalarMap.to_rgba(subdata)
        scalarMap.set_array(subdata)

        # if applicable, apply alpha values
        alpha_cid = kwargs.get('cid_alpha', None)
        if isinstance(alpha_cid, int):
            print('applying alpha')
            alpha = self.parman.parsets[alpha_cid]
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

        norm = kwargs.get('norm', None)

        collection = mpl.collections.PolyCollection(
            all_xz,
            edgecolor=fcolors,
            facecolor=fcolors,
            linewidth=0.0,
            cmap=cmap,
            norm=norm,
            rasterized=rasterize,
        )
        collection.set_cmap(cmap)
        ax.add_collection(collection)
        no_elecs = kwargs.get('no_elecs', False)
        if self.grid.electrodes is not None and no_elecs is not True:
            ax.scatter(
                self.grid.electrodes[:, 1],
                self.grid.electrodes[:, 2],
                color=self.grid.props['electrode_color'],
                # clip_on=False,
            )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(zmin, zmax)
        ax.set_xlabel(kwargs.get('xlabel', 'x'))
        ax.set_ylabel(kwargs.get('zlabel', 'z'))
        ax.set_aspect('equal')
        ax.set_title(
            kwargs.get('title', '')
        )

        if kwargs.get('plot_colorbar', False):
            divider = make_axes_locatable(ax)
            cbposition = kwargs.get('cbposition', 'vertical')
            if cbposition == 'horizontal':
                ax_cb = divider.new_vertical(
                    size=0.1, pad=0.4, pack_start=True
                )
            elif cbposition == 'vertical':
                ax_cb = divider.new_horizontal(
                    size=0.1, pad=0.4,
                )
            else:
                raise Exception('cbposition not recognized')

            ax.get_figure().add_axes(ax_cb)

            cb = fig.colorbar(
                scalarMap,
                cax=ax_cb,
                orientation=cbposition,
                label=kwargs.get('cblabel', ''),
                ticks=mpl.ticker.MaxNLocator(kwargs.get('cbnrticks', 3)),
                format=kwargs.get('cbformat', None),
                extend='both',
            )

            return fig, ax, cnorm, cmap, cb, scalarMap

        return fig, ax, cnorm, cmap, scalarMap


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
    data_converted[indices_lt_zero] = -np.log10(-data[indices_lt_zero])
    return indices_gt_zero, indices_lt_zero, data_converted


def converter_abs_log10(data):
    """Return log10(abs(data))
    """
    return np.log10(np.abs(data))
