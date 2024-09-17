# *-* coding: utf-8 *-*
"""Manage node and element plots

"""
import numpy as np
import scipy.interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from crtomo.mpl_setup import *
import crtomo.mpl
import crtomo.grid as CRGrid
import crtomo.parManager as pM
import crtomo.nodeManager as nM
plt, mpl = crtomo.mpl.setup()


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
        ax : axes object, optional
            axes to plot to. If not provided, generate a new figure and axes
        nid : int
            node id pointing to the respective data set
        cmap : string, optional
            color map to use. Default: jet
        cbmin : float, optional
            Minimum colorbar value
        cbmax : float, optional
            Maximum colorbar value
        plot_colorbar : bool, optional
            if true, plot a colorbar next to the plot
        use_aspect: bool, optional
            plot grid in correct aspect ratio. Default: True
        fill_contours: bool, optional (true)
            If True, the fill the contours (contourf)

        """
        # if 'ax' not in kwargs:
        #     fig, ax = plt.subplots(1, 1)
        # else:
        #     ax = kwargs.get('ax')

        fig = ax.get_figure()
        x = self.grid.nodes['presort'][:, 1]
        z = self.grid.nodes['presort'][:, 2]
        xz = np.vstack((x, z)).T

        # generate grid
        X, Z = np.meshgrid(
            np.linspace(x.min(), x.max(), 1000),
            np.linspace(z.min(), z.max(), 1000),
        )

        values = np.array(self.nodeman.nodevals[nid])
        if kwargs.get('converter', None) is not None:
            values = kwargs['converter'](values)
        # linear
        # cubic
        cint = scipy.interpolate.griddata(
            xz,
            values,
            (X, Z),
            method='cubic',
            # method='linear',
            # method='nearest',
            fill_value=np.nan,
        )
        cint_ma = np.ma.masked_invalid(cint)

        cmap = mpl.cm.get_cmap('turbo', 1)
        if kwargs.get('fill_contours', True):
            pc = ax.contourf(
                X, Z, cint_ma,
                cmap=kwargs.get('cmap', 'jet'),
                vmin=kwargs.get('cbmin', None),
                vmax=kwargs.get('cbmax', None),
                levels=kwargs.get('cblevels', None)
            )
        else:
            pc = ax.contour(
                X, Z, cint_ma,
                cmap=cmap,
                vmin=kwargs.get('cbmin', None),
                vmax=kwargs.get('cbmax', None),
                levels=kwargs.get('cblevels', None),
                alpha=kwargs.get('alpha', 1.0),
            )

        # plot electrodes
        ax.scatter(
            self.grid.electrodes[:, 1],
            self.grid.electrodes[:, 2],
        )

        # pc = ax.pcolormesh(
        #     X, Z, cint_ma,
        #     vmin=-40,
        #     vmax=40,
        # )
        if kwargs.get('use_aspect', True):
            ax.set_aspect('equal')
        ax.set_xlabel(kwargs.get('xlabel', 'x [m]'))
        ax.set_ylabel(kwargs.get('zlabel', 'z [m]'))
        # if kwargs.get('plot_colorbar', False):
        #     fig = ax.get_figure()
        #     cb = fig.colorbar(pc, ax=ax)
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

            fig.add_axes(ax_cb)

            cb = fig.colorbar(
                pc,
                cax=ax_cb,
                orientation=cbposition,
                label=kwargs.get('cblabel', ''),
                ticks=mpl.ticker.MaxNLocator(kwargs.get('cbnrticks', 3)),
                format=kwargs.get('cbformat', None),
                extend='both',
            )

        if kwargs.get('plot_colorbar', False):
            return fig, ax, pc, cb
        return fig, ax, pc

    def plot_nodes_current_streamlines_to_ax(
            self, ax, cid, model_pid, **kwargs):
        """

        """
        # node locations
        x = self.grid.nodes['presort'][:, 1]
        z = self.grid.nodes['presort'][:, 2]
        xz = np.vstack((x, z)).T

        # generate a fine grid
        X, Z = np.meshgrid(
            np.linspace(x.min(), x.max(), 1000),
            np.linspace(z.min(), z.max(), 1000),
        )

        values = np.array(self.nodeman.nodevals[cid])

        # linear
        # cubic
        cint = scipy.interpolate.griddata(
            xz,
            values,
            (X, Z),
            method='cubic',
            # method='linear',
            # method='nearest',
            fill_value=np.nan,
        )
        cint_ma = np.ma.masked_invalid(cint)

        # now compute the gradients in both directions, using the fine
        # interpolation
        U, V = np.gradient(cint_ma)

        resistivity = self.parman.parsets[model_pid]
        res = scipy.interpolate.griddata(
            self.grid.get_element_centroids(),
            resistivity,
            (X, Z),
            method='cubic',
            # method='linear',
            # method='nearest',
            fill_value=np.nan,
        )

        jx = -U / res
        jz = -V / res
        # jx = U
        # jz = V

        # current = np.sqrt(U ** 2 + V ** 2)
        # start_points = np.array((
        #     (1.0, 0.0),
        #     (11.0, 0.0),
        # ))
        # print(start_points.shape)
        ax.streamplot(
            X,
            Z,
            jz,
            jx,
            density=kwargs.get('density', 2.0),
            linewidth=kwargs.get('linewidth', 1.0),
            minlength=kwargs.get('minlength', 0.5),
            broken_streamlines=kwargs.get('broken_streamlines', False),
            # start_points=start_points,
        )

        # ax.contour(
        #     X,
        #     Z,
        #     current,
        # )

        # pc = ax.contourf(
        #     X,
        #     Z,
        #     cint,
        #     N=10,
        # )
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
        if kwargs.get('use_aspect', True):
            ax.set_aspect('equal')
        # if kwargs.get('plot_colorbar', False):
        #     fig = ax.get_figure()
        #     cb = fig.colorbar(pc)
        #     return cb

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
        cid_alpha : int, optional
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
        plot_colorbar : bool, optional
            if true, plot a colorbar next to the plot
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
        aspect: 'auto'|'equal', optional default: 'equal'
            Aspect of the plot region
        cb_pad: float, optional
            Padding of colorbar. Defaults to 0.4

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

        if kwargs.get('converter', None) is not None:
            subdata = kwargs['converter'](subdata)

        # import IPython
        # IPython.embed()
        # color map
        cmap_name = kwargs.get('cmap_name', 'viridis')
        # cmap = mpl.colormaps[cmap_name].resampled(
        #     kwargs.get('cbsegments', 256)
        # ).copy()

        cmap = mpl.cm.get_cmap(
            cmap_name,
            kwargs.get('cbsegments', None)
        ).copy()
        over = kwargs.get('over', cmap(1.0))
        under = kwargs.get('under', cmap(0.0))

        bad = kwargs.get('bad', 'white')
        cmap.set_over(over)
        cmap.set_under(under)
        cmap.set_bad(bad)

        # normalize data
        data_min = kwargs.get('cbmin', np.nanmin(subdata))
        data_max = kwargs.get('cbmax', np.nanmax(subdata))
        if (data_min is not None and data_max is not None
                and data_min == data_max):
            data_min -= 1
            data_max += 1
        cnorm = mpl.colors.Normalize(vmin=data_min, vmax=data_max)
        scalarMap = mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap)
        fcolors = scalarMap.to_rgba(subdata)
        scalarMap.set_array(subdata)

        # if applicable, apply alpha values
        alpha_cid = kwargs.get('cid_alpha', None)
        if isinstance(alpha_cid, int):
            alpha = self.parman.parsets[alpha_cid]

            sens_log_threshold_upper = kwargs.get(
                'alpha_sens_threshold', 3
            )
            sens_log_threshold_lower = kwargs.get(
                'alpha_sens_threshold_lower', None
            )
            # make sure this data set is normalized between 0 and 1
            if np.nanmin(alpha) < 0 or np.nanmax(alpha) > 1:
                normcov = np.divide(
                    np.abs(alpha),
                    sens_log_threshold_upper
                )
                indices_upper = np.where(normcov > 1)

                if sens_log_threshold_lower is not None:
                    normcov_lower = np.divide(
                        np.abs(alpha),
                        sens_log_threshold_lower
                    )
                    indices_lower = np.where(normcov_lower <= 1)
                    # these will be fully transparent
                    normcov[indices_lower] = 0

                # these will be fully opaque
                normcov[indices_upper] = 1

                alpha = np.subtract(1, normcov)
                # raise Exception(
                #     'alpha data set must be normalized between 0 and 1'
                # )

            fcolors[:, 3] = alpha
        elif isinstance(alpha_cid, np.ndarray):
            alpha = alpha_cid
            fcolors[:, 3] = alpha

        all_xz = []
        for x, z in zip(self.grid.grid['x'], self.grid.grid['z']):
            tmp = np.vstack((x, z)).T
            all_xz.append(tmp)

        norm = kwargs.get('norm', None)

        # the actual plotting
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
                clip_on=False,
            )

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(zmin, zmax)
        ax.set_xlabel(kwargs.get('xlabel', 'x [m]'))
        ax.set_ylabel(kwargs.get('zlabel', 'z [m]'))
        ax.set_aspect(kwargs.get('aspect', 'equal'))
        ax.set_title(
            kwargs.get('title', ''),
            fontsize=7,
        )

        if kwargs.get('plot_colorbar', False):
            divider = make_axes_locatable(ax)
            cbposition = kwargs.get('cbposition', 'vertical')
            if cbposition == 'horizontal':
                ax_cb = divider.new_vertical(
                    size=0.1,
                    pad=kwargs.get('cb_pad', 0.4),
                    pack_start=True
                )
            elif cbposition == 'vertical':
                ax_cb = divider.new_horizontal(
                    size=0.1,
                    pad=kwargs.get('cb_pad', 0.4),
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


def converter_pm_log10(data, verbose_return=False):
    """Convert the given data to:

        log10(subdata) for subdata > 0
        log10(-subdata') for subdata' < 0
        0 for subdata'' == 0

    Parameters
    ----------
    data: array
        input data
    verbose_return : bool
        if True, then also return the indices for cells larger/smaller than
        zero

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
    # import IPython
    # IPython.embed()
    if verbose_return:
        return indices_gt_zero, indices_lt_zero, data_converted
    else:
        return data_converted


def converter_log10_to_lin(data):
    """Return 10 ** data"""
    return 10 ** data


def converter_abs_log10(data):
    """Return log10(abs(data))
    """
    return np.log10(np.abs(data))


def converter_change_sign(data):
    """Reverse the sign of the data. Useful for plotting phase values
    """
    return -data


def converter_asinh(data):
    norm = np.max(np.abs(data))
    dyn = np.abs(
        np.min(
            np.log10(
                np.abs(
                    data
                )
            )
        )
    )

    data_transformed = np.arcsinh(
        10 ** dyn * data / norm
    ) / np.arcsinh(
        10 ** dyn
    )
    return data_transformed


def converter_sensitivity(data):
    """

    """
    norm_value = np.abs(data).max()
    sens_normed = data / norm_value

    indices_gt_zero = data > 1e-5
    indices_lt_zero = data < -1e-5

    # map all values greater than zero to the range [0.5, 1]
    x = np.log10(sens_normed[indices_gt_zero])
    # log_norm_factor = np.abs(x).max()
    log_norm_factor = -5
    y1 = 1 - x / (2 * log_norm_factor)

    # map all values smaller than zero to the range [0, 0.5]
    x = np.log10(np.abs(sens_normed[indices_lt_zero]))
    y = x / (2 * log_norm_factor)

    y2 = np.abs(y)

    # reassign values
    data_transformed = np.ones_like(data) * 0.5
    data_transformed[indices_gt_zero] = y1
    data_transformed[indices_lt_zero] = y2
    return data_transformed
