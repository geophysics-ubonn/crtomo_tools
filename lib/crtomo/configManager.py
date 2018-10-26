# *-* coding: utf-8 *-*
"""Manage measurement configurations and corresponding measurements.

Geometric Factors
-----------------

.. plot:: pyplots/plot_K_vs_dipol_sep.py

Literature
----------

* Noel, Mark and Biwen Xu; Archaeological investigation by electrical
    resistivity tomography: a preliminary study. Geophys J Int 1991; 107 (1):
    95-102. doi: 10.1111/j.1365-246X.1991.tb01159.x
* Stummer, Peter, Hansruedi Maurer, and Alan G. Green. “Experimental Design:
  Electrical Resistivity Data Sets That Provide Optimum Subsurface
  Information.” Geophysics 69, no. 1 (January 1, 2004): 120–120.
  doi:10.1190/1.1649381.
* Roy, A., and A. Apparao. “DEPTH OF INVESTIGATION IN DIRECT CURRENT METHODS.”
  GEOPHYSICS 36, no. 5 (January 1, 1971): 943–59. doi:10.1190/1.1440226.

"""
import itertools

import scipy.interpolate as si
import numpy as np

import crtomo.mpl
plt, mpl = crtomo.mpl.setup()
import reda.utils.geometric_factors as edfK
import reda.utils.filter_config_types as fT


class ConfigManager(object):
    """The class`ConfigManager` manages four-point measurement configurations.
    Measurements can be saved/loaded from CRMod/CRTomo files, and new
    configurations can be created.
    """

    def __init__(self, nr_of_electrodes=None):
        # store the configs as a Nx4 numpy array
        self.configs = None
        # store measurements in a list of size N arrays
        self.measurements = {}
        # each measurement can store additional data here
        self.metadata = {}
        # global counter for measurements
        self.meas_counter = - 1
        # number of electrodes
        self.nr_electrodes = nr_of_electrodes

    def _get_next_index(self):
        self.meas_counter += 1
        return self.meas_counter

    def clear_measurements(self):
        """Remove all measurements from self.measurements. Reset the
        measurement counter. All ID are invalidated.
        """
        keys = list(self.measurements.keys())
        for key in keys:
            del(self.measurements[key])
        self.meas_counter = -1

    def clear_configs(self):
        """Remove all configs. This implies deleting all measurements.
        """
        self.clear_measurements()
        del(self.configs)
        self.configs = None

    def delete_measurements(self, mid):
        del(self.measurements[mid])

    @property
    def nr_of_configs(self):
        """Return number of configurations

        Returns
        -------
        nr_of_configs: int
            number of configurations stored in this instance

        """
        if self.configs is None:
            return 0
        else:
            return self.configs.shape[0]

    def add_noise(self, cid, **kwargs):
        """Add noise to a data set and return a new cid for the noised data.

        Parameters
        ----------
        cid: int
            ID for the data set to add noise to
        positive: bool
            if True, then set measurements to np.nan that are negative
        seed: int, optional
            set the seed used to initialize the random number generator
        relative: float
            standard deviation of error
        absolute: float
            mean value of normal distribution


        Returns
        -------
        cid_noise: int
            ID pointing to noised data set

        Note
        ----

        This function is a stub at the moment and has NO functionality.

        """
        pass

    def add_measurements(self, measurements):
        """Add new measurements to this instance

        Parameters
        ----------
        measurements: numpy.ndarray
            one or more measurement sets. It must either be 1D or 2D, with the
            first dimension the number of measurement sets (K), and the second
            the number of measurements (N): K x N

        Returns
        -------
        mid: int
            measurement ID used to extract the measurements later on

        Examples
        --------
        >>> import numpy as np
            import crtomo.configManager as CRconfig
            config = CRconfig.ConfigManager(nr_of_electrodes=10)
            config.gen_dipole_dipole(skipc=0)
            # generate some random noise
            random_measurements = np.random.random(config.nr_of_configs)
            mid = config.add_measurements(random_measurements)
            # retrieve using mid
            print(config.measurements[mid])

        """
        subdata = np.atleast_2d(measurements)

        if self.configs is None:
            raise Exception(
                'must read in configuration before measurements can be stored'
            )

        # we try to accommodate transposed input
        if subdata.shape[1] != self.configs.shape[0]:
            if subdata.shape[0] == self.configs.shape[0]:
                subdata = subdata.T
            else:
                raise Exception(
                    'Number of measurements does not match number of configs'
                )

        return_ids = []
        for dataset in subdata:
            cid = self._get_next_index()
            self.measurements[cid] = dataset.copy()
            return_ids.append(cid)

        if len(return_ids) == 1:
            return return_ids[0]
        else:
            return return_ids

    def _crmod_to_abmn(self, configs):
        """convert crmod-style configurations to a Nx4 array

        CRMod-style configurations merge A and B, and M and N, electrode
        numbers into one large integer each:

        .. math ::

            AB = A \cdot 10^4 + B

            MN = M \cdot 10^4 + N

        Parameters
        ----------
        configs: numpy.ndarray
            Nx2 array holding the configurations to convert

        Examples
        --------

        >>> import numpy as np
            import crtomo.configManager as CRconfig
            config = CRconfig.ConfigManager(nr_of_electrodes=5)
            # generate some CRMod-style configurations
            crmod_configs = np.array((
                (10002, 40003),
                (10010, 30004),
            ))
            abmn = config._crmod_to_abmn(crmod_configs)
            print(abmn)
        [[  2.   1.   3.   4.]
         [ 10.   1.   4.   3.]]

        """
        A = configs[:, 0] % 1e4
        B = np.floor(configs[:, 0] / 1e4).astype(int)
        M = configs[:, 1] % 1e4
        N = np.floor(configs[:, 1] / 1e4).astype(int)
        ABMN = np.hstack((
            A[:, np.newaxis],
            B[:, np.newaxis],
            M[:, np.newaxis],
            N[:, np.newaxis]
        ))
        return ABMN

    def load_configs(self, filename):
        """Load configurations from a file with four columns a b m n

        Parameters
        ----------
        filename: string
            absolute or relative path to a config file with four columns
        """
        configs = np.loadtxt(filename)
        self.add_to_configs(configs)

    def load_crmod_config(self, filename):
        """Load a CRMod configuration file

        Parameters
        ----------
        filename: string
            absolute or relative path to a crmod config.dat file

        """
        with open(filename, 'r') as fid:
            nr_of_configs = int(fid.readline().strip())
            configs = np.loadtxt(fid)
            print('loaded configs:', configs.shape)
            if nr_of_configs != configs.shape[0]:
                raise Exception(
                    'indicated number of measurements does not equal ' +
                    'to actual number of measurements'
                )
            ABMN = self._crmod_to_abmn(configs[:, 0:2])
            self.configs = ABMN

    def load_crmod_volt(self, filename):
        """Load a CRMod measurement file (commonly called volt.dat)

        Parameters
        ----------
        filename: string
            path to filename

        Returns
        -------
        list
            list of measurement ids
        """
        with open(filename, 'r') as fid:
            nr_of_configs = int(fid.readline().strip())
            measurements = np.loadtxt(fid)
            if nr_of_configs != measurements.shape[0]:
                raise Exception(
                    'indicated number of measurements does not equal ' +
                    'to actual number of measurements'
                )
        ABMN = self._crmod_to_abmn(measurements[:, 0:2])
        if self.configs is None:
            self.configs = ABMN
        else:
            # check that configs match
            if not np.all(ABMN == self.configs):
                raise Exception(
                    'previously stored configurations do not match new ' +
                    'configurations'
                )

        # add data
        cid_mag = self.add_measurements(measurements[:, 2])
        cid_pha = self.add_measurements(measurements[:, 3])
        return [cid_mag, cid_pha]

    def _get_crmod_abmn(self):
        """return a Nx2 array with the measurement configurations formatted
        CRTomo style
        """
        ABMN = np.vstack((
            self.configs[:, 0] * 1e4 + self.configs[:, 1],
            self.configs[:, 2] * 1e4 + self.configs[:, 3],
        )).T.astype(int)
        return ABMN

    def write_crmod_volt(self, filename, mid):
        """Write the measurements to the output file in the volt.dat file
        format that can be read by CRTomo.

        Parameters
        ----------
        filename: string
            output filename
        mid: int or [int, int]
            measurement ids of magnitude and phase measurements. If only one ID
            is given, then the phase column is filled with zeros

        """
        ABMN = self._get_crmod_abmn()

        if isinstance(mid, (list, tuple)):
            mag_data = self.measurements[mid[0]]
            pha_data = self.measurements[mid[1]]
        else:
            mag_data = self.measurements[mid]
            pha_data = np.zeros(mag_data.shape)

        all_data = np.hstack((
            ABMN,
            mag_data[:, np.newaxis],
            pha_data[:, np.newaxis]
        ))

        with open(filename, 'wb') as fid:
            fid.write(
                bytes(
                    '{0}\n'.format(ABMN.shape[0]),
                    'utf-8',
                )
            )
            np.savetxt(fid, all_data, fmt='%i %i %f %f')

    def write_crmod_config(self, filename):
        """Write the configurations to a configuration file in the CRMod format
        All configurations are merged into one previor to writing to file

        Parameters
        ----------
        filename: string
            absolute or relative path to output filename (usually config.dat)
        """
        ABMN = self._get_crmod_abmn()

        with open(filename, 'wb') as fid:
            fid.write(
                bytes(
                    '{0}\n'.format(ABMN.shape[0]),
                    'utf-8',
                )
            )
            np.savetxt(fid, ABMN.astype(int), fmt='%i %i')

    def gen_dipole_dipole(
            self, skipc, skipv=None, stepc=1, stepv=1, nr_voltage_dipoles=10,
            before_current=False, start_skip=0, N=None):
        """Generate dipole-dipole configurations

        Parameters
        ----------
        skipc: int
            number of electrode positions that are skipped between electrodes
            of a given dipole
        skipv: int
            steplength between subsequent voltage dipoles. A steplength of 0
            will produce increments by one, i.e., 3-4, 4-5, 5-6 ...
        stepc: int
            steplength between subsequent current dipoles. A steplength of 0
            will produce increments by one, i.e., 3-4, 4-5, 5-6 ...
        stepv: int
            steplength between subsequent voltage dipoles. A steplength of 0
            will produce increments by one, i.e., 3-4, 4-5, 5-6 ...
        nr_voltage_dipoles: int
            the number of voltage dipoles to generate for each current
            injection dipole
        before_current: bool, optional
            if set to True, also generate voltage dipoles in front of current
            dipoles.
        start_skip: int, optional
            how many electrode to skip before/after the first/second current
            electrode.
        N: int, optional
            number of electrodes, must be given if not already known by the
            config instance

        Examples
        --------

        .. plot::
            :include-source:

            import crtomo.configManager as CRconfig
            config = CRconfig.ConfigManager(nr_of_electrodes=10)
            config.gen_dipole_dipole(skipc=2)
            config.plot_pseudodepths()

        """
        if N is None and self.nr_electrodes is None:
            raise Exception('You must provide the number of electrodes')
        elif N is None:
            N = self.nr_electrodes

        # by default, current voltage dipoles have the same size
        if skipv is None:
            skipv = skipc

        configs = []
        # current dipoles
        for a in range(0, N - skipv - skipc - 3, stepc):
            b = a + skipc + 1
            nr = 0
            # potential dipoles before current injection
            if before_current:
                for n in range(a - start_skip - 1, -1, -stepv):
                    nr += 1
                    if nr > nr_voltage_dipoles:
                        continue
                    m = n - skipv - 1
                    if m < 0:
                        continue
                    quadpole = np.array((a, b, m, n)) + 1
                    configs.append(quadpole)

            # potential dipoles after current injection
            nr = 0
            for m in range(b + start_skip + 1, N - skipv - 1, stepv):
                nr += 1
                if nr > nr_voltage_dipoles:
                    continue
                n = m + skipv + 1
                quadpole = np.array((a, b, m, n)) + 1
                configs.append(quadpole)

        configs = np.array(configs)
        # now add to the instance
        if self.configs is None:
            self.configs = configs
        else:
            self.configs = np.vstack((self.configs, configs))
        return configs

    def _pseudodepths_wenner(self, configs, spacing=1, grid=None):
        """Given distances between electrodes, compute Wenner pseudo
        depths for the provided configuration

        The pseudodepth is computed after Roy & Apparao, 1971, as 0.11 times
        the distance between the two outermost electrodes. It's not really
        clear why the Wenner depths are different from the Dipole-Dipole
        depths, given the fact that Wenner configurations are a complete subset
        of the Dipole-Dipole configurations.

        """
        if grid is None:
            xpositions = (configs - 1) * spacing
        else:
            xpositions = grid.get_electrode_positions()[configs - 1, 0]

        z = np.abs(
            np.max(xpositions, axis=1) - np.min(xpositions, axis=1)
        ) * -0.11
        x = np.mean(xpositions, axis=1)
        return x, z

    def _pseudodepths_schlumberger(self, configs, spacing=1, grid=None):
        """Given distances between electrodes, compute Schlumberger pseudo
        depths for the provided configuration

        The pseudodepth is computed after Roy & Apparao, 1971, as 0.125 times
        the distance between the two outermost electrodes.

        """
        if grid is None:
            xpositions = (configs - 1) * spacing
        else:
            xpositions = grid.get_electrode_positions()[configs - 1, 0]

        x = np.mean(xpositions, axis=1)
        z = np.abs(
            np.max(xpositions, axis=1) - np.min(xpositions, axis=1)
        ) * -0.125
        return x, z

    def _pseudodepths_dd_simple(self, configs, spacing=1, grid=None):
        """Given distances between electrodes, compute dipole-dipole pseudo
        depths for the provided configuration

        The pseudodepth is computed after Roy & Apparao, 1971, as 0.195 times
        the distance between the two outermost electrodes.

        """
        if grid is None:
            xpositions = (configs - 1) * spacing
        else:
            xpositions = grid.get_electrode_positions()[configs - 1, 0]

        z = np.abs(
            np.max(xpositions, axis=1) - np.min(xpositions, axis=1)
        ) * -0.195
        x = np.mean(xpositions, axis=1)
        return x, z

    def plot_pseudodepths(self, spacing=1, grid=None, ctypes=None,
                          dd_merge=False, **kwargs):
        """Plot pseudodepths for the measurements. If grid is given, then the
        actual electrode positions are used, and the parameter 'spacing' is
        ignored'

        Parameters
        ----------
        spacing: float, optional
            assumed distance between electrodes. Default=1
        grid: crtomo.grid.crt_grid instance, optional
            grid instance. Used to infer real electrode positions
        ctypes: list of strings, optional
            a list of configuration types that will be plotted. All
            configurations that can not be sorted into these types will not be
            plotted! Possible types:

            * dd
            * schlumberger

        dd_merge: bool, optional
            if True, merge all skips. Otherwise, generate individual plots for
            each skip

        Returns
        -------
        figs: matplotlib.figure.Figure instance or list of Figure instances
            if only one type was plotted, then the figure instance is return.
            Otherwise, return a list of figure instances.
        axes: axes object or list of axes ojects
            plot axes

        Examples
        --------

        .. plot::
            :include-source:

            import crtomo.configManager as CRconfig
            config = CRconfig.ConfigManager(nr_of_electrodes=48)
            config.gen_dipole_dipole(skipc=2)
            fig, ax = config.plot_pseudodepths(
                spacing=0.3,
                ctypes=['dd', ],
            )

        .. plot::
            :include-source:

            import crtomo.configManager as CRconfig
            config = CRconfig.ConfigManager(nr_of_electrodes=48)
            config.gen_schlumberger(M=24, N=25)
            fig, ax = config.plot_pseudodepths(
                spacing=1,
                ctypes=['schlumberger', ],
            )

        """
        # for each configuration type we have different ways of computing
        # pseudodepths
        pseudo_d_functions = {
            'dd': self._pseudodepths_dd_simple,
            'schlumberger': self._pseudodepths_schlumberger,
            'wenner': self._pseudodepths_wenner,
        }

        titles = {
            'dd': 'dipole-dipole configurations',
            'schlumberger': 'Schlumberger configurations',
            'wenner': 'Wenner configurations',
        }

        # sort the configurations into the various types of configurations
        only_types = ctypes or ['dd', ]
        results = fT.filter(
            self.configs,
            settings={
                'only_types': only_types,
            }
        )

        # loop through all measurement types
        figs = []
        axes = []
        for key in sorted(results.keys()):
            print('plotting: ', key)
            if key == 'not_sorted':
                continue
            index_dict = results[key]
            # it is possible that we want to generate multiple plots for one
            # type of measurement, i.e., to separate skips of dipole-dipole
            # measurements. Therefore we generate two lists:
            # 1) list of list of indices to plot
            # 2) corresponding labels
            if key == 'dd' and not dd_merge:
                plot_list = []
                labels_add = []
                for skip in sorted(index_dict.keys()):
                    plot_list.append(index_dict[skip])
                    labels_add.append(
                        ' - skip {0}'.format(skip)
                    )
            else:
                # merge all indices
                plot_list = [np.hstack(index_dict.values()), ]
                print('schlumberger', plot_list)
                labels_add = ['', ]

            # generate plots
            for indices, label_add in zip(plot_list, labels_add):
                if len(indices) == 0:
                    continue
                ddc = self.configs[indices]
                px, pz = pseudo_d_functions[key](ddc, spacing, grid)

                fig, ax = plt.subplots(figsize=(15 / 2.54, 5 / 2.54))
                ax.scatter(px, pz, color='k', alpha=0.5)

                # plot electrodes
                if grid is not None:
                    electrodes = grid.get_electrode_positions()
                    ax.scatter(
                        electrodes[:, 0],
                        electrodes[:, 1],
                        color='b',
                        label='electrodes',
                    )
                else:
                    ax.scatter(
                        np.arange(0, self.nr_electrodes) * spacing,
                        np.zeros(self.nr_electrodes),
                        color='b',
                        label='electrodes',
                    )
                ax.set_title(titles[key] + label_add)
                ax.set_aspect('equal')
                ax.set_xlabel('x [m]')
                ax.set_ylabel('x [z]')

                fig.tight_layout()
                figs.append(fig)
                axes.append(ax)

        if len(figs) == 1:
            return figs[0], axes[0]
        else:
            return figs, axes

    def plot_pseudosection_type1(self, mid, spacing=1, grid=None, ctypes=None,
                                 dd_merge=False, cb=False, **kwargs):
        """Create a pseudosection type 1 plot for a given measurement.

        This type of pseudosection uses the pseudodepth, as given in the
        literature for certain configuration types, to generate (x,y)
        coordinates for each measurement data. The x-coordinate is computed as
        the mean of all four electrode x positions.

        Note that not all configurations can be plotted using type 1 plots.

        Parameters
        ----------
        mid
        spacing
        grid
        ctypes
        dd_merge
        cb


        Examples
        --------

        .. plot::
            :include-source:

            import numpy as np
            import crtomo.configManager as CRConfig
            config = CRConfig.ConfigManager(nr_of_electrodes=48)
            config.gen_dipole_dipole(skipc=1, stepc=2)
            # generate random measurements
            measurements = np.random.random(config.nr_of_configs)
            mid = config.add_measurements(measurements)
            config.plot_pseudosection_type1(mid, spacing=1)

        """

        pseudo_d_functions = {
            'dd': self._pseudodepths_dd_simple,
            'schlumberger': self._pseudodepths_schlumberger,
            'wenner': self._pseudodepths_wenner,
        }

        titles = {
            'dd': 'dipole-dipole configurations',
            'schlumberger': 'Schlumberger configurations',
            'wenner': 'Wenner configurations',
        }

        # for now sort data and only plot dipole-dipole
        only_types = ctypes or ['dd', ]
        if 'schlumberger' in only_types:
            raise Exception(
                'plotting of pseudosections not implemented for ' +
                'Schlumberger configurations!'
            )
        results = fT.filter(
            self.configs,
            settings={
                'only_types': only_types,
            },
        )

        plot_objects = []
        for key in sorted(results.keys()):
            print('plotting: ', key)
            if key == 'not_sorted':
                continue
            index_dict = results[key]
            # it is possible that we want to generate multiple plots for one
            # type of measurement, i.e., to separate skips of dipole-dipole
            # measurements. Therefore we generate two lists:
            # 1) list of list of indices to plot
            # 2) corresponding labels
            if key == 'dd' and not dd_merge:
                plot_list = []
                labels_add = []
                for skip in sorted(index_dict.keys()):
                    plot_list.append(index_dict[skip])
                    labels_add.append(
                        ' - skip {0}'.format(skip)
                    )
            else:
                # merge all indices
                plot_list = [np.hstack(index_dict.values()), ]
                print('schlumberger', plot_list)
                labels_add = ['', ]

            # generate plots
            for indices, label_add in zip(plot_list, labels_add):
                if len(indices) == 0:
                    continue

                ddc = self.configs[indices]
                px, pz = pseudo_d_functions[key](ddc, spacing, grid)
                print('pxpz', px, pz)

                # take 200 points for the new grid in every direction. Could be
                # adapted to the actual ratio
                xg = np.linspace(px.min(), px.max(), 200)
                zg = np.linspace(pz.min(), pz.max(), 200)

                x, z = np.meshgrid(xg, zg)

                plot_data = self.measurements[mid][indices]
                cmap_name = kwargs.get('cmap_name', 'jet')
                cmap = mpl.cm.get_cmap(cmap_name)

                # normalize data
                data_min = kwargs.get('cbmin', plot_data.min())
                data_max = kwargs.get('cbmax', plot_data.max())
                cnorm = mpl.colors.Normalize(vmin=data_min, vmax=data_max)
                scalarMap = mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap)
                fcolors = scalarMap.to_rgba(plot_data)

                image = si.griddata((px, pz), fcolors, (x, z), method='linear')

                cmap = mpl.cm.get_cmap('jet_r')

                data_ratio = np.abs(
                    px.max() - px.min()
                ) / np.abs(
                    pz.min()
                )
                print(np.abs(px.max() - px.min()))
                print(np.abs(pz.min()))
                print('ratio:', data_ratio)

                fig_size_y = 15 / data_ratio + 6 / 2.54
                print('SIZE', fig_size_y)
                fig = plt.figure(figsize=(15, fig_size_y))

                fig_top = 1 / 2.54 / fig_size_y
                fig_left = 2 / 2.54 / 15
                fig_right = 1 / 2.54 / 15
                if cb:
                    fig_bottom = 3 / 2.54 / fig_size_y
                else:
                    fig_bottom = 0.05

                ax = fig.add_axes([
                    fig_left,
                    fig_bottom + fig_top * 2,
                    1 - fig_left - fig_right,
                    1 - fig_top - fig_bottom - fig_top * 2
                ])

                im = ax.imshow(
                    image[::-1],
                    extent=(xg.min(), xg.max(), zg.min(), zg.max()),
                    interpolation='none',
                    aspect='auto',
                    # vmin=10,
                    # vmax=300,
                    cmap=cmap,
                )
                ax.set_ylim(pz.min(), 0)

                # colorbar
                if cb:
                    print('plotting colorbar')
                    # the colorbar has 3 cm on the bottom
                    ax_cb = fig.add_axes(
                        [
                            fig_left * 4,
                            fig_top * 2,
                            1 - fig_left * 4 - fig_right * 4,
                            fig_bottom - fig_top * 2
                        ]
                    )
                    # from mpl_toolkits.axes_grid1 import make_axes_locatable
                    # divider = make_axes_locatable(ax)
                    # ax_cb = divider.append_axes("bottom", "5%", pad="3%")
                    # (ax_cb, kw) = mpl.colorbar.make_axes_gridspec(
                    #     ax,
                    #     orientation='horizontal',
                    #     fraction=fig_bottom,
                    #     pad=0.3,
                    #     shrink=0.9,
                    #     # location='bottom',
                    # )
                    cb = mpl.colorbar.ColorbarBase(
                        ax=ax_cb,
                        cmap=cmap,
                        norm=cnorm,
                        orientation='horizontal',
                        # **kw
                    )
                    cblabel = kwargs.get('cblabel', '')
                    cb.set_label(cblabel)
                else:
                    fig_bottom = 0.05

                # 1cm on top

                # # 3 cm on bottom for colorbar
                # fig.subplots_adjust(
                #     top=1 - fig_top,
                #     bottom=fig_bottom,
                # )

                ax.set_title(titles[key] + label_add)
                ax.set_aspect('equal')
                ax.set_xlabel('x [m]')
                ax.set_ylabel('x [z]')
                plot_objects.append((fig, ax, im))

        return plot_objects

    def gen_gradient(self, skip=0, step=1, vskip=0, vstep=1):
        """Generate gradient measurements

        Parameters
        ----------
        skip: int
            distance between current electrodes
        step: int
            steplength between subsequent current dipoles
        vskip: int
            distance between voltage electrodes
        vstep: int
            steplength between subsequent voltage dipoles

        """
        N = self.nr_electrodes
        quadpoles = []
        for a in range(1, N - skip, step):
            b = a + skip + 1
            for m in range(a + 1, b - vskip - 1, vstep):
                n = m + vskip + 1
                quadpoles.append((a, b, m, n))

        configs = np.array(quadpoles)
        if configs.size == 0:
            return None

        self.add_to_configs(configs)
        return configs

    def gen_all_voltages_for_injections(self, injections_raw):
        """For a given set of current injections AB, generate all possible
        unique potential measurements.

        After Noel and Xu, 1991, for N electrodes, the number of possible
        voltage dipoles for a given current dipole is :math:`(N - 2)(N - 3) /
        2`. This includes normal and reciprocal measurements.

        If current dipoles are generated with
        ConfigManager.gen_all_current_dipoles(), then :math:`N \cdot (N - 1) /
        2` current dipoles are generated. Thus, this function will produce
        :math:`(N - 1)(N - 2)(N - 3) / 4` four-point configurations ABMN, half
        of which are reciprocals (Noel and Xu, 1991).

        All generated measurements are added to the instance.

        Use ConfigManager.split_into_normal_and_reciprocal() to split the
        configurations into normal and reciprocal measurements.

        Parameters
        ----------
        injections: numpy.ndarray
            Kx2 array holding K current injection dipoles A-B

        Returns
        -------
        configs: numpy.ndarray
            Nax4 array holding all possible measurement configurations

        """
        injections = injections_raw.astype(int)

        N = self.nr_electrodes
        all_quadpoles = []
        for idipole in injections:
            # sort current electrodes and convert to array indices
            Icurrent = np.sort(idipole) - 1

            # voltage electrodes
            velecs = list(range(1, N + 1))

            # remove current electrodes
            del(velecs[Icurrent[1]])
            del(velecs[Icurrent[0]])

            # permutate remaining
            voltages = itertools.permutations(velecs, 2)
            for voltage in voltages:
                all_quadpoles.append(
                    (idipole[0], idipole[1], voltage[0], voltage[1])
                )
        configs_unsorted = np.array(all_quadpoles)
        # sort AB and MN
        configs_sorted = np.hstack((
            np.sort(configs_unsorted[:, 0:2], axis=1),
            np.sort(configs_unsorted[:, 2:4], axis=1),
        ))
        configs = self.remove_duplicates(configs_sorted)

        self.add_to_configs(configs)
        self.remove_duplicates()
        return configs

    def gen_all_current_dipoles(self):
        """Generate all possible current dipoles for the given number of
        electrodes (self.nr_electrodes). Duplicates are removed in the process.

        After Noel and Xu, 1991, for N electrodes, the number of possible
        unique configurations is :math:`N \cdot (N - 1) / 2`. This excludes
        duplicates in the form of switches current/voltages electrodes, as well
        as reciprocal measurements.

        Returns
        -------
        configs: Nx2 numpy.ndarray
            all possible current dipoles A-B
        """
        N = self.nr_electrodes
        celecs = list(range(1, N + 1))
        AB_list = itertools.permutations(celecs, 2)
        AB = np.array([ab for ab in AB_list])
        AB.sort(axis=1)

        # now we need to filter duplicates
        AB = np.unique(
            AB.view(AB.dtype.descr * 2)
        ).view(AB.dtype).reshape(-1, 2)

        return AB

    def remove_duplicates(self, configs=None):
        """remove duplicate entries from 4-point configurations. If no
        configurations are provided, then use self.configs. Unique
        configurations are only returned if configs is not None.

        Parameters
        ----------
        configs: Nx4 numpy.ndarray, optional
            remove duplicates from these configurations instead from
            self.configs.

        Returns
        -------
        configs_unique: Kx4 numpy.ndarray
            unique configurations. Only returned if configs is not None

        """
        if configs is None:
            c = self.configs
        else:
            c = configs
        struct = c.view(c.dtype.descr * 4)
        configs_unique = np.unique(struct).view(c.dtype).reshape(-1, 4)
        if configs is None:
            self.configs = configs_unique
        else:
            return configs_unique

    def gen_schlumberger(self, M, N, a=None):
        """generate one Schlumberger sounding configuration, that is, one set
        of configurations for one potential dipole MN.

        Parameters
        ----------
        M: int
            electrode number for the first potential electrode
        N: int
            electrode number for the second potential electrode
        a: int, optional
            stepping between subsequent voltage electrodes. If not set,
            determine it as a = abs(M - N)

        Returns
        -------
        configs: Kx4 numpy.ndarray
            array holding the configurations

        Examples
        --------
        from crtomo.mpl_setup import *
        import crtomo.configManager as CRconfig
        config = CRconfig.ConfigManager(nr_of_electrodes=40)
        config.gen_schlumberger(M=20, N=21)


        .. plot::

            import numpy as np
            from crtomo.mpl_setup import *
            import crtomo.tdManager as CRman
            import crtomo.grid as CRGrid
            grid = CRGrid.crt_grid.create_surface_grid(
                nr_electrodes=50,
                spacing=0.5,
                depth=25,
                right=20,
                left=20,
                char_lengths=[0.1, 5, 5, 5],
            )
            man = CRman.tdMan(grid=grid)
            man.configs.gen_schlumberger(M=20, N=21)
            K = man.configs.compute_K_factors(spacing=0.5)
            # pseudo depth after Knödel et al for Schlumberger configurations
            pdepth = np.abs(
                np.max(
                    man.configs.configs, axis=1
                ) - np.min(
                    man.configs.configs, axis=1
                )
            ) * 0.19
            fig, axes = plt.subplots(2, 1, figsize=(15 / 2.54, 10 / 2.54))
            ax = axes[0]
            for contrast in (2, 5, 10):
                pid_mag, pid_pha = man.add_homogeneous_model(
                    magnitude=1000, phase=0)
                man.clear_measurements()
                man.parman.modify_area(
                    pid_mag, -100, 100, -40, -3, 1000 / contrast)
                # man.parman.modify_area(
                #    pid_mag, -100, 100, -40, -10, 500 / contrast)
                ax.plot(pdepth, man.measurements()[:, 0] * K, '.-')
            ax.axvline(x=3, color='k', linestyle='dashed')
            # ax.axvline(x=10, color='k', linestyle='dashed')
            ax.set_xlabel('pseudo depth [m]')
            ax.set_ylabel('measurement [$\Omega$]')
            ax.set_title(
                'Schlumberger sounding for different layer resistivities')
            ax = axes[1]
            # grid.plot_grid_to_ax(ax)
            man.plot.plot_elements_to_ax(
                pid_mag,
                ax=ax,
                plot_colorbar=True,
            )
            fig.tight_layout()
            fig.savefig('schlumberger_sounding.png', dpi=300)


        """
        if a is None:
            a = np.abs(M - N)

        nr_of_steps_left = int(min(M, N) - 1 / a)
        nr_of_steps_right = int((self.nr_electrodes - max(M, N)) / a)
        configs = []
        for i in range(0, min(nr_of_steps_left, nr_of_steps_right)):
            A = min(M, N) - (i + 1) * a
            B = max(M, N) + (i + 1) * a
            configs.append(
                (A, B, M, N)
            )
        configs = np.array(configs)
        self.add_to_configs(configs)
        return configs

    def gen_wenner(self, a):
        """Generate Wenner measurement configurations.

        Parameters
        ----------
        a: int
            distance (in electrodes) between subsequent electrodes of each
            four-point configuration.

        Returns
        -------
        configs: Kx4 numpy.ndarray
            array holding the configurations
        """
        configs = []
        for i in range(1, self.nr_electrodes - 3 * a + 1):
            configs.append(
                (i, i + a, i + 2 * a, i + 3 * a),
            )
        configs = np.array(configs)
        self.add_to_configs(configs)
        return configs

    def add_to_configs(self, configs):
        """Add one or more measurement configurations to the stored
        configurations

        Parameters
        ----------
        configs: list or numpy.ndarray
            list or array of configurations

        Returns
        -------
        configs: Kx4 numpy.ndarray
            array holding all configurations of this instance
        """
        if len(configs) == 0:
            return None

        if self.configs is None:
            self.configs = np.atleast_2d(configs)
        else:
            configs = np.atleast_2d(configs)
            self.configs = np.vstack((self.configs, configs))
        return self.configs

    def split_into_normal_and_reciprocal(
            self, pad=False, return_indices=False):
        """Split the stored configurations into normal and reciprocal
        measurements

        ** *Rule 1: the normal configuration contains the smallest electrode
        number of the four involved electrodes in the current dipole* **

        Parameters
        ----------
        pad: bool, optional
            if True, add numpy.nan values to the reciprocals for non-existent
            measuremnts
        return_indices: bool, optional
            if True, also return the indices of normal and reciprocal
            measurments. This can be used to extract corresponding
            measurements.

        Returns
        -------
        normal: numpy.ndarray
            Nnx4 array. If pad is True, then Nn == N (total number of
            unique measurements). Otherwise Nn is the number of normal
            measurements.
        reciprocal: numpy.ndarray
            Nrx4 array. If pad is True, then Nr == N (total number of
            unique measurements). Otherwise Nr is the number of reciprocal
            measurements.
        nor_indices: numpy.ndarray, optional
            Nnx1 array containing the indices of normal measurements. Only
            returned if return_indices is True.
        rec_indices: numpy.ndarray, optional
            Nrx1 array containing the indices of normal measurements. Only
            returned if return_indices is True.

        """
        # for simplicity, we create an array where AB and MN are sorted
        configs = np.hstack((
            np.sort(self.configs[:, 0:2], axis=1),
            np.sort(self.configs[:, 2:4], axis=1)
        ))

        ab_min = configs[:, 0]
        mn_min = configs[:, 2]

        # rule 1
        indices_normal = np.where(ab_min < mn_min)[0]

        # now look for reciprocals
        indices_used = []
        normal = []
        normal_indices = []
        reciprocal_indices = []
        reciprocal = []
        duplicates = []
        for index in indices_normal:
            indices_used.append(index)
            normal.append(self.configs[index, :])
            normal_indices.append(index)

            # look for reciprocal configuration
            index_rec = np.where(
                # A == M, B == N, M == A, N == B
                (configs[:, 0] == configs[index, 2]) &
                (configs[:, 1] == configs[index, 3]) &
                (configs[:, 2] == configs[index, 0]) &
                (configs[:, 3] == configs[index, 1])
            )[0]
            if len(index_rec) == 0 and pad:
                reciprocal.append(np.ones(4) * np.nan)
            elif len(index_rec) == 1:
                reciprocal.append(self.configs[index_rec[0], :])
                indices_used.append(index_rec[0])
                reciprocal_indices.append(index_rec[0])
            elif len(index_rec > 1):
                # take the first one
                reciprocal.append(self.configs[index_rec[0], :])
                reciprocal_indices.append(index_rec[0])
                duplicates += list(index_rec[1:])
                indices_used += list(index_rec)

        # now determine all reciprocal-only parameters
        set_all_indices = set(list(range(0, configs.shape[0])))
        set_used_indices = set(indices_used)
        reciprocal_only_indices = set_all_indices - set_used_indices
        for index in reciprocal_only_indices:
            if pad:
                normal.append(np.ones(4) * np.nan)
            reciprocal.append(self.configs[index, :])

        normals = np.array(normal)
        reciprocals = np.array(reciprocal)

        if return_indices:
            return normals, reciprocals, normal_indices, reciprocal_indices
        else:
            return normals, reciprocals

    def gen_reciprocals(self, quadrupoles):
        """For a given set of quadrupoles, generate and return reciprocals
        """
        reciprocals = quadrupoles[:, ::-1].copy()
        reciprocals[:, 0:2] = np.sort(reciprocals[:, 0:2], axis=1)
        reciprocals[:, 2:4] = np.sort(reciprocals[:, 2:4], axis=1)
        return reciprocals

    def plot_error_pars(self, mid):
        """ ???

        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))

        def plot_error_pars(axes, a, b, R, label=''):
            dR = a * R + b
            dlogR = np.abs(a + b / R)
            ax = axes[0]
            ax.scatter(R, dR / R * 100, label=label)
            ax = axes[1]
            ax.scatter(R, dlogR / np.abs(np.log(R)) * 100, label=label)
            ax.set_xscale('log')
            ax.set_yscale('log')

        for b in np.linspace(R.min(), np.percentile(R, 10), 5):
            plot_error_pars(axes, 0.05, b, R, label='b={0:.4f}'.format(b))
        axes[0].set_xlabel(r'$R [\Omega]$')
        axes[0].set_ylabel(r'$(\Delta R/R) \cdot 100 [\%]$')
        axes[1].set_xlabel(r'$log_{10}(R [\Omega])$')
        axes[1].set_ylabel(
            r'$log_{10}(R) / \Delta log_{10}(R) \cdot 100 [\%]$'
        )
        axes[0].axhline(y=100)
        axes[1].axhline(y=100)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2)
        axes[0].legend(
            loc="lower center",
            ncol=4,
            bbox_to_anchor=(0, 0, 1, 1),
            bbox_transform=fig.transFigure
        )

        # fig.savefig('out.png', dpi=300)
        return fig

    def compute_K_factors(self, spacing=None, configs=None, numerical=False,
                          elem_file=None, elec_file=None):
        """Compute analytical geometrical factors.

        TODO: use real electrode positions from self.grid
        """
        if configs is None:
            use_configs = self.configs
        else:
            use_configs = configs

        if numerical:
            settings = {
                'elem': elem_file,
                'elec': elec_file,
                'rho': 100,
            }
            K = edfK.compute_K_numerical(use_configs, settings)
        else:
            K = edfK.compute_K_analytical(use_configs, spacing=spacing)
        return K

    def gen_configs_permutate(self, injections_raw,
                              only_same_dipole_length=False,
                              ignore_crossed_dipoles=False):
        """
        Create measurement configurations out of a pool of current injections.
        Use only the provided dipoles for potential dipole selection. This
        means that we have always reciprocal measurements.

        Remove quadpoles where electrodes are used both as current and voltage
        dipoles.

        Parameters
        ----------
        injections_raw: Nx2 array
            current injections
        only_same_dipole_length: bool, optional
            if True, only generate permutations for the same dipole length
        ignore_crossed_dipoles: bool, optional
            If True, potential dipoles will be ignored that lie between current
            dipoles,  e.g. 1-4 3-5. In this case it is possible to not have
            full normal-reciprocal coverage.

        Returns
        -------
        configs: Nx4 array
            quadrupoles generated out of the current injections

        """
        injections = np.atleast_2d(injections_raw).astype(int)
        N = injections.shape[0]

        measurements = []

        for injection in range(0, N):
            dipole_length = np.abs(
                injections[injection][1] -
                injections[injection][0]
            )

            # select all dipole EXCEPT for the injection dipole
            for i in set(range(0, N)) - set([injection]):
                test_dipole_length = np.abs(
                    injections[i, :][1] - injections[i, :][0]
                )
                if(only_same_dipole_length and
                   test_dipole_length != dipole_length):
                    continue
                quadpole = np.array(
                    [
                        injections[injection, :],
                        injections[i, :]
                    ]
                ).flatten()
                if ignore_crossed_dipoles is True:
                    # check if we need to ignore this dipole
                    # Note: this could be wrong if electrode number are not
                    # ascending!
                    if(quadpole[2] > quadpole[0] and
                       quadpole[2] < quadpole[1]):
                        print('A - ignoring', quadpole)
                    elif(quadpole[3] > quadpole[0] and
                         quadpole[3] < quadpole[1]):
                        print('B - ignoring', quadpole)
                    else:
                        measurements.append(quadpole)
                else:
                    # add very quadpole
                    measurements.append(quadpole)

        # check and remove double use of electrodes
        filtered = []
        for quadpole in measurements:
            if (not set(quadpole[0:2]).isdisjoint(set(quadpole[2:4]))):
                print('Ignoring quadrupole because of repeated electrode use:',
                      quadpole)
                pass
            else:
                filtered.append(quadpole)
        self.add_to_configs(filtered)
        return np.array(filtered)

    @staticmethod
    def _get_unique_identifiers(ee_raw):
        """

        """
        ee = ee_raw.copy()

        # sort order of dipole electrodes, i.e., turn 2-1 into 1-2
        ee_s = np.sort(ee, axis=1)

        # get unique dipoles
        eeu = np.unique(
            ee_s.view(ee_s.dtype.descr * 2)
        ).view(ee_s.dtype).reshape(-1, 2)

        # sort according to first electrode number
        eeu_s = eeu[
            np.argsort(eeu[:, 0]),
            :
        ]

        # differences
        eeu_diff = np.abs(eeu_s[:, 0] - eeu_s[:, 1])
        # important: use mergesort here, as this is a stable sort algorithm,
        # i.e., it preserves the order of equal values
        indices = np.argsort(eeu_diff, kind='mergesort')

        # final arrangement
        eeu_final = eeu_s[indices, :]

        ee_ids = {
            key: value for key, value in zip(
                (eeu_final[:, 0] * 1e5 + eeu_final[:, 1]).astype(int),
                range(0, eeu_final.shape[0]),
            )
        }
        return ee_ids

    def test_get_unique_identifiers(self):
        np.random.seed(1)
        results = []
        for i in range(0, 10):
            ab = np.random.permutation(c[:, 0:2])
            print(ab)
            q = self._get_unique_identifiers(ab)
            for key in sorted(q.keys()):
                print(key, q[key])
            results.append(q)
        # compare the results
        print('checking results:')
        for x in results:
            if x != results[0]:
                print('error')

    def plot_pseudosection_type2(self, mid, **kwargs):
        """Create a pseudosection plot of type 2.

        For a given measurement data set, create plots that graphically show
        the data in a 2D color plot. Hereby, x and y coordinates in the plot
        are determined by the current dipole (x-axis) and voltage dipole
        (y-axis) of the corresponding measurement configurations.

        This type of rawdata plot can plot any type of measurement
        configurations, i.e., it is not restricted to certain types of
        configurations such as Dipole-dipole or Wenner configurations. However,
        spatial inferences cannot be made from the plots for all configuration
        types.

        Coordinates are generated by separately sorting the dipoles
        (current/voltage) along the first electrode, and then subsequently
        sorting by the difference (skip) between both electrodes.

        Note that this type of raw data plot does not take into account the
        real electrode spacing of the measurement setup.

        Type 2 plots can show normal and reciprocal data at the same time.
        Hereby the upper left triangle of the plot area usually contains normal
        data, and the lower right triangle contains the corresponding
        reciprocal data. Therefore a quick assessment of normal-reciprocal
        differences can be made by visually comparing the symmetry on the 1:1
        line going from the lower left corner to the upper right corner.

        Note that this interpretation usually only holds for Dipole-Dipole data
        (and the related Wenner configurations).

        Parameters
        ----------

        mid: integer or numpy.ndarray
            Measurement ID of stored measurement data, or a numpy array (size
            N) with data that will be plotted. Must have the same length as
            number of configurations.
        ax: matplotlib.Axes object, optional
            axes object to plot to. If not provided, a new figure and axes
            object will be created and returned
        nocb: bool, optional
            if set to False, don't plot the colorbar
        cblabel: string, optional
            label for the colorbar
        cbmin: float, optional
            colorbar minimum
        cbmax: float, optional
            colorbar maximum
        xlabel: string, optional
            xlabel for the plot
        ylabel: string, optional
            ylabel for the plot
        do_not_saturate: bool, optional
            if set to True, then values outside the colorbar range will not
            saturate with the respective limit colors. Instead, values lower
            than the CB are colored "cyan" and vaues above the CB limit are
            colored "red"
        log10: bool, optional
            if set to True, plot the log10 values of the provided data

        Returns
        -------
        fig:
            figure object
        ax:
            axes object

        Examples
        --------

        You can just supply a data vector to the plot function:

        .. plot::
            :include-source:

            import numpy as np
            import crtomo.configManager as CRConfig
            configs = CRConfig.ConfigManager(nr_of_electrodes=48)
            configs.gen_dipole_dipole(skipc=1, stepc=2)
            measurements = np.random.random(configs.nr_of_configs)
            configs.plot_pseudosection_type2(
                mid=measurements,
            )

        Generate a simple type 2 plot:

        .. plot::
            :include-source:

            import numpy as np
            import crtomo.configManager as CRConfig
            configs = CRConfig.ConfigManager(nr_of_electrodes=48)
            configs.gen_dipole_dipole(skipc=1, stepc=2)
            measurements = np.random.random(configs.nr_of_configs)
            mid = configs.add_measurements(measurements)
            configs.plot_pseudosection_type2(
                mid,
                cblabel='this label',
                xlabel='xlabel',
                ylabel='ylabel',
            )

        You can also supply axes to plot to:

        .. plot::
            :include-source:

            import numpy as np
            from crtomo.mpl_setup import *
            import crtomo.configManager as CRConfig

            configs = CRConfig.ConfigManager(nr_of_electrodes=48)
            configs.gen_dipole_dipole(skipc=1, stepc=2)
            K = configs.compute_K_factors(spacing=1)
            measurements = np.random.random(configs.nr_of_configs)
            mid = configs.add_measurements(measurements)

            fig, axes = plt.subplots(1, 2)

            configs.plot_pseudosection_type2(
                mid,
                ax=axes[0],
                cblabel='this label',
                xlabel='xlabel',
                ylabel='ylabel',
            )
            configs.plot_pseudosection_type2(
                K,
                ax=axes[1],
                cblabel='K factor',
                xlabel='xlabel',
                ylabel='ylabel',
            )
            fig.tight_layout()




        """
        c = self.configs

        AB_ids = self._get_unique_identifiers(c[:, 0:2])
        MN_ids = self._get_unique_identifiers(c[:, 2:4])

        ab_sorted = np.sort(c[:, 0:2], axis=1)
        mn_sorted = np.sort(c[:, 2:4], axis=1)

        AB_coords = [
            AB_ids[x] for x in
            (ab_sorted[:, 0] * 1e5 + ab_sorted[:, 1]).astype(int)
        ]
        MN_coords = [
            MN_ids[x] for x in
            (mn_sorted[:, 0] * 1e5 + mn_sorted[:, 1]).astype(int)
        ]

        # check for duplicate positions
        ABMN_coords = np.vstack((AB_coords, MN_coords)).T.copy()
        _, counts = np.unique(
            ABMN_coords.view(
                ABMN_coords.dtype.descr * 2
            ),
            return_counts=True,
        )
        if np.any(counts > 1):
            print('found duplicate coordinates!')
            duplicate_configs = np.where(counts > 1)[0]
            print('duplicate configs:')
            print('A B M N')
            for i in duplicate_configs:
                print(c[i, :])

        # prepare matrix
        if isinstance(mid, int):
            plot_values = self.measurements[mid]
        elif isinstance(mid, np.ndarray):
            plot_values = np.squeeze(mid)
        else:
            raise Exception('Data in parameter "mid" not understood')

        if kwargs.get('log10', False):
            plot_values = np.log10(plot_values)

        C = np.zeros((len(MN_ids.items()), len(AB_ids))) * np.nan
        C[MN_coords, AB_coords] = plot_values

        # for display purposes, reverse the first dimension
        C = C[::-1, :]

        ax = kwargs.get('ax', None)
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(15 / 2.54, 10 / 2.54))
        fig = ax.get_figure()

        cmap = mpl.cm.get_cmap('viridis')
        if kwargs.get('do_not_saturate', False):
            cmap.set_over(
                color='r'
            )
            cmap.set_under(
                color='c'
            )
        im = ax.matshow(
            C,
            interpolation='none',
            cmap=cmap,
            aspect='auto',
            vmin=kwargs.get('cbmin', None),
            vmax=kwargs.get('cbmax', None),
            extent=[
                0, max(AB_coords),
                0, max(MN_coords),
            ],
        )

        max_xy = max((max(AB_coords), max(MN_coords)))
        ax.plot(
            (0, max_xy),
            (0, max_xy),
            '-',
            color='k',
            linewidth=1.0,
        )

        if not kwargs.get('nocb', False):
            cb = fig.colorbar(im, ax=ax)
            cb.set_label(
                kwargs.get('cblabel', '')
            )

        ax.set_xlabel(
            kwargs.get('xlabel', 'current dipoles')
        )
        ax.set_ylabel(
            kwargs.get('ylabel', 'voltage dipoles')
        )

        return fig, ax

    def write_configs(self, filename):
        """Write configs to file in four columns
        """
        np.savetxt(filename, self.configs, fmt='%i %i %i %i')

    @property
    def get_unique_injections(self):
        return np.unique(self.configs[:, 0:2], axis=1)

    def gen_voltage_dipoles_skip(self, cinj, skip=0):
        """For given current injections, generate all possible voltage dipoles
        with skip **skip**.

        Parameters
        ----------
        cinj : :py:class:`numpy.ndarray`
            Nx2 array containing the current injection electrodes a,b
        skip : int, optional
            Skip used for voltage electrodes. Default: 0

        """
        new_configs = []
        for ab in cinj:
            for i in range(1, self.nr_electrodes + 1):
                m = i
                n = i + skip + 1
                if m in ab or n in ab:
                    continue
                if n > self.nr_electrodes:
                    continue
                new_configs.append((ab[0], ab[1], m, n))
        configs = np.array(new_configs)
        self.add_to_configs(configs)
        return configs
