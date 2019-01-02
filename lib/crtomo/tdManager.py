# *-* coding: utf-8 *-*
"""A digital representation of a tomodir, i.e., a single-frequency inversion.

** Modelling **

* save to one large binary file

    * I suggest to use pytables (tables) for that.
    * need to figure out how to store the individual modules in there, and
      retrieve them later.

* save-to-tomodir

    * including modeling results

* on-demand generation of modeling results

* provide easy plotting of pseudo sections for voltages:

    * magnitude (linear/log)
    * phase (liner)
    * rho (analytic/numerical K factor)
    * sigma'
    * sigma''

  plots should be generated for dipole-dipole only, and all additional
  configurations that we can find reliable z-estimators for.

  Provide corresponding filter functions, or use reda for that (do we want to
  depend on reda? I think it's ok to do this).

  In case only a subset of data is plotted, note the number of missing values
  in the plot!

  We could also provide means to generate the (x/z) coordinates by means of
  averaging over sensitivity distributions (see old script for that).

* provide functionality to check voltages against sensitivity sums

* plot sensitivities

    * perhaps provide different transformation functions, for example the one
      from Johannes Kenkel

* plot potential distributions

    * also plot current lines via streamlines?
    * can we generate current densities? not sure how to mix element data
      (sigma) and potential data (nodes) in j = sigma E

"""
from glob import glob
import re
import os
import tempfile
import subprocess
from io import StringIO
import itertools

import matplotlib.colors
import matplotlib.cm
import numpy as np
import pandas as pd

import crtomo.mpl
plt, mpl = crtomo.mpl.setup()

import crtomo.binaries as CRBin
import crtomo.grid as CRGrid
import crtomo.nodeManager as nM
import crtomo.parManager as pM
import crtomo.configManager as cConf
import crtomo.cfg as CRcfg
import crtomo.plotManager as PlotManager


class noise_model(object):
    """noise model as contained in the crt.noisemod file

    Notes
    -----

    1. line: 1       # Ensemble seed
    2. line: 0.300 # Relative error resistance A (noise) [%] dR=AR+B
    3. line: 0.100E-01 # Absolute errior resistance B (noise) [Ohm m]
    4. line: 0.00  # Phase error parameter A1 [mRad/Ohm/m] dp=A1*R^B1+A2*p+p0
    5. line: 0.00        # Phase error parameter B1 (noise) []
    6. line: 0.00        # Relative phase error A2 (noise) [%]
    7. line: 0.00        # Absolute phase error p0 (noise) [mRad]

    """
    def __init__(
            self, seed,
            mag_rel=0, mag_abs=0,
            pha_a1=0, pha_b1=0, pha_rel=0, pha_abs=0):
        self.seed = seed
        self.mag_rel = mag_rel
        self.mag_abs = mag_abs
        self.pha_a1 = pha_a1
        self.pha_b1 = pha_b1
        self.pha_rel = pha_rel
        self.pha_abs = pha_abs

    def write_crt_noisemod(self, filename):
        with open(filename, 'w') as fid:
            for value in (self.seed, self.mag_rel, self.mag_abs,
                          self.pha_a1, self.pha_b1, self.pha_rel,
                          self.pha_abs):
                fid.write('{}\n'.format(value))


class tdMan(object):
    """Manage tomodirs

    """
    def __init__(self, **kwargs):
        """The following initialization permutations are possible:

            * initialize empty
            * initialize from an existing tomodir
                * tomodir: [tomodir path]
            * supply one or more of the building blocks of a tomodir
                * grid: crtomo.grid.crt_grid instance
                * crmod_cfg: crtomo.cfg.crmod_cfg instance
                * crtomo_cfg: crmod.cfg.crtomo_cfg instance

        Parameters
        ----------

        **kwargs :
            grid: crtomo.grid.crt_grid
                A fully initialized grid object
            tempdir : string|None, optional
                if set, use this directory to create all temporary directories
                in. For example, settings this to /dev/shm can result in faster
                generation of large Jacobians
        """
        # these variables will be filled later
        self.grid = None
        self.crmod_cfg = None
        self.crtomo_cfg = None
        self.nodeman = None
        self.parman = None
        self.configs = None
        self.plot = None
        self.crmod_cfg = None
        self.crtomo_cfg = None
        self.tempdir = kwargs.get('tempdir', None)
        # we need a struct to organize the assignments
        self.assignments = {
            # should contain a two-item list with ids in parman
            'forward_model': None,
            # should contain a two-item list with ids of magnitude and phase
            # measurements (which are stored in self.configs)
            'measurements': None,
            # store sensitivity cids here
            'sensitivities': None,
            # store potential nids here
            'potentials': None,
            # store the ID of the resolution matrix
            'resm': None,
            # last iteration of inversion results
            'inversion': None
        }
        # short-cut
        self.a = self.assignments

        # if set, use this class for the decoupled error model
        self.noise_model = kwargs.get('noise_model', None)

        # indicates if all information for modeling are present
        self.can_model = False
        # indicates if all information for inversion are present
        self.can_invert = False

        self._initialize_components(kwargs)

    def _initialize_components(self, kwargs):
        """initialize the various components using the supplied \*\*kwargs

        Parameters
        ----------
        kwargs: dict
            \*\*kwargs dict as received by __init__()

        """
        tomodir = None

        # load/assign grid
        if 'tomodir' in kwargs:
            # load grid
            tomodir = kwargs.get('tomodir')
            print('importing tomodir {}'.format(tomodir))
            assert os.path.isdir(tomodir)
            grid = CRGrid.crt_grid(
                tomodir + os.sep + 'grid' + os.sep + 'elem.dat',
                tomodir + os.sep + 'grid' + os.sep + 'elec.dat',
            )
            self.grid = grid
        elif 'grid' in kwargs:
            self.grid = kwargs.get('grid')
        elif 'elem_file' in kwargs and 'elec_file' in kwargs:
            grid = CRGrid.crt_grid()
            grid.load_grid(
                kwargs['elem_file'],
                kwargs['elec_file'],
            )
            self.grid = grid
        else:
            raise Exception(
                'You must provide either a grid instance or ' +
                'elem_file/elec_file file paths'
            )

        crmod_cfg = kwargs.get('crmod_cfg', CRcfg.crmod_config())
        self.crmod_cfg = crmod_cfg

        crtomo_cfg = kwargs.get('crtomo_cfg', CRcfg.crtomo_config())
        self.crtomo_cfg = crtomo_cfg

        parman = kwargs.get('parman', pM.ParMan(self.grid))
        self.parman = parman

        nodeman = kwargs.get('nodeman', nM.NodeMan(self.grid))
        self.nodeman = nodeman

        configs_abmn = kwargs.get('configs_abmn', None)
        config = cConf.ConfigManager(
            nr_of_electrodes=self.grid.nr_of_electrodes
        )
        if configs_abmn is not None:
            config.add_to_configs(configs_abmn)
        self.configs = config

        config_file = kwargs.get('config_file', None)
        if config_file is not None:
            self.configs.load_crmod_config(config_file)

        voltage_file = kwargs.get('volt_file', None)
        if voltage_file is not None:
            cids = self.configs.load_crmod_volt(voltage_file)
            self.assignments['measurements'] = cids

        self.plot = PlotManager.plotManager(
            grid=self.grid,
            nm=self.nodeman,
            pm=self.parman,
        )

        # if we load from a tomodir, also load configs and inversion results
        if tomodir is not None:
            print('importing tomodir results')
            # forward configurations
            config_file = tomodir + os.sep + 'config' + os.sep + 'config.dat'
            if os.path.isfile(config_file):
                self.configs.load_crmod_config(config_file)
            # load inversion results
            self.read_inversion_results(tomodir)

    def create_tomodir(self, directory):
        """Create a tomodir subdirectory structure in the given directory
        """
        pwd = os.getcwd()
        if not os.path.isdir(directory):
            os.makedirs(directory)
        os.chdir(directory)

        directories = (
            'config',
            'exe',
            'grid',
            'mod',
            'mod/pot',
            'mod/sens',
            'rho',
        )
        for directory in directories:
            if not os.path.isdir(directory):
                os.makedirs(directory)

        os.chdir(pwd)

    def _check_state(self):
        """Check if this instance can model and/or can invert
        """
        if(self.grid is not None and
           self.configs.configs is not None and
           self.assignments['forward_model'] is not None):
            self.can_model = True

        if(self.grid is not None and
           self.assignments['measurements'] is not None):
            self.can_invert = True

    def load_rho_file(self, filename):
        """Load a forward model from a rho.dat file

        Parameters
        ----------
        filename: string
            filename to rho.dat file

        Returns
        -------
        pid_mag: int
            parameter id for the magnitude model
        pid_pha: int
            parameter id for the phase model

        """
        pids = self.parman.load_from_rho_file(filename)
        self.register_magnitude_model(pids[0])
        self.register_phase_model(pids[1])
        return pids

    def save_to_tomodir(self, directory):
        """Save the tomodir instance to a directory structure.

        Note
        ----

        Test cases:

            * modeling only
            * inversion only
            * modeling and inversion

        """
        self.create_tomodir(directory)

        self.grid.save_elem_file(
            directory + os.sep + 'grid/elem.dat'
        )

        self.grid.save_elec_file(
            directory + os.sep + 'grid/elec.dat'
        )

        # modeling
        if self.configs.configs is not None:
            self.configs.write_crmod_config(
                directory + os.sep + 'config/config.dat'
            )

        if self.assignments['forward_model'] is not None:
            self.parman.save_to_rho_file(
                directory + os.sep + 'rho/rho.dat',
                self.assignments['forward_model'][0],
                self.assignments['forward_model'][1],
            )

        self.crmod_cfg.write_to_file(
            directory + os.sep + 'exe/crmod.cfg'
        )

        if self.assignments['measurements'] is not None:
            self.configs.write_crmod_volt(
                directory + os.sep + 'mod/volt.dat',
                self.assignments['measurements']
            )

        if self.assignments['sensitivities'] is not None:
            self._save_sensitivities(
                directory + os.sep + 'mod/sens',
            )

        if self.assignments['potentials'] is not None:
            self._save_potentials(
                directory + os.sep + 'mod/pot',
            )

        # inversion
        self.crtomo_cfg.write_to_file(
            directory + os.sep + 'exe/crtomo.cfg'
        )

        if self.noise_model is not None:
            self.noise_model.write_crt_noisemod(
                directory + os.sep + 'exe/crt.noisemod'
            )

        if not os.path.isdir(directory + os.sep + 'inv'):
            os.makedirs(directory + os.sep + 'inv')

    def _save_sensitivities(self, directory):
        """save sensitivities to a directory
        """
        print('saving sensitivities')
        digits = int(np.ceil(np.log10(self.configs.configs.shape[0])))
        for i in range(0, self.configs.configs.shape[0]):
            sens_data, meta_data = self.get_sensitivity(i)
            filename_raw = 'sens{0:0' + '{0}'.format(digits) + '}.dat'
            filename = directory + os.sep + filename_raw.format(i + 1)

            grid_xz = self.grid.get_element_centroids()
            all_data = np.vstack((
                grid_xz[:, 0],
                grid_xz[:, 0],
                sens_data[0],
                sens_data[1],
            )).T
            with open(filename, 'wb') as fid:
                fid.write(bytes(
                    '{0} {1}\n'.format(meta_data[0], meta_data[1]),
                    'utf-8'
                ))
                np.savetxt(fid, all_data)

    def _save_potentials(self, directory):
        """save potentials to a directory
        """
        print('saving potentials')
        digits = int(np.ceil(np.log10(self.configs.configs.shape[0])))
        for i in range(0, self.configs.configs.shape[0]):
            pot_data = self.get_potential(i)
            filename_raw = 'pot{0:0' + '{0}'.format(digits) + '}.dat'
            filename = directory + os.sep + filename_raw.format(i + 1)

            nodes = self.grid.nodes['sorted'][:, 1:3]
            all_data = np.hstack((
                nodes,
                pot_data[0][:, np.newaxis],
                pot_data[1][:, np.newaxis],
            ))
            with open(filename, 'wb') as fid:
                np.savetxt(fid, all_data)

    def clear_measurements(self):
        """Forget any previous measurements
        """
        mid_list = self.assignments.get('measurements', None)
        if mid_list is not None:
            for mid in mid_list:
                self.configs.delete_measurements(mid=mid)
            self.assignments['measurements'] = None

    def measurements(self):
        """Return the measurements associated with this instance.

        if measurements are not present, check if we can model, and then
        run CRMod to load the measurements.
        """
        # check if we have measurements
        mid = self.assignments.get('measurements', None)
        if mid is None:
            return_value = self.model(
                voltages=True,
                sensitivities=False,
                potentials=False,
            )
            if return_value is None:
                print('cannot model')
                return

        # retrieve measurements
        cids = self.assignments['measurements']
        measurements = np.vstack((
            self.configs.measurements[cids[0]],
            self.configs.measurements[cids[1]],
        )).T
        return measurements

    def _read_modeling_results(self, directory, silent=False):
        """Read modeling results from a given mod/ directory. Possible values
        to read in are:

            * voltages
            * potentials
            * sensitivities

        """

        voltage_file = directory + os.sep + 'volt.dat'
        if os.path.isfile(voltage_file):
            if not silent:
                print('reading voltages')
            self.read_voltages(voltage_file)

        sens_files = sorted(glob(
            directory + os.sep + 'sens' + os.sep + 'sens*.dat')
        )
        # check if there are sensitivity files, and that the nr corresponds to
        # the nr of configs
        if(len(sens_files) > 0 and
           len(sens_files) == self.configs.nr_of_configs):
            print('reading sensitivities')
            self._read_sensitivities(directory + os.sep + 'sens')

        # same for potentials
        pot_files = sorted(glob(
            directory + os.sep + 'pot' + os.sep + 'pot*.dat')
        )
        # check if there are sensitivity files, and that the nr corresponds to
        # the nr of configs
        if(len(pot_files) > 0 and
           len(pot_files) == self.configs.nr_of_configs):
            print('reading potentials')
            self._read_potentials(directory + os.sep + 'pot')

    def _read_sensitivities(self, sens_dir):
        """import sensitivities from a directory

        Note
        ----

        * check that signs are correct in case CRMod switches potential
          electrodes

        """
        if self.assignments['sensitivities'] is not None:
            print('Sensitivities already imported. Will not overwrite!')
            return
        else:
            self.assignments['sensitivities'] = {}

        sens_files = sorted(glob(sens_dir + os.sep + 'sens*.dat'))
        for nr, filename in enumerate(sens_files):
            with open(filename, 'r') as fid:
                metadata = np.fromstring(
                    fid.readline().strip(), sep=' ', count=2
                )
                meta_re = metadata[0]
                meta_im = metadata[1]

                sens_data = np.loadtxt(fid)

                cids = self.parman.add_data(
                    sens_data[:, 2:4],
                    [meta_re, meta_im],
                )
                # store cids for later retrieval
                self.assignments['sensitivities'][nr] = cids

    def _read_potentials(self, pot_dir):
        """import potentials from a directory
        """
        if self.assignments['potentials'] is not None:
            print('Potentials already imported. Will not overwrite!')
            return
        else:
            self.assignments['potentials'] = {}

        pot_files = sorted(glob(pot_dir + os.sep + 'pot*.dat'))
        for nr, filename in enumerate(pot_files):
            with open(filename, 'r') as fid:
                pot_data = np.loadtxt(fid)

                nids = self.nodeman.add_data(
                    pot_data[:, 2:4],
                )
                # store cids for later retrieval
                self.assignments['potentials'][nr] = nids

    def get_potential(self, config_nr):
        """Return potential data for a given measurement configuration.

        Parameters
        ----------
        config_nr: int
            Number of the configurations. Starts at 0

        Returns
        -------
        pot_data: list with two numpy.ndarrays
            First array: magnitude potentials, second array: phase potentials

        """
        if self.assignments['potentials'] is None:
            self._check_state()
            if self.can_model:
                self.model(potentials=True)

        nids = self.assignments['potentials'][config_nr]
        pot_data = [self.nodeman.nodevals[nid] for nid in nids]
        return pot_data

    def get_sensitivity(self, config_nr):
        """return a sensitivity, as well as corresponding metadata, for a given
        measurement configuration. Indices start at zero.
        """
        if self.assignments['sensitivities'] is None:
            self._check_state()
            if self.can_model:
                self.model(sensitivities=True)
        cids = self.assignments['sensitivities'][config_nr]
        sens_data = [self.parman.parsets[cid] for cid in cids]
        meta_data = [self.parman.metadata[cid] for cid in cids]

        return sens_data, meta_data

    def plot_sensitivity(self, config_nr=None, sens_data=None,
                         mag_only=False, absv=False,
                         **kwargs):
        """Create a nice looking plot of the sensitivity distribution for the
        given configuration nr. Configs start at 1!

        Parameters
        ----------
        config_nr : int, optional
            The configuration number (starting with 0) to compute the
            sensitivity for.
        sens_data : Nx2 numpy.ndarray, optional
            If provided, use this data as sensitivity data (do not compute
            anything)
        mag_only : bool, optional
            Plot only the magnitude sensitivities
        absv : bool, optional
            If true, plot absolute values of sensitivity

        Returns
        -------
        fig
        ax

        Examples
        --------

        .. plot::

            import crtomo.debug
            import crtomo
            grid = crtomo.debug.get_grid(key=20)
            td = crtomo.tdMan(grid=grid)
            td.configs.add_to_configs([1, 5, 9, 13])
            cid_mag, cid_pha = td.add_homogeneous_model(25, 0)
            td.register_forward_model(cid_mag, cid_pha)
            td.model(sensitivities=True)
            fig, axes = td.plot_sensitivity(0)
            fig.tight_layout()
            fig.savefig('sens_plot.pdf', bboch_inches='tight')

        """

        def _rescale_sensitivity(sens_data):
            norm_value = np.abs(sens_data).max()
            sens_normed = sens_data / norm_value

            indices_gt_zero = sens_data > 1e-5
            indices_lt_zero = sens_data < -1e-5

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
            sens_data[:] = 0.5
            sens_data[indices_gt_zero] = y1
            sens_data[indices_lt_zero] = y2
            return sens_data

        assert config_nr is not None or sens_data is not None
        assert not (config_nr is not None and sens_data is not None)

        if config_nr is not None:
            cids = self.assignments['sensitivities'][config_nr]

            sens_mag = self.parman.parsets[cids[0]]
            sens_pha = self.parman.parsets[cids[1]]
        else:
            sens_mag = sens_data[:, 0]
            sens_pha = sens_data[:, 1]

        if absv:
            sens_mag = np.log10(np.abs(sens_mag) / np.abs(sens_mag).max())
            sens_pha = np.log10(np.abs(sens_pha) / np.abs(sens_pha).max())
            cbmin = sens_mag.min()
            cbmax = sens_mag.max()
        else:
            _rescale_sensitivity(sens_mag)
            _rescale_sensitivity(sens_pha)
            cbmin = 0
            cbmax = 1

        cmap_jet = matplotlib.cm.get_cmap('jet')
        colors = [cmap_jet(i) for i in np.arange(0, 1.1, 0.1)]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            'jetn9', colors, N=9)
        over = kwargs.get('over', 'orange')
        under = kwargs.get('under', 'cyan')
        bad = kwargs.get('bad', 'white')
        cmap.set_over(over)
        cmap.set_under(under)
        cmap.set_bad(bad)
        if mag_only:
            Nx = 1
        else:
            Nx = 2

        fig, axes = plt.subplots(1, Nx, figsize=(15 / 2.54, 12 / 2.54))
        axes = np.atleast_1d(axes)
        # magnitude
        ax = axes[0]
        cid = self.parman.add_data(sens_mag)
        fig, ax, cnorm, cmap, cb, sM = self.plot.plot_elements_to_ax(
            cid=cid,
            ax=ax,
            plot_colorbar=True,
            # cmap_name='seismic',
            cmap_name='jet_r',
            cbsegments=18,
            cbmin=cbmin,
            cbmax=cbmax,
            bad='white',
            # cbmin=-cblim,
            # cbmax=cblim,
            # converter=converter_pm_log10,
            # norm = colors.SymLogNorm(
            #     linthresh=0.03,
            #     linscale=0.03,
            #     vmin=-1.0,
            #     vmax=1.0
            # ),
            # xmin=-0.25,
            # xmax=10,
            # zmin=-2,
        )
        if not absv:
            cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
            cb.set_ticklabels([
                '-1',
                r'$-10^{-2.5}$',
                '0',
                r'$10^{-2.5}$',
                '1',
            ])

        # self.plot.plot_elements_to_ax(
        #     cid=cids[0],
        #     ax=ax,
        # )

        if not mag_only:
            cid = self.parman.add_data(sens_pha)
            # plot phase
            ax = axes[1]
            fig, ax, cnorm, cmap, cb, sM = self.plot.plot_elements_to_ax(
                cid=cid,
                ax=ax,
                plot_colorbar=True,
                # cmap_name='seismic',
                cmap_name='jet_r',
                cbsegments=18,
                cbmin=cbmin,
                cbmax=cbmax,
                bad='white',
            )
            if not absv:
                cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
                cb.set_ticklabels([
                    '-1',
                    r'$-10^{-2.5}$',
                    '0',
                    r'$10^{-2.5}$',
                    '1',
                ])

        fig.tight_layout()

        return fig, axes

    def read_voltages(self, voltage_file):
        """import voltages from a volt.dat file

        Parameters
        ----------
        voltage_file : string
            Path to volt.dat file
        """

        measurements_raw = np.loadtxt(
            voltage_file,
            skiprows=1,
        )
        measurements = np.atleast_2d(measurements_raw)

        # extract measurement configurations
        A = (measurements[:, 0] / 1e4).astype(int)
        B = (measurements[:, 0] % 1e4).astype(int)
        M = (measurements[:, 1] / 1e4).astype(int)
        N = (measurements[:, 1] % 1e4).astype(int)
        ABMN = np.vstack((A, B, M, N)).T

        if self.configs.configs is None:
            self.configs.configs = ABMN
        else:
            # configurations don't match
            if not np.all(ABMN == self.configs.configs):
                for nr, (old_config, new_config) in enumerate(zip(
                        self.configs.configs, ABMN)):

                    if np.all(old_config == new_config):
                        continue
                    # check polarity
                    current_electrodes_are_equal = np.all(
                        old_config[0:2] == new_config[0:2]
                    )
                    voltage_electrodes_are_switched = np.all(
                        old_config[2:4] == new_config[4:1:-1]
                    )

                    if(current_electrodes_are_equal and
                       voltage_electrodes_are_switched):

                        if len(self.configs.measurements.keys()) > 0:
                            raise Exception(
                                'need to switch electrode polarity, but ' +
                                'there are already measurements stored for ' +
                                'the old configuration!')
                        else:
                            # switch M/N in configurations
                            self.configs.configs[nr, :] = new_config
                    else:
                        raise Exception(
                            'There was an error matching configurations of ' +
                            'voltages with configurations already imported'
                        )

        # add measurements to the config instance
        mid_mag = self.configs.add_measurements(
            measurements[:, 2]
        )
        mid_pha = self.configs.add_measurements(
            measurements[:, 3]
        )

        self.assignments['measurements'] = [mid_mag, mid_pha]

    def _model(self, voltages, sensitivities, potentials, tempdir,
               silent=False):
        self._check_state()
        if self.can_model:
            if not silent:
                print('attempting modeling')
            pwd = os.getcwd()
            os.chdir(tempdir)
            # store the configuration temporarily
            cfg_save = self.crmod_cfg.copy()

            self.crmod_cfg['write_volts'] = voltages
            self.crmod_cfg['write_pots'] = potentials
            self.crmod_cfg['write_sens'] = sensitivities

            self.save_to_tomodir('.')
            os.chdir('exe')
            binary = CRBin.get('CRMod')
            return_text = subprocess.check_output(
                binary,
                shell=True,
                stderr=subprocess.STDOUT,
            )
            return_text
            # restore the configuration
            self.crmod_cfg = cfg_save
            # if return_code != 0:
            #     raise Exception('There was an error using CRMod')

            os.chdir(pwd)
            self._read_modeling_results(
                tempdir + os.sep + 'mod', silent=silent)
            return 1
        else:
            print(
                'Sorry, no measurements present, cannot model yet'
            )
            return None

    def model(self,
              voltages=True,
              sensitivities=False,
              potentials=False,
              output_directory=None,
              silent=False,
              ):
        """Forward model the tomodir and read in the results
        """
        self._check_state()
        if self.can_model:
            if output_directory is not None:
                if not os.path.isdir(output_directory):
                    os.makedirs(output_directory)
                    tempdir = output_directory
                    self._model(voltages, sensitivities, potentials, tempdir)
                else:
                    raise IOError(
                        'output directory already exists: {0}'.format(
                            output_directory
                        )
                    )
            else:
                with tempfile.TemporaryDirectory(dir=self.tempdir) as tempdir:
                    self._model(
                        voltages, sensitivities, potentials, tempdir,
                        silent=silent
                    )

            return 1
        else:
            print('Sorry, not all required information to model are present')
            print('Check:')
            print('1) configurations present: self.configs.configs')
            print('2) is a model present')
            return None

    def _invert(self, tempdir, catch_output=True, **kwargs):
        """Internal function than runs an inversion using CRTomo.

        Parameters
        ----------
        tempdir : string
            directory which to use as a tomodir
        catch_output : bool, optional
            if True, catch all outputs of the CRTomo call (default: True)
        cores : int, optional
            how many cores to use. (default 2)
        """
        nr_cores = kwargs.get('cores', 2)
        print('attempting inversion in directory: {0}'.format(tempdir))
        pwd = os.getcwd()
        os.chdir(tempdir)

        self.save_to_tomodir('.')
        os.chdir('exe')
        binary = CRBin.get('CRTomo')
        print('Using binary: {0}'.format(binary))
        print('calling CRTomo')
        # store env variable
        env_omp = os.environ.get('OMP_NUM_THREADS', '')
        os.environ['OMP_NUM_THREADS'] = '{0}'.format(nr_cores)
        if catch_output:
            subprocess.check_output(
                binary,
                shell=True,
                stderr=subprocess.STDOUT,
            )
        else:
            subprocess.call(
                binary,
                shell=True,
            )
        # reset environment variable
        os.environ['OMP_NUM_THREADS'] = env_omp

        print('finished')

        os.chdir(pwd)
        self.read_inversion_results(tempdir)

    def invert(self, output_directory=None, catch_output=True, **kwargs):
        """Invert this instance, and import the result files

        No directories/files will be overwritten. Raise an IOError if the
        output directory exists.

        Parameters
        ----------
        output_directory: string, optional
            use this directory as output directory for the generated tomodir.
            If None, then a temporary directory will be used that is deleted
            after import.
        catch_output: bool, optional
            Do not show CRTomo output
        cores: int, optional
            how many cores to use for CRTomo

        Returns
        -------
        return_code: bool
            Return 0 if the inversion completed successfully. Return 1 if no
            measurements are present.
        """
        self._check_state()
        if self.can_invert:
            if output_directory is not None:
                if not os.path.isdir(output_directory):
                    os.makedirs(output_directory)
                    tempdir = output_directory
                    self._invert(tempdir, catch_output, **kwargs)
                else:
                    raise IOError(
                        'output directory already exists: {0}'.format(
                            output_directory
                        )
                    )
            else:
                with tempfile.TemporaryDirectory(dir=self.tempdir) as tempdir:
                    self._invert(tempdir, catch_output, **kwargs)

            return 0
        else:
            print(
                'Sorry, no measurements present, cannot model yet'
            )
            return 1

    def read_inversion_results(self, tomodir):
        """Import inversion results from a tomodir into this instance

        WARNING: Not finished!
        """
        self._read_inversion_results(tomodir)
        self._read_inv_ctr(tomodir)
        self._read_resm_m(tomodir)
        self._read_eps_ctr(tomodir)

    def _read_inversion_results(self, tomodir):
        """Import resistivity magnitude/phase and real/imaginary part of
        conductivity for all iterations
        """
        basedir = tomodir + os.sep + 'inv' + os.sep
        print(basedir)
        inv_mag = sorted(glob(basedir + 'rho*.mag'))
        inv_pha = sorted(glob(basedir + 'rho*.pha'))
        inv_sig = sorted(glob(basedir + 'rho*.sig'))

        assert len(inv_mag) == len(inv_pha)
        assert len(inv_mag) == len(inv_sig)

        pids_mag = [self.parman.load_inv_result(filename) for filename in
                    inv_mag]
        pids_pha = [self.parman.load_inv_result(filename) for filename in
                    inv_pha]
        pids_sig = [self.parman.load_inv_result(filename, columns=[0, 1]) for
                    filename in inv_sig]
        self.assignments['inversion'] = {
            'rmag': pids_mag,
            'rpha': pids_pha,
            'cre_cim': pids_sig,
        }

    def plot_eps_data_hist(self, dfs):
        """Plot histograms of data residuals and data error weighting

        TODO:
            * add percentage of data below/above the RMS value
        """
        # check if this is a DC inversion
        if 'datum' in dfs[0]:
            dc_inv = True
        else:
            dc_inv = False

        nr_y = len(dfs)
        size_y = 5 / 2.54 * nr_y
        if dc_inv:
            nr_x = 1
        else:
            nr_x = 3
        size_x = 15 / 2.54

        fig, axes = plt.subplots(nr_y, nr_x, figsize=(size_x, size_y))
        axes = np.atleast_2d(axes)

        # plot initial data errors
        df = dfs[0]
        if dc_inv:
            ax = axes[0, 0]
            ax.hist(
                df['datum'] / df['eps_r'],
                100,
            )
            ax.set_xlabel(r'$-log(|R|) / \epsilon_r$')
            ax.set_ylabel(r'count')
        else:
            # complex inversion
            ax = axes[0, 0]
            ax.hist(
                df['-log(|R|)'] / df['eps'],
                100,
            )
            ax.set_xlabel(r'$-log(|R|)$')
            ax.set_ylabel(r'count')

            ax = axes[0, 1]
            ax.hist(
                df['-log(|R|)'] / df['eps_r'],
                100,
            )
            ax.set_xlabel(r'$-log(|R|) / \epsilon_r$')
            ax.set_ylabel(r'count')

            ax = axes[0, 2]
            phase_data = df['-Phase(rad)'] / df['eps_p']
            if not np.all(np.isinf(phase_data) | np.isnan(phase_data)):
                ax.hist(
                    phase_data,
                    100,
                )
            ax.set_xlabel(r'$-\phi[rad] / \epsilon_p$')
            ax.set_ylabel(r'count')

        # iterations
        for it, df in enumerate(dfs[1:]):
            ax = axes[1 + it, 0]
            ax.hist(
                df['psi'],
                100
            )
            rms = np.sqrt(
                1 / df['psi'].shape[0] *
                np.sum(
                    df['psi'] ** 2
                )
            )
            ax.axvline(rms, color='k', linestyle='dashed')
            ax.set_title('iteration: {0}'.format(it))
            ax.set_xlabel('psi')
            ax.set_ylabel(r'count')

            ax = axes[1 + it, 1]
            Rdat = df['Re(d)']
            Rmod = df['Re(f(m))']

            ax.scatter(
                Rdat,
                Rmod,
            )
            ax.set_xlabel(r'$log(R_{data}~[\Omega])$')
            ax.set_ylabel(r'$log(R_{mod}~[\Omega])$')

            ax = axes[1 + it, 2]
            phidat = df['Im(d)']
            phimod = df['Im(f(m))']

            ax.scatter(
                phidat,
                phimod,
            )
            ax.set_xlabel(r'$\phi_{data}~[mrad]$')
            ax.set_ylabel(r'$\phi_{mod}~[mrad]$')

        fig.tight_layout()
        fig.savefig('eps_plot_hist.png', dpi=300)

    def plot_eps_data(self, dfs):
        # check if this is a DC inversion
        if 'datum' in dfs[0]:
            dc_inv = True
        else:
            dc_inv = False

        nr_y = len(dfs)
        size_y = 5 / 2.54 * nr_y
        if dc_inv:
            nr_x = 1
        else:
            nr_x = 3
        size_x = 15 / 2.54

        fig, axes = plt.subplots(nr_y, nr_x, figsize=(size_x, size_y))
        axes = np.atleast_2d(axes)

        # plot initial data errors
        df = dfs[0]
        if dc_inv:
            ax = axes[0, 0]
            ax.scatter(
                df['datum'] / df['eps_r'],
                df['eps_r'],
            )
            ax.set_xlabel(r'$-log(|R|)$')
            ax.set_ylabel(r'$eps_r$')
        else:
            # complex inversion
            ax = axes[0, 0]
            ax.scatter(
                df['-log(|R|)'] / df['eps'],
                df['eps'],
            )
            ax.set_xlabel(r'$-log(|R|)$')
            ax.set_ylabel(r'$eps$')

            ax = axes[0, 1]
            ax.scatter(
                df['-log(|R|)'] / df['eps_r'],
                df['eps_p'],
            )
            ax.set_xlabel(r'$-log(|R|)$')
            ax.set_ylabel(r'$eps$')

            ax = axes[0, 2]
            ax.scatter(
                df['-Phase(rad)'] / df['eps_p'],
                df['eps_p'],
            )
            ax.set_xlabel(r'$-log(|R|)$')
            ax.set_ylabel(r'$eps$')

        # iterations
        for it, df in enumerate(dfs[1:]):
            ax = axes[1 + it, 0]
            ax.scatter(
                range(0, df.shape[0]),
                df['psi'],
            )
            rms = np.sqrt(
                1 / df['psi'].shape[0] *
                np.sum(
                    df['psi'] ** 2
                )
            )
            ax.axhline(rms, color='k', linestyle='dashed')
            ax.set_title('iteration: {0}'.format(it))
            ax.set_xlabel('config nr')
            ax.set_ylabel(r'$\Psi$')

        fig.tight_layout()
        fig.savefig('eps_plot.png', dpi=300)

    @staticmethod
    def _read_eps_ctr(tomodir):
        """Parse a CRTomo eps.ctr file.

        TODO: change parameters to only provide eps.ctr file

        Parameters
        ----------
        tomodir: string
            Path to directory path

        Returns
        -------


        """
        epsctr_file = tomodir + os.sep + 'inv' + os.sep + 'eps.ctr'
        if not os.path.isfile(epsctr_file):
            print('eps.ctr not found: {0}'.format(epsctr_file))
            print(os.getcwd())
            return 1

        with open(epsctr_file, 'r') as fid:
            lines = fid.readlines()
        group = itertools.groupby(lines, lambda x: x == '\n')
        dfs = []
        # group
        for x in group:
            # print(x)
            if not x[0]:
                data = [y for y in x[1]]
                if data[0].startswith('IT') or data[0].startswith('PIT'):
                    del(data[0])
                data[0] = data[0].replace('-Phase (rad)', '-Phase(rad)')
                tfile = StringIO(''.join(data))
                df = pd.read_csv(
                    tfile,
                    delim_whitespace=True,
                    na_values=['Infinity'],
                )
                dfs.append(df)
        return dfs

    def _read_inv_ctr(self, tomodir):
        """Read in selected results of the inv.ctr file

        Parameters
        ----------
        tomodir: string
            directory path to a tomodir

        Returns
        -------
        inv_ctr:    ?
            structure containing inv.ctr data

        """
        invctr_file = tomodir + os.sep + 'inv' + os.sep + 'inv.ctr'
        if not os.path.isfile(invctr_file):
            print('inv.ctr not found: {0}'.format(invctr_file))
            print(os.getcwd())
            return 1

        # read header
        with open(invctr_file, 'r') as fid:
            lines = fid.readlines()

        # check for robust inversion
        is_robust_inversion = False
        nr_of_data_points = None
        for i, line in enumerate(lines):
            if line.startswith('***PARAMETERS***'):
                raw_value = lines[i + 7].strip()[0]
                if raw_value == 'T':
                    is_robust_inversion = True
            if line.startswith('# Data points'):
                nr_of_data_points = int(line[15:].strip())

        print('is robust', is_robust_inversion)

        # find section that contains the iteration data
        for i, line in enumerate(lines):
            if line.strip().startswith('ID it.'):
                break

        # TODO: check for robust iteration

        # we have three types of lines:
        # 1. first iteration line
        # 2. other main iteration lines
        # 3. update lines

        # prepare regular expressions for these three types, each in two
        # flavors: robust and non-robust

        """
! first iteration, robust
100 FORMAT (t1,a3,t5,i3,t11,g10.4,t69,g10.4,t81,g10.4,t93,i4,t105,g9.3)
! first iteration, non-robust
101 FORMAT (t1,a3,t5,i3,t11,g10.4,t69,g10.4,t81,g10.4,t93,i4)

! other iterations, robust
110 FORMAT (t1,a3,t5,i3,t11,g10.4,t23,g10.4,t34,g10.4,t46,g10.4,t58,&
i6,t69,g10.4,t81,g10.4,t93,i4,t105,g9.3,t117,f5.3)
! other iterations, non-robust
111 FORMAT (t1,a3,t5,i3,t11,g10.4,t23,g10.4,t34,g10.4,t46,g10.4,t58,&
i6,t69,g10.4,t81,g10.4,t93,i4,t105,f5.3)

! update iterations, non-robust
105 FORMAT (t1,a3,t5,i3,t11,g10.4,t23,g9.3,t34,g10.4,t46,g10.4,t58,&
i6,t105,f5.3)
! update iterations, robust
106 FORMAT (t1,a3,t5,i3,t11,g10.4,t23,g9.3,t34,g10.4,t46,g10.4,t58,&
i6,t105,g9.3,t117,f5.3)

        """

        # this identifies a float number, or a NaN value
        reg_float = ''.join((
            '((?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)',
            '|',
            '(?:NaN))'
        ))

        reg_int = '(\d{1,3})'

        # (t1,a3,t5,i3,t11,g10.4,t69,g10.4,t81,g10.4,t93,i4)
        # first iteration line of non-robust inversion
        reg_it1_norob = ''.join((
            '([a-zA-Z]{1,3})',
            ' *' + reg_int,
            ' *' + reg_float,
            ' *' + reg_float,
            ' *' + reg_float,
            ' *' + reg_int,
        ))
        # first iteration line of robust inversion
        reg_it1_robust = ''.join((
            '([a-zA-Z]{1,3})',
            ' *(\d{1,3})',
            ' *' + reg_float,   # data RMS
            ' *' + reg_float,   # mag RMS
            ' *' + reg_float,   # pha RMS
            ' *' + reg_int,     # nr excluded data
            ' *' + reg_float,   # L1-ratio
        ))

        # second-to-last iterations, robust
        reg_it2plus_rob = ''.join((
            '([a-zA-Z]{1,3})',
            ' *(\d{1,3})',
            ' *' + reg_float,   # data RMS
            ' *' + reg_float,   # stepsize
            ' *' + reg_float,   # lambda
            ' *' + reg_float,   # roughness
            ' *' + reg_int,     # CG-steps
            ' *' + reg_float,   # mag RMS
            ' *' + reg_float,   # pha RMS
            ' *' + reg_int,     # nr excluded data
            ' *' + reg_float,   # l1-ratio
            ' *' + reg_float,   # steplength
        ))

        # second-to-last iterations, non-robustk
        # (t1,a3,t5,i3,t11,g10.4,t23,g10.4,t34,g10.4,t46,g10.4,t58,&
        # i6,t69,g10.4,t81,g10.4,t93,i4,t105,f5.3)
        reg_it2plus_norob = ''.join((
            '([a-zA-Z]{1,3})',
            ' *(\d{1,3})',
            ' *' + reg_float,   # data RMS
            ' *' + reg_float,   # stepsize
            ' *' + reg_float,   # lambda
            ' *' + reg_float,   # roughness
            ' *' + reg_int,     # CG-steps
            ' *' + reg_float,   # mag RMS
            ' *' + reg_float,   # pha RMS
            ' *' + reg_int,     # nr excluded data
            ' *' + reg_float,   # steplength
        ))

        # update robust
        reg_update_rob = ''.join((
            '([a-zA-Z]{1,3})',
            ' *(\d{1,3})',
            ' *' + reg_float,  # data RMS
            ' *' + reg_float,  # stepsize
            ' *' + reg_float,  # lambda
            ' *' + reg_float,  # roughness
            ' *' + reg_int,  # CG-steps
            ' *' + reg_float,  # l1ratio
        ))
        # update non-robust
        reg_update_norob = ''.join((
            '([a-zA-Z]{1,3})',
            ' *(\d{1,3})',
            ' *' + reg_float,  # data RMS
            ' *' + reg_float,  # stepsize
            ' *' + reg_float,  # lambda
            ' *' + reg_float,  # roughness
            ' *' + reg_int,  # CG-steps
            ' *' + reg_float,  # steplength
        ))

        # iteration counter
        current_iteration = 0
        iterations = []

        for line in lines[i:]:
            linec = line.strip()
            if linec.startswith('IT') or linec.startswith('PIT'):
                if linec[0:3].strip() == 'IT':
                    it_type = 'DC/IP'
                else:
                    it_type = 'FPI'

                values = None

                # main iterations
                if is_robust_inversion:
                    if current_iteration == 0:
                        # first iteration, robust
                        g = re.compile(reg_it1_robust).search(linec).groups()
                        keyfuncs = [
                            (None, None),
                            ('iteration', int),
                            ('dataRMS', float),
                            ('magRMS', float),
                            ('phaRMS', float),
                            ('nrdata', int),
                            ('l1ratio', float),
                        ]
                        values = {}
                        for value, (key, func) in zip(g, keyfuncs):
                            if key is not None:
                                values[key] = func(value)
                    else:
                        # second-to-last iterations, robust
                        g = re.compile(
                            reg_it2plus_rob
                        ).search(linec).groups()
                        keyfuncs = [
                            (None, None),
                            ('iteration', int),
                            ('dataRMS', float),
                            ('stepsize', float),
                            ('lambda', float),
                            ('roughness', float),
                            ('cgsteps', int),
                            ('magRMS', float),
                            ('phaRMS', float),
                            ('nrdata', int),
                            ('l1ratio', float),
                            ('steplength', float),
                        ]
                        values = {}
                        for value, (key, func) in zip(g, keyfuncs):
                            if key is not None:
                                values[key] = func(value)
                    values['type'] = 'main'
                    values['main_iteration'] = current_iteration
                    values['it_type'] = it_type
                    iterations.append(values)
                    current_iteration += 1
                else:
                    if current_iteration == 0:
                        # non-robust, first iteration
                        g = re.compile(reg_it1_norob).search(linec).groups()

                        keyfuncs = [
                            (None, None),
                            ('iteration', int),
                            ('dataRMS', float),
                            ('magRMS', float),
                            ('phaRMS', float),
                            ('nrdata', int)
                        ]
                        values = {}
                        for value, (key, func) in zip(g, keyfuncs):
                            if key is not None:
                                values[key] = func(value)
                    else:
                        g = re.compile(
                            reg_it2plus_norob
                        ).search(linec).groups()
                        keyfuncs = [
                            (None, None),
                            ('iteration', int),
                            ('dataRMS', float),
                            ('stepsize', float),
                            ('lambda', float),
                            ('roughness', float),
                            ('cgsteps', int),
                            ('magRMS', float),
                            ('phaRMS', float),
                            ('nrdata', int),
                            ('steplength', float),
                        ]
                        values = {}
                        for value, (key, func) in zip(g, keyfuncs):
                            if key is not None:
                                values[key] = func(value)
                    values['type'] = 'main'
                    values['it_type'] = it_type
                    values['main_iteration'] = current_iteration
                    iterations.append(values)
                    current_iteration += 1
            elif linec.startswith('UP'):
                # update iterations
                if is_robust_inversion:
                    # robust
                    g = re.compile(
                        reg_update_rob
                    ).search(linec).groups()
                    keyfuncs = [
                        (None, None),
                        ('iteration', int),
                        ('dataRMS', float),
                        ('stepsize', float),
                        ('lambda', float),
                        ('roughness', float),
                        ('cgsteps', int),
                        ('l1-ratio', float),
                    ]
                    values = {}
                    for value, (key, func) in zip(g, keyfuncs):
                        if key is not None:
                            values[key] = func(value)
                else:
                    g = re.compile(
                        reg_update_norob
                    ).search(linec).groups()
                    keyfuncs = [
                        (None, None),
                        ('iteration', int),
                        ('dataRMS', float),
                        ('stepsize', float),
                        ('lambda', float),
                        ('roughness', float),
                        ('cgsteps', int),
                        ('steplength', float),
                    ]
                    values = {}
                    for value, (key, func) in zip(g, keyfuncs):
                        if key is not None:
                            values[key] = func(value)
                values['type'] = 'update'
                values['it_type'] = it_type
                values['main_iteration'] = current_iteration
                iterations.append(values)

        df = pd.DataFrame(iterations)
        df = df.reindex_axis([
            'iteration',
            'main_iteration',
            'it_type',
            'type',
            'dataRMS',
            'magRMS',
            'phaRMS',
            'lambda',
            'roughness',
            'cgsteps',
            'nrdata',
            'steplength',
            'stepsize',
            'l1ratio',
        ], axis=1)

        df['nrdata'] = nr_of_data_points - df['nrdata']
        return df

    def plot_inversion_evolution(self, df, filename):
        g = df.groupby('iteration')
        fig, ax = plt.subplots()
        # update iterations
        for name, group in g:
            # plot update evolution
            updates = group.query('type == "update"')
            ax.scatter(
                np.ones(updates.shape[0]) * name,
                updates['lambda'],
                color='k',
            )

        # main iterations
        main = df.query('type == "main"')
        ax.plot(
            main['main_iteration'],
            main['lambda'],
            '.-',
            color='k',
        )
        ax.set_ylabel(r'$\lambda$')
        ax2 = ax.twinx()
        ax2.plot(
            main['main_iteration'],
            main['dataRMS'],
            '.-',
            color='r',
        )
        ax2.set_ylabel('data RMS')

        fig.tight_layout()
        fig.savefig('inversion_evolution.png', dpi=300)

    def _read_resm_m(self, tomodir):
        """Read in the resolution matrix of an inversion

        Parameters
        ----------
        tomodir: string
            directory path to a tomodir

        """
        resm_file = tomodir + os.sep + 'inv' + os.sep + 'res_m.diag'
        if not os.path.isfile(resm_file):
            print('res_m.diag not found: {0}'.format(resm_file))
            print(os.getcwd())
            return 1

        # read header
        with open(resm_file, 'rb') as fid:
            first_line = fid.readline().strip()
            header_raw = np.fromstring(first_line, count=4, sep=' ')
            header_raw
            # nr_cells = int(header_raw[0])
            # lam = float(header_raw[1])

            subdata = np.genfromtxt(fid)
            print(subdata.shape)
            pid = self.parman.add_data(subdata[:, 0])
            self.assignments['resm'] = pid

    def register_measurements(self, mag, pha=None):
        """Register measurements as magnitude/phase measurements used for the
        inversion

        Parameters
        ----------
        mag: int|numpy.ndarray
            magnitude measurement id for the corresponding measurement data in
            self.configs.measurements. If mag is a numpy.ndarray, assume
            mag to be the data itself an register it
        pha: int, optional
            phase measurement id for the corresponding measurement data in
            self.configs.measurements. If not present, a new measurement set
            will be added with zeros only.
        """
        assert len(mag.squeeze().shape) == 1
        if isinstance(mag, np.ndarray):
            mid_mag = self.configs.add_measurements(mag)
        else:
            mid_mag = mag

        if pha is not None:
            if isinstance(pha, np.ndarray):
                mid_pha = self.configs.add_measurements(pha)
            else:
                mid_pha = pha
        else:
            mid_pha = self.configs.add_measurements(
                np.zeros_like(
                    self.configs.measurements[mid_mag]
                )
            )
        self.assignments['measurements'] = [mid_mag, mid_pha]

    def register_forward_model(self, pid_mag, pid_pha):
        """Register parameter sets as the forward models for magnitude and
        phase

        Parameters
        ----------
        pid_mag: int
            parameter id corresponding to the magnitude model
        pid_pha: int
            parameter id corresponding to the phase model
        """
        self.register_magnitude_model(pid_mag)
        self.register_phase_model(pid_pha)

    def register_magnitude_model(self, pid):
        """Set a given parameter model to the forward magnitude model
        """
        if self.assignments['forward_model'] is None:
            self.assignments['forward_model'] = [None, None]

        self.assignments['forward_model'][0] = pid

    def register_phase_model(self, pid):
        """Set a given parameter model to the forward phase model
        """
        if self.assignments['forward_model'] is None:
            self.assignments['forward_model'] = [None, None]

        self.assignments['forward_model'][1] = pid

    def add_homogeneous_model(self, magnitude, phase=0):
        """Add a homogeneous resistivity model to the tomodir. This is useful
        for synthetic measurements.

        Parameters
        ----------
        magnitude : float
            magnitude [Ohm m] value of the homogeneous model
        phase : float, optional
            phase [mrad] value of the homogeneous model


        Returns
        -------
        pid_mag : int
            ID value of the parameter set of the magnitude model
        pid_pha : int
            ID value of the parameter set of the phase model

        Note that the parameter sets are automatically registered as the
        forward models for magnitude and phase values.
        """
        if self.assignments['forward_model'] is not None:
            print('model already set, will overwrite')

        # generate distributions
        magnitude_model = np.ones(self.grid.nr_of_elements) * magnitude
        phase_model = np.ones(self.grid.nr_of_elements) * phase
        pid_mag = self.parman.add_data(magnitude_model)
        pid_pha = self.parman.add_data(phase_model)

        self.assignments['forward_model'] = [pid_mag, pid_pha]
        return pid_mag, pid_pha

    def check_measurements_against_sensitivities(
            self, magnitude, phase=0, return_plot=False):
        """Check for all configurations if the sensitivities add up to a given
        homogeneous model

        Parameters
        ----------
        magnitude : float
            magnitude used for the homogeneous model
        phase : float, optional, default=0
            phase value used for the homogeneous model
        return_plot : bool, optional, default=False
            create a plot analyzing the differences

        Returns
        -------
        results : Nx6 numpy.ndarray
            Results of the analysis.

            * magnitude measurement [Ohm]
            * sum of sensitivities [Volt]
            * relative deviation of sensitivity-sum from measurement [in
              percent]

        fig : matplotlib.figure, optional
            figure object. Only returned of return_plot=True
        axes : list
            list of axes corresponding to the figure

        Examples
        --------

        >>> #!/usr/bin/python
            import crtomo.tdManager as CRtdMan
            tdm = CRtdMan.tdMan(
                    elem_file='grid/elem.dat',
                    elec_file='grid/elec.dat',
                    config_file='config/config.dat',
            )
            results, fig, axes = tdm.check_measurements_against_sensitivities(
                    magnitude=100,
                    phase=-10,
                    return_plot=True
            )
            fig.savefig('sensitivity_comparison.png', dpi=300)

        """
        # generate a temporary tdMan instance
        tdm = tdMan(
            grid=self.grid,
            configs=self.configs,
        )
        tdm.add_homogeneous_model(magnitude, phase)
        measurements = tdm.measurements()

        Z = measurements[:, 0] * np.exp(1j * measurements[:, 1] / 1000)

        results = []
        for nr in range(0, tdm.configs.nr_of_configs):
            sensitivities = tdm.get_sensitivity(nr)
            sens_re = sensitivities[0][0]
            sens_im = sensitivities[0][1]

            sens_mag = 1.0 / measurements[nr, 0] * (
                np.real(Z[nr]) * sens_re + np.imag(Z[nr]) * sens_im
            )

            V_mag_from_sens = sens_mag.sum() / magnitude
            if phase != 0:
                outer = 1 / (1 + (np.imag(Z[nr]) / np.real(Z[nr])) ** 2)
                inner1 = - sens_re / np.real(Z[nr]) ** 2 * np.imag(Z[nr])
                inner2 = sens_im * np.real(Z[nr])
                sens_pha = outer * (inner1 + inner2)

                V_pha_from_sens = sens_pha.sum() / phase
            else:
                V_pha_from_sens = None

            print(
                'WARNING: We still do not know where the minus sign comes ' +
                'from!'
            )
            V_mag_from_sens *= -1

            results.append((
                measurements[nr][0],
                V_mag_from_sens,
                (measurements[nr][0] - V_mag_from_sens) / measurements[nr][0] *
                100,
                measurements[nr][1],
                V_pha_from_sens,
                (measurements[nr][1] - V_mag_from_sens) / measurements[nr][1] *
                100,
            ))
        results = np.array(results)

        if return_plot:
            nr_x = 2
            if phase == 0:
                nr_x = 1
            fig, axes = plt.subplots(1, nr_x, figsize=(15 / 2.54, 7 / 2.54))
            fig.suptitle('Comparison sum of sensitivities to measurements')
            # plot phase first
            if phase != 0:
                ax = axes[1]
                ax.plot(results[:, 5], '.')
                ax.set_xlabel('configuration number')
                ax.set_ylabel(
                    r'$\frac{V_i^{\mathrm{pha}} - ' +
                    r' \sum s_{ij}^{\mathrm{pha}} \cdot ' +
                    r'\phi_0}{V_i}~[\%]$'
                )

                # set ax for magnitude plot
                ax = axes[0]
            else:
                ax = axes

            ax.plot(results[:, 2], '.')
            ax.set_xlabel('configuration number')
            # ax.set_ylabel('deviation from magnitude measurement [\%]')
            ax.set_ylabel(
                r'$\frac{V_i^{\mathrm{mag}} - ' +
                r'\sum s_{ij}^{\mathrm{mag}} \cdot ' +
                r'\sigma_0}{V_i}~[\%]$'
            )

            fig.tight_layout()
            return results, fig, axes
        else:
            return results

    def show_parset(self, pid):
        """Plot a given parameter set
        """
        fig, ax = plt.subplots()
        self.plot.plot_elements_to_ax(pid, ax=ax)
        return fig, ax

    def plot_forward_models(self):
        """Plot the forward models (requires the forward models to be loaded)
        """
        pids_rho = self.assignments.get('forward_model', None)
        if pids_rho is None:
            raise Exception('you need to load the forward model first')
        fig, axes = plt.subplots(1, 2, figsize=(16 / 2.54, 8 / 2.54))
        ax = axes[0]
        self.plot.plot_elements_to_ax(
            pids_rho[0],
            ax=ax,
            plot_colorbar=True,
            cblabel=r'$|\rho| [\Omega m]$',
        )

        ax = axes[1]
        self.plot.plot_elements_to_ax(
            pids_rho[1],
            ax=ax,
            plot_colorbar=True,
            cblabel=r'$\phi [mrad]$',
        )
        fig.tight_layout()
        return fig, axes

