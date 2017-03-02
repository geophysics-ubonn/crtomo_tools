# *-* coding: utf-8 *-*
"""A digital representation of a tomodir, i.e., a single-frequency inversion.

TODO
----

Modelling
---------

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

  Provide corresponding filter functions, or use EDF for that (do we want to
  depend on EDF? I think it's ok to do this).

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
import glob
import os
import tempfile
import numpy as np
import subprocess

from crtomo.mpl_setup import *
import crtomo.binaries as CRBin
import crtomo.grid as CRGrid
import crtomo.nodeManager as nM
import crtomo.parManager as pM
import crtomo.configManager as cConf
import crtomo.cfg as CRcfg
import crtomo.plotManager as PlotManager


class tdMan(object):
    """Manage tomodirs

    """
    def __init__(self, **kwargs):
        """The following initialization permutations are possible:

            * initialize empty
            * initialize from an exisiting tomodir
                * tomodir: [tomodir path]
            * supply one or more of the building blocks of a tomodir
                * grid: crtomo.grid.crt_grid instance
                * crmod_cfg: crtomo.cfg.crmod_cfg instance
                * crtomo_cfg: crmod.cfg.crtomo_cfg instance

        Parameters
        ----------

        grid: crtomo.grid.crt_grid
            A fully initialized grid object
        """
        # these variables will be filled later
        self.grid = None
        self.crmod_cfg = None
        self.crtomo_cfg = None
        self.nodeman = None
        self.parman = None
        self.configs = None
        self.plotman = None
        # we need a struct to organize the assignments
        self.assignments = {
            # should contain a two-item list with ids in parman
            'forward_model': None,
            # should contain one id for the config object
            'measurements': None,
            # store sensitivity cids here
            'sensitivities': None,
            # store potential nids here
            'potentials': None,
        }
        # indicates if all information for modeling are present
        self.can_model = False
        # indicates if all information for inversion are present
        self.can_invert = False

        self._initialize_components(kwargs)

    def _initialize_components(self, kwargs):
        """initialize the various components using the supplied **kwargs
        """

        # load/assign grid
        if 'grid' in kwargs:
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

        config = kwargs.get(
            'configs',
            cConf.ConfigManager(nr_of_electrodes=self.grid.nr_of_electrodes)
        )
        self.configs = config

        config_file = kwargs.get('config_file', None)
        if config_file is not None:
            self.configs.load_crmod_config(config_file)

        voltage_file = kwargs.get('volt_file', None)
        if voltage_file is not None:
            cids = self.configs.load_crmod_volt(voltage_file)
            self.assignments['measurements'] = cids

        self.plotman = PlotManager.plotManager(
            grid=self.grid,
            np=self.nodeman,
            pm=self.parman,
        )

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

    def save_to_tomodir(self, directory):
        """Save the tomodir instance to a directory structure.

        TODO
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

    def _save_sensitivities(self, directory):
        """save sensitivities to a directory
        """
        print('saving sensitivities')
        digits = int(np.ceil(np.log10(self.configs.configs.shape[0])))
        for i in range(0, self.configs.configs.shape[0]):
            sens_data, meta_data = self.get_sensitivity(i)
            filename_raw = 'sens{0:0' + '{0}'.format(digits) + '}.dat'
            filename = directory + os.sep + filename_raw.format(i + 1)

            grid_x, grid_z = self.grid.get_element_centroids()
            all_data = np.vstack((
                grid_x,
                grid_z,
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
                print('Cannot model')
                return

        # retrieve measurements
        cids = self.assignments['measurements']
        measurements = np.vstack((
            self.configs.measurements[cids[0]],
            self.configs.measurements[cids[1]],
        )).T
        return measurements

    def _read_modeling_results(self, directory):
        """Read modeling results from a given mod/ directory. Possible values
        to read in are:

            * voltages
            * potentials
            * sensitivities

        """

        voltage_file = directory + os.sep + 'volt.dat'
        if os.path.isfile(voltage_file):
            print('reading voltages')
            self._read_voltages(voltage_file)

        sens_files = sorted(glob.glob(
            directory + os.sep + 'sens' + os.sep + 'sens*.dat')
        )
        # check if there are sensitivity files, and that the nr corresponds to
        # the nr of configs
        if(len(sens_files) > 0 and
           len(sens_files) == self.configs.nr_of_configs):
            print('reading sensitivities')
            self._read_sensitivities(directory + os.sep + 'sens')

        # same for potentials
        pot_files = sorted(glob.glob(
            directory + os.sep + 'pot' + os.sep + 'pot*.dat')
        )
        print('pot files', pot_files)
        # check if there are sensitivity files, and that the nr corresponds to
        # the nr of configs
        if(len(pot_files) > 0 and
           len(pot_files) == self.configs.nr_of_configs):
            print('reading potentials')
            self._read_potentials(directory + os.sep + 'pot')

    def _read_sensitivities(self, sens_dir):
        """import sensitivities from a directory

        TODO:
        -----

        * check that signs are correct in case CRMod switches potential
        electrodes
        """
        if self.assignments['sensitivities'] is not None:
            print('Sensitivities already imported. Will not overwrite!')
            return
        else:
            self.assignments['sensitivities'] = {}

        sens_files = sorted(glob.glob(
            sens_dir + os.sep + 'sens*.dat')
        )
        for nr, filename in enumerate(sens_files):
            with open(filename, 'r') as fid:
                metadata = np.fromstring(
                    fid.readline().strip(), sep=' ', count=2
                )
                meta_mag = metadata[0]
                meta_pha = metadata[1]

                sens_data = np.loadtxt(fid)

                cids = self.parman.add_data(
                    sens_data[:, 2:4],
                    [meta_mag, meta_pha],
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

        pot_files = sorted(glob.glob(
            pot_dir + os.sep + 'pot*.dat')
        )
        for nr, filename in enumerate(pot_files):
            with open(filename, 'r') as fid:
                pot_data = np.loadtxt(fid)

                nids = self.nodeman.add_data(
                    pot_data[:, 2:4],
                )
                # store cids for later retrieval
                self.assignments['potentials'][nr] = nids

    def get_potential(self, config_nr):
        """

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

    def plot_sensitivity(self, config_nr):
        """Create a nice looking plot of the sensitivity distribution for the
        given configuration nr. Configs start at 1!
        """
        cids = self.assignments['sensitivities'][config_nr]

        fig, axes = plt.subplots(1, 2, figsize=(15 / 2.54, 15 / 2.54))
        # magnitude
        ax = axes[0]
        self.plotman.plot_elements_to_ax(
            ax,
            cids[0],
            config={},
        )

        # phase
        ax = axes[1]
        self.plotman.plot_elements_to_ax(
            ax,
            cids[1],
            config={},
        )

        return fig, axes

    def _read_voltages(self, voltage_file):
        """import voltages from a volt.dat file
        """

        measurements = np.loadtxt(
            voltage_file,
            skiprows=1,
        )
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
                # check polarity
                current_electrodes_are_equal = np.all(
                    self.configs.configs[:, 0:2] == ABMN[:, 0:2]
                )
                voltage_electrodes_are_switched = np.all(
                    self.configs.configs[:, 2:4] == ABMN[:, 4:1:-1]
                )
                if(current_electrodes_are_equal and
                   voltage_electrodes_are_switched):

                    if len(self.configs.measurements.keys()) > 0:
                        raise Exception(
                            'need to switch electrode polarity, but there ' +
                            'are already measurements stored for the ' +
                            'old configuration!')
                    else:
                        # switch M/N in configurations
                        self.configs.configs = ABMN
                else:
                    raise Exception(
                        'There was an error matching configurations of ' +
                        'voltages with configurations already imported'
                    )

        # add measurements to the config instance
        mid_mag = self.configs.add_measurements(
            measurements[:, 2]
        )[0]
        mid_pha = self.configs.add_measurements(
            measurements[:, 3]
        )[0]

        self.assignments['measurements'] = [mid_mag, mid_pha]

    def model(self,
              voltages=True,
              sensitivities=False,
              potentials=False):
        """Forward model the tomodir and read in the results
        """
        self._check_state()
        if self.can_model:
            with tempfile.TemporaryDirectory() as tempdir:
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
                return_code = subprocess.call(
                    binary, shell=True
                )
                # restore the configuration
                self.crmod_cfg = cfg_save
                if return_code != 0:
                    raise Exception('There was an error using CRMod')

                os.chdir(pwd)
                self._read_modeling_results(tempdir + os.sep + 'mod')
                return 1
        else:
            print(
                'Sorry, no measurements present, cannot model yet'
            )
            return None

    def add_homogeneous_model(self, magnitude, phase=0):
        """Add a homogeneous resistivity model to the tomodir. This is useful
        for synthetic measurements.
        """
        if self.assignments['forward_model'] is not None:
            print('model already set, will overwrite')

        # generate distributions
        magnitude_model = np.ones(self.grid.nr_of_elements) * magnitude
        phase_model = np.ones(self.grid.nr_of_elements) * phase
        pid_mag = self.parman.add_data(magnitude_model)[0]
        pid_pha = self.parman.add_data(phase_model)[0]

        self.assignments['forward_model'] = [pid_mag, pid_pha]
