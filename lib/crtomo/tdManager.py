# *-* coding: utf-8 *-*
"""A digital representation of a tomodir, i.e., a single-frequency inversion.


"""
import os
import tempfile
import numpy as np
import subprocess

import crtomo.binaries as CRBin
import crtomo.grid as CRGrid
import crtomo.nodeManager as nM
import crtomo.parManager as pM
import crtomo.configManager as cConf
import crtomo.cfg as CRcfg


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
        # we need a struct to organize the assignments
        self.assignments = {
            # should contain a two-item list with ids in parman
            'forward_model': None,
            # should contain one id for the config object
            'measurements': None,
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

        print('nrelc', self.grid.nr_of_electrodes)
        config = kwargs.get(
            'configs',
            cConf.ConfigManager(nr_of_electrodes=self.grid.nr_of_electrodes)
        )
        self.config = config

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

    def save_to_tomodir(self, directory):
        """Save the tomodir instance to a directory structure.

        Test cases:

            * modeling only
            * inversion only
            * modeling and inversion

        """
        self.create_tomodir(directory)

        # determine which set to save (modeling and/or inversion)
        # requirements for modeling: grid, configs, rho
        # requirements for inversion: grid, measurements

        self.grid.save_elem_file(
            directory + os.sep + 'grid/elem.dat'
        )

        self.grid.save_elec_file(
            directory + os.sep + 'grid/elec.dat'
        )

        save_modeling = False
        # modeling
        if(self.grid is not None and
           self.config.configs is not None and
           self.assignments['forward_model'] is not None):
            save_modeling = True

            self.config.write_crmod_config(
                directory + os.sep + 'config/config.dat'
            )

            self.parman.save_to_rho_file(
                directory + os.sep + 'rho/rho.dat',
                self.assignments['forward_model'][0],
                self.assignments['forward_model'][1],
            )

            self.crmod_cfg.write_to_file(
                directory + os.sep + 'exe/crmod.cfg'
            )

        # inversion
        if(self.grid is not None and
           (save_modeling or self.assignments['measurements'] is not None)
           ):
            self.crtomo_cfg.write_to_file(
                directory + os.sep + 'exe/crtomo.cfg'
            )
            # todo: measurements

    def measurements(self):
        """Return the measurements associated with this instance.

        if measurements are not present, check if we can model, and then
        run CRMod to load the measurements.
        """
        mid = self.assignments.get('measurements', None)
        if mid is not None:
            cids = self.assignments['measurements']
            measurements = np.hstack((
                self.config.measurements[cids[0]],
                self.config.measurements[cids[1]],
            )).T
            return measurements
        else:
            if self.can_model:
                with tempfile.TemporaryDirectory() as tempdir:

                    self.model(
                        directory=tempdir,
                        voltages=True,
                        sensitivities=False,
                        potentials=False,
                    )
                    # read data from the temp directory
                    measurements = np.loadtxt(
                        tempdir + os.sep + 'mod/volt.dat',
                        skiprows=1,
                    )
                    print(measurements.shape)
                    mid_mag = self.config.add_measurements(
                        measurements[:, 2]
                    )
                    mid_pha = self.config.add_measurements(
                        measurements[:, 3]
                    )
                    self.assignments['measurements'] = [mid_mag, mid_pha]
                    # hmm, this doesn't look good, perhaps we should rethink
                    # the way we store measurements?
                    return measurements[:, 2:4]
            else:
                print(
                    'Sorry, no measurements present, cannot model yet'
                )

    def model(self, directory, voltages=True, sensitivities=False,
              potentials=False):
        """Forward model the tomodir
        """
        pwd = os.getcwd()
        os.chdir(directory)
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
