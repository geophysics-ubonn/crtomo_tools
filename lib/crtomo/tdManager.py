# *-* coding: utf-8 *-*
"""A digital representation of a tomodir, i.e., a single-frequency inversion.


"""
import os

import crtomo.grid as CRGrid
import crtomo.nodeManager as nM
import crtomo.parManager as pM
import crtomo.configManager as cConf


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
        # we need a struct to organize the assigments

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

        parman = kwargs.get('parman', pM.ParMan(self.grid))
        self.parman = parman

        nodeman = kwargs.get('nodeman', nM.NodeMan(self.grid))
        self.nodeman = nodeman

        config = kwargs.get('configs', cConf.ConfigManager())
        self.config = config

    def create_tomodir(self, directory):
        """Create a tomodir subdirectory structure in the given directory
        """
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
