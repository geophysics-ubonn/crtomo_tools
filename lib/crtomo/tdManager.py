# *-* coding: utf-8 *-*
"""A digital representation of a tomodir, i.e., a single-frequency inversion.


"""
import os
# import crtomo.nodeManager as nM
# import crtomo.parManager as pM


class tdMan(object):
    """Manage tomodirs

    """
    def __init__(self, kwargs):
        """The following initialization permutations are possible:

            * initialize empty
            * initialize from an exisiting tomodir
                * tomodir: [tomodir path]
            * supply one or more of the building blocks of a tomodir
                * grid: crtomo.grid.crt_grid instance
                * crmod_cfg: crtomo.cfg.crmod_cfg instance
                * crtomo_cfg: crmod.cfg.crtomo_cfg instance

        """
        # these variables will be filled later
        self.grid = None
        self.crmod_cfg = None
        self.crtomo_cfg = None
        self.nodeman = None
        self.parman = None
        # configs
        # measurements
        # we need a struct to organize the assigments

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
