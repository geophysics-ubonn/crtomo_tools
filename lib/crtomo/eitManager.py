# *-* coding: utf-8 *-*
"""A digital representation of a sEIT inversion directory.

A sEIT-dir (or sipdir) has the following structure:



"""
import os

import numpy as np

import crtomo.grid as CRgrid
import crtomo.cfg as CRcfg
import crtomo.tdManager as CRman


class eitMan(object):
    """Manage sEIT data

    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        frequencies: numpy.ndarray
            frequencies that we work with
        """
        # load/assign grid
        if 'grid' in kwargs:
            self.grid = kwargs.get('grid')
        elif 'elem_file' in kwargs and 'elec_file' in kwargs:
            grid = CRgrid.crt_grid()
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

        self.frequencies = None

        # the following variables will be partially duplicated in the tomodir
        # instances down below. Thus, they are used primarily to store blue
        # prints or "parent" files for the various frequencies. For example:
        # mask files that are then parameterized using some kind of SIP model
        # such as the Cole-Cole model.
        self.crmod_cfg = kwargs.get('crmod_cfg', None)
        self.crtomo_cfg = kwargs.get('crtomo_cfg', CRcfg.crtomo_config())
        self.parman = kwargs.get('parman', None)

        # for each frequency we have a separate tomodir object
        self.tds = {}

    def _init_frequencies(self, frequencies):
        self.frequencies = frequencies
        for frequency in frequencies:
            td = CRman.tdMan(grid=self.grid)
            self.tds[frequency] = td

    def load_data_crt_files(self, data_dict):
        """Load sEIT data from .ctr files (volt.dat files readable by CRTomo,
        produced by CRMod)

        Parameters
        ----------
        data_dict: dict
            Data files that are imported. See example down below

        Examples
        --------

        >>> import glob
            data_files = {}
            data_files['frequencies'] = 'data/frequencies.dat'
            files = sorted(glob.glob('data/volt_*.crt'))
            data_files['crt'] = files

        """
        if isinstance(data_dict, str):
            raise Exception('Parameter must be a dict!')

        frequencies = np.loadtxt(data_dict['frequencies'])
        if frequencies.size != len(data_dict['crt']):
            raise Exception(
                'number of frequencies does not match the number of data files'
            )
        self._init_frequencies(frequencies)

        for frequency, filename in zip(frequencies, data_dict['crt']):
            subdata = np.loadtxt(filename, skiprows=1)
            # extract configurations
            A = (subdata[:, 0] / 1e4).astype(int)
            B = (subdata[:, 0] % 1e4).astype(int)
            M = (subdata[:, 1] / 1e4).astype(int)
            N = (subdata[:, 1] % 1e4).astype(int)

            ABMN = np.vstack((A, B, M, N)).T
            print(ABMN.shape)

            magnitudes = subdata[:, 2]
            phases = subdata[:, 3]

            self.tds[frequency].configs.add_to_configs(ABMN)
            self.tds[frequency].register_measurements(magnitudes, phases)

    def apply_crtomo_cfg(self):
        """Set the global crtomo_cfg for all frequencies
        """
        for key in sorted(self.tds.keys()):
            self.tds[key].crtomo_cfg = self.crtomo_cfg

    def save_to_eitdir(self, directory):
        """Save the eit data into a eit/sip directory structure
        """
        if os.path.isdir(directory):
            raise Exception('output directory already exists')

        os.makedirs(directory)
        np.savetxt(directory + os.sep + 'frequencies.dat', self.frequencies)

        invmod_dir = directory + os.sep + 'invmod'
        os.makedirs(invmod_dir)
        for nr, key in enumerate(sorted(self.tds.keys())):
            outdir = invmod_dir + os.sep + '{0:02}_{1:.6}'.format(nr, key)
            print(outdir)
            self.tds[key].save_to_tomodir(outdir)
