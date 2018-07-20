# *-* coding: utf-8 *-*
"""A digital representation of a sEIT inversion directory.

A sEIT-dir (or sipdir) has the following structure:
"""
import os
from glob import glob

import pandas as pd
import numpy as np

from crtomo.grid import crt_grid
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
        crmod_cfg: ?, optional

        crtomo_cfg ?, optional

        parman ?, optional

        grid: crtomo.crt_grid.crt_grid, optional
            A grid instance
        elem_file: string, optional
            Path to elem file
        elec_file: string, optional
            Path to elec file
        crt_data_dir: string, optional
            if given, then try to load data from this directory. Expect a
            'frequencies.dat' file and corresponding 'volt_*.dat' files.
        """
        # the following variables will be partially duplicated in the tomodir
        # instances down below. Thus, they are used primarily to store blue
        # prints or "parent" files for the various frequencies. For example:
        # mask-files that are then parameterized using some kind of SIP model
        # such as the Cole-Cole model.
        self.crmod_cfg = kwargs.get('crmod_cfg', None)
        self.crtomo_cfg = kwargs.get('crtomo_cfg', CRcfg.crtomo_config())
        self.parman = kwargs.get('parman', None)
        self.noise_model = kwargs.get('noise_model', None)

        # for each frequency we have a separate tomodir object
        self.tds = {}
        # when we load data from inversion results, we need to store the
        # corresponding parameter ids for the various quantities
        self.assigments = {}
        # shortcut
        self.a = self.assigments

        self.frequencies = kwargs.get('frequencies', None)

        # # now load data/initialize things

        seit_dir = kwargs.get('seitdir', None)
        if seit_dir is not None:
            # load the grid from the first invmod subdirectory
            tdirs = sorted(glob(seit_dir + '/invmod/*'))
            grid = crt_grid(
                elem_file=tdirs[0] + '/grid/elem.dat',
                elec_file=tdirs[0] + '/grid/elec.dat'
            )
            self.grid = grid

            self.load_inversion_results(seit_dir)

        else:
            # load/assign grid
            if 'grid' in kwargs:
                self.grid = kwargs.get('grid')
            elif 'elem_file' in kwargs and 'elec_file' in kwargs:
                grid = crt_grid()
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

        crt_data_dir = kwargs.get('crt_data_dir', None)
        if crt_data_dir is not None:
            data_files = {}
            data_files['frequencies'] = '{}/frequencies.dat'.format(
                crt_data_dir)
            files = sorted(glob('{}/volt_*.crt'.format(crt_data_dir)))
            data_files['crt'] = files
            self.load_data_crt_files(data_files)

    def _init_frequencies(self, frequencies):
        self.frequencies = frequencies
        for frequency in frequencies:
            td = CRman.tdMan(grid=self.grid)
            self.tds[frequency] = td

    def set_sip_parameterization(self, ):
        """DEFUNCT Parameterize the eit instance by supplying one or more
        SIP spectra

        Parameters
        ----------
        data: numpy.ndarray
            data, in the format specified in the 'format' string
        format:string
            ?

        """
        pass

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

        frequency_data = data_dict['frequencies']
        if isinstance(frequency_data, str):
            frequencies = np.loadtxt(data_dict['frequencies'])
        else:
            # if this is not a string, assume it to be the data
            frequencies = frequency_data

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

            magnitudes = subdata[:, 2]
            phases = subdata[:, 3]

            self.tds[frequency].configs.add_to_configs(ABMN)
            self.tds[frequency].register_measurements(magnitudes, phases)

    def apply_crtomo_cfg(self):
        """Set the global crtomo_cfg for all frequencies
        """
        for key in sorted(self.tds.keys()):
            self.tds[key].crtomo_cfg = self.crtomo_cfg

    def apply_noise_models(self):
        """Set the global noise_model for all frequencies
        """
        for key in sorted(self.tds.keys()):
            self.tds[key].noise_model = self.noise_model

    def save_to_eitdir(self, directory):
        """Save the eit data into a eit/sip directory structure

        Parameters
        ----------
        directory: string|path
            output directory
        """
        if os.path.isdir(directory):
            raise Exception('output directory already exists')

        os.makedirs(directory)
        np.savetxt(directory + os.sep + 'frequencies.dat', self.frequencies)

        invmod_dir = directory + os.sep + 'invmod'
        os.makedirs(invmod_dir)
        for nr, key in enumerate(sorted(self.tds.keys())):
            outdir = invmod_dir + os.sep + '{0:02}_{1:.6f}'.format(nr, key)
            print(outdir)
            self.tds[key].save_to_tomodir(outdir)

    def load_inversion_results(self, sipdir):
        """Given an sEIT inversion directory, load inversion results and store
        the corresponding parameter ids in self.assignments

        Note that all previous data stored in this instance of the eitManager
        will be overwritten, if required!
        """
        # load frequencies and initialize tomodir objects for all frequencies
        frequency_file = sipdir + os.sep + 'frequencies.dat'
        frequencies = np.loadtxt(frequency_file)
        self._init_frequencies(frequencies)

        # cycle through all tomodirs on disc and load the data
        for nr, (key, item) in enumerate(sorted(self.tds.items())):
            for label in ('rmag', 'rpha', 'cre', 'cim'):
                if label not in self.assigments:
                    self.a[label] = {}

            tdir = sipdir + os.sep + 'invmod' + os.sep + '{:02}_{:.6f}'.format(
                nr, key) + os.sep
            print(tdir, tdir)

            rmag_file = sorted(glob(tdir + 'inv/*.mag'))[-1]
            rmag_data = np.loadtxt(rmag_file, skiprows=1)[:, 2]
            pid_rmag = item.parman.add_data(rmag_data)
            self.a['rmag'][key] = pid_rmag

            rpha_file = sorted(glob(tdir + 'inv/*.pha'))[-1]
            rpha_data = np.loadtxt(rpha_file, skiprows=1)[:, 2]
            pid_rpha = item.parman.add_data(rpha_data)
            self.a['rpha'][key] = pid_rpha

            sigma_file = sorted(glob(tdir + 'inv/*.sig'))[-1]
            sigma_data = np.loadtxt(sigma_file, skiprows=1)
            pid_cre = item.parman.add_data(sigma_data[:, 0])
            pid_cim = item.parman.add_data(sigma_data[:, 1])
            self.a['cre'][key] = pid_cre
            self.a['cim'][key] = pid_cim


    def extract_points(self, label, points):
        """Extract data points along a given line

        Parameters
        ----------
        label: str

        points: Nx2 numpy.ndarray
            (x, y) pairs

        Returns
        -------

        """
        if isinstance(label, str):
            label = [label, ]

        data_list = []
        for label_key in label:
            value_list = {}
            for key, item in sorted(self.tds.items()):
                values = item.parman.extract_points(
                    pid=self.a[label_key][key],
                    points=points
                )
                value_list[key] = values
            df = pd.DataFrame(value_list)
            df['x'] = points[:, 0]
            df['z'] = points[:, 1]
            df['key'] = label_key
            data_list.append(df)
        df_all = pd.concat(data_list)

        return df_all

