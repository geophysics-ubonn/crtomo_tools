# *-* coding: utf-8 *-*
"""A digital representation of a sEIT inversion directory.

A sEIT-dir (or sipdir) has the following structure:
"""
from numbers import Number
import os
from glob import glob

import pandas as pd
import numpy as np

import pylab as plt

from crtomo.grid import crt_grid
import crtomo.cfg as CRcfg
import crtomo.tdManager as CRman

import sip_models.res.cc as cc_res
from reda.eis.plots import sip_response
# this is the same object as sip_response, but for legacy reasons we have
# multiple locations for it
from sip_models.sip_response import sip_response as sip_response2


class eitMan(object):
    """Manage sEIT data

    """
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        seitdir: string, optional

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
        self.assigments = {
            'rmag': {},
            'rpha': {},
            'cre': {},
            'cim': {},
            'forward_rmag': {},
            'forward_rpha': {},
        }
        # shortcut
        self.a = self.assigments

        self.frequencies = kwargs.get('frequencies', None)

        # # now load data/initialize things
        seit_dir = kwargs.get('seitdir', None)
        crt_data_dir = kwargs.get('crt_data_dir', None)

        # these are the principle ways to add data
        if seit_dir is not None:
            # load the grid from the first invmod subdirectory
            tdirs = sorted(glob(seit_dir + '/invmod/*'))
            grid = crt_grid(
                elem_file=tdirs[0] + '/grid/elem.dat',
                elec_file=tdirs[0] + '/grid/elec.dat'
            )
            self.grid = grid

            self.load_inversion_results(seit_dir)
        elif crt_data_dir is not None:
            data_files = {}
            data_files['frequencies'] = '{}/frequencies.dat'.format(
                crt_data_dir)
            files = sorted(glob('{}/volt_*.crt'.format(crt_data_dir)))
            data_files['crt'] = files
            self.load_data_crt_files(data_files)
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

            # initialize frequency tomodirs
            if 'frequencies' in kwargs:
                self._init_frequencies(
                    kwargs.get('frequencies'),
                    kwargs.get('configs_abmn', None)
                )

    def _init_frequencies(self, frequencies, configs_abmn=None):
        self.frequencies = frequencies
        kwargs = {}

        for frequency in frequencies:
            if self.crtomo_cfg is not None:
                kwargs['crtomo_cfg'] = self.crtomo_cfg.copy()
            td = CRman.tdMan(
                grid=self.grid,
                configs_abmn=configs_abmn,
                **kwargs
            )
            self.tds[frequency] = td

    def set_area_to_sip_signature(self, xmin, xmax, zmin, zmax, spectrum):
        """Parameterize the eit instance by supplying one
        SIP spectrum and the area to apply to.

        Parameters
        ----------
        xmin: float
            Minimum x coordinate of the area
        xmax: float
            Maximum x coordinate of the area
        zmin: float
            Minimum z coordinate of the area
        zmax: float
            Maximum z coordinate of the area
        spectrum: sip_response
            SIP spectrum to use for parameterization

        """
        assert isinstance(spectrum, (sip_response, sip_response2))
        assert np.all(self.frequencies == spectrum.frequencies)
        for frequency, rmag, rpha in zip(
                self.frequencies, spectrum.rmag, spectrum.rpha):
            td = self.tds[frequency]
            pidm, pidp = td.a['forward_model']
            td.parman.modify_area(pidm, xmin, xmax, zmin, zmax, rmag)
            td.parman.modify_area(pidp, xmin, xmax, zmin, zmax, rpha)

    def set_area_to_single_colecole(self, xmin, xmax, zmin, zmax, ccpars):
        objcc = cc_res.cc(self.frequencies)
        response = objcc.response(ccpars)
        self.set_area_to_sip_signature(xmin, xmax, zmin, zmax, response)

    def add_homogeneous_model(self, magnitude, phase=0, frequency=None):
        """Add homogeneous models to one or all tomodirs. Register those as
        forward models

        Parameters
        ----------
        magnitude: float
            Value of homogeneous magnitude model
        phase: float, optional
            Value of homogeneous phase model. Default 0
        frequency: float, optional
            Frequency of of the tomodir to use. If None, then apply to all
            tomodirs. Default is None.
        """
        if frequency is None:
            frequencies = self.frequencies
        else:
            assert isinstance(frequency, Number)
            frequencies = [frequency, ]

        for freq in frequencies:
            pidm, pidp = self.tds[freq].add_homogeneous_model(magnitude, phase)
            self.a['forward_rmag'][freq] = pidm
            self.a['forward_rpha'][freq] = pidp

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
            subdata = np.atleast_2d(np.loadtxt(filename, skiprows=1))
            print(subdata.shape)
            if subdata.shape[0] == 0:
                continue
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
            self.tds[key].crtomo_cfg = self.crtomo_cfg.copy()

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
        for nr, (frequency_key, item) in enumerate(sorted(self.tds.items())):
            for label in ('rmag', 'rpha', 'cre', 'cim'):
                if label not in self.assigments:
                    self.a[label] = {}

            tdir = sipdir + os.sep + 'invmod' + os.sep + '{:02}_{:.6f}'.format(
                nr, frequency_key) + os.sep

            rmag_file = sorted(glob(tdir + 'inv/*.mag'))[-1]
            rmag_data = np.loadtxt(rmag_file, skiprows=1)[:, 2]
            pid_rmag = item.parman.add_data(rmag_data)
            self.a['rmag'][frequency_key] = pid_rmag

            rpha_file = sorted(glob(tdir + 'inv/*.pha'))[-1]
            rpha_data = np.loadtxt(rpha_file, skiprows=1)[:, 2]
            pid_rpha = item.parman.add_data(rpha_data)
            self.a['rpha'][frequency_key] = pid_rpha

            sigma_file = sorted(glob(tdir + 'inv/*.sig'))[-1]
            sigma_data = np.loadtxt(sigma_file, skiprows=1)
            pid_cre = item.parman.add_data(sigma_data[:, 0])
            pid_cim = item.parman.add_data(sigma_data[:, 1])
            self.a['cre'][frequency_key] = pid_cre
            self.a['cim'][frequency_key] = pid_cim

    def extract_polygon_area(self, label, polygon):
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

    def extract_points(self, label, points):
        """Extract data points along a given line

        Parameters
        ----------
        label : str
            the label for the assignments.
        points : Nx2 numpy.ndarray
            (x, y) pairs

        Returns
        -------
            df_all : pandas.DataFrame
                A dataframe with the extracted data

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

    def plot_forward_models(self, maglim=None, phalim=None, **kwargs):
        """Create plots of the forward models

        Returns
        -------
        mag_fig: dict
            Dictionary containing the figure and axes objects of the magnitude
            plots

        """
        return_dict = {}

        N = len(self.frequencies)
        nrx = min(N, 4)
        nrz = int(np.ceil(N / nrx))

        for index, key, limits in zip(
                (0, 1), ('rmag', 'rpha'), (maglim, phalim)):
            if limits is None:
                cbmin = None
                cbmax = None
            else:
                cbmin = limits[0]
                cbmax = limits[1]

            fig, axes = plt.subplots(
                nrz, nrx,
                figsize=(16 / 2.54, nrz * 3 / 2.54),
                sharex=True, sharey=True,
            )
            for ax in axes.flat:
                ax.set_visible(False)

            for ax, frequency in zip(axes.flat, self.frequencies):
                ax.set_visible(True)
                td = self.tds[frequency]
                pids = td.a['forward_model']
                td.plot.plot_elements_to_ax(
                    pids[index],
                    ax=ax,
                    plot_colorbar=True,
                    cbposition='horizontal',
                    cbmin=cbmin,
                    cbmax=cbmax,
                    **kwargs
                )
            for ax in axes[0:-1, :].flat:
                ax.set_xlabel('')

            for ax in axes[:, 1:].flat:
                ax.set_ylabel('')

            fig.tight_layout()
            return_dict[key] = {
                'fig': fig,
                'axes': axes,
            }

        return return_dict

    def add_to_configs(self, configs):
        """Add configurations to all tomodirs

        Parameters
        ----------
        configs : :class:`numpy.ndarray`
            Nx4 numpy array with abmn configurations

        """
        for f, td in self.tds.items():
            td.configs.add_to_configs(configs)

    def model(self, **kwargs):
        """Run the forward modeling for all frequencies.

        Use :py:func:`crtomo.eitManager.eitMan.measurements` to retrieve the
        resulting synthetic measurement spectra.

        Parameters
        ----------
        **kwargs : dict, optional
            All kwargs are directly provide to the underlying
            :py:func:`crtomo.tdManager.tdMan.model` function calls.

        """
        for key, td in self.tds.items():
            td.model(**kwargs)

    def measurements(self):
        """Return modeled measurements

        1. dimension: frequency
        2. dimension: config-number
        3. dimension: 2: magnitude and phase (resistivity)

        """
        m_all = np.array([self.tds[key].measurements() for key in
                          sorted(self.tds.keys())])
        return m_all

    def get_measurement_responses(self):
        """Return a dictionary of sip_responses for the modeled SIP spectra

        Note that this function does NOT check that each frequency contains the
        same configurations!

        Returns
        -------
        responses : dict
            Dictionary with configurations as keys

        """
        # take configurations from first tomodir
        configs = self.tds[sorted(self.tds.keys())[0]].configs.configs

        measurements = self.measurements()
        responses = {}
        for config, sip_measurement in zip(configs,
                                           np.rollaxis(measurements, 1)):
            sip = sip_response(
                frequencies=self.frequencies,
                rmag=sip_measurement[:, 0],
                rpha=sip_measurement[:, 1]
            )
            responses[tuple(config)] = sip
        return responses

