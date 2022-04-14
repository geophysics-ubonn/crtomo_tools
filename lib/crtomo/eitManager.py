# *-* coding: utf-8 *-*
"""A digital representation of a sEIT inversion directory.

A sEIT-dir (or sipdir) has the following structure:
"""
from numbers import Number
import os
import shutil
from glob import glob

import pandas as pd
import numpy as np

import pylab as plt

from crtomo.grid import crt_grid
import crtomo.cfg as CRcfg
import crtomo.tdManager as CRman
# from crtomo.status import seitdir_is_finished

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
        Data/frequency input: Note that only one of the parameters seitdir,
        crt_data_dir, seit_data will be used during initialization, in the
        stated priority.

        Initialization diagram:

        .. blockdiag::

            diagram {
                group g_grid {
                    par_grid -> a_grid [label="loads"];
                    group g_elmc {
                        par_elem -> a_grid [label="loads"];
                        par_elec -> a_grid [label="loads"];
                    }
                }
                a_grid [label="grid"]
                seitdir[label="sEIT inversion directory"];
                seitdir -> a_grid [label="grid"];
                seitdir -> frequencies [label="loads"];
                seitdir -> inv_results [label="loads"];
                seitdir -> seit_data [label="loads"];
                crt_data_dir -> seit_data [label="loads"];
                inv_results[label="loads"];
                crt_data_dir[label="CRTomo style multi-frequency data"];
                crt_data_dir -> frequencies [label="loads"];
                crt_data_dir -> seit_data [label="loads"];
                dict_seit_data [label="data dictionary"];
                par_frequencies -> frequencies [label="loads"];

           }

        Parameters
        ----------
        seitdir : string, optional
            Load frequencies, grid, and data from an existing sEIT directory.
            Honors the shallow_import parameter.
        shallow_import : bool, optional
            In combination with seitdir, only import the last iteration
            inversion results (faster).
        crt_data_dir : string, optional
            if given, then try to load data from this directory. Expect a
            'frequencies.dat' file and corresponding 'volt_*.dat' files.
        only_frequency_ids : numpy.ndarray|None
            Load only the frequencies associated with these indices
            (zero-based). Only works with crt_data_dir.
        seit_data : dict
            A dictionary with frequencies as keys, and numpy arrays as items.
            Each array must have 6 columns: a,b,m,n,magnitude[Ohm],phase[mrad]
        frequencies : numpy.ndarray, optional
            frequencies that we work with
        grid : crtomo.crt_grid.crt_grid, optional
            A grid instance
        elem_file : string, optional
            Path to elem file
        elec_file : string, optional
            Path to elec file
        decouplings_file : str, optional
            Path to decoupling file: For not will be copied to the exe/ subdir
            of each tomodir written
        crmod_cfg : ?, optional

        crtomo_cfg : ?, optional

        parman : ?, optional

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

        self.decoupling_file = kwargs.get('decouplings_file', None)

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
        seit_data = kwargs.get('seit_data', None)

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
            self.grid = None

        # these are the principle ways to add data
        if seit_dir is not None:
            # load the grid from the first invmod subdirectory
            tdirs = sorted(glob(seit_dir + '/invmod/*'))
            grid = crt_grid(
                elem_file=tdirs[0] + '/grid/elem.dat',
                elec_file=tdirs[0] + '/grid/elec.dat'
            )
            self.grid = grid

            self.load_inversion_results(
                seit_dir, shallow_import=kwargs.get('shallow_import', False)
            )
        elif crt_data_dir is not None:
            data_files = {}
            data_files['frequencies'] = '{}/frequencies.dat'.format(
                crt_data_dir)
            files = sorted(glob('{}/volt_*.crt'.format(crt_data_dir)))
            only_frequency_ids = kwargs.get('only_use_frequency_ids', None)
            data_files['crt'] = files
            self.load_data_crt_files(
                data_files, only_frequency_ids=only_frequency_ids
            )
        elif seit_data is not None:
            frequencies = sorted(seit_data.keys())
            self._init_frequencies(frequencies)

            for frequency in frequencies:
                abmn = seit_data[frequency][:, 0:4]
                rmag = seit_data[frequency][:, 4]
                rpha = seit_data[frequency][:, 5]

                self.tds[frequency].configs.add_to_configs(abmn)
                self.tds[frequency].register_measurements(rmag, rpha)
        else:
            # initialize frequency tomodirs
            if 'frequencies' in kwargs:
                self._init_frequencies(
                    kwargs.get('frequencies'),
                    kwargs.get('configs_abmn', None)
                )

        if self.grid is None:
            raise Exception(
                'You must provide either a grid instance or '
                'elem_file/elec_file file paths'
            )

    def _init_frequencies(self, frequencies, configs_abmn=None):
        """Initialize the tdMan instances associated with each frequency

        Note that existing tds will not be deleted/overwritten

        Parameters
        ----------
        frequencies : Nx1 numpy.ndarray
            Frequencies in ascending order
        configs_abmn : None|numpy.ndarray (Mx4)
            Measurement configurations to provide to the generated tdMan
            instances

        """
        self.frequencies = frequencies
        kwargs = {}

        for frequency in frequencies:
            if frequency in self.tds:
                # already present, do not generate a new tdMan instance
                continue
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
        xmin : float
            Minimum x coordinate of the area
        xmax : float
            Maximum x coordinate of the area
        zmin : float
            Minimum z coordinate of the area
        zmax : float
            Maximum z coordinate of the area
        spectrum : sip_response
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
        """Parameterize a rectangular area of the forward model using a
        single-term Cole-Cole response.

        Parameters
        ----------

        """
        objcc = cc_res.cc(self.frequencies)
        response = objcc.response(ccpars)
        self.set_area_to_sip_signature(xmin, xmax, zmin, zmax, response)

    def add_homogeneous_model(self, magnitude, phase=0, frequency=None):
        """Add homogeneous models to one or all tomodirs. Register those as
        forward models

        Parameters
        ----------
        magnitude : float
            Value of homogeneous magnitude model
        phase : float, optional
            Value of homogeneous phase model. Default 0
        frequency : float, optional
            Frequency of of the tomodir to use. If None, then apply to all
            tomodirs. Default is None.

        Returns
        -------
        None
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

    def register_forward_model(self, frequency, mag_model, phase_model):
        """Register a magnitude parameter set and optionally a phase parameter
        set as a forward model for a given frequency.

        Parameters
        ----------
        frequency : float
            Frequency. Must match the frequencies in eitMan.tds
        mag_model : numpy.ndarray
            The magnitude model (linear scale)
        phase_model : numpy.ndarray
            The phase model [mrad]
        """
        assert frequency in self.tds.keys(), "Frequency does not match any td"
        td = self.tds[frequency]

        pid_mag = td.register_magnitude_model(mag_model)
        pid_pha = td.register_phase_model(phase_model)
        self.a['forward_rmag'][frequency] = pid_mag
        self.a['forward_rpha'][frequency] = pid_pha

    def load_data_crt_files(self, data_dict, **kwargs):
        """Load sEIT data from .ctr files (volt.dat files readable by CRTomo,
        produced by CRMod)

        Parameters
        ----------
        data_dict : dict
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
        only_frequency_ids = kwargs.get('only_frequency_ids', None)
        if only_frequency_ids is not None:
            frequencies = frequencies[only_frequency_ids]
            data_dict['crt'] = [
                data_dict['crt'][i] for i in only_frequency_ids
            ]
        self._init_frequencies(frequencies)

        for frequency, filename in zip(frequencies, data_dict['crt']):
            subdata = np.atleast_2d(np.loadtxt(filename, skiprows=1))
            if subdata.size == 0:
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
            # HACK: if available, copy the decouplings file
            if self.decoupling_file is not None:
                shutil.copy(
                    self.decoupling_file,
                    outdir + os.sep + 'exe' + os.sep + 'decouplings.dat'
                )

    def reset_tds(self):
        """Reset the data stored in all tds"""
        for frequency, td in self.tds.items():
            td.reset_data()
        for key in ('rmag', 'rpha', 'cre', 'cim'):
            if key in self.a:
                self.a[key] = {}

    def load_inversion_results(self, sipdir, shallow_import=True):
        """Given an sEIT inversion directory, load inversion results and store
        the corresponding parameter ids in self.assignments

        Note that all previous data stored in this instance of the eitManager
        will be overwritten, if required!

        Parameters
        ----------
        sipdir : string
            path to a CRTomo sip-invdir (i.e., a full sEIT inversion directory)

        shallow_import : bool, optional
            If set to True, then only import the last inversion result, nothing
            else.
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

            # some old inversion directories are one-indexed, while new ones
            # start with zero. As such, try both approaches
            for test_nr in (nr, nr + 1):
                tdir = ''.join((
                    sipdir + os.sep,
                    'invmod' + os.sep,
                    '{:02}_{:.6f}'.format(test_nr, frequency_key) + os.sep
                ))
                if os.path.isdir(tdir):
                    break
            if not os.path.isdir(tdir):
                raise Exception('tdir not found: {}'.format(tdir))

            if shallow_import:
                rmag_file = sorted(glob(tdir + 'inv/*.mag'))
                if len(rmag_file) > 0:
                    rmag_data = np.loadtxt(rmag_file[-1], skiprows=1)[:, 2]
                    pid_rmag = item.parman.add_data(rmag_data)
                else:
                    pid_rmag = None
                self.a['rmag'][frequency_key] = pid_rmag

                rpha_file = sorted(glob(tdir + 'inv/*.pha'))
                if len(rpha_file) > 0:
                    rpha_data = np.loadtxt(rpha_file[-1], skiprows=1)[:, 2]
                    pid_rpha = item.parman.add_data(rpha_data)
                else:
                    pid_rpha = None
                self.a['rpha'][frequency_key] = pid_rpha

                sigma_files = sorted(glob(tdir + 'inv/*.sig'))
                if len(sigma_files) > 0:
                    sigma_data = np.loadtxt(sigma_files[-1], skiprows=1)
                    pid_cre = item.parman.add_data(sigma_data[:, 0])
                    pid_cim = item.parman.add_data(sigma_data[:, 1])
                elif len(rmag_file) == 0 and len(rpha_file) == 0:
                    pid_cre = None
                    pid_cim = None
                else:
                    # very old CRTomo runs...
                    crho = item.parman.parsets[
                        pid_rmag
                    ] * np.exp(1j * item.parman.parsets[pid_rpha] / 1000)
                    csigma = 1 / crho
                    pid_cre = item.parman.add_data(csigma.real)
                    pid_cim = item.parman.add_data(csigma.imag)
                self.a['cre'][frequency_key] = pid_cre
                self.a['cim'][frequency_key] = pid_cim
            else:
                crtomo_cfg_file = tdir + os.sep + 'exe' + os.sep + 'crtomo.cfg'
                if os.path.isfile(crtomo_cfg_file):
                    item.crtomo_cfg.import_from_file(crtomo_cfg_file)
                # forward configurations
                config_file = tdir + os.sep + 'config' + os.sep + 'config.dat'
                if os.path.isfile(config_file):
                    item.configs.load_crmod_config(config_file)

                # load data/modeling results
                item._read_modeling_results(tdir + os.sep + 'mod')

                item.read_inversion_results(tdir)

                if len(item.a['inversion']['rmag']) > 0:
                    self.a['rmag'][frequency_key] = item.a[
                        'inversion']['rmag'][-1]
                    self.a['rpha'][frequency_key] = item.a[
                        'inversion']['rpha'][-1]
                    self.a['cre'][frequency_key] = item.a[
                        'inversion']['cre'][-1]
                    self.a['cim'][frequency_key] = item.a[
                        'inversion']['cim'][-1]
                else:
                    self.a['rmag'][frequency_key] = None
                    self.a['rpha'][frequency_key] = None
                    self.a['cre'][frequency_key] = None
                    self.a['cim'][frequency_key] = None

    def extract_polygon_area(self, label, polygon_points):
        """DEFUNCT

        Parameters
        ----------
        label : str
            the label (data type) to extract. This corresponds to a key in
            eitMan.assignments. Possible values are rmag, rpha, cre,
            cim
        polygon_points : list of (x,y) floats
            list of points that form the polygon

        Returns
        -------

        """
        raise Exception('Not implemented')
        """
        if isinstance(label, str):
            label = [label, ]

        data_list = []
        for label_key in label:
            value_list = {}
            for frequency, tdobj in sorted(self.tds.items()):
                values = tdobj.parman.extract_polygon_area(
                    pid=self.a[label_key][frequency],
                    polygon_points=polygon_points
                )
                value_list[key] = values
            df = pd.DataFrame(value_list)
            df['x'] = points[:, 0]
            df['z'] = points[:, 1]
            df['key'] = label_key
            data_list.append(df)
        df_all = pd.concat(data_list)

        return df_all
        """

    def extract_all_spectra(self, label):
        """Extract all SIP spectra, and return frequencies and a numpy array

        Parameters
        ----------
        label : str
            the label (data type) to extract. This corresponds to a key in
            eitMan.assignments. Possible values are rmag, rpha, cre, cim

        """
        if isinstance(label, str):
            label = [label, ]

        data_list = {}
        for label_key in label:
            frequencies = []
            value_list = []
            for frequency, item in sorted(self.tds.items()):
                pid = self.a[label_key][frequency]
                value_list.append(item.parman.parsets[pid])
                frequencies.append(frequency)
            data_list[label_key] = (frequencies, np.array(value_list))
        return data_list

    def extract_points(self, label, points):
        """Extract data points (i.e., SIP spectra) for one or more points.

        Parameters
        ----------
        label : str
            the label (data type) to extract. This corresponds to a key in
            eitMan.assignments. Possible values are rmag, rpha, cre,
            cim
        points : Nx2 numpy.ndarray
            (x, y) pairs

        Returns
        -------
        df_all : pandas.DataFrame
            A dataframe with the extracted data

        """
        if isinstance(label, str):
            label = [label, ]

        points = np.atleast_2d(points)

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
        return_dict : dict
            Dictionary containing the figure and axes objects of the magnitude
            plots

        """
        return_dict = {}

        N = len(self.frequencies)
        nrx = min(N, 3)
        nrz = int(np.ceil(N / nrx))

        labels = [
            r'$\rho~[\Omega m]$',
            r'$\phi~[mrad]$',
        ]

        cmaps = [
            'turbo',
            'jet',
        ]

        # try to select a suitable colorbar position
        (xmin, xmax), (zmin, zmax) = self.grid.get_minmax()
        if np.abs(xmax - xmin) > np.abs(zmax - zmin):
            cbposition = 'horizontal'
        else:
            cbposition = 'vertical'

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
                figsize=(16 / 2.54, nrz * 3.5 / 2.54),
                sharex=True, sharey=True,
            )
            axes = np.atleast_2d(axes)
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
                    cbposition=cbposition,
                    cbmin=cbmin,
                    cbmax=cbmax,
                    title='{:.3f} Hz'.format(frequency),
                    cblabel=labels[index],
                    cmap_name=cmaps[index],
                    **kwargs,
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

    def measurements(self, **kwargs):
        """Return modeled measurements. If not already done, call CRMod for
        each frequency to actually compute the forward response.

        Returns
        -------
        m_all : numpy.ndarray
            1. dimension: frequency
            2. dimension: config-number
            3. dimension: 2: magnitude and phase (resistivity)

        """
        m_all = np.array([self.tds[key].measurements(**kwargs) for key in
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

    def save_measurements_to_directory(self, directory):
        """Store the measurements (either from a previous import, or from
        forward modeling, in one directory in the CRTomo 'crt'-style.

        Frequencies are stored in a file frequencies.dat.

        For each frequency, data is stored in a file volt_%.2i.dat in the
        CRTomo format.
        """
        os.makedirs(directory, exist_ok=True)
        np.savetxt(directory + os.sep + 'frequencies.dat', self.frequencies)
        for nr, (frequency, td) in enumerate(sorted(self.tds.items())):
            td.save_measurements(
                directory + os.sep + 'volt_{:02}_f{:08.3f}.dat'.format(
                    nr, frequency
                )
            )

    def plot_all_result_spectra(self, directory):
        os.makedirs(directory, exist_ok=True)
        N = self.tds[self.frequencies[0]].configs.configs.shape[0]
        for i in range(N):
            print('Plotting {}/{}'.format(i + 1, N))
            self.plot_result_spectrum(
                directory + '/spectrum_{:03}.jpg'.format(i), i)

    def plot_result_spectrum(self, filename, nr):
        """Plot a given data and inversion response spectrum to file.

        WARNING: At this point does not take into account missing
        configurations for certain frequencies...

        Parameters
        ----------
        filename : str
            Output filename
        nr : int
            Index of spectrum to plot. Starts at 0

        """
        rpha = np.vstack(
            [td.configs.measurements[
                td.a['inversion']['fwd_response_rpha'][-1]
            ] for f, td in sorted(self.tds.items())]).T
        rmag = np.vstack(
            [td.configs.measurements[
                td.a['inversion']['fwd_response_rmag'][-1]
            ] for f, td in sorted(self.tds.items())]).T
        spec1 = sip_response(
            self.frequencies, rmag=rmag[nr, :], rpha=rpha[nr, :])
        rmag_true = np.vstack(
            [td.configs.measurements[
                td.a['measurements'][0]
            ] for f, td in sorted(self.tds.items())]).T
        rpha_true = np.vstack(
            [td.configs.measurements[
                td.a['measurements'][1]
            ] for f, td in sorted(self.tds.items())]).T
        spec_true = sip_response(
            self.frequencies, rmag=rmag_true[nr, :], rpha=rpha_true[nr, :])
        spec_true.plot(
            filename,
            reciprocal=spec1,
            title='abmn: {}-{}-{}-{}'.format(
                *self.tds[self.frequencies[0]].configs.configs[nr]),
            label_nor='data',
            label_rec='inversion response',
        )

    def set_electrode_capacitances(self, capacitance):
        """Zimmermann et al 2018"""
        for frequency, td in self.tds.items():
            td.electrode_admittance = 2 * np.pi * frequency * capacitance

    def assign_sip_signatures_using_mask(self, mask_raw, lookup_table):
        """

        Parameters
        ----------
        lookup_table : dict

        spectrum : sip_response
            SIP spectrum to use for parameterization

        """
        assert isinstance(mask_raw, np.ndarray), "mask must be numpy array"
        # these are indices - we need them to be integers
        mask = mask_raw.astype(int)
        assert len(mask.shape) == 1, "mask must be an 1D array"
        assert (
            mask.size == self.grid.nr_of_elements
        ), "mask must be of the same size as number of mesh elements"

        assert isinstance(
            lookup_table, dict), "parameter lookup_table must be a dict"

        # check the lookup table
        for key, item in lookup_table.items():
            assert isinstance(
                item, (sip_response, sip_response2)
            ), "the item with key {} is not a sip_response!".format(key)
            assert np.all(self.frequencies == item.frequencies), \
                "The frequencies in the spectrum of key " + \
                "{} do not match".format(key)

        assert len(lookup_table) == np.unique(mask).size, \
            "Number of entries in the lookup table does not match number " + \
            "of unique mask entries"

        assert np.all(
            np.sort(
                np.unique(mask)
            ) == np.sort(np.unique(list(lookup_table.keys())))
            ), \
            "entries in mask and lookup_table do not match"

        if len(self.a['forward_rmag']) == 0:
            print('No forward models registered yet. Creating empty ones')
            self.add_homogeneous_model(0, 0)

        # loop over assignments
        for assignment in np.unique(mask):
            pixel_indices = np.where(mask == assignment)[0]

            spectrum = lookup_table[assignment]
            for frequency, rmag, rpha in zip(
                    self.frequencies, spectrum.rmag, spectrum.rpha):
                td = self.tds[frequency]
                pid_mag, pid_phase = td.a['forward_model']
                td.parman.modify_pixels(
                    pid_mag,
                    pixel_indices,
                    rmag
                )
                td.parman.modify_pixels(
                    pid_phase,
                    pixel_indices,
                    rpha
                )
