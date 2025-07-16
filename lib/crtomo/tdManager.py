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
import time
import datetime
from glob import glob
import re
import os
import tempfile
import subprocess
import io
from io import StringIO
import itertools
import functools
import tarfile

import matplotlib.colors
import matplotlib.cm
import numpy as np
import pandas as pd
import reda
from reda.main import units as reda_units

import crtomo.mpl
from crtomo.mpl import get_mpl_version

import crtomo.binaries as CRBin
import crtomo.grid as CRGrid
import crtomo.nodeManager as nM
import crtomo.parManager as pM
import crtomo.configManager as cConf
import crtomo.cfg as CRcfg
import crtomo.plotManager as PlotManager

mpl_version = get_mpl_version()
plt, mpl = crtomo.mpl.setup()
mpl_version = crtomo.mpl.get_mpl_version()


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

        .. blockdiag::

           diagram {
            inv_cjg -> NI;
            cov -> NI;
            cov_mag_fpi -> NI;
            ata_diag -> NI;
            ata_reg_diag -> NI;
            cov1_m_diag -> NI;
            cov2_m_diag -> NI;
            res_m_diag -> tdMan_parset_resm;
            eps -> tdMan_eps_data;
            inv_ctr -> tdMan_inv_stats;
            modl -> NI;
            mag -> tdMan_inv_mag;
            pha -> tdMan_inv_pha;
            run_ctr -> NI;
            volt_files -> NI;
            cre_cim -> tdMag_inv_cre_cim;

            tdMan_inv_stats[label="tdMan.inv_stats"];
            tdMan_eps_data[label="tdMan.eps_data"];
            tdMan_parset_resm[label="tdMan.parman.parsets[tdMan.a['res_m']]"];
            inv_cjg[label="conjugate gradient information"];
            NI[label="not implemented"];
           }

        http://www2.geo.uni-bonn.de/~mweigand/dashboard/content/crtomo_doc/crtomo/files.html#inv

        Keyword Arguments
        -----------------
        grid: crtomo.grid.crt_grid
            A fully initialized grid object
        tempdir : string|None, optional
            if set, use this directory to create all temporary directories
            in. For example, settings this to /dev/shm can result in faster
            generation of large Jacobians
        volt_file : None|str|numpy.ndarray
            if this is None, assume we didn't get any measurement data. If
            this is a str, assume it to be the filename to a CRTomo
            volt.dat file. If it is a numpy array, assume 6 columns:
            a,b,m,n,rmag,rpha
        volt_data : synonym for volt_file parameter
            see description for volt_file
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
            # should contain a two-item tuple with ids of data errors (stored
            # in self.configs)
            'measurement_errors': None,
            # these are the normalization factors for the individual errors
            'error_norm_factor_mag': None,
            'error_norm_factor_pha': None,
            # store sensitivity cids here
            'sensitivities': None,
            # store potential nids here
            'potentials': None,
            # last iteration of inversion results
            'inversion': None,
            # if not None, this should be a tuple (pid_rmag, pid_rpha)
            # containing pids to the prior model
            'prior_model': None,
        }
        # short-cut
        self.a = self.assignments

        # if set, use this class for the decoupled error model
        self.noise_model = kwargs.get('noise_model', None)

        # indicates if all information for modeling are present
        self.can_model = False
        # indicates if all information for inversion are present
        self.can_invert = False

        self.eps_data = None
        self.inv_stats = None
        # additional information from inversion results
        self.header_res_m = None
        self.header_l1_dw_log10_norm = None

        # if we did measure electrode capacitances, store the average electrode
        # capacitance here
        # See Zimmermann et al. 2018 for more information
        # For now this is only one value [S/m] = omega * C and will be
        # multiplied for each electrode
        self.electrode_admittance = None

        # when calling CRTomo, store output here
        self.crtomo_error_msg = None
        self.crtomo_output = None
        self._initialize_components(kwargs)

    def __repr__(self):
        """Return meaningful information on current state of the object"""
        str_list = []
        str_list.append(80 * '-')
        str_list.append('tdMan instance')
        str_list.append(80 * '-')
        # status of grid
        str_list.append('GRID:')
        if self.grid is None:
            str_list.append('no grid loaded')
        else:
            str_list.append(self.grid.__repr__())

        # status of configs
        str_list.append('')
        str_list.append('CONFIGS:')
        if self.configs.configs is None:
            str_list.append('no configs present')
        else:
            str_list.append(
                '{} configs present'.format(self.configs.configs.shape[0])
            )

        # status of parsets
        str_list.append('')
        str_list.append('PARSETS:')
        str_list.append(
            '{} parsets loaded'.format(len(self.parman.parsets.keys()))
        )

        str_list.append('')
        str_list.append('ASSIGNMENTS:')
        str_list.append('{}'.format(self.assignments))

        str_list.append(80 * '-')
        return '\n'.join(str_list)

    def _initialize_components(self, kwargs):
        r"""initialize the various components using the supplied \*\*kwargs

        Parameters
        ----------
        kwargs : dict
            kwargs dict as received by __init__()

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
                'You must provide either a grid instance or '
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
            self.configs.load_crmod_or_4c_configs(config_file)

        # we can load data either from file, or directly from a numpy array
        voltage_file = kwargs.get('volt_file', None)
        voltage_data = kwargs.get('volt_data', voltage_file)

        # did we get either a file name OR data? then import it
        if voltage_data is not None:
            cids = self.configs.load_crmod_data(voltage_data)
            self.assignments['measurements'] = cids

        resistances = kwargs.get('resistances', None)

        if resistances is not None:
            cid_mag = self.configs.add_measurements(resistances)
            self.register_measurements(mag=cid_mag)

        self.plot = PlotManager.plotManager(
            grid=self.grid,
            nm=self.nodeman,
            pm=self.parman,
        )

        # store decoupling data for the inversion here
        # will become a Yx3 array
        self.decouplings = None

        # if we load from a tomodir, also load configs and inversion results
        if tomodir is not None:
            print('importing tomodir results')
            # if present, read crtomo.cfg file
            crtomo_cfg_file = tomodir + os.sep + 'exe' + os.sep + 'crtomo.cfg'
            if os.path.isfile(crtomo_cfg_file):
                self.crtomo_cfg.import_from_file(crtomo_cfg_file)

            # forward configurations
            config_file = tomodir + os.sep + 'config' + os.sep + 'config.dat'
            if os.path.isfile(config_file):
                self.configs.load_crmod_config(config_file)

            # forward model
            rho_file = tomodir + os.sep + 'rho' + os.sep + 'rho.dat'
            if os.path.isfile(rho_file):
                pid_mag, pid_pha = self.parman.load_from_rho_file(rho_file)
                self.register_forward_model(pid_mag, pid_pha)

            # load data/modeling results
            self._read_modeling_results(tomodir + os.sep + 'mod')

            # load inversion results
            self.read_inversion_results(tomodir)

            self.read_decouplings_file(
                tomodir + os.sep + 'exe' + os.sep + 'decouplings.dat'
            )

        # if tomodir_tarxz is not None:
        #     # read from a tarxz file/BytesIO file
        #     raise Exception('Reading from tar.xz files is not supported yet')

    def read_decouplings_file(self, filename):
        """Import decoupling data for the inversion. This is usally a file
        called decouplings.dat in the exe/ directory of a tomodir, but we can
        also read from an BytesIO object.

        Do nothing if the file does not exist

        Overwrite any existing decouplings
        """
        if not isinstance(filename, io.BytesIO):
            if not os.path.isfile(filename):
                return
        decouplings = np.loadtxt(filename, skiprows=1)
        assert decouplings.shape[1] == 3
        if self.decouplings is not None:
            print('WARNING: overwriting existing decouplings')
            self.decouplings = decouplings

    def save_decouplings_file(self, filename):
        if self.decouplings is None:
            return
        if isinstance(filename, io.BytesIO):
            fid = filename
        else:
            fid = open(filename, 'w')

        fid.write('{}\n'.format(self.decouplings.shape[0]))
        np.savetxt(fid, self.decouplings, fmt='%i %i %f')

        if not isinstance(filename, io.BytesIO):
            fid.close()

    def add_to_decouplings(self, new_decouplings):
        """
        """
        assert new_decouplings.shape[1] == 3
        if self.decouplings is None:
            self.decouplings = new_decouplings
        self.decouplings = np.vstack((
            self.decouplings,
            new_decouplings
        ))

    def inv_get_last_pid(self, parameter):
        """Return the pid of the parameter set corresponding to the final
        inversion results of a given parameter. Return None if the parameter
        type does not exist, or no inversion result was registered.

        Parameters
        ----------
        parameter : str
            The requested parameter type: cre, cim, rmag, rpha

        Returns
        -------
        pid : int|None
            The parameter id, or None
        """
        if ('inversion' in self.a and parameter in self.a['inversion'] and
                len(self.a['inversion'][parameter]) > 0):
            pid = self.a['inversion'][parameter][-1]
            return pid
        return None

    def inv_last_rmag_parset(self):
        """Return the resistivity magnitude of the last iteration. None if no
        inversion data exists.

        Example
        -------
        >>> import crtomo
        ... tdm = crtomo.tdMan('tomodir/')
        ... tdm.inv_last_rmag_parset()

        Returns
        -------
        inv_last_rmag : numpy.ndarray|None
        """
        pid = self.inv_get_last_pid('rmag')
        if pid is None:
            return None
        else:
            return self.parman.parsets[pid]

    def inv_last_rpha_parset(self):
        """Return the phase magnitude of the last inversion iteration.
         None if no inversion data exists.

        Returns
        -------
        inv_last_rpha : numpy.ndarray|None
        """
        pid = self.inv_get_last_pid('rpha')
        if pid is None:
            return None
        else:
            return self.parman.parsets[pid]

    def inv_last_cre_parset(self):
        """Return the real part of the complex resistivity of the last
         inversion iteration. None if no inversion data exists.

        Returns
        -------
        inv_last_cre : numpy.ndarray|None
        """
        pid = self.inv_get_last_pid('cre')
        if pid is None:
            return None
        else:
            return self.parman.parsets[pid]

    def inv_last_cim_parset(self):
        """Return the imaginary part of the complex resistivity of the last
         inversion iteration.None if no inversion data exists.

        Returns
        -------
        inv_last_cim : numpy.ndarray|None
        """
        pid = self.inv_get_last_pid('cim')
        if pid is None:
            return None
        else:
            return self.parman.parsets[pid]

    def reset_data(self):
        """Attempts to reset (delete) all inversion data currently stored in
        the tdMan instance. This is mostly attempted for the impedance data
        (magnitudes, phases, conductivity real and imaginary parts), but could
        be extended to other data (this is currently not done due to complexity
        and missing demand).
        Forward models are also deleted.
        """
        # deletes data actually stored
        self.parman.reset()

        if 'inversion' in self.a and self.a['inversion'] is not None:
            for key in ('rmag', 'rpha', 'cre', 'cim', 'cre_cim'):
                self.a['inversion'][key] = {}

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
        if (self.grid is not None and
           self.configs.configs is not None and
           self.assignments['forward_model'] is not None):
            self.can_model = True

        if (self.grid is not None and
           self.assignments['measurements'] is not None):
            self.can_invert = True

    def load_parset_from_file(self, filename, columns=0):
        """
        Parameters
        ----------
        filename : string
            filename to rho.dat file
        columns : int|list of int
            which columns to use/treat as parameter sets. This settings uses
            zero-indexing, i.e., the first column is 0. Default: column 0

        Returns
        -------
        pids : int|list
            the parameter ids of the imported files

        """
        pids = self.parman.load_model_from_file(filename, columns=columns)
        return pids

    def load_rho_file(self, filename):
        """Load a forward model from a rho.dat file

        Parameters
        ----------
        filename : string
            filename to rho.dat file

        Returns
        -------
        pid_mag : int
            parameter id for the magnitude model
        pid_pha : int
            parameter id for the phase model

        """
        pids = self.parman.load_from_rho_file(filename)
        self.register_magnitude_model(pids[0])
        self.register_phase_model(pids[1])
        return pids

    def save_to_tarfile(self):
        """Save the current modeling/inversion data into a tarfile. Return the
        file as an io.BytesIO object. The tar file is generated completely in
        memory - no files are written to the disc

        Returns
        -------
        tomodir_tar : io.BytesIO
            Tomodir stored in tar file
        """
        tomodir = io.BytesIO()
        tar = tarfile.open(fileobj=tomodir, mode='w:xz')

        # prepare buffers and write them to the tar file
        elem_data = io.BytesIO()
        self.grid.save_elem_file(elem_data)
        info = tarfile.TarInfo()
        info.name = 'grid/elem.dat'
        info.mtime = time.time()
        info.size = elem_data.tell()
        info.type = tarfile.REGTYPE
        elem_data.seek(0)
        tar.addfile(info, elem_data)

        elec_data = io.BytesIO()
        self.grid.save_elec_file(elec_data)
        info = tarfile.TarInfo()
        info.name = 'grid/elec.dat'
        info.mtime = time.time()
        info.size = elec_data.tell()
        info.type = tarfile.REGTYPE
        elec_data.seek(0)
        tar.addfile(info, elec_data)

        crtomo_cfg = io.BytesIO()
        self.crtomo_cfg.write_to_file(crtomo_cfg)
        info = tarfile.TarInfo()
        info.name = 'exe/crtomo.cfg'
        info.mtime = time.time()
        info.size = crtomo_cfg.tell()
        info.type = tarfile.REGTYPE
        crtomo_cfg.seek(0)
        tar.addfile(info, crtomo_cfg)

        decouplings = io.BytesIO()
        self.save_decouplings_file(decouplings)
        info = tarfile.TarInfo()
        info.name = 'exe/decouplings.dat'
        info.mtime = time.time()
        info.size = crtomo_cfg.tell()
        info.type = tarfile.REGTYPE
        decouplings.seek(0)
        tar.addfile(info, decouplings)

        volt_data = io.BytesIO()
        self.save_measurements(volt_data)
        info = tarfile.TarInfo()
        info.name = 'mod/volt.dat'
        info.mtime = time.time()
        info.size = volt_data.tell()
        info.type = tarfile.REGTYPE
        volt_data.seek(0)
        tar.addfile(info, volt_data)

        tar.close()
        tomodir.seek(0)
        return tomodir

        # modelling
        """
        config/config.dat
        rho/rho.dat

        exe/crmod.cfg
        exe/crtomo.cfg
        exe/crt.noisemod
        exe/electrode_capactitances.dat
        exe/decouplings.dat
        exe/prior.model
        [TODO]exe/crt.lamnull

        mod/sens/*
        [TODO] mod/volt.dat
        [TODO] mod/pot/*

        inv/cjg.ctr
        inv/coverage.mag
        coverage.mag.fpi
        [TODO] inv/ata.diag
        [TODO] inv/ata_reg.diag
        [TODO] inv/cov1_m.diag
        [TODO] inv/cov2_m.diag
        [TODO] inv/res_m.diag
        [TODO] inv/eps.ctr
        [TODO] inv/inv.ctr
        [TODO] inv/*.model
        [TODO] inv/*.mag
        [TODO] inv/*.pha
        [TODO] inv/run.ctr
        [TODO] inv/voltXX.dat
        """
        # file1 = io.BytesIO()
        # content_file1 = 'Hi there\nNew line\n'.encode('utf-8')
        # file1.write(content_file1)
        # file1.seek(0)

        # tar = tarfile.open(fileobj=target_file, mode='w:xz')

        # info1 = tarfile.TarInfo()
        # info1.name = 'subdir/File1.txt'
        # info1.mtime = time.time()
        # info1.size = len(content_file1)
        # info1.type = tarfile.REGTYPE

        # tar.addfile(info1, file1)
        # tar.close()

        # target_file.seek(0)
        # print('Reading tar file from memory:')
        # print(
        #     target_file.read()
        #     )
        #     with open('test.tar.xz', 'wb') as fid:
        #         target_file.seek(0)
        #             fid.write(target_file.read())
        #             # extract with tar xvJf test.tar.xz

        #             ## Reading
        #             tar = tarfile.open(fileobj=target_file, mode='r')

    def save_to_tomodir(self, directory, only_for_inversion=False):
        """Save the tomodir instance to a directory structure.

        At this point forward modeling results (voltages, potentials and
        sensitivities) will be saved.

        Inversion results will, at this stage, not be saved (TODO!!!).

        Parameters
        ----------
        directory : str
            Output directory (tomodir). Will be created of it does not exists,
            but otherwise files we be overwritten in any existing directories
        only_for_inversion : bool, optional
            If True, save only files required for an inversion (i.e., omit any
            forward modeling files and results not necessary). Default: False

        Note
        ----

        Test cases:

            * modeling only
            * inversion only
            * modeling and inversion

        """
        self.create_tomodir(directory)

        self.grid.save_elem_file(
            directory + os.sep + 'grid' + os.sep + 'elem.dat'
        )

        self.grid.save_elec_file(
            directory + os.sep + 'grid' + os.sep + 'elec.dat'
        )

        # modeling
        if not only_for_inversion:
            if (self.configs.configs is not None and
                    self.assignments['forward_model'] is not None):
                self.configs.write_crmod_config(
                    directory + os.sep + 'config' + os.sep + 'config.dat'
                )

            if self.assignments['forward_model'] is not None:
                self.parman.save_to_rho_file(
                    directory + os.sep + 'rho' + os.sep + 'rho.dat',
                    self.assignments['forward_model'][0],
                    self.assignments['forward_model'][1],
                )

            self.crmod_cfg.write_to_file(
                directory + os.sep + 'exe' + os.sep + 'crmod.cfg'
            )

            if self.assignments['sensitivities'] is not None:
                self._save_sensitivities(
                    directory + os.sep + 'mod' + os.sep + 'sens',
                )

            if self.assignments['potentials'] is not None:
                self._save_potentials(
                    directory + os.sep + 'mod' + os.sep + 'pot',
                )

        # we always want to save the measurements
        self.save_measurements(
            directory + os.sep + 'mod' + os.sep + 'volt.dat'
        )

        # inversion
        self.crtomo_cfg.write_to_file(
            directory + os.sep + 'exe' + os.sep + 'crtomo.cfg'
        )

        if self.electrode_admittance is not None:
            self._write_el_admittance(directory)

        if self.noise_model is not None:
            self.noise_model.write_crt_noisemod(
                directory + os.sep + 'exe' + os.sep + 'crt.noisemod'
            )

        self.save_decouplings_file(
            directory + os.sep + 'exe' + os.sep + 'decouplings.dat'
        )

        if not os.path.isdir(directory + os.sep + 'inv'):
            os.makedirs(directory + os.sep + 'inv')

        if self.a['prior_model'] is not None:
            self.parman.save_to_rho_file(
                directory + os.sep + 'inv/prior.model',
                *self.a['prior_model']
            )

    def _write_el_admittance(self, directory):
        filename = 'electrode_capacitances.dat'
        with open(directory + os.sep + 'exe' + os.sep + filename, 'w') as fid:
            fid.write('{}\n'.format(self.grid.nr_of_electrodes))
            np.savetxt(
                fid,
                np.ones_like(
                    self.grid.nr_of_electrodes
                ) * self.electrode_admittance
            )

    def save_measurements(self, filename):
        """Save measurements in a file. Use the CRTomo format.

        Do not save anything if no measurements are present in this tdMan
        instance.

        Parameters
        ----------
        filename : string
            path to output filename
        """
        if self.assignments['measurements'] is not None:
            if self.assignments['measurement_errors'] is not None:
                # save individual errors
                self.configs.write_crmod_volt_with_individual_errors(
                    filename,
                    self.assignments['measurements'],
                    self.assignments['measurement_errors'],
                )
            else:
                self.configs.write_crmod_volt(
                    filename,
                    self.assignments['measurements']
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

    def measurements(self, silent=False):
        """Return the measurements associated with this instance.

        If measurements are not present, check if we can model, and then
        run CRMod to load the measurements.

        Parameters
        ----------
        silent : bool, optional
            If False, suppress CRMod output
        """
        # check if we have measurements
        mid = self.assignments.get('measurements', None)
        if mid is None:
            return_value = self.model(
                voltages=True,
                sensitivities=False,
                potentials=False,
                silent=silent,
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

    def _read_modeling_results(self, mod_directory, silent=False):
        """Read modeling results from a given mod/ directory. Possible values
        to read in are:

            * voltages
            * potentials
            * sensitivities

        Parameters
        ----------
        mod_directory : str
            Path to the directory containing the volt.dat file
        silent : bool, optional
            If True, suppress some debug output

        Returns
        -------
        None

        """

        voltage_file = mod_directory + os.sep + 'volt.dat'
        if os.path.isfile(voltage_file):
            if not silent:
                print('reading voltages')
            self.read_voltages(voltage_file)

        sens_files = sorted(glob(
            mod_directory + os.sep + 'sens' + os.sep + 'sens*.dat')
        )
        # check if there are sensitivity files, and that the nr corresponds to
        # the nr of configs
        if (len(sens_files) > 0 and
           len(sens_files) == self.configs.nr_of_configs):
            print('reading sensitivities')
            self._read_sensitivities(mod_directory + os.sep + 'sens')

        # same for potentials
        pot_files = sorted(glob(
            mod_directory + os.sep + 'pot' + os.sep + 'pot*.dat')
        )
        # check if there are sensitivity files, and that the nr corresponds to
        # the nr of configs
        if (len(pot_files) > 0 and
           len(pot_files) == self.configs.nr_of_configs):
            print('reading potentials')
            self._read_potentials(mod_directory + os.sep + 'pot')

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
                self.model(sensitivities=True, silent=True)
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
            fig.savefig('sens_plot.pdf', bbox_inches='tight')

        """

        def _rescale_sensitivity(sens_data):
            norm_value = np.abs(sens_data).max()
            if norm_value == 0:
                sens_normed = sens_data * 0
            else:
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

            sens_mag = self.parman.parsets[cids[0]].copy()
            sens_pha = self.parman.parsets[cids[1]].copy()
        else:
            sens_mag = sens_data[:, 0]
            sens_pha = sens_data[:, 1]

        if absv:
            sens_mag = np.log10(np.abs(sens_mag) / np.abs(sens_mag).max())
            sens_pha = np.log10(np.abs(sens_pha) / np.abs(sens_pha).max())
            cbmin = sens_mag.min()
            cbmax = sens_mag.max()
        else:
            sens_mag /= np.abs(sens_mag).max()
            # _rescale_sensitivity(sens_mag)
            # _rescale_sensitivity(sens_pha)
            cbmin = 0
            cbmax = 1

        # https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.9.0.html#top-level-cmap-registration-and-access-functions-in-mpl-cm
        if mpl_version[0] <= 3 and mpl_version[1] < 9:
            cmap_jet = mpl.cm.get_cmap('jet')
        else:
            cmap_jet = mpl.colormaps['jet']

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

        axes = kwargs.get('axes', None)
        if axes:
            # assume axes to plot to were provided
            if Nx == 2:
                assert len(axes) == 2, "We need two axes"
                fig = axes[0].get_figure()
            else:
                fig = axes.get_figure()
        else:
            fig, axes = plt.subplots(1, Nx, figsize=(15 / 2.54, 12 / 2.54))
        axes = np.atleast_1d(axes)
        # magnitude
        ax = axes[0]
        cid = self.parman.add_data(sens_mag)

        # we always plot the first subplot, the magnitude
        fig, ax, cnorm, cmap, cb, sM = self.plot.plot_elements_to_ax(
            cid=cid,
            ax=ax,
            plot_colorbar=True,
            # cmap_name='seismic',
            cmap_name='jet_r',
            cbsegments=18,
            cbmin=cbmin,
            cbmax=cbmax,
            cblabel='asinh-transformed sensitivity',
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
            converter=PlotManager.converter_asinh,
        )
        if not absv:
            # for the asinh converter
            cb.set_ticks([-1, 0, 1])
            cb.set_ticklabels([
                '-1',
                '0',
                '1',
            ])
            pass
            # cb.set_ticks([0, 0.25, 0.5, 0.75, 1])
            # cb.set_ticklabels([
            #     '-1',
            #     r'$-10^{-2.5}$',
            #     '0',
            #     r'$10^{-2.5}$',
            #     '1',
            # ])

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
        """Import voltages from a volt.dat file

        Parameters
        ----------
        voltage_file : str
            Path to volt.dat file
        """
        if isinstance(voltage_file, (StringIO, )):
            fid = voltage_file
            fid.seek(0)
            working_on_file = False
        else:
            fid = open(voltage_file, 'r')
            working_on_file = True

        items_first_line = fid.readline().strip().split(' ')

        # rewind for reading of complete file later on
        fid.seek(0)

        if int(items_first_line[0]) == 0:
            if working_on_file:
                fid.close()
            # empty file
            return

        individual_errors = False
        if len(items_first_line) == 1:
            # regular volt.dat file
            measurements_raw = np.loadtxt(
                fid,
                skiprows=1,
            )
        elif len(items_first_line) == 2 and items_first_line[1] == 'T':
            individual_errors = True
            # Individual data errors
            measurements_raw = np.genfromtxt(
                fid,
                skip_header=1,
                max_rows=int(items_first_line[0]),
            )
            fid.seek(0)
            (norm_mag, norm_pha) = np.genfromtxt(fid, max_rows=1)

        if working_on_file:
            fid.close()

        measurements = np.atleast_2d(measurements_raw)

        # extract measurement configurations
        A = (measurements[:, 0] / 1e4).astype(int)
        B = (measurements[:, 0] % 1e4).astype(int)
        M = (measurements[:, 1] / 1e4).astype(int)
        N = (measurements[:, 1] % 1e4).astype(int)
        ABMN = np.vstack((A, B, M, N)).T

        # it may happen that we need to switch signs of measurements
        switch_signs = []

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

                    if (current_electrodes_are_equal and
                       voltage_electrodes_are_switched):

                        if len(self.configs.measurements.keys()) > 0:
                            # print('asdasd')
                            # import IPython
                            # IPython.embed()
                            # raise Exception(
                            #     'need to switch electrode polarity, but ' +
                            #     'there are already measurements stored for '
                            #     + 'the old configuration!')
                            # exit()
                            switch_signs += [nr]
                        else:
                            # switch M/N in configurations
                            self.configs.configs[nr, :] = new_config
                    else:
                        raise Exception(
                            'There was an error matching configurations of ' +
                            'voltages with configurations already imported'
                        )

        # change sign of R measurements that require it
        measurements[switch_signs, 2] *= -1

        # add measurements to the config instance
        mid_mag = self.configs.add_measurements(measurements[:, 2])
        if measurements.shape[1] >= 4:
            mid_pha = self.configs.add_measurements(measurements[:, 3])
        else:
            mid_pha = None
        # register those measurements as 'the measurements', used, e.g., for a
        # subsequent inversion
        self.assignments['measurements'] = [mid_mag, mid_pha]

        if individual_errors:
            mid_mag_error = self.configs.add_measurements(measurements[:, 4])
            mid_pha_error = self.configs.add_measurements(measurements[:, 5])
            self.register_data_errors(
                mid_mag_error, mid_pha_error, norm_mag, norm_pha
            )

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
            if not silent:
                print(return_text)

            # print('Return text:', return_text)
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

        Note that this function will always do the full modeling. No caching
        involved.

        Please use .measurements to accessed cached measurements.

        Parameters
        ----------
        voltages : bool, optional
            if True, compute voltages for registered quadrupoles. Default: True
        sensitivities : bool, optional
            if True, compute sensitivities for registered quadrupoles. Default:
            False
        potentials : bool, optional
            if True, compute potential fields for all current injections.
            Default: False
            TODO: check if implemented in the Python wrapper
        output_directory : str|None, optional
            if this is a string, treat it as an output directory in which the
            tomodir used for the modeling is saved.
            Will raise an exception if the output directory already exists.
            Default: None
        silent : bool, optional
            if True, suppress most of the output. Default: False

        Returns
        -------
        None

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
        tempdir : str
            directory which to use as a tomodir
        catch_output : bool, optional
            if True, catch all outputs of the CRTomo call (default: True)
        cores : int, optional
            how many cores to use. (default 2)

        Returns
        -------
        success: bool
            False if an error was detected
        error_msg: str
            Error message. None if not error was encountered.
        output: str
            Output of the actual CRTomo call
        """
        nr_cores = kwargs.get('cores', 2)
        print('Attempting inversion in directory: {0}'.format(tempdir))
        pwd = os.getcwd()
        os.chdir(tempdir)

        self.save_to_tomodir('.')
        os.chdir('exe')
        binary = CRBin.get('CRTomo')
        print('Using binary: {0}'.format(binary))
        print('Calling CRTomo')
        # store env variable
        env_omp = os.environ.get('OMP_NUM_THREADS', '')
        os.environ['OMP_NUM_THREADS'] = '{0}'.format(nr_cores)
        try:
            output = subprocess.check_output(
                binary,
                shell=True,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as error:
            print('CRTomo returned a non-zero exit code')
            print('Return code: {}'.format(error.returncode))
            print(error.output)
            raise Exception('CRTomo calling error')

        if catch_output:
            print(output)

        # reset environment variable
        os.environ['OMP_NUM_THREADS'] = env_omp

        print('Inversion attempt finished')
        if os.path.isfile('error.dat'):
            error_message = open('error.dat', 'r').read()
            success = False
        else:
            success = True
            error_message = None

        os.chdir(pwd)
        if success:
            print('Attempting to import the results')
            self.read_inversion_results(tempdir)
            print('Statistics of last iteration:')
            print(self.inv_stats.iloc[-1])
        else:
            print('There was an error while trying to invert')
            print('Please see .crtomo_error_msg and .crtomo_output')

        self.crtomo_output = output
        self.crtomo_error_msg = error_message

        return success

    def invert(self, output_directory=None, catch_output=True, **kwargs):
        """Invert this instance, and import the result files

        No directories/files will be overwritten. Raise an IOError if the
        output directory exists.

        Parameters
        ----------
        output_directory : string, optional
            use this directory as output directory for the generated tomodir.
            If None, then a temporary directory will be used that is deleted
            after data import.
        catch_output : bool, optional
            if True, do not show CRTomo output. Default:True
        cores : int, optional
            how many cores to use for CRTomo
        **kwargs : Are supplied to :py:meth:`crtomo.tdMan._invert`

        Returns
        -------
        return_code : bool
            Return 0 if the inversion completed successfully. Return 1 if no
            measurements are present.
        """
        self._check_state()
        if self.can_invert:
            if output_directory is not None:
                if not os.path.isdir(output_directory):
                    os.makedirs(output_directory)
                    tempdir = output_directory
                    success = self._invert(
                        tempdir, catch_output, **kwargs
                    )
                else:
                    raise IOError(
                        'Output directory already exists: {0}'.format(
                            output_directory
                        )
                    )
            else:
                with tempfile.TemporaryDirectory(dir=self.tempdir) as tempdir:
                    success = self._invert(
                        tempdir, catch_output, **kwargs
                    )

            if success:
                return 0
            else:
                return 1
        else:
            print(
                'Sorry, no measurements present, cannot model yet'
            )
            return 1

    def invert_with_crhydra(self):
        import cr_hydra.settings
        import crh_add
        import crh_retrieve
        import hashlib
        import platform

        cr_hydra_config, error_code = cr_hydra.settings.get_config(True)
        if error_code != 0:
            assert ('No cr_hydra config file found. Cannot proceed')
        print('cr_hydra config found. Proceeding')

        print('Creating in-memory .tar.xz file of tomodir')
        tarxz = self.save_to_tarfile()

        # upload the simulation
        crh_settings = {
            'datetime_init': '{}'.format(
                datetime.datetime.now(tz=datetime.timezone.utc)
            ),
        }
        crh_settings['source_computer'] = platform.node()
        crh_settings['sim_type'] = 'inv'
        crh_settings['crh_file'] = 'jupyter-lab'
        crh_settings['username'] = 'crtomo-tools'

        engine = crh_add._get_db_engine(cr_hydra_config)
        connection = engine.connect()

        archive_file = 'dasdasd'
        file_id = crh_add._upload_binary_data(
            tarxz, archive_file, connection
        )
        crh_settings['tomodir_unfinished_file'] = file_id

        sim_id = crh_add._upload_simulation(crh_settings, connection)
        crh_settings['sim_id'] = sim_id
        crh_add._activate_simulation(sim_id, connection)

        print('cr_hydra simulation id:', sim_id)

        # wait until the inversion finished
        is_finished = None
        while is_finished is None:
            print('Waiting for inversion to finish')
            time.sleep(5)
            is_finished = crh_retrieve._is_finished(sim_id, connection)
        print('Inversion finished')

        print('Downloading')
        # retrieve the inversion
        # NOTE: This should be called from crh_retrieve
        result = connection.execute(
            crh_retrieve.text(
                'select hash, data from binary_data where index=:data_id;'
            ),
            parameters={
                'data_id': is_finished,
            },
        )
        assert result.rowcount == 1
        file_hash, binary_data = result.fetchone()

        # check hash
        m = hashlib.sha256()
        m.update(binary_data)
        assert file_hash == m.hexdigest(), "sha256 does not match"

        result_data = io.BytesIO(bytes(binary_data))
        with open('output.tar.xz', 'wb') as fid:
            fid.write(result_data.read())

    def read_inversion_results(self, tomodir):
        """Import inversion results from a tomodir into this instance

        .. warning::
            This function is not finished yet and does not import ALL crtomo
            information yet.

        Parameters
        ----------
        tomodir : str
            Path to tomodir

        """
        print('Reading inversion results')
        self._read_inversion_results(tomodir)
        self._read_inversion_fwd_responses(tomodir)
        self.inv_stats = self._read_inv_ctr(tomodir)
        self._read_resm_m(tomodir)
        self._read_l1_coverage(tomodir)
        self._read_l2_coverage(tomodir)
        self.eps_data = self._read_eps_ctr(tomodir)
        # for simplicity, add configurations to psi data
        has_eps_data = False
        if isinstance(self.eps_data, list):
            if len(self.eps_data) > 1:
                has_eps_data = True
        if self.configs.configs is not None and len(
                self.configs.configs) > 0 and has_eps_data:
            for iteration in range(1, len(self.eps_data)):
                for index, key in enumerate('abmn'):
                    self.eps_data[
                        iteration
                    ][key] = self.configs.configs[:, index]

    def _read_inversion_fwd_responses(self, tomodir):
        """Import the forward responses for all iterations of a given inversion

        Parameters
        ----------
        tomodir : str
            Path to tomodir
        """
        basedir = tomodir + os.sep + 'inv' + os.sep
        volt_files = sorted(glob(basedir + 'volt*.dat'))
        pids_rmag = []
        pids_rpha = []
        pids_wdfak = []
        for filename in volt_files:
            pids = self.configs.load_crmod_data(
                filename, is_forward_response=True, try_fix_signs=True)
            pids_rmag.append(pids[0])
            pids_rpha.append(pids[1])
            pids_wdfak.append(pids[2])

        # self.a['inversions'] already created by .read_inversion_results
        self.assignments['inversion']['fwd_response_rmag'] = pids_rmag
        self.assignments['inversion']['fwd_response_rpha'] = pids_rpha
        self.assignments['inversion']['fwd_response_wdfak'] = pids_wdfak

    def _read_inversion_results(self, tomodir):
        """Import resistivity magnitude/phase and real/imaginary part of
        conductivity for all iterations

        Parameters
        ----------
        tomodir : str
            Path to tomodir
        """
        basedir = tomodir + os.sep + 'inv' + os.sep
        inv_mag = sorted(glob(basedir + 'rho*.mag'))
        inv_pha = sorted(glob(basedir + 'rho*.pha'))
        inv_sig = sorted(glob(basedir + 'rho*.sig'))

        assert len(inv_pha) == 0 or len(inv_mag) == len(inv_pha)
        assert len(inv_sig) == 0 or len(inv_mag) == len(inv_sig)

        pids_mag = [
            self.parman.load_inv_result(filename, is_log10=True) for filename
            in inv_mag
        ]
        pids_pha = [self.parman.load_inv_result(filename) for filename in
                    inv_pha]
        pids_sig = [
            self.parman.load_inv_result(
                filename, columns=[0, 1], is_log10=False
            ) for filename in inv_sig
        ]

        if len(pids_sig) > 0:
            pids_cre = [x[0] for x in pids_sig]
            pids_cim = [x[1] for x in pids_sig]
        else:
            if len(pids_pha) > 0:
                pids_cre = []
                pids_cim = []
                for pid_rmag, pid_rpha in zip(pids_mag, pids_pha):
                    # compute the admittances by hand
                    impedance = self.parman.parsets[pid_rmag] * np.exp(
                        1j * self.parman.parsets[pid_rpha] / 1000
                    )
                    admittance = 1 / impedance
                    pids_cre += [self.parman.add_data(np.real(admittance))]
                    pids_cim += [self.parman.add_data(np.imag(admittance))]
            else:
                pids_cre = None
                pids_cim = None

        self.assignments['inversion'] = {
            'rmag': pids_mag,
            'rpha': pids_pha,
            'cre_cim': pids_sig,
            'cre': pids_cre,
            'cim': pids_cim,
        }

    def plot_eps_data_hist(self, filename=None):
        """Plot histograms of data residuals and data error weighting

        TODO :
            * add percentage of data below/above the RMS value

        Parameters
        ----------
        filename : string|None
            if not None, then save plot to this file

        Returns
        -------
        figure : matplotlib.Figure|None
            if filename is None, then return the generated figure

        """
        assert self.eps_data is not None
        dfs = self.eps_data
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
        if filename is not None:
            fig.savefig(filename, dpi=300)
        else:
            return fig

    def plot_eps_data(self, filename=None):
        """Plot data residuals and data error weighting

        Parameters
        ----------
        filename : string|None
            if not None, then save plot to this file

        Returns
        -------
        figure : matplotlib.Figure|None
            if filename is None, then return the generated figure

        """
        assert self.eps_data is not None
        dfs = self.eps_data
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
        if filename is not None:
            fig.savefig(filename, dpi=300)
        else:
            return fig

    @staticmethod
    def _read_eps_ctr(tomodir):
        """Parse a CRTomo eps.ctr file.

        TODO: change parameters to only provide eps.ctr file

        Parameters
        ----------
        tomodir : string
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
                    del (data[0])
                data[0] = data[0].replace('-Phase (rad)', '-Phase(rad)')
                tfile = StringIO(''.join(data))
                df = pd.read_csv(
                    tfile,
                    sep=r'\s+',
                    # delim_whitespace=True,
                    na_values=['Infinity'],
                )
                dfs.append(df)
        return dfs

    def _read_inv_ctr(self, tomodir):
        """Read in selected results of the inv.ctr file

        Parameters
        ----------
        tomodir : string
            directory path to a tomodir

        Returns
        -------
        inv_ctr :    ?
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
            r'((?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)',
            '|',
            '(?:NaN))'
        ))

        reg_int = r'(\d{1,3})'

        # (t1,a3,t5,i3,t11,g10.4,t69,g10.4,t81,g10.4,t93,i4)
        # first iteration line of non-robust inversion
        reg_it1_norob = ''.join((
            r'([a-zA-Z]{1,3})',
            ' *' + reg_int,
            ' *' + reg_float,
            ' *' + reg_float,
            ' *' + reg_float,
            ' *' + reg_int,
        ))
        # first iteration line of robust inversion
        reg_it1_robust = ''.join((
            '([a-zA-Z]{1,3})',
            r' *(\d{1,3})',
            ' *' + reg_float,   # data RMS
            ' *' + reg_float,   # mag RMS
            ' *' + reg_float,   # pha RMS
            ' *' + reg_int,     # nr excluded data
            ' *' + reg_float,   # L1-ratio
        ))

        # second-to-last iterations, robust
        reg_it2plus_rob = ''.join((
            r'([a-zA-Z]{1,3})',
            r' *(\d{1,3})',
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
            r' *(\d{1,3})',
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
            r' *(\d{1,3})',
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
            r' *(\d{1,3})',
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
        df = df.reindex([
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

    def plot_inversion_evolution(self, filename=None):
        """Plot the evolution of inversion properties (lambda, RMS, ...)
        Parameters
        ----------
        filename : string|None
            if not None, save plot to this file

        Returns
        -------
        fig : matplotlib.Figure|None
            if filename is None, return the figure object
        """
        assert self.inv_stats is not None
        df = self.inv_stats
        fig, ax = plt.subplots(figsize=(16 / 2.54, 9 / 2.54), dpi=300)

        plot_objs = []

        # main iterations
        main = df.query('type == "main"')
        obj = ax.plot(
            main['main_iteration'],
            main['lambda'],
            '.-',
            color='k',
            label=r'$\lambda$'
        )
        # import IPython
        # IPython.embed()
        plot_objs.append(obj[0])
        ax.set_ylabel(r'$\lambda$')
        ax2 = ax.twinx()
        obj = ax2.plot(
            main['main_iteration'],
            main['dataRMS'],
            '.-',
            color='r',
            label='dataRMS',
        )
        plot_objs.append(obj[0])
        ax2.set_ylabel('data RMS')
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('axes', 1.2))

        # update iterations
        g = df.groupby('main_iteration')

        # https://matplotlib.org/stable/api/prev_api_changes/api_changes_3.9.0.html#top-level-cmap-registration-and-access-functions-in-mpl-cm
        if mpl_version[0] <= 3 and mpl_version[1] < 9:
            cm = mpl.cm.get_cmap('jet')
        else:
            cm = mpl.colormaps['jet']

        SM = mpl.cm.ScalarMappable(norm=None, cmap=cm)
        colors = SM.to_rgba(np.linspace(0, 1, g.ngroups))
        for color, (name, group) in zip(colors, g):
            # plot update evolution
            updates = group.query('type == "update"')
            print('#############################')
            print(name)
            print(updates)
            obj = ax3.scatter(
                range(0, updates.shape[0]),
                updates['lambda'],
                color=color,
                label='it {}'.format(name),
            )
            plot_objs.append(obj)
        ax3.set_ylabel('update lambdas')

        ax.set_xlabel('iteration number')

        ax.grid(None)
        ax2.grid(None)
        ax3.grid(None)

        # added these three lines
        labs = [label.get_label() for label in plot_objs]
        ax.legend(plot_objs, labs, loc=0, fontsize=6.0)

        fig.tight_layout()
        if filename is None:
            return fig
        else:
            fig.savefig(filename, dpi=300)

    def _read_l1_coverage(self, tomodir):
        """Read in the L1 data-weighted coverage (or cumulated sensitivity)
        of an inversion

        Parameters
        ----------
        tomodir : str
            directory path to a tomodir
        """
        l1_dw_coverage_file = os.sep.join(
            (tomodir, 'inv', 'coverage.mag')
        )
        if not os.path.isfile(l1_dw_coverage_file):
            print(
                'Info: coverage.mag not found: {0}'.format(l1_dw_coverage_file)
            )
            print(os.getcwd())
            return 1

        try:
            nr_cells, max_sens = np.loadtxt(l1_dw_coverage_file, max_rows=1)
            l1_dw_log10_norm = np.loadtxt(
                l1_dw_coverage_file, skiprows=1)[:, 2]
        except Exception:
            # maybe old format - ignore for now
            return
        self.header_l1_dw_log10_norm = {
            'max_value': max_sens,
        }
        pid = self.parman.add_data(l1_dw_log10_norm)
        if 'inversion' not in self.a:
            self.a['inversion'] = {}

        self.a['inversion']['l1_dw_log10_norm'] = pid

    def _read_l2_coverage(self, tomodir):
        """Read in the L2 data-weighted coverage (or cumulated sensitivity)
        of an inversion

        Parameters
        ----------
        tomodir : str
            directory path to a tomodir
        """
        l2_dw_coverage_file = os.sep.join(
            (tomodir, 'inv', 'ata.diag')
        )
        if not os.path.isfile(l2_dw_coverage_file):
            print(
                'Info: ata.diag not found: {0}'.format(l2_dw_coverage_file)
            )
            print(os.getcwd())
            return 1

        nr_cells, min_l2, max_l2 = np.loadtxt(l2_dw_coverage_file, max_rows=1)
        l2_dw_log10_norm = np.loadtxt(l2_dw_coverage_file, skiprows=1)[:, 1]
        self.header_l1_dw_log10_norm = {
            'min_value': min_l2,
            'max_value': max_l2,
        }
        pid = self.parman.add_data(l2_dw_log10_norm)
        if 'inversion' not in self.a:
            self.a['inversion'] = {}

        self.a['inversion']['l2_dw_log10_norm'] = pid

    def _read_resm_m(self, tomodir):
        """Read in the diagonal entries of the resolution matrix of an
        inversion

        Parameters
        ----------
        tomodir : str
            directory path to a tomodir

        """
        resm_file = tomodir + os.sep + 'inv' + os.sep + 'res_m.diag'
        if not os.path.isfile(resm_file):
            print('Info: res_m.diag not found: {0}'.format(resm_file))
            print(os.getcwd())
            return 1

        with open(resm_file, 'rb') as fid:
            first_line = fid.readline().strip()
            header_raw = np.fromstring(first_line, count=4, sep=' ')
            # nr_cells = int(header_raw[0])
            self.header_res_m = {
                'r_lambda': float(header_raw[1]),
                'r_min': float(header_raw[2]),
                'r_max': float(header_raw[3]),
            }

            subdata = np.genfromtxt(fid)
            pid = self.parman.add_data(subdata[:, 0])
            if 'inversion' not in self.a:
                self.a['inversion'] = {}

            self.a['inversion']['resm'] = pid

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
        if isinstance(mag, np.ndarray):
            # make sure that this array is 1D at the most
            # the 0 indicates only one measurement
            assert len(mag.squeeze().shape) in (0, 1)
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

    def show_parset(self, pid, **kwargs):
        """Plot a given parameter set. Thin wrapper around
        :py:meth:`crtomo.plotManager.plotManager.plot_elements_to_ax`.

        kwargs will be directly supplied to the plot function
        """
        if 'ax' in kwargs:
            ax = kwargs.get('ax')
            kwargs.pop('ax')
            fig = ax.get_figure()
        else:
            fig, ax = plt.subplots()
        self.plot.plot_elements_to_ax(pid, ax=ax, **kwargs)
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

    @functools.wraps(pM.ParMan.extract_points)
    def extract_points(self, pid, points):
        values = self.parman.extract_points(pid, points)
        return values

    @functools.wraps(pM.ParMan.extract_along_line)
    def extract_along_line(self, pid, xy0, xy1, N=10):
        values = self.parman.extract_along_line(pid, xy0, xy1, N)
        return values

    def set_prior_model(self, pid_rmag, pid_rpha):
        self.assignments['prior_model'] = (pid_rmag, pid_rpha)
        self.crtomo_cfg['prior_model'] = '../inv/prior.model'

    def set_starting_model(self, pid_rmag, pid_rpha):
        self.set_prior_model(pid_rmag, pid_rpha)

    def register_data_errors(
            self, mid_rmag_error, mid_rpha_error=None, norm_mag=1, norm_pha=1):
        """Register individual data errors.

        The normalization factors are used as follows:

            mag_error -> mag_error / norm_mag ** 2
            pha_error -> pha_error / norm_pha ** 2

        That means that you need to decrease the values in order to increase
        individual errors.

        Parameters
        ----------
        mid_mag_error : int
            ID to the corresponding measurement in tdMan.configs holding the
            magnitude errors (linear)
        mid_pha_error : int
            ID to the corresponding measurement in tdMan.configs holding the
            phase errors (linear)
        norm_mag : float
            Magnitude errors can be normalized by the square of this value.
        norm_pha : float
            Phase errors can be normalized by the square of this value.

        """
        self.assignments['measurement_errors'] = [
            mid_rmag_error, mid_rpha_error
        ]
        self.assignments['error_norm_factor_mag'] = norm_mag
        self.assignments['error_norm_factor_pha'] = norm_pha

    def copy(self):
        """Provide a copy of yourself. Do not copy modeling or inversion
        results, but copy everything that can be used for modeling or
        inversion
        """
        tdm_copy = tdMan(grid=self.grid)
        tdm_copy.crtomo_cfg = self.crtomo_cfg.copy()
        tdm_copy.crmod_cfg = self.crmod_cfg.copy()
        # configs

        # forward model
        if self.a['forward_model'] is not None:
            raise Exception('not implemented yet')

        # data
        if len(self.a['measurements']) > 1:
            raise Exception('not implemented yet')

        return tdm_copy

    def sensitivity_center_of_masses(self, mode='none'):
        """

        """
        assert 'sensitivities' in self.a, \
            "This function requires sensitivities"
        mag_sens_indices = [
            self.a['sensitivities'][key][0] for key in sorted(
                self.a['sensitivities'].keys()
            )
        ]
        coms = self.parman.center_of_mass_value_multiple(
            mag_sens_indices,
            mode=mode,
        )
        return coms

    def plot_inversion_result_rmag(
            self, figsize=None, log10=False, overlay_coverage=False,
            **kwargs):
        """Plot the final inversion results, magnitude-results only

        Parameters
        ----------
        figsize: (float, float)|None
            Figure size of the matplotlib figure in inches.
        log10: bool, default: False
            Shortcut to force a log10 conversion of the resistivity data
        overlay_coverage: bool, default: False
            If True, use the cumulated coverage to adjust the transparency of
            the plot
        **kwargs: dict
            will be propagated into self.plot.plot_elements_to_ax

        Returns
        -------
        fig: matplotlib.Figure
            The created figure
        ax: matplotlib.Axes
            Plot axes

        """
        assert self.assignments['inversion'] is not None, \
            'need inversion results to plot anything'
        pid_mag = self.assignments['inversion']['rmag'][-1]

        if 'plot_colorbar' not in kwargs:
            kwargs['plot_colorbar'] = True
        if 'cmap_name' not in kwargs:
            kwargs['cmap_name'] = 'turbo'
        if 'cblabel' not in kwargs:
            kwargs['cblabel'] = reda_units.get_label('rho', log10=log10)

        if log10:
            kwargs['converter'] = PlotManager.converter_abs_log10

        if figsize is None:
            figsize = (16 / 2.54, 8 / 2.54)

        if overlay_coverage:
            key = 'l1_dw_log10_norm'
            if key in self.a['inversion']:
                abscov = np.abs(
                    self.parman.parsets[self.a['inversion'][key]]
                )
                normcov = np.divide(abscov, 3)
                normcov[np.where(normcov > 1)] = 1
                alpha_channel = np.subtract(1, normcov)

                kwargs['cid_alpha'] = alpha_channel

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        self.plot.plot_elements_to_ax(
            pid_mag,
            ax=ax,
            **kwargs,
        )

        fig.tight_layout()
        return fig, ax

    def plot_coverage(self, figsize=None, **kwargs):
        """Plot the final cumulated coverage

        Parameters
        ----------
        figsize: (float, float)|None
            Figure size of the matplotlib figure in inches.
        **kwargs: dict
            will be propagated into self.plot.plot_elements_to_ax

        Returns
        -------
        fig: matplotlib.Figure
            The created figure
        ax: matplotlib.Axes
            Plot axes

        """
        assert self.assignments['inversion'] is not None, \
            'need inversion results to plot anything'

        if 'plot_colorbar' not in kwargs:
            kwargs['plot_colorbar'] = True
        if 'cmap_name' not in kwargs:
            kwargs['cmap_name'] = 'turbo'
        # if 'cblabel' not in kwargs:
        #     kwargs['cblabel'] = r'$|\rho| [\Omega m]$'

        if figsize is None:
            figsize = (16 / 2.54, 8 / 2.54)

        key = 'l1_dw_log10_norm'
        if key in self.a['inversion']:
            pid_l1_cov = self.a['inversion'][key]

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        self.plot.plot_elements_to_ax(
            pid_l1_cov,
            ax=ax,
            **kwargs,
        )

        fig.tight_layout()
        return fig, ax

    def plot_decouplings_to_ax(self, ax, plot_transistions=True):
        """Visualize decouplings. Usually you may want to plot the mesh to this
        axis first

        Parameters
        ----------
        ax: matplotlib.Axes
            Axis to plot to
        plot_transistions: bool (True)
            If True, then also visualize the decoupled transitions

        """
        if self.decouplings is None:
            return

        if plot_transistions:
            centroids = self.grid.get_element_centroids()
            for A, B, strength in self.decouplings:
                A = int(A)
                B = int(B)
                ax.plot(
                    [centroids[A][0], centroids[B][0]],
                    [centroids[A][1], centroids[B][1]],
                    color='green',
                    linewidth=1.0 * strength + 0.1,
                )

        for element1, element2, strength in self.decouplings:
            element1 = int(element1)
            element2 = int(element2)

            nodes = np.intersect1d(
                self.grid.elements[element1],
                self.grid.elements[element2]
            )
            (a_x, a_y) = self.grid.nodes['presort'][nodes[0]][1:3]
            (b_x, b_y) = self.grid.nodes['presort'][nodes[1]][1:3]
            ax.plot([a_x, b_x], [a_y, b_y], color='r')

    def plot_inversion_misfit_pseudosection(self):
        if self.eps_data is None:
            return
        psi = self.eps_data[-1][['a', 'b', 'm', 'n', 'psi']]
        rms = self.inv_stats.iloc[-1]['dataRMS']
        from reda.plotters.pseudoplots import plot_pseudosection_type2
        fig, ax, cb = plot_pseudosection_type2(
            psi,
            'psi',
            markersize=100,
            cmap='seismic',
            cbmin=0,
            cbmax=2,
            title='RMS: {} Mag-Error: {} % + {}'.format(
                rms,
                self.crtomo_cfg['mag_rel'],
                self.crtomo_cfg['mag_abs'],
            ),
        )

        # import numpy as np
        # import matplotlib.pylab as plt
        # rms = 1 / np.sqrt(
        # psi.shape[0]) * np.sqrt(np.sum(psi['psi'] ** 2))
        # print('error weighted RMS:', rms)

        # fig, ax = plt.subplots()
        # _ = ax.hist(psi['psi'], 100)

        return fig, ax, cb

    def get_fwd_reda_container(self):
        """Return a REDA container, either reda.ERT, or reda.CR, with modeled
        data

        Returns
        -------
        container : reda.ERT|reda.CR|None
        """
        m_ids = self.a['measurements']
        if m_ids is None:
            return None

        if len(m_ids) == 1:
            # ERT
            ert = reda.ERT()
            data = np.hstack((
                self.configs.configs,
                self.configs.measurements[m_ids[0]][:, np.newaxis],
            ))
            df = pd.DataFrame(
                data,
                columns=['a', 'b', 'm', 'n', 'r', 'rpha'],
            )
            ert.add_dataframe(df)
            return ert
        elif len(m_ids) == 2:
            # Complex-resistivity
            cr = reda.CR()
            data = np.hstack((
                self.configs.configs,
                self.configs.measurements[m_ids[0]][:, np.newaxis],
                self.configs.measurements[m_ids[1]][:, np.newaxis],
            ))
            df = pd.DataFrame(
                data,
                columns=['a', 'b', 'm', 'n', 'r', 'rpha'],
            )
            cr.add_dataframe(df)
            return cr
        return None

    def plot_pseudo_locs(self, spacing=1.0):
        """Plot pseudo-locations of measurement configurations.
        This function does not take into account real electrode locations.
        However, for surface configurations a 'spacing' parameter can be
        provided

        Parameters
        ----------
        spacing: float, default=1.0
            Electrode spacing
        """
        ert = reda.ERT()
        data = np.hstack((
            self.configs.configs,
            np.ones(self.configs.configs.shape[0])[:, np.newaxis],
        ))
        df = pd.DataFrame(
            data,
            columns=['a', 'b', 'm', 'n', 'r'],
        )
        ert.add_dataframe(df)
        fig, ax, cb = ert.pseudosection_type2(
            markersize=100,
            spacing=spacing,
            xlabel='X-Center [m]',
            ylabel='Pseudodepth [m]',
        )
        cb.remove()
        return fig, ax
