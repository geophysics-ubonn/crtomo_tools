"""Representations of CRMod and CRTomo configurations.

Examples
--------

    >>> import crtomo.cfg as CRcfg
        crmod_cfg = CRcfg.crmod_config()
        print(crmod_cfg)
    ***FILES***       !  mswitch
    ../grid/elem.dat       !  elem
    ../grid/elec.dat       !  elec
    ../rho/rho.dat       !  rho
    ../config/config.dat       !  config
    F       !  write_pot
    ../mod/pot/pot.dat       !  pot_file
    T       !  write_volts
    ../mod/volt.dat       !  volt_file
    F       !  write_sens
    ../mod/sens/sens.dat       !  sens_file
    F       !  another_dataset
    1       !  2D
    F       !  fictitious_sink
    1660       !  sink_node
    F       !  boundary_values
    boundary.dat       !  boundary_file

TODO
----

* we could also add help texts here for each parameter

"""
import io


class crmod_config(dict):
    """
    Write CRMod configuration files (crmod.cfg).

    This class is essentially a dict of CRMod configurations with a few extra
    functions that know how to write a proper crmod.cfg file.

    Examples
    --------

    >>> import crtomo.cfg as cCFG
        crmod_cfg = cCfg.crmod_config()
        crmod_cfg.write_to_file('crmod.cfg')

    """
    def __init__(self, *arg, **kw):
        super(crmod_config, self).__init__(*arg, **kw)
        self.set_defaults()

        self.key_order = (
            'mswitch',
            'elem',
            'elec',
            'rho',
            'config',
            'write_pots',
            'pot_file',
            'write_volts',
            'volt_file',
            'write_sens',
            'sens_file',
            'another_dataset',
            '2D',
            'fictitious_sink',
            'sink_node',
            'boundary_values',
            'boundary_file'
        )

        # boolean options
        self.bools = (
            'write_pots',
            'write_volts',
            'write_sens',
        )

    def _check_and_convert_bools(self):
        """Replace boolean variables by the characters 'F'/'T'
        """
        replacements = {
            True: 'T',
            False: 'F',
        }

        for key in self.bools:
            if isinstance(self[key], bool):
                self[key] = replacements[self[key]]

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        new_copy = crmod_config()
        # translate the keys
        for key in self.keys():
            new_copy[key] = self[key]
        return new_copy

    def __deepcopy__(self, memo):
        print('deepcopy')
        raise Exception('not implemented')

    def set_defaults(self):
        """
        Fill the dictionary with all defaults
        """
        self['mswitch'] = '***FILES***'
        self['elem'] = '../grid/elem.dat'
        self['elec'] = '../grid/elec.dat'
        self['rho'] = '../rho/rho.dat'
        self['config'] = '../config/config.dat'
        self['write_pots'] = 'F'  # ! potentials ?
        self['pot_file'] = '../mod/pot/pot.dat'
        self['write_volts'] = 'T'  # ! measurements ?
        self['volt_file'] = '../mod/volt.dat'
        self['write_sens'] = 'F'  # ! sensitivities ?
        self['sens_file'] = '../mod/sens/sens.dat'
        self['another_dataset'] = 'F'  # ! another dataset ?
        self['2D'] = '1'  # ! 2D (=0) or 2.5D (=1)
        self['fictitious_sink'] = 'F'  # ! fictitious sink ?
        self['sink_node'] = '1660'  # ! fictitious sink node number
        self['boundary_values'] = 'F'  # ! boundary values ?
        self['boundary_file'] = 'boundary.dat'

    def write_to_file(self, filename):
        """
        Write the configuration to a file. Use the correct order of values.
        """
        self._check_and_convert_bools()

        if isinstance(filename, io.BytesIO):
            fid = filename
        else:
            fid = open(filename, 'wb')

        for key in self.key_order:
            if (key == -1):
                fid.write(bytes('\n', 'utf-8'))
            else:
                fid.write(
                    bytes(
                        '{0}\n'.format(self[key]),
                        'utf-8',
                    )
                )

        # do not close BytesIO stream
        if not isinstance(filename, io.BytesIO):
            fid.close()

    def __repr__(self):
        self._check_and_convert_bools()
        representation = ''
        for key in self.key_order:
            if key == -1:
                representation += '\n'
            else:
                representation += '{0}       !  {1}\n'.format(self[key], key)
        return representation

    def __str__(self):
        return self.__repr__()


class crtomo_config(dict):
    """
    Write CRTomo configuration files (crtomo.cfg).

    This class is essentially a dict of CRTomo configurations with a few extra
    functions that know how to write a proper crtomo.cfg file.
    """
    def __init__(self, *arg, **kw):
        super(crtomo_config, self).__init__(*arg, **kw)
        self.set_defaults()

        # -1 indicates an empty line
        self.key_order = (
            'mswitch', 'elem', 'elec', 'volt', 'inv_dir', 'diff_inv',
            -1, 'prior_model', -1, 'iseed_var', 'cells_x', 'cells_z',
            'ani_x', 'ani_z', 'max_it', 'dc_inv', 'robust_inv',
            'fpi_inv', 'mag_rel', 'mag_abs', 'pha_a1', 'pha_b',
            'pha_rel', 'pha_abs', 'hom_bg', 'hom_mag', 'hom_pha',
            'another_ds', 'd2_5', 'fic_sink', 'fic_sink_node',
            'boundaries', 'boundaries_file', 'mswitch2', 'lambda',
        )

        self.mswitch_values = {
            'l1_cov_dw': 1,
            'lcov1': 2,
            'res_m': 4,
            'lcov2': 8,
            'ols_gauss': 16,
            'err_ellipse': 32,
            'force_neg_phase': 128,
            'lsytop': 256,
            'lvario': 512,
            'verbose': 1024,
            'verbose_dat': 2048,
        }

    def set_defaults(self):
        """Fill the dictionary with all defaults
        """
        self['mswitch'] = 1
        self['elem'] = '../grid/elem.dat'
        self['elec'] = '../grid/elec.dat'
        self['volt'] = '../mod/volt.dat'
        self['inv_dir'] = '../inv'
        self['prior_model'] = ''
        self['diff_inv'] = 'F ! difference inversion?'
        self['iseed_var'] = 'iseed variance'
        self['cells_x'] = '0    ! # cells in x-direction'
        self['cells_z'] = '0    ! # cells in z-direction'
        self['ani_x'] = '1.000  ! smoothing parameter in x-direction'
        self['ani_z'] = '1.000  ! smoothing parameter in z-direction'
        self['max_it'] = '20    ! max. nr of iterations'
        self['dc_inv'] = 'F     ! DC inversion?'
        self['robust_inv'] = 'T     ! robust inversion?'
        self['fpi_inv'] = 'F     ! final phase improvement?'
        self['mag_rel'] = '5'
        self['mag_abs'] = '1e-3'
        self['pha_a1'] = 0
        self['pha_b'] = 0
        self['pha_rel'] = 0
        self['pha_abs'] = 0
        self['hom_bg'] = 'F'
        self['hom_mag'] = '10.00'
        self['hom_pha'] = '0.00'
        self['another_ds'] = 'F'
        self['d2_5'] = '1'
        self['fic_sink'] = 'F'
        self['fic_sink_node'] = '10000'
        self['boundaries'] = 'F'
        self['boundaries_file'] = 'boundary.dat'
        self['mswitch2'] = '1'
        self['lambda'] = 'lambda'

    def set_mswitch(self, key, active):
        """The mswitch can enable/disable various functions of CRTomo, mainly
        concerned with the computation of various resolution parameters.
        The switch itself is implemented as a binary switch, meaning that each
        function can be enabled/disabled by setting its corresponding bit.

        This function simplifies control of these functions by providing a
        simple boolean interface for these options.

        Possible keys: {}

        Parameters
        ----------
        key : str
            Function to control
        active : bool
            Activate (True) or deactivate (False) the feature
        """.format(
            ['{}'.format(x) for x in self.mswitch_values.keys()]
        )

        assert key in self.mswitch_values
        if active:
            self['mswitch'] = (
                self['mswitch'] % self.mswitch_values[key]
            ) + self.mswitch_values[key]
        else:
            self['mswitch'] = (self['mswitch'] % self.mswitch_values[key])

    def write_to_file(self, filename):
        """ Write the configuration to a file. Use the correct order of values.
        """
        if isinstance(filename, io.BytesIO):
            fid = filename
        else:
            fid = open(filename, 'wb')

        for key in self.key_order:
            if (key == -1):
                fid.write(bytes('\n', 'utf-8'))
            else:
                fid.write(
                    bytes(
                        '{0}\n'.format(self[key]),
                        'utf-8',
                    )
                )

        if not isinstance(filename, io.BytesIO):
            fid.close()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        new_copy = crtomo_config()
        # translate the keys
        for key in self.keys():
            new_copy[key] = self[key]
        return new_copy

    def __deepcopy__(self, memo):
        print('deepcopy')
        raise Exception('not implemented')

    def __repr__(self):
        representation = ''
        for key in self.key_order:
            if key == -1:
                representation += '\n'
            else:
                representation += '{0}       !  {1}\n'.format(self[key], key)
        return representation

    def __str__(self):
        return self.__repr__()

    def help(key):
        """Return the help text specific to a certain key
        """
        help_dict = {

        }
        return_text = help_dict.get(key, 'no help available')
        return return_text

    def import_from_file(self, filename):
        """Import a CRTomo configuration from an existing crtomo.cfg file

        Parameters
        ----------
        filename : str
            Path to crtomo.cfg file
        """
        if isinstance(filename, io.BytesIO):
            line_data = filename.readlines()
        else:
            with open(filename, 'r') as fid:
                line_data = fid.readlines()
        lines_raw = [x.strip() for x in line_data]
        key_index = 0
        for line in lines_raw:
            # ignore comments
            if line.startswith('#'):
                continue
            else:
                # check if we have inline comments
                index_inline_comment = line.find('!')
                if index_inline_comment != -1:
                    line = line[0:index_inline_comment].strip()
            key = self.key_order[key_index]
            if key == -1:
                pass
            else:
                self[key] = line
            # increment the key_index
            key_index += 1
