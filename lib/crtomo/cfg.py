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
        fid = open(filename, 'w')

        for key in self.key_order:
            if(key == -1):
                fid.write('\n')
            else:
                fid.write('{0}\n'.format(self[key]))

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
            -1, -1, -1, 'iseed_var', 'cells_x', 'cells_z',
            'ani_x', 'ani_z', 'max_it', 'dc_inv', 'robust_inv',
            'fpi_inv', 'mag_rel', 'mag_abs', 'pha_a1', 'pha_b',
            'pha_rel', 'pha_abs', 'hom_bg', 'hom_mag', 'hom_pha',
            'another_ds', 'd2_5', 'fic_sink', 'fic_sink_node',
            'boundaries', 'boundaries_file', 'mswitch2', 'lambda',
        )

    def set_defaults(self):
        """Fill the dictionary with all defaults
        """
        self['mswitch'] = 1
        self['elem'] = '../grid/elem.dat'
        self['elec'] = '../grid/elec.dat'
        self['volt'] = '../mod/volt.dat'
        self['inv_dir'] = '../inv'
        self['diff_inv'] = 'F ! difference inversion?'
        self['iseed_var'] = 'iseed variance'
        self['cells_x'] = '0    ! # cells in x-direction'
        self['cells_z'] = '-1    ! # cells in z-direction'
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

    def write_to_file(self, filename):
        """ Write the configuration to a file. Use the correct order of values.
        """
        fid = open(filename, 'w')

        for key in self.key_order:
            if(key == -1):
                fid.write('\n')
            else:
                fid.write('{0}\n'.format(self[key]))

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
