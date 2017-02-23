"""Representations of CRMod and CRTomo configurations.

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
            'write_pot',
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

    def set_defaults(self):
        """
        Fill the dictionary with all defaults
        """
        self['mswitch'] = '***FILES***'
        self['elem'] = '../grid/elem.dat'
        self['elec'] = '../grid/elec.dat'
        self['rho'] = '../rho/rho.dat'
        self['config'] = '../config/config.dat'
        self['write_pot'] = 'F'  # ! potentials ?
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
        fid = open(filename, 'w')

        for key in self.key_order:
            if(key == -1):
                fid.write('\n')
            else:
                fid.write('{0}\n'.format(self[key]))

        fid.close()

    def __repr__(self):
        representation = ''
        for key in self.key_order:
            if key == -1:
                representation += '\n'
            else:
                representation += '{0}       !  {1}\n'.format(self[key], key)
        return representation


class crtomo_cfg(dict):
    """
    Write CRTomo configuration files (crtomo.cfg).

    This class is essentially a dict of CRTomo configurations with a few extra
    functions that know how to write a proper crtomo.cfg file.
    """
    def __init__(self, *arg, **kw):
        super(crtomo_cfg, self).__init__(*arg, **kw)
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
        self['cells_z'] = '0    ! # cells in z-direction'
        self['ani_x'] = '1.000  ! smoothing parameter in x-direction'
        self['ani_z'] = '1.000  ! smoothing parameter in z-direction'
        self['max_it'] = '20    ! max. nr of iterations'
        self['dc_inv'] = 'F     ! DC inversion?'
        self['robust_inv'] = 'F     ! robust inversion?'
        self['fpi_inv'] = 'T     ! final phase improvement?'
        self['mag_rel'] = '5'
        self['mag_abs'] = '1e-3'
        self['pha_a1'] = 0
        self['pha_b'] = 0
        self['pha_rel'] = 0
        self['pha_abs'] = 0.5
        self['hom_bg'] = 'F'
        self['hom_mag'] = '10.00'
        self['hom_pha'] = '0.00'
        self['another_ds'] = 'F'
        self['d2_5'] = '0'
        self['fic_sink'] = 'T'
        self['fic_sink_node'] = '6467'
        self['boundaries'] = 'F'
        self['boundaries_file'] = 'boundary.dat'
        self['mswitch2'] = '1'
        self['lambda'] = 'lambda'

    def write_to_file(self, filename):
        """
        Write the configuration to a file. Use the correct order of values.
        """
        fid = open(filename, 'w')

        for key in self.key_order:
            if(key == -1):
                fid.write('\n')
            else:
                fid.write('{0}\n'.format(self[key]))

        fid.close()

    def __repr__(self):
        representation = ''
        for key in self.key_order:
            if key == -1:
                representation += '\n'
            else:
                representation += '{0}       !  {1}\n'.format(self[key], key)
        return representation
