#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
init_sim.py
initialize a new simulation directory structure using `pwd` as the root
directory.
If there are files present in this directory defined in the directory
structure, they will me moved to the corresponding directories
command -s: if this parameter reads "silent", don't print any warnings.
'''
import os
import shutil
from optparse import OptionParser


def handle_cmd_options():
    '''
    Get  the options from the command line.
    '''
    parser = OptionParser()
    parser.add_option("-s", "--silent", action="store_true", dest="silent",
                      help="print any warnings", default=False)
    (options, args) = parser.parse_args()
    return options, args


def move(fname, folder, options):
    """Move file to dir if existing
    """
    if os.path.isfile(fname):
        shutil.move(fname, folder)
    else:
        if options.silent is False:
            print('{0} missing'.format(fname))


def sip_copy(fname, options):
    '''
    If fname missing use default one in ~/.sip
    '''
    if options.silent is False:
        print('{0} missing, using default one stored in ~/.sip'.format(fname))

    if os.path.isfile('~/.sip/{0}'.format(fname)):
        shutil.copy('~/.sip/{0}'.format(fname), 'exe')


def main():
    (options, args) = handle_cmd_options()
    for i in ['config', 'exe', 'grid', 'mod', 'mod/sens', 'mod/pot',
              'inv', 'rho']:
        os.mkdir(i)
    files = ['config.dat', 'elem.dat', 'elec.dat', 'rho.dat', 'volt.dat',
             'crt.noisemod', 'decoupling.dat']
    folders = ['config', 'grid', 'grid', 'rho', 'mod', 'exe', 'exe']

    for i, j in zip(files, folders):
        move(i, j, options)

    if os.path.isfile('crmod.cfg'):
        shutil.move('crmod.cfg', 'exe')
    elif os.path.isfile('config/config.dat'):
        sip_copy('crmod.cfg', options)

    if os.path.isfile('crtomo.cfg'):
        shutil.move('crtomo.cfg', 'exe')
    else:
        sip_copy('crtomo.cfg', options)


if __name__ == '__main__':
    main()
