#!/usr/bin/env python
# *-* coding: utf-8 *-*
"""Look for all unfinished tomodirs in the present directory (and
subdirectories), and run them subsequently by calling CRmod/CRTomo. The number
of cores to use can be specified using a command line option.

Note that this script does not allow any kind of parallel calling of CRTomo, or
remote execution of inversions.

This file is self-contained: it does not need any additional python scripts,
and can thus be also used without installing crtomo-tools. The only requirement
is that CRMod/CRTomo is available in the $PATH variable, i.e., that
CRMod/CRTomo can be called.
"""
import os
import subprocess

# determine binary paths
try:
    import crtomo.binaries as cBin
    crmod_binary = cBin.get('CRMod')
    crtomo_binary = cBin.get('CRTomo')
except ImportError as e:
    crmod_binary = 'CRMod'
    crtomo_binary = 'CRTomo'


def is_tomodir(subdirectories):
    """provided with the subdirectories of a given directory, check if this is
    a tomodir
    """
    required = (
        'exe',
        'config',
        'rho',
        'mod',
        'inv'
    )
    is_tomodir = True
    for subdir in required:
        if subdir not in subdirectories:
            is_tomodir = False
    return is_tomodir


def check_if_needs_modeling(tomodir):
    """check of we need to run CRMod in a given tomodir
    """
    print('check for modeling', tomodir)
    required_files = (
        'config' + os.sep + 'config.dat',
        'rho' + os.sep + 'rho.dat',
        'grid' + os.sep + 'elem.dat',
        'grid' + os.sep + 'elec.dat',
        'exe' + os.sep + 'crmod.cfg',
    )

    not_allowed = (
        'mod' + os.sep + 'volt.dat',
    )
    needs_modeling = True
    for filename in not_allowed:
        if os.path.isfile(tomodir + os.sep + filename):
            needs_modeling = False

    for filename in required_files:
        full_file = tomodir + os.sep + filename
        if not os.path.isfile(full_file):
            print('does not exist: ', full_file)
            needs_modeling = False

    return needs_modeling


def check_if_needs_inversion(tomodir):
    """check of we need to run CRTomo in a given tomodir
    """
    required_files = (
        'grid' + os.sep + 'elem.dat',
        'grid' + os.sep + 'elec.dat',
        'exe' + os.sep + 'crtomo.cfg',
    )

    needs_inversion = True

    for filename in required_files:
        if not os.path.isfile(tomodir + os.sep + filename):
            needs_inversion = False

    # check for crmod OR modeling capabilities
    if not os.path.isfile(tomodir + os.sep + 'mod' + os.sep + 'volt.dat'):
        if not check_if_needs_modeling(tomodir):
            print('no volt.dat and no modeling possible')
            needs_inversion = False

    # check if finished
    inv_ctr_file = tomodir + os.sep + 'inv' + os.sep + 'inv.ctr'
    if os.path.isfile(inv_ctr_file):
        inv_lines = open(inv_ctr_file, 'r').readlines()
        print('inv_lines', inv_lines[-1])
        if inv_lines[-1].startswith('***finished***'):
            needs_inversion = False

    return needs_inversion


def find_unfinished_tomodirs(directory):
    needs_modeling = []
    needs_inversion = []
    for root, dirs, files in os.walk(directory):
        if is_tomodir(dirs):
            print('found tomodir: ', root)
            if check_if_needs_modeling(root):
                needs_modeling.append(root)
            if check_if_needs_inversion(root):
                needs_inversion.append(root)

    return needs_modeling, needs_inversion


def run_CRMod(tomodirs):
    pwd = os.getcwd()
    for tomodir in tomodirs:
        os.chdir(tomodir + os.sep + 'exe')
        subprocess.call(crmod_binary, shell=True)
        os.chdir(pwd)


def run_CRTomo(tomodirs):
    pwd = os.getcwd()
    for tomodir in tomodirs:
        os.chdir(tomodir + os.sep + 'exe')
        subprocess.call(crtomo_binary, shell=True)
        os.chdir(pwd)


def main():
    needs_modeling, needs_inversion = find_unfinished_tomodirs('.')
    print('modeling:', needs_modeling)
    print('inversion:', needs_inversion)

    run_CRMod(needs_modeling)
    run_CRTomo(needs_inversion)


if __name__ == '__main__':
    main()
