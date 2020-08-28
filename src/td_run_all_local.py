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
import re
import os
import subprocess
import multiprocessing
from optparse import OptionParser


# determine binary paths
try:
    import crtomo.binaries as cBin
    crmod_binary = cBin.get('CRMod')
    crtomo_binary = cBin.get('CRTomo')
except ImportError:
    print('Couldn\'t find crtomo.binaries - falling back to defaults')
    crmod_binary = 'CRMod'
    crtomo_binary = 'CRTomo'


def handle_cmd_options():
    parser = OptionParser()
    parser.add_option(
        '-t', "--threads",
        dest="number_threads",
        type="int",
        help="number of threads EACH CRMod/CRTomo instance uses. If not " +
        "set, will be determined automatically",
        default=None,
    )

    parser.add_option(
        "-n", "--number",
        dest="number_processes",
        help="How many CRMod/CRTomo instances to start in parallel. " +
        "Default: number of detected CPUs/2",
        type='int',
        default=None,
    )

    parser.add_option(
        "-f", "--filter",
        dest="regex_filter",
        help="Process only tomodirs whose path matches the regex",
        type='str',
        default=None,
    )

    parser.add_option(
        "-r", "--reverse",
        dest="reverse_lists",
        help="Reverse directory lists before working with them",
        action='store_true',
    )

    parser.add_option(
        "-c", "--confirm",
        dest="confirm_start",
        help="Prompt for user input before starting",
        action='store_true',
        default=False,
    )

    (options, args) = parser.parse_args()
    return options


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

    Parameters
    ----------
    tomodir : str
        Tomodir to check

    Returns
    -------
    needs_inversion : bool
        True if not finished yet
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
        dirs.sort()
        if is_tomodir(dirs):
            print('found tomodir: ', root)
            if check_if_needs_modeling(root):
                needs_modeling.append(root)
            if check_if_needs_inversion(root):
                needs_inversion.append(root)

    return sorted(needs_modeling), sorted(needs_inversion)


def _run_crmod_in_tomodir(tomodir):
    pwd = os.getcwd()
    os.chdir(tomodir + os.sep + 'exe')
    print('Calling CRMod in {}'.format(pwd))
    subprocess.check_output(
        crmod_binary, shell=True, stderr=subprocess.STDOUT, )
    os.chdir(pwd)


def _get_mp_settings(options):
    cpu_count = os.cpu_count()
    if options.number_threads is not None:
        os.environ['OMP_NUM_THREADS'] = '{0}'.format(
            options.number_threads
        )
    else:
        os.environ['OMP_NUM_THREADS'] = '{0}'.format(
            int(cpu_count / 2)
        )

    if options.number_processes is not None:
        number_of_concurrent_processes = options.number_processes
    else:
        number_of_concurrent_processes = int(cpu_count / 2)
    return number_of_concurrent_processes


def run_CRMod(tomodirs, options):
    number_of_concurrent_processes = _get_mp_settings(options)
    pool = multiprocessing.Pool(number_of_concurrent_processes)
    pool.map(_run_crmod_in_tomodir, tomodirs)


def _run_crtomo_in_tomodir(tomodir):
    # check once again if the TD is not yet finished
    if not check_if_needs_inversion(tomodir):
        return
    pwd = os.getcwd()
    os.chdir(tomodir + os.sep + 'exe')
    print('Calling CRTomo in {}'.format(pwd))
    subprocess.check_output(
        crtomo_binary, shell=True, stderr=subprocess.STDOUT, )
    os.chdir(pwd)


def run_CRTomo(tomodirs, options):
    number_of_concurrent_processes = _get_mp_settings(options)
    pool = multiprocessing.Pool(number_of_concurrent_processes)
    pool.map(_run_crtomo_in_tomodir, tomodirs)


def main():
    options = handle_cmd_options()
    needs_modeling, needs_inversion = find_unfinished_tomodirs('.')
    print('-' * 20)
    print('modeling:', needs_modeling)
    print('inversion:', needs_inversion)
    print('-' * 20)

    if options.reverse_lists:
        needs_modeling = reversed(needs_modeling)
        needs_inversion = reversed(needs_inversion)

    if options.regex_filter:
        prog = re.compile(options.regex_filter)
        print('Applying filter:', prog)
        needs_inversion_filtered = list(filter(prog.search, needs_inversion))
        needs_modeling_filtered = list(filter(prog.search, needs_modeling))
    else:
        needs_modeling_filtered = needs_modeling
        needs_inversion_filtered = needs_inversion

    if options.confirm_start:
        print('-' * 20)
        print('Tomodirs to model (after filtering):')
        print(needs_modeling_filtered)
        print('-' * 20)
        print('Tomodirs to invert (after filtering):')
        print(needs_inversion_filtered)
        input('Press any key to proceed')
    run_CRMod(needs_modeling_filtered, options)
    run_CRTomo(needs_inversion_filtered, options)


if __name__ == '__main__':
    main()
