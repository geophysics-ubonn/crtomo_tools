#!/usr/bin/env python
# *-* coding: utf-8 *-*
"""provide a simple OS agnostic interface to various binary paths, e.g. for
GMSH oder CRTomo.

At the moment all paths are hardcoded, but this interface could be extended
using a configuration file.

At the moment the following binaries have hardcoded default paths:

    * gmsh
    * CRTomo
    * CRMod
    * CutMcK

Examples
--------

>>> import crtomo.binaries as cBin
    gmsh_binary = cBin.get('gmsh')
    print(gmsh_binary)
/usr/bin/gmsh'

"""
import platform
import os
import shutil

# this dictionary contains the hard-coded paths to our binaries for both
# Windows and Linux. Note that we could easily add lambda functions to the
# lists and check for the later in the 'get' function. This would allow us to
# actively search for the binaries, i.e., using shutil.which or some other
# means.
binaries = {
    # binary key
    'gmsh': {
        # platform
        'Linux': [
            # list of possible locations
            'gmsh',
        ],
        'Windows': [
            r'C:\crtomo\bin\gmsh.exe',
            'gmsh.exe',
        ],
    },
    'CRTomo': {
        'Linux': [
            'CRTomo',
            '/usr/bin/CRTomo_dev',
        ],
        'Windows': [
            r'C:\crtomo\bin\crtomo.exe',
        ]
    },
    'CRMod': {
        'Linux': [
            'CRMod',
            'CRMod_dev',
            '/usr/bin/CRMod_dev',
        ],
        'Windows': [
            r'C:\crtomo\bin\crmod.exe',
        ]
    },
    'CutMcK': {
        'Linux': [
            'CutMcK',
            '/usr/bin/CutMcK_dev',
        ],
        'Windows': [
            r'C:\crtomo\bin\cutmck.exe',
        ]
    },

}


def get(binary_name):
    """return a valid path to the given binary. Return an error if no existing
    binary can be found.

    Parameters
    ----------
    binary_name: string
        a binary name used as a key in the 'binaries' dictionary above

    Return
    ------
    string
        full path to binary
    """
    if binary_name not in binaries:
        raise Exception('binary_name: {0} not found'.format(binary_name))

    system = platform.system()
    binary_list = binaries[binary_name][system]

    # check list for a valid entry
    for filename in binary_list:
        valid_file = shutil.which(filename)
        if valid_file:
            return os.path.abspath(valid_file)

