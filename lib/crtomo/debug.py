"""This file contains certain functionality used for testing purposes.

One example is the interface to return grid objects based on CRTomo grids that
are distributed with crtomo_tools.
"""

import pkg_resources as pk

import crtomo.grid as CRGrid

# if pk.resource_exists(module, rfile):
# sys.stdout.write(pk.resource_filename(module, rfile))


grid_files = {
    20: {
        'elem': 'debug_data/elem_20elecs.dat',
        'elec': 'debug_data/elec_20elecs.dat',
    },
    40: {
        'elem': 'debug_data/elem_40elecs.dat',
        'elec': 'debug_data/elec_40elecs.dat',
    },
}


def get_grid(key):
    """Return a :class:`crtomo.grid.crt_grid` instance, with a debug grid that
    is distributed with the crtomo package. Multiple grids are available:

        * key=20 - a 20 electrode grid with 1m spacing, 4 elements between
          electrodes, rectangular elements.
        * key=40 - a 40 electrode grid with 1m spacing, 4 elements between
          electrodes, rectangular elements.

    Parameters
    ----------
    key: string
        key that identifies the grid


    Returns
    -------
    grid: :class:`crtomo.grid.crt_grid` instance
        loaded grid object

    """
    rbase = grid_files[key]
    elem_file = pk.resource_filename('crtomo', rbase['elem'])
    elec_file = pk.resource_filename('crtomo', rbase['elec'])

    grid = CRGrid.crt_grid(elem_file=elem_file, elec_file=elec_file)
    return grid
