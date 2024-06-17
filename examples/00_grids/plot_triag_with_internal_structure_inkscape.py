#!/usr/bin/env python
# *-* coding: utf-8 *-*
"""
Using Inkscape to define internal structure
===========================================

It is possible to use Inkscape to define some of the internal structure of a FE
mesh. This can be used to:

    * define irregularly-shaped regions within a forward mesh
    * define internal geometry that is later used for decoupling of the
      regularization

More information: https://geophysics-ubonn.github.io/crtomo_tools/grid_creation.html#introducing-structures-into-a-mesh-using-inkscape


The basic procedure is as follows:

    * Create input files for triangular mesh generation as usual:
      **boundaries.dat**, **electrodes.dat**
    * Following that, generate a .svg template of the mesh boundary by running
      *grid_convert_boundary_to_svg*

"""
###############################################################################
import os
import shutil

import reda
import subprocess
import numpy as np
import crtomo
from shapely.geometry import Polygon
import matplotlib.pylab as plt
###############################################################################
if os.path.isdir('tmp_triag_inkscape'):
    shutil.rmtree('tmp_triag_inkscape')

###############################################################################
with reda.CreateEnterDirectory('tmp_triag_inkscape'):
    with open('electrodes.dat', 'w') as fid:
        fid.write("""0.0 0.0
5.0 0.0
10.0 0.0
15.0 0.0
""")

    with open('boundaries.dat', 'w') as fid:
        fid.write("""-10.0000 0.0000 12
0.0000 0.0000 12
5.0000 0.0000 12
10.0000 0.0000 12
15.0000 0.0000 12
25.0000 0.0000 11
25.0000 -10.0000 11
-10.0000 -10.0000 11""")


with reda.CreateEnterDirectory('tmp_triag_inkscape'):
    subprocess.call('grid_convert_boundary_to_svg', shell=True)

with reda.CreateEnterDirectory('tmp_triag_inkscape'):
    shutil.copy('../data_02/out_modified2.svg', 'out_modified2.svg')

with reda.CreateEnterDirectory('tmp_triag_inkscape'):
    subprocess.call('grid_parse_svg_to_files', shell=True)

with reda.CreateEnterDirectory('tmp_triag_inkscape'):
    shutil.copy('constraint_1.dat', 'extra_lines.dat')

with reda.CreateEnterDirectory('tmp_triag_inkscape'):
    subprocess.call('cr_trig_create grid', shell=True)

###############################################################################
with reda.CreateEnterDirectory('tmp_triag_inkscape'):
    grid = crtomo.crt_grid(
        elem_file='grid/elem.dat',
        elec_file='grid/elec.dat',
    )
    pm = crtomo.ParMan(grid)
    pid = pm.add_empty_dataset(100)
    region_lines = np.loadtxt('mdl_constraint_1.dat')

    poly = Polygon(
        np.vstack(
            (region_lines[:, 0:2], region_lines[-1, 2:4])
        )
    )
    pm.modify_polygon(pid, poly, 10)

    plotman = crtomo.pltMan(pm=pm, grid=grid)

    fig, ax = plt.subplots()
    plotman.plot_elements_to_ax(
        pid,
        plot_colorbar=True,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig('model.jpg', dpi=300)

with reda.CreateEnterDirectory('tmp_triag_inkscape'):
    subprocess.call(
        'grid_extralines_gen_decouplings -e grid/elem.dat '
        '-l constraint_1.dat --debug_plot --eta 0.7',
        shell=True
    )
