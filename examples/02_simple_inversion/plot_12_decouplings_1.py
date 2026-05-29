#!/usr/bin/env python
r"""
Gegularization decouplings 1
============================

"""
###############################################################################
import crtomo
import numpy as np
from shapely.geometry import Polygon

###############################################################################
# Define the electrode positions and boundaries
# See :ref:`sphx_glr__examples_00_grids_plot_mesh_generation.py` for more
# information.
electrodes = np.array((
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 0),
    (6, 0),
))
boundaries = np.array((
    (-1, 0, 12),
    (0, 0, 12),
    (1, 0, 12),
    (2, 0, 12),
    (3, 0, 12),
    (4, 0, 12),
    (5, 0, 12),
    (6, 0, 12),
    (7, 0, 12),
    (7, -2, 11),
    (-1, -2, 11),
))

###############################################################################
# Additionally, we define one extra line (x0, y0, x1, y1):

extra_lines = [
    [-0, -0.20, 3, -0.45],
]

# Throws error: TODO
# poly1 = Polygon((
#    (0, 0.0),
#    (5, -0.75),
#    (5, -3),
#    (0, -3),
#    #(-2, 0),
# ))

###############################################################################
poly1 = Polygon((
    (0, -0.20),
    (3, -0.45),
    (3, -2),
    (-1, -2),
))

"""
mesh_d = crtomo.mesh_gen.gen_mesh(
    boundaries,
    electrodes,
    # vertical line shoudl not start on horizontal line
    # additional_lines=extra_lines,
    # polygons=[poly1],
    char_lengths=[0.25, 0.5, 0.5, 0.5],
)
"""
###############################################################################
# Generate the mesh
mesh_d, _ = crtomo.mesh_gen.gen_mesh_with_polygons(
    boundaries,
    electrodes,
    # vertical line should not start on horizontal line
    # additional_lines=extra_lines,
    polygons=[poly1],
    char_lengths=[0.15, 0.25, 0.25, 0.25],

)

print(mesh_d)
mesh_d.plot_grid()

###############################################################################
#
decouplings = crtomo.mesh_decouplings.get_decouplings(
    mesh_d, extra_lines
)

###############################################################################
tdm = crtomo.tdMan(grid=mesh_d)
ab = tdm.configs.gen_all_current_dipoles()
tdm.configs.gen_all_voltages_for_injections(ab)
pid_mag, _ = tdm.add_homogeneous_model(300)
tdm.parman.modify_polygon(
    pid_mag,
    poly1,
    30,
)
tdm.plot_forward_models()

###############################################################################
rmag = tdm.measurements(silent=True)[:, 0]
np.random.seed(2048)

# absolute component in [Ohm ]
noise_level_rmag_absolute = 0.01
# relative component [0, 1]
noise_level_rmag_relative = 0.05

noise_rmag = np.random.normal(
    loc=0,
    scale=rmag * noise_level_rmag_relative + noise_level_rmag_absolute
)

rmag_with_noise = rmag + noise_rmag

tdm.register_measurements(rmag_with_noise)

tdm.remove_negative_resistance_measurements()

###############################################################################
tdm.crtomo_cfg['robust_inv'] = 'F'
tdm.add_to_decouplings(decouplings)

tdm.invert(cores=4, catch_output=True)
###############################################################################
_ = tdm.plot_inversion_result_rmag()
