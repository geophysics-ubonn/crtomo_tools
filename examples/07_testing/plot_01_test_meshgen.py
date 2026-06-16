#!/usr/bin/env python
# coding: utf-8
"""
CRTomo-Meshgen Tests - Extra lines
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Various tests for adding extra lines to generated meshes
"""
import numpy as np
import crtomo

# ###########################################################################


def test_mesh(mesh):
    """Conduct a simple forward modelling
    """
    tdm = crtomo.tdMan(grid=mesh)
    tdm.add_homogeneous_model(100, 0)
    tdm.configs.gen_dipole_dipole(skipc=0)
    tdm.model(silent=True)
    return True


# ###########################################################################

electrodes = np.array((
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
))

boundaries = np.array((
    (-1, 0, 12),
    (0, 0, 12),
    (1, 0, 12),
    (2, 0, 12),
    (3, 0, 12),
    (4, 0, 11),
    (4, -4, 11),
    (-1, -4, 11),
))
# ###########################################################################

mesh = crtomo.mesh_gen.gen_mesh(
    boundaries,
    electrodes,
    char_lengths=[0.5, 0.5, 0.5, 0.5],
)
test_mesh(mesh)

# ###########################################################################

mesh = crtomo.mesh_gen.gen_mesh(
    boundaries,
    electrodes,
)
test_mesh(mesh)

# ###########################################################################


# test for over-extending lines
mesh = crtomo.mesh_gen.gen_mesh(
    boundaries,
    electrodes,
    extra_lines=[
        # horizontal
        [-5, -1, 5, 1],
    ]
)
test_mesh(mesh)


# ###########################################################################

# extra line through electrode
mesh = crtomo.mesh_gen.gen_mesh(
    boundaries,
    electrodes,
    extra_lines=[
        [0, -5, 0, 1],
    ]
)
test_mesh(mesh)


# ###########################################################################


# extra lines that cross
mesh = crtomo.mesh_gen.gen_mesh(
    boundaries,
    electrodes,
    extra_lines=[
        [-5, -1, 5, -1],
        [0, -5, 0, 1],
    ]
)
test_mesh(mesh)


# ###########################################################################


# extra lines that cross (multiple times)
mesh = crtomo.mesh_gen.gen_mesh(
    boundaries,
    electrodes,
    extra_lines=[
        [-5, -1, 5, -1],
        [-5, -2, 5, -2],
        [0, -5, 0, 1],
    ]
)
test_mesh(mesh)

# ###########################################################################


# extra lines that cross (multiple times)
mesh = crtomo.mesh_gen.gen_mesh(
    boundaries,
    electrodes,
    extra_lines=[
        # three horizontal lines
        [-5, -1, 5, -1],
        [-5, -2, 5, -2],
        [-5, -3, 5, -3],
        # one vertical line
        [0, -5, 0, 1],
    ]
)
test_mesh(mesh)


# parallel extra lines, one of which starts within the other
mesh = crtomo.mesh_gen.gen_mesh(
    boundaries,
    electrodes,
    extra_lines=[
        [-5, -1, 1, -1],
        [0, -1, 5, -1],
    ]
)
test_mesh(mesh)

# lines ending/starting at some point
mesh = crtomo.mesh_gen.gen_mesh(
    boundaries,
    electrodes,
    extra_lines=[
        [-5, -1, 0, -1],
        [0, -1, 5, -1],
    ]
)
test_mesh(mesh)
