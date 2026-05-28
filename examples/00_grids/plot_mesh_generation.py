#!/usr/bin/env python
# *-* coding: utf-8 *-*
"""
Creating meshes with the mesh_gen object
========================================

"""
###############################################################################
# The top level crtomo import suffices for most tasks
import crtomo
import numpy as np
###############################################################################
# Next, we need to define our electrode positions and boundary geometry
electrodes = np.array((
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0),
))

# the last number indicates the type of boundary element towards the next node
# 12: Neumann no-flow boundaries (for surface boundaries)
# 11: mixed boundary types (subsurface boundaries)
boundaries = np.array((
    (-1, 0, 12),
    (0, 0, 12),
    (1, 0, 12),
    (2, 0, 12),
    (3, 0, 12),
    (4, 0, 11),
    (4, -2, 11),
    (-1, -2, 11),
))
###############################################################################
mesh = crtomo.mesh_gen.gen_mesh(
    boundaries,
    electrodes,
    char_lengths=[
        0.5,
        0.45,
        0.45,
        0.45,
    ],
)
_ = mesh.plot()
