#!/usr/bin/env python
# *-* coding: utf-8 *-*
"""
This is my example script
=========================

This example doesn't do much, it just makes a simple plot
"""

import crtomo
crtomo

grid = crtomo.crt_grid.create_surface_grid(nr_electrodes=10, spacing=1.5)

grid.plot_grid()

