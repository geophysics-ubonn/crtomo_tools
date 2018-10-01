#!/usr/bin/env python
# *-* coding: utf-8 *-*
"""
Creating and handling grids
===========================

"""
###############################################################################
# The toplevel crtomo import suffices for most tasks
import crtomo

###############################################################################
# Create simple surface grids with this wrapper
grid = crtomo.crt_grid.create_surface_grid(nr_electrodes=10, spacing=1.5)
grid.plot_grid()
# number the electrodes (useful for numerical studies)
grid.plot_grid(plot_electrode_numbers=True)

# save this grid to disc
grid.save_elem_file('elem.dat')
grid.save_elec_file('elec.dat')


###############################################################################
# Grid can be read from disk:
grid1 = crtomo.crt_grid('elem.dat', 'elec.dat')
print(grid1)
