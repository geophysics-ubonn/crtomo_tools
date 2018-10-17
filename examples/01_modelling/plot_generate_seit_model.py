#!/usr/bin/env python
# *-* coding: utf-8 *-*
"""
Generating sEIT forward models
==============================

The eit manager can be used to easily create forward models using different
parameterizations.
"""
###############################################################################
# imports
import numpy as np
import crtomo

###############################################################################
# we need a FE grid
grid = crtomo.crt_grid.create_surface_grid(nr_electrodes=15, spacing=1)
grid.plot_grid()

###############################################################################
# define frequencies
frequencies = np.logspace(-3, 3, 10)

# create the eit manager
eitman = crtomo.eitMan(frequencies=frequencies, grid=grid)

###############################################################################
# start with a homogeneous complex resistivity distribution
eitman.add_homogeneous_model(magnitude=100, phase=0)

r = eitman.plot_forward_models(maglim=[90, 110])
print(r)

# save to files
r['rmag']['fig'].savefig('fwd_model_hom_rmag.png', dpi=300)
r['rpha']['fig'].savefig('fwd_model_hom_rpha.png', dpi=300)

###############################################################################
# now we can start parameterizing the subsurface
eitman.set_area_to_single_colecole(
    0, 5, -2, 0,
    [100, 0.1, 0.04, 0.8]
)

r = eitman.plot_forward_models(maglim=[90, 110], phalim=[-30, 0])

# save to files
r['rmag']['fig'].savefig('fwd_model_par_rmag.png', dpi=300)
r['rpha']['fig'].savefig('fwd_model_par_rpha.png', dpi=300)

###############################################################################
# add configurations
configs = np.array((
    (1, 3, 5, 4),
    (5, 7, 10, 8),
))
eitman.add_to_configs(configs)

###############################################################################
# conduct forward modeling
eitman.model()
measurements = eitman.measurements()

###############################################################################
# modeled SIP signatures can be retrieved as a dict:
sip_sigs = eitman.get_measurement_responses()
print(sip_sigs)

###############################################################################
# plot modeled SIP signatures

for key, obj in eitman.get_measurement_responses().items():
        obj.plot(filename='mod_sip_{}.png'.format(key), dtype='r')

###############################################################################
# Extract SIP signature at one point from the forward model
sip_one_p = eitman.extract_points(
    ['forward_rmag', 'forward_rpha'],
    np.atleast_2d(np.array((1, -1)))
)

# import IPython
# IPython.embed()
