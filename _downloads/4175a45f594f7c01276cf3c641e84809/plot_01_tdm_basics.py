#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
Using the tdManager to investigate inversion results
====================================================

One of the purposes of the tdManager is to provide an easy interface to analyze
inversion results.
Inversion results can either be directly generated using the
:meth:`crtomo.tdMan.invert` function, or by loading from an already-finished
inversion directory (the tomodir).

In this example we will explore the various functionalities of the tdManager.

"""
###############################################################################
# we need the crtomo module to continue

import crtomo
# the pprint module is used just to prettify the output of this example, please
# ignore for further use
import pprint

###############################################################################
# load inversion results from an already-finished tomodir
tdm = crtomo.tdMan(tomodir='tomodir')

###############################################################################
# The first order of business is to access the convergence behavior of the
# inversion. This data is loaded into a :py:class:`pandas.DataFrame`:
print(tdm.inv_stats)

# TODO Plot inv stats

###############################################################################
# Inversion results are stored as parameter sets in the ParMan instance
# (:py:class:`crtomo.ParMan`):
print(tdm.parman)

###############################################################################
# the ParMan stores multiple parameter sets (i.e., data sets which assign each
# cell of the FE-mesh a given value), identified by an integer index number,
# the parameter id, pid.
# The pids for the various parameters imported from the inversion results are
# stored in dictionary entries:
pprint.pprint(tdm.a['inversion'])

# We can see several entries in the dictionary: The resistivity magnitude
# (rmag), the phase value (rpha), the real part of the complex conductivity
# (cre), the imaginary part of the complex conductivity (cim) and an entry
# which combines cre and cim (cre_cim). For each inversion, the parameter sets
# have an own pid to make the results accessible.

###############################################################################
# To access a given parameter set, extract the pid from the dict and then
# access the data in the ParMan instance. If we would like to access the
# phase values for e.g. the second inversion run, we have to choose the index
# accordingly:
pid_rpha_2nd = tdm.a['inversion']['rpha'][1]
print('pid of the second phase inversion result: ', pid_rpha_2nd)

rpha_second_inversion = tdm.parman.parsets[pid_rpha_2nd]
print(
    'Pixel values of the second phase inverion result: ',
    rpha_second_inversion
)

###############################################################################
# The last inversion result can be accessed using the index -1
rpha_last_inversion = tdm.parman.parsets[tdm.a['inversion']['rmag'][-1]]

###############################################################################
# Most of the time, it is of interest to get the final inversion result. This
# can be done in two seperate ways. Either by settings the index to [-1] or by
# using the built-in retriever function from crtomo-tools:

rmag_final_inversion = tdm.inv_last_rmag_parset()

# The function returns a numpy-array with the inversion results. A similar
# function exists to extract the final results for the phase value (rpha),
# the real part of the complex conductivity (cre) and the imaginary
# part of the complex conductivity (cim):

rpha_final_inversion = tdm.inv_last_rpha_parset()
cre_final_inversion = tdm.inv_last_cre_parset()
cim_final_inversion = tdm.inv_last_cim_parset()

###############################################################################
# TODO: Extract from point, line, polygon, rectangle
