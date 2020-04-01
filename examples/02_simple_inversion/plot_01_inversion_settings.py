#!/usr/bin/env python
# *-* coding: utf-8 *-*
"""

.. _example_crmod_crtomo_cfg:

Setting modeling and inversion settings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example discusses details of settings various settings for modeling or
inversion using the single-frequency tdManager
"""
###################################################################
# we create an empty tdMan instance, which lacks a model and measurement
# configurations in order to run a forward modeling and subsequent inversion.
# However, we still can set inversion settings
import crtomo

grid = crtomo.crt_grid.create_surface_grid(
    nr_electrodes=20, spacing=1, char_lengths=[0.3, 1, 1, 1]
)

# create the tdManager instance used for the inversion
tdm = crtomo.tdMan(grid=grid)

###############################################################################
# Forward modeling settings are stored in tdm.crmod_cfg. This object is an
# extended dict, containing the settings for each modeling setting.
# Documentation :py:class:`crtomo.cfg.crmod_config`
# This class wraps the file-based CRMod configuration file crmod.cfg, which is
# described here: :ref:`description_crmod_cfg`
print('Type:', type(tdm.crmod_cfg))

###############################################################################
# Access the settings just as with a normal dict:
print(tdm.crmod_cfg.keys())

###############################################################################
# Note that printing it will print all keys and current settings
print(tdm.crmod_cfg)

###############################################################################
# Examples for setting parameters
tdm.crmod_cfg['write_pots'] = True
tdm.crmod_cfg['2D'] = True

###############################################################################
# CRTomo (inversion) settings are stored in tdm.crtomo_cfg
# Documentation :py:class:`crtomo.cfg.crtomo_config`
print('Type:', type(tdm.crtomo_cfg))

###############################################################################
# Access the settings just as with a normal dict:
print(tdm.crtomo_cfg.keys())

###############################################################################
# Note that printing it will print all keys and current settings
print(tdm.crtomo_cfg)

###############################################################################
# Settings of the mswitch can be set in a simple way
tdm.crtomo_cfg.set_mswitch('res_m', True)
