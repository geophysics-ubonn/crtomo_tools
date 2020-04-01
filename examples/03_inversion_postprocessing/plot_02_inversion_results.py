#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
Plot inversion results from a tomodir
=====================================

"""
import crtomo
tdm = crtomo.tdMan(tomodir='tomodir')

###############################################################################
# Plot the last magnitude and phase iteration the quick and dirty way.
# Note that all iterations are stored in the tdm.a['inversion'][KEY] list
tdm.show_parset(tdm.a['inversion']['rmag'][-1])
tdm.show_parset(tdm.a['inversion']['rpha'][-1])
###############################################################################
# Let's do this the nice way (for phase values only)
import matplotlib.pylab as plt
# extract parameter set ids
pid_pha = tdm.a['inversion']['rpha'][-1]

fig, ax = plt.subplots(1, 1, figsize=(16 / 2.54, 7 / 2.54))
tdm.plot.plot_elements_to_ax(
    cid=pid_pha,
    ax=ax,
    plot_colorbar=True,
    cmap_name='jet',
    xmin=-0.0,
    xmax=3.5,
    zmin=-1.0,
    cblabel=r'$\phi [mrad]$',
)

###############################################################################
# Plot values can be transformed (converted) on the fly using a
# converter-function, provided using the converter parameter.
# As an example, lets change the sign of the phase values (note that we also
# swap the colormap)
from crtomo.plotManager import converter_change_sign

fig, ax = plt.subplots(1, 1, figsize=(16 / 2.54, 7 / 2.54))
tdm.plot.plot_elements_to_ax(
    cid=tdm.inv_last_rpha_parset(),
    ax=ax,
    cmap_name='jet_r',
    plot_colorbar=True,
    xmin=-0.0,
    xmax=3.5,
    zmin=-1.0,
    cblabel=r'$-\phi [mrad]$',
    converter=converter_change_sign,
)

# Other converters:

# from crtomo.plotManager import converter_pm_log10
# from crtomo.plotManager import converter_log10_to_lin

###############################################################################
# Lets plot real and imaginary parts of the conductivity as log10 values
# Note that we now use the numpy.log10 function as a converter
import numpy as np

fig, axes = plt.subplots(2, 1, figsize=(16 / 2.54, 9 / 2.54))
ax = axes[0]

tdm.plot.plot_elements_to_ax(
    cid=tdm.inv_last_cre_parset(),
    ax=ax,
    cmap_name='viridis',
    plot_colorbar=True,
    xmin=-0.0,
    xmax=3.5,
    zmin=-1.0,
    cblabel=r"$log_{10}(\sigma' [S/m])$",
    converter=np.log10,
)

ax = axes[1]

tdm.plot.plot_elements_to_ax(
    cid=tdm.inv_last_cim_parset(),
    ax=ax,
    cmap_name='viridis',
    plot_colorbar=True,
    xmin=-0.0,
    xmax=3.5,
    zmin=-1.0,
    cblabel=r"$log_{10}(\sigma' [S/m])$",
    converter=np.log10,
)

###############################################################################
# Create a depth cut at x = 4 m, down to 2 m depth
pid_pha = tdm.a['inversion']['rpha'][-1]
results = tdm.parman.extract_along_line(pid_pha, [4, -2], [4, 0])
# x y value
print(results)

import pylab as plt
fig, ax = plt.subplots(figsize=(12 / 2.54, 8 / 2.54))
ax.plot(-results[:, 2], results[:, 1], '.-')
ax.set_xlabel(r'$-\phi [mrad]$')
ax.set_ylabel('depth [m]')

###############################################################################
# Resolution assessment
# Note that you must explicitly tell CRTomo to compute these measures. This can
# be done using the mswitch: (note that the following line only works if you
# actually invert -- here we load from an existing tomodir!) ::
#
#    tdm.crtomo_cfg.set_mswitch('res_m', True)
from crtomo.plotManager import converter_abs_log10
fig, axes = plt.subplots(3, 1, figsize=(20 / 2.54, 13 / 2.54))
ax = axes[0]

tdm.plot.plot_elements_to_ax(
    cid=tdm.a['inversion']['resm'],
    ax=ax,
    cmap_name='viridis',
    plot_colorbar=True,
    xmin=-0.0,
    xmax=3.5,
    zmin=-1.0,
    cblabel=r"$log_{10}(diag(R))$",
    converter=converter_abs_log10,
)

ax = axes[1]

tdm.plot.plot_elements_to_ax(
    cid=tdm.a['inversion']['l1_dw_log10_norm'],
    ax=ax,
    cmap_name='viridis',
    plot_colorbar=True,
    xmin=-0.0,
    xmax=3.5,
    zmin=-1.0,
    cblabel=r"$L1 Coverage$",
    # converter=np.log10,
)

ax = axes[2]

tdm.plot.plot_elements_to_ax(
    cid=tdm.a['inversion']['l2_dw_log10_norm'],
    ax=ax,
    cmap_name='viridis',
    plot_colorbar=True,
    xmin=-0.0,
    xmax=3.5,
    zmin=-1.0,
    cblabel=r"$L2 Coverage$",
    # converter=np.log10,
)
