#!/usr/bin/env python3
'''
Example for processing field data from Sycal
============================================
'''
import reda.exporters.crtomo as crto
import reda.utils.norrec as NR
import reda
import matplotlib.pyplot as plt

###############################################################################
# Daten einlesen und berechnen
# ----------------------------

# reda container erstellen
ert = reda.ERT()
# normal und reciproke Daten einlesen
ert.import_syscal_bin('data_nor.bin')
ert.import_syscal_bin('data_rec.bin', reciprocals=48)
# K-Faktor berechnen
K = reda.utils.geometric_factors.compute_K_analytical(ert.data, spacing=2.5)
# Rho_a berechnen
reda.utils.geometric_factors.apply_K(ert.data, K)
# negative Widerstände bei negativen K umdrehen
reda.utils.fix_sign_with_K.fix_sign_with_K(ert.data)
# calculate diffeence between nor and rec measurement
ert.data = NR.assign_norrec_diffs(ert.data, ('r', 'rho_a'))


###############################################################################
# Histogramme plotten
# -------------------

fig = ert.histogram(('r', 'rdiff', 'rho_a', 'rho_adiff', 'Iab'))
plt.savefig('histogramms.png')

###############################################################################
# Pseudosektionen plotten
# -----------------------

fig = ert.pseudosection('r')
plt.savefig('pseudo_r.png')
fig = ert.pseudosection('rdiff')
plt.savefig('pseudo_rdiff.png')
fig = ert.pseudosection('rho_a')
plt.savefig('pseudo_rhoa.png')
fig = ert.pseudosection('rho_adiff')
plt.savefig('pseudo_rhoadiff.png')
fig = ert.pseudosection('Iab')
plt.savefig('pseudo_I.png')


###############################################################################
# Filtern
# -------

ert.data.query('r > 0', inplace=True) # ert.filter
ert.data.query('r < 1', inplace=True)
ert.data.query('rho_a > 0', inplace=True)
ert.data.query('rho_a < 60', inplace=True)
ert.data.query('rdiff > 0.2', inplace=True)
ert.query('norrec == "nor"', inplace=True)
ert.query('rpha < 10 and rpha > -50', inplace=True)
ert.query('norrec == "nor"', inplace=True)


###############################################################################
# Datenfile in CRTomo Format ausgeben (volt.dat)
# ----------------------------------------------

crto.write_files_to_directory(ert.data, '.')


###############################################################################
# Arbeiten im Tomodir
# -------------------
import crtomo

###############################################################################
# Gitter und Daten einlesen
# -------------------------

td_obj = crtomo.tdMan(
    elem_file='elem.dat',
    elec_file='elec.dat')
td_obj.read_voltages('volt.dat')


###############################################################################
# Inversionseinstellungen
# -----------------------

td_obj.crtomo_cfg['robust_inv'] = 'F'
td_obj.crtomo_cfg['dc_inv'] = 'T'
td_obj.crtomo_cfg['cells_z'] = '-1'
td_obj.crtomo_cfg['mag_rel'] = '10'
td_obj.crtomo_cfg['mag_abs'] = '0.5'
td_obj.crtomo_cfg['fpi_inv'] = 'F'
td_obj.crtomo_cfg['pha_a1'] = '10'
td_obj.crtomo_cfg['pha_b'] = '-1.5'
td_obj.crtomo_cfg['pha_rel'] = '10'
td_obj.crtomo_cfg['pha_abs'] = '0'



###############################################################################
# Inversion
# ----------

td_obj.invert()


###############################################################################
# Tomodir speichern
# -----------------

td_obj.save_to_tomodir('td')