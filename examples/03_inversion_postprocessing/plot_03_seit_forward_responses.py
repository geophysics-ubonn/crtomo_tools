#!/usr/bin/env python3
# *-* coding: utf-8 *-*
"""
Analyzing data response of inversion result
===========================================

"""
import crtomo
import reda

seit = crtomo.eitMan(seitdir='seitdir', shalllow_import=False)
with reda.CreateEnterDirectory('output_03'):
    seit.plot_result_spectrum('spectrum_01.jpg', 0)
