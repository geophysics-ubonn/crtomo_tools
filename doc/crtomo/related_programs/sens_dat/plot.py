#!/usr/bin/python
# -*- coding: utf-8 -*-
import crlab_py.elem as elem
import numpy as np

if __name__ == '__main__':

    elem.load_elem_file('elem.dat')
    elem.load_elec_file('elec.dat')
    indices = elem.load_column_file_to_elements_advanced('sens0112.dat', [2,3], False)
#    elem.plt_opt.title = 'L1 - Coverage'
#    elem.plt_opt.cblabel = r'$\sum \frac{dV_j}{d\rho_i}$'

    elem.plot_elements_to_file('sens_real.png', indices[0], scale='log')
    elem.plot_elements_to_file('sens_imag.png', indices[1], scale='log')
