#!/usr/bin/python
# -*- coding: utf-8 -*-
from crlab_py.mpl import *
import crlab_py.elem as elem
import numpy as np

if __name__ == '__main__':
    center = np.loadtxt('center.dat')
    colors = np.loadtxt('data.dat')

    elem.load_elem_file('elem.dat')
    elem.load_elec_file('elec.dat')
    nr_elements = len(elem.element_type_list[0])
    elem.element_data = np.zeros((nr_elements, 1)) * np.nan

    elem.plt_opt.title = ''

    elem.plt_opt.cblabel = r'fill'
    elem.plt_opt.reverse = True

    # plot the previously loaded data
    fig = plt.figure()
    ax = fig.add_subplot(111)


    ax,pm,cb = elem.plot_element_data_to_ax(0, ax, scale='linear', no_cb=True)

    cb_pos = mpl_get_cb_bound_next_to_plot(ax)
    ax1 = fig.add_axes(cb_pos, frame_on=True)

    scatter = ax.scatter(center[:,0], center[:,1], c=colors, s=100)
    cmap = mpl.cm.jet_r
    norm = mpl.colors.Normalize(vmin=np.nanmin(colors), vmax=np.nanmax(colors))
    cb1 = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='vertical')
    fig.savefig('sensitivity.png')

    # Some options need to be reset before plotting the second column
    #elem.plt_opt.cblabel = r'fill'  # automatic addition of norm_fac and dyn_fac

    #elem.plot_elements_to_file('sensitivity_2.png', indices[1])
