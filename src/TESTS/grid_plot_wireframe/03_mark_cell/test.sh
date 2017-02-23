#!/bin/bash

grid_plot_wireframe.py -e data_rect/elem.dat -t data_rect/elec.dat -o wire_rect_cutmck.png --mark_cell 5
grid_plot_wireframe.py -e data_rect/elem.dat.orig -t data_rect/elec.dat.orig -o wire_rect_nocutmck.png --mark_cell 5

MARK_WIDTH=0.1 grid_plot_wireframe.py -e data_tri/elem.dat -t data_tri/elec.dat -o wire_tri_cutmck.png --mark_cell 4
MARK_WIDTH=0.6 grid_plot_wireframe.py -e data_tri/elem.dat -t data_tri/elec.dat -o wire_tri_cutmck_thick.png --mark_cell 4
grid_plot_wireframe.py -e data_tri/elem.dat.orig -t data_tri/elec.dat.orig -o wire_tri_nocutmck.png --mark_cell 4
