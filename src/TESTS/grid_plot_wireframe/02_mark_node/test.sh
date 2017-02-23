#!/bin/bash

grid_plot_wireframe.py -e data_rect/elem.dat -t data_rect/elec.dat -o wire_rect_cutmck.png --mark_node 5
grid_plot_wireframe.py -e data_rect/elem.dat.orig -t data_rect/elec.dat.orig -o wire_rect_nocutmck.png --mark_node 5

