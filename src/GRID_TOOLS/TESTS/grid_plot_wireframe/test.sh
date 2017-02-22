#!/bin/bash

grid_plot_wireframe.py -e orig_rect/elem.dat -t orig_rect/elec.dat -o wire_rect.png
grid_plot_wireframe.py -e orig_rect/elem.dat.orig -t orig_rect/elec.dat.orig -o wire_rect_orig.png
grid_plot_wireframe.py -e orig_tri/elem.dat -t orig_tri/elec.dat -o wire_tri.png

grid_plot_wireframe.py -e orig_rect/elem.dat -t orig_rect/elec.dat --fancy -o wire_rect_fancy.png
grid_plot_wireframe.py -e orig_rect/elem.dat.orig -t orig_rect/elec.dat.orig --fancy -o wire_rect_orig_fancy.png
grid_plot_wireframe.py -e orig_tri/elem.dat -t orig_tri/elec.dat --fancy -o wire_tri_fancy.png
