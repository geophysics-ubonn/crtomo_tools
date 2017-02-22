#!/bin/bash

grid_plot_wireframe.py --fancy --elem orig_rect/elem.dat --elec orig_rect/elec.dat -o output_rect.png
grid_rotate.py -e orig_rect/elem.dat -a 45 -o elem_rect_rot.dat
grid_plot_wireframe.py --fancy --elem elem_rect_rot.dat --elec orig_rect/elec.dat -o output_rect_rot.png
montage -geometry 1000x output_rect*.png -tile 2x -geometry 1000x summary_rect.png

grid_plot_wireframe.py --fancy --elem orig_tri/elem.dat --elec orig_tri/elec.dat -o output_tri.png
grid_rotate.py -e orig_tri/elem.dat -a 45 -o elem_tri_rot.dat
grid_plot_wireframe.py --fancy --elem elem_tri_rot.dat --elec orig_tri/elec.dat -o output_tri_rot.png
montage -geometry 1000x output_tri*.png -tile 2x -geometry 1000x summary_tri.png
