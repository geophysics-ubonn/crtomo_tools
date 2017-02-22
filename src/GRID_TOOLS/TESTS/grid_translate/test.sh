#!/bin/bash

rm -f *.png

grid_plot_wireframe.py --elem orig_rect/elem.dat --elec orig_rect/elec.dat -o "output_rect.png" --fancy
grid_translate.py -e orig_rect/elem.dat -x 10 -o elem_rect_trans.dat
grid_plot_wireframe.py --elem elem_rect_trans.dat --elec orig_rect/elec.dat -o "output_rect_trans.png" --fancy
montage -geometry 1000x output_rect*.png -tile 2x -geometry 2000x summary_rect.png


grid_plot_wireframe.py --elem orig_tri/elem.dat --elec orig_tri/elec.dat -o "output_tri.png" --fancy
grid_translate.py -e orig_tri/elem.dat -x 5 -z -4 -o elem_tri_trans.dat
grid_plot_wireframe.py --elem elem_tri_trans.dat --elec orig_tri/elec.dat -o "output_tri_trans.png" --fancy
montage -geometry 1000x output_tri*.png -tile 2x -geometry 2000x summary_tri.png
