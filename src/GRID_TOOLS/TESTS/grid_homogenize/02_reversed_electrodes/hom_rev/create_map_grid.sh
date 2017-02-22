#!/bin/bash
cr_trig_create.py grid
grid_translate.py -e grid/elem.dat --dx 35.25 --dz 0.0 -o elem_trans1.dat
grid_rotate.py -e elem_trans1.dat -a -13.8688750796 -o elem_trans1_rot1.dat
grid_translate.py -e elem_trans1_rot1.dat --dx 35.25 --dz 149.038071059 -o elem_trans1_rot1_trans2.dat
grid_plot_wireframe.py --fancy -t grid/elec.dat -e elem_trans1_rot1_trans2.dat -o trans1_rot1_trans2.png