#!/bin/bash
cr_trig_create.py grid
grid_translate.py -e grid/elem.dat --dx 0.0 --dz 19.5891791065 -o elem_trans1.dat
grid_rotate.py -e elem_trans1.dat -a 23.4265352879 -o elem_trans1_rot1.dat
grid_translate.py -e elem_trans1_rot1.dat --dx 0.0 --dz 163.863099687 -o elem_trans1_rot1_trans2.dat
grid_plot_wireframe.py --fancy -t grid/elec.dat -e elem_trans1_rot1_trans2.dat -o trans1_rot1_trans2.png
