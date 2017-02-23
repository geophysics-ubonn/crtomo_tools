#!/bin/bash

cd 2/
ELEC_SIZE=20.0 grid_plot_wireframe.py --fancy
cd ..
grid_translate_model.py --old 3/ --new 2/ -o 2/rho.dat
plot_magpha.py --elem 2/elem.dat --elec 2/elec.dat -s 2/rho.dat

cd 3/
ELEC_SIZE=20.0 grid_plot_wireframe.py --fancy
plot_magpha.py -s rho.dat
cd ..

cd 6/
ELEC_SIZE=20.0 grid_plot_wireframe.py --fancy
cd ..
grid_translate_model.py --old 3/ --new 6/ -o 6/rho.dat
plot_magpha.py --elem 6/elem.dat --elec 6/elec.dat -s 6/rho.dat


## create montage
montage -geometry 1000x 2/grid.png 3/grid.png 6/grid.png 2/rho.dat_mag.png 3/rho.dat_mag.png 6/rho.dat_mag.png -tile 3x -geometry 1000x grid_translate.png
