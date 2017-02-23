#!/bin/bash

grid_translate_model.py --old 3/ --new 3_offset/ -o 3_offset/rho.dat
plot_magpha.py --elem 3_offset/elem.dat --elec 3_offset/elec.dat -s 3_offset/rho.dat

# plot
cd 3_offset/
ELEC_SIZE=20.0 grid_plot_wireframe.py --fancy
cd ..
cd 3/
ELEC_SIZE=20.0 grid_plot_wireframe.py --fancy
plot_magpha.py -s rho.dat
cd ..

## create montage
montage -geometry 1000x 3/grid.png 3_offset/grid.png 3/rho.dat_mag.png 3_offset/rho.dat_mag.png -tile 2x -geometry 1000x grid_translate.png
