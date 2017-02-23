#!/bin/bash
odir="16elecs"
ndir="24elecs"

cd "${odir}"/
ELEC_SIZE=20.0 grid_plot_wireframe.py
cd ..
cd "${ndir}"
ELEC_SIZE=20.0 grid_plot_wireframe.py
cd ..

grid_translate_model.py --old "${odir}" --new "${ndir}" -o "${ndir}"/rho.dat
plot_magpha.py --elem "${ndir}"/elem.dat --elec "${ndir}"/elec.dat -s "${ndir}"/rho.dat
plot_magpha.py --elem "${odir}"/elem.dat --elec "${odir}"/elec.dat -s "${odir}"/rho.dat

## create montage
montage -geometry 1000x "${odir}"/grid.png "${ndir}"/grid.png "${odir}"/rho.dat_mag.png "${odir}"/rho.dat_mag.png  -tile 2x -geometry 1000x grid_translate.png
