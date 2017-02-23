#!/bin/bash
cd data_master
clean_sim.sh
cd exe
CRMod_master_knuet_gfortran
cd ..
cd ..

cd data_dev
clean_sim.sh
cd exe
# CRMod_dev_knuet_gfortran
CRMod_dev
cd ..
cd ..

test -d pot && rm -r pot
test -d mod && rm -r mod
cr_get_analytical_solutions.py -e data_master/grid/elem.dat -t data_master/grid/elec.dat --config data_master/config/config.dat --rho 100 -o output -p -v
head data_master/mod/pot/pot1.dat data_dev/mod/pot/pot1.dat pot/pot1.dat

