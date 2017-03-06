#!/bin/bash

rm *.png
test -d grid_hom && rm -r grid_hom

grid_homogenize -d data --dx 50 --dy 100 -o grid_hom
cp char_length.dat grid_hom/
cd grid_hom
# cr_trig_create.py grid
# mkdir t1
# cp grid/elem.dat t1/
# cp grid/elec.dat t1/
