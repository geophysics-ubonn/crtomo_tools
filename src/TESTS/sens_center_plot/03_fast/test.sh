#!/bin/bash

rm *.png

sens_center_plot.py --elem grid/elem.dat --elec grid/elec.dat --config config/config.dat -c
