#!/bin/bash
cr_get_modelling_errors.py --elem grid1/elem.dat --elec grid1/elec.dat --config grid1/config.dat -o grid1_modelling_error.png
cr_get_modelling_errors.py --elem grid2/elem.dat --elec grid2/elec.dat --config grid2/config.dat -o grid2_modelling_error.png
