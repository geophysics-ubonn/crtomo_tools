#!/bin/bash
grid_gen_depth_decoupling.py \
   	--elem grid/elem.dat.orig \
   	--decfile zdec.dat \
	--output dec_spike.dat

grid_gen_depth_decoupling.py \
   	--elem grid/elem.dat.orig \
   	--decfile zdec_linear.dat \
	--output dec_linear.dat

grid_gen_depth_decoupling.py \
   	--elem grid/elem.dat.orig \
   	--decfile zdec_orig.dat \
	--output dec_orig.dat
