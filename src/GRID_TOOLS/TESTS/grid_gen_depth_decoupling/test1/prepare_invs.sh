#!/bin/bash

sdir="skeleton_td"


# base
test -d td_base && rm -r td_base
cp -r "${sdir}" td_base
cd td_base/exe
CRMod
CRTomo
cd ..
../plot.sh
cp inv/rho04.mag.png ../result_1_base.png
cp rho/rho.dat_mag.png ../result_0_forward.png
cd ..


test -d td_spike && td_spike
cp -r "${sdir}" td_spike
cd td_spike/exe
cp ../../dec_spike.dat decouplings.dat
CRMod
CRTomo_grid_decoupling_pidwig2
cd ..
../plot.sh
cp inv/rho05.mag.png ../result_2_spike.png
cd ..

test -d td_linear && rm -r td_linear
cp -r "${sdir}" td_linear
cd td_linear/exe
cp ../../dec_linear.dat decouplings.dat
CRMod
CRTomo_grid_decoupling_pidwig2
cd ..
../plot.sh
cp inv/rho04.mag.png ../result_3_linear.png
cd ..


test -d td_orig && rm -r td_orig
cp -r "${sdir}" td_orig
cd td_orig/exe
cp ../../dec_orig.dat decouplings.dat
CRMod
CRTomo_grid_decoupling_pidwig2
cd ..
../plot.sh
cp inv/rho04.mag.png ../result_4_orig.png
cd ..
