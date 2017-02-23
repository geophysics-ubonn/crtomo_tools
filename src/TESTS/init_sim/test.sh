#!/bin/bash
# init_sim.sh
outdir="sh_version"
test -d "${outdir}" && rm -r "${outdir}"
mkdir "${outdir}"
cd "${outdir}"
init_sim.sh
rm -r config exe grid mod/sens mod/pot mod inv rho
cd ..
rm -r "${outdir}"


#Check if init_sim.py works
outdir="py_version"
test -d "${outdir}" && rm -r "${outdir}"
mkdir "${outdir}"
cd "${outdir}"
init_sim.py
rm -r config exe grid mod/sens mod/pot mod inv rho
cd ..
rm -r "${outdir}"
