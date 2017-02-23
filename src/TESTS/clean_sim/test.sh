#!/bin/bash

# test if clean_sim work on a specfic tomodir
function test_td() {
	pwdx="${PWD}"
	dir="${1}"
	cd ${dir}
	echo "---------------------------------------------------------"
	clean_sim.py
	echo "after"
	tree
	echo "---------------------------------------------------------"
	cd "${pwdx}"
}

test -d tomodir_cmplx && rm -r tomodir_cmplx
test -d tomodir_cmplx || tar xvjf ../../../../sample_data/Inversions/tomodir_cmplx.tar.bz2
test_td tomodir_cmplx

base="tomodir_fpi"
test -d "${base}" && rm -r "${base}"
tar xvjf ../../../../sample_data/Inversions/"${base}".tar.bz2
test_td "${base}"

base="tomodir_dc"
test -d "${base}" && rm -r "${base}"
tar xvjf ../../../../sample_data/Inversions/"${base}".tar.bz2
test_td "${base}"
