#!/bin/bash

function test_m()
{
	m="${1}"
	outdir="grid_${m}"
	test -d "${outdir}" && rm -r "${outdir}"
	cr_trig_create.py -m "${m}" "${outdir}"
	infile="${outdir}/triangle_grid.png"
	test -e "${infile}" && cp "${infile}" "${outdir}.png"

}

test_m 0
test_m 1
test_m 2
test_m 3
