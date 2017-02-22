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

# test_m 0
# test_m 1
# test_m 2
# test_m 3

montage -geometry 1500x \
	-pointsize 72 -label "-m 0" grid_0.png\
   	-label "-m 1" grid_1.png\
   	-label "-m 2" grid_2.png\
   	-label "-m 3" grid_3.png\
   	-tile 2x -geometry 1200x additional_nodes.png
