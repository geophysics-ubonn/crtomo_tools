#!/bin/bash

cp extra_nodes.dat.save extra_nodes.dat
test -d grid_w_nodes && rm -r grid_w_nodes
cr_trig_create.py grid_w_nodes

test -e extra_nodes.dat && rm extra_nodes.dat
test -d grid_wo_nodes && rm -r grid_wo_nodes
cr_trig_create.py grid_wo_nodes

montage -geometry 1000x -pointsize 50\
	-label "without extra node" grid_wo_nodes/triangle_grid.png \
	-label "with extra nodes" grid_w_nodes/triangle_grid.png \
	-tile 2x -geometry 1200x fig_extra_nodes.png
