#!/bin/bash

rm *.png

test -d hom && rm -r hom
grid_homogenize.py -d orig/ --dy 20 -o hom

test -d hom_rev && rm -r hom_rev
grid_homogenize.py -d orig_rev/ --dy 20 -o hom_rev
