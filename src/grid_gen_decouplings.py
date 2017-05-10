#!/usr/bin/env python
# *-* coding: utf-8 *-Ü
"""For a regular grid, create decouplings along a horizontal line in the grid.

Two CMD parameters are required:

    1) depth (in elements) at which the line is located. The first line is 1.
    2) grid width in element numbers


Example:

    >>> grid_gen_decouplings 2 5

    Generate a decoupling line in 2m depths, with the grid consisting of 5
    elements in x direction.


Usage
-----

Pipe the output directly to a decouplings.dat file: ::

    echo "40" > decouplings.dat && grid_gen_decouplings.py 5 40 >> \
    decouplings.dat

The first line denotes the number of decouplings.

Chain multiple calls to decouple multiple lines ::

    echo "80" > decouplings.dat && grid_gen_decouplings.py 5 40 >> \
    decouplings.dat && grid_gen_decouplings.py 8 40 >> \
    decouplings.dat

"""
import sys


def main():
    depth = int(sys.argv[1]) - 1
    width = int(sys.argv[2])

    if depth <= 0:
        raise Exception("First parameter must be larger than 1!")

    assert(depth > 0)

    start = depth * width + 1
    for i in range(start, start + width):
        print('{0} {1} 0'.format(i, i + width))
