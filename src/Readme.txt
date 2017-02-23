command line program to create borehole irregular grids
=======================================================

TODO
----

Check and sort boundary polygon:

http://stackoverflow.com/questions/13935324/sorting-clockwise-polygon-points-in-matlab

http://stackoverflow.com/questions/1165647/how-to-determine-if-a-list-of-polygon-points-are-in-clockwise-order


Usage
=====

Create the files described down below (electrodes.dat, boundaries.dat, (opt)
char_length.dat, (opt) gmsh_commands.dat.

Then run

    ::
        cr_trig_create.py

The program will create a subdirectory with a unique name and store the final
grid files (and all intermediate files) in there.


Files
=====

electrodes.dat
--------------

(X,Y) coordinates of all electrodes. Each electrode in one line:

::

    24.0000 -34.0000
    24.0000 -35.0000
    24.0000 -36.0000

boundaries.dat
--------------

(X,Y) coordinates of the nodes comprising the boundaries of the grid. If
electrodes lie on the surface, also specify them in the boundaries.dat file.
Doublets will be automatically removed. Specify the boundary elements clockwise!

TODO: Implement sorting routines.

The third column denotes the boundary element type of the boundary described by
the node and the next node. The last node denotes the boundary type of the
element from the last to the first node.

::

    -1.0000 1.7700  12
    1.0000 1.7100   12
    3.0000 1.5100   12
    5.0000 1.4300   12

char_length.dat (optional)
--------------------------

If this file exists, only one floating point number is expected: The
characteristic length which will be assigned to the electrodes.

Default value is 1

Larger values increase element size (and decrease number of elements, and
drecrease homogeneity of the grid)

gmsh_commands.dat (optional)
----------------------------

If this file exists, the content will be appended to the commands.geo file used
to create the grid.
