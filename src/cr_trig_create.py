#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create irregular (triangular) grids for CRMod/CRTomo from simple ascii input
files.

Create the input files "electrodes.dat", "boundaries.dat", "char_length.dat"
according to the CRLab manual. An optional file "gmsh_commands.dat" file can be
used to append GMSH commands at the end of the generated gmsh command file.
Note that these command will be executed only after the initial meshing, but
can be used to refine the mesh.

Then run cr_trig_create.py.

If an (optional) parameter is provided, use this as the output directory. This
directory must not exist. In case no parameter is given, create a random
directory using the uuid module.

The output directory is printed in the last line of program output.

Examples:
---------

>>> # note that this is a shell example, not Python!
    cat electrodes.dat
1.0 0.0
2.0 0.0
>>> cat boundaries.dat
0.0 0.0 12
1.0 0.0 12
2.0 0.0 11
2.0 -2.0 11
0.0 -2.0 11
>>> cat char_length.dat
0.5
>>> cat extra_lines.dat
0.5 -0.5 1.5 -1.5
>>> cr_trig_create.py


TODO
----

* this file needs to be restructured. The complexity has outgrown the initial
  structure.
* investigate the ' attractor field' to set characteristic lengths (see notes
  below) http://gmsh.info/doc/texinfo/gmsh.html
"""
import numpy as np
import uuid
import shutil
import os
import subprocess
from optparse import OptionParser

import crtomo.binaries as cBin


def handle_cmd_options():
    parser = OptionParser()
    parser.add_option("-m", "--add_stabilizer_nodes", dest="add_stab_nodes",
                      help="(for surface electrodes only!) How many nodes " +
                      "to put between electrodes", metavar="NR",
                      default=None,
                      type=int)

    (options, args) = parser.parse_args()

    return options, args


class Mesh():
    """
    GMSH distinguishes three types of objects:

    * Points on the boundary (i.e., points that create the boundary)
    * Lines that connect Points to form the actual boundaries
    * Points in Surface, nodes (or points) that lie IN the grid and need to
      be connected to the grid

    Note
    ----

    TODO: I think we don't need to add electrodes as POINTS, only as POINT IN
    SURFACE, because POINTs that do not belong to a line will not be included
    in the final mesh, or? Check.
    Possible Answer: We need to define the Points before we can add them with
    the IN SURFACe command.

    Check for duplicate entries in input files

    Can we test if all electrodes and extra-nodes lie in the boundaries?

    Sort boundaries

    Add extra-nodes (for inner-grid-structure)

    Add extra-lines (for inner-grid-structure)

    """

    def __init__(self):
        self.Points = []
        self.Charlengths = []
        self.Lines = []
        self.Electrodes = []
        self.Boundaries = []
        self.BoundaryIndices = []
        self.ExtraNodes = []
        self.ExtraLineIndices = []

    def get_point_id(self, p, char_length):
        """
        Return the id of the given point (x,y) tuple.
        """
        print('Checking point', p)
        # TODO: This search loop NEEDS to be replaced with something sane
        index = -1
        for nr, i in enumerate(self.Points):
            if(np.all(i == p)):
                print('Point already in list at index {0}'.format(nr))
                if self.Charlengths[nr] > char_length:
                    print('Updating characteristic length')
                    self.Charlengths[nr] = char_length
                return nr

        if(index == -1):
            print('adding point:', p)
            self.Points.append(p)
            self.Charlengths.append(char_length)
            return len(self.Points) - 1

    def add_boundary(self, p1, p2, btype):
        """
        Add a boundary line
        """
        index = self.add_line(p1, p2, self.char_lengths['boundary'])
        # self.Boundaries.append((p1_id,p2_id,btype))
        self.BoundaryIndices.append(index)
        self.Boundaries.append((p1, p2, btype))

    def add_line(self, p1, p2, char_length):
        """
        Add a line to the list. Check if the nodes already exist, and add them
        if not.

        Return the line index (1-indixed, starting with 1)
        """
        p1_id = self.get_point_id(p1, char_length)
        p2_id = self.get_point_id(p2, char_length)
        self.Lines.append((p1_id, p2_id))
        return len(self.Lines)

    def is_in(self, search_list, pair):
        """
        If pair is in search_list, return the index. Otherwise return -1
        """
        index = -1
        for nr, i in enumerate(search_list):
            if(np.all(i == pair)):
                return nr
        return index

    def read_electrodes(self, electrodes):
        """
        Read in electrodes, check if points already exist
        """
        for nr, electrode in enumerate(electrodes):
            index = self.get_point_id(
                electrode, self.char_lengths['electrode'])
            self.Electrodes.append(index)

    def read_extra_nodes(self, filename):
        """Read extra nodes in. Format: x y

        What happens if we add nodes on the boundaries, which are not included
        in the boundaries?
        """
        data = np.atleast_2d(np.loadtxt(filename))
        for nr, pair in enumerate(data):
            index = self.get_point_id(pair, self.char_lengths['extra_node'])
            self.ExtraNodes.append(index)

    def read_extra_lines(self, filename):
        """Read extra lines from the given filename. Each line is defined in
        one line with four coordinates: x1 y1 x2 y2. (x1,y1) denotes the
        starting point, (x2, y2) the end point of the line.
        """
        data = np.atleast_2d(np.loadtxt(filename))
        for nr, coords in enumerate(data):
            p1 = [coords[0], coords[1]]
            p2 = [coords[2], coords[3]]
            index = self.add_line(p1, p2, self.char_lengths['extra_line'])
            self.ExtraLineIndices.append(index)

    def write_electrodes(self, filename):
        """
        Write X Y coordinates of electrodes
        """
        fid = open(filename, 'w')
        for i in self.Electrodes:
            fid.write('{0} {1}\n'.format(self.Points[i][0], self.Points[i][1]))
        fid.close()

    def write_boundaries(self, filename):
        """
        Write boundary lines X1 Y1 X2 Y2 TYPE to file
        """
        fid = open(filename, 'w')
        for i in self.Boundaries:
            print(i)
            # fid.write('{0} {1} {2}\n'.format(i[0], i[1], i[2]))
            fid.write(
                '{0} {1} {2} {3} {4}\n'.format(
                    i[0][0], i[0][1], i[1][0], i[1][1], i[2]))
        fid.close()

    def read_char_lengths(self, filename, electrode_filename):
        """Read characteristic lengths from the given file.

        The file is expected to have either 1 or 4 entries/lines with
        characteristic lengths > 0 (floats). If only one value is encountered,
        it is used for all four entities. If four values are encountered, they
        are assigned, in order, to:

            1) electrode nodes
            2) boundary nodes
            3) nodes from extra lines
            4) nodes from extra nodes

        Note that in case one node belongs to multiple entities, the smallest
        characteristic length will be used.

        If four values are used and the electrode length is negative, then the
        electrode positions will be read in (todo: we open the electrode.dat
        file two times here...) and the minimal distance between all electrodes
        will be multiplied by the absolute value of the imported value, and
        used as the characteristic length:

        .. math::

            l_{electrodes} = min(pdist(electrodes)) * |l_{electrodes}^{from
            file}|

        The function scipy.spatial.distance.pdist is used to compute the global
        minimal distance between any two electrodes.

        It is advisable to only used values in the range [-1, 0) for the
        automatic char length option.
        """

        if os.path.isfile(filename):
            data = np.atleast_1d(np.loadtxt(filename))
            if data.size == 4:
                characteristic_length = data
                # check sign of first (electrode) length value
                if characteristic_length[0] < 0:
                    try:
                        elec_positions = np.loadtxt(electrode_filename)
                    except:
                        raise IOError(
                            'The was an error opening the electrode file')
                    import scipy.spatial.distance
                    distances = scipy.spatial.distance.pdist(elec_positions)
                    characteristic_length[0] = min(distances) * np.abs(
                        characteristic_length[0])
                    if characteristic_length[0] == 0:
                        raise Exception(
                            'Error computing electrode ' +
                            'distances (got a minimal distance of zero')

            else:
                characteristic_length = np.ones(4) * data[0]
        else:
            characteristic_length = np.ones(4)

        if np.any(characteristic_length <= 0):
            raise Exception('No negative characteristic lengths allowed ' +
                            '(except for electrode length')

        self.char_lengths = {}
        for key, item in zip(('electrode',
                              'boundary',
                              'extra_line',
                              'extra_node'),
                             characteristic_length):
            self.char_lengths[key] = item

    def write_points(self, fid):
        """
        Write the grid points to the GMSH-command file.

        Parameters
        ----------
        fid: file object for the command file (.geo)

        """
        for nr, point in enumerate(self.Points):
            fid.write(
                'Point({0}) = {{{1}, {2}, 0, {3}}};\n'.format(
                    nr + 1, point[0], point[1], self.Charlengths[nr]))

    def write_lines(self, fid):
        for nr, line in enumerate(self.Lines):
            fid.write(
                'Line({0}) = {{{1},{2}}};\n'.format(
                    nr + 1, line[0] + 1, line[1] + 1))

    def write_in_plane_nodes(self, fid):
        for nr, line in enumerate(self.Electrodes):
            fid.write('Point {{{0}}} In Surface {{7}};\n'.format(line + 1))

    def write_extra_nodes(self, fid):
        for nr, line in enumerate(self.ExtraNodes):
            fid.write('Point {{{0}}} In Surface {{7}};\n'.format(line + 1))

    def write_geo_file(self, filename):
        """
        Write the .geo file
        """
        fid = open(filename, 'w')
        # 2D mesh algorithm (1=MeshAdapt, 2=Automatic, 5=Delaunay, 6=Frontal,
        # 7=bamg, 8=delquad)
        # according to the GMSH-mailing list the frontal algorithm should be
        # one of the best in terms of grid quality
        fid.write('Mesh.Algorithm = 6;\n')

        self.write_points(fid)
        self.write_lines(fid)

        # fid.write('Coherence;\n')
        # write line loop
        fid.write('Line Loop(1) = {')
        fid.write(','.join(['{0}'.format(x) for x in self.BoundaryIndices]))
        # for i in self.BoundaryIndices:
        #     fid.write('{0},'.format(i))
        fid.write('};\n')
        # # fid.write('{0}}};\n'.format(len(self.Lines)))
        fid.write('Plane Surface(7) = {1};\n')

        self.write_in_plane_nodes(fid)
        # fid.write('Coherence;\n')
        self.write_extra_nodes(fid)
        # fid.write('Coherence;\n')
        for index in self.ExtraLineIndices:
            fid.write('Line {' + '{0}'.format(index) + '} In Surface {7};\n')

        # Lloyd mesh optimisation crashes
        # fid.write('Mesh.Lloyd = 1;\n')

        # run the mesher
        fid.write('Mesh 7;')

        if os.path.isfile('../gmsh_commands.dat'):
            fid2 = open('../gmsh_commands.dat', 'r')
            additional_commands = fid2.read()
            fid2.close()

            fid.write('\n')
            fid.write(additional_commands)

        fid.close()


def check_boundaries(boundaries):
    # generate a complex number for each (x,y) pair
    xy = boundaries[:, 0] + 1j * boundaries[:, 1]
    """
    # for numpy > 1.9 , use:
    unique_values, indices, indices_rev, counts = np.unique(
        xy,
        return_index=True,
        return_inverse=True,
        return_counts=True)
    """
    # numpy 1.8 -->
    unique_values, indices, indices_rev = np.unique(
        xy,
        return_index=True,
        return_inverse=True)
    # now bin the indices we use to reconstruct the original array. Each index
    # that is present more than once will manifest in a bin with a number
    # larger than one. Use as many bins as there are indices.
    nr_bins = np.abs(indices_rev.min() - indices_rev.max()) + 1
    counts, b = np.histogram(indices_rev, bins=nr_bins)
    # <!-- numpy 1.8

    doublets = np.where(counts > 1)
    if doublets[0].size > 0:
        print('ERROR: Duplicate boundary coordinates found!')
        print('ERROR: Debug information')
        for doublet in doublets[0]:
            print('================')
            print('x y type:')
            print(boundaries[doublet, :])
            print('lines: ')
            print(np.where(indices_rev == doublet)[0])
        raise Exception('Duplicate boundary coordinates found!')


def add_stabilizer_nodes(boundaries_raw, electrodes, nr_nodes_between):
    """
    Segmentation of nodes:
        we have the existing nodes
        N.F is the ratio of required nodes and existing nodes
        first, add N nodes to each segment
        then, add one more node to the F first segments

    * assume ordered boundaries
    """
    boundaries = []

    boundaries = boundaries_raw
    # find first electrode in boundary
    for nr in range(electrodes.shape[0] - 1):
        index0 = np.where(
            (boundaries[:, 0] == electrodes[nr, 0]) &
            (boundaries[:, 1] == electrodes[nr, 1])
        )[0]

        index1 = np.where(
            (boundaries[:, 0] == electrodes[nr + 1, 0]) &
            (boundaries[:, 1] == electrodes[nr + 1, 1])
        )[0]

        index0 = index0[0]
        index1 = index1[0]
        if index1 - index0 < 0:
            index0, index1 = index1, index0
        running_index = index0
        nr_nodes = index1 - index0 - 1
        while nr_nodes < nr_nodes_between:
            # determine line equation
            xy0 = boundaries[running_index, 0:2]
            xy1 = boundaries[running_index + 1, 0:2]

            direction = xy1 - xy0
            heading = direction / np.sqrt(np.sum(direction ** 2))

            # new node
            xy_new = xy0 + heading * direction / 2.0
            a = boundaries[running_index, 2][np.newaxis]
            xyb = np.hstack((xy_new, a))
            boundaries = np.insert(boundaries, running_index + 1, xyb, axis=0)

            # 2, because we have to count the new one
            running_index += 2
            index1 += 1
            nr_nodes += 1

            if running_index == index1:
                running_index = index0

    return boundaries


def main():
    options, args = handle_cmd_options()

    # determine a unique directory name
    pwdx = os.getcwd()

    electrodes = np.loadtxt('electrodes.dat')
    boundaries_raw = np.loadtxt('boundaries.dat')

    check_boundaries(boundaries_raw)

    if options.add_stab_nodes:
        boundaries = add_stabilizer_nodes(boundaries_raw,
                                          electrodes,
                                          options.add_stab_nodes)
    else:
        boundaries = boundaries_raw

    # create output directory
    directory = 'tmp_grid_' + str(uuid.uuid4())
    print('Using directory: ' + directory)
    os.makedirs(directory)

    shutil.copy('electrodes.dat', directory + os.sep + 'electrodes.dat')
    shutil.copy('boundaries.dat', directory + os.sep + 'boundaries.dat')

    if os.path.isfile('extra_nodes.dat'):
        shutil.copy('extra_nodes.dat', directory + os.sep + 'extra_nodes.dat')
    if(os.path.isfile('char_length.dat')):
        shutil.copy('char_length.dat', directory + os.sep + 'char_length.dat')
    if os.path.isfile('extra_lines.dat'):
        shutil.copy('extra_lines.dat',
                    directory + os.sep + 'extra_lines.dat')
    if os.path.isfile('gmsh_commands.dat'):
        shutil.copy('gmsh_commands.dat',
                    directory + os.sep + 'gmsh_commands.dat')

    os.chdir(directory)

    mesh = Mesh()
    mesh.read_char_lengths('char_length.dat', 'electrodes.dat')

    # create boundary lines from boundary nodes
    for index in range(0, boundaries.shape[0]):
        print(index, (index + 1) % boundaries.shape[0])
        a = index
        b = (index + 1) % boundaries.shape[0]

        mesh.add_boundary(
            boundaries[a, 0:2], boundaries[b, 0:2], boundaries[a, 2])

    os.makedirs('step1')

    mesh.read_electrodes(electrodes)

    if os.path.isfile('extra_nodes.dat'):
        mesh.read_extra_nodes('extra_nodes.dat')

    if os.path.isfile('extra_lines.dat'):
        mesh.read_extra_lines('extra_lines.dat')

    mesh.write_geo_file('step1/commands.geo')
    mesh.write_electrodes('step1/electrode_positions.dat')
    mesh.write_boundaries('step1/boundary_lines.dat')
    os.chdir('step1')

    gmsh_binary = cBin.get('gmsh')
    print('Calling binary: {0}'.format(gmsh_binary))
    return_code = subprocess.call(
        '{0} -format msh2 -2 commands.geo'.format(gmsh_binary),
        shell=True
    )
    if return_code != 0:
        raise Exception('There was an error during execution of GMSH')
    os.makedirs('step2')
    os.chdir('step2')
    import cr_trig_parse_gmsh
    cr_trig_parse_gmsh.main()
    # return_code = subprocess.call('cr_trig_parse_gmsh.py', shell=True)
    # if return_code != 0:
    #     raise Exception(
    #         'There was an error during execution of cr_trig_parse_gmsh.sh'
    #     )

    #
    os.makedirs('grid')
    shutil.copy('elec.dat', 'grid/elec.dat')
    shutil.copy('../elem.dat', 'grid/elem.dat')
    os.chdir('grid')
    cutmck_binary = cBin.get('CutMcK')
    print('Calling binary: {0}'.format(cutmck_binary))
    return_code = subprocess.call(cutmck_binary, shell=True)
    if return_code != 0:
        raise Exception(
            'There was an error during execution of CutMcK'
        )

    # for convenience, copy the final grid files to the top level
    shutil.copy('elem.dat', '../../../elem.dat')
    shutil.copy('elec.dat', '../../../elec.dat')
    os.chdir(pwdx)

    print('The final grid can be found in:')

    # if we have one parameter, use this as the output directory
    if len(args) == 1:
        output_dir = args[0]
        if not os.path.isdir(output_dir):
            shutil.move(directory, output_dir)
            print(output_dir)
        else:
            print('temporary directory:')
            print(directory)
            raise Exception('output directory already exists!')
    else:
        print(directory)


if __name__ == '__main__':
    main()
