#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parse a GMSH mesh and produce a FEM grid for CRMod/CRTomo. This script should
not be called directly, as it requires the input and output files from
"cr_trig_create.py".

TODO
----

* the program at some point looks for nodes on the boundary line. Perhaps some
  funtions in "grid_extralines_gen_decouplings.py" can be used here (code
  deduplication and speed improvements)


"""
from crtomo.mpl_setup import *
mpl.rcParams['font.size'] = 6.0
import numpy as np


def parse_gmsh(filename, boundary_file):
    """
    Parse a GMSH .msh file and return a dictionary containing the data
    neccessary to create CRTomo grids
    """
    mesh = {}

    fid = open(filename, 'r')
    line = fid.readline()
    while(line):
        if(line.startswith('$MeshFormat')):
            pass
        elif(line.startswith('$Nodes')):
            nodes = []
            line = fid.readline()
            nr_nodes = np.fromstring(line, dtype=int, count=1, sep=r'\n')
            nr_nodes
            while(line):
                line = fid.readline()
                if(line.startswith('$EndNodes')):
                    break
                node = np.fromstring(line, dtype=float, sep=' ')
                nodes.append(node)
            mesh['nodes'] = nodes
        elif(line.startswith('$Elements')):
            """
            Create a dictionary with the element types as keys. E.g.:
            elements['15'] provides all elements of type 15 (Points)
            """
            elements = {}
            line = fid.readline()
            nr_elements = np.fromstring(line, dtype=int, count=1, sep=r'\n')
            nr_elements
            while(line):
                line = fid.readline()
                if(line.startswith('$EndElements')):
                    break
                element = np.fromstring(line, dtype=int, sep=' ')
                # el_nr = element[0]
                el_type = element[1]
                el_nr_tags = element[2]
                # el_tags = element[3:3 + el_nr_tags]
                el_nodes = element[3 + el_nr_tags:]

                # now decide where to put it
                key = str(el_type)
                if(key in elements.keys()):
                    elements[key].append(el_nodes)
                else:
                    elements[key] = []
                    elements[key].append(el_nodes)

            mesh['elements'] = elements
        line = fid.readline()

    fid.close()

    # if boundary_file is != None, then sort the lines (element type 1)
    # according to the element types
    boundaries = {}

    if(boundary_file is not None):
        # load the original boundary lines
        # it is possible that GMSH added additional nodes on these lines, and
        # that is why we need to find all mesh lines that lie on these original
        # lines.
        bids = np.loadtxt(boundary_file)

        for btype in ('12', '11'):
            # select all original boundaries with this type
            a = np.where(bids[:, 4] == int(btype))[0]
            boundaries[btype] = []
            # for each of those lines, find all lines of the mesh that belong
            # here
            for orig_line in bids[a, :]:
                # print('Find all lines lying on the line: ')
                found_one_line = False
                # print(orig_line)
                # construct line equation

                # x1 == x2 ?
                # split into coordinates
                ox1 = orig_line[0]
                ox2 = orig_line[2]
                oy1 = orig_line[1]
                oy2 = orig_line[3]

                if(orig_line[0] == orig_line[2]):
                    # special case: we only need to find all lines with x1 ==
                    # x2 == x1_orig and y_min >= y_orig_min and y_max <=
                    # <_orig_max
                    for line in elements['1']:
                        if(btype == '11'):
                            if(line[0] == 48 and line[1] == 150):
                                pass
                                # print('Find all lines lying on the line: ')
                                # print('This is the line')

                        # it doesn't matter any more to be able to assign x ->
                        # y values. Thus we can sort the y values and just
                        # check
                        # if the new line lies in between the original one
                        oy1, oy2 = np.sort([orig_line[1], orig_line[3]])
                        x1, x2 = np.sort(
                            [
                                mesh['nodes'][line[0] - 1][1],
                                mesh['nodes'][line[1] - 1][1]
                            ]
                        )
                        y1, y2 = np.sort(
                            [
                                mesh['nodes'][line[0] - 1][2],
                                mesh['nodes'][line[1] - 1][2]
                            ]
                        )

                        if np.isclose(x1, x2) and np.isclose(x2, ox1):
                            if(y1 >= oy1 and y2 <= oy2):
                                found_one_line = True
                                boundaries[btype].append(line)

                else:
                    # print('checking with full line equation')
                    # no vertical line
                    # we need the full check using the line equation
                    slope = (orig_line[1] - orig_line[3]) / (
                        orig_line[0] - orig_line[2])
                    intersect = orig_line[1] - (slope * orig_line[0])
                    # print('Slope', slope, ' Intercept ', intersect)
                    for line in elements['1']:
                        x1 = mesh['nodes'][line[0] - 1][1]
                        y1 = mesh['nodes'][line[0] - 1][2]
                        x2 = mesh['nodes'][line[1] - 1][1]
                        y2 = mesh['nodes'][line[1] - 1][2]

                        # print(x1, x2, y1, y1)
                        check = False
                        # check if x coordinates of the test line fit in the
                        # original line
                        if(ox1 < ox2):
                            if(x1 < x2):
                                if((np.isclose(x1, ox1) or x1 > ox1) and
                                   (np.isclose(x2, ox2) or x2 < ox2)):
                                    check = True
                            else:
                                if((np.isclose(x2, ox1) or x2 >= ox1) and
                                   (np.isclose(x1, ox2) or x1 <= ox2)):
                                    check = True
                        else:
                            if(x1 < x2):
                                if((np.isclose(x1, ox2) or x1 >= ox2) and
                                   (np.isclose(x2, ox1) or x2 <= ox1)):
                                    check = True
                            else:
                                if((np.isclose(x2, ox2) or x2 >= ox2) and
                                   (np.isclose(x1, ox1) or x1 <= ox1)):
                                    check = True

                        # print('boundary check:', check)
                        if(check):
                            # the line lies within the x-range of the orig line
                            ytest1 = slope * x1 + intersect
                            ytest2 = slope * x2 + intersect
                            if(np.around(ytest1 - y1, 5) == 0 and
                               np.around(ytest2 - y2, 5) == 0):
                                boundaries[btype].append(line)
                                # found = True
                                found_one_line = True
                                # print('found it new', line)
                # add a weak check: we need to find at least one line in the
                # mesh corresponding to this boundary line:
                if not found_one_line:
                    raise Exception('no mesh line found for this boundary')

            print('Total number of boundaries of this type:',
                  len(boundaries[btype]))
    mesh['boundaries'] = boundaries
    return mesh


class _line():
    def __init__(self, p1, p2):
        self.p1_x = p1[0]
        self.p1_y = p1[1]

        self.p2_x = p2[0]
        self.p2_y = p2[1]

    def get_diff(self):
        diff = []
        diff.append(self.p1_x - self.p2_x)
        diff.append(self.p1_y - self.p2_y)
        return np.array(diff)

    def get_distance(self):
        diff = self.get_diff()
        dist = np.sqrt(np.sum(diff ** 2))
        return dist

    def get_center(self):
        """
        Return (x,y) coordinate of central point
        """
        dist = self.get_distance()
        diff = self.get_diff()
        direction = diff / dist

        center_x = self.p2_x + (dist * 0.5) * direction[0]
        center_y = self.p2_y + (dist * 0.5) * direction[1]

        return np.array([center_x, center_y])


def debug_plot_mesh(mesh, boundary_elements):
    plot_large = True

    # prepare nodes
    nodes = np.array(mesh['nodes'])
    tx = nodes[:, 1]
    ty = nodes[:, 2]

    # adapt height of plot to size of grid
    tx_size = np.abs(np.max(tx) - np.min(tx))
    ty_size = np.abs(np.max(ty) - np.min(ty))

    if(plot_large):
        width = 10
    else:
        width = 7
    size_x = width
    size_y = width * (ty_size / tx_size) * 1.5

    # plot triangles
    triangles = np.array(mesh['elements']['2']) - 1

    fig, ax = plt.subplots(1, 1, figsize=(size_x, size_y))
    ax.triplot(tx, ty, triangles, color='k', linewidth=1.0)

    # plot boundaries in red
    # lines = np.array(mesh['elements']['1']) - 1
    lines = np.array(mesh['boundaries']['12'] + mesh['boundaries']['11']) - 1
    lx = nodes[lines, 1]
    ly = nodes[lines, 2]

    for index in range(0, lx.shape[0]):
        # for index in range(0, 1):
        ax.plot(lx[index, :], ly[index, :], 'r.', alpha=0.4)
        # draw a line to the neighbour
        line_obj = _line([lx[index, 0], ly[index, 0]],
                         [lx[index, 1], ly[index, 1]])
        center = line_obj.get_center()
        ntri = np.array(mesh['elements']['2'])[int(boundary_elements[index])]
        ntri_x = nodes[ntri - 1, 1]
        ntri_y = nodes[ntri - 1, 2]

        # compute centroid of element adjacent to the boundary element
        centroidx = np.sum(ntri_x) / 3
        centroidy = np.sum(ntri_y) / 3

        ax.plot([centroidx, center[0]], [centroidy, center[1]], color='g',
                label='adjacent elements')
        ax.annotate('{0}'.format(index), xy=(centroidx, centroidy),
                    fontsize=6.0, color='k')

    ax.set_xlim([np.min(tx) - 0.1, np.max(tx) + 0.1])
    ax.set_ylim([np.min(ty) - 0.1, np.max(ty) + 0.1])
    ax.set_aspect('equal')

    # plot electrodes
    electrodes = np.loadtxt('../electrode_positions.dat')
    ax.scatter(electrodes[:, 0], electrodes[:, 1], s=30,
               color='blue',
               label='electrodes')

    fig.tight_layout()
    if(plot_large):
        dpi = 600
    else:
        dpi = 300
    fig.savefig('../../triangle_grid.png', dpi=dpi)


def get_header(mesh):
    """
    For now we only know Element types 1 (line) and 2 (triangle)
    """
    nr_all_nodes = len(mesh['nodes'])
    nr_types = 1  # triangles

    # compute bandwidth
    bandwidth = -1
    for triangle in mesh['elements']['2']:
        diff1 = abs(triangle[0] - triangle[1])
        diff2 = abs(triangle[0] - triangle[2])
        diff3 = abs(triangle[1] - triangle[2])
        diffm = max(diff1, diff2, diff3)
        if(diffm > bandwidth):
            bandwidth = diffm

    el_infos = []
    # triangles
    for element in ('2',):
        el = mesh['elements'][element]
        if(element == '2'):
            el_type = 3
        elif(element == '1'):
            el_type = 12  # Neumann
        nr = len(el)
        nr_nodes = len(el[0])
        el_infos.append((el_type, nr, nr_nodes))

    # boundary elements
    for btype in ('12', '11'):
        if(btype in mesh['boundaries']):
            el_type = int(btype)
            nr = len(mesh['boundaries'][btype])
            nr_nodes = 2
            if(nr > 0):
                nr_types += 1
                el_infos.append((el_type, nr, nr_nodes))

    # now convert to string
    str_header = ''
    for a, b, c in [(nr_all_nodes, nr_types, bandwidth), ] + el_infos:
        str_header = str_header + '{0}  {1} {2}\n'.format(a, b, c)
    return str_header


def get_nodes(mesh):
    nodes = np.array(mesh['nodes'])

    str_nodes = ''
    for node_nr in range(0, nodes.shape[0]):
        str_nodes += '{0}   {1} {2}\n'.format(node_nr + 1, nodes[node_nr, 1],
                                              nodes[node_nr, 2])
    return str_nodes


def get_elements(mesh):
    # print('Get elements')
    str_elements = ''
    # triangles
    for element in ('2'):
        el = mesh['elements'][element]
        for item in el:
            # the reverse should ensure counter-clockwise element nodes
            if(element == '1'):
                # boundary elements
                lst = item
            else:
                lst = reversed(item)

            for i in lst:
                str_elements += '{0}  '.format(i)
            str_elements += '\n'
    # boundary elements
    for btype in ('12', '11'):
        for line in mesh['boundaries'][btype]:
            str_elements += '{0} {1}\n'.format(line[0], line[1])

    return(str_elements)


def get_ajd_bound(mesh):
    """
    Determine triangular elements adjacend to the boundary elements
    """
    print('Get elements adjacent to boundaries')
    boundary_elements = []
    str_adj_boundaries = ''
    # for boundary in mesh['elements']['1']:
    boundaries = mesh['boundaries']['12'] + mesh['boundaries']['11']
    for boundary in boundaries:
        # now find the triangle ('2') with two nodes equal to this boundary
        indices = [nr if (boundary[0] in x and boundary[1] in x) else np.nan
                   for (nr, x) in enumerate(mesh['elements']['2'])]
        indices = np.array(indices)[~np.isnan(indices)]
        if(len(indices) != 1):
            print('More than one neighbour found!')
        elif(len(indices) == 0):
            print('No neighbour found!')
        boundary_elements.append(indices[0])
        str_adj_boundaries += '{0}\n'.format(int(indices[0]) + 1)
    return str_adj_boundaries, boundary_elements


def write_elec_file(filename, mesh):
    """
    Read in the electrode positions and return the indices of the electrodes

    # TODO: Check if you find all electrodes
    """
    elecs = []
    # print('Write electrodes')
    electrodes = np.loadtxt(filename)
    for i in electrodes:
        # find
        for nr, j in enumerate(mesh['nodes']):
            if np.isclose(j[1], i[0]) and np.isclose(j[2], i[1]):
                elecs.append(nr + 1)

    fid = open('elec.dat', 'w')
    fid.write('{0}\n'.format(len(elecs)))
    for i in elecs:
        fid.write('{0}\n'.format(i))
    fid.close()


def main():
    mesh = parse_gmsh('../commands.msh', '../boundary_lines.dat')

    # now create the CRTomo grid
    """
    1. Header
    2. Nodes
    3. Elements: Triangles, Boundary elements
    4. Element ids for adjoining boundary elements
    """

    str_header = get_header(mesh)
    str_nodes = get_nodes(mesh)
    str_elements = get_elements(mesh)
    str_adj_boundaries, boundary_elements = get_ajd_bound(mesh)

    crt_mesh = str_header + str_nodes + str_elements + str_adj_boundaries

    fid = open('../elem.dat', 'w')
    fid.write(crt_mesh)
    fid.close()

    write_elec_file('../electrode_positions.dat', mesh)
    debug_plot_mesh(mesh, boundary_elements)
