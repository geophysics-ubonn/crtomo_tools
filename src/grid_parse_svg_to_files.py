#!/usr/bin/env python
"""TODO: Description

This scripts parses a .svg file and converts layers into individual geometries
that can be used to

- add these geometries in newly created meshes (for forward modelings)
- modify subsurface models using these geometries
- decouple regularisation in the inversion

Further information
-------------------

https://developer.mozilla.org/en-US/docs/Web/SVG/Tutorial/Paths

Output files
------------

For each layer specified in the parsed svg file, the following files will be
written:

    * all_[REGION_NAME].dat - the lines of the specified region, unmodified
                              from the svg
    * lne_[REGION_NAME].dat - all lines located within the boundary region. Use
                              this file to create extra_lines.dat files for
                              grid creation
    * pts_[REGION_NAME].dat - the points of the intersection of the region with
                              the boundary

"""
import os
import shutil
from argparse import ArgumentParser

from xml.dom import minidom
# import shapely
import matplotlib.pylab as plt
import numpy as np
from shapely.geometry import Polygon
# from shapely.geometry import Point
from shapely.geometry import LineString
import shapely.plotting


def handle_cmd_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-i',
        "--svg",
        # dest="",
        type=str,
        help="SVG file to parse",
        default='out_modified2.svg',
    )

    return parser.parse_args()


def parse_svg_path(path_str):
    """
    """
    line_commands = ['M', 'm', 'L', 'l', 'H', 'h', 'V', 'v', 'Z', 'z']
    close_path = False
    cmd = None
    pos = np.array([0.0, 0.0])
    pts = []
    for token in path_str.split(' '):
        if token in line_commands:
            cmd = token
            if cmd == 'Z' or cmd == 'z':
                close_path = True
                continue

        else:
            if cmd == 'M':
                pos = np.array([float(s) for s in token.split(',')])
            elif cmd == 'm':
                pos += [float(s) for s in token.split(',')]
            elif cmd == 'L':
                pos = np.array([float(s) for s in token.split(',')])
            elif cmd == 'l':
                pos += np.array([float(s) for s in token.split(',')])
            elif cmd == 'H':
                pos[0] = float(token)
            elif cmd == 'h':
                pos[0] += float(token)
            elif cmd == 'V':
                pos[1] = float(token)
            elif cmd == 'v':
                pos[1] += float(token)

            pts.append(pos.round(6))
            # if round is not used, copy pos array:
            # pts.append(pos.copy())

    if close_path:
        pts.append(pts[0])

    return np.array(pts)


def main():
    args = handle_cmd_args()
    # import IPython
    # IPython.embed()

    if not os.path.isfile('boundaries_orig.dat'):
        if os.path.isfile('boundaries.dat'):
            print(
                'boundaries.dat already exists, moving to boundaries_orig.dat'
            )
            shutil.move('boundaries.dat', 'boundaries_orig.dat')
    else:
        if os.path.isfile('boundaries.dat'):
            print(
                'WARNING: boundaries_orig.dat AND boundaries.dat already exit'
            )
            print(
                'stopping here.'
            )
            exit()

    boundaries = np.loadtxt('boundaries_orig.dat')
    fig, ax = plt.subplots()
    ax.plot(boundaries[:, 0], boundaries[:, 1])

    infile = args.svg
    print('Using file for input: {}'.format(infile))
    assert os.path.isfile(infile), "Input file does not exist: {}".format(
        infile
    )
    doc = minidom.parse(infile)  # parseString also exists

    svg = doc.getElementsByTagName('svg')[0]
    offsetx = float(svg.attributes['crtomo_offset_x'].value)
    offsety = float(svg.attributes['crtomo_offset_y'].value)
    # width = float(svg.attributes['crtomo_width'].value)
    height = float(svg.attributes['crtomo_height'].value)

    svg_layers_to_parse = (
        'special_',
        'constraint_',
        'region_',
    )
    print(
        'Looking for the following layers within the .svg file:',
        svg_layers_to_parse
    )

    region_str_list = {}
    for x in doc.getElementsByTagName('g'):
        label = x.getAttribute('inkscape:label')
        print('Found layer: ', label)
        # if label.startswith('region_'):
        # if label.startswith('region_') or label.startswith('constraint_'):
        for svg_layer in svg_layers_to_parse:
            if label.startswith(svg_layer):
                region_str = x.getElementsByTagName(
                    'path'
                )[0].getAttribute('d')
                region_str_list[label] = region_str

    for region_name, region_str in region_str_list.items():
        print('-' * 80)
        print('REGION', region_name, region_str)
        print('')
        # points_raw = []
        # is_relative = False
        # close_poly = False
        # import IPython
        # IPython.embed()

        # # this is where we parse the svg path
        # for token in region_str.split(' '):
        #     if token == 'm':
        #         # this indicate a relative movement
        #         is_relative = True
        #     elif token == 'M':
        #         # Move to command
        #         pass
        #     elif token.find(',') > 0:
        #         tmp = token.split(',')
        #         x = float(tmp[0])
        #         y = float(tmp[1])
        #         points_raw += [(x, y)]
        #     elif token in ('z', 'Z'):
        #         close_poly = True
        # points = []
        # if is_relative:
        #     points += [points_raw[0]]
        #     for tmp in points_raw[1:]:
        #         points += [
        #             (tmp[0] + points[-1][0], tmp[1] + points[-1][1])
        #         ]
        # else:
        #     points = points_raw

        # if close_poly:
        #     points += [points[0]]

        # poly = np.array(points)
        poly = parse_svg_path(region_str)
        poly[:, 0] += offsetx
        poly[:, 1] = -poly[:, 1] + height + offsety

        # fix boundaries: start
        boundary_area = Polygon(boundaries[:, 0:2])
        boundary_line = boundary_area.boundary

        fig, ax = plt.subplots()
        shapely.plotting.plot_polygon(boundary_area, ax=ax)

        lines_inside = []
        lines_unmodified = []

        for i in range(poly.shape[0] - 1):
            lines_unmodified += [np.array([poly[i], poly[i + 1]]).flatten()]
            line = LineString([poly[i], poly[i + 1]])
            # print('Checking line: ', line)
            shapely.plotting.plot_line(line, ax=ax)
            # the line starts/ends IN the boundary, but crosses it
            partially_inside = (
                not boundary_area.contains(line)
            ) & boundary_area.intersects(line)

            is_outside = (
                not boundary_area.contains(line)
            ) & (not boundary_area.intersects(line))
            # print('    outside: {} partially_inside: {}'.format(
            #     is_outside,
            #     partially_inside
            # ))
            if partially_inside:
                # print('    NEEDS FIXING')
                p1 = poly[i]
                p2 = poly[i + 1]
                line = LineString([p1, p2])
                # import IPython
                # IPython.embed()
                # exit()
                # we need to shorten the line
                line_limited = boundary_area.intersection(line)
                if isinstance(line_limited, shapely.geometry.MultiLineString):
                    line_complete = shapely.geometry.LineString([
                        [
                            line_limited.geoms[0].xy[0][0],
                            line_limited.geoms[0].xy[1][0],
                        ],
                        [
                            line_limited.geoms[-1].xy[0][1],
                            line_limited.geoms[-1].xy[1][1],
                        ]
                    ])
                    assert line_complete.equals(line_limited)
                    line_limited = line_complete

                # import IPython
                # IPython.embed()
                # we need to change our boundary
                j = 0
                # for j in range(boundaries.shape[0]):
                # fix the boundaries
                while j < boundaries.shape[0]:
                    # print('checking boundary', j)
                    bline = LineString([
                        boundaries[j, 0:2],
                        boundaries[(j + 1) % (boundaries.shape[0]), 0:2]
                    ])
                    if bline.intersects(line):
                        # print('    INTER', bline)
                        shapely.plotting.plot_line(bline, ax=ax, color='green')
                        newp = bline.intersection(line)
                        # print('    newp:', newp)
                        # insert the new node after the current i-th node
                        boundaries = np.vstack(
                            (
                                boundaries[0:j + 1],
                                [
                                    newp.xy[0][0],
                                    newp.xy[1][0],
                                    boundaries[j, 2]
                                ],
                                boundaries[j+1:])
                            )
                        # we added one boundary, move the index forward
                        j += 1
                    j += 1
                # jetzt use the line_limited as the new line
                coords = np.vstack(line_limited.xy).flatten(order='F').tolist()
                p1 = coords[0:2]
                p2 = coords[2:4]

                try:
                    pass
                    # print(
                    #   'FIX:', i, poly[i], poly[i + 1], line_limited, newp)
                except Exception as e:
                    print(e)
                    fig, ax = plt.subplots()
                    ax.set_title('DEBUG')
                    shapely.plotting.plot_line(boundary_line, ax=ax)
                    shapely.plotting.plot_line(line, ax=ax)
                    fig.savefig('debug_grid_parse.jpg', dpi=300)
                    fig.show()
                    import IPython
                    IPython.embed()

                shapely.plotting.plot_line(line_limited, ax=ax, color='red')

                # lines_inside += [np.array([poly[i], poly[i + 1]]).flatten()]
                lines_inside += [np.array([p1, p2]).flatten()]

            elif is_outside:
                pass
                # print('Completely outside')
            else:
                lines_inside += [np.array([poly[i], poly[i + 1]]).flatten()]

        # shapely.plotting.plot_polygon(Polygon(boundaries[:, 0:2]), ax=ax)
        fig.savefig('grid_parse_region_{}.jpg'.format(region_name), dpi=300)
        fig.show()

        fig, ax = plt.subplots()
        shapely.plotting.plot_polygon(Polygon(boundaries[:, 0:2]), ax=ax)
        for line in lines_inside:
            l0 = line
            ax.plot(
                [l0[0], l0[2]], [l0[1], l0[3]],
                color='black',
                linewidth=5
            )

        ax.set_title(
            'lines inside the mesh',
            loc='left',
            fontsize=8,
        )
        fig.tight_layout()

        fig.savefig('grid_parse_lines.jpg', dpi=300)
        fig.show()

        # fixing end

        # this file contains everything, unmodified
        np.savetxt(
            'all_{}.dat'.format(region_name),
            np.array(lines_unmodified),
            fmt="%.4f %.4f %.4f %.4f",
        )

        # only the lines inside the mesh
        np.savetxt(
            'lne_{}.dat'.format(region_name),
            np.array(lines_inside),
            fmt="%.4f %.4f %.4f %.4f",
        )

        # compute the intersection of mesh and region and write out the points
        # only compute a polygon-intersection for more than one line
        if len(lines_unmodified) > 1:
            poly_region = Polygon(
                np.array(lines_unmodified)[:, 0:2]
            ).normalize()
            area_reduced_to_boundary = boundary_area.intersection(
                poly_region
            ).normalize()
            region_points = np.array(
                area_reduced_to_boundary.boundary.coords
            )[:-1, :]

            filename = 'pts_{}.dat'.format(region_name)
            np.savetxt(filename, region_points, fmt="%.4f")

        # ?
        ax.plot(poly[:, 0], poly[:, 1])

    if os.path.isfile('boundaries.dat'):
        print('boundaries.dat already exists, moving to boundaries_orig.dat')
        shutil.move('boundaries.dat', 'boundaries_orig.dat')

    np.savetxt(
        'boundaries.dat', boundaries,
        fmt="%.4f %.4f %i",
    )

    fig.savefig('grid_parse_final.jpg', dpi=300)
    fig.show()
