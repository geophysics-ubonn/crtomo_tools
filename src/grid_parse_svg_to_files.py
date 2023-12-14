#!/usr/bin/env python
"""

"""
import os
import shutil

from xml.dom import minidom
# import shapely
import matplotlib.pylab as plt
import numpy as np
from shapely.geometry import Polygon
# from shapely.geometry import Point
from shapely.geometry import LineString
import shapely.plotting


def main():
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

    doc = minidom.parse('out_modified2.svg')  # parseString also exists

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

    region_str_list = {}
    for x in doc.getElementsByTagName('g'):
        label = x.getAttribute('inkscape:label')
        print(label)
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
        points_raw = []
        is_relative = False
        close_poly = False
        for token in region_str.split(' '):
            if token == 'm':
                is_relative = True
            elif token == 'M':
                pass
            elif token.find(',') > 0:
                print('got a coordinate', token)
                tmp = token.split(',')
                x = float(tmp[0])
                y = float(tmp[1])
                points_raw += [(x, y)]
            elif token in ('z', 'Z'):
                close_poly = True
        points = []
        if is_relative:
            points += [points_raw[0]]
            for tmp in points_raw[1:]:
                points += [
                    (tmp[0] + points[-1][0], tmp[1] + points[-1][1])
                ]
        else:
            points = points_raw

        if close_poly:
            points += [points[0]]

        poly = np.array(points)
        print("poly raw:", poly)
        poly[:, 0] += offsetx
        poly[:, 1] = -poly[:, 1] + height + offsety
        print("poly:", poly)

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
            print('Checking line: ', line)
            shapely.plotting.plot_line(line, ax=ax)
            # the line starts/ends IN the boundary, but crosses it
            partially_inside = (
                not boundary_area.contains(line)
            ) & boundary_area.intersects(line)

            is_outside = (
                not boundary_area.contains(line)
            ) & (not boundary_area.intersects(line))
            print('    outside: {} partially_inside: {}'.format(
                is_outside,
                partially_inside
            ))
            if partially_inside:
                print('    NEEDS FIXING')
                p1 = poly[i]
                p2 = poly[i + 1]
                line = LineString([p1, p2])
                # import IPython
                # IPython.embed()
                # exit()
                # we need to shorten the line
                line_limited = boundary_area.intersection(line)
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
                        print('    INTER', bline)
                        shapely.plotting.plot_line(bline, ax=ax, color='green')
                        newp = bline.intersection(line)
                        print('    newp:', newp)
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
                    print('FIX:', i, poly[i], poly[i + 1], line_limited, newp)
                except Exception as e:
                    print(e)
                    fig, ax = plt.subplots()
                    ax.set_title('DEBUG')
                    shapely.plotting.plot_line(boundary_line, ax=ax)
                    shapely.plotting.plot_line(line, ax=ax)
                    fig.show()
                    import IPython
                    IPython.embed()

                shapely.plotting.plot_line(line_limited, ax=ax, color='red')

                # lines_inside += [np.array([poly[i], poly[i + 1]]).flatten()]
                lines_inside += [np.array([p1, p2]).flatten()]

            elif is_outside:
                print('Completely outside')
            else:
                lines_inside += [np.array([poly[i], poly[i + 1]]).flatten()]

        # shapely.plotting.plot_polygon(Polygon(boundaries[:, 0:2]), ax=ax)
        fig.show()
        fig, ax = plt.subplots()
        shapely.plotting.plot_polygon(Polygon(boundaries[:, 0:2]), ax=ax)
        l0 = lines_inside[0]
        ax.plot([l0[0], l0[2]], [l0[1], l0[3]], color='black', linewidth=10)

        fig.show()

        # fixing end

        # this file contains everything, unmodified
        np.savetxt(
            'mdl_{}.dat'.format(region_name),
            np.array(lines_unmodified),
            fmt="%.4f %.4f %.4f %.4f",
        )

        np.savetxt(
            '{}.dat'.format(region_name),
            np.array(lines_inside),
            fmt="%.4f %.4f %.4f %.4f",
        )

        # with open('mdl_{}.dat'.format(region_name), 'w') as fid:
        #     for i in range(0, poly.shape[0] - 1):
        #         fid.write('{} {} {} {}\n'.format(
        #             poly[i][0], poly[i][1],
        #             poly[i + 1][0], poly[i + 1][1],
        #         ))

        ax.plot(poly[:, 0], poly[:, 1])

    if os.path.isfile('boundaries.dat'):
        print('boundaries.dat already exists, moving to boundaries_orig.dat')
        shutil.move('boundaries.dat', 'boundaries_orig.dat')

    np.savetxt(
        'boundaries.dat', boundaries,
        fmt="%.4f %.4f %i",
    )

    fig.show()
