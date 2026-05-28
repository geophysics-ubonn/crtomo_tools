"""

"""
import tempfile
import numpy as np
import subprocess
import os
import crtomo

import shapely
from shapely.ops import split


def poly_to_linestring(in_polygon):
    xx, yy = in_polygon.exterior.coords.xy
    # construct lines
    lines = []
    for index in range(0, len(xx) - 1):
        lines += [
            [
                xx[index], yy[index]
            ],
            [
                xx[index + 1], yy[index + 1]
            ]
        ]
    return shapely.geometry.LineString(lines)


def linestring_to_poly(in_linestring):
    return shapely.geometry.Polygon(in_linestring.coords[::2])


def linestring_to_crt_lines(in_ls):
    lines = []
    for index in range(0, len(in_ls.coords), 2):

        index2 = (index + 1) % len(in_ls.coords)

        line = np.hstack((in_ls.coords[index], in_ls.coords[index2]))
        lines += [line]
    lines = np.array(lines)
    return lines


def explode_linestring(in_ls):
    lines = []
    for pt1, pt2 in zip(in_ls.coords, in_ls.coords[1:]):
        if pt1 != pt2:
            lines += [shapely.geometry.LineString([pt1, pt2])]
    return lines


class CRTomoGMSHMeshGenerator():
    """An interface to cr_trig_create
    """
    def __init__(self):
        pass

    def check_electrodes_in_or_on_boundary(self, boundaries, electrodes):
        """Given a numpy array with boundary inputs, check that all electrodes
        are located within the region marked by the boundary

        Returns
        -------
        Returns True if all electrodes lie on, or within, the boundary region.
        False otherwise

        """
        # create a polygon out of the boundary nodes
        boundary_polygon = self._boundaries_to_polygon(boundaries)

        for (x, y) in electrodes:
            check = boundary_polygon.intersects(
                shapely.geometry.Point((x, y))
            )
            if not check:
                return False
        return True

    def gen_mesh(self, boundaries, electrodes, char_lengths=None,
                 extra_lines=None, extra_nodes=None,
                 return_output=False,
                 ):
        pwd = os.getcwd()

        if not self.check_electrodes_in_or_on_boundary(boundaries, electrodes):
            print(
                'ERROR: Not all electrodes lie on, or within, the boundary ' +
                'region!'
            )
            if return_output:
                return None, None
            return None

        if extra_lines is not None:
            extra_lines_crop = self.crop_lines_to_boundary(
                boundaries,
                extra_lines
            )
            boundaries = self._extra_lines_add_to_boundaries(
                boundaries, extra_lines_crop
            )
        else:
            extra_lines_crop = None

        # debug
        # workdir = 'mesh_gen'
        # os.makedirs(workdir, exist_ok=True)
        # os.chdir(workdir)
        # debug end

        workdir = tempfile.TemporaryDirectory()
        os.chdir(workdir.name)

        np.savetxt('boundaries.dat', boundaries)
        np.savetxt('electrodes.dat', electrodes)
        if char_lengths is not None:
            np.savetxt('char_length.dat', char_lengths)
        if extra_lines_crop is not None and len(extra_lines) > 0:
            np.savetxt('extra_lines.dat', extra_lines_crop)
        if extra_nodes is not None:
            np.savetxt('extra_nodes.dat', extra_nodes)

        print('Calling the mesh generation routines...')
        try:
            output = subprocess.check_output(
                'cr_trig_create mesh',
                shell=True
            )
        except subprocess.CalledProcessError:
            os.chdir(pwd)
            return None
        print('   done')
        if os.path.isfile('mesh/elem.dat') and os.path.isfile('mesh/elec.dat'):
            mesh = crtomo.crt_grid('mesh/elem.dat', 'mesh/elec.dat')
        else:
            mesh = None
        os.chdir(pwd)
        if return_output:
            return mesh, output

        return mesh

    def _boundaries_to_lines(self, boundaries):
        """

        """
        # print('boundaries to linestring')
        bound_ls = []
        for index in range(0, boundaries.shape[0]):
            index2 = (index + 1) % boundaries.shape[0]
            line = shapely.geometry.LineString(
                ((boundaries[index, 0:2], boundaries[index2, 0:2]))
            )  # .normalize().reverse()
            bound_ls += [[line, boundaries[index, 2].item()]]
        # print(bound_ls)
        # print('now done')
        return bound_ls

    def _linestring_to_boundaries(self, bls):
        pass

    def _boundaries_to_polygon(self, boundaries):
        return shapely.geometry.Polygon(boundaries[:, 0:2])

    def _lines_to_crt_boundaries(self, boundary_lines):
        """
        boundary_lines: list of lists
            Each entry should contain a two-list/tuple with a
            shapely.geometry.LineString, boundary type content
        """
        boundaries = []
        for line, btype in boundary_lines:
            # print(line)
            boundaries += [[
                line.coords[0][0],
                line.coords[0][1],
                btype
            ]]

        return np.array(boundaries)

    def fix_extra_lines(self, extra_lines_in):
        # print('fix_extra_lines')

        # first fix: find parallel, overlapping lines and split them
        # accordingly
        index = 0
        while index < len(extra_lines_in):
            line1c = extra_lines_in[index]
            line1 = shapely.geometry.LineString(
                [
                    [line1c[0], line1c[1]],
                    [line1c[2], line1c[3]],
                ]
            )

            # print('line1', line1)
            # look only following lines, we already dealt with all previous
            # ones
            index2 = index + 1
            while index2 < len(extra_lines_in):
                line2c = extra_lines_in[index2]
                line2 = shapely.geometry.LineString(
                    [
                        [line2c[0], line2c[1]],
                        [line2c[2], line2c[3]],
                    ]
                )
                if line1.intersects(line2):
                    # print('intersecting lines')
                    intersection = line1.intersection(line2)
                    if type(intersection) is shapely.geometry.LineString:
                        # print('PAR', index2)
                        l2diff = line2.difference(line1)
                        extra_lines_in = np.vstack((
                            extra_lines_in[0:index2],
                            np.array(((
                                l2diff.coords[0][0],
                                l2diff.coords[0][1],
                                l2diff.coords[1][0],
                                l2diff.coords[1][1],
                            ))),
                            extra_lines_in[index2+1:, :],
                        ))
                        index2 += 1

                index2 += 1
            index += 1

        # second fix: find lines that have their start point in existing lines
        # print('------------------------------------------------')
        # print('Second fix')
        index = 0
        while index < len(extra_lines_in):
            line1c = extra_lines_in[index]
            line1 = shapely.geometry.LineString(
                [
                    [line1c[0], line1c[1]],
                    [line1c[2], line1c[3]],
                ]
            )

            # print('line1', line1)
            index2 = 0
            while index2 < len(extra_lines_in):
                if index == index2:
                    index2 += 1
                    continue
                line2c = extra_lines_in[index2]
                line2 = shapely.geometry.LineString(
                    [
                        [line2c[0], line2c[1]],
                        [line2c[2], line2c[3]],
                    ]
                )
                if line1.intersects(line2):
                    # print('  intersecting lines')
                    intersection = line1.intersection(line2)
                    if type(intersection) is shapely.geometry.Point:
                        # print('  Got a point intersection')
                        # check for start/end points
                        check_endpoints = (
                            intersection == line2.coords[0] or
                            intersection == line2.coords[1]
                        )
                        if check_endpoints:
                            # print('   endpoint check failed')
                            index2 += 1
                            continue

                        # make sure we are not only dealing with endpoints
                        # TODO: This check seems odd
                        check_con_end = (
                            line1.coords[0] == line2.coords[0] or
                            line1.coords[0] == line2.coords[1] or
                            line1.coords[1] == line2.coords[0] or
                            line1.coords[1] == line2.coords[1]
                        )
                        if check_con_end:
                            # print('   line just connect at ends')
                            index2 += 1
                            continue
                        check_ends_here = (
                            intersection.coords[0] == line1.coords[0] or
                            intersection.coords[0] == line1.coords[1]
                        )
                        if check_ends_here:
                            # split line2
                            # print('   splitting line2', index, index2)
                            line2_split = split(line2, intersection)
                            # print(line2_split)
                            elines = np.array((
                                (
                                    line2_split.geoms[0].coords[0][0],
                                    line2_split.geoms[0].coords[0][1],
                                    line2_split.geoms[0].coords[1][0],
                                    line2_split.geoms[0].coords[1][1],
                                ),
                                (
                                    line2_split.geoms[1].coords[0][0],
                                    line2_split.geoms[1].coords[0][1],
                                    line2_split.geoms[1].coords[1][0],
                                    line2_split.geoms[1].coords[1][1],
                                ),
                            ))

                            extra_lines_in = np.vstack((
                                extra_lines_in[0:index2],
                                elines,
                                extra_lines_in[index2+1:, :],
                            ))
                        index2 += 1

                index2 += 1
            index += 1

        # print('done')
        return extra_lines_in

    def crop_lines_to_boundary(self, boundaries, extra_lines, round_dec=4):
        """Take all extra_lines and make sure the either lie completely in the
        boundary region, or crop them to lie on the boundary

        Round coordinates to four decimal points.

        """
        ls_boundary = self._boundaries_to_polygon(boundaries)

        extra_lines_new = []
        for line_coords in extra_lines:
            line = shapely.geometry.LineString(
                (
                    (line_coords[0], line_coords[1]),
                    (line_coords[2], line_coords[3]),
                )
            )
            if ls_boundary.contains(line):
                extra_lines_new += [line_coords]
            else:
                line_crop = ls_boundary.intersection(line)
                extra_lines_new += [
                    [
                        line_crop.coords[0][0],
                        line_crop.coords[0][1],
                        line_crop.coords[1][0],
                        line_crop.coords[1][1],
                    ]
                ]
        extra_lines_new = np.array(extra_lines_new).round(round_dec)
        return extra_lines_new

    def _extra_lines_add_to_boundaries(self, boundaries, extra_lines):
        """Add all start/end points of the extra lines lying on the boundary to
        the boundary, of not already present
        """
        boundary_lines = self._boundaries_to_lines(boundaries)

        points = []
        for (x1, y1, x2, y2) in extra_lines:
            points += [shapely.geometry.Point((x1, y1))]
            points += [shapely.geometry.Point((x2, y2))]

        for point in points:
            index = 0
            while index < len(boundary_lines):

                bline = boundary_lines[index][0]
                btype = boundary_lines[index][1]

                # check if line is crossed by line_cross
                if point.intersects(bline):
                    # now make sure the point is not already start/end point
                    check_endpoints = (
                        point.x == bline.coords[0][0] and
                        point.y == bline.coords[0][1]
                    ) or (
                        point.x == bline.coords[1][0] and
                        point.y == bline.coords[1][1]
                    )
                    if not check_endpoints:
                        print(
                            'POINT {} must be inserted on boundary', point
                        )
                        splits = [
                            [x, btype] for x in split(bline, point).geoms
                        ]
                        boundary_lines = boundary_lines[
                            0:index
                        ] + splits + boundary_lines[
                            index + 1:
                        ]
                        index += 1

                index += 1
            boundaries_new = self._lines_to_crt_boundaries(boundary_lines)

        return boundaries_new

    def gen_mesh_with_polygons(self, boundaries, electrodes, char_lengths=None,
                               polygons=None, additional_lines=None):
        """

        Parameters
        ----------
        additional_lines: None|list
            Add shapely.geometry.LineString lines here that shall be added to
            the extra_lines

        TODO: Externalize the "add-to-boundary" function so we can apply it to
        LineStrings in additional_lines
        """
        ls_boundary = self._boundaries_to_polygon(boundaries)
        boundary_lines = self._boundaries_to_lines(boundaries)

        # fig, ax = plt.subplots()
        # shapely.plotting.plot_polygon(ls_boundary)
        # return fig
        in_extra_lines = []
        # create extra_lines and updated boundaries
        if polygons is None:
            polygons = []
        for polygon in polygons:
            # check if the polygon touches the boundary
            if not ls_boundary.contains_properly(polygon):
                # it does, we need to add all intersection points to the
                # boundaries

                # print('intersetion with boundary')
                # 1. crop polygon to boundary
                poly_int = ls_boundary.intersection(polygon)

                # fig, ax = plt.subplots()
                # shapely.plotting.plot_polygon(ls_boundary)
                # shapely.plotting.plot_polygon(poly_int)

                # 2. find intersection points/lines
                # 3. add points to boundaries
                index = 0
                while index < len(boundary_lines):

                    # check if line is crossed by line_cross
                    line = boundary_lines[index][0]

                    if poly_int.intersects(line):
                        # print('Found a crossing', index)
                        intersection = poly_int.intersection(line)
                        btype = boundary_lines[index][1]

                        if type(intersection) is shapely.geometry.Point:
                            splits = [
                                [x, btype] for x in split(
                                    line, poly_int.intersection(line)).geoms
                            ]
                            boundary_lines = boundary_lines[
                                0:index
                            ] + splits + boundary_lines[
                                index + 1:
                            ]
                            # print(boundary_lines)
                        else:
                            # print('TODO: overlapping lines')

                            # print('total line', line)
                            # print(line.difference(intersection))
                            # print(intersection)

                            # check if the order fits
                            if intersection.coords[0] == line.coords[0]:
                                new_lines = [
                                    [intersection, btype],
                                    [line.difference(intersection), btype],
                                ]
                            else:
                                new_lines = [
                                    [line.difference(intersection), btype],
                                    [intersection, btype],
                                ]
                            boundary_lines = boundary_lines[
                                0:index
                            ] + new_lines + boundary_lines[
                                index + 1:
                            ]
                            # return
                        # advance once to prevent duplicate action
                        index += 1
                    index += 1
                boundaries = self._lines_to_crt_boundaries(boundary_lines)
                # print('---- boundaries')

                # print(boundaries)
                # 4. add lines to extra_lines
                ls_int = explode_linestring(poly_to_linestring(poly_int))

                for line in ls_int:
                    if ls_boundary.contains(line):
                        # use this as extra line
                        # print('GOT EXTRA LINE', line)
                        in_extra_lines += [linestring_to_crt_lines(line)]
            else:
                # print('no intersection')
                out_ls = poly_to_linestring(polygon)
                in_extra_lines += [linestring_to_crt_lines(out_ls)]

        extra_lines_raw = np.vstack(in_extra_lines)

        if additional_lines is not None:
            add_lines = []
            for entry in additional_lines:
                assert isinstance(entry, shapely.geometry.LineString)
                add_lines = linestring_to_crt_lines(entry)

            extra_lines_raw = np.vstack((
                extra_lines_raw,
                add_lines
            ))
        # print('extra lines raw')
        # print(extra_lines_raw)
        extra_lines = self.fix_extra_lines(extra_lines_raw)
        # print(extra_lines)
        # print('boundaries')
        # print(boundaries)

        mesh = self.gen_mesh(
            boundaries,
            electrodes,
            char_lengths=char_lengths,
            extra_lines=extra_lines[0:, :],
        )
        return mesh, extra_lines
