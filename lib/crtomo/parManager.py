# *-* coding: utf-8 *-*
"""Manage parameter sets for a given grid. A parameter set is a set of values
that correspond to the elements of the grid. Usually this is a resistivity or
phase distribution, but can also be a cumulated sensitivity distribution.
"""
import os
from collections.abc import Iterable

from scipy.stats import multivariate_normal
import scipy.interpolate as spi
import numpy as np
import shapely.geometry as shapgeo


class ParMan(object):
    """manage one or more parameter sets for a given
    :class:`crtomo.grid.crt_grid` object
    """
    def __init__(self, grid_obj):
        """
        Parameters
        ----------
        grid_obj: crtomo.grid.crt_grid
            The FE grid that the parameter sets refer to
        """
        # check if grid is already initialized
        if grid_obj.grid is None:
            raise Exception('Grid object is not initialized!')
        self.grid = grid_obj
        # we store the parameter sets in here
        self.parsets = {}
        self.metadata = {}
        # we assign indices to each data set stored in the manager. This index
        # should be unique over the life time of each instance. Therefore we
        # increase the counter for each added data set. This way we also
        # prevent conflicts when data sets are removed.
        self.index = -1

    def _get_next_index(self):
        self.index += 1
        return self.index

    def reset(self):
        """Resets the ParMan instance. This process deletes all data and
        metadata, and resets the index variable
        """
        self.index = -1
        del (self.parsets)
        self.parsets = {}
        del (self.metadata)
        self.metadata = {}

    def add_data(self, data, metadata=None):
        """Add data to the parameter set

        Parameters
        ----------
        data : numpy.ndarray
            one or more parameter sets. It must either be 1D or 2D, with the
            first dimension the number of parameter sets (K), and the second
            the number of elements (N): K x N
        metadata : object, optional
            the provided object will be stored in in the metadata dict and can
            be received with the ID that is returned. If multiple (K) datasets
            are added at ones, provide a list of objects with len K.

        Returns
        -------
        int, ID
            ID which can be used to access the parameter set

        Examples
        --------

        >>> # suppose that grid is a fully initialized grid oject with 100
            # elements
            parman = ParMan(grid)
            #
            one_data_set = np.ones(100)
            cid = parman.add_data(one_data_set)
            print(parman.parsets[cid])
            two_data_sets = np.ones((2, 100))
            cids = parman.add_data(two_data_sets)
            print(cids)
        [0, ]
        [1, 2]

        """
        subdata = np.atleast_2d(data)

        # we try to accommodate transposed input
        if subdata.shape[1] != self.grid.nr_of_elements:
            if subdata.shape[0] == self.grid.nr_of_elements:
                subdata = subdata.T
            else:
                raise Exception(
                    'Number of values does not match the number of '
                    'elements in the grid'
                )

        # now make sure that metadata can be zipped with the subdata
        K = subdata.shape[0]
        if metadata is not None:
            if K > 1:
                if (not isinstance(
                   metadata, (list, tuple)) or len(metadata) != K):
                    raise Exception('metadata does not fit the provided data')
            else:
                # K == 1
                metadata = [metadata, ]

        if metadata is None:
            metadata = [None for i in range(0, K)]

        return_ids = []
        for dataset, meta in zip(subdata, metadata):
            cid = self._get_next_index()
            self.parsets[cid] = dataset
            self.metadata[cid] = meta
            return_ids.append(cid)

        if len(return_ids) == 1:
            return return_ids[0]
        else:
            return return_ids

    def load_from_rho_file(self, filename):
        """Convenience function that loads two parameter sets from a rho.dat
        file, as used by CRMod for forward resistivity/phase models.

        Parameters
        ----------
        filename: string, file path
            filename of rho.dat file

        Returns
        -------
        cid_mag: int
            ID of magnitude parameter set
        cid_phase: int
            ID of phase parameter set

        """
        data = np.loadtxt(filename, skiprows=1)
        cid_mag = self.add_data(data[:, 0])
        cid_pha = self.add_data(data[:, 1])
        return cid_mag, cid_pha

    def load_inv_result(self, filename, columns=2, is_log10=False):
        """Load one parameter set from a rho*.mag or rho*.pha file produced by
        CRTomo.

        Parameters
        ----------
        filename : str, file path
            Filename to loaded data from
        columns : int or iterable of ints, optional
            column(s) to add to the manager. Defaults to 2 (third column).
        is_log10 : bool, optional
            If set to True, assume values to be in log10 and convert the
            imported values to linear before importing

        Returns
        -------
        pid : int or list of ints
            ID(s) of parameter set

        """
        assert os.path.isfile(filename)
        try:
            iterator = iter(columns)
        except TypeError:
            # not iterable
            iterator = [columns, ]

        data = np.loadtxt(filename, skiprows=1)
        pid_list = []
        for column in iterator:
            if is_log10:
                data[:, column] = 10 ** data[:, column]
            pid = self.add_data(data[:, column])
            pid_list.append(pid)

        if len(pid_list) == 1:
            return pid_list[0]
        else:
            return pid_list

    def load_model_from_file(self, filename, columns=0):
        """Load one parameter set from a file which contains one value per line

        No row is skipped.

        Parameters
        ----------
        filename : string, file path
            Filename to loaded data from
        columns : int or iterable of ints, optional
            column(s) to add to the manager. Defaults to 0 (first column)

        Returns
        -------
        pid : int or list of ints
            ID(s) of parameter set
        """
        try:
            iterator = iter(columns)
        except TypeError:
            # not iterable
            iterator = [columns, ]

        assert os.path.isfile(filename)
        data = np.loadtxt(filename).squeeze()
        # make sure data is NxC, where C is the number of datasets (1-?)
        if len(data.shape) == 1:
            data = data[:, np.newaxis]
        # check that all columns can be added
        assert data.shape[1] > np.max(columns)

        pid_list = []
        for column in iterator:
            pid = self.add_data(data[:, column])
            pid_list.append(pid)

        if len(pid_list) == 1:
            return pid_list[0]
        else:
            return pid_list

    def load_from_sens_file(self, filename):
        """Load real and imaginary parts from a sens.dat file generated by
        CRMod

        Parameters
        ----------
        filename: string
            filename of sensitivity file

        Returns
        -------
        nid_re: int
            ID of real part of sensitivities
        nid_im: int
            ID of imaginary part of sensitivities
        """
        sens_data = np.loadtxt(filename, skiprows=1)
        nid_re = self.add_data(sens_data[:, 2])
        nid_im = self.add_data(sens_data[:, 3])
        return nid_re, nid_im

    def save_to_rho_file(self, filename, cid_mag, cid_pha=None):
        """Save one or two parameter sets in the rho.dat forward model format

        Parameters
        ----------
        filename: string (file path)
            output filename
        cid_mag: int
            ID of magnitude parameter set
        cid_pha: int, optional
            ID of phase parameter set. If not set, will be set to zeros.

        """
        mag_data = self.parsets[cid_mag]
        if cid_pha is None:
            pha_data = np.zeros(mag_data.shape)
        else:
            pha_data = self.parsets[cid_pha]

        with open(filename, 'wb') as fid:
            fid.write(
                bytes(
                    '{0}\n'.format(self.grid.nr_of_elements),
                    'utf-8',
                )
            )
            np.savetxt(
                fid,
                np.vstack((
                    mag_data,
                    pha_data,
                )).T,
                fmt='%f %f'
            )

    def add_empty_dataset(self, value=1):
        """Create an empty data set. Empty means: all elements have the same
        value.

        Parameters
        ----------
        value: float, optional
            which value to assign all element parameters. Default is one.
        """
        subdata = np.ones(self.grid.nr_of_elements) * value
        pid = self.add_data(subdata)
        return pid

    def _clean_pid(self, pid):
        """if pid is a number, don't do anything. If pid is a list with one
        entry, strip the list and return the number. If pid contains more than
        one entries, do nothing.
        """
        if isinstance(pid, (list, tuple)):
            if len(pid) == 1:
                return pid[0]
            else:
                return pid
        return pid

    def modify_area(self, pid, xmin, xmax, zmin, zmax, value):
        """Modify the given dataset in the rectangular area given by the
        parameters and assign all parameters inside this area the given value.

        Partially contained elements are treated as INSIDE the area, i.e., they
        are assigned new values.

        Parameters
        ----------
        pid: int
            id of the parameter set to modify
        xmin: float
            smallest x value of the area to modify
        xmax: float
            largest x value of the area to modify
        zmin: float
            smallest z value of the area to modify
        zmin: float
            largest z value of the area to modify
        value: float
            this value is assigned to all parameters of the area

        Examples
        --------

        >>> import crtomo.tdManager as CRtdm
            tdman = CRtdm.tdMan(
                    elem_file='GRID/elem.dat',
                    elec_file='GRID/elec.dat',
            )
            pid = tdman.parman.add_empty_dataset(value=1)
            tdman.parman.modify_area(
                    pid,
                    xmin=0,
                    xmax=2,
                    zmin=-2,
                    zmin=-0.5,
                    value=2,
            )
            fig, ax = tdman.plot.plot_elements_to_ax(pid)
            fig.savefig('out.png')

        """
        area_polygon = shapgeo.Polygon(
            ((xmin, zmax), (xmax, zmax), (xmax, zmin), (xmin, zmin))
        )
        self.modify_polygon(pid, area_polygon, value)

    def modify_polygon(self, pid, polygon, value):
        """Modify parts of a parameter set by modifying all elements within a
        provided :class:`shapely.geometry.Polygon` instance. Hereby, an element
        is modified if its center lies within the polygon.

        Parameters
        ----------
        pid: int
            id of parameter set to vary
        polygon: :class:`shapely.geometry.Polygon` instance
            polygon that determines the area to modify
        value: float
            value that is assigned to all elements in the polygon

        Examples
        --------
        >>> import shapely.geometry
            polygon = shapely.geometry.Polygon((
                (2, 0), (4, -1), (2, -1)
            ))
            tdman.parman.modify_polygon_centroids(pid, polygon, 3)

        """
        centroids = self.grid.get_element_centroids()
        # now determine elements in area
        elements_in_area = []
        for nr, centroid in enumerate(centroids):
            if polygon.contains(shapgeo.Point(centroid)):
                elements_in_area.append(nr)
        # change the values
        pid_clean = self._clean_pid(pid)
        self.parsets[pid_clean][elements_in_area] = value

    def modify_polygon_old(self, pid, polygon, value):
        """Modify parts of a parameter set by setting all parameters located
        in, or touching, the provided :class:`shapely.geometry.Polygon`
        instance.

        WARNING: This implementation often leads to ragged borders in the
        selected polygons. Use the new modify_polygon function!

        Parameters
        ----------
        pid: int
            id of parameter set to vary
        polygon: :class:`shapely.geometry.Polygon` instance
            polygon that determines the area to modify
        value: float
            value that is assigned to all elements in the polygon

        Examples
        --------
        >>> import shapely.geometry
            polygon = shapely.geometry.Polygon((
                (2, 0), (4, -1), (2, -1)
            ))
            tdman.parman.modify_polygon(pid, polygon, 3)

        """
        # create grid polygons
        grid_polygons = []
        for x, z in zip(self.grid.grid['x'], self.grid.grid['z']):
            coords = [(a, b) for a, b in zip(x, z)]
            grid_polygons.append(
                shapgeo.Polygon(coords)
            )

        # now determine elements in area
        elements_in_area = []
        for nr, element in enumerate(grid_polygons):
            if polygon.contains(element):
                elements_in_area.append(nr)
            elif polygon.equals(element):
                elements_in_area.append(nr)
            elif polygon.crosses(element):
                elements_in_area.append(nr)
                # only take crossed elements with at least A % overlap
                # int_area = polygon.intersect(element).area
                # print('overlap: ',
                #       int_area,
                #       element.area,
                #       element.area / int_area
                #       )

        # change the values
        pid_clean = self._clean_pid(pid)
        self.parsets[pid_clean][elements_in_area] = value

    def modify_pixels(self, pid, pixels, new_value):
        """Replace the value of a given pixel by a new one.

        Parameters
        ----------
        pid : int
            Parset id
        pixel : int|itearble
            Pixel index (zero-indexed)
        new_value : float
            New value that is assigned to the pixel
        """
        if not isinstance(pixels, Iterable):
            pixels = [pixels, ]
        for pixel in pixels:
            self.parsets[pid][pixel] = new_value

    def extract_points(self, pid, points):
        """Extract values at certain points in the grid from a given parameter
        set. Cells are selected by interpolating the centroids of the cells
        towards the line using a "nearest" scheme.

        Note that data is only returned for the points provided. If you want to
        extract multiple data points along a line, defined by start and end
        point, use the **extract_along_line** function.

        Parameters
        ----------
        pid : int
            The parameter id to extract values from
        points : Nx2 numpy.ndarray
            (x, y) pairs

        Returns
        -------
        values : numpy.ndarray (n x 1)
            data values for extracted data points
        """
        xy = self.grid.get_element_centroids()
        data = self.parsets[pid]

        iobj = spi.NearestNDInterpolator(xy, data)
        values = iobj(points)
        return values

    def extract_along_line(self, pid, xy0, xy1, N=10):
        """Extract parameter values along a given line.

        Parameters
        ----------
        pid : int
            The parameter id to extract values from
        xy0 : tuple
            A tupe with (x,y) start point coordinates
        xy1 : tuple
            A tupe with (x,y) end point coordinates
        N : integer, optional
            The number of values to extract along the line (including start and
            end point). Default: 10

        Returns
        -------
        values : numpy.ndarray (n x 3)
            data values for extracted data points. First column: x-coordinates,
            second column: z-coordinates, third column: extracted values
        """
        assert N >= 2
        xy0 = np.array(xy0).squeeze()
        xy1 = np.array(xy1).squeeze()
        assert xy0.size == 2
        assert xy1.size == 2

        # compute points
        points = [(x, y) for x, y in zip(
            np.linspace(xy0[0], xy1[0], N), np.linspace(xy0[1], xy1[1], N)
        )]
        result = self.extract_points(pid, points)

        results_xyv = np.hstack((
            points,
            result[:, np.newaxis]
        ))
        return results_xyv

    def extract_polygon_area(self, pid, polygon_points):
        """Extract all data points whose element centroid lies within the given
        polygon.

        Parameters
        ----------
        pid : int
            The parameter id to extract values from
        polygon_points : list of (x,y) floats
            list of points that form the polygon

        Returns
        -------
        elements_in_polygon_array : numpy.ndarray
            list with elements numbers (zero-indexed, starting from 0) which
            lie within the polygon
        values : list
            the data values corresponding to the elements in
            elements_in_polygon
        """
        polygon = shapgeo.Polygon(polygon_points)
        xy = self.grid.get_element_centroids()
        elements_in_polygon = []

        for nr, point in enumerate(xy):
            if shapgeo.Point(point).within(polygon):
                elements_in_polygon.append(nr)

        values = self.parsets[pid][elements_in_polygon]
        elements_in_polygon_array = np.array(elements_in_polygon)

        return elements_in_polygon_array, values

    def create_parset_with_gaussian_anomaly(
            self, center, width, max_value, background):
        """

        Parameters
        ----------
        center : [float, float]
            Center of the anomaly
        width : float| [float, float]
            The spatial width of the anomaly (x/z directions). Equivalent to
            the standard deviations of the underlying distribution.
        max_value : float
            The maximum value that the anomaly is normalized to
        background : float
            Background value

        Returns
        -------
        pid : int
            ID of newly created parameter set

        """
        pid = self.add_empty_dataset(background)
        self.add_gaussian_anomaly_to_parset(
            pid, center, np.sqrt(width), max_value
        )
        return pid

    def add_gaussian_anomaly_to_parset(
            self, pid, center, width, max_value):
        """

        Parameters
        ----------
        pid : int
            Id of parameter set
        center : [float, float]
            Center of the anomaly
        width : float| [float, float]
            The spatial width of the anomaly (x/z directions). Equivalent to
            the standard deviations of the underlying distribution.
        max_value : float
            The maximum value that the anomaly is normalized to

        """
        xy = self.grid.get_element_centroids()
        # generates a 2D Gaussian distribution with
        covariances = np.array(width) ** 2
        rv = multivariate_normal(
            center,
            covariances,
        )
        grid_values = rv.pdf(xy)

        if isinstance(width, (list, tuple)):
            assert len(width) == 2, "need either one number or two"
            std_x = width[0]
            std_z = width[1]
        else:
            std_x = width
            std_z = width
        # peak value of stretched normal distribution
        peak_x = 1 / (np.sqrt(2 * np.pi) * std_x)
        peak_z = 1 / (np.sqrt(2 * np.pi) * std_z)
        # resulting 2D peak
        peak_value = peak_x * peak_z
        grid_values /= peak_value
        grid_values *= max_value
        self.parsets[pid] = self.parsets[pid] + grid_values

    def add_gaussian_line(
                self, pid,
                p0,
                length,
                rotation,
                spacing,
                anomaly_peak,
                anomaly_std,
            ):
        xpos = np.arange(p0[0], length, step=spacing)
        zpos = np.ones_like(xpos) * p0[1]

        # rotate
        xr = xpos * np.cos(
            rotation * np.pi / 180) - zpos * np.sin(rotation * np.pi / 180)
        zr = zpos * np.cos(
            rotation * np.pi / 180) + xpos * np.sin(rotation * np.pi / 180)

        for x, z in zip(xr, zr):
            self.add_gaussian_anomaly_to_parset(
                pid,
                [x, z],
                width=anomaly_std,
                max_value=anomaly_peak,
            )

        return xr, zr

    def add_2d_cos_anomaly_line(
            self, pid, p0, anomaly_width, anomaly_height, peak_value,
            area='only_one_y',
            only_one_y_line=True, only_one_x_line=False, whole_mesh=False):
        """Add one line of cos(x)cos(y) anomalies to a given parameter set. The
        wavelength refers to half a period, with the maximum of the anomaly
        centered on p0=[x0, z0].

        Parameters
        ----------
        whole_mesh : bool, optional
            If True, then fill the whole mesh with a cos(x)cos(y) pattern.


        """
        coords = self.grid.get_element_centroids()
        norm = 2 * np.pi

        wavelength_x = anomaly_width * 2
        wavelength_z = anomaly_height * 2

        if area == 'all':
            indices = np.arange(0, coords.shape[0]).astype(int)
        elif area == 'only_one_y':
            boundary_y_min = p0[1] - (anomaly_height / 2)
            boundary_y_max = p0[1] + (anomaly_height / 2)
            # restrict to to within a rectangular area
            indices = np.where(
                (coords[:, 1] > boundary_y_min) &
                (coords[:, 1] < boundary_y_max)
            )
        elif area == 'only_one_x':
            boundary_x_min = p0[0] - (anomaly_width / 2)
            boundary_x_max = p0[0] + (anomaly_width / 2)
            # restrict to to within a rectangular area
            indices = np.where(
                (coords[:, 0] > boundary_x_min) &
                (coords[:, 0] < boundary_x_max)
            )

        anomaly = np.cos(
            (coords[indices, 0] - p0[0]) * norm / wavelength_x
        ) * np.cos(
            (coords[indices, 1] - p0[1]) * norm / wavelength_z
        ) * peak_value

        paradd = np.zeros(coords.shape[0])
        paradd[indices] += anomaly.squeeze()

        self.parsets[pid] += paradd

    def add_checkerboard_pattern(
            self, pid, p0, anomaly_width, anomaly_height, peak_value,
            ):
        """

        Note that if p0 is larger than a few anomaly sizes its possible that
        the checkerboard pattern will only be applied to parts of the mesh!

        """
        xlims, zlims = self.grid.get_minmax()

        x_positions = np.arange(
            p0[0],
            np.abs(xlims[1] - xlims[0]) + 4 * anomaly_width, anomaly_width * 2
        ) - (
            anomaly_width * 2
        ) * np.ceil(np.abs(xlims[0] - p0[0]) / (anomaly_width * 2))

        depth_sign = np.sign(zlims[0] - zlims[1])
        z_positions = np.arange(
            p0[1],
            (zlims[0] - zlims[1]) +
            depth_sign * 4 * anomaly_height, depth_sign * anomaly_height * 2
        )
        print(z_positions)
        offset = (anomaly_height * 2) * np.ceil(np.abs(zlims[0] - p0[1]) / (
            anomaly_height * 2))
        print(offset)

        coords = self.grid.get_element_centroids()
        paradd = np.zeros(coords.shape[0])

        for pos_x in x_positions:
            for pos_z in z_positions:
                indices = np.where(
                    (coords[:, 0] >= pos_x) &
                    (coords[:, 0] <= pos_x + anomaly_width) &
                    (coords[:, 1] <= pos_z) &
                    (coords[:, 1] >= (pos_z - anomaly_height))
                )
                paradd[indices] = peak_value

        self.parsets[pid] += paradd

    def _com_data_trafo_mode(self, subdata, mode):
        if mode == "none":
            # just take the absolute values
            s = np.abs(subdata)
        elif mode == "log10":
            # log10 values of absolutes
            s = np.abs(np.log10(np.abs(subdata)))
        elif mode == "sqrt":
            # square roots of absolutes
            s = np.sqrt(np.abs(subdata))
        return s

    def center_of_mass_value(self, pid, mode='log10'):
        """Compute the center of mass value of a given parameter set.
        """
        centroids = self.grid.get_element_centroids()

        # compute the integrative value
        subdata = self.parsets[pid]
        s = self._com_data_trafo_mode(subdata, mode)

        com_x = np.sum(centroids[:, 0] * s) / np.sum(s)
        com_y = np.sum(centroids[:, 1] * s) / np.sum(s)
        # to check: do we need to norm here?

        return [com_x, com_y]

    def center_of_mass_value_multiple(self, pid_list, mode='none'):
        print('center_of_mass_value_multiple')
        centroids = self.grid.get_element_centroids()
        values = np.vstack(
            [self.parsets[index] for index in pid_list]
        )
        s = self._com_data_trafo_mode(values, mode)
        xy = np.dot(s, centroids)

        s_sum = np.sum(s, axis=1)

        s_sum_stacked = np.tile(s_sum, (2, 1))
        s_sum_T = s_sum_stacked.T

        com = xy / s_sum_T
        return com
        # import IPython
        # IPython.embed()
