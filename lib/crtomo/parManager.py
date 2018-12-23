# *-* coding: utf-8 *-*
"""Manage parameter sets for a given grid. A parameter set is a set of values
that correspond to the elements of the grid. Usually this is a resistivity or
phase distribution, but can also be a cumulated sensitivity distribution.
"""
import os
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
        # increase the counter for each added data set. We also ensure
        # conflicts when data sets are removed.
        self.index = -1

    def _get_next_index(self):
        self.index += 1
        return self.index

    def add_data(self, data, metadata=None):
        """Add data to the parameter set

        Parameters
        ----------
        data: numpy.ndarray
            one or more parameter sets. It must either be 1D or 2D, with the
            first dimension the number of parameter sets (K), and the second
            the number of elements (N): K x N
        metadata: object, optional
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
                    'Number of values does not match the number of ' +
                    'elements in the grid'
                )

        # now make sure that metadata can be zipped with the subdata
        K = subdata.shape[0]
        if metadata is not None:
            if K > 1:
                if(not isinstance(metadata, (list, tuple)) or
                   len(metadata) != K):
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

    def load_inv_result(self, filename, columns=2):
        """Load one parameter set from a rho*.mag or rho*.pha file produced by
        CRTomo.

        Parameters
        ----------
        filename : string, file path
            Filename to loaded data from
        columns : int or iterable of ints, optional
            column(s) to add to the manager. Defaults to 2 (third column).

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

        pid_list = []
        for column in iterator:
            data = np.loadtxt(filename, skiprows=1)
            pid = self.add_data(data[:, column])
            pid_list.append(pid)

        if len(pid_list) == 1:
            return pid_list[0]
        else:
            return pid_list

    def load_model_from_file(self, filename):
        """Load one parameter set from a file which contains one value per line

        No row is skipped.

        Parameters
        ----------
        filename : string, file path
            Filename to loaded data from

        Returns
        -------
        pid : int
            ID of parameter set
        """
        assert os.path.isfile(filename)
        data = np.loadtxt(filename).squeeze()
        assert len(data.shape) == 1
        pid = self.add_data(data)
        return pid

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
        """Modify parts of a parameter set by setting all parameters located
        in, or touching, the provided :class:`shapely.geometry.Polygon`
        instance.

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

    def extract_points(self, pid, points):
        """Extract values at certain points in the grid from a given parameter
        set. Cells are selected by interpolating the centroids of the cells
        towards the line using a "nearest" scheme.

        Note that data is only returned for the points provided. If you want to
        extract multiple data points along a line, defined by start and end
        point, use the **extract_along_line** function.

        Parameters
        ----------
        pid: int
            The parameter id to extract values from
        points: Nx2 numpy.ndarray
            (x, y) pairs

        Returns
        -------
        values: numpy.ndarray (n x 1)
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
        pid: int
            The parameter id to extract values from
        xy0: tuple
            A tupe with (x,y) start point coordinates
        xy1: tuple
            A tupe with (x,y) end point coordinates
        N: integer, optional
            The number of values to extract along the line (including start and
            end point)

        Returns
        -------
        values: numpy.ndarray (n x 1)
            data values for extracted data points
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

        Returns
        -------
        """
        polygon = shapgeo.Polygon(polygon_points)
        xy = self.grid.get_element_centroids()
        in_poly = []
        for nr, point in enumerate(xy):
            if shapgeo.Point(point).within(polygon):
                in_poly.append(nr)

        values = self.parsets[pid][in_poly]
        return np.array(in_poly), values

