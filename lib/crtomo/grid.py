"""Manage and plot CRMod/CRTomo grids

Creating grids
--------------

Grids can be created in various ways:

    * Regular grids can be created using Griev (which is not supported any
      more)
    * Irregular (triangular) grids are created using the command line tool
      'cr_trig_create'
    * the crt_grid class provides some simple wrapper functions for
      cr_trig_create that directory load created grids

Notes
-----

After loading an elem.dat file, i.e., the grid, grid information are exposed
via various structures:

    * the header is stored in the dict self.header::

          >>> self.header.keys()
              dict_keys(
                ['bandwidth', 'element_infos', 'cutmck', 'nr_element_types',
                'nr_nodes']
              )

    * nodes are stored in self.nodes. Various sortings are available. If in
      doubt use self.nodes['presort'].
    * elements are stored in self.elements (as node numbers)


Examples
--------

>>> import crtomo.grid as CRTGrid
    grid = CRTGrid.crt_grid()
    grid.load_grid('elem.dat', 'elec.dat')

>>> # extracting element coordinates in the order of rho.dat file
    import crtomo.grid as CRGrid
    grid = CRGrid.crt_grid(elem_file='elem.dat', elec_file='elec.dat')
    # 1. element (oben links)
    grid.nodes['presort'][grid.elements[0], :]
    # columns: node number - x position - z position
"""
import tempfile
import subprocess
import os
import time
import io

import numpy as np
import scipy.sparse
from scipy.spatial.distance import pdist
import scipy.spatial
import pandas as pd

import crtomo.mpl
plt, mpl = crtomo.mpl.setup()


class crt_grid(object):
    """The :class:`crt_grid` holds and manages one Finite-Element grid.

    This class provides only basic plotting routines. Advanced plotting
    routines can be found in :class:`crtomo.plotManager`.

    Examples
    --------

    >>> import crtomo.grid as CRGrid
        grid = CRGrid(elem_file='elem.dat', elec_file='elec.dat')
        print(grid)
    CRMod/CTRomo grid instance
    number of elements: 2728
    number of nodes: 1440
    number of electrodes: 60
    grid dimsensions:
        X: -7.5 37.5
        Z: -15.0 0.0


    """
    def __init__(self, elem_file=None, elec_file=None, empty=False):
        """

        Parameters
        ----------


        """
        self.electrodes = None
        self.header = None
        self.nodes = None
        self.elements = None
        # store the neighbors (by zeroed index) of each element
        self.element_neighbors_data = None
        # for each neighbor, store the nodes common to both elements
        self.element_neighbors_edges = None
        self.grid_is_rectangular = None
        self.grid_data = []

        self.props = {
            'electrode_color': 'k',
        }
        self.grid = None
        self.nr_of_elements = None
        self.nr_of_nodes = None
        self.nr_of_electrodes = None

        # can hold an KDTree of the element centroids
        self.centroid_tree = None
        # can hold an KDTree of the node coordinates
        self.node_tree = None

        # can hold the distance matrix of the mesh, as a csr_array
        self.distance_matrix = None

        # will be used for caching by .get_element_centroids
        self.centroids = None

        if elem_file is None and not empty:
            # check if elem.dat is present
            if os.path.isfile('elem.dat'):
                elem_file = 'elem.dat'

        if elec_file is None and not empty:
            # check if elec.dat is present
            if os.path.isfile('elec.dat'):
                elec_file = 'elec.dat'

        if elem_file is not None:
            self.load_elem_file(elem_file)

        if elec_file is not None:
            self.load_elec_file(elec_file)

    def __repr__(self):
        output_str = 'CRMod/CTRomo grid instance\n'
        output_str += 'number of elements: {0}\n'.format(self.nr_of_elements)
        output_str += 'number of nodes: {0}\n'.format(self.nr_of_nodes)
        output_str += 'number of electrodes: {0}\n'.format(
            self.nr_of_electrodes
        )
        output_str += 'grid dimsensions: \n'
        xlim, zlim = self.get_minmax()
        output_str += 'X: {0} {1} \n'.format(*xlim)
        output_str += 'Z: {0} {1} \n'.format(*zlim)

        return output_str

    class element:
        def __init__(self):
            self.nodes = np.empty(1)
            self.xcoords = np.empty(1)
            self.zcoords = np.empty(1)

        def center(self):
            xcenter = np.mean(self.xcoords)
            zcenter = np.mean(self.zcoords)
            return (xcenter, zcenter)

        def length_line(self):
            assert len(self.xcoords) == 2
            length = np.sqrt(
                np.sum(
                    np.diff(self.xcoords) ** 2 + np.diff(self.zcoords) ** 2
                )
            )
            return length

        def vector_line(self):
            """Return the vector of the boundary element"""
            assert len(self.xcoords) == 2
            diff_x = self.xcoords[1] - self.xcoords[0]
            diff_z = self.zcoords[1] - self.zcoords[0]
            vec = np.hstack((diff_x, diff_z))
            return vec

        def vector_norm_line(self):
            return self.vector_line() / self.length_line()

        def outer_normal_line(self):
            vector = self.vector_norm_line()
            # element coords are provided counter-clockwise.
            # for outer normal, we need to rotate by 90 degree to the left
            vector[1] *= -1
            # swap x and z coords
            return vector[::-1]

    def add_grid_data(self, data):
        """ Return id
        """
        self.grid_data.append(data)
        return len(self.grid_data) - 1

    def _read_elem_header(self, fid):
        header = {}
        # first header line
        firstline = fid.readline().lstrip()
        nr_of_nodes, nr_of_element_types, bandwidth = np.fromstring(
            firstline, dtype=int, sep='   ')
        header['nr_nodes'] = nr_of_nodes
        header['nr_element_types'] = nr_of_element_types
        header['bandwidth'] = bandwidth

        # read header lines for each element type
        element_infos = np.zeros((nr_of_element_types, 3), dtype=int)
        for element_type in range(0, nr_of_element_types):
            element_line = fid.readline().lstrip()
            element_infos[element_type, :] = np.fromstring(
                element_line,
                dtype=int,
                sep='  ',
            )
        header['element_infos'] = element_infos
        self.header = header

    def _read_elem_nodes(self, fid):
        """ Read the nodes from an opened elem.dat file. Correct for CutMcK
        transformations.

        We store three typed of nodes in the dict 'nodes':

            * "raw" : as read from the elem.dat file
            * "presort" : pre-sorted so we can directly read node numbers from
              a elec.dat file and use them as indices.
            * "sorted" : completely sorted as in the original grid (before any
                       CutMcK)

        For completeness, we also store the following keys:

            * "cutmck_index" : Array containing the indices in "presort" to
              obtain the "sorted" values:
              nodes['sorted'] = nodes['presort'] [nodes['cutmck_index'], :]
            * "rev_cutmck_index" : argsort(cutmck_index)
        """
        nodes = {}

        # read in nodes
        nodes_raw = np.empty((self.header['nr_nodes'], 3), dtype=float)
        for nr in range(0, self.header['nr_nodes']):
            node_line = fid.readline().lstrip()
            nodes_raw[nr, :] = np.fromstring(
                node_line, dtype=float, sep='    ')

        # round node coordinates to 5th decimal point. Sometimes this is
        # important when we deal with mal-formatted node data
        nodes_raw[:, 1:3] = np.round(nodes_raw[:, 1:3], 5)

        # check for CutMcK
        # The check is based on the first node, but if one node was renumbered,
        # so were all the others.
        if (
                ~np.all(
                    nodes_raw[:, 0] == list(
                        range(1, nodes_raw.shape[0] + 1)))):
            self.header['cutmck'] = True
            print(
                'This grid was sorted using CutMcK. The nodes were resorted!')
        else:
            self.header['cutmck'] = False

        # Rearrange nodes when CutMcK was used.
        if (self.header['cutmck']):
            nodes_cutmck = np.empty_like(nodes_raw)
            nodes_cutmck_index = np.zeros(nodes_raw.shape[0], dtype=int)
            for node in range(0, self.header['nr_nodes']):
                new_index = np.where(nodes_raw[:, 0].astype(int) == (node + 1))
                nodes_cutmck[new_index[0], 1:3] = nodes_raw[node, 1:3]
                nodes_cutmck[new_index[0], 0] = new_index[0]
                nodes_cutmck_index[node] = new_index[0]
            # sort them
            nodes_sorted = nodes_cutmck[nodes_cutmck_index, :]
            nodes['presort'] = nodes_cutmck
            nodes['cutmck_index'] = nodes_cutmck_index
            nodes['rev_cutmck_index'] = np.argsort(nodes_cutmck_index)
        else:
            nodes_sorted = nodes_raw
            nodes['presort'] = nodes_raw

        # prepare node dict
        nodes['raw'] = nodes_raw
        nodes['sorted'] = nodes_sorted

        self.nodes = nodes
        self.nr_of_nodes = nodes['raw'].shape[0]

    def _read_elem_elements(self, fid):
        """Read all FE elements from the file stream. Elements are stored in
        the self.element_data dict. The keys refer to the element types:

         *  3: Triangular grid (three nodes)
         *  8: Quadrangular grid (four nodes)
         * 11: Mixed boundary element
         * 12: Neumann (no-flow) boundary element

        """
        elements = {}

        # read elements
        for element_type in range(0, self.header['nr_element_types']):
            element_list = []
            for element_coordinates in range(
                    0, self.header['element_infos'][element_type, 1]):
                element_coordinates_line = fid.readline().lstrip()
                tmp_element = self.element()
                tmp_element.nodes = np.fromstring(element_coordinates_line,
                                                  dtype=int, sep=' ')
                tmp_element.xcoords = self.nodes['presort'][tmp_element.nodes -
                                                            1, 1]
                tmp_element.zcoords = self.nodes['presort'][tmp_element.nodes -
                                                            1, 2]
                element_list.append(tmp_element)
            element_type_number = self.header['element_infos'][element_type, 0]
            elements[element_type_number] = element_list
        self.element_data = elements

    def _prepare_grids(self):
        """
        depending on the type of grid (rectangular or triangle), prepare grids
        or triangle lists

        TODO: We want some nice way of not needing to know in the future if we
              loaded triangles or quadratic elements.
        """
        if (self.header['element_infos'][0, 2] == 3):
            print('Triangular grid found')
            self.grid_is_rectangular = False

            triangles = self.element_data[3]
            triangles = [x.nodes for x in triangles]
            # python starts arrays with 0, but elem.dat with 1
            triangles = np.array(triangles) - 1
            self.elements = triangles
            tri_x = self.nodes['presort'][triangles, 1]
            tri_z = self.nodes['presort'][triangles, 2]
            self.grid = {}
            self.grid['x'] = tri_x
            self.grid['z'] = tri_z

        else:
            print('Rectangular grid found')
            self.grid_is_rectangular = True
            quads_raw = [x.nodes for x in self.element_data[8]]
            quads = np.array(quads_raw) - 1
            self.elements = quads
            quads_x = self.nodes['presort'][quads, 1]
            quads_z = self.nodes['presort'][quads, 2]
            self.grid = {}
            self.grid['x'] = quads_x
            self.grid['z'] = quads_z

            # calculate the dimensions of the grid
            try:
                self.calculate_dimensions()
            except Exception as e:
                e
                self.nr_nodes_x = None
                self.nr_nodes_z = None
                self.nr_elements_x = None
                self.nr_elements_z = None
        self.nr_of_elements = self.grid['x'].shape[0]

    def calculate_dimensions(self):
        """For a regular grid, calculate the element and node dimensions
        """
        x_coordinates = np.sort(self.grid['x'][:, 0])  # first x node
        self.nr_nodes_z = np.where(x_coordinates == x_coordinates[0])[0].size
        self.nr_elements_x = self.elements.shape[0] / (self.nr_nodes_z - 1)
        self.nr_nodes_x = self.nr_elements_x + 1
        self.nr_elements_z = self.nr_nodes_z - 1

    def get_minmax(self):
        """Return min/max x/z coordinates of grid

        Returns
        -------
        x: [float, float]
            min, max values of grid dimensions in x direction (sideways)
        z: [float, float]
            min, max values of grid dimensions in z direction (downwards)

        """
        x_minmax = [np.min(self.grid['x']), np.max(self.grid['x'].max())]
        z_minmax = [np.min(self.grid['z']), np.max(self.grid['z'].max())]
        return x_minmax, z_minmax

    def _read_elem_neighbors(self, fid):
        """Read the boundary-element-neighbors from the end of the file
        """
        # get number of boundary elements
        # types 11 and 12 are boundary elements
        self.neighbors = {}
        try:
            for key in (11, 12):
                if key not in self.element_data.keys():
                    continue
                length = len(self.element_data[key])
                self.neighbors[key] = []
                for i in range(length):
                    self.neighbors[key].append(
                        int(fid.readline().strip())
                    )
        except Exception as e:
            e
            raise Exception('Not enough neighbors in file')

    def load_elem_file(self, elem_file):
        """Load a CRTomo/CRMod elem.dat mesh file from either a file, or from
        stringIO

        """
        if isinstance(elem_file, (io.StringIO, io.BytesIO,)):
            # the StringIO object can directly be used
            fid = elem_file
        else:
            fid = open(elem_file, 'r')
        self._read_elem_header(fid)
        self._read_elem_nodes(fid)
        self._read_elem_elements(fid)
        self._read_elem_neighbors(fid)
        fid.close()

        self._prepare_grids()

    def save_elem_file(self, output):
        """Save elem.dat to file.
        The grid is saved as read in, i.e., with or without applied cutmck.
        If you want to change node coordinates, use self.nodes['raw']

        Parameters
        ----------
        filename: string
            output filename
        """
        if isinstance(output, (io.StringIO, io.BytesIO,)):
            fid = output
        else:
            fid = open(output, 'wb')
        self._write_elem_header(fid)
        self._write_nodes(fid)
        self._write_elements(fid)
        self._write_neighbors(fid)
        if not isinstance(fid, io.BytesIO):
            fid.close()

    def save_elec_file(self, filename):
        if isinstance(filename, (io.StringIO, io.BytesIO,)):
            fid = filename
        else:
            fid = open(filename, 'wb')
        fid.write(
            bytes(
                '{0}\n'.format(int(self.electrodes.shape[0])),
                'utf-8',
            )
        )
        # the + 1 fixes the zero-indexing
        np.savetxt(
            fid, self.electrodes[:, 0].astype(int) + 1, fmt='%i',
            encoding='utf-8',
        )
        if not isinstance(fid, io.BytesIO):
            fid.close()

    def _write_neighbors(self, fid):
        for key in (11, 12):
            if key in self.neighbors:
                np.savetxt(
                    fid, self.neighbors[key], fmt='%i', encoding='utf-8'
                )

    def _write_elements(self, fid):
        for dtype in self.header['element_infos'][:, 0]:
            for elm in self.element_data[dtype]:
                np.savetxt(
                    fid, np.atleast_2d(elm.nodes), fmt='%i', encoding='utf-8')

    def _write_nodes(self, fid):
        np.savetxt(fid, self.nodes['raw'], fmt='%i %f %f', encoding='utf-8')

    def _write_elem_header(self, fid):
        fid.write(
            bytes(
                '{0} {1} {2}\n'.format(
                    self.header['nr_nodes'],
                    self.header['nr_element_types'],
                    self.header['bandwidth']),
                'utf-8',
            )
        )
        np.savetxt(
            fid, self.header['element_infos'], fmt='%i', encoding='utf-8')

    def load_elec_file(self, elec_file):
        if isinstance(elec_file, (io.StringIO, io.BytesIO,)):
            # the StringIO object can directly be used
            fid = elec_file
        else:
            fid = open(elec_file, 'r')
        electrode_nodes_raw = np.loadtxt(fid, skiprows=1, dtype=int) - 1
        fid.close()
        self.electrodes = self.nodes['presort'][electrode_nodes_raw]
        self.nr_of_electrodes = self.electrodes.shape[0]

    def load_grid(self, elem_file, elec_file):
        """Load elem.dat and elec.dat
        """
        self.load_elem_file(elem_file)
        self.load_elec_file(elec_file)

    def get_electrode_node(self, electrode):
        """
        For a given electrode (e.g. from a config.dat file), return the true
        node number as in self.nodes['sorted']
        """
        elec_node_raw = int(self.electrodes[electrode - 1][0])
        if (self.header['cutmck']):
            elec_node = self.nodes['rev_cutmck_index'][elec_node_raw]
        else:
            elec_node = elec_node_raw - 1
        return int(elec_node)

    def plot_grid_to_ax(self, ax, **kwargs):
        """
        Other Parameters
        ---------------
        plot_electrode_numbers: bool, optional
            Plot electrode numbers in the grid, default: False
        """
        all_xz = []
        for x, z in zip(self.grid['x'], self.grid['z']):
            tmp = np.vstack((x, z)).T
            all_xz.append(tmp)
        collection = mpl.collections.PolyCollection(
            all_xz,
            edgecolor='k',
            facecolor='none',
            linewidth=0.4,
        )
        ax.add_collection(collection)
        if self.electrodes is not None:
            ax.scatter(
                self.electrodes[:, 1],
                self.electrodes[:, 2],
                color=self.props['electrode_color'],
                clip_on=False,
            )
        ax.set_xlim(self.grid['x'].min(), self.grid['x'].max())
        ax.set_ylim(self.grid['z'].min(), self.grid['z'].max())
        ax.set_xlabel('x [m]')
        ax.set_ylabel('z [m]')
        # ax.autoscale_view()
        ax.set_aspect('equal')

        if kwargs.get('plot_electrode_numbers', False):
            for nr, xy in enumerate(self.electrodes[:, 1:3]):
                ax.text(
                    xy[0], xy[1],
                    format(nr + 1),
                    bbox=dict(boxstyle='circle', facecolor='red', alpha=0.8)
                )

    def plot_grid(self, **kwargs):
        """Plot the mesh

        Parameters
        ----------

        Other Parameters
        ----------------
        plot_electrode_numbers: bool, optional
            Plot electrode numbers in the grid, default: False

        Returns
        -------
        fig: matplotlib.Figure
            The Figure object
        ax: matplotlib.Axes
            The axes object the mesh was plotted to
        """
        fig, ax = plt.subplots(1, 1)
        self.plot_grid_to_ax(ax, **kwargs)
        fig.tight_layout()
        return fig, ax

    def test_plot(self):
        # play with plot routines
        fig, ax = plt.subplots(1, 1)
        all_xz = []
        for x, z in zip(self.grid['x'], self.grid['z']):
            tmp = np.vstack((x, z)).T
            all_xz.append(tmp)
        collection = mpl.collections.PolyCollection(all_xz, edgecolor='r')
        ax.add_collection(collection)
        ax.scatter(self.electrodes[:, 1], self.electrodes[:, 2])
        ax.autoscale_view()
        ax.set_aspect('equal')

        fig.savefig('test.png', dpi=300)
        return fig, ax

    def get_element_centroids(self):
        """return the central points of all elements

        Returns
        -------
        centroids: numpy.ndarray
            Nx2 array x/z coordinates for all (N) elements
        """
        if self.centroids is None:
            self.centroids = np.vstack((
                np.mean(self.grid['x'], axis=1),
                np.mean(self.grid['z'], axis=1)
            )).T

        return self.centroids

    @staticmethod
    def _get_area_polygon(points_x, points_z):
        """Return the area of a polygon. The node coordinates must be ordered,
        but the direction does not matter - we return the abs-value

        >>> points_x = [4,  4,  8,  8, -4, -4]
        >>> points_z = [6, -4, -4, -8, -8, 6]
        >>> self._get_area_polygon(points_x, points_z)
        ... 128
        """
        area = 0
        j = len(points_x) - 1
        for i in range(len(points_x)):
            area = area + (
                points_x[j] + points_x[i]
            ) * (points_z[j] - points_z[i])
            j = i
        return np.abs(area / 2)

    def get_element_areas(self):
        """return the areas of the elements

        note that this formula is generic, see CRMod code for a simpler one

        Returns
        -------
        areas : numpy.ndarray
        """
        areas = [
            self._get_area_polygon(
                points_x, points_z
            ) for points_x, points_z in zip(self.grid['x'], self.grid['z'])
        ]
        return np.array(areas)

    def get_electrode_positions(self):
        """Return the electrode positions in an numpy.ndarray

        Returns
        -------
        positions: numpy.ndarray
            Nx2 array, where N is the number of electrodes. The first column
            contains x positions, the second z positions.
        """
        return self.electrodes[:, 1:3]

    def get_min_max_electrode_distances(self):
        """Return the minimal and the maximal electrode distance of the grid

        Returns
        -------
        amin : float
            minimal electrode distance
        amax : float
            maximal electrode distance
        """
        distances = pdist(self.get_electrode_positions())
        return distances.min(), distances.max()

    def get_internal_angles(self):
        """Compute all internal angles of the grid

        Returns
        -------
        numpy.ndarray
            NxK array with N the number of elements, and K the number of nodes,
            filled with the internal angles in degrees
        """

        angles = []

        for elx, elz in zip(self.grid['x'], self.grid['z']):
            el_angles = []
            xy = np.vstack((elx, elz))
            for i in range(0, elx.size):
                i1 = (i - 1) % elx.size
                i2 = (i + 1) % elx.size

                a = (xy[:, i] - xy[:, i1])
                b = (xy[:, i2] - xy[:, i])
                # note that nodes are ordered counter-clockwise!
                angle = np.pi - np.arctan2(
                    a[0] * b[1] - a[1] * b[0],
                    a[0] * b[0] + a[1] * b[1]
                )
                el_angles.append(angle * 180 / np.pi)
            angles.append(el_angles)
        return np.array(angles)

    def analyze_internal_angles(self, return_plot=False):
        """Analyze the internal angles of the grid. Angles shouldn't be too
        small because this can cause problems/uncertainties in the
        Finite-Element solution of the forward problem. This function prints
        the min/max values, as well as quantiles, to the command line, and can
        also produce a histogram plot of the angles.

        Parameters
        ----------
        return_plot: bool
            if true, return (fig, ax) objects of the histogram plot

        Returns
        -------
        fig: matplotlib.figure
            figure object
        ax: matplotlib.axes
            axes object

        Examples
        --------

            >>> import crtomo.grid as CRGrid
                grid = CRGrid.crt_grid()
                grid.load_elem_file('elem.dat')
                fig, ax = grid.analyze_internal_angles(Angles)
            This grid was sorted using CutMcK. The nodes were resorted!
            Triangular grid found
            Minimal angle: 22.156368696965796 degrees
            Maximal angle: 134.99337326279496 degrees
            Angle percentile 10%: 51.22 degrees
            Angle percentile 20%: 55.59 degrees
            Angle percentile 30%: 58.26 degrees
            Angle percentile 40%: 59.49 degrees
            Angle percentile 50%: 59.95 degrees
            Angle percentile 60%: 60.25 degrees
            Angle percentile 70%: 61.16 degrees
            Angle percentile 80%: 63.44 degrees
            Angle percentile 90%: 68.72 degrees
            generating plot...
            >>> # save to file with
                fig.savefig('element_angles.png', dpi=300)

        """
        angles = self.get_internal_angles().flatten()

        print('Minimal angle: {0} degrees'.format(np.min(angles)))
        print('Maximal angle: {0} degrees'.format(np.max(angles)))
        # print out quantiles
        for i in range(10, 100, 10):
            print('Angle percentile {0}%: {1:0.2f} degrees'.format(
                i,
                np.percentile(angles, i),
            ))

        if return_plot:
            print('generating plot...')
            fig, ax = plt.subplots(1, 1, figsize=(12 / 2.54, 8 / 2.54))
            ax.hist(angles, int(angles.size / 10))
            ax.set_xlabel('angle [deg]')
            ax.set_ylabel('count')
            fig.tight_layout()
            # fig.savefig('plot_element_angles.jpg', dpi=300)
            return fig, ax

    @property
    def element_neighbors(self):
        if self.element_neighbors_data is not None:
            return self.element_neighbors_data

        if self.centroid_tree is None:
            centroids = self.get_element_centroids()
            self.centroid_tree = scipy.spatial.cKDTree(centroids)

        max_nr_edges = self.header['element_infos'][0, 2]

        A = self.get_element_areas().max()
        # A = pi * r^2
        radius = np.sqrt(A / np.pi)

        # initialize the neighbor array
        self.element_neighbors_data = []
        self.element_neighbors_edges = []

        # print('Looking for neighbors (V2 with cKDTree)')
        time_start = time.perf_counter()
        for nr, element_nodes in enumerate(self.elements):
            neighbors_edges = []
            neighbors_indices = []
            # r = set(self.centroid_tree.query(centroids[nr], max_nr_edges)[1])
            r = set(
                self.centroid_tree.query_ball_point(
                    centroids[nr],
                    2 * radius
                )
            )
            for index in list(r):
                el_test = self.elements[index]
                intersect_nodes = np.intersect1d(element_nodes, el_test)
                if len(intersect_nodes) == 2:
                    neighbors_indices += [index]
                    neighbors_edges += [intersect_nodes]
                # if len(intersect_nodes) in (0, 1, 3):
                #     r.remove(index)
                # neighbors_edges.append(intersect_nodes)
                if len(neighbors_indices) == max_nr_edges:
                    break
            self.element_neighbors_edges.append(neighbors_edges)
            self.element_neighbors_data.append(neighbors_indices)

        # import IPython
        # IPython.embed()
        time_end = time.perf_counter()
        print('elapsed time: {} s'.format(time_end - time_start))
        return self.element_neighbors_data

    @property
    def element_neighbors_old(self):
        """Return a list with element numbers (zero indexed) of neighboring
        elements. Note that the elements are not sorted. No spacial orientation
        can be inferred from the order of neighbors.

        WARNING: This function is slow due to a nested loop. This would be a
        good starting point for further optimizations.

        In order to speed things up, we could search using the raw data, i.e.,
        with CutMcK enabled sorting, and then restrict the loops to 2x the
        bandwidth (before - after).

        While not being returned, this function also sets the variable
        self.element_neighbors_edges, in which the common nodes with each
        neighbor are stored.

        Returns
        -------
        neighbors : list
            a list (length equal to nr of elements) with neighboring elements

        Examples
        --------


        """
        if self.element_neighbors_data is not None:
            return self.element_neighbors_data

        max_nr_edges = self.header['element_infos'][0, 2]

        # initialize the neighbor array
        self.element_neighbors_data = []
        self.element_neighbors_edges = []

        # determine neighbors
        print('Looking for neighbors')
        time_start = time.time()
        for nr, element_nodes in enumerate(self.elements):
            # print('element {0}/{1}'.format(nr + 1, self.nr_of_elements))
            # print(element_nodes)
            neighbors = []
            neighbors_edges = []  # store the edges to this neighbor
            for nr1, el in enumerate(self.elements):
                # we look for elements that have two nodes in common with this
                # element
                intersection = np.intersect1d(element_nodes, el)
                if intersection.size == 2:
                    neighbors.append(nr1)
                    neighbors_edges.append(intersection)
                    # stop if we reached the maximum number of possible edges
                    # this saves us quite some loop iterations
                    if len(neighbors) == max_nr_edges:
                        break
            self.element_neighbors_data.append(neighbors)
            self.element_neighbors_edges.append(neighbors_edges)
        time_end = time.time()
        print('elapsed time: {} s'.format(time_end - time_start))
        return self.element_neighbors_data

    def Wm(self):
        r"""Return the smoothing regularization matrix Wm of the grid

        See PhD Thesis Roland Blaschek, eq. 3.48ff

        .. math::

            \Psi_m = ||\underline{\underline{W}}_m \underline{m}||^2_2\\
            = \sum_{j=1}^M \sum_{i=nb(j)} \alpha_{r_ij} |\frac{m_j -
            m_i}{\Delta c_{ij}|^2 \Delta b_{ij} \Delta c_{ij}

        Note that the anisotropic regularization parameter :math:`\alpha` is
        not implemented yet.

        i and j are swapped in the implementation.

        See Fig 3.5 in the thesis.

        """
        centroids = self.get_element_centroids()

        Wm = scipy.sparse.lil_matrix(
            (self.nr_of_elements, self.nr_of_elements)
        )
        # Wm = np.zeros((self.nr_of_elements, self.nr_of_elements))
        for i, nb in enumerate(self.element_neighbors):
            for j, edges in zip(nb, self.element_neighbors_edges[i]):
                # side length
                edge_coords = self.nodes['presort'][edges][:, 1:]
                edge_length = np.linalg.norm(
                    edge_coords[1, :] - edge_coords[0, :]
                )
                distance = np.linalg.norm(centroids[i] - centroids[j])

                # main diagonal
                Wm[i, i] += np.sqrt(edge_length / distance)
                # side diagonals
                Wm[i, j] -= np.sqrt(edge_length / distance)
        return Wm

    def Wm_mgs(self, m, beta):
        r"""Return the MGS regularization matrix Wm of the grid

        See PhD Thesis Roland Blaschek, eq. 3.50

        Note that alpha and f are set 1

        Parameters
        ----------
        m : numpy.ndarray
            model parameters to compute MGS matrix for
        beta : float
            MGS beta parameter

        Returns
        -------
        Wm : scipy.sparse.csr_matrix
            Sparse MGS matrix
        """
        assert m is not None
        assert beta is not None
        centroids = self.get_element_centroids()

        Wm = scipy.sparse.csr_matrix(
            (self.nr_of_elements, self.nr_of_elements)
        )
        for j, nb in enumerate(self.element_neighbors):
            for i, edges in zip(nb, self.element_neighbors_edges[j]):
                # side length
                edge_coords = self.nodes['presort'][edges][:, 1:]
                # b_ij
                edge_length = np.linalg.norm(
                    edge_coords[1, :] - edge_coords[0, :]
                )
                # c_ij
                distance = np.linalg.norm(centroids[j] - centroids[i])

                # |model difference|
                m_diff = np.abs(m[i] - m[j])

                term = np.sqrt(
                    edge_length / distance * 1 / (
                        (m_diff / distance) ** 2 + beta ** 2
                    )
                )

                # main diagonal
                Wm[j, j] += term
                # side diagonals
                Wm[j, i] -= term
        return Wm

    @staticmethod
    def create_surface_grid(nr_electrodes=None,
                            spacing=None,
                            electrodes_x=None,
                            electrodes_z=None,
                            depth=None,
                            left=None,
                            right=None,
                            char_lengths=None,
                            lines=None,
                            internal_lines=None,
                            debug=False,
                            workdir=None,
                            force_neumann_only=False):
        """This is a simple wrapper for cr_trig_create to create simple surface
        grids.

        Electrode and boundary positions are rounded to the third digit.

        Parameters
        ----------
        nr_electrodes : int, optional
            the number of surface electrodes
        spacing : float, optional
            the spacing between electrodes, usually in [m], required if nr of
            electrodes is given
        electrodes_x : array, optional
            x-electrode positions can be provided here, e.g., for
            non-equidistant electrode distances
        electrodes_z : array, optional
            z-electrode positions can be provided here, e.g., for
            non-equidistant electrode distances. Only useful in combination
            with electrodes_x
        depth : float, optional
            the depth of the grid, relative to the minimum z-value.
            If not given, this is computed as half the
            maximum distance between electrodes
        left : float, optional
            the space allocated left of the first electrode. If not given,
            compute as a fourth of the maximum inter-electrode distance
        right : float, optional
            the space allocated right of the first electrode. If not given,
            compute as a fourth of the maximum inter-electrode distance
        char_lengths : float|list of 4 floats, optional
            characteristic lengths, as used by cr_trig_create
        lines: list of floats, optional
            at the given depths, add horizontal lines in the grid. Note that
            all positive values will be multiplied by -1!
        internal_lines : list of 4-tuple
            extra lines to add to the grid.  Important: These lines must NOT
            touch the outer edge of the grid (this is not supported by this
            function, but can be accomplished by manually building the grid)
        debug : bool, optional
            default: False. If true, don't hide the output of cr_trig_create
        workdir : string, optional
            if set, use this directory to create the grid. Don't delete files
            afterwards.
        force_neumann_only : bool, optional
            sometimes we want to use only Neumann boundary conditions. Setting
            this switch to True will force all boundaries to Neumann. Use only
            if you know what you are doing! default: False

        Returns
        -------
        grid: :class:`crtomo.grid.crt_grid` instance
            the generated grid

        Examples
        --------
        >>> from crtomo.grid import crt_grid
        >>> grid = crt_grid.create_surface_grid(40, spacing=0.25, depth=5,
        ...     left=2, right=2, char_lengths=[0.1, 0.5, 0.1, 0.5],
        ...     lines=[0.4, 0.8], debug=False, workdir=None)
        >>> import pylab as plt
        >>> fig, ax = plt.subplots()
        >>> grid.plot_grid_to_ax(ax)

        """
        # check if all required information are present
        if (electrodes_x is None and
           (nr_electrodes is None or spacing is None)):
            raise Exception(
                'You must provide either the parameter "electrodes_" or ' +
                'the parameters "nr_electrodes" AND "spacing"'
            )

        if electrodes_x is None:
            electrodes = np.array(
                [(x, 0.0) for x in np.arange(0.0, nr_electrodes)]
            )
            electrodes[:, 0] = electrodes[:, 0] * spacing
            electrodes = np.round(electrodes, 3)
        else:
            # we have individual electrode positions
            nr_electrodes = len(electrodes_x)
            electrodes = np.vstack(
                (electrodes_x, np.zeros_like(electrodes_x))
            ).T
            if electrodes_z is not None:
                electrodes[:, 1] = electrodes_z

        max_distance = np.abs(
            np.max(electrodes[:, 0]) - np.min(electrodes[:, 0])
        )
        minx = electrodes[:, 0].min()
        maxx = electrodes[:, 0].max()

        if left is None:
            left = max_distance / 4
        if right is None:
            right = max_distance / 4
        if depth is None:
            depth = np.abs(np.min(electrodes[:, 1]) - max_distance / 2)

        # min/max coordinates of final grid
        minimum_x = minx - left
        maximum_x = maxx + left
        minimum_z = np.min(electrodes[:, 1]) - depth
        # maximum_z = 0

        boundary_noflow = 12
        boundary_mixed = 11

        if force_neumann_only:
            boundary_mixed = 12

        # prepare extra lines
        extra_lines = []
        add_boundary_nodes_left = []
        add_boundary_nodes_right = []

        if electrodes_x is not None and lines is not None:
            raise Exception(
                'Extra lines not tested with individual electrode positions'
            )

        if lines is not None:
            lines = np.array(lines)
            lines[np.where(np.array(lines) < 0)] *= -1
            lines = sorted(lines)
            for line_depth in lines:
                extra_lines.append(
                    (minimum_x, -line_depth, maximum_x, -line_depth)
                )
                add_boundary_nodes_left.append(
                    (minimum_x, -line_depth, boundary_mixed)
                )
                add_boundary_nodes_right.append(
                    (maximum_x, -line_depth, boundary_mixed)
                )
            # reverse direction of nodes on the left side of the grid
            add_boundary_nodes_left = np.array(add_boundary_nodes_left)[::-1]
            add_boundary_nodes_right = np.array(add_boundary_nodes_right)
            print(add_boundary_nodes_left)
            print(add_boundary_nodes_right)

        if internal_lines is not None:
            extra_lines = extra_lines + internal_lines

        surface_electrodes = np.hstack((
            electrodes, boundary_noflow * np.ones((electrodes.shape[0], 1))
        ))

        # start assembling the boundaries here
        boundaries = np.vstack((
            # left corner
            (minimum_x, surface_electrodes[0, 1], boundary_noflow),
            # electrodes
            surface_electrodes,
            # right corner
            (maximum_x, surface_electrodes[-1, 1], boundary_mixed),
        ))

        if len(add_boundary_nodes_right) != 0:
            boundaries = np.vstack((
                boundaries,
                add_boundary_nodes_right,
            ))

        boundaries = np.vstack((
            boundaries,
            (maximum_x, minimum_z, boundary_mixed),
            (minimum_x, minimum_z, boundary_mixed),
        ))
        if len(add_boundary_nodes_left) != 0:
            boundaries = np.vstack(
                (
                    boundaries,
                    add_boundary_nodes_left,
                )
            )

        if char_lengths is None:
            assert spacing is not None, \
                "you must provide a spacing parameter or the char_lengths"
            char_lengths = [spacing / 3.0, ]
        else:
            assert isinstance(char_lengths, (float, int)) or len(
                    char_lengths) in [1, 4], \
                "char_lengths must be either of size 1 or size 4"

        if workdir is None:
            tempdir_obj = tempfile.TemporaryDirectory()
            tempdir = tempdir_obj.name
        else:
            if not os.path.isdir(workdir):
                os.makedirs(workdir)
            tempdir = workdir

        np.savetxt(
            tempdir + os.sep + 'electrodes.dat', electrodes,
            fmt='%.3f %.3f',
            encoding='utf-8',
        )
        np.savetxt(
            tempdir + os.sep + 'boundaries.dat', boundaries,
            fmt='%.3f %.3f %i',
            encoding='utf-8',
        )
        np.savetxt(
            tempdir + os.sep + 'char_length.dat',
            np.atleast_1d(char_lengths),
            encoding='utf-8',
        )
        if extra_lines:
            np.savetxt(
                tempdir + os.sep + 'extra_lines.dat',
                np.atleast_2d(extra_lines),
                fmt='%.3f %.3f %.3f %.3f',
                encoding='utf-8',
            )
        pwd = os.getcwd()
        os.chdir(tempdir)
        try:
            if debug:
                subprocess.call(
                    'cr_trig_create grid',
                    shell=True,
                )
            else:
                subprocess.check_output(
                    'cr_trig_create grid',
                    shell=True,
                    # stdout=subprocess.STDOUT,
                    # stderr=subprocess.STDOUT,
                )
        except subprocess.CalledProcessError as e:
            print('there was an error generating the grid')
            print(e.returncode)
            print(e.output)
            import shutil
            shutil.copytree(tempdir, pwd + os.sep + 'GRID_FAIL')
            exit()
        finally:
            os.chdir(pwd)
        grid = crt_grid(
            elem_file=tempdir + os.sep + 'grid' + os.sep + 'elem.dat',
            elec_file=tempdir + os.sep + 'grid' + os.sep + 'elec.dat',
        )
        if workdir is None:
            tempdir_obj.cleanup()

        return grid

    def get_element_indices_along_line(self, p0, p1, N):
        points = np.array(
            [(x, y) for x, y in zip(
                np.linspace(p0[0], p1[0], N), np.linspace(p0[1], p1[1], N)
            )]
        )
        centroids = self.get_element_centroids()

        if self.centroid_tree is None:
            self.centroid_tree = scipy.spatial.cKDTree(centroids)

        element_indices = self.centroid_tree.query(points)[1]
        return element_indices

    def get_element_indices_within_rectangle(self, xmin, xmax, zmin, zmax):
        """Return the indices of all elements whose center is located within
        the rectangle defined by the parameters.

        The indices can then be used, e.g., to select values from inversion
        results.

        Parameters
        ----------
        xmin : float
            Minimum x coordinate of accepted elements
        xmax : float
            Maximum x coordinate of accepted elements
        zmin : float
            Minimum z coordinate of accepted elements
        zmax : float
            Maximum z coordinate of accepted elements

        Returns
        -------
        indices : np.array
            Array with indices (zero-indexed)
        """
        centroids = self.get_element_centroids()
        indices_list = []
        for nr, (x, z) in enumerate(centroids):
            if x >= xmin and x <= xmax and z >= zmin and z <= zmax:
                indices_list.append(nr)
        return np.array(indices_list)

    @staticmethod
    def interpolate_grid(ingrid, outgrid, data, method='nearest'):
        """
        Function for interpolating data from one grid to another, using the
        cell midpoints as datapoint locations. Standard method for
        interpolating is set to nearest-neighbour.

        Parameters
        ----------
        ingrid : :class:`crtomo.grid.crt_grid` instance
            CRT grid that matches the input data
        outgrid : :class:`crtomo.grid.crt_grid` instance
            CRT grid to interpolate to
        data : pandas.dataframe
            input data that matches the input grid (in pandas dataframe format)
        method : interpolation method, optional
            Standard interpolation method is nearest-neighbour ('nearest').
            Other possible methods are 'linear' and 'cubic'.

        Returns
        -------
        interpolated data : pandas.dataframe
            returned dataframe with interpolated data
        """

        midpoints_in = ingrid.get_element_centroids()
        midpoints_out = outgrid.get_element_centroids()

        interpolated_data = pd.DataFrame()
        for i in list(data):
            interpolated_data[i] = scipy.interpolate.griddata(
                midpoints_in, data[i], midpoints_out, method=method)
        print(interpolated_data)
        return interpolated_data

    def reverse_node_order(self, element_type):
        """Reverses the order of the nodes of all elements of a given type.
        This can be used to fix CRTomo/CRMod errors regarding the computation
        of the determinant during FE system compilation.

        Use only if you know what you are doing!

        Parameters
        ----------
        element_type : int
            Element type to change. Usually 3 or 4 (triangles or quads)

        """
        assert element_type in self.element_data, \
            "element type {} not registered Available types :{}".format(
                element_type, self.element_data.keys()
            )
        for i in range(len(self.element_data[element_type])):
            self.element_data[
                element_type
            ][i].nodes = self.element_data[
                element_type
            ][i].nodes[::-1]

            self.element_data[
                element_type
            ][i].xcoords = self.element_data[
                element_type
            ][i].xcoords[::-1]

            self.element_data[
                element_type
            ][i].zcoords = self.element_data[
                element_type
            ][i].zcoords[::-1]

    def get_distance_matrix(self):
        if self.distance_matrix is not None:
            return self.distance_matrix

        N = self.nr_of_nodes
        self.distance_matrix = scipy.sparse.lil_array((N, N))
        for element in self.elements:
            # works only for triangles!
            for a, b in zip((0, 1, 2), (1, 2, 0)):
                index_a = element[a]
                index_b = element[b]

                coords_a = self.nodes['presort'][element[a]][1:3]
                coords_b = self.nodes['presort'][element[b]][1:3]

                # compute distance between nodes
                distance = np.linalg.norm(
                    coords_a - coords_b
                )

                self.distance_matrix[index_a, index_b] = distance
                self.distance_matrix[index_b, index_a] = distance
        return self.distance_matrix

    def get_node_tree(self):
        if self.node_tree is not None:
            return self.node_tree
        self.node_tree = scipy.spatial.KDTree(
            self.nodes['presort'][:, 1:3]
        )
        return self.node_tree

    def find_nearest_node(self, coords, return_node_coords=True):
        node_tree = self.get_node_tree()
        result = node_tree.query(coords, 1)[1]
        if return_node_coords:
            return result, self.nodes['presort'][result][1:3]
        return result

    def determine_path_along_nodes(self, start_coordinate, end_coordinate):
        """

        """
        start, (sx, sy) = self.find_nearest_node(start_coordinate, True)
        end, (ex, ey) = self.find_nearest_node(end_coordinate, True)

        path_dist, path_pred = scipy.sparse.csgraph.shortest_path(
            self.get_distance_matrix(),
            directed=False,
            return_predecessors=True,
            unweighted=False,
        )
        path_nodes = []
        path_pred[start]

        # determine the nodes of the path
        N = self.nr_of_elements
        index = end
        path_nodes += [end]
        for i in range(N):
            next_index = path_pred[start][index]
            if next_index > 0:
                path_nodes += [path_pred[start][index]]
                index = next_index
            else:
                break

        # determine elements adjacent to the path
        el_pairs = []
        for i in range(len(path_nodes) - 1):
            a1 = path_nodes[i]
            a2 = path_nodes[i+1]
            relevant_elements = np.intersect1d(
                np.where(np.any(self.elements == a1, axis=1))[0],
                np.where(np.any(self.elements == a2, axis=1))[0]
            )

            if len(relevant_elements) == 2:
                el_pairs += [np.hstack(([a1, a2], relevant_elements))]
        return np.array(el_pairs)

    def get_polygon_from_file(self, filename):
        pass
