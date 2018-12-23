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

import numpy as np
import scipy.sparse

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
    def __init__(self, elem_file=None, elec_file=None):
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

        #   # prepare nodes
        #   nodes_sorted = np.zeros((number_of_nodes, 3), dtype=float)
        #   nodes = np.zeros((number_of_nodes, 3), dtype=float)

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
        if(nodes_raw[:, 0] != list(range(1, nodes_raw.shape[0]))):
            self.header['cutmck'] = True
            print(
                'This grid was sorted using CutMcK. The nodes were resorted!')
        else:
            self.header['cutmck'] = False

        # Rearrange nodes when CutMcK was used.
        if(self.header['cutmck']):
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
        if(self.header['element_infos'][0, 2] == 3):
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
        sizes = sum([len(self.element_data[key]) for key in (11, 12) if
                     self.element_data.get(key, None) is not None])
        self.neighbors = []

        try:
            for i in range(0, sizes):
                self.neighbors.append(int(fid.readline().strip()))
        except Exception as e:
            raise Exception('Not enough neighbors in file')

    def load_elem_file(self, elem_file):
        """

        """
        with open(elem_file, 'r') as fid:
            self._read_elem_header(fid)
            self._read_elem_nodes(fid)
            self._read_elem_elements(fid)
            self._read_elem_neighbors(fid)
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
        with open(output, 'wb') as fid:
            self._write_elem_header(fid)
            self._write_nodes(fid)
            self._write_elements(fid)
            self._write_neighbors(fid)

    def save_elec_file(self, filename):
        with open(filename, 'wb') as fid:
            fid.write(
                bytes(
                    '{0}\n'.format(int(self.electrodes.shape[0])),
                    'utf-8',
                )
            )
            # the + 1 fixes the zero-indexing
            np.savetxt(fid, self.electrodes[:, 0].astype(int) + 1, fmt='%i')

    def _write_neighbors(self, fid):
        np.savetxt(fid, self.neighbors, fmt='%i')

    def _write_elements(self, fid):
        for dtype in self.header['element_infos'][:, 0]:
            for elm in self.element_data[dtype]:
                np.savetxt(fid, np.atleast_2d(elm.nodes), fmt='%i')

    def _write_nodes(self, fid):
        np.savetxt(fid, self.nodes['raw'], fmt='%i %f %f')

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
        np.savetxt(fid, self.header['element_infos'], fmt='%i')

    def load_elec_file(self, elec_file):
        electrode_nodes_raw = np.loadtxt(elec_file, skiprows=1, dtype=int) - 1
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
        if(self.header['cutmck']):
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
        """
        Other Parameters
        ----------------
        plot_electrode_numbers: bool, optional
            Plot electrode numbers in the grid, default: False
        """
        fig, ax = plt.subplots(1, 1)
        self.plot_grid_to_ax(ax, **kwargs)
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
        Nx2 array
            x/z coordinates for all (N) elements
        """
        centroids = np.vstack((
            np.mean(self.grid['x'], axis=1), np.mean(self.grid['z'], axis=1)
        )).T

        return centroids

    def get_electrode_positions(self):
        """Return the electrode positions in an numpy.ndarray

        Returns
        -------
        positions: numpy.ndarray
            Nx2 array, where N is the number of electrodes. The first column
            contains x positions, the second z positions.
        """
        return self.electrodes[:, 1:3]

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
        return self.element_neighbors_data

    def Wm(self):
        """Return the smoothing regularization matrix Wm of the grid

        """
        centroids = self.get_element_centroids()

        Wm = scipy.sparse.csr_matrix(
            (self.nr_of_elements, self.nr_of_elements))
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
                Wm[i, i] += edge_length / distance
                # side diagonals
                Wm[i, j] -= edge_length / distance
        return Wm

    @staticmethod
    def create_surface_grid(nr_electrodes=None, spacing=None,
                            electrodes_x=None,
                            depth=None,
                            left=None,
                            right=None,
                            char_lengths=None,
                            lines=None,
                            debug=False,
                            workdir=None):
        """This is a simple wrapper for cr_trig_create to create simple surface
        grids.

        Parameters
        ----------
        nr_electrodes: int, optional
            the number of surface electrodes
        spacing: float, optional
            the spacing between electrodes, usually in [m], required if nr of
            electrodes is given
        electrodes_x: array, optional
            x-electrode positions can be provided here, e.g., for
            non-equidistant electrode distances
        depth: float, optional
            the depth of the grid. If not given, this is computed as half the
            maximum distance between electrodes
        left: float, optional
            the space allocated left of the first electrode. If not given,
            compute as a fourth of the maximum inter-electrode distance
        right: float, optional
            the space allocated right of the first electrode. If not given,
            compute as a fourth of the maximum inter-electrode distance
        char_lengths: float|list of 4 floats, optional
            characteristic lengths, as used by cr_trig_create
        lines: list of floats, optional
            at the given depths, add horizontal lines in the grid. Note that
            all positive values will be multiplied by -1!
        debug: bool, optional
            default: False. If true, don't hide the output of cr_trig_create
        workdir: string, optional
            if set, use this directory to create the grid. Don't delete files
            afterwards.

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
        if(electrodes_x is None and
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
        else:
            nr_electrodes = len(electrodes_x)
            electrodes = np.hstack((electrodes_x, np.zeros_like(electrodes_x)))

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
            depth = max_distance / 2

        # min/max coordinates of final grid
        minimum_x = minx - left
        maximum_x = maxx + left
        minimum_z = -depth
        maximum_z = 0

        boundary_noflow = 11
        boundary_mixed = 12
        # prepare extra lines
        extra_lines = []
        add_boundary_nodes_left = []
        add_boundary_nodes_right = []

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
            # reverse direction of right nodes
            add_boundary_nodes_left = np.array(add_boundary_nodes_left)[::-1]
            add_boundary_nodes_right = np.array(add_boundary_nodes_right)

        surface_electrodes = np.hstack((
            electrodes, boundary_noflow * np.ones((electrodes.shape[0], 1))
        ))
        # import IPython
        # IPython.embed()
        boundaries = np.vstack((
            (minimum_x, 0, boundary_noflow),
            surface_electrodes,
            (maximum_x, maximum_z, boundary_mixed),
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
                    add_boundary_nodes_left,
                )
            )

        if char_lengths is None:
            char_lengths = [spacing / 3.0, ]

        if workdir is None:
            tempdir_obj = tempfile.TemporaryDirectory()
            tempdir = tempdir_obj.name
        else:
            if not os.path.isdir(workdir):
                os.makedirs(workdir)
            tempdir = workdir

        np.savetxt(tempdir + os.sep + 'electrodes.dat', electrodes)
        np.savetxt(tempdir + os.sep + 'boundaries.dat', boundaries,
                   fmt='%.4f %.4f %i')
        np.savetxt(
            tempdir + os.sep + 'char_length.dat',
            np.atleast_1d(char_lengths)
        )
        if extra_lines:
            np.savetxt(
                tempdir + os.sep + 'extra_lines.dat',
                np.atleast_2d(extra_lines),
                fmt='%.4f %.4f %.4f %.4f'
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
