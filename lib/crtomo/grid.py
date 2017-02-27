"""Manage and plot CRMod/CRTomo grids

Examples
--------

>>> import crtomo.grid as CRTGrid
    grid = CRTGrid.crt_grid()
    grid.load_grid('elem.dat', 'elec.dat')

"""
import numpy as np
from .mpl_setup import *


class crt_grid(object):
    """The :class:`crt_grid` holds and manages one Finite-Element grid.

    This class provides only basic plotting routines. Advanced plotting
    routines can be found in XX.

    """
    def __init__(self):
        self.electrodes = None
        self.header = None
        self.nodes = None
        self.elements = None
        self.grid_is_rectangular = None
        self.grid_data = []

        self.props = {
            'electrode_color': 'k',
        }
        self.grid = None
        self.nr_of_elements = None
        self.nr_of_nodes = None
        self.nr_of_electrodes = None

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
            element_infos[element_type, :] = np.fromstring(element_line,
                                                           dtype=int,
                                                           sep='  ')
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

        For completness, we also store the following keys:

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
        if(nodes_raw[0, 0] != 1 or nodes_raw[1, 0] != 2):
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
            except:
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
        except:
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
            np.savetxt(fid, self.electrodes[:, 0].astype(int), fmt='%i')

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
        elec_node_raw = self.electrodes[electrode - 1][0]
        if(self.header['cutmck']):
            elec_node = self.nodes['rev_cutmck_index'][elec_node_raw]
        else:
            elec_node = elec_node_raw - 1
        return int(elec_node)

    def plot_grid_to_ax(self, ax):
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
            )
        ax.set_xlim(self.grid['x'].min(), self.grid['x'].max())
        ax.set_ylim(self.grid['z'].min(), self.grid['z'].max())
        # ax.autoscale_view()
        ax.set_aspect('equal')

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

