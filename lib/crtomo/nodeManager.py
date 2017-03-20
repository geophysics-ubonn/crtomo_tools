# *-* coding: utf-8 *-*
"""Manage node value sets for a given grid. Node data assigns a numeric value
to each node of a grid. This could, for example, be potential distributions or
sensitivities for single files.
"""
import numpy as np


class NodeMan(object):
    """manage one or more node value sets for a given
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
        # we store the node value sets in here
        self.nodevals = {}
        # here we can store metadata for a node value set
        self.metadata = {}
        # we assign indices to each data set stored in the manager. This index
        # should be unique over the life time of each instance. Therefore we
        # increase the counter for each added data set. We also ensure
        # conflicts when data sets are removed.
        self.index = -1

    def _get_next_index(self):
        self.index += 1
        return self.index

    def add_data(self, data):
        """Add data to the node value sets

        Parameters
        ----------
        data: numpy.ndarray
            one or more node value sets. It must either be 1D or 2D, with the
            first dimension the number of parameter sets (K), and the second
            the number of elements (Z): K x Z

        Examples
        --------

        >>> # suppose that grid is a fully initialized grid oject with 50 nodes
            nodeman = NodeMan(grid)
            #
            one_data_set = np.ones(50)
            cid = nodeman.add_data(one_data_set)
            print(nodeman.parsets[cid])
            two_data_sets = np.ones((2, 50))
            cids = nodeman.add_data(two_data_sets)
            print(cids)
        [0, ]
        [1, 2]

        """
        subdata = np.atleast_2d(data)

        # we try to accommodate transposed input
        if subdata.shape[1] != self.grid.nr_of_nodes:
            if subdata.shape[0] == self.grid.nr_of_nodes:
                subdata = subdata.T
            else:
                raise Exception(
                    'Number of values does not match the number of ' +
                    'nodes in the grid {0} grid nodes vs {1} data'.format(
                        self.grid.nr_of_nodes, subdata.shape,
                    )
                )

        return_ids = []
        for dataset in subdata:
            cid = self._get_next_index()
            self.nodevals[cid] = dataset.copy()
            return_ids.append(cid)

        if len(return_ids) == 1:
            return return_ids[0]
        else:
            return return_ids
