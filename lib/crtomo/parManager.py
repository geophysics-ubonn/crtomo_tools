# *-* coding: utf-8 *-*
"""Manage parameter sets for a given grid. A parameter set is a set of values
that correspond to the elements of the grid. Usually this is a resistivity or
phase distribution, but can also be a cumulated sensitivity distribution.
"""
import numpy as np


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
        # we assign indices to each data set stored in the manager. This index
        # should be unique over the life time of each instance. Therefore we
        # increase the counter for each added data set. We also ensure
        # conflicts when data sets are removed.
        self.index = -1

    def _get_next_index(self):
        self.index += 1
        return self.index

    def add_data(self, data):
        """Add data to the parameter set

        Parameters
        ----------
        data: numpy.ndarray
            one or more parameter sets. It must either be 1D or 2D, with the
            first dimension the number of parameter sets (K), and the second
            the number of elements (N): K x N

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

        return_ids = []
        for dataset in subdata:
            cid = self._get_next_index()
            self.parsets[cid] = dataset
            return_ids.append(cid)
        return return_ids
