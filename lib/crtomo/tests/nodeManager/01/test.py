import numpy as np

import crtomo.nodeManager as nM
import crtomo.grid as CRGrid


class test_nodeManager():
    def setup(self):
        grid = CRGrid.crt_grid()
        grid.load_elem_file('tomodir/grid/elem.dat')

        self.nodeMan = nM.NodeMan(grid)

    def test_adding_potentials(self):
        pot1 = np.loadtxt('tomodir/mod/pot/pot01.dat')

        cid = self.nodeMan.add_data(pot1[:, 2])
        assert cid == [0, ]

        cids = self.nodeMan.add_data(pot1[:, 2:4])
        assert cids == [1, 2]

        # we also allow transposed input
        cids = self.nodeMan.add_data(pot1[:, 2:4].T)
        assert cids == [3, 4]

