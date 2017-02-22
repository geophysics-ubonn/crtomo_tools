import numpy as np

import crtomo.parManager as pM
import crtomo.grid as CRGrid


class test_parManager():
    def setup(self):
        grid = CRGrid.crt_grid()
        grid.load_elem_file('tomodir/grid/elem.dat')

        self.parman = pM.ParMan(grid)

    def test_adding_potentials(self):
        par = np.loadtxt('tomodir/inv/rho02.mag', skiprows=1)

        cid = self.parman.add_data(par[:, 2])
        assert cid == [0, ]
