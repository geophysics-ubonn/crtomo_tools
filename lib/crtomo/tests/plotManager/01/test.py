import crtomo.parManager as pM
import crtomo.nodeManager as nM
import crtomo.grid as CRGrid
import crtomo.plotManager as plotMan


class test_plotManager():
    def setup(self):
        grid = CRGrid.crt_grid()
        grid.load_elem_file('tomodir/grid/elem.dat')
        grid.load_elec_file('tomodir/grid/elec.dat')
        self.grid = grid

    def test_initialization(self):
        # provide a grid instance
        plotman = plotMan.plotManager(
            grid=self.grid
        )
        assert plotman.grid is not None

        plotman = plotMan.plotManager(
            elem_file='tomodir/grid/elem.dat',
            elec_file='tomodir/grid/elec.dat',
        )
        assert plotman.grid is not None

        # test node manager
        plotman = plotMan.plotManager(
            grid=self.grid,
        )
        assert plotman.nodeman is not None

        nm = nM.NodeMan(self.grid)
        plotman = plotMan.plotManager(
            grid=self.grid,
            nm=nm,
        )
        assert plotman.nodeman is not None

        # test par manager
        plotman = plotMan.plotManager(
            grid=self.grid,
        )
        assert plotman.parman is not None

        pm = pM.ParMan(self.grid)
        plotman = plotMan.plotManager(
            grid=self.grid,
            pm=pm,
        )
        assert plotman.nodeman is not None
