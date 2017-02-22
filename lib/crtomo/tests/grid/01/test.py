import crtomo.grid as CRGrid


def setup_func():
    pass


def teardown_func():
    pass


def test_loading_of_grid():
    grid = CRGrid.crt_grid()
    grid.load_grid('data/elem.dat', 'data/elec.dat')


def test_loading_of_elem_only():
    grid = CRGrid.crt_grid()
    grid.load_elem_file('data/elem.dat')


def test_number_elements():
    grid = CRGrid.crt_grid()
    grid.load_grid('data/elem.dat', 'data/elec.dat')
    assert grid.nr_of_elements == 74


def test_number_nodes():
    grid = CRGrid.crt_grid()
    grid.load_grid('data/elem.dat', 'data/elec.dat')
    assert grid.nr_of_nodes == 48
