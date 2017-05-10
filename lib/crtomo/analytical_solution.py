r"""Analytical potential distribution over a half-space

For a homogeneous full space, the potential distribution over a point source is
given as:

:math:`\varphi(r) = I \frac{1}{4 \pi \sigma r}`

"""
import numpy as np


def pot_ana(r, rho):
    """Return the analytical potential in distance r over a homogeneous
    half-space
    """
    I = 1.0
    sigma = 1.0 / rho
    phi = np.divide(I, (2.0 * np.pi * sigma * r))
    return phi


def compute_potentials_analytical_hs(grid, configs_raw, rho):
    """Compute the potential superpositions of each current dipole in the
    configurations, using the provided resistivity

    Parameters
    ----------
    grid:
        crt_grid object with loaded FE grid. Used for the electrode positions
    configs_raw: numpy.ndarray
        Nx4 array containing N four-point spreads
    rho: float
        resistivity of half-space

    Returns
    -------
    potentials: list
        List containing N arrays, each of size M (nr of grid nodes)

    """
    potentials = []
    nodes_sorted = grid.nodes['sorted']
    nodes_raw = grid.nodes['sorted']

    for config in configs_raw:
        print('potential configs', config)
        # determine distance of all nodes to both electrodes
        e1_node = grid.get_electrode_node(config[0])
        print('e1_node', e1_node)
        electrode1 = nodes_sorted[e1_node][1:3]
        # electrode1 = nodes_sorted[config[0]][1:3]
        r1 = np.sqrt(
            (nodes_raw[:, 1] - electrode1[0]) ** 2 +
            (nodes_raw[:, 2] - electrode1[1]) ** 2
        )
        # electrode2 = nodes_sorted[config[1]][1:3]
        e2_node = grid.get_electrode_node(config[1])
        print('e2_node', e2_node)
        electrode2 = nodes_sorted[e2_node][1:3]
        r2 = np.sqrt(
            (nodes_raw[:, 1] - electrode2[0]) ** 2 +
            (nodes_raw[:, 2] - electrode2[1]) ** 2
        )
        pot1 = pot_ana(r1, rho)
        pot2 = - pot_ana(r2, rho)
        pot12 = pot1 + pot2
        potentials.append(pot12)
    return potentials


def compute_voltages(grid, configs_raw, potentials_raw):
    """Given a list of potential distribution and corresponding four-point
    spreads, compute the voltages

    Parameters
    ----------
    grid:
        crt_grid object the grid is used to infer electrode positions
    configs_raw: Nx4 array
        containing the measurement configs (1-indexed)
    potentials_raw: list with N entries
        corresponding to each measurement, containing the node potentials of
        each injection dipole.
    """
    # we operate on 0-indexed arrays, config holds 1-indexed values
    # configs = configs_raw - 1
    voltages = []
    for config, potentials in zip(configs_raw, potentials_raw):
        print('config', config)
        e3_node = grid.get_electrode_node(config[2])
        e4_node = grid.get_electrode_node(config[3])
        print(e3_node, e4_node)
        print('pot1', potentials[e3_node])
        print('pot2', potentials[e4_node])
        voltage = potentials[e3_node] - potentials[e4_node]
        voltages.append(voltage)
    return voltages
