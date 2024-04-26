#!/usr/bin/env python
import crtomo
import numpy as np
import scipy.sparse

mesh = crtomo.crt_grid()
mesh_large = crtomo.crt_grid(
    '../grid_neighbors/large/elem.dat',
    '../grid_neighbors/large/elec.dat'
)
mesh_large
#mesh = mesh_large
N = mesh.nr_of_nodes
dist_matrix = scipy.sparse.lil_array((N, N))
for element in mesh.elements:
    print('element:', element)
    for a, b in zip((0, 1, 2), (1, 2, 0)):
        #print(element[a], element[b])
        # compute distance between nodes
        distance = np.linalg.norm(
            mesh.nodes['presort'][element[b]][1:3] - mesh.nodes['presort'][element[a]][1:3]
        )
        # print(distance)
        index_a = element[a]
        index_b = element[b]
        dist_matrix[index_a, index_b] = distance
        dist_matrix[index_b, index_a] = distance
import matplotlib.pylab as plt

fig, ax = plt.subplots()
# ax.spy(dist_matrix.toarray())
mesh.plot_grid_to_ax(
    ax
)
fig.tight_layout()
fig.show()
