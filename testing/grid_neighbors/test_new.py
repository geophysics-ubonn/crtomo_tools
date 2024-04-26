#!/usr/bin/env python
import time

import crtomo

mesh = crtomo.crt_grid()
mesh1 = crtomo.crt_grid()

start = time.perf_counter()
neighbors_v1 = mesh.element_neighbors
end = time.perf_counter()
print('V1 neighbors took {}s'.format(end - start))

start = time.perf_counter()
neighbors_v2 = mesh1.element_neighbors_v2
end = time.perf_counter()
print('V2 neighbors took {}s'.format(end - start))

assert len(neighbors_v1) == len(neighbors_v2), "Lengths do not match"
for index, (entry_v1, entry_v2) in enumerate(zip(neighbors_v1, neighbors_v2)):
    # print(index, entry_v1, entry_v2)
    assert set(entry_v1) == set(entry_v2), 'Problem: {} {} {}'.format(
        index, entry_v1, entry_v2
    )
