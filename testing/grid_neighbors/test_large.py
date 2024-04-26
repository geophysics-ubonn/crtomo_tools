#!/usr/bin/env python
import time

import crtomo

mesh = crtomo.crt_grid('large/elem.dat', 'large/elec.dat')

start = time.perf_counter()
neighbors = mesh.element_neighbors_v2
end = time.perf_counter()
print('V2 neighbors took {}s'.format(end - start))
