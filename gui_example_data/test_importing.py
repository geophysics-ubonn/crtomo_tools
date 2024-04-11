#!/usr/bin/env python
import io
elem_file_str = io.BytesIO(bytes(open('elem.dat', 'r').read(), 'utf-8'))
elem_file_bytes = io.BytesIO(bytes(open('elem.dat', 'r').read(), 'utf-8'))

elec_file_str = io.BytesIO(bytes(open('elec.dat', 'r').read(), 'utf-8'))
elec_file_bytes = io.BytesIO(bytes(open('elec.dat', 'r').read(), 'utf-8'))

import crtomo
mesh_str = crtomo.crt_grid(elem_file_str, elec_file_str)
mesh_bytes = crtomo.crt_grid(elem_file_bytes, elec_file_bytes)

tdm = crtomo.tdMan(grid=mesh_str)

volt_file_str = io.StringIO(open('volt.dat', 'r').read())
volt_file_bytes = io.BytesIO(bytes(open('volt.dat', 'r').read(), 'utf-8'))
tdm.read_voltages(volt_file_str)
