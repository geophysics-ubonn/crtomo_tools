#!/usr/bin/env python
import numpy as np


def main():
    print('Loading boundaries.dat file')
    boundaries = np.loadtxt('boundaries.dat')
    # translate
    offset_x = np.min(boundaries[:, 0])
    offset_y = np.min(boundaries[:, 1])

    width = boundaries[:, 0].max() - boundaries[:, 0].min()
    height = boundaries[:, 1].max() - boundaries[:, 1].min()

    path_d = 'M '
    for (x, y, rtype) in boundaries:
        # note reversed y-coordinates due to svg-coordinate origin in the upper
        # left corner
        path_d += ' {:.4f},{:.4f}'.format(x - offset_x, -y + offset_y + height)
    path_d += ' Z'

    filename = 'out.svg'

    with open(filename, 'w') as fid:

        fid.write(
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
        )

        fid.write('<svg\n')
        fid.write('width="{}mm"\n'.format(width))
        fid.write('height="{}mm"\n'.format(height))
        fid.write('viewBox="0 0 {} {}"\n'.format(width, height))
        fid.write('crtomo_offset_x="{}"\n'.format(offset_x))
        fid.write('crtomo_offset_y="{}"\n'.format(offset_y))
        fid.write('crtomo_width="{}"\n'.format(width))
        fid.write('crtomo_height="{}"\n'.format(height))
        fid.write('version="1.1"\n')
        fid.write('id="svg5"\n')
        fid.write('  xmlns="http://www.w3.org/2000/svg"\n')
        fid.write('  xmlns:svg="http://www.w3.org/2000/svg">\n')
        fid.write(' <defs\n')
        fid.write('    id="defs2" />\n')

        fid.write('<g\n')
        fid.write('id="boundary">\n')
        fid.write('<path\n')
        fid.write(
            'style="fill:none;stroke:#000000;stroke-width:0.264583px;'
        )
        fid. write(
            'stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1"\n'
        )
        fid.write('d="{}"\n'.format(path_d))
        fid.write('id="mesh_outline" />\n')
        fid.write('</g>\n')
        fid.write('</svg>\n')

    print('Output .svg file written to: {}'.format(filename))
