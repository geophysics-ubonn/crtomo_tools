#!/usr/bin/env python
# *-* coding: utf-8 *-*
"""Test element plotting functions of the plotManager

Note that this is not (yet) a nose testing module because we do not want to
slow down the unit tests

Todo
----

* color handling
* vmin/vmax handling
* alpha values for color?

"""
import numpy as np
import pylab as plt

import crtomo.plotManager as pM


class plot_nodes():
    def setup(self):
        self.plotman = pM.plotManager(
            elem_file='tomodir/grid/elem.dat',
            elec_file='tomodir/grid/elec.dat',
        )
        # add some node data
        self.plotman.parman.add_data(
            np.loadtxt('tomodir/inv/rho02.mag', skiprows=1)[:, 2]
        )

    def test_elements(self):
        fig, ax = plt.subplots()
        self.plotman.plot_elements_to_ax(
            ax,
            0,
            config={

            },
        )
        fig.savefig('nodes_contour1.jpg', dpi=300)
        fig.savefig('nodes_contour1.png', dpi=300)


if __name__ == '__main__':
    obj = plot_nodes()
    obj.setup()
    obj.test_elements()
