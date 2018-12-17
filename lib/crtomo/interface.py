"""This is meant as a simple interface to CRMod, i.e. the forward operator and
its Jacobian. Nothing more.

It can be used to experiment with alternative inversion strategies outside of
the old FORTRAN code, but bear in mind that the CRMod binary is used to compute
everything. This is slow and involves writing things to disc. If possible use
another, better interfaced forward code such as pygimli.

The plan is to support both resistivity-only and complex (magnitude/phase)
versions.
"""
import numpy as np

import crtomo
from crtomo.grid import crt_grid


class crmod_interface(object):
    def __init__(self, grid, configs):
        assert isinstance(configs, np.ndarray)
        assert isinstance(grid, crt_grid)

        self.grid = grid
        self.configs = configs

    def _get_tdm(self, m):
        m = np.atleast_2d(m)
        tdm = crtomo.tdMan(grid=self.grid)
        tdm.configs.add_to_configs(self.configs)

        pid_mag = tdm.parman.add_data(m[0, :])
        tdm.register_magnitude_model(pid_mag)
        if m.shape[0] == 2:
            pid_pha = tdm.parman.add_data(m[1, :])
        else:
            pid_pha = tdm.parman.add_data(np.zeros(m.shape[1]))
        tdm.register_phase_model(pid_pha)
        return tdm

    def forward_complex(self, log_sigma):
        """Compute a model response, i.e. complex impedances

        Parameters
        ----------
        log_sigma : 1xN or 2xN numpy.ndarray
            Model parameters log sigma, N the number of cells. If first
            dimension is of length one, assume phase values to be zero

        Returns
        -------

        """
        m = 1.0 / np.exp(log_sigma)
        tdm = self._get_tdm(m)
        measurements = tdm.measurements()
        # import IPython
        # IPython.embed()
        # convert R to logR
        measurements[:, 0] = np.log(1.0 / measurements[:, 0])
        return measurements

    def J(self, log_sigma):
        """Return the sensitivity matrix

        Parameters
        ----------

        """
        m = 1.0 / np.exp(log_sigma)
        tdm = self._get_tdm(m)

        tdm.model(
            sensitivities=True,
            # output_directory=stage_dir + 'modeling',
        )

        measurements = tdm.measurements()

        # build up the sensitivity matrix
        sens_list = []
        for config_nr, cids in sorted(
                tdm.assignments['sensitivities'].items()):
            sens_list.append(tdm.parman.parsets[cids[0]])

        sensitivities_lin = np.array(sens_list)
        # now convert to the log-sensitivities relevant for CRTomo and the
        # resolution matrix
        sensitivities_log = sensitivities_lin
        # multiply measurements on first dimension
        measurements_rep = np.repeat(
            measurements[:, 0, np.newaxis],
            sensitivities_lin.shape[1],
            axis=1)
        # sensitivities_log = sensitivities_log * mfit

        # multiply resistivities on second dimension
        m_rep = np.repeat(
            m[np.newaxis, :], sensitivities_lin.shape[0], axis=0
        )

        # eq. 3.41 in Kemna, 2000: notice that m_rep here is in rho, not sigma
        factor = - 1 / (m_rep * measurements_rep)
        sensitivities_log = factor * sensitivities_lin

#         import IPython
#         IPython.embed()

        return sensitivities_log
