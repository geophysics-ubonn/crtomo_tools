import os
import shutil
import subprocess

import numpy as np


class CRTomoDecouplingLines():
    """Interface to grid_extralines_gen_decouplings
    """
    def __init__(self):
        pass

    def get_decouplings(self, mesh, dec_lines_raw, return_output=False):
        """
        Parameters
        ----------
        """
        dec_lines = np.atleast_2d(dec_lines_raw)
        pwd = os.getcwd()

        workdir = 'tmp_dec'
        if os.path.isdir(workdir):
            print('Removing old worktree')
            shutil.rmtree(workdir)

        os.makedirs(workdir)
        os.chdir(workdir)
        mesh.save_elem_elec_files()
        np.savetxt('extra_lines.dat', dec_lines)

        try:
            output = subprocess.check_output(
                'grid_extralines_gen_decouplings',
                shell=True,
            )
            decouplings = np.loadtxt('decouplings.dat', skiprows=1)

        except Exception as e:
            print('ERROR')
            print(e)
            decouplings = None
        os.chdir(pwd)
        if return_output:
            return decouplings, output
        return decouplings
