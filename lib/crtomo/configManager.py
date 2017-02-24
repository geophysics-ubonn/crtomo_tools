# *-* coding: utf-8 *-*
"""Manage measurement configurations
"""
import itertools

import numpy as np


class ConfigManager(object):
    """The class`ConfigManager` manages four-point measurement configurations.
    Measurements can be saved/loaded from CRMod/CRTomo files, and new
    configurations can be created.
    """

    def __init__(self):
        # store the configs as a Nx4 numpy array
        self.configs = None
        # store measurements in a list of size N arrays
        self.measurements = []
        # global counter for measurements
        self.meas_counter = - 1

    def _get_next_index(self):
        self.meas_counter += 1
        return self.meas_counter

    def add_measurements(self, measurements):
        """Add new measurements

        Parameters
        ----------
        measurements: numpy.ndarray
            one or more measurement sets. It must either be 1D or 2D, with the
            first dimension the number of measurement sets (K), and the second
            the number of measurements (N): K x N

        """
        subdata = np.atleast_2d(measurements)

        if self.configs is None:
            raise Exception(
                'must read in configuration before measurements can be stored'
            )

        # we try to accommodate transposed input
        if subdata.shape[1] != self.configs.shape[0]:
            if subdata.shape[0] == self.configs.shape[0]:
                subdata = subdata.T
            else:
                raise Exception(
                    'Number of measurements does not match number of configs'
                )

        return_ids = []
        for dataset in subdata:
            cid = self._get_next_index()
            self.measurements[cid] = dataset.copy()
            return_ids.append(cid)
        return return_ids

    def _crmod_to_abmn(self, configs):
        """convert crmod-style configurations to a Nx4 array
        """
        A = configs[:, 0] % 1e4
        B = np.floor(configs[:, 0] / 1e4).astype(int)
        M = configs[:, 1] % 1e4
        N = np.floor(configs[:, 1] / 1e4).astype(int)
        ABMN = np.hstack((A, B, M, N))
        return ABMN

    def load_crmod_config(self, filename):
        """Load a CRMod configuration file
        """
        with open(filename, 'r') as fid:
            nr_of_configs = int(fid.readline().strip())
            configs = np.loadtxt(fid)
            if nr_of_configs != configs.shape[0]:
                raise Exception(
                    'indicated number of measurements does not equal ' +
                    'to actual number of measurements'
                )
            ABMN = self._crmod_to_abmn(configs)
            self.configs = ABMN

    def load_crmod_volt(self, filename):
        """Load a CRMod measurement file (commonly called volt.dat)

        Parameters
        ----------
        filename: string
            path to filename

        Returns
        -------
        list
            list of measurement ids
        """
        with open(filename, 'r') as fid:
            nr_of_configs = int(fid.readline().strip())
            measurements = np.loadtxt(fid)
            if nr_of_configs != measurements.shape[0]:
                raise Exception(
                    'indicated number of measurements does not equal ' +
                    'to actual number of measurements'
                )
        ABMN = self._crmod_to_abmn(measurements[:, 0:2])
        if self.configs is None:
            self.configs = ABMN
        else:
            # check that configs match
            if not np.all(ABMN == self.configs):
                raise Exception(
                    'previously stored configurations do not match new ' +
                    'configurations'
                )

        # add data
        cid_mag = self.add_measurements(measurements[:, 2])
        cid_pha = self.add_measurements(measurements[:, 3])
        return [cid_mag, cid_pha]

    def write_crmod_config(self, filename):
        """Write the configurations to a configuration file in the CRMod format
        """

        with open(filename, 'w') as fid:
            fid.write('{0}\n'.format(self.configs.shape[0]))

            ABMN = np.vstack((
                self.configs[:, 0] * 1e4 + self.configs[:, 1],
                self.configs[:, 2] * 1e4 + self.configs[:, 3],
            )).T
            np.savetxt(fid, ABMN, fmt='%i %i')

    def gen_dipole_dipole(
            self,
            N, skip, step=1, nr_voltage_dipoles=10,
            skipv=0):
        """Generate dipole-dipole configurations

        Parameters
        ----------
        N: int
            number of electrodes
        skip: int
            number of electrode positions that are skipped between electrodes
            of a given dipole
        step: int
            steplength between subsequent current dipoles. A steplength of 0
            will produce increments by one, i.e., 3-4, 4-5, 5-6 ...
        nr_voltage_dipoles: int
            the number of voltage dipoles to generate for each  current
            injection dipole
        skipv: int
            steplength between subsequent voltage dipoles. A steplength of 0
            will produce increments by one, i.e., 3-4, 4-5, 5-6 ...

        """
        configs = []
        # current dipoles
        for a in range(0, N - (3 * skip) - 3, step):
            b = a + skip + 1
            nr = 0
            # potential dipoles
            for m in range(b + 1 + skipv, N - skip - 1, step):
                nr += 1
                if nr > nr_voltage_dipoles:
                    continue
                n = m + skip + 1
                quadpole = np.array((a, b, m, n)) + 1
                configs.append(quadpole)
        configs = np.array(configs)
        return configs

    def get_gradient(self, N, skip=0, step=1, vskip=0, vstep=1):
        """
        Parameters
        ---------
        N: int
            number of electrodes
        skip: int
            distance between current electrodes
        step: int
            steplength between subsequent current dipoles
        vskip: int
            distance between voltage electrodes
        vstep: int
            steplength between subsequent voltage dipoles
        """
        quadpoles = []
        for a in range(1, N - skip, step):
            b = a + skip + 1
            for m in range(a + 1, b - vskip - 1, vstep):
                n = m + vskip + 1
                quadpoles.append((a, b, m, n))
        return np.array(quadpoles)

# old functions that must be vetted for usefulness


def full_voltages(injections, N):
    """

    """
    all_quadpoles = []
    for idipole in injections:
        # sort current electrodes and convert to array indices
        I = np.sort(idipole) - 1

        # voltage electrodes
        velecs = range(1, N + 1)

        # remove current electrodes
        del(velecs[I[1]])
        del(velecs[I[0]])

        # permutate remaining
        voltages = itertools.permutations(velecs, 2)

        voltages = np.array([x for x in voltages])
        voltages_sorted = np.sort(voltages, axis=1)
        sorted_u = 1e4 * voltages_sorted[:, 0] + voltages_sorted[:, 1]
        duplicates, ret_index = np.unique(sorted_u, return_index=True)
        sorted_voltages = voltages_sorted[ret_index, :]

        current = np.resize(np.array(idipole), sorted_voltages.shape)
        quadpoles = np.hstack((current, sorted_voltages))
        all_quadpoles.append(quadpoles)

    quadpoles = np.vstack(all_quadpoles)
    return quadpoles


def create_current_dipoles(x0, skip, dx, N):
    """
    Parameters
    ----------
    x0: starting electrodes
    skip: skip of dipole
    dx: how many electrodes to skip between injection dipoles
    N: total number of electrodes
    """
    starting_elecs = [x0]
    while(True):
        i = starting_elecs[-1] + dx
        if i <= N:
            starting_elecs.append(i)
        else:
            break
    N1 = N + 1
    dipoles_raw = [(x, x + skip + 1) for x in starting_elecs]
    dipoles0 = [(x[0] % N1, x[1] % N1) for x in dipoles_raw]

    dipoles = []
    for x in dipoles0:
        if(x[0] == 0):
            x0 = 1
        else:
            x0 = x[0]

        if(x[1] == 0):
            x1 = 1
        else:
            x1 = x[1]

        dipoles.append((x0, x1))
    return dipoles


def syscal_write_electrode_coords(fid, spacing, N):
    fid.write('# X Y Z\n')
    for i in range(0, N):
        fid.write('{0} {1} {2} {3}\n'.format(i + 1, i * spacing, 0, 0))


def syscal_write_quadpoles(fid, quadpoles):
    fid.write('# A B M N\n')
    for nr, quadpole in enumerate(quadpoles):
        fid.write(
            '{0} {1} {2} {3} {4}\n'.format(
                nr, quadpole[0], quadpole[1], quadpole[2], quadpole[3]))


def get_dipole_dipole(
        N, skip, step=1, nr_voltage_dipoles=10,
        skipv=0):
    configs = []
    for a in range(0, N - (3 * skip) - 3, step):
        b = a + skip + 1
        nr = 0
        for m in range(b + 1 + skipv, N - skip - 1, step):
            nr += 1
            if nr > nr_voltage_dipoles:
                continue
            n = m + skip + 1
            quadpole = np.array((a, b, m, n)) + 1
            configs.append(quadpole)
    configs = np.array(configs)
    return configs


def get_gradient(N, skip=0, step=1, vskip=0, vstep=1):
    """
    Parameter
    ---------
    N: number of electrodes
    skip: distance between current electrodes
    step: steplength between subsequent current dipoles
    vskip: distance between voltage electrodes
    vstep: steplength between subsequent voltage dipoles
    """
    quadpoles = []
    for a in range(1, N - skip, step):
        b = a + skip + 1
        for m in range(a + 1, b - vskip - 1, vstep):
            n = m + vskip + 1
            quadpoles.append((a, b, m, n))
    return np.array(quadpoles)


def save_to_config_txt(filename, liste):
    quadrupoles = np.vstack(liste)
    print('Number of measurements: ', quadrupoles.shape[0])

    with open(filename, 'w') as fid:
        syscal_write_electrode_coords(fid, 1, 48)
        syscal_write_quadpoles(fid, quadrupoles)


def create_gradient_configs():
    liste = []
    liste.append(
        get_gradient(
            N=48,
            skip=6,
            step=2,
            vskip=0,
            vstep=1,
        )
    )
    liste.append(
        get_gradient(
            N=48,
            skip=10,
            step=2,
            vskip=1,
            vstep=2,
        )
    )

    save_to_config_txt('config_grad.txt', liste)


def create_dd_configs():
    liste = []

    liste.append(
        get_dipole_dipole(
            N=48,
            skip=1,
            step=1,
            nr_voltage_dipoles=16,
            skipv=1,
        )
    )

    liste.append(
        get_dipole_dipole(
            N=48,
            skip=3,
            step=1,
            nr_voltage_dipoles=60,
            skipv=1,
        )
    )

    for x in liste[1][0:30]:
        print(x)

    liste.append(
        get_dipole_dipole(
            N=48,
            skip=7,
            step=1,
            nr_voltage_dipoles=5,
            skipv=1,
        )
    )

    save_to_config_txt('configs_dd.txt', liste)


def create_dd_configs_v2():
    liste = []

    liste.append(
        get_dipole_dipole(
            N=48,
            skip=1,
            step=1,
            nr_voltage_dipoles=16,
            skipv=1,
        )
    )

    liste.append(
        get_dipole_dipole(
            N=48,
            skip=3,
            step=1,
            nr_voltage_dipoles=60,
            skipv=1,
        )
    )

    liste.append(
        get_dipole_dipole(
            N=48,
            skip=5,
            step=1,
            nr_voltage_dipoles=60,
            skipv=1,
        )
    )

    liste.append(
        get_dipole_dipole(
            N=48,
            skip=7,
            step=1,
            nr_voltage_dipoles=5,
            skipv=1,
        )
    )

    save_to_config_txt('configs_dd_v2.txt', liste)
    # write_crmod_config('configs_v2.dat', liste)
