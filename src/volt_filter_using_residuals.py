#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
import os
import shutil
import datetime

import numpy as np
import pandas as pd
from argparse import ArgumentParser


def handle_cmd_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-t',
        "--top",
        dest="filter_top_nr",
        type=int,
        help="Filter the top N residuals",
        default=0,
    )

    return parser.parse_args()


def read_lastmodfile(directory):
    """
    Return the number of the final inversion result.
    """
    filename = '{0}/exe/inv.lastmod'.format(directory)
    # filename HAS to exist. Otherwise the inversion was not finished
    if not os.path.isfile(filename):
        return None

    linestring = open(filename, 'r').readline().strip()
    linestring = linestring.replace("\n", '')
    linestring = linestring.replace(".mag", '')
    linestring = linestring.replace("../inv/rho", '')
    return linestring


def open_data():
    content = np.loadtxt('mod/volt.dat', skiprows=1)
    return content  # , array


def open_inv():
    num = read_lastmodfile('.')
    content = np.loadtxt('inv/volt' + num + '.dat', skiprows=1)
    return content  # , array


def main():
    args = handle_cmd_args()
    assert args.filter_top_nr > 0, "Nothing to do, -t == 0"

    data = open_data()
    inv = open_inv()
    ab_are_equal = (data[:, 0] - inv[:, 0] == 0).all()
    mn_are_equal = (data[:, 1] - inv[:, 1] == 0).all()
    if not (ab_are_equal and mn_are_equal):
        raise Exception(
            "There is a mismatch between mod/volt.dat and inv/volt*.dat"
        )

    rdata = pd.DataFrame((data[:, 0] / 1e4).astype(int), columns=['a', ])
    rdata['b'] = (data[:, 0] % 1e4).astype(int)
    rdata['m'] = (data[:, 1] / 1e4).astype(int)
    rdata['n'] = (data[:, 1] % 1e4).astype(int)
    rdata['r'] = data[:, 2]
    rdata['pha'] = data[:, 3]
    rdata['ab_pos'] = (rdata['b'] - rdata['a']) / 2 + rdata['a']
    rdata['mn_pos'] = (rdata['n'] - rdata['m']) / 2 + rdata['m']
    rdata['diff_res'] = data[:, 2] - inv[:, 2]
    rdata['diff_pha'] = data[:, 3] - inv[:, 3]
    rdata['diff_res_abs'] = np.abs(rdata['diff_res'])
    rdata['diff_pha_abs'] = np.abs(rdata['diff_pha'])

    rdata = rdata.sort_values('diff_res_abs', ascending=False)

    n = args.filter_top_nr

    filtered_data = rdata.iloc[n:].sort_index()
    filtered_data['ab'] = (
        filtered_data['a'] * 1e4 + filtered_data['b']).astype(int)
    filtered_data['mn'] = (
        filtered_data['m'] * 1e4 + filtered_data['n']).astype(int)

    filtered_data = filtered_data[['ab', 'mn', 'r', 'pha']]

    # make a backup copy
    backup_file = 'mod/volt_{}.dat'.format(
        datetime.datetime.now().strftime('%Y%m%d_%H%M_%S'),
    )
    assert not os.path.isfile(backup_file), \
        "Backup file already exists: {}".format(backup_file)
    print('Moving mod/volt.dat to {}'.format(backup_file))
    shutil.move('mod/volt.dat', backup_file)

    print('Writing filtered mod/volt.dat file')
    with open('mod/volt.dat', 'w') as fid:
        fid.write('{}\n'.format(filtered_data.shape[0]))
        np.savetxt(
            fid,
            filtered_data.values,
            fmt='%i %i %f %f',
        )

    # import IPython
    # IPython.embed()


if __name__ == '__main__':
    main()
