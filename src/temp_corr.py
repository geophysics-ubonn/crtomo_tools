#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from optparse import OptionParser
import modul_cleaner as mc
import shutil
import math
import os


def handle_cmd_options():
    parser = OptionParser()
    # options for formula
    parser.add_option("-m",
                      dest="m",
                      type="float",
                      help="User override for coefficient m",
                      default=0.021,
                      )
    parser.add_option("-T",
                      action="store",
                      dest="T_std",
                      type="float",
                      help="User override for standard temperature",
                      default=10.0,
                      )
    # input options
    parser.add_option("--temp",
                      action="store",
                      dest="temp_file",
                      help="temperature profile",
                      default='temp/tprofile.mag',
                      )
    parser.add_option("--filename",
                      dest="filename",
                      help='filename of input file',
                      metavar="file",
                      type='string',
                      default=None,
                      )
    # output options
    parser.add_option("-o",
                      "--output",
                      dest="output",
                      help="Output file (default: rho/rho_T.mag)",
                      metavar="FILE",
                      default="rho/rho_T.dat",
                      )

    (options, args) = parser.parse_args()
    return options


def read_iter(use_fpi):
    '''Return the path to the final .mag file either for the complex or the fpi
    inversion.
    '''
    filename_rhosuffix = 'exe/inv.lastmod_rho'
    filename = 'exe/inv.lastmod'
    # filename HAS to exist. Otherwise the inversion was not finished
    if(not os.path.isfile(filename)):
        print('Inversion was not finished! No last iteration found.')

    if(use_fpi is True):
        if(os.path.isfile(filename_rhosuffix)):
            filename = filename_rhosuffix

    linestring = open(filename, 'r').readline().strip()
    linestring = linestring.replace('\n', '')
    linestring = linestring.replace('../', '')
    return linestring


def readin_temp(temp_file):
    """The temperature file should be in mag-format: header + 3 columns with
    coordinates and value of temperature. The coordinates  have to be the same
    as from the rho-file.
    """
    with open(temp_file, 'r') as fid:
        temp = np.loadtxt(fid, skiprows=1, usecols=[2])

    return temp


def readin_rho(magfile):
    """Read in the values in Ohmm to which the temperature should be added.
    """

    with open('inv/' + magfile, 'r') as fid:
        magcontent = np.loadtxt(fid, skiprows=1)
    mag = magcontent[:, 2]  # extract only magnitudes
    mag = [10 ** m for m in mag]  # remove logarithm

    return mag


def calc_correction(data, T_std, m, temp):
    """Add the temperature effect to the data. The data is given in Ohmm.
    """
#    if options.add:
#        data_i = (m * (T_std - 25) + 1) / (m * (temp - 25) + 1) * data
#        return data_i
#    else:
    data_std = (m * (temp - 25) + 1) / (m * (T_std - 25) + 1) * data
    return data_std


def save_mag_to_file(mag, filename):
    """Save the new values in rho.dat (add) or rho.mag (sub).
    """
    # bring data in shape
    null = np.zeros(len(mag))
    result = np.transpose(np.vstack((mag, null)))

    # save datapoints
    with open(filename, 'w') as fid:
        fid.write('{0}\n'.format(mag.shape[0]))
    with open(filename, 'ab') as fid:
        np.savetxt(fid, np.array(result), fmt='%f')

    if not options.add:
        # bring data in shape
        magfile = mc.get_magfile('.')
        with open('inv/' + magfile, 'r') as fid:
            coor = np.loadtxt(fid, skiprows=1, usecols=[0, 1])
        mag_log = [math.log(d, 10) for d in mag]  # calculated back to log
        content = np.column_stack((coor[:, 0], coor[:, 1], mag_log))

        # Overwrite mag-file
        try:
            open('inv/' + magfile + '_old', 'r')
        except:
            shutil.copy('inv/' + magfile, 'inv/' + magfile + '_old')
        with open('inv/' + magfile, 'w') as fid:
            fid.write('{0}\n'.format(content.shape[0]))
        with open('inv/' + magfile, 'ab') as fid:
            np.savetxt(fid, np.array(content), fmt='%f')


def consider_temperature(temp_file, T_std, m, output):
    """Main function: get data, calculate new values, save result
    """
    temp = readin_temp(temp_file)
    mag = readin_rho()
    mag_cor = calc_correction(mag, T_std, m, temp)
    save_mag_to_file(mag_cor, output)

def main():
    pass

if __name__ == '__main__':
    main()
