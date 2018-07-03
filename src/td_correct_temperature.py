#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Tool to add or substract the temperature effect on/of resistivity data.
The calculation is done after Hayley (2007):

    rho_std_i = (m * (T_i - 25°) + 1) / (m * (T_std - 25°) + 1) * rho_i

Options:

* m: choose coeffcient m
* T_std: choose standard temperatur to which to correct the data
* add: add the effect of temperature instead of substracting it (for modelling)
* aniso: resistivity data is anisotropic (three columns)
* temp: file with temperature information in 3rd column with a headerline
        (mag-format)
* filename: file with resistivity information
* rho: resistivity input and output files are in rho- instead of mag-format
* output: output file

'''
import numpy as np
from optparse import OptionParser
import math


def handle_options():
    '''Handle options.
    '''
    parser = OptionParser()
    parser.set_defaults(add=False)
    parser.set_defaults(rhofile=False)
    parser.set_defaults(aniso=False)

    parser.add_option("-m",
                      dest="m",
                      type="float",
                      help="User override for coefficient m",
                      default=0.021,
                      )
    parser.add_option("-T",
                      dest="T_std",
                      type="float",
                      help="User override for standard temperature",
                      default=10.0,
                      )
    parser.add_option("--add",
                      help="add temperature effect, default=sub",
                      dest='add',
                      action='store_true',
                      default=False
                      )
    parser.add_option("--aniso",
                      help="resistivity data in anisotropic (three columns)",
                      dest='aniso',
                      action='store_true',
                      default=False
                      )
    # input options
    parser.add_option("--temp",
                      dest="temp_file",
                      help="file with temperature data",
                      metavar="file",
                      default='temp/tprofile.mag',
                      )
    parser.add_option("--filename",
                      dest='filename',
                      help='file with resistivity data',
                      metavar="file",
                      type='string',
                      default=None,
                      )
    parser.add_option('-r',
                      "--rho",
                      help="define input file as a rho-file, default=mag-file",
                      dest='rhofile',
                      action='store_true',
                      )
    # output options
    parser.add_option("-o",
                      "--output",
                      dest="output",
                      help="output file",
                      metavar="file",
                      default="temp/rho_T.mag",
                      )

    (options, args) = parser.parse_args()
    return options


def readin_temp(temp_file):
    """The temperature file should be in mag-format: header + 3 columns with
    coordinates and value of temperature. The coordinates  have to be the same
    as from the resistivity data.
    Such a temperature file can be produced with #############
    """
    with open(temp_file, 'r') as fid:
        temp = np.loadtxt(fid, skiprows=1, usecols=[2])

    return temp


def read_iter():
    '''Return the path to the final rho*.mag file from the tomodir.
    '''
    filename = 'exe/inv.lastmod'
    linestring = open(filename, 'r').readline().strip()
    linestring = linestring.replace('\n', '')
    linestring = linestring.replace('../', '')
    return linestring


def readin_rho(filename, rhofile=True, aniso=False):
    """Read in the values of the resistivity in Ohmm.
    The format is variable: rho-file or mag-file.
    """
    if aniso:
        a = [[0, 1, 2], [2, 3, 4]]
    else:
        a = [0, 2]
    if rhofile:
        if filename is None:
            filename = 'rho/rho.dat'
        with open(filename, 'r') as fid:
            mag = np.loadtxt(fid, skiprows=1, usecols=(a[0]))

    else:
        if filename is None:
            filename = read_iter()
        with open(filename, 'r') as fid:
            mag = np.power(10, np.loadtxt(fid, skiprows=1, usecols=(a[1])))

    return mag


def calc_correction(temp, mag, add=False, T_std=10, m=0.021):
    """Function to add or substract the temperature effect to given data. The
    function can be called in python scripts. For application via command line
    in a file system use the script td_correct_temperature.py. The data is
    taken and given in Ohmm.

    rho_std_i = (m * (T_i - 25°) + 1) / (m * (T_std - 25°) + 1) * rho_i
    rho_i = (m * (T_std - 25°) + 1) / (m * (T_i - 25°) + 1) * rho_std_i

    Hayley (2007)

    Parameters:
        temp: temperature values corresponding to the individual resistivity
              values
        mag: resistivity values to be corrected
        add: switch for adding instead of substracting the effect
        T_std: standard temperature t or from which to correct (default=10°)
        m:coeffcient (default=0.021)
    """
    if mag.shape[1] == 3:
        if add:
            data_x = (m * (T_std - 25) + 1) / (m * (temp - 25) + 1) * mag[:, 0]
            data_y = (m * (T_std - 25) + 1) / (m * (temp - 25) + 1) * mag[:, 1]
            data_z = (m * (T_std - 25) + 1) / (m * (temp - 25) + 1) * mag[:, 2]
            return np.column_stack((data_x, data_y, data_z))
        else:
            data_x = (m * (temp - 25) + 1) / (m * (T_std - 25) + 1) * mag[:, 0]
            data_y = (m * (temp - 25) + 1) / (m * (T_std - 25) + 1) * mag[:, 1]
            data_z = (m * (temp - 25) + 1) / (m * (T_std - 25) + 1) * mag[:, 2]
            return np.column_stack((data_x, data_y, data_z))
    else:
        if add:
            data_i = (m * (T_std - 25) + 1) / (m * (temp - 25) + 1) * mag
            return data_i
        else:
            data_std = (m * (temp - 25) + 1) / (m * (T_std - 25) + 1) * mag
            return data_std


def save_mag_to_file(mag, filename, rhofile):
    """Save the values in rho- or mag-format.
    """
    if rhofile:
        # bring data in shape
        null = np.zeros(len(mag))
        if mag.shape[1] == 3:
            null = np.column_stack((null, null, null, null))
        result = np.column_stack((mag, null))

        # save datapoints
        with open(filename, 'w') as fid:
            fid.write('{0}\n'.format(mag.shape[0]))
        with open(filename, 'ab') as fid:
            np.savetxt(fid, np.array(result), fmt='%f')

    else:
        # bring data in shape
        with open('inv/rho00.mag', 'r') as fid:
            coor = np.loadtxt(fid, skiprows=1, usecols=[0, 1])
        # calculated back to log
        if mag.shape[1] == 3:
            logx = [math.log(d, 10) for d in mag[:, 0]]
            logy = [math.log(d, 10) for d in mag[:, 1]]
            logz = [math.log(d, 10) for d in mag[:, 2]]
            mag_log = np.column_stack((logx, logy, logz))
        else:
            mag_log = [math.log(d, 10) for d in mag]
        content = np.column_stack((coor[:, 0], coor[:, 1], mag_log))

        # save datapoints
        with open(filename, 'w') as fid:
            fid.write('{0}\n'.format(content.shape[0]))
        with open(filename, 'ab') as fid:
            np.savetxt(fid, np.array(content), fmt='%f')


def main():
    """Function to add or substract the temperature effect to data in a tomodir
    """
    options = handle_options()

    # read in temperature and resistivity data
    tempdata = readin_temp(options.temp_file)
    magdata = readin_rho(options.filename,
                         options.rhofile,
                         aniso=options.aniso)
    # calculate corrected data
    mag_corr = calc_correction(temp=tempdata,
                               mag=magdata,
                               add=options.add,
                               T_std=options.T_std,
                               m=options.m,)
    # save data
    save_mag_to_file(mag_corr,
                     options.output,
                     options.rhofile)


if __name__ == '__main__':
    main()
