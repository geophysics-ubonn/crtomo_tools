#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Tool to substract the temperature effect of given rawdata in a volt.dat file.
The calculation is done after Hayley (2010):

    d_obs^TC = d_obs + (d_est^TC - d_est)

Necessary Input:
* d_obs: measured field data to correct
* d_est: synthetic data of inversion result from d_obs
* d_estTC: synthetic data of temperature corrected inversion result of d_obs

Options:
* output: output file
'''
import numpy as np
from optparse import OptionParser


def handle_options():
    '''Handle options.
    '''
    parser = OptionParser()
    parser.set_defaults(aniso=False)

    parser.add_option("--dobs",
                      dest="d_obs",
                      help="field data",
                      metavar="file",
                      default="mod/volt.dat",
                      )
    parser.add_option("--dest",
                      dest="d_est",
                      help="synthetic data of inversion result",
                      metavar="file",
                      )
    parser.add_option("--desttc",
                      dest="d_estTC",
                      help="synthetic data of corrected inversionr esult",
                      metavar="file",
                      )
    # output options
    parser.add_option("-o",
                      "--output",
                      dest="output",
                      help="output file",
                      metavar="file",
                      default="temp/volt.dat",
                      )

    (options, args) = parser.parse_args()
    return options


def readin_volt(filename):
    """Read in measurement data from a volt.dat file and return electrodes and
    measured resistance.
    """
    with open(filename, 'r') as fid:
        content = np.loadtxt(fid, skiprows=1, usecols=[0, 1, 2])
        volt = content[:, 2]
        elecs = content[:, 0:2]
    return elecs, volt


def calc_correction(volt1, volt2, volt3):
    """Remove the temperature effect from field data using inversion results of
    that data:

        print(volt[0])
        d_obs^TC = d_obs + (d_est^TC - d_est)

    Parameters
    ----------
    d_obs:
        measured field data to correct (volt1)
    d_est:
        synthetic data of inversion result from d_obs (volt2)
    d_estTC:
        synthetic data of temperature corrected inversion result of d_obs
        (volt3)
    """
    volt = np.array([a - b + c for a, b, c in zip(volt1, volt2, volt3)])
    volt[np.where(volt < 0)] = 0.000001

    return volt


def save_volt(elecs, volt, filename):
    """Save the values in volt-format.
    """
    # bring data in shape
    content = np.column_stack((elecs, volt, np.zeros(len(volt))))

    # save datapoints
    with open(filename, 'w') as fid:
        fid.write('{0}\n'.format(content.shape[0]))
    with open(filename, 'ab') as fid:
        np.savetxt(fid, np.array(content), fmt='%i %i %f %f')


def main():
    """Function to remove temperature effect from field data
    """
    options = handle_options()

    # read in observed and synthetic data
    elecs, d_obs = readin_volt(options.d_obs)
    elecs, d_est = readin_volt(options.d_est)
    elecs, d_estTC = readin_volt(options.d_estTC)
    # calculate corrected data
    volt_corr = calc_correction(d_obs,
                                d_est,
                                d_estTC,
                                )
    # save data
    save_volt(elecs,
              volt_corr,
              options.output,
              )


if __name__ == '__main__':
    main()
