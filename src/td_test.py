#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""
from optparse import OptionParser


def handle_cmd_options():
    parser = OptionParser()
    parser.add_option("-e", "--elem", dest="elem_file", type="string",
                      help="elem.dat file (default: elem.dat)",
                      default="elem.dat")
    parser.add_option("-t", "--elec", dest="elec_file", type="string",
                      help="elec.dat file (default: elec.dat)",
                      default="elec.dat")
    parser.add_option("--config", dest="config_file", type="string",
                      help="config.dat file (default: config.dat)",
                      default="config.dat")
    parser.add_option("-i", "--use_first_line", action="store_true",
                      dest="use_first_line", default=False,
                      help="Normally the first line of the config file is " +
                      "ignored, but if set to True, it will be used. " +
                      "Default: False")
    parser.add_option('-s', "--sink", dest="sink", type="int",
                      help="Fictitious sink node nr, implies 2D mode",
                      default=None)
    parser.add_option("--data", dest="data_file", type="string",
                      help="Data file (default: volt.dat)",
                      default='volt.dat')
    parser.add_option("-f", "--frequency", dest="frequency", type="int",
                      help="Frequency/Column in volt.dat, starting from 0 " +
                      "(default: 2)", default=2)

    parser.add_option("-o", "--output", dest="output_file", type="string",
                      help="Output file (plot) (default: sens_center.png)",
                      default='sens_center.png')

    parser.add_option("--cblabel", dest="cblabel", type="string",
                      help=r"ColorbarLabel (default: $Data$)",
                      default=r'$Data$')
    parser.add_option("--label", dest="label", type="string",
                      help=r"Label (default: none)", default=r'$ $')
    parser.add_option("-w", "--weight", dest="weight_int", type="int",
                      help="Choose the weights used : 0 - unweighted, 1 - " +
                      "abs, 2 -log10, 3 - sqrt", default=0)
    parser.add_option("-c", "--plot_configurations", action="store_true",
                      dest="plot_configurations", default=False,
                      help="Plots every configuration sensitivity center in " +
                      "a single file. Default: False")
    parser.add_option("--no_plot", action="store_true",
                      dest="no_plot", default=False,
                      help="Do not create center plot (only text output)")

    (options, args) = parser.parse_args()
    return options


def main():
    options = handle_cmd_options()
    print('NO PLOT', options.no_plot)
