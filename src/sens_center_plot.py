#!/usr/bin/python
# -*- coding: utf-8 -*-
""" For each measurement configuration, the sensitivity distribution and the
center of mass of its values is computed.

Then for all measurements sensitivities and centers of mass are plotted in the
grid. This might give a better overview on the sensitivities of our measurement
configuration.

Different weights for the sensitivities can be used (--weight):

    - 0: unweighted,
    - 1: abs,
    - 2: log10,
    - 3: square,

by invoking the command line options.

Use sens_center_plot.py -h for help or take a look at the tests provided in
TESTS/sens_center_plot.

Examples
--------

Plot center plot, and single measurement sensitivities: ::

    sens_center_plot.py --elem elem.dat --elec elec.dat --config config.dat -c

Disable plots: ::

    sens_center_plot.py --no_plot

Use alternative weighting functions:


    sens_center_plot.py --weight 0
    sens_center_plot.py --weight 1
    sens_center_plot.py --weight 2
    sens_center_plot.py --weight 3


"""
import crtomo.mpl
plt, mpl = crtomo.mpl.setup()
from optparse import OptionParser
import numpy as np
import shutil

import crtomo.grid as CRGrid
import crtomo.cfg as CRcfg
# import crlab_py.elem as elem
# import crlab_py.CRMod as CRMod


def handle_cmd_options():
    parser = OptionParser()
    parser.add_option(
        "-e", "--elem",
        dest="elem_file",
        type="string",
        help="elem.dat file (default: elem.dat)",
        default="elem.dat"
    )
    parser.add_option(
        "-t", "--elec",
        dest="elec_file",
        type="string",
        help="elec.dat file (default: elec.dat)",
        default="elec.dat"
    )
    parser.add_option(
        "--config", dest="config_file",
        type="string",
        help="config.dat file (default: config.dat)",
        default="config.dat"
    )
    parser.add_option(
        "-i", "--use_first_line",
        action="store_true",
        dest="use_first_line",
        default=False,
        help="Normally the first line of the config file is " +
        "ignored, but if set to True, it will be used. " +
        "Default: False"
    )
    parser.add_option(
        '-s', "--sink",
        dest="sink",
        type="int",
        help="Fictitious sink node nr, implies 2D mode",
        default=None
    )
    parser.add_option(
        "--data", dest="data_file",
        type="string",
        help="Data file (default: volt.dat)",
        default='volt.dat'
    )
    parser.add_option(
        "-f", "--frequency",
        dest="frequency",
        type="int",
        help="Frequency/Column in volt.dat, starting from 0 " +
        "(default: 2)",
        default=2
    )

    parser.add_option(
        "-o", "--output",
        dest="output_file",
        type="string",
        help="Output file (plot) (default: sens_center.png)",
        default='sens_center.png'
    )

    parser.add_option(
        "--cblabel", dest="cblabel",
        type="string",
        help=r"ColorbarLabel (default: $Data$)",
        default=r'$Data$'
    )
    parser.add_option(
        "--label",
        dest="label",
        type="string",
        help=r"Label (default: none)",
        default=r'$ $'
    )
    parser.add_option(
        "-w", "--weight",
        dest="weight_int",
        type="int",
        help="Choose the weights used : 0 - unweighted, 1 - " +
        "abs, 2 -log10, 3 - sqrt",
        default=0
    )
    parser.add_option(
        "-c", "--plot_configurations",
        action="store_true",
        dest="plot_configurations",
        default=False,
        help="Plots every configuration sensitivity center in " +
        "a single file. Default: False"
    )
    parser.add_option(
        "--no_plot",
        action="store_true",
        dest="no_plot",
        default=False,
        help="Do not create center plot (only text output)"
    )

    (options, args) = parser.parse_args()
    return options


class sens_center:

    def __init__(self, elem_file, elec_file, options, weight):
        self.options = options
        self.elem_file = elem_file
        self.elec_file = elec_file
        self.weight = weight
        self.cblabel = None
        self.output_file = None
        self.grid = CRGrid.crt_grid(elem_file, elec_file)

    def plot_single_configuration(self, config_nr, sens_file):
        """
        plot sensitivity distribution with center of mass for
        a single configuration. The electrodes used are colored.

        Parameters
        ----------
        config_nr: int
            number of configuration
        sens_file: string, file path
            filename to sensitvity file

        """
        indices = elem.load_column_file_to_elements_advanced(
            sens_file, [2, 3],
            False,
            False
        )

        elem.plt_opt.title = ''
        elem.plt_opt.reverse = True
        elem.plt_opt.cbmin = -1
        elem.plt_opt.cbmax = 1
        elem.plt_opt.cblabel = r'fill'
        elem.plt_opt.xlabel = 'x (m)'
        elem.plt_opt.ylabel = 'z (m)'

        fig = plt.figure(figsize=(5, 7))
        ax = fig.add_subplot(111)
        ax, pm, cb = elem.plot_element_data_to_ax(
            indices[0],
            ax,
            scale='asinh',
            no_cb=False,
        )
        ax.scatter(
            self.sens_centers[config_nr, 0],
            self.sens_centers[config_nr, 1],
            marker='*',
            s=50,
            color='w',
            edgecolors='w',
        )

        self.color_electrodes(config_nr, ax)

        # Output
        sensf = sens_file.split('sens')[-1]
        sensf = sensf.split('.')[0]
        out = 'sens_center_' + sensf + '.png'
        fig.savefig(out, bbox_inches='tight', dpi=300)
        fig.clf()
        plt.close(fig)

    def plot_sens_center(self, frequency=2):
        """
        plot sensitivity center distribution for all configurations in
        config.dat.  The centers of mass are colored by the data given in
        volt_file.
        """
        try:
            colors = np.loadtxt(self.volt_file, skiprows=1)
        except IOError:
            print('IOError opening {0}'.format(volt_file))
            exit()

        # check for 1-dimensionality
        if(len(colors.shape) > 1):
            print('Artificial or Multi frequency data')
            colors = colors[:, frequency].flatten()

        colors = colors[~np.isnan(colors)]

        elem.load_elem_file(self.elem_file)
        elem.load_elec_file(self.elec_file)
        nr_elements = len(elem.element_type_list[0])
        elem.element_data = np.zeros((nr_elements, 1)) * np.nan

        elem.plt_opt.title = ' '
        elem.plt_opt.reverse = True
        elem.plt_opt.cbmin = -1
        elem.plt_opt.cbmax = 1
        elem.plt_opt.cblabel = self.cblabel
        elem.plt_opt.xlabel = 'x (m)'
        elem.plt_opt.ylabel = 'z (m)'

        fig = plt.figure(figsize=(5, 7))
        ax = fig.add_subplot(111)
        ax, pm, cb = elem.plot_element_data_to_ax(0, ax, scale='linear',
                                                  no_cb=True)
        ax.scatter(self.sens_centers[:, 0], self.sens_centers[:, 1], c=colors,
                   s=100, edgecolors='none')

        cb_pos = mpl_get_cb_bound_next_to_plot(ax)
        ax1 = fig.add_axes(cb_pos, frame_on=True)
        cmap = mpl.cm.jet_r
        norm = mpl.colors.Normalize(vmin=np.nanmin(colors),
                                    vmax=np.nanmax(colors))
        mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm,
                                  orientation='vertical')

        fig.savefig(self.output_file, bbox_inches='tight', dpi=300)

    def color_electrodes(self, config_nr, ax):
        """
        Color the electrodes used in specific configuration.
        Voltage electrodes are yellow, Current electrodes are red ?!
        """
        electrodes = np.loadtxt(options.config_file, skiprows=1)
        electrodes = self.configs[~np.isnan(self.configs).any(1)]
        electrodes = electrodes.astype(int)

        conf = []
        for dim in range(0, electrodes.shape[1]):
            c = electrodes[config_nr, dim]
            # c = c.partition('0')
            a = np.round(c / 10000) - 1
            b = np.mod(c, 10000) - 1
            conf.append(a)
            conf.append(b)
        Ex, Ez = elem.get_electrodes()
        color = ['#ffed00', '#ffed00', '#ff0000', '#ff0000']
        ax.scatter(Ex[conf], Ez[conf], c=color, marker='s', s=60,
                   clip_on=False, edgecolors='k')

    def compute_sens(self, elem_file, elec_file, configs):
        """
        Compute the sensitivities for the given input data.
        A CRMod instance is called to create the sensitivity files.
        """
        CRMod_config = CRMod.config()
        # activate 2D mode and set sink nr
        if self.options.sink is not None:
            print('2D mode with sink {0}'.format(self.options.sink))
            CRMod_config['2D'] = 0
            CRMod_config['fictitious_sink'] = 'T'
            CRMod_config['sink_node'] = self.options.sink

        CRMod_config['write_sens'] = 'T'
        CRMod_instance = CRMod.CRMod(CRMod_config)
        CRMod_instance.elemfile = elem_file
        CRMod_instance.elecfile = elec_file
        CRMod_instance.configdata = configs

        resistivity = 100
        # get number of elements
        fid = open(elem_file, 'r')
        fid.readline()
        elements = int(fid.readline().strip().split()[1])
        fid.close()

        # create rho.dat file
        rhodata = '{0}\n'.format(elements)
        for i in range(0, elements):
            rhodata += '{0}   0\n'.format(resistivity)
        CRMod_instance.rhodata = rhodata

        CRMod_instance.run_in_tempdir()
        volt_file = CRMod_instance.volt_file
        sens_files = CRMod_instance.sens_files
        return sens_files, volt_file, CRMod_instance.temp_dir

    def compute_center_of_mass(self, filename):
        """
        Center of mass is computed using the sensitivity data output from CRMod
        Data weights can be applied using command line options
        """
        sens = np.loadtxt(filename, skiprows=1)

        X = sens[:, 0]
        Z = sens[:, 1]
        # C = (np.abs(sens[:,2]))# ./ np.max(np.abs(sens[:,2]))
        C = sens[:, 2]

        x_center = 0
        z_center = 0
        sens_sum = 0

        for i in range(0, C.shape[0]):
            # unweighted
            if(self.weight == 0):
                weight = (C[i])
            # abs
            if(self.weight == 1):
                weight = np.abs(C[i])
            # log10
            if(self.weight == 2):
                weight = np.log10(np.abs(C[i]))
            # sqrt
            if(self.weight == 3):
                weight = np.sqrt(np.abs(C[i]))

            x_center += (X[i] * weight)
            z_center += (Z[i] * weight)
            sens_sum += weight
        x_center /= sens_sum
        z_center /= sens_sum

        return (x_center, z_center)

    def get_configs(self, filename, use_first_line):
        # 1. compute sensitivities of config file
        if(options.use_first_line):
            skiprows = 0
        else:
            skiprows = 1

        # 2. load configuration file
        configs = np.loadtxt(options.config_file, skiprows=skiprows)
        configs = configs[~np.isnan(configs).any(1)]  # remove nans
        self.configs = configs

    def get_sens_centers(self, sens_files):
        center = []
        for sens_f in sens_files:
            c = self.compute_center_of_mass(sens_f)
            center.append(c)

        center = np.array(center)
        self.sens_centers = center

    def compute_sensitivity_centers(self):
        # compute the sensitivity centers (CRMod is called!) and store
        # them in sens_centers. Save to 'center.dat'
        sens_files, volt_file, temp_dir = self.compute_sens(self.elem_file,
                                                            self.elec_file,
                                                            self.configs)

        self.sens_files = sens_files
        self.temp_dir = temp_dir
        self.volt_file = volt_file
        self.get_sens_centers(sens_files)

    def plot_sensitivities_to_file(self):
        for k, sens_f in enumerate(self.sens_files):
            print('Plotting' + sens_f)
            self.plot_single_configuration(k, sens_f)

    def remove_tmp_dir(self, directory):
        """
        Remove the directory if it is located in /tmp
        """
        if(not directory.startswith('/tmp/')):
            print('Directory not in /tmp')
            exit()

        print('Deleting directory: ' + directory)

        shutil.rmtree(directory)

    def clean(self):
        self.remove_tmp_dir(self.temp_dir)


def main():
    options = handle_cmd_options()

    center_obj = sens_center(
        options.elem_file,
        options.elec_file,
        options,
        weight=options.weight_int,
    )
    center_obj.cblabel = options.cblabel
    center_obj.output_file = options.output_file

    center_obj.get_configs(options.config_file, options.use_first_line)
    center_obj.compute_sensitivity_centers()
    np.savetxt('center.dat', center_obj.sens_centers)

    if not options.no_plot:
        print('Creating center plot')
        center_obj.plot_sens_center(options.frequency)

    if(options.plot_configurations):
        print('Plotting single configuration sensitivities')
        center_obj.plot_sensitivities_to_file()

    center_obj.clean()


if __name__ == '__main__':
    main()
