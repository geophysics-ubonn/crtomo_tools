#!/usr/bin/env python
"""

"""
import os
import shutil
import io
import codecs
from copy import deepcopy
import pickle
import importlib.resources

import ipywidgets as widgets
from IPython.display import display
from IPython.display import IFrame
from ipywidgets import GridspecLayout
from ipywidgets import GridBox, Layout
import pylab as plt

import reda
import crtomo


def do_we_run_in_ipython():
    we_run_in_ipython = False
    try:
        # flake8-error F821 complains about missing definition of
        # get_ipython
        # that's the point: this function is only defined when we run within
        # ipython
        we_run_in_ipython = hasattr(get_ipython(), 'kernel')  # noqa: F821
    except Exception:
        pass
    return we_run_in_ipython


class base_step(object):
    def __init__(self, persistent_directory=None):
        self.persistent_directory = persistent_directory
        if persistent_directory is not None:
            os.makedirs(self.persistent_directory, exist_ok=True)

        # we need to identify steps, e.g. for checking if required, previous
        # steps have already run. No space, no fancy characters
        self.name = None

        # A user-readable title for this step
        self.title = None

        # identifier for the help page to show when showing the associated gui
        self.help_page = None

        # we have a few status variables
        self.has_run = False

        # we differentiate between settings that have already be applied (i.e.,
        # those that correspond to self.results), and those that will be
        # applied to generate new results at some time
        self.input_applied = {}
        self.input_new = {}

        # This variable will be populated with a dictionary of the same
        # structure as self.input_applied and self.input_new. Instead of actual
        # values, the key:value pairs hold the data types of the items. This
        # way we a) have a reference for the expected input structure and b)
        # could implement a validity check
        self.input_skel = None

        # we store results of this step here
        self.results = {}

        # steps are stored in a linked list (or tree? We allow multiple next
        # items)
        self.next_step = []
        self.prev_step = None

        # these steps must reside somewhere in the prev-branch and be finished
        # before this step can run
        # Set to None if no steps are required for this one
        self.required_steps = []

        # we inherently provide a Jupyter Widget-based GUI to each step
        # this gui element can be embedded into larger notebook guis, e.g. for
        # a complete workflow
        self.jupyter_gui = None
        self.widgets = {}

        # this callback can be used to trigger events outside of this step
        # object
        self.callback_step_ran = None

    def can_run(self):
        """Check if all required steps have been finished
        """
        if self.required_steps is None:
            return True

        def find_required_steps(branch, search_results):
            if branch.name in search_results.keys():
                search_results[branch.name] = branch.has_run

            if branch.prev_step is not None:
                search_results = find_required_steps(
                    branch.prev_step, search_results
                )
            return search_results

        search_results = find_required_steps(
            self.prev_step,
            {key: None for key in self.required_steps}
        )
        # print('Search results:')
        # print(search_results)
        can_run = True
        for key, item in search_results.items():
            if item is None:
                print('[{}] Required step not found: {}'.format(
                    self.name, key
                ))
                return False
            # print('testing:', can_run, item)
            can_run = can_run & item
            # print('   result:', can_run)
        return can_run

    def set_input_new(self, input_new):
        """Apply a new set of inputs

        TODO: This is the place to check the input_new dictionary for
        consistency with self.input_skel

        """
        assert isinstance(input_new, dict), "input_new must be a dict"
        self.input_new = input_new

    def transfer_input_new_to_applied(self):
        """Make a copy of the self.input_new dict and store in
        self.input_applied

        This is complicated because some objects cannot be easily copied (e.g.,
        io.BytesIO). Therefore, each step must implement this function by
        itself.
        """
        # this should suffice for simple input dicts
        self.input_applied = deepcopy(self.input_new)

    def apply_next_input(self):
        """Actually execute the step based in self.input_new

        """
        raise Exception('Must be implemented by deriving class')

    def create_ipywidget_gui(self):
        """

        """
        raise Exception('Must be implemented by deriving class')

    def persistency_store(self):
        """Store the current state of the widget in the given persistency
        directory
        """
        print('Persistent store - base version', self.name)
        if self.persistent_directory is None:
            return

        stepdir = self.persistent_directory + os.sep + 'step_' + self.name
        print('Writing data to:', stepdir)
        os.makedirs(stepdir, exist_ok=True)

        # store settings in pickle
        with open(stepdir + os.sep + 'settings.pickle', 'wb') as fid:
            pickle.dump(self.input_applied, fid)

        # store state
        with open(stepdir + os.sep + 'state.dat', 'w') as fid:
            fid.write('{}\n'.format(int(self.has_run)))

    def persistency_load(self):
        """Load state of this step from peristency directory
        """
        print('Persistent load - base version', self.name)
        if self.persistent_directory is None:
            return

        stepdir = self.persistent_directory + os.sep + 'step_' + self.name

        if not os.path.isdir(stepdir):
            return

        # load settings from pickle
        with open(stepdir + os.sep + 'settings.pickle', 'rb') as fid:
            # really to input_new ???? makes only sense if has_run=True
            self.input_new = pickle.load(fid)

        # load state
        state = bool(
            open(stepdir + os.sep + 'state.dat', 'r').readline().strip()
        )
        print('has_run  from load:', state)
        if state:
            print('applying next input from persistent storage')
            self.apply_next_input()

    def find_previous_step(self, starting_step, search_name):
        if starting_step.name == search_name:
            return starting_step
        if starting_step.prev_step is None:
            return None
        result = self.find_previous_step(starting_step.prev_step, search_name)
        return result


class step_fe_mesh(base_step):
    """Load, or generate, an FE mesh for CRTomo
    """
    def __init__(self, persistent_directory=None):
        super().__init__(persistent_directory=persistent_directory)
        self.name = 'fe_mesh'
        self.title = 'FE Mesh'
        self.required_steps = None
        self.help_page = 'fe_meshes.html'

        self.input_skel = {
            'elem_data': io.BytesIO,
            'elec_data': io.BytesIO,
        }

    def create_ipywidget_gui(self):

        self.widgets['label_intro'] = widgets.Label(
            'This tab allows you to load, or generate, an FE mesh for CRTomo'
        )
        # Variant 1: directly load elem/elec.dat files
        self.widgets['label_elem'] = widgets.Label(
            "Upload elem.dat"
        )
        self.widgets['file_elem'] = widgets.FileUpload(
            accept='.dat',
            multiple=False
        )
        self.widgets['label_elec'] = widgets.Label(
            "Upload elec.dat"
        )
        self.widgets['file_elec'] = widgets.FileUpload(
            accept='.dat',
            multiple=False
        )

        self.widgets['button_upload'] = widgets.Button(
            description='Import elem.dat and elec.dat',
            disabled=False,
            # 'success', 'info', 'warning', 'danger' or ''
            button_style='',
            tooltip='Click me',
            icon='check'  # (FontAwesome names without the `fa-` prefix)
        )
        self.widgets['button_upload'].on_click(
            self.apply_next_input_from_gui
        )
        self.widgets['label_feedback'] = widgets.Label()
        self.widgets['output_meshimg'] = widgets.Output()

        self.jupyter_gui = widgets.VBox([
            GridBox(
                children=[
                    self.widgets['label_intro'],
                    widgets.HBox([
                        self.widgets['label_elem'],
                        self.widgets['file_elem'],
                    ]),
                    widgets.HBox([
                        self.widgets['label_elec'],
                        self.widgets['file_elec'],
                    ]),
                    widgets.HBox([
                        self.widgets['button_upload'],
                        self.widgets['label_feedback'],
                    ]),
                    self.widgets['output_meshimg'],
                ],
                layout=Layout(
                    width='100%',
                    grid_template_columns='auto',
                    grid_template_rows='50px 50px 50px 50px auto',
                    grid_gap='5px 10px'
                 )
            )
            ]
        )

        # self.jupyter_gui = GridspecLayout(
        #     5, 3
        # )
        # self.jupyter_gui[0, :] = self.widgets['label_intro']
        # self.jupyter_gui[1, 0] = self.widgets['label_elem']
        # self.jupyter_gui[1, 1] = self.widgets['file_elem']
        # self.jupyter_gui[2, 0] = self.widgets['label_elec']
        # self.jupyter_gui[2, 1] = self.widgets['file_elec']
        # self.jupyter_gui[3, 0:2] = self.widgets['button_upload']
        # self.jupyter_gui[3, 2] = self.widgets['label_feedback']
        # self.jupyter_gui[4, :] = self.widgets['output_meshimg']

        # Variant 2: Create a mesh using electrodes.dat, boundaries.dat,
        # char_length.dat, extra_nodes.dat, extra_lines.dat
        # TODO

    def apply_next_input(self):
        """

        """
        print('FE MESH: apply_next_input')
        if not self.can_run():
            return

        # load mesh into a grid object
        mesh = crtomo.crt_grid(
            self.input_new['elem_data'],
            self.input_new['elec_data'],
        )
        self.results['mesh'] = mesh

        fig, ax = mesh.plot_grid()
        self.results['mesh_fig'] = fig
        self.results['mesh_ax'] = ax

    def apply_next_input_from_gui(self, button):
        """Generate an input dict from the gui elements and apply those new
        inputs
        """
        print('Applying input from GUI')
        feedback = self.widgets['label_feedback']

        for widget_name in ('file_elem', 'file_elec'):
            widget = self.widgets[widget_name]
            if len(widget.value) == 0 or widget.value[0]['size'] == 0:
                feedback.value = 'You must provide both elem and elec files'
                return

        # get the file data from GUI elements
        settings = {
            'elem_data': io.BytesIO(
                self.widgets['file_elem'].value[0].content
            ),
            'elec_data': io.BytesIO(
                self.widgets['file_elec'].value[0].content
            ),
        }

        self.set_input_new(settings)

        # do not plot anything interactively
        with plt.ioff():
            self.apply_next_input()

        with self.widgets['output_meshimg']:
            fig_mesh = self.results['mesh_fig']
            display(fig_mesh)

        feedback.value = 'Mesh was loaded'

        # notify external objects
        if self.callback_step_ran is not None:
            self.callback_step_ran(self)


class step_data_import(base_step):
    def __init__(self, persistent_directory=None):
        super().__init__(persistent_directory=persistent_directory)
        self.name = 'data_import'
        self.title = 'Data Import'
        self.help_page = 'data_import.html'
        self.required_steps = None

        self.input_skel = {
            # we allow for two inputs to accommodate normal and reciprocal
            # measurements
            'data_1': io.BytesIO,
            'data_2': io.BytesIO,
            # for now, we only handle Syscal binary files
            'importer': str,
            # importer-specific settings
            'importer_settings': {
                'syscal_bin': {
                    'data_1': {
                        'reciprocals': None,
                    },
                    'data_2': {
                        'reciprocals': None,
                    },
                },
            },
        }

    def transfer_input_new_to_applied(self):
        """Make a copy of the self.input_new dict and store in
        self.input_applied

        This is complicated because some objects cannot be easily copied (e.g.,
        io.BytesIO). Therefore, each step must implement this function by
        itself.
        """
        self.input_applied = deepcopy(self.input_new)
        self.input_applied['data_1'].seek(0)
        if self.input_applied['data_2'] is not None:
            self.input_applied['data_2'].seek(0)

        # previously, we attempted to copy by hand. Should work now with
        # deepcopy
        # self.input_applied = {}
        # data_copy = io.BytesIO()
        # data1_old = self.input_new['data_1']
        # data1_old.seek(0)
        # data_copy.write(data1_old.read())
        # data_copy.seek(0)
        # self.input_applied['data_1'] = data_copy

        # data_copy = io.BytesIO()
        # data2_old = self.input_new['data_2']
        # data2_old.seek(0)
        # data_copy.write(data2_old.read())
        # data_copy.seek(0)
        # self.input_applied['data_2'] = data_copy

        # self.input_applied['importer'] = self.input_new['importer']
        # # we can deep-copy the importer settings because there are only
        # simple objects in there
        # self.input_applied['importer_settings'] = deepcopy(
        #     self.input_new['importer_settings']
        # )

    def apply_next_input(self):
        """

        """
        if not self.can_run():
            return

        import_settings = self.input_new['importer_settings']['syscal_bin']
        TDIP1 = reda.TDIP()
        TDIP1.import_syscal_bin(
            self.input_new['data_1'],
            reciprocals=import_settings['data_1']['reciprocals'],
        )
        CR1 = TDIP1.to_cr()
        self.results['cr1'] = CR1

        if self.input_new['data_2'] is not None:
            TDIP2 = reda.TDIP()
            TDIP2.import_syscal_bin(
                self.input_new['data_2'],
                reciprocals=import_settings['data_2']['reciprocals'],
            )
            CR2 = TDIP2.to_cr()
            self.results['cr2'] = CR2

            CR_merge = reda.CR()
            CR_merge.add_dataframe(CR1.data)
            CR_merge.add_dataframe(CR2.data)
            self.results['cr_merge'] = CR_merge
        else:
            self.results['cr2'] = None
            self.results['cr_merge'] = CR1

        self.transfer_input_new_to_applied()
        self.has_run = True

    def create_ipywidget_gui(self):
        self.jupyter_gui = GridspecLayout(
            4, 3
        )

        self.widgets['label_intro'] = widgets.Label(
            'Here you can upload your .bin data files and import them into ' +
            'the system.'
        )
        self.jupyter_gui[0, :] = self.widgets['label_intro']

        self.widgets['label_data1'] = widgets.Label(
            "Syscal .bin file, data file 1"
        )
        self.jupyter_gui[1, 0] = self.widgets['label_data1']
        self.widgets['upload_data1'] = widgets.FileUpload(
            accept='.bin',
            multiple=False
        )
        self.jupyter_gui[1, 1] = self.widgets['upload_data1']

        self.widgets['label_data2'] = widgets.Label(
            "Syscal .bin file, data file 2"
        )
        self.jupyter_gui[2, 0] = self.widgets['label_data2']
        self.widgets['upload_data2'] = widgets.FileUpload(
            accept='.bin',
            multiple=False
        )
        self.jupyter_gui[2, 1] = self.widgets['upload_data2']

        self.widgets['button_import'] = widgets.Button(
            description='Import Data',
            disabled=False,
            # 'success', 'info', 'warning', 'danger' or ''
            button_style='',
            tooltip='Import data into the system',
            icon='check'  # (FontAwesome names without the `fa-` prefix)
        )
        self.jupyter_gui[3, 0] = self.widgets['button_import']
        self.widgets['button_import'].on_click(
            self.apply_next_input_from_gui
        )

        self.widgets['label_feedback'] = widgets.Label('')
        self.jupyter_gui[3, 1] = self.widgets['label_feedback']

    def apply_next_input_from_gui(self, button):
        """Generate an input dict from the gui elements and apply those new
        inputs
        """
        print('Applying input from GUI')
        feedback = self.widgets['label_feedback']

        upload1 = self.widgets['upload_data1']
        if len(upload1.value) == 0 or upload1.value[0]['size'] == 0:
            feedback.value = 'Data 1 MUST be provided'
            return
        data1 = io.BytesIO(upload1.value[0].content)
        print('Data1 tell', data1.tell())

        settings = {
            'data_1': data1,
            'data_2': None,
            'importer': 'syscal_bin',
            'importer_settings': {
                'syscal_bin': {
                    'data_1': {
                        'reciprocals': None,
                    },
                    'data_2': {
                        'reciprocals': 48,
                    },
                }
            }
        }
        self.set_input_new(settings)
        self.apply_next_input()

        feedback.value = 'Data was successfully imported'

        # notify external objects
        if self.callback_step_ran is not None:
            self.callback_step_ran(self)

    def persistency_store(self):
        print('Persistent store', self.name)
        if self.persistent_directory is None:
            return

        stepdir = self.persistent_directory + os.sep + 'step_' + self.name
        print('Writing data to:', stepdir)
        os.makedirs(stepdir, exist_ok=True)

        # store settings in pickle
        with open(stepdir + os.sep + 'settings.pickle', 'wb') as fid:
            pickle.dump(self.input_applied, fid)

        # store state
        with open(stepdir + os.sep + 'state.dat', 'w') as fid:
            fid.write('{}\n'.format(int(self.has_run)))

        # it would be nice to also store results, but for now we will
        # re-generate the results upon loading
        # with open(stepdir + os.sep + 'results.pickle', 'wb') as fid:
        #     pickle.dump(self.results, fid)

    def persistency_load(self):
        print('Persistent load', self.name)
        if self.persistent_directory is None:
            return

        stepdir = self.persistent_directory + os.sep + 'step_' + self.name
        if not os.path.isdir(stepdir):
            print('stepdir does not exist')
            return
        print('Reading data to:', stepdir)

        # load settings from pickle
        with open(stepdir + os.sep + 'settings.pickle', 'rb') as fid:
            # really to input_new ???? makes only sense if has_run=True
            self.input_new = pickle.load(fid)

        # load state
        state = bool(
            open(stepdir + os.sep + 'state.dat', 'r').readline().strip()
        )
        print('has_run  from load:', state)
        if state:
            print('applying next input from persistent storage')
            self.apply_next_input()


class step_raw_visualization(base_step):
    def __init__(self, persistent_directory=None):
        super().__init__(persistent_directory=persistent_directory)
        self.name = 'raw_vis'
        self.title = 'Data Visualisation/Filtering'
        self.help_page = 'filtering.html'

        self.required_steps = [
            'data_import',
        ]

        self.input_skel = {

        }

    def apply_next_input(self):
        """

        """
        if not self.can_run():
            return

        step_import = self.find_previous_step(self, 'data_import')
        # note: we already checked that the previous step finished
        cr = step_import.results['cr_merge'].create_copy()

        # apply filters
        cr.filter('r <= 0')

        plot_r = cr.plot_histogram(column='r', log10=True)
        self.results['hist_r_log10'] = plot_r

        if 'rpha' in cr.data.columns:
            plot_rpha = cr.plot_histogram(column='rpha', log10=True)
            self.results['hist_rpha'] = plot_rpha

        fig_pseudo_log10_r = cr.pseudosection_type1(column='r', log10=True)
        self.results['ps_log10_r'] = fig_pseudo_log10_r

        self.results['cr'] = cr

        self.transfer_input_new_to_applied()
        self.has_run = True

    def create_ipywidget_gui(self):
        self.jupyter_gui = GridspecLayout(
            4, 3
        )

        self.widgets['label_intro'] = widgets.Label(
            'This tab visualises the raw data and allows you to apply data ' +
            'filters'
        )
        self.jupyter_gui[0, :] = self.widgets['label_intro']

        self.widgets['output'] = widgets.Output()
        self.jupyter_gui[1:3, :] = self.widgets['output']

        self.widgets['button_plot'] = widgets.Button(
            description='Plot',
            disabled=False,
            # 'success', 'info', 'warning', 'danger' or ''
            button_style='',
            tooltip='Click me',
            icon='check'  # (FontAwesome names without the `fa-` prefix)
        )
        self.jupyter_gui[3, 0] = self.widgets['button_plot']
        self.widgets['button_plot'].on_click(
            self.apply_next_input_from_gui
        )

        self.widgets['label_feedback'] = widgets.Label('')
        self.jupyter_gui[3, 1] = self.widgets['label_feedback']

    def apply_next_input_from_gui(self, button):
        """Generate an input dict from the gui elements and apply those new
        inputs
        """
        print('Applying input from GUI')
        feedback = self.widgets['label_feedback']

        settings = {

        }
        self.set_input_new(settings)
        # do not plot anything interactively
        with plt.ioff():
            self.apply_next_input()

        self.widgets['output'].clear_output()
        with self.widgets['output']:
            fig_rmag = self.results['hist_r_log10']['all']
            display(fig_rmag)
            if 'hist_rpha' in self.results:
                display(self.results['hist_rpha']['all'])

            if 'ps_log10_r' in self.results:
                # first entry is the fig object
                display(self.results['ps_log10_r'][0])

        feedback.value = 'Plots were generated'

        # notify external objects
        if self.callback_step_ran is not None:
            self.callback_step_ran(self)


class step_inversion_settings(base_step):
    def __init__(self, persistent_directory=None):
        super().__init__(persistent_directory=persistent_directory)
        self.name = 'inv_prepare'
        self.title = 'Inversion'
        self.help_page = 'inversion_settings.html'

        self.required_steps = [
            'raw_vis',
        ]

        self.input_skel = {
            'err_rmag_abs': float,
            'err_rmag_rel': float,
            'err_rpha_rel': float,
            'err_rpha_abs': float,
            'robust_inv': bool,
            'inv_location': str,
        }

    def apply_next_input(self):
        """

        """
        if not self.can_run():
            return

        fe_step = self.find_previous_step(self, 'fe_mesh')
        data_step = self.find_previous_step(self, 'raw_vis')

        assert fe_step is not None
        mesh = fe_step.results['mesh']
        cr = data_step.results['cr']
        measurements = cr.data[['a', 'b', 'm', 'n', 'r', 'rpha']].values

        self.tdm = crtomo.tdMan(
            grid=mesh
        )

        cid_mag, cid_pha = self.tdm.configs.load_crmod_data(measurements)
        self.tdm.register_measurements(cid_mag, cid_pha)

        if self.input_new['inv_location'] == 'local':
            result = self.tdm.invert(
                # output_directory='tmp_inv'
            )

        self.results['inv_success'] = result
        self.results['inv_error_msg'] = self.tdm.crtomo_error_msg
        self.results['inv_output'] = self.tdm.crtomo_output

        # note: we already checked that the previous step finished
        # cr = self.prev_step.results['cr_merge']

        # plot_r = cr.plot_histogram(column='r', log10=True)
        # self.results['hist_r_log10'] = plot_r

        # if 'rpha' in cr.data.columns:
        #     plot_rpha = cr.plot_histogram(column='rpha', log10=True)
        #     self.results['hist_rpha'] = plot_rpha

        self.transfer_input_new_to_applied()
        self.has_run = True

    def create_ipywidget_gui(self):
        # self.jupyter_gui = GridspecLayout(
        #     5, 3
        # )

        self.widgets['label_intro'] = widgets.Label(
            'This tab allows you to load, or generate, an FE mesh for CRTomo'
        )

        self.widgets['err_rmag_rel'] = widgets.FloatText(
            value=5,
            description='Magnitude relative error estimate [%]:',
            disabled=False
        )
        self.widgets['err_rmag_abs'] = widgets.FloatText(
            value=1e-3,
            description=r'Magnitude absolute error estimate [$\Omega$]:',
            disabled=False
        )
        self.widgets['err_rpha_rel'] = widgets.FloatText(
            value=5,
            description='Phase Error estimate [%]:',
            disabled=False
        )
        self.widgets['err_rpha_abs'] = widgets.FloatText(
            value=1,
            description='Phase Error estimate [%]:',
            disabled=False
        )
        self.widgets['check_robust_inv'] = widgets.Checkbox(
            value=False,
            description='Robust Inversion',
            disabled=False,
            indent=False
        )

        self.widgets['invert_local'] = widgets.Button(
            description='Invert local',
            disabled=False,
            # 'success', 'info', 'warning', 'danger' or ''
            button_style='',
            tooltip='Click me',
            icon='check'  # (FontAwesome names without the `fa-` prefix)
        )
        self.widgets['invert_local'].on_click(
            self.apply_next_input_from_gui
        )

        self.widgets['invert_crhydra'] = widgets.Button(
            description='Invert crhydra',
            disabled=False,
            # 'success', 'info', 'warning', 'danger' or ''
            button_style='',
            tooltip='Click me',
            icon='check'  # (FontAwesome names without the `fa-` prefix)
        )
        self.widgets['invert_crhydra'].on_click(
            self.apply_next_input_from_gui
        )

        self.widgets['label_feedback'] = widgets.Label()
        self.widgets['inv_error_msg'] = widgets.Output()

        # layout the widgets
        self.jupyter_gui = widgets.VBox(
            [
                self.widgets['label_intro'],
                widgets.HBox([
                    self.widgets['err_rmag_rel'],
                    self.widgets['err_rmag_abs'],
                ]),
                widgets.HBox([
                    self.widgets['err_rpha_rel'],
                    self.widgets['err_rpha_abs'],
                ]),
                widgets.HBox([
                    self.widgets['invert_local'],
                    self.widgets['invert_crhydra'],
                ]),
                self.widgets['check_robust_inv'],
                self.widgets['label_feedback'],
                self.widgets['inv_error_msg'],
            ]
        )

    def apply_next_input_from_gui(self, button):
        """Generate an input dict from the gui elements and apply those new
        inputs
        """
        print('Applying input from GUI')
        feedback = self.widgets['label_feedback']
        print(button.description)
        feedback.value = 'Starting inversion...please wait'

        settings = {
            'err_rmag_abs': self.widgets['err_rmag_rel'].value,
            'err_rmag_rel': self.widgets['err_rmag_rel'].value,
            'err_rpha_rel': self.widgets['err_rmag_rel'].value,
            'err_rpha_abs': self.widgets['err_rmag_rel'].value,
            'robust_inv': self.widgets['check_robust_inv'].value,
        }

        if button.description == 'Invert local':
            settings['inv_location'] = 'local'
        else:
            settings['inv_location'] = 'crhydra'

        self.set_input_new(settings)
        # # do not plot anything interactively
        # with plt.ioff():
        self.apply_next_input()

        if self.results['inv_success'] == 0:
            feedback.value = 'Inversion finished successfully'
        else:
            feedback.value = 'Inversion aborted with an error!'
            with self.widgets['inv_error_msg']:
                print(self.tdm.crtomo_error_msg)

        # notify external objects
        if self.callback_step_ran is not None:
            self.callback_step_ran(self)


class step_inversion_analysis(base_step):
    def __init__(self, persistent_directory=None):
        super().__init__(persistent_directory=persistent_directory)
        self.name = 'inv_analysis'
        self.title = 'Inv-Results'

        self.required_steps = [
            'inv_prepare',
        ]

        self.input_skel = {

        }

    def apply_next_input(self):
        """

        """
        if not self.can_run():
            return

        inv_step = self.find_previous_step(self, 'inv_prepare')

        fig, ax = inv_step.tdm.plot_inversion_result_rmag()
        self.results['fig_rmag'] = fig

        self.transfer_input_new_to_applied()
        self.has_run = True

    def create_ipywidget_gui(self):
        # self.jupyter_gui = GridspecLayout(
        #     5, 3
        # )

        self.widgets['label_intro'] = widgets.Label(
            'This tab allows you to load, or generate, an FE mesh for CRTomo'
        )

        self.widgets['output_fig_rmag'] = widgets.Output()

        self.widgets['button_plot'] = widgets.Button(
            description='Plot Result',
            disabled=False,
            # 'success', 'info', 'warning', 'danger' or ''
            button_style='',
            tooltip='Click me',
            icon='check'  # (FontAwesome names without the `fa-` prefix)
        )
        self.widgets['button_plot'].on_click(
            self.apply_next_input_from_gui
        )

        self.widgets['label_feedback'] = widgets.Label()

        # layout the widgets
        self.jupyter_gui = widgets.VBox(
            [
                self.widgets['label_intro'],
                self.widgets['output_fig_rmag'],
                self.widgets['label_feedback'],
                self.widgets['button_plot'],
            ]
        )

    def apply_next_input_from_gui(self, button):
        """Generate an input dict from the gui elements and apply those new
        inputs
        """
        print('Applying input from GUI')
        feedback = self.widgets['label_feedback']

        settings = {

        }
        self.set_input_new(settings)
        # do not plot anything interactively
        with plt.ioff():
            self.apply_next_input()

        self.widgets['output_fig_rmag'].clear_output()
        with self.widgets['output_fig_rmag']:
            fig_rmag = self.results['fig_rmag']
            display(fig_rmag)

        feedback.value = 'Plots were generated'

        # notify external objects
        if self.callback_step_ran is not None:
            self.callback_step_ran(self)


class processing_workflow_v1(object):
    """Provide one workflow for processing, inversion, and visualization of CR
    data.

    Input is provided only by dictionaries (or, optional, via json strings).

    Please note that this workflow represents only one specific way of handling
    complex-resistivity data. Other workflows may be more appropriate for
    specific applications.

    # Executing the workflow

    As a first rule of thumb, each step can only be executed once the previous
    one is finished. There may be exceptions when it comes to
    plotting/analysing inversion results.

    # Steps of the workflow:

    * step 1: Data import:


    * step 2: Raw data visualisation

        step_raw_visualisation = {
            'plot_histograms': True,
            'plot_pseudosections': True,
            // default: type 1
            'pseudosection_type': int(1),

        }

    """
    def __init__(self, persistent_directory=None, prepare_gui=True):
        """
        Parameters
        ----------
        persistent_directory : None|str
            If given, store input data and, if possible, intermediate results,
            in a persistent directory. Data is then loaded from this directory
            during the next initialisation

        """
        # print('CR Workflow V1')
        self.persistent_directory = persistent_directory

        # define steps
        self.step_fe_mesh = step_fe_mesh(persistent_directory)
        self.step_fe_mesh.callback_step_ran = self.callback_step_ran

        self.step_data_import = step_data_import(persistent_directory)
        self.step_data_import.callback_step_ran = self.callback_step_ran

        self.step_raw_visualization = step_raw_visualization(
            persistent_directory)
        self.step_raw_visualization.callback_step_ran = self.callback_step_ran

        self.step_inversion = step_inversion_settings(persistent_directory)
        self.step_inversion.callback_step_ran = self.callback_step_ran

        self.step_inv_analysis = step_inversion_analysis(persistent_directory)
        self.step_inv_analysis.callback_step_ran = self.callback_step_ran

        # put all steps within a list for easy access
        # IMPORTANT: The order must match the order of tabs in the jupyter gui
        # tab widget
        self.step_list = [
            self.step_fe_mesh,
            self.step_data_import,
            self.step_raw_visualization,
            self.step_inversion,
            self.step_inv_analysis,
        ]

        # define step association
        self.step_fe_mesh.next_step = [self.step_data_import, ]

        self.step_data_import.prev_step = self.step_fe_mesh
        self.step_data_import.next_step = [self.step_raw_visualization, ]

        self.step_raw_visualization.prev_step = self.step_data_import
        self.step_raw_visualization.next_step = [
            self.step_inversion,
        ]

        self.step_inversion.prev_step = self.step_raw_visualization
        self.step_inversion.next_step = [self.step_inv_analysis, ]

        self.step_inv_analysis.prev_step = self.step_inversion

        # root step
        self.root = self.step_fe_mesh

        # prepare the manual html pages
        self.html_base = '_tmp_crtomo_gui_manual' + os.sep

        manual_package_path = ''.join((
            str(importlib.resources.files('crtomo')),
            os.sep,
            'notebook',
            os.sep,
            'manual' + os.sep + 'html' + os.sep
        ))
        if os.path.isdir(self.html_base):
            shutil.rmtree(self.html_base)
        shutil.copytree(manual_package_path, self.html_base)

        # now we can check if there is anything to load from the storage
        def traverse_tree_call_pers_load(child):
            print('Call persistent load for', child.name)
            child.persistency_load()
            for subchild in child.next_step:
                traverse_tree_call_pers_load(subchild)

        if do_we_run_in_ipython():
            output = widgets.Output()
            with output:
                traverse_tree_call_pers_load(self.root)
        else:
            traverse_tree_call_pers_load(self.root)

        self.jupyter_gui = None

        if prepare_gui:
            self.prepare_jupyter_gui()

    def print_step_tree(self):

        def print_linked_tree(branch, level):
            if branch is None:
                return
            print(' ' * level * 3 + branch.name)
            for child in branch.next_step:
                print_linked_tree(child, level + 1)

        print_linked_tree(self.root, 0)

    def prepare_jupyter_gui(self):
        """

        """
        self.help_widget = widgets.Output()
        # test service a help page
        # with self.help_widget:
        #     display(
        #         IFrame(
        #             src='https://geophysics-ubonn.github.io/reda/',
        #             width=700,
        #             height=1000
        #         )
        #     )

        self.jupyter_tabs = widgets.Tab()
        self.jupyter_tabs.tabs = []
        self.jupyter_tabs.titles = []

        for step in self.step_list:
            step.create_ipywidget_gui()

        self.jupyter_tabs.children = [
            step.jupyter_gui for step in self.step_list]

        self.jupyter_tabs.titles = [
            step.title for step in self.step_list
        ]

        self.external_help_links = widgets.HTML(
            '<a href="http://uni-bonn.de" target="_blank">Help CRTomo</a>' +
            ' - ' +
            '<a href="http://uni-bonn.de" target="_blank">Help REDA</a>'
        )
        self.ext_help_output = widgets.Output()
        with self.ext_help_output:
            display(self.external_help_links)

        self.help_line = widgets.HBox(
            [
                widgets.Label("CR Workflow V1"),
                self.ext_help_output,
            ]
        )

        self.jupyter_gui = widgets.VBox([
            self.help_line,
            GridBox(
                children=[
                    self.jupyter_tabs,
                    self.help_widget,
                ],
                layout=Layout(
                    width='100%',
                    grid_template_columns='auto 700px',
                    grid_template_rows='auto auto',
                    grid_gap='5px 10px'
                 )
            )
            ]
        )
        self._set_help_page(self.step_list[0].help_page)

    def _set_help_page(self, page):
        page = self.html_base + page
        self.help_widget.clear_output()
        # print('HELP PAGE:', page)
        with self.help_widget:
            display(
                IFrame(
                    src=page,
                    width=700,
                    height=1000
                )
            )

    def update_help_page(self, changee):
        if changee['name'] != 'selected_index':
            return
        index = changee['new']
        step = self.step_list[index]

        if step is not None and step.help_page is not None:
            page = step.help_page
        else:
            page = 'index.html'
        self._set_help_page(page)

    def show_gui(self):
        display(self.jupyter_gui)
        self.jupyter_tabs.observe(self.update_help_page)

    def callback_step_ran(self, step):
        """This function is called from within a given step when this step
        applied new settings (i.e., it runs)

        Do the following:
            * invalidate all steps further down
        """
        print('Workflow callback called')
        print('from', step.name)
        pass


class crtomo_gui_jupyter(object):
    def __init__(self):
        self.prepare_widgets()

    def prepare_widgets(self):
        # https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html#file-upload
        self.file_elem = widgets.FileUpload(
            # Accepted file extension e.g. '.txt', '.pdf', 'image/*',
            # 'image/*,.pdf'
            accept='.dat',
            multiple=False
        )
        self.file_elec = widgets.FileUpload(
            accept='.dat',
            multiple=False
        )
        self.file_volt = widgets.FileUpload(
            accept='.dat',
            multiple=False
        )

        self.button_prep_inv = widgets.Button(
            description='Prepare Inversion',
            disabled=False,
            # 'success', 'info', 'warning', 'danger' or ''
            button_style='',
            tooltip='Click me',
            icon='check'  # (FontAwesome names without the `fa-` prefix)
        )
        self.button_prep_inv.on_click(
            self.prepare_inv
        )
        self.vbox_inv = widgets.VBox(
            [
                self.file_elem,
                self.file_elec,
                self.file_volt,
                self.button_prep_inv,
            ]
        )

    def show(self):
        display(self.vbox_inv)

    def prepare_inv(self, button):
        """Load mesh and data, and then show the inversion settings
        """
        print('Prepare inversion')
        self.button_prep_inv.disabled = True

        # Error checking
        for widget in (self.file_elec, self.file_elem, self.file_volt):
            print(widget.value)
            if len(widget.value) == 0 or widget.value[0]['size'] == 0:
                print('Bad file upload')

        mesh = crtomo.crt_grid(
            io.BytesIO(self.file_elem.value[0].content),
            io.BytesIO(self.file_elec.value[0].content),
        )
        self.tdm = crtomo.tdMan(grid=mesh)
        self.tdm.read_voltages(
            io.StringIO(
                codecs.decode(
                    self.file_volt.value[0].content,
                    'utf-8'
                )
            )
        )

        self.button_run_inv = widgets.Button(
            description='Run Inversion',
            disabled=False,
            visible=True,
            # 'success', 'info', 'warning', 'danger' or ''
            button_style='',
            tooltip='Click me',
            icon='check'  # (FontAwesome names without the `fa-` prefix)
        )
        print('Displaying inversion button')
        display(self.button_run_inv)
        # self.vbox_inv_2 = widgets.VBox(
        #     [
        #         self.button_run_inv,
        #     ]
        # )
        self.button_run_inv.on_click(
            self.run_inv
        )
        # display(self.vbox_inv_2)

    def run_inv(self, button):
        """

        """
        self.tdm.invert(catch_output=False)
