#!/usr/bin/env python
"""

"""
import os
import shutil
import io
import codecs
import importlib.resources

import ipywidgets as widgets
from IPython.display import display
from IPython.display import IFrame
from ipywidgets import GridBox, Layout
import pylab as plt

# import reda
import crtomo

from .steps.base_step import base_step
from .steps.fe_mesh import step_fe_mesh
from .steps.data_import import step_data_import
from .steps.raw_data_visualization import step_raw_visualization


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
        # tab widget (defined down below in the corresponding gui-building
        # function)
        self.step_list = [
            self.step_fe_mesh,
            self.step_data_import,
            self.step_raw_visualization,
            self.step_inversion,
            self.step_inv_analysis,
        ]

        # define step association
        # note that this process here does not define HARD dependencies, but
        # merely indicates a suggested path through the tabs. Hard dependencies
        # (i.e., steps that must run before a given step can be worked with)
        # are defined in the step init functions.
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

        self.jupyter_gui = None

        if prepare_gui:
            self.prepare_jupyter_gui()

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
            '<a href="https://geophysics-ubonn.github.io/crtomo_tools/" ' +
            'target="_blank">CRTomo Online Help</a>' +
            ' - ' +
            '<a href="https://geophysics-ubonn.github.io/reda/" ' +
            'target="_blank">REDA Online help</a>'
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
    """OLD DEPRECATED

    These were first testing stages.

    """
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
