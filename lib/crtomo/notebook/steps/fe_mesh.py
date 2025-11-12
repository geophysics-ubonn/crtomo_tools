import io
from copy import deepcopy

from IPython.display import display
import ipywidgets as widgets
from .base_step import base_step
from ipywidgets import GridBox, Layout
import pylab as plt
import crtomo


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
            'This tab allows you to load CRTomo Finite-Element meshes'
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

    def transfer_input_new_to_applied(self):
        """Make a copy of the self.input_new dict and store in
        self.input_applied

        This is complicated because some objects cannot be easily copied (e.g.,
        io.BytesIO). Therefore, each step must implement this function by
        itself.
        """
        self.input_applied = deepcopy(self.input_new)
        self.input_applied['elem_data'].seek(0)
        self.input_applied['elec_data'].seek(0)

    def apply_next_input(self):
        """

        """
        print('FE MESH: apply_next_input')
        if not self.can_run():
            return False

        print('TELL 1', self.input_new['elem_data'].tell())
        # load mesh into a grid object
        mesh = crtomo.crt_grid(
            self.input_new['elem_data'],
            self.input_new['elec_data'],
        )
        # rewind positions for future use
        self.input_new['elem_data'].seek(0)
        self.input_new['elec_data'].seek(0)

        print('TELL 2', self.input_new['elem_data'].tell())
        self.results['mesh'] = mesh

        with plt.ioff():
            fig, ax = mesh.plot_grid()
        self.results['mesh_fig'] = fig
        self.results['mesh_ax'] = ax

        with self.widgets['output_meshimg']:
            fig_mesh = self.results['mesh_fig']
            display(fig_mesh)

        self.has_run = True
        self.transfer_input_new_to_applied()
        self.persistency_store()
        return True

    def apply_next_input_from_gui(self, button):
        """Generate an input dict from the GUI elements and apply those new
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

        ret_value = self.apply_next_input()

        if ret_value:
            feedback.value = 'Mesh was loaded'
            # notify external objects
            if self.callback_step_ran is not None:
                self.callback_step_ran(self)
        else:
            feedback.value = 'There was an error loading the mesh'
