import os
import pickle
from copy import deepcopy
import io

import reda

from .base_step import base_step

import ipywidgets as widgets
from ipywidgets import GridspecLayout


class step_data_import(base_step):
    """A simple data importer step. Loads only Syscal .bin files at the moment.

    """
    def __init__(self, persistent_directory=None):
        super().__init__(persistent_directory=persistent_directory)
        self.name = 'data_import'
        self.title = 'Data Import'
        self.help_page = 'data_import.html'
        self.required_steps = [
            'fe_mesh',
        ]

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
        Returns True only for a successful application of new input

        """
        if not self.can_run():
            print("ERROR: CANNOT RUN")
            return False

        step_fe_mesh = self.find_previous_step(self, 'fe_mesh')

        import_settings = self.input_new['importer_settings']['syscal_bin']
        TDIP1 = reda.TDIP()
        TDIP1.import_syscal_bin(
            self.input_new['data_1'],
            reciprocals=import_settings['data_1']['reciprocals'],
        )
        CR1 = TDIP1.to_cr()
        print('Computing K factors for CR1')
        CR1.compute_K_numerical(
            {'mesh': step_fe_mesh.results['mesh'], },
            fe_code='crtomo'
        )
        print('done')
        print(CR1.data)
        self.results['cr1'] = CR1

        if self.input_new['data_2'] is not None:
            TDIP2 = reda.TDIP()
            TDIP2.import_syscal_bin(
                self.input_new['data_2'],
                reciprocals=import_settings['data_2']['reciprocals'],
            )
            CR2 = TDIP2.to_cr()
            CR2.compute_K_numerical(
                {'mesh': step_fe_mesh.results['mesh'], },
                fe_code='crtomo'
            )
            self.results['cr2'] = CR2

            CR_merge = reda.CR()
            CR_merge.add_dataframe(CR1.data)
            CR_merge.add_dataframe(CR2.data)
            CR_merge.compute_K_numerical(
                {'mesh': step_fe_mesh.results['mesh'], },
                fe_code='crtomo'
            )
            self.results['cr_merge'] = CR_merge
        else:
            self.results['cr2'] = None
            self.results['cr_merge'] = CR1

        self.transfer_input_new_to_applied()
        self.has_run = True

        self.persistency_store()

        return True

    def create_ipywidget_gui(self):
        self.jupyter_gui = GridspecLayout(
            4, 4
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
            "Syscal .bin file, data file 2 (optional, usually reciprocal file)"
        )
        self.jupyter_gui[2, 0] = self.widgets['label_data2']
        self.widgets['upload_data2'] = widgets.FileUpload(
            accept='.bin',
            multiple=False
        )
        self.jupyter_gui[2, 1] = self.widgets['upload_data2']

        self.widgets['dat2_check_is_reciprocal'] = widgets.Checkbox(
            value=True,
            description="data set is reciprocal",
        )
        self.widgets['dat2_nr_elecs'] = widgets.BoundedIntText(
            value=48,
            min=1,
            step=1,
            description="Nr of electrodes",
        )

        self.jupyter_gui[2, 2] = self.widgets['dat2_check_is_reciprocal']
        self.jupyter_gui[2, 3] = self.widgets['dat2_nr_elecs']

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
        return_value = self.apply_next_input()

        if return_value:
            feedback.value = 'Data was successfully imported'
            # notify external objects
            if self.callback_step_ran is not None:
                self.callback_step_ran(self)
        else:
            feedback.value = ''.join((
                'There was an error importing the data. ',
                'Check the Jupyter log',
            ))

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
