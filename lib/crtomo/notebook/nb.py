#!/usr/bin/env python
"""

"""
import io
import codecs

import ipywidgets as widgets
from IPython.core.display import display

import crtomo


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
