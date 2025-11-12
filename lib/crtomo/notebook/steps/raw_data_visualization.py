from .base_step import base_step
import pylab as plt

from IPython.display import display
import ipywidgets as widgets
# from ipywidgets import GridspecLayout


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
            "filters": {
                'r_min': 0,
                'r_max': None,
                'rhoa_min': None,
                'rhoa_max': None,
                'pha_min': None,
                'pha_max': None,
            },

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
        if (filt_rmin := self.input_new['filters'].get("r_min", None)):
            print('Applying rmin filter', filt_rmin)
            cr.filter("r <= {}".format(filt_rmin))
        if (filt_rmax := self.input_new['filters'].get("r_max", None)):
            print('Applying rmax filter', filt_rmax)
            cr.filter("r >= {}".format(filt_rmax))

        # if(filt_rhoa_min := self.input_new['filters'].get("rhoa_min", None)):
        #     cr.filter("rho_a <= {}".format(filt_rhoa_min))
        # if(filt_rhoa_max := self.input_new['filters'].get("rhoa_max", None)):
        #     cr.filter("rho_a <= {}".format(filt_rhoa_max))

        # TODO: Phase

        plot_r = cr.plot_histogram(column='r')
        self.results['hist_r_log10'] = plot_r

        if 'rpha' in cr.data.columns:
            plot_rpha = cr.plot_histogram(column='rpha')
            self.results['hist_rpha'] = plot_rpha

        fig_pseudo_log10_r = cr.pseudosection_type1(column='r', log10=True)
        self.results['ps_log10_r'] = fig_pseudo_log10_r

        self.results['cr'] = cr

        self.transfer_input_new_to_applied()
        self.has_run = True

    def create_ipywidget_gui(self):
        self.widgets['label_intro'] = widgets.Label(
            'This tab visualises the raw data and allows you to apply data ' +
            'filters'
        )

        # filter widgets
        self.widgets['filter_r_min_useit'] = widgets.Checkbox(
            value=True,
            description="Use this filter",
        )
        self.widgets['filter_r_min'] = widgets.FloatText(
            value=0,
            description="Filter R min",
        )
        self.widgets['filter_r_max_useit'] = widgets.Checkbox(
            value=False,
            description="Use this filter",
        )
        self.widgets['filter_r_max'] = widgets.FloatText(
            value=None,
            description="Filter R max",
        )
        self.widgets['filter_vbox'] = widgets.VBox(
            [
                widgets.HBox([
                    self.widgets['filter_r_min'],
                    self.widgets['filter_r_min_useit'],
                ]),
                widgets.HBox([
                    self.widgets['filter_r_max'],
                    self.widgets['filter_r_max_useit'],
                ]),
            ],
        )

        # the visualization output
        self.widgets['output'] = widgets.Output()

        self.widgets['button_plot'] = widgets.Button(
            description='Plot',
            disabled=False,
            # 'success', 'info', 'warning', 'danger' or ''
            button_style='',
            tooltip='Click me',
            icon='check'  # (FontAwesome names without the `fa-` prefix)
        )
        self.widgets['button_plot'].on_click(
            self.apply_next_input_from_gui
        )

        self.widgets['label_feedback'] = widgets.Label('')

        self.jupyter_gui = widgets.VBox([
            self.widgets['label_intro'],
            self.widgets['filter_vbox'],
            self.widgets['button_plot'],
            self.widgets['label_feedback'],
            self.widgets['output'],
        ])

    def apply_next_input_from_gui(self, button):
        """Generate an input dict from the gui elements and apply those new
        inputs
        """
        print('Applying input from GUI')
        feedback = self.widgets['label_feedback']

        settings = {
            "filters": {},
        }
        if self.widgets['filter_r_min_useit'].value:
            settings['filters']['r_min'] = self.widgets['filter_r_min'].value
        if self.widgets['filter_r_max_useit'].value:
            settings['filters']['r_max'] = self.widgets['filter_r_max'].value
        print('Submitting new settings:', settings)

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
