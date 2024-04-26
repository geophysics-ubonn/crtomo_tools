#!/usr/bin/env python
import io
import crtomo.notebook.nb as crtnb


wflow = crtnb.processing_workflow_v1()

data1 = io.BytesIO()
data1.write(open('data/20181130_01_profil1_normal.bin', 'rb').read())
data1.seek(0)
data2 = io.BytesIO()
data2.write(open('data/20181130_03_profil2_reziprok.bin', 'rb').read())
data2.seek(0)

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

wflow.step_data_import.set_input_new(settings)
print('Step1, can we run?', wflow.step_data_import.can_run())
print('Step2, can we run?', wflow.step_raw_visualization.can_run())
print('Running step1')
wflow.step_data_import.apply_next_input()

print('Step2, can we run?', wflow.step_raw_visualization.can_run())
wflow.step_raw_visualization.apply_next_input()
