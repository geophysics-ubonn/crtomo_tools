#!/usr/bin/env python
import os
import subprocess


def test_script():
    return_code = subprocess.call(
        'volt_histogram',
        shell=True,
    )
    if os.path.isfile('volt_histogram.png'):
        os.unlink('volt_histogram.png')
    assert return_code == 0
