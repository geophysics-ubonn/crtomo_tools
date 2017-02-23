import os
import shutil
import subprocess
import glob


def check_grid_creation(directory, expect_return_code):
    pwd = os.getcwd()
    os.chdir(directory)
    return_code = subprocess.call(
        'cr_trig_create.py',
        shell=True,
    )
    # clean output directories
    output_dirs = glob.glob('tmp_grid_*')
    for directory in output_dirs:
        if os.path.isdir(directory):
            shutil.rmtree(directory)
    os.chdir(pwd)
    assert return_code == expect_return_code


def test_success_generator():
    """Look for all 'Test_*' directories and try to create grids in them. We
    expect those test to run through, i.e., return a code 0.
    """
    directories = sorted(glob.glob('Test_*'))
    for directory in directories:
        yield check_grid_creation, directory, 0


def test_fail_generator():
    """Look for all 'TestF_*' directories and try to create grids in them. We
    expect those test to FAIL, i.e., return a code of 1.
    """
    directories = sorted(glob.glob('TestF_*'))
    for directory in directories:
        yield check_grid_creation, directory, 1
