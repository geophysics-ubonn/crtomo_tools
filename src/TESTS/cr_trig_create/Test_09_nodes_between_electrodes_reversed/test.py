import os
import shutil
import subprocess
import glob


def check_grid_creation(expect_return_code, m):
    # TODO: Fix?
    pwd = os.getcwd()
    test_dir = 'Test_09_nodes_between_electrodes_reversed'
    if not os.path.basename(os.getcwd()) == test_dir:
        os.chdir(test_dir)
    print(os.getcwd())
    return_code = subprocess.call(
        'cr_trig_create -m {0}'.format(m),
        shell=True,
    )
    # clean output directories
    output_dirs = glob.glob('tmp_grid_*')
    for directory in output_dirs:
        if os.path.isdir(directory):
            shutil.rmtree(directory)
    os.chdir(pwd)
    assert return_code == expect_return_code


def test_m_generator():
    """Test different -m switches
    """
    m = (0, 1, 2, 3)
    for mval in m:
        yield check_grid_creation, 0, mval
