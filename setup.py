#!/usr/bin/env python
from setuptools import setup
import glob
# from setuptools import find_packages
# find_packages

# under windows, run
# python.exe setup.py bdist --format msi
# to create a windows installer

scripts = glob.glob('src/*.py')
# scripts = ['src/GRID_TOOLS/td_test.py', ]
# scripts = [],

version_short = '0.1'
version_long = '0.1.0'

# generate entry points
entry_points = {'console_scripts': []}


if __name__ == '__main__':
    setup(
        name='crtomo_tools',
        version=version_long,
        description='???',
        author='Maximilian Weigand',
        license='GPL-3',
        author_email='mweigand@geo.uni-bonn.de',
        url='http://www.geo.uni-bonn.de/~mweigand',
        entry_points={
            'console_scripts': [
                'td_test = td_test:main',
            ],
        },
        # package_dir={'': 'src/', 'crtomo': 'lib/crtomo'},
        package_dir={
            '': 'src',
            'crtomo': 'lib/crtomo'
        },
        # packages=[''],
        # package_dir={'': 'lib', 'grid_tools': 'src/GRID_TOOLS'},
        # packages=find_packages(),
        packages=['crtomo', ],
        py_modules=[
            'td_test',
        ],
        # py_modules=[
        #     splitext(basename(i))[0] for i in glob.glob("src/*.py")
        # ]
        # packages=[
        #     'crtomo',
        # ],
        scripts=scripts,
        # install_requires=['numpy', 'scipy', 'matplotlib'],
    )
