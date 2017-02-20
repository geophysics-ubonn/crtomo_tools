#!/usr/bin/env python
from setuptools import setup
import glob
# from setuptools import find_packages
# find_packages

# under windows, run
# python.exe setup.py bdist --format msi
# to create a windows installer

scripts = glob.glob('src/*.py')

version_short = '0.1'
version_long = '0.1.0'

if __name__ == '__main__':
    setup(name='crtomo_tools',
          version=version_long,
          description='???',
          author='Maximilian Weigand',
          license='GPL-3',
          author_email='mweigand@geo.uni-bonn.de',
          url='http://www.geo.uni-bonn.de/~mweigand',
          # find_packages() somehow does not work under Win7 when creating a
          # msi # installer
          # packages=find_packages(),
          package_dir={'': 'lib'},
          packages=[
          ],
          scripts=scripts,
          # install_requires=['numpy', 'scipy', 'matplotlib'],
          )
