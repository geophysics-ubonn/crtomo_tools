## CRTomo tools

CRTomo user space tools and libraries. This repository provides various
Python-based tools and libraries for interacting with the complex electrical
resistivity code CRTomo, originally created by Prof. Andreas Kemna, Geophysics
Section, University of Bonn.

The actual forward modelling and inversion code CRTomo can be found here:
https://github.com/geophysics-ubonn/crtomo_stable


Capabilities
------------

Please refer to the examples-section of the documentation:

https://geophysics-ubonn.github.io/crtomo_tools/


Platform availability
---------------------

The code is mainly developed and tested on the Linux operating system, but is
increasingly also used under Windows and MacOS. Please report any issues.


Installation
------------

The package is distributed via pypi and can be installed with the package
manager of your choice:

    pip install crtomo_tools

Linux install from source::

	mkvirtualenv --python /usr/bin/python3 crtomo
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install .

