Developing the crtomo-tools and the documentation
=================================================

Documentation
-------------

For documentation purposes we use the sphinx documentation system
(https://www.sphinx-doc.org) and various plugins. Most notable we use the
sphinx-gallery (https://sphinx-gallery.github.io/) to present examples (located
in the **examples/** subdirectory of the repository.

In order to build the documentation it should suffice to install the required
packages listed in the files **requirements.txt** and **requirements_doc.txt**.
If you use virtualenvs, a helper script **recreate_venv.sh** creates a
virtualenv *crtomo* that should be readily usable.

Build the documentation by entering the **doc** directory and call::

   make html

The initial build will take some time, as the examples include some modeling
and inversion runs.
However, results will be cached unless the specific example scripts are
changed, and subsequent calls to ```make html``` should be significantly
faster.

.. note::

   It is advisable to rebuild the entire documentation before opening a merge
   request. Sometimes errors only show after a complete rebuild.
