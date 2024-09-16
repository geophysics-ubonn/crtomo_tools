Contributing to the crtomo-tools
================================

As an open-source project, crtomo_tools always welcomes contributions from the
community. Here, we offer guidance for 3 different ways of contributing with
increasing levels of required coding proficiency.

A. Submit a bug report
----------------------

If you experience issues with the crtomo_tools, or miss a certain feature,
please `open a new issue on GitHub
<https://github.com/geophysics-ubonn/crtomo_tools/issues>`__. To do so, you
need to `create a GitHub account <https://github.com/join>`__.

B. Send us your example
-----------------------

We are constantly looking for interesting usage examples of crtomo_tools. If
you have used the package and would like to contribute your work to the
examples, please attach your script to a new github issue. Make sure that the
individual steps in your Python script are documented according
to the `sphinx-gallery syntax
<http://sphinx-gallery.readthedocs.io/en/latest/tutorials/plot_notebook.html>`__.

C. Contribute to the code
-------------------------

.. note::

    To avoid redundant work, please contact us before you start working on a
    non-trivial feature.

The preferred way to contribute to the crtomo_tools code base is via the *fork
and pull* collaborative model pull request (PR) on GitHub, described `here
<https://help.github.com/en/articles/about-collaborative-development-models>`__.
The general concept of working with pull requests is explained `here
<https://guides.github.com/introduction/flow>`__ and involves the following
steps:

1. Fork the repository
++++++++++++++++++++++

If you are a first-time contributor, you need `a GitHub account
<https://github.com/join>`__ and your own copy ("fork") of the code.
To do so, go to https://github.com/geophysics-ubonn/crtomo_tools and click the
"Fork button" in the upper right corner.
This will create an identical copy of the complete code base under your
username on the GitHub server.
Clone this repository to your local disk:

.. code:: bash

  git clone https://github.com/YOUR_USERNAME/crtomo_tools

After that you can install the software as usual.

2. Create a feature branch
++++++++++++++++++++++++++

Go to the source folder and create a feature branch to hold your changes. It is
advisable to give it a sensible name describing the overall topic of the
proposed changes, such as ``new_plot_script``.

.. code:: bash

  cd crtomo_tools
  git checkout -b new_plot_script

3. Start making your changes
++++++++++++++++++++++++++++

Go nuts! Add and modify files and regularly commit your changes with meaningful
commit messages.
Remember that you are working in your own personal copy and in case you break
something, you can always go back.

.. code:: bash

  git add new_file1 new_file2 modified_file1
  git commit -m "implement new plot method method after Authors et al 2019"

4. Test your code
+++++++++++++++++

Make sure that everything works as expected. New functions should always contain
a docstring with a test:

.. code:: python

  def sum(a, b):
      """Return the sum of `a` and `b`.

      Examples
      --------
      >>> a = 1
      >>> b = 2
      >>> sum(a,b)
      3
      """
      return a + b

Docstrings are not yet run automatically, but will be in the future.

5. Submit a pull request
++++++++++++++++++++++++

Once you implemented a functioning new feature, make sure your GitHub repository
contains all your commits:

.. code:: bash

  git push origin new_plot_script

After pushing, you can go to GitHub and you will see a green PR button.
Describe your changes in more detail.
Once reviewed by the core developers, your PR will be merged to the main
repository.

6. Updating your work with changes from upstream
++++++++++++++++++++++++++++++++++++++++++++++++

While you work on your forked repository, sometimes changes are commited to the
main repository (usually called **upstream**).
You do NOT need to delete your forked repository and refork to apply these
changes to your own fork.
Follow the procedure described `here
<https://help.github.com/en/articles/syncing-a-fork>`__

Only the first time, add the main repository as a remote to your local (cloned)
git repository:

   git remote add upstream https://github.com/geophysics-ubonn/crtomo_tools.git

Then, to update the local branch **new_plot_script** with the newest changes
of the upstream branch **master**, execute the following  commands:

.. code:: bash

   git fetch upstream
   git checkout new_plot_script
   git merge upstream/master

Update your forked repository branch **new_plot_script** on github:

.. code:: bash

   git push


D. Extending the documentation
------------------------------

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
