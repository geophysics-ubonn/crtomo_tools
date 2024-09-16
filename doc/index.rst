.. crtomo tools documentation master file, created by
   sphinx-quickstart on Tue Feb 21 17:06:04 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to crtomo tools's documentation!
========================================

This document describes the scripts and libraries of the *crtomo_tools* package.
These tools are written in the Python programming language and provide command
line tools for common tasks such as generating triangular finite-element meshes
for CRMod/CRTomo, or modifying existing grids.

Also included is the Python module *crtomo*, which aims at providing an
interface to all functions of CRMod and CRTomo. Thus, CRMod/CRTomo can be fully
controlled using Python. This also includes retrieving any data output produced
by either of the programs.

Getting started
---------------

* Read through the rest of this page and make yourself familiar with

  * the different semi-separate modules/programs that form the complete
    software stack
  * the basics of geoelectrical tomography. Please also refer to the additional
    literature listed near the end for further information

* Install the required tools
* Head over to the examples section, select a few examples, and try to execute
  and understand them in your work environment

  * :doc:`_examples/02_simple_inversion/plot_05_single_freq_inversion_CR`

Application/Module Overview
---------------------------
The workflow for the analysis of geoelectric measurements makes use of three different software packages. Each software package implements a key aspect of the workflow:
* CRTomoMod (https://github.com/geophysics-ubonn/crtomo_stable): This is the
  core modeling and inversion code, implementing the Finite-Element based
  forward solver, as well as the non-linear, iterative, Gauss-Newton inversion
  scheme.
* crtomo-tools (https://github.com/geophysics-ubonn/crtomo_tools): This is a
  Python wrapper and tool library built around CRTomoMod.
* REDA (https://github.com/geophysics-ubonn/reda): Reda is the 'Reproducible
  Electrical Data Analysis' that deals with data import/export and processing,
  as well as the generation of measurement configurations. This library is
  mostly required when dealing with real-world data.

A few notes on the structure of the Python interfaces
-----------------------------------------------------

The *crtomo-tools* software package consists of one Python library, **crtomo**,
and multiple, executable scripts.
While the Python library forms the core functionality of the *crtomo-tools*,
the scripts are meant to be executed within a terminal and operate on certain
files and/or directory structures.

When working in a Jupyter environment, usually only the **crtomo** library is
required. However, certain functionality is only available through the scripts,
sometimes requiring the use of either a command invocation from within Python,
or the use of a terminal application (note that the Jupyter Lab environment
also provides a terminal window!).

Functionality within the **crtomo** library is distributed over various
submodules. In the beginning, the :py:class:`crtomo.tdManager.tdMan` instance
is most relevant. It represents on complex-resistivity modeling and/or
inversion instance and provides means to handle/create meshes, modify the
subsurface models, generate or load measurement configurations and measurement
data, conduct the actual modeling/inversion, and finally, assess and plot the
results.
Therefore, most usages of the library start with the following lines of code: ::

    import crtomo
    tdm = crtomo.tdMan()

The ``tdm`` instance has multiple instances of relevant submodules:

* ``tdm.grid`` points to a :py:class:`crtomo.grid.crt_grid` instance for
  handling meshes

A short introduction to electrical modeling and inversion
---------------------------------------------------------

.. mermaid::

   flowchart LR
   mod
   mod("State of physical parameter in the subsurface")
   inv
   inv("Observations / Measurements")
   mod --> |'Forward modeling'| inv
   inv --> |'Inverse modeling'| mod

Electrical Modeling
^^^^^^^^^^^^^^^^^^^

We here understand the term *electrical modeling* as the process of computing
electrical potentials in a certain region (e.g., the subsurface), given the
location of a current injection dipole.
Correspondingly, we need the following information to proceed:

* a discretisation of the subsurface in the form of a Finite-Element mesh.
  Usually this mesh is comprised of triangles, but formulations for quadrangles
  also exist.
* an electrical subsurface model which associates a constant electrical
  (complex) resistivity with each of the mesh cells.
  The electrical resistivities are collected in a **model vector**
  :math:`\textbf{m}`, with the number of model parameters being :math:`M`.
* the location of injection and modeling electrodes. Usually we model
  four-point spreads, consisting of two electrodes to inject current into the
  soil, and another two electrodes to measure a corresponding voltage between
  them.
  The current electrodes are commonly denoted by *A* and *B*, while the
  measurement dipole is comprised of the electrodes *M* and *N*.
* it is common to employ tomographic measurement setups with tens, or even
  hundreds, of electrodes. Then, multiple four-point spreads are defined from
  this set of electrodes. A group of four-point spreads is called a
  **configuration**, or **config**.
* the actual forward modeling then solves the Poisson equation for the given
  geometry and resistivity distribution by means of a Finite-Element
  formulation. The primary result is distribution of electrical potentials at
  the *mesh nodes*. By selecting potentials at the location of potential
  electrodes, and computing a difference, a synthetic **measurement voltage**
  is computed.
* The data array consisting of all four-point measurements are referred to as
  the data vector :math:`\textbf{d}`, with the number of measurements being
  :math:`D`.
* Important: The forward modeling of the geoelectrical problem is *unique*.
  Given a set of model parameters and measurement configurations, there is only
  one set of measurements to be generated on the basis of the physical
  description of the problem.

.. note::

   It is common to compute the **transfer resistance** by dividing the
   measurement voltage :math:`U_{MN}` by the (known) injection current
   :math:`I_{AB}`:

   .. math::

        R_T = \frac{U_{MN}}{I_{AB}}

   When complex measurements are conducted, both the injection current and the
   resulting measured voltage are complex values, leading to a complex
   **transfer impedance**:

   .. math::

        \hat{Z}_T = \frac{\hat{U}_{MN}}{\hat{I}_{AB}}


Inverse Modeling
^^^^^^^^^^^^^^^^

We refer the process of estimating a best-fitting (defined later) model
:math:`\textbf{m}`, given a set of measurements :math:`\textbf{d}`.
The inverse step is not as straight forward as the forward modeling step, as it
is inherently *non-unique*: Usually there are far more model cells than there
are independent measurements available, leading to an under-determined system
of equations.
In addition, the forward model is of a non-linear nature, requiring an
iterative approach to solving the inverse problem.
A further complication arises due to data noise, which can take various forms
and further complicates the computation of **robust** and **reliable** inverse
model results.

The inverse step, here, is approached using a **non-linear least-square
regression** method, the *Gauss-Newton Inversion Scheme*.
The objective here is to iteratively minimise the cost function

.. math::

   \Psi(\mathbf{m}) = \Psi_d(\mathbf{m}) + \lambda \Psi_m{\mathbf{m}}

   \Psi_d(\mathbf{m}) = \sum_{i=1}^D \left|
        \frac{d_i - f_i(\mathbf{m})}{\epsilon_i}
       \right|_2^2

**TODO: finish**

RMS Error
^^^^^^^^^

The root-mean-square error corresponding to the data misfit function is defined
as:

.. math::

    \epsilon_\text{RMS} = \sqrt{
        \frac{1}{N}\sum_i^N \left|
            \frac{d_i - f(\textbf{m})_i}{\epsilon_i}
         \right|^2
    }


The RMS is a measure of how good or bad the forward model was fitted to the
measurements, within the specified error margins.
It sums the differences between the measurements and the response of the
forward model (synthetic measurements) weighted with the assumed error.
A sum of one indicates that the measurements are correctly described by the
forward model generated by the inversion.
Note that this measure is of an averaging nature, and a few measurements will
have individual misfits (error weighted residuals) well different from one.
This is what we seek to do.

.. note:

   The choice of assumed error values is often a hard one, as robust and
   reliable statistical means of determining data errors are mostly not
   available.

   It is quite common to adapt a heuristic approach, in which error parameters
   are succcessively decreased until a data rms error of 1 ist reached, or
   until obvious image artifacts emerge in the final image results.

A RMS value smaller than one means that the sum of differences between
measurements and model responses is smaller than the assumed error, on average.
We call this 'over-fitted'. Either the error assumption is wrong (the actual
error is smaller than expected), or the inversion introduced artificial data
into the process that has nothing to do with structures found in the
subsurface. In most cases the error values are just too small.

A RMS value greater than one indicates that the data was not described within
the given error estimates, and shows larger average residuals (:math:`|d_k -
f_k|`) than expected by the error estimates :math:`\epsilon_k`: :math:`|d_k
-f_k| < \epsilon_k`. This holds true ONLY for the average of all measurements,
and therefore some data points :math:`d_i` can be estimated within the expected
error margin while some show larger residuals.

Complex impedances
^^^^^^^^^^^^^^^^^^

The main feature of CRTomo is that it deals with complex impedances as input
and output parameters.
This allows the easy modeling and inversion of electrical low-frequency
polarisation effects, such as encountered in the spectral electrical impedance
spectroscopy (SIP), spectral electrical impedance tomography (sEIT), or complex
resistivity imaging.

The complex electrical impedance consists of an in-phase and an out-phase
component, but can also be written terms of a magnitude and a phase value:

:math:`\hat{Z} = Z' + 1j Z'' = |\hat{Z}| e^{1j \phi}`

.. note::

   The common electrical resistivity tomopgrahy, ERT, only deals with the
   magnitude of the electrical impedance, :math:`|\hat{Z}|`.

Further Reading
---------------

* Kemna, A.: Tomographic inversion of complex resistivity – theory and
  application, Ph.D.  thesis, Ruhr-Universität Bochum,
  https://doi.org/10.1111/1365-2478.12013, 2000.
* Binley, A.: Resistivity and Induced Polarization -- Theory and Applications to the Near-Surface Earth
  https://www.cambridge.org/core/books/resistivity-and-induced-polarization/A767136D8C584D3820D1A810381891ED

Contents
--------

.. toctree::
   :maxdepth: 2

   crtomo/crtomo.rst
   grid_creation.rst
   _examples/index.rst
   contributing.rst
   scripts/grid_tools/modules.rst
   api/crtomo.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

