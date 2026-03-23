fem2geo
=======

*From geomechanical models to structural-geology interpretation.*


.. image:: https://img.shields.io/badge/GitHub-Repository-blue?logo=github
   :target: https://github.com/pabloitu/fem2geo
   :alt: GitHub Repository

.. image:: https://img.shields.io/pypi/pyversions/fem2geo
   :alt: PyPI - Python Version

.. image:: https://img.shields.io/pypi/v/fem2geo
   :target: https://pypi.org/project/fem2geo
   :alt: PyPI - Version

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.19096967.svg
   :target: https://doi.org/10.5281/zenodo.19096967
   :alt: DOI


.. |start| image:: https://img.icons8.com/nolan/64/software-installer.png
   :target: installation.html
   :height: 42px

.. |theory| image:: https://img.icons8.com/nolan/64/literature.png
   :target: theory/index.html
   :height: 42px

.. |concepts| image:: https://img.icons8.com/nolan/64/blocks.png
   :target: concepts/index.html
   :height: 42px

.. |tutorials| image:: https://img.icons8.com/nolan/64/bicycle.png
   :target: tutorials/principal-directions.html
   :height: 42px

.. |jobs| image:: https://img.icons8.com/nolan/64/work.png
   :target: jobs/fracture.html
   :height: 42px

.. |api| image:: https://img.icons8.com/nolan/64/code--v2.png
   :target: reference/model.html
   :height: 42px


Contents
--------

+----------------------------------------------+----------------------------------------------+
| |start| **Installation Guide**               |  |theory| **Theory and Conventions**         |
+----------------------------------------------+----------------------------------------------+
| |concepts| **Core Concepts**                 | |tutorials| **Tutorials**                    |
+----------------------------------------------+----------------------------------------------+
| |jobs| **Jobs Reference**                    | |api| **API Reference**                      |
+----------------------------------------------+----------------------------------------------+

.. admonition:: Quickstart
   :class: tip

   Create an environment and install:

   .. code-block:: console

      $ python -m venv venv
      $ source venv/bin/activate
      $ pip install fem2geo

   Download the tutorials:

   .. code-block:: console

      $ fem2geo download-tutorials

   Navigate to any tutorial on ``tutorials/*`` and run:

   .. code-block:: console

      $ fem2geo config.yaml

What is fem2geo?
----------------

``fem2geo`` is a Python package for structural geology analyses on finite-element and
boundary-element model outputs. It provides a workflow to load model results, extract regions
of interest, project data into the model space, compute geomechanical quantities, and generate
multiple comparisons with geological observations.


**EXAMPLE FIGURE HERE (MODEL + PROBE + STEREOPLOT)**

Main capabilities
-----------------

- Probe principal stress/strain directions in a selected region
- Compare multiple models at the same location
- Overlay measured fracture datasets
- Compare observed and predicted fault-slip directions
- Compute slip, dilation, and combined tendency fields
- Compare fault-based Kostrov tensors with model stress or strain axes


Useful links
------------

+---------------------------------------------------------+------------------------------------------+
| .. image:: https://img.icons8.com/nolan/64/github.png   | **GitHub repository**                    |
|   :height: 42px                                         |                                          |
|   :target: https://github.com/pabloitu/fem2geo          |                                          |
+---------------------------------------------------------+------------------------------------------+
| .. image:: https://img.icons8.com/nolan/64/code.png     | **PyPI package**                         |
|   :height: 42px                                         |                                          |
|   :target: https://pypi.org/project/fem2geo             |                                          |
+---------------------------------------------------------+------------------------------------------+
| .. image:: https://img.icons8.com/nolan/64/idea.png     | **Issue tracker**                        |
|   :height: 42px                                         |                                          |
|   :target: https://github.com/pabloitu/fem2geo/issues   |                                          |
+---------------------------------------------------------+------------------------------------------+


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Get Started

   intro/installation
   intro/theory
   intro/concepts


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorials

   tutorials/principal-directions
   tutorials/model-comparison
   tutorials/fracture-analysis
   tutorials/tendency
   tutorials/resolved-shear
   tutorials/kostrov


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Jobs

   jobs/principal-directions
   jobs/fracture
   jobs/tendency
   jobs/resolved-shear
   jobs/kostrov


.. sidebar-links::
   :caption: Help & Reference
   :github:

   reference/api_reference
   Getting Help <https://github.com/pabloitu/fem2geo/issues>
   License <https://github.com/pabloitu/fem2geo/blob/master/LICENSE>