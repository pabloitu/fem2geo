Installation
============

.. important::

   This library supports ``3.10 <= python <= 3.13``.


Latest Version
--------------

This option is recommended for development, trying the newest features, and contributing.

1. Using only ``pip``
~~~~~~~~~~~~~~~~~~~~~

First, clone the **fem2geo** source code into a new directory:

.. code-block:: console

   $ git clone https://github.com/pabloitu/fem2geo
   $ cd fem2geo

Create and activate a virtual environment, then install **fem2geo** in editable mode:

.. code-block:: console

   $ python -m venv venv
   $ source venv/bin/activate
   $ pip install -e .

.. note::

   To *update* **fem2geo** at a later date, execute:

   .. code-block:: console

      $ git pull
      $ pip install -e . -U


2. Developer install (recommended for contributors)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For development, install with the ``dev`` extras. This installs testing and documentation dependencies (including Sphinx):

.. code-block:: console

   $ git clone https://github.com/pabloitu/fem2geo
   $ cd fem2geo
   $ python -m venv venv
   $ source venv/bin/activate
   $ pip install -e ".[dev]"

This enables running tests and building documentation locally.

.. tip::

   Run the test suite:

   .. code-block:: console

      $ pytest

   Build the documentation:

   .. code-block:: console

      $ make -C docs html


Latest Stable Release
---------------------

This option is recommended for regular users.

From the ``PyPI`` repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create and activate a virtual environment, then install **fem2geo**:

.. code-block:: console

   $ python -m venv venv
   $ source venv/bin/activate
   $ pip install fem2geo


Tutorials (download separately)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Tutorials are distributed separately (not included in the PyPI package). To download the tutorials for the **latest GitHub release** into your current directory:

.. code-block:: console

   $ fem2geo download-tutorials

This will download and extract the tutorials into ``./tutorials``.


For Developers
--------------

We recommend using a dedicated virtual environment for development. For contributing to the **fem2geo** codebase, please consider `forking the repository <https://docs.github.com/articles/fork-a-repo>`_
and `creating pull requests <https://docs.github.com/articles/creating-a-pull-request>`_ from there.

.. code-block:: console

   $ git clone https://github.com/${your_fork}/fem2geo
   $ cd fem2geo
   $ python -m venv venv
   $ source venv/bin/activate
   $ pip install -e ".[dev]"

This will install and configure the unit-testing, linting, and documentation packages.