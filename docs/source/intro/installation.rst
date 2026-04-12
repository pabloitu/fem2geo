Installation
============

.. important::

   ``fem2geo`` supports Python ``3.10`` to ``3.13``.


.. tab-set::

   .. tab-item:: Latest
      :sync: latest

      Recommended for trying the newest features, following
      development, or contributing back.

      Clone the repository:

      .. code-block:: console

         $ git clone https://github.com/pabloitu/fem2geo
         $ cd fem2geo

      Create and activate a virtual environment, then install in
      editable mode:

      .. code-block:: console

         $ python -m venv venv
         $ source venv/bin/activate
         $ pip install -e .


   .. tab-item:: Stable (PyPI)
      :sync: stable

      Recommended for regular use.

      Create and activate a virtual environment, then install from
      PyPI:

      .. code-block:: console

         $ python -m venv venv
         $ source venv/bin/activate
         $ pip install fem2geo


Tutorials
---------

Tutorials are distributed separately from the PyPI package. To download the tutorials from the **latest GitHub release** into your current directory:

.. code-block:: console

   $ fem2geo download-tutorials

This will download and extract the tutorials into ``./tutorials``.


Developer install
-----------------

For contributing to ``fem2geo``, install with the ``dev`` extras. This
adds the testing and documentation dependencies.

.. code-block:: console

   $ git clone https://github.com/pabloitu/fem2geo
   $ cd fem2geo
   $ python -m venv venv
   $ source venv/bin/activate
   $ pip install -e ".[dev]"


Running the tests
~~~~~~~~~~~~~~~~~

From the repository root, with the dev environment active:

.. code-block:: console

   $ pytest tests/

Building the documentation
~~~~~~~~~~~~~~~~~~~~~~~~~~

From the repository root, with the dev environment active:

.. code-block:: console

   $ make clean
   $ make html

Open ``docs/build/html/index.html`` in a browser to preview it.