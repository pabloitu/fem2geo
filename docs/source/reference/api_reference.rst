API Documentation
=================

This page is the full reference for the public Python API. It is
auto-generated from the docstrings in the source.

Most users use ``fem2geo`` through YAML configs and never touch these
modules directly. If you are writing a custom workflow in Python, the
modules below are the building blocks every job is assembled from.

Model
-----

The :class:`~fem2geo.model.Model` class is the main handler of
``fem2geo``. It wraps a solver output mesh together with a schema and
exposes cell-wise arrays (stress, strain, displacement, ...) through
canonical attribute names.

.. automodule:: fem2geo.model
   :members:
   :no-index:
   :undoc-members:

Structural data
---------------

Lightweight containers for structural measurements loaded from CSVs.
Used as the ``data`` entry of a site.

.. automodule:: fem2geo.data
   :members:


Schema
------

Maps solver-specific array names to the canonical names that
``fem2geo`` uses everywhere. Users writing custom schemas interact with
:class:`~fem2geo.internal.schema.ModelSchema` through the
:meth:`~fem2geo.internal.schema.ModelSchema.load` dispatcher, which
accepts a built-in name, a file path, an inline dict, or an existing
instance.

.. automodule:: fem2geo.internal.schema
   :members:
   :undoc-members:

Tensor
------

Pure tensor math. These functions take numpy arrays (averaged tensors,
plane orientations, fault rakes) and return scalars, vectors, or new
tensors. They are solver-agnostic.

.. automodule:: fem2geo.utils.tensor
   :members:
   :undoc-members:

Transform
---------

Geometric transformations between coordinate conventions: line and
plane representations, ENU ↔ NED, rake ↔ spherical, grid builders for
stereonet sampling.

.. automodule:: fem2geo.utils.transform
   :members:
   :undoc-members:

Plots
-----

Stereonet plotting. Every function takes a matplotlib
``Axes`` (with the ``stereonet`` projection) as its first argument and
draws into it.

.. automodule:: fem2geo.plots
   :members:
   :undoc-members:

Projector
---------

Translates, scales, reprojects, and optionally rotates georeferenced
data into a local model frame. Used by the ``project`` job.

.. automodule:: fem2geo.projector
   :members:
   :undoc-members:

Projections
-----------

Helpers for the ``project`` job: unit conversions, CRS reprojection,
bounding-box filtering, and related utilities.

.. automodule:: fem2geo.utils.projections
   :members:
   :undoc-members:

IO
--

File loaders for solver outputs, meshes, rasters, and structural CSVs.

.. automodule:: fem2geo.internal.io
   :members:
   :undoc-members:

Runner
------

Top-level entry point for YAML-driven jobs. :func:`~fem2geo.runner.run`
loads a config file, dispatches to the requested job module, and
writes the output. :func:`~fem2geo.runner.main` is the CLI entry point
bound to the ``fem2geo`` command.

.. automodule:: fem2geo.runner
   :members:
   :undoc-members:

