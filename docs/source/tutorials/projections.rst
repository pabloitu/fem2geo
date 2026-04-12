.. _projections:

Projecting onto a Model Frame
=============================

Geophysical datasets come in a variety of coordinate systems: catalogs in
lon/lat with depth in kilometers, surface meshes in meters, rasters in
geographic or projected CRSs. Finite-element models live in a local
cartesian frame for convenience.

The ``project`` job bridges the two. It takes a georeferenced input —
catalog, mesh, or GeoTIFF raster — and transforms it into a local model
frame (or a model with a different real-world CRS).

Every input is translated and rotated using a real-world anchor
point that lands at a chosen local-frame point. The *anchor* is a known
point whose coordinates you have in both frames. The rotation is optional. An intermediate
azimuthal equidistant projection centered on the anchor is used automatically.


Input data types
----------------

The input kind is declared as a sub-block under ``data:``:

- ``catalog`` for ``.csv`` point sets
- ``mesh`` for ``.vtp`` / ``.vtu`` / ``.vtk`` surface or volumetric meshes
- ``raster`` for ``.tif`` / ``.tiff`` GeoTIFF rasters

This tutorial walks through three configurations from ``tutorials/7_data_to_fem_space/``, one
for each input type.

Projecting a catalog
--------------------

Catalogs are CSV files with one point per row. The configuration declares
which columns hold the coordinates:

.. literalinclude:: ../../../tutorials/7_data_to_fem_space/project_catalog.yaml
   :language: yaml

Run it with:

.. code-block:: console

   $ fem2geo project_catalog.yaml

In this example:

- ``data.catalog.columns`` lists the column names holding longitude,
  latitude, and depth, in that order. All other numeric columns in the
  CSV become point-data arrays on the output.
- ``src`` describes the coordinates already in the file: geographic CRS,
  degrees for XY, kilometers for depth, positive downward.
- ``src.bbox`` drops points outside the region before projection.
  Bboxes are always in lon/lat/depth_km, regardless of the source CRS.
- ``dst.anchor`` pins the point ``(lon=-71.07, lat=-20.09, 15.6 km deep)``
  to the local-frame coordinate ``(0, 0, -14)``.
- ``dst.anchor.rotation_deg: -20`` rotates the model frame 20° clockwise
  around the vertical axis through the anchor (a negative value is
  clockwise; positive is counter-clockwise).

The output is a ``.vtp`` file with one point per catalog entry and every
numeric column preserved as point data.

.. figure:: ../_static/projections_catalog.png
   :alt: Projected earthquake catalog
   :width: 60%
   :align: center

   Earthquake catalog projected into the local model frame.

Projecting a mesh
-----------------

Surface meshes (e.g., slab interfaces) are read directly
from VTK formats. The ``mesh:`` sub-block carries the file path:

.. literalinclude:: ../../../tutorials/7_data_to_fem_space/project_slab.yaml
   :language: yaml

Run it with:

.. code-block:: console

   $ fem2geo project_slab.yaml

Points of note:

- The slab mesh was stored in UTM zone 19S (``epsg:32719``), with XY and Z
  both in meters.
- Meshes are assumed to be ENU (Z positive up) with a single unit for all
  three components, so ``src.z_units`` and ``src.z_positive`` are not
  needed and can be omitted.
- Because the bbox is specified in lon/lat/depth_km, it works identically
  whether the mesh is in UTM meters or any other CRS.
- The output keeps the mesh topology and all original point data, written
  as ``.vtu``.

.. figure:: ../_static/projections_slab.png
   :alt: Projected slab interface
   :width: 60%
   :align: center

   Slab interface projected into the local model frame.

Projecting a raster
-------------------

GeoTIFF rasters are triangulated into a surface mesh with one point per
pixel, then projected. The elevation comes from whichever band the config
selects:

.. literalinclude:: ../../../tutorials/7_data_to_fem_space/project_topo.yaml
   :language: yaml

Run it with:

.. code-block:: console

   $ fem2geo project_topo.yaml

A few raster-specific details:

- ``data.raster.z_band`` picks the band (1-indexed) that drives the z
  coordinate. Omit it for a flat grid positioned at the anchor.
- All raster bands become point-data arrays on the output, so additional
  layers (e.g. slope, classification) ride along with the elevation.
- ``src.xy_units`` describes the GeoTIFF's CRS units (``deg`` for a
  geographic raster, ``m`` or ``km`` for a projected one), while
  ``src.z_units`` describes the band value. The two are independent —
  a lon/lat DEM with elevations in meters is a normal case.
- The output is a ``.vtp`` surface mesh ready to visualize in ParaView
  alongside the model, and the slab and catalog outputs.

.. figure:: ../_static/projections_topo.png
   :alt: Projected topography raster
   :width: 60%
   :align: center

   Topography raster projected into the local model frame.

Understanding the configuration
-------------------------------

Every ``project`` configuration has four blocks: ``data``, ``src``, ``dst``,
and ``output``. They describe, in order, *what* to process, *how the data
is currently georeferenced*, *where the model frame lives*, and *where to
write the result*.

The ``data`` block
^^^^^^^^^^^^^^^^^^

Pick exactly one of ``catalog``, ``mesh``, or ``raster`` as a sub-block.
Each carries its own ``file`` and any kind-specific options.

.. code-block:: yaml

   data:
     mesh:
       file: path/to/input.vtu

For catalogs, declare the coordinate columns:

.. code-block:: yaml

   data:
     catalog:
       file: data/cat.csv
       columns: [longitude, latitude, depth]

For rasters, optionally pick the elevation band:

.. code-block:: yaml

   data:
     raster:
       file: data/dem.tif
       z_band: 1

The file extension must match the declared kind: ``.csv`` for catalog,
``.vtp``/``.vtu``/``.vtk`` for mesh, ``.tif``/``.tiff`` for raster.

The ``src`` block
^^^^^^^^^^^^^^^^^

Describes the coordinates inside the input file:

.. code-block:: yaml

   src:
     crs: epsg:32719
     xy_units: m
     z_units: m
     z_positive: up
     bbox:
       lon: [-72.5, -68.0]
       lat: [-22.0, -18.0]
       depth_km: [-10, 60]

- ``crs`` accepts any format ``pyproj`` understands: ``epsg:32719``,
  ``EPSG:4326``, proj strings, WKT, and so on.
- ``xy_units`` is ``deg`` for a geographic CRS, ``m`` or ``km`` for a
  projected CRS.
- ``z_units`` is ``m`` or ``km``, independent of ``xy_units``.
- ``z_positive`` lets the file use either convention: ``up`` for
  elevations (negative underground) or ``down`` for depths (positive
  underground). The tool handles the flip internally.
- ``bbox`` is always specified in lon/lat/depth_km *regardless of the
  source CRS*, so configs stay comparable across datasets. Any axis left
  out is unconstrained. ``depth_km`` is always positive-down.

.. note::

   Mesh inputs are assumed to be ENU: Z positive up, with XY and Z in
   the same unit.

The ``dst`` block
^^^^^^^^^^^^^^^^^

Describes the destination model frame:

.. code-block:: yaml

   dst:
     units: km
     anchor:
       data:
         lon: -71.07
         lat: -20.09
         depth_km: 15.6
       model: [0, 0, -21]
       rotation_deg: -10

- ``units`` applies to both XY and Z of the output, ``m`` or ``km``.
- ``anchor.data`` is where the anchor sits in the *input* coordinate
  system. Use ``lon``/``lat`` when the anchor is naturally a geographic
  point; use ``x``/``y`` when the anchor is given in the same projected
  CRS as ``src.crs``. ``depth_km`` is always positive-down.
- ``anchor.model`` is where that same point lands in the *output* frame,
  in ``dst.units`` with z positive up.
- ``rotation_deg`` rotates the model frame around the vertical (z) axis
  through the anchor. Positive values rotate counter-clockwise as seen
  from above, negative values clockwise. Defaults to zero.

The ``output`` block
^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   output:
     dir: results/
     file: topo_fem.vtu

Catalogs are always written as ``.vtp``. Meshes and rasters are written
as ``.vtu`` or ``.vtp`` depending on the geometry type; if the requested
extension doesn't match, the tool warns and corrects it.

See also
--------

- :doc:`../intro/user_guide`