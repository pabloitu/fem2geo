"""
Job: project
============
Project georeferenced data into a local cartesian model frame. The model frame is
pinned to a real-world anchor point and optionally rotated around it. The input kind
is declared as a sub-block under ``data:``: ``catalog`` for CSV point sets, ``mesh``
for VTK files, ``raster`` for GeoTIFFs.

For rasters, ``src.xy_units`` applies to the GeoTIFF's CRS (``deg`` for lon/lat
sources, ``m`` or ``km`` for projected) while ``src.z_units`` applies to the band
value; the two can differ.

Meshes are assumed to be ENU (Z positive up) with XY and Z in the same unit.
``src.z_units`` and ``src.z_positive`` are therefore ignored for mesh inputs and may
be omitted.

An intermediate azimuthal equidistant projection centered on the anchor is used
automatically; the user does not specify it. The anchor always lands at (0,
0) in that projection.

An optional ``src.bbox`` filter in lon/lat/depth_km is applied before projection,
regardless of the source CRS.

Config reference
----------------
job: project

data:
  # Pick exactly one of mesh / catalog / raster.
  mesh:
    file: ../data/topo.vtu
  # catalog:
  #   file: ../data/eq.csv
  #   columns: [longitude, latitude, depth]
  # raster:
  #   file: ../data/dem.tif
  #   z_band: 1                     # omit for a flat grid

src:
  crs: epsg:32719
  xy_units: m                     # deg | m | km
  z_units: m                      # m | km (omit for mesh)
  z_positive: up                  # up | down (omit for mesh)
  bbox:                           # optional
    lon: [-72.5, -68.0]
    lat: [-22.0, -18.0]
    depth_km: [-10, 60]

dst:
  units: km                       # m | km, applies to XY and Z
  anchor:
    data:                         # where the anchor sits in the input data
      lon: -71.07                 # or: x: 283554.3 (in src.crs / src.xy_units)
      lat: -20.09                 # or: y: 7777215.8
      depth_km: 15.6              # always km, positive down
    model: [0, 0, -21]            # same point in the model frame
    rotation_deg: -10             # optional, defaults to 0

output:
  dir: results/
  file: topo_fem.vtu

Example
-------
fem2geo config.yaml
"""

import logging

import numpy as np
import pyvista as pv
from pyproj import CRS

from fem2geo.data import CatalogData
from fem2geo.internal.io import load_catalog_csv, load_mesh, load_raster
from fem2geo.projector import Projector
from fem2geo.runner import resolve_output
from fem2geo.utils.projections import (
    bbox_mask,
    bbox_to_crs_bounds,
    bbox_to_src_bounds,
    flip_z,
    to_lonlat,
    unit_factor,
)

log = logging.getLogger("fem2geoLogger")

KIND_EXTS = {
    "catalog": {".csv"},
    "mesh": {".vtp", ".vtu", ".vtk"},
    "raster": {".tif", ".tiff"},
}


def require(d, prefix, *keys):
    """Raise ValueError if any of `keys` is missing from `d`."""
    for k in keys:
        if k not in d:
            raise ValueError(f"{prefix}.{k} is required.")


def run(cfg, job_dir):
    out = resolve_output(cfg, job_dir)
    out_dir = out["dir"]

    # data (parsed first to know which src keys are required)
    path, kind, sub = parse_data(cfg.get("data", {}), job_dir)

    # src
    src = dict(cfg.get("src", {}))
    if kind == "mesh":
        require(src, "src", "crs", "xy_units")
        if "z_units" in src and src["z_units"] != src["xy_units"]:
            raise ValueError(
                "src.z_units must equal src.xy_units for mesh inputs "
                "(or be omitted)."
            )
        if "z_positive" in src and src["z_positive"] != "up":
            raise ValueError(
                "src.z_positive must be 'up' for mesh inputs (ENU), " "or be omitted."
            )
        src["z_units"] = src["xy_units"]
        src["z_positive"] = "up"
    else:
        require(src, "src", "crs", "xy_units", "z_units", "z_positive")

    bbox = src.get("bbox")
    if bbox is not None and not isinstance(bbox, dict):
        raise ValueError("src.bbox must be a mapping with lon/lat/depth_km keys.")

    # dst
    dst = cfg.get("dst", {})
    require(dst, "dst", "units", "anchor")
    anchor = dst["anchor"]
    require(anchor, "dst.anchor", "data", "model")
    require(anchor["data"], "dst.anchor.data", "depth_km")
    model = anchor["model"]
    if len(model) != 3:
        raise ValueError("dst.anchor.model must be [x, y, z].")

    lon0, lat0 = parse_anchor_data(anchor["data"], src)
    depth_km = float(anchor["data"]["depth_km"])
    rotation_deg = anchor.get("rotation_deg", 0.0) or None

    # projector
    aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    proj = Projector(
        src_crs=src["crs"],
        dst_crs=aeqd,
        src_xy_units=src["xy_units"],
        dst_xy_units=dst["units"],
        src_z_units=src["z_units"],
        dst_z_units=dst["units"],
        src_z_positive=src["z_positive"],
        dst_z_positive="up",
        anchor_geo=(lon0, lat0, depth_km),
        anchor_local=tuple(model),
        rotation_deg=rotation_deg,
    )

    # dispatch
    if kind == "catalog":
        cat = load_catalog_csv(path, columns=sub["columns"])
        log.info(f"  loaded {len(cat)} points")
        if bbox:
            cat = filter_catalog(cat, bbox, src)
        cat = proj.transform_catalog(cat)
        result = pv.PolyData(np.c_[cat.x, cat.y, cat.z])
        for name, arr in cat.attrs.items():
            result.point_data[name] = arr

    elif kind == "mesh":
        mesh = load_mesh(path)
        if bbox:
            bounds = bbox_to_src_bounds(
                bbox,
                src["crs"],
                src["xy_units"],
                src["z_units"],
                src["z_positive"],
            )
            mesh = mesh.clip_box(bounds, invert=False)
            log.info(f"  bbox: kept {mesh.n_points} points, {mesh.n_cells} cells")
            if mesh.n_points == 0:
                raise ValueError("No geometry left after bbox filtering.")
        result = proj.transform_mesh(mesh)

    elif kind == "raster":
        window = raster_window(path, bbox) if bbox else None
        poly = load_raster(path, z_band=sub.get("z_band"), window=window)
        result = poly.copy()
        result.points = proj.transform_points(poly.points)

    out_path = output_path(out_dir / out.get("file", "projected.vtp"), result)
    result.save(str(out_path))
    log.info(f"Saved: {out_path}")


# config parsing


def parse_anchor_data(ad, src):
    has_lonlat = "lon" in ad or "lat" in ad
    has_xy = "x" in ad or "y" in ad

    if has_lonlat and has_xy:
        raise ValueError("dst.anchor.data cannot mix lon/lat with x/y; pick one form.")
    if not has_lonlat and not has_xy:
        raise ValueError("dst.anchor.data must specify either lon/lat or x/y.")

    if has_lonlat:
        require(ad, "dst.anchor.data", "lon", "lat")
        return float(ad["lon"]), float(ad["lat"])

    require(ad, "dst.anchor.data", "x", "y")
    if CRS.from_user_input(src["crs"]).is_geographic:
        raise ValueError(
            "dst.anchor.data.x/y cannot be used with a geographic src.crs; "
            "use lon/lat instead."
        )
    lon, lat = to_lonlat(
        [float(ad["x"])],
        [float(ad["y"])],
        src["crs"],
        src["xy_units"],
    )
    return float(lon[0]), float(lat[0])


def parse_data(d, job_dir):
    declared = [k for k in KIND_EXTS if k in d]
    if not declared:
        raise ValueError(
            f"data must declare one of {sorted(KIND_EXTS)} as a sub-block."
        )
    if len(declared) > 1:
        raise ValueError(f"data has multiple sub-blocks ({declared}); pick one.")
    kind = declared[0]
    sub = d[kind] or {}

    require(sub, f"data.{kind}", "file")
    path = (job_dir / sub["file"]).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = path.suffix.lower()
    if ext not in KIND_EXTS[kind]:
        raise ValueError(
            f"data.{kind}.file has extension '{ext}', expected one of "
            f"{sorted(KIND_EXTS[kind])}."
        )

    if kind == "catalog":
        require(sub, "data.catalog", "columns")
        cols = sub["columns"]
        if len(cols) != 3:
            raise ValueError(
                "data.catalog.columns must be a length-3 list " "[x_col, y_col, z_col]."
            )
        sub = dict(sub)
        sub["columns"] = tuple(cols)

    elif kind == "raster":
        z_band = sub.get("z_band")
        if z_band is not None:
            z_band = int(z_band)
            if z_band < 1:
                raise ValueError("data.raster.z_band must be >= 1 (1-indexed).")
            sub = dict(sub)
            sub["z_band"] = z_band

    return path, kind, sub


# pipeline helpers


def filter_catalog(cat, bbox, src):
    lon, lat = to_lonlat(cat.x, cat.y, src["crs"], src["xy_units"])
    depth_km = (
        flip_z(
            cat.z * unit_factor(src["z_units"]),
            src["z_positive"],
            "down",
        )
        / 1000.0
    )
    mask = bbox_mask(
        lon,
        lat,
        depth_km,
        lon_range=bbox.get("lon"),
        lat_range=bbox.get("lat"),
        depth_range_km=bbox.get("depth_km"),
    )
    out = CatalogData(
        x=cat.x[mask],
        y=cat.y[mask],
        z=cat.z[mask],
        attrs={k: v[mask] for k, v in cat.attrs.items()},
    )
    log.info(f"  bbox: kept {len(out)} points")
    if len(out) == 0:
        raise ValueError("No points left after bbox filtering.")
    return out


def raster_window(path, bbox):
    import rasterio
    from rasterio.errors import WindowError
    from rasterio.windows import Window, from_bounds

    with rasterio.open(str(path)) as ds:
        bounds = bbox_to_crs_bounds(bbox, ds.crs)
        if bounds is None:
            bounds = (
                ds.bounds.left,
                ds.bounds.bottom,
                ds.bounds.right,
                ds.bounds.top,
            )
        left, bottom, right, top = bounds
        win = from_bounds(left, bottom, right, top, ds.transform)
        win = win.round_offsets().round_lengths()
        full = Window(0, 0, ds.width, ds.height)
        try:
            win = win.intersection(full)
        except WindowError:
            raise ValueError("No raster pixels left after bbox filtering.")
        if win.width <= 0 or win.height <= 0:
            raise ValueError("No raster pixels left after bbox filtering.")
        return win


def output_path(path, mesh):
    ext = path.suffix.lower()
    is_poly = isinstance(mesh, pv.PolyData)
    if is_poly and ext == ".vtu":
        new = path.with_suffix(".vtp")
        log.warning(f"  output is PolyData; writing {new.name} instead")
        return new
    if (not is_poly) and ext == ".vtp":
        new = path.with_suffix(".vtu")
        log.warning(f"  output is UnstructuredGrid; writing {new.name} instead")
        return new
    return path
