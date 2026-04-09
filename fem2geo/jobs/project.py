"""
Job: project
============
Project georeferenced data into a local cartesian model frame. The
model frame is pinned to a real-world anchor point and optionally
rotated around it. Dispatches on the input file extension: ``.csv``
is treated as a point catalog, ``.vtp``/``.vtu``/``.vtk`` as a mesh,
and ``.tif``/``.tiff`` as a GeoTIFF raster.

For rasters, ``src.xy_units`` applies to the GeoTIFF's CRS (``deg``
for lon/lat sources, ``m`` or ``km`` for projected) while
``src.z_units`` applies to the band value; the two can differ.

An intermediate azimuthal equidistant projection centered on the
anchor is used automatically; the user does not specify it. The
anchor always lands at (0, 0) in that projection.

An optional ``src.bbox`` filter in lon/lat/depth_km is applied before
projection, regardless of the source CRS.

Config reference
----------------
job: project

data:
  file: ../data/topo.vtp
  # For catalogs, declare the format and its column mapping:
  # catalog:
  #   columns: [longitude, latitude, depth]
  # For GeoTIFF rasters, optionally pick a band to drive elevation:
  # raster:
  #   z_band: 1          # omit for a flat grid positioned by the anchor

src:
  crs: epsg:32719
  xy_units: m                     # deg | m | km
  z_units: m                      # m | km
  z_positive: up                  # up | down
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
from collections import namedtuple
from pathlib import Path

import numpy as np
import pyvista as pv
from pyproj import CRS

from fem2geo.data import CatalogData
from fem2geo.internal.io import load_catalog_csv
from fem2geo.projector import Projector
from fem2geo.runner import parse_config
from fem2geo.utils.projections import (
    unit_factor, flip_z, to_lonlat, bbox_mask, reproject_xy,
)

log = logging.getLogger("fem2geoLogger")


# format handlers

Handler = namedtuple("Handler", ["name", "read", "filter", "project"])


# main entry

def run(cfg, job_dir):
    _, _, _, _, out = parse_config(cfg, job_dir)
    out_dir = out["dir"]

    _reject_legacy_schema(cfg)

    src = _parse_src(cfg.get("src", {}))
    dst = _parse_dst(cfg.get("dst", {}), src)
    data_cfg = _parse_data(cfg.get("data", {}), job_dir)
    proj = _build_projector(src, dst)

    handler = data_cfg["handler"]
    obj = handler.read(data_cfg)
    if src.get("bbox"):
        obj = handler.filter(obj, src["bbox"], src)
    result = handler.project(obj, proj)

    out_path = _fix_extension(out_dir / out.get("file", "projected.vtp"), result)
    result.save(str(out_path))
    log.info(f"Saved: {out_path}")


# schema validation

def _reject_legacy_schema(cfg):
    if "projector" in cfg:
        raise ValueError(
            "Legacy 'projector:' block is no longer supported. "
            "Use 'src:' and 'dst:' blocks instead — see the job docstring."
        )
    if "input" in cfg and "data" not in cfg:
        raise ValueError(
            "Legacy 'input:' block is no longer supported. Rename to 'data:'."
        )


# src block

def _parse_src(s):
    for k in ("crs", "xy_units", "z_units", "z_positive"):
        if k not in s:
            raise ValueError(f"src.{k} is required.")
    bbox = s.get("bbox")
    if bbox is not None and not isinstance(bbox, dict):
        raise ValueError("src.bbox must be a mapping with lon/lat/depth_km keys.")
    return {
        "crs": s["crs"],
        "xy_units": s["xy_units"],
        "z_units": s["z_units"],
        "z_positive": s["z_positive"],
        "bbox": bbox,
    }


# dst block

def _parse_dst(d, src):
    if "units" not in d:
        raise ValueError("dst.units is required ('m' or 'km').")
    if "anchor" not in d:
        raise ValueError(
            "dst.anchor is required. Pure CRS reprojection is not yet "
            "supported; specify an anchor with data and model keys."
        )

    a = d["anchor"]
    if "data" not in a or "model" not in a:
        raise ValueError("dst.anchor must have both 'data' and 'model' keys.")

    anchor_lon, anchor_lat = _parse_anchor_data(a["data"], src)

    if "depth_km" not in a["data"]:
        raise ValueError("dst.anchor.data.depth_km is required.")
    anchor_depth_km = float(a["data"]["depth_km"])

    model = a["model"]
    if len(model) != 3:
        raise ValueError("dst.anchor.model must be [x, y, z].")

    return {
        "units": d["units"],
        "anchor_lon": anchor_lon,
        "anchor_lat": anchor_lat,
        "anchor_depth_km": anchor_depth_km,
        "anchor_local": tuple(model),
        "rotation_deg": a.get("rotation_deg", 0.0),
    }


def _parse_anchor_data(ad, src):
    has_lonlat = "lon" in ad or "lat" in ad
    has_xy = "x" in ad or "y" in ad

    if has_lonlat and has_xy:
        raise ValueError(
            "dst.anchor.data cannot mix lon/lat with x/y; pick one form."
        )
    if not has_lonlat and not has_xy:
        raise ValueError(
            "dst.anchor.data must specify either lon/lat or x/y."
        )

    if has_lonlat:
        if "lon" not in ad or "lat" not in ad:
            raise ValueError("dst.anchor.data needs both lon and lat.")
        return float(ad["lon"]), float(ad["lat"])

    if "x" not in ad or "y" not in ad:
        raise ValueError("dst.anchor.data needs both x and y.")
    if CRS.from_user_input(src["crs"]).is_geographic:
        raise ValueError(
            "dst.anchor.data.x/y cannot be used with a geographic src.crs; "
            "use lon/lat instead."
        )
    lon, lat = to_lonlat(
        [float(ad["x"])], [float(ad["y"])],
        src["crs"], src["xy_units"],
    )
    return float(lon[0]), float(lat[0])


# data block

def _parse_data(d, job_dir):
    if "file" not in d:
        raise ValueError("data.file is required.")
    path = (job_dir / d["file"]).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = path.suffix.lower()
    if ext not in EXT_HANDLERS:
        raise ValueError(
            f"Unsupported input extension '{ext}'. "
            f"Use one of: {sorted(EXT_HANDLERS)}."
        )
    handler = EXT_HANDLERS[ext]

    declared = None
    for k in HANDLERS:
        if k in d:
            if declared is not None:
                raise ValueError(
                    f"data block has multiple format sub-blocks "
                    f"({declared}, {k}); pick one."
                )
            declared = k
    if declared is not None and declared != handler.name:
        raise ValueError(
            f"data.{declared} declared, but file extension '{ext}' is a "
            f"{handler.name}. Declare data.{handler.name} instead "
            f"(or remove the sub-block)."
        )

    out = {"file": path, "handler": handler}
    if handler.name == "catalog":
        cat_block = d.get("catalog") or {}
        cols = cat_block.get("columns")
        if not cols or len(cols) != 3:
            raise ValueError(
                "data.catalog.columns must be a length-3 list "
                "[x_col, y_col, z_col]."
            )
        out["columns"] = tuple(cols)
    elif handler.name == "raster":
        ras_block = d.get("raster") or {}
        z_band = ras_block.get("z_band")
        if z_band is not None:
            z_band = int(z_band)
            if z_band < 1:
                raise ValueError("data.raster.z_band must be >= 1 (1-indexed).")
        out["z_band"] = z_band
    return out


# projector construction

def _build_projector(src, dst):
    lon0, lat0 = dst["anchor_lon"], dst["anchor_lat"]
    aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    return Projector(
        src_crs=src["crs"],
        dst_crs=aeqd,
        src_xy_units=src["xy_units"],
        dst_xy_units=dst["units"],
        src_z_units=src["z_units"],
        dst_z_units=dst["units"],
        src_z_positive=src["z_positive"],
        dst_z_positive="up",
        anchor_geo=(lon0, lat0, dst["anchor_depth_km"]),
        anchor_local=dst["anchor_local"],
        rotation_deg=dst["rotation_deg"] or None,
    )


# catalog handler

def _read_catalog(data_cfg):
    cat = load_catalog_csv(data_cfg["file"], columns=data_cfg["columns"])
    log.info(f"  loaded {len(cat)} points")
    return cat


def _filter_catalog(cat, bbox, src):
    lon, lat = to_lonlat(cat.x, cat.y, src["crs"], src["xy_units"])
    depth_km = flip_z(
        cat.z * unit_factor(src["z_units"]),
        src["z_positive"], "down",
    ) / 1000.0
    mask = bbox_mask(
        lon, lat, depth_km,
        lon_range=bbox.get("lon"),
        lat_range=bbox.get("lat"),
        depth_range_km=bbox.get("depth_km"),
    )
    out = CatalogData(
        x=cat.x[mask], y=cat.y[mask], z=cat.z[mask],
        attrs={k: v[mask] for k, v in cat.attrs.items()},
    )
    log.info(f"  bbox: kept {len(out)} points")
    if len(out) == 0:
        raise ValueError("No points left after bbox filtering.")
    return out


def _project_catalog(cat, proj):
    cat = proj.transform_catalog(cat)
    poly = pv.PolyData(np.c_[cat.x, cat.y, cat.z])
    for name, arr in cat.attrs.items():
        poly.point_data[name] = arr
    return poly


# mesh handler

def _read_mesh(data_cfg):
    mesh = pv.read(str(data_cfg["file"]))
    if mesh.n_points == 0:
        raise ValueError(f"Empty mesh: {data_cfg['file']}")
    log.info(f"  loaded {mesh.n_points} points, {mesh.n_cells} cells")
    return mesh


# shared gridded pipeline (mesh, and later raster via pv.StructuredGrid)

def _filter_gridded(mesh, bbox, src):
    bounds = _bbox_in_mesh_frame(bbox, src)
    out = mesh.clip_box(bounds, invert=False)
    log.info(f"  bbox: kept {out.n_points} points, {out.n_cells} cells")
    if out.n_points == 0:
        raise ValueError("No geometry left after bbox filtering.")
    return out


def _project_gridded(mesh, proj):
    return proj.transform_mesh(mesh)


def _bbox_in_mesh_frame(bbox, src):
    """
    Convert a lon/lat/depth_km bbox into a mesh-frame
    ``(xmin, xmax, ymin, ymax, zmin, zmax)`` tuple for ``clip_box``.
    Any missing axis becomes an infinite bound.
    """
    lon_r = bbox.get("lon")
    lat_r = bbox.get("lat")
    d_r = bbox.get("depth_km")

    if lon_r is None and lat_r is None:
        x_min = y_min = -np.inf
        x_max = y_max = np.inf
    else:
        lo0, lo1 = lon_r if lon_r is not None else (-180.0, 180.0)
        la0, la1 = lat_r if lat_r is not None else (-85.0, 85.0)
        corner_lons = np.array([lo0, lo1, lo1, lo0])
        corner_lats = np.array([la0, la0, la1, la1])
        cx, cy = reproject_xy(corner_lons, corner_lats, "epsg:4326", src["crs"])
        cx = cx / unit_factor(src["xy_units"])
        cy = cy / unit_factor(src["xy_units"])
        x_min, x_max = (
            (cx.min(), cx.max()) if lon_r is not None else (-np.inf, np.inf)
        )
        y_min, y_max = (
            (cy.min(), cy.max()) if lat_r is not None else (-np.inf, np.inf)
        )

    if d_r is None:
        z_min, z_max = -np.inf, np.inf
    else:
        z_vals = flip_z(
            np.array(d_r, dtype=float) * 1000.0,
            "down", src["z_positive"],
        ) / unit_factor(src["z_units"])
        z_min, z_max = float(z_vals.min()), float(z_vals.max())

    return [x_min, x_max, y_min, y_max, z_min, z_max]


# raster handler

def _read_raster(data_cfg):
    import rasterio

    path = data_cfg["file"]
    z_band = data_cfg.get("z_band")

    with rasterio.open(str(path)) as ds:
        poly = _raster_to_grid(ds, z_band, window=None)

    log.info(
        f"  loaded {poly.n_points} points, {poly.n_cells} triangles "
        f"({len(poly.point_data) - 1} bands)"
    )
    return poly


def _filter_raster(grid, bbox, src):
    import rasterio
    from rasterio.windows import from_bounds, Window
    from rasterio.errors import WindowError

    path = grid.field_data["_source_file"][0]
    z_band = grid.field_data["_z_band"]
    z_band = int(z_band[0]) if len(z_band) else None

    with rasterio.open(str(path)) as ds:
        left, bottom, right, top = _bbox_in_raster_crs(bbox, ds)
        window = from_bounds(left, bottom, right, top, ds.transform)
        window = window.round_offsets().round_lengths()
        full = Window(0, 0, ds.width, ds.height)
        try:
            window = window.intersection(full)
        except WindowError:
            raise ValueError("No raster pixels left after bbox filtering.")
        if window.width <= 0 or window.height <= 0:
            raise ValueError("No raster pixels left after bbox filtering.")
        out = _raster_to_grid(ds, z_band, window=window)

    log.info(
        f"  bbox: kept {out.n_points} points, {out.n_cells} triangles"
    )
    return out


def _project_raster(poly, proj):
    for k in ("_source_file", "_z_band"):
        if k in poly.field_data:
            del poly.field_data[k]
    out = poly.copy()
    pts = np.asarray(out.points, dtype=float)
    X, Y, Z = proj.transform(pts[:, 0], pts[:, 1], pts[:, 2])
    out.points = np.c_[X, Y, Z]
    return out


def _raster_to_grid(ds, z_band, window):
    import rasterio
    from rasterio.windows import Window

    if window is None:
        window = Window(0, 0, ds.width, ds.height)

    w, h = int(window.width), int(window.height)
    if w < 2 or h < 2:
        raise ValueError(
            f"Raster window too small to triangulate: {w}x{h} (need >= 2x2)."
        )

    tfm = rasterio.windows.transform(window, ds.transform)
    cols = np.arange(w, dtype=float) + 0.5
    rows = np.arange(h, dtype=float) + 0.5
    cc, rr = np.meshgrid(cols, rows)
    xs = tfm.a * cc + tfm.b * rr + tfm.c
    ys = tfm.d * cc + tfm.e * rr + tfm.f

    bands = {}
    for i in range(1, ds.count + 1):
        arr = ds.read(i, window=window, masked=True).astype(float)
        arr = np.ma.filled(arr, np.nan)
        bands[_band_name(ds, i)] = arr

    if z_band is not None:
        if z_band > ds.count:
            raise ValueError(
                f"data.raster.z_band={z_band} but raster has {ds.count} band(s)."
            )
        zs = bands[_band_name(ds, z_band)]
    else:
        zs = np.zeros_like(xs)

    # flip rows so j increases northward (VTK winding)
    xs = xs[::-1, :]
    ys = ys[::-1, :]
    zs = zs[::-1, :]
    bands = {k: v[::-1, :] for k, v in bands.items()}

    valid = np.isfinite(zs)
    z_fill = np.where(valid, zs, 0.0)
    points = np.c_[xs.ravel(), ys.ravel(), z_fill.ravel()]

    faces = _build_quad_faces(valid, w, h)
    if faces.size == 0:
        raise ValueError("No valid faces built from raster (all pixels masked?).")

    poly = pv.PolyData(points, faces)
    for name, arr in bands.items():
        poly.point_data[name] = arr.ravel()
    poly.point_data["valid"] = valid.astype(np.uint8).ravel()

    poly.field_data["_source_file"] = np.array([str(ds.name)])
    poly.field_data["_z_band"] = (
        np.array([z_band], dtype=int) if z_band is not None
        else np.array([], dtype=int)
    )
    return poly


def _build_quad_faces(valid, w, h):
    v00 = valid[:-1, :-1]
    v10 = valid[:-1, 1:]
    v01 = valid[1:, :-1]
    v11 = valid[1:, 1:]
    ok = v00 & v10 & v01 & v11

    j, i = np.nonzero(ok)
    p00 = j * w + i
    p10 = j * w + (i + 1)
    p01 = (j + 1) * w + i
    p11 = (j + 1) * w + (i + 1)

    n = p00.size
    tris = np.empty((2 * n, 4), dtype=np.int64)
    tris[0::2, 0] = 3
    tris[0::2, 1] = p00
    tris[0::2, 2] = p10
    tris[0::2, 3] = p11
    tris[1::2, 0] = 3
    tris[1::2, 1] = p00
    tris[1::2, 2] = p11
    tris[1::2, 3] = p01
    return tris.ravel()


def _band_name(ds, i):
    desc = ds.descriptions[i - 1] if ds.descriptions else None
    if desc:
        return str(desc)
    return f"band_{i}"


def _bbox_in_raster_crs(bbox, ds):
    lon_r = bbox.get("lon")
    lat_r = bbox.get("lat")
    if lon_r is None and lat_r is None:
        return ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top

    lo0, lo1 = lon_r if lon_r is not None else (-180.0, 180.0)
    la0, la1 = lat_r if lat_r is not None else (-85.0, 85.0)
    corner_lons = np.array([lo0, lo1, lo1, lo0])
    corner_lats = np.array([la0, la0, la1, la1])
    cx, cy = reproject_xy(corner_lons, corner_lats, "epsg:4326", ds.crs)
    return float(cx.min()), float(cy.min()), float(cx.max()), float(cy.max())


# handler registry

HANDLER_CATALOG = Handler(
    name="catalog",
    read=_read_catalog,
    filter=_filter_catalog,
    project=_project_catalog,
)

HANDLER_MESH = Handler(
    name="mesh",
    read=_read_mesh,
    filter=_filter_gridded,
    project=_project_gridded,
)

HANDLER_RASTER = Handler(
    name="raster",
    read=_read_raster,
    filter=_filter_raster,
    project=_project_raster,
)

HANDLERS = {
    "catalog": HANDLER_CATALOG,
    "mesh": HANDLER_MESH,
    "raster": HANDLER_RASTER,
}

EXT_HANDLERS = {
    ".csv": HANDLER_CATALOG,
    ".vtp": HANDLER_MESH,
    ".vtu": HANDLER_MESH,
    ".vtk": HANDLER_MESH,
    ".tif": HANDLER_RASTER,
    ".tiff": HANDLER_RASTER,
}



def _fix_extension(out_path, mesh):
    ext = out_path.suffix.lower()
    is_poly = isinstance(mesh, pv.PolyData)
    if is_poly and ext == ".vtu":
        new_path = out_path.with_suffix(".vtp")
        log.warning(f"  output is PolyData; writing {new_path.name} instead")
        return new_path
    if (not is_poly) and ext == ".vtp":
        new_path = out_path.with_suffix(".vtu")
        log.warning(
            f"  output is UnstructuredGrid; writing {new_path.name} instead"
        )
        return new_path
    return out_path