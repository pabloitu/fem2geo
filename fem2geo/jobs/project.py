"""
Job: project
============
Project georeferenced data (point catalogs or PyVista meshes) into a FEM
coordinate frame. Dispatches on the input file extension: ``.csv`` is
treated as a point catalog, ``.vtp``/``.vtu``/``.vtk`` as a mesh.

The transform is configured by a Projector built from the ``projector:``
block. An optional ``bbox:`` filter is applied in lon/lat/depth_km
*before* projection, regardless of the source CRS.

Output is always a PyVista file. Mismatches between the requested
extension and the actual mesh type (PolyData vs UnstructuredGrid) are
auto-corrected with a warning.

Config reference
----------------
job: project

projector:
  src_crs: epsg:4326                # required
  dst_crs: epsg:32719               # required
  src_xy_units: deg                 # deg | m | km
  dst_xy_units: km                  # m | km
  src_z: {units: km, positive: down}
  dst_z: {units: km, positive: up}
  anchor:                           # optional, both keys required together
    geo: [lon, lat, depth_km]
    fem: [x, y, z]
  rotation_deg: -10                 # optional, requires anchor

input:
  file: ../data/cat.csv             # csv | vtp | vtu | vtk
  columns: [longitude, latitude, depth]   # csv only
  mesh_units: m                     # mesh only, defaults to projector.src_xy_units
  bbox:                             # optional pre-filter, all keys optional
    lon: [-75.5, -60.0]
    lat: [-22.0, -18.0]
    depth_km: [0, 60]

output:
  dir: results/
  file: out.vtp

Example
-------
fem2geo config.yaml
"""

import logging
from pathlib import Path

import numpy as np
import pyvista as pv

from fem2geo.data import CatalogData
from fem2geo.internal.io import load_catalog_csv
from fem2geo.projector import Projector
from fem2geo.runner import parse_config
from fem2geo.utils.projections import (
    unit_factor, flip_z, to_lonlat, bbox_mask,
)

log = logging.getLogger("fem2geoLogger")

CSV_EXTS = {".csv"}
MESH_EXTS = {".vtp", ".vtu", ".vtk"}


def run(cfg: dict, job_dir: Path) -> None:
    _, _, _, _, out = parse_config(cfg, job_dir)
    out_dir = out["dir"]

    proj = _build_projector(cfg.get("projector", {}))

    inp = cfg.get("input", {})
    if "file" not in inp:
        raise ValueError("input.file is required.")
    in_path = (job_dir / inp["file"]).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    bbox = inp.get("bbox")
    out_path = out_dir / out.get("file", "projected.vtp")
    ext = in_path.suffix.lower()

    if ext in CSV_EXTS:
        result = _project_catalog(proj, in_path, inp.get("columns"), bbox)
    elif ext in MESH_EXTS:
        result = _project_mesh(proj, in_path, inp.get("mesh_units"), bbox)
    else:
        raise ValueError(
            f"Unsupported input extension '{ext}'. "
            f"Use one of: {sorted(CSV_EXTS | MESH_EXTS)}."
        )

    out_path = _fix_extension(out_path, result)
    result.save(str(out_path))
    log.info(f"Saved: {out_path}")


def _build_projector(pcfg: dict) -> Projector:
    if "src_crs" not in pcfg or "dst_crs" not in pcfg:
        raise ValueError("projector.src_crs and projector.dst_crs are required.")
    src_z = pcfg.get("src_z", {})
    dst_z = pcfg.get("dst_z", {})
    anchor = pcfg.get("anchor", {})
    return Projector(
        src_crs=pcfg["src_crs"],
        dst_crs=pcfg["dst_crs"],
        src_xy_units=pcfg.get("src_xy_units", "deg"),
        dst_xy_units=pcfg.get("dst_xy_units", "m"),
        src_z_units=src_z.get("units", "km"),
        dst_z_units=dst_z.get("units", "m"),
        src_z_positive=src_z.get("positive", "down"),
        dst_z_positive=dst_z.get("positive", "up"),
        anchor_geo=tuple(anchor["geo"]) if "geo" in anchor else None,
        anchor_fem=tuple(anchor["fem"]) if "fem" in anchor else None,
        rotation_deg=pcfg.get("rotation_deg"),
    )


def _project_catalog(proj, in_path, columns, bbox) -> pv.PolyData:
    if not columns or len(columns) != 3:
        raise ValueError("input.columns must be a length-3 list for CSV input.")

    cat = load_catalog_csv(in_path, columns=tuple(columns))
    log.info(f"  loaded {len(cat)} points")

    if bbox:
        lon, lat = to_lonlat(cat.x, cat.y, proj.src_crs, proj.src_xy_units)
        depth_km = flip_z(
            cat.z * unit_factor(proj.src_z_units),
            proj.src_z_positive, "down",
        ) / 1000.0
        mask = bbox_mask(
            lon, lat, depth_km,
            lon_range=bbox.get("lon"),
            lat_range=bbox.get("lat"),
            depth_range_km=bbox.get("depth_km"),
        )
        cat = CatalogData(
            x=cat.x[mask], y=cat.y[mask], z=cat.z[mask],
            attrs={k: v[mask] for k, v in cat.attrs.items()},
        )
        log.info(f"  bbox: kept {len(cat)} points")
        if len(cat) == 0:
            raise ValueError("No points left after bbox filtering.")

    cat = proj.transform_catalog(cat)
    poly = pv.PolyData(np.c_[cat.x, cat.y, cat.z])
    for name, arr in cat.attrs.items():
        poly.point_data[name] = arr
    return poly


def _project_mesh(proj, in_path, mesh_units, bbox) -> pv.DataSet:
    mesh = pv.read(str(in_path))
    if mesh.n_points == 0:
        raise ValueError(f"Empty mesh: {in_path}")
    log.info(f"  loaded {mesh.n_points} points, {mesh.n_cells} cells")

    if mesh_units is None:
        mesh_units = proj.src_xy_units

    if bbox:
        pts = np.asarray(mesh.points, dtype=float)
        scale = unit_factor(mesh_units) / unit_factor(proj.src_xy_units)
        lon, lat = to_lonlat(pts[:, 0] * scale, pts[:, 1] * scale,
                             proj.src_crs, proj.src_xy_units)
        depth_km = -pts[:, 2] * unit_factor(mesh_units) / 1000.0
        mask = bbox_mask(
            lon, lat, depth_km,
            lon_range=bbox.get("lon"),
            lat_range=bbox.get("lat"),
            depth_range_km=bbox.get("depth_km"),
        )
        mesh = mesh.extract_points(mask, adjacent_cells=True)
        log.info(f"  bbox: kept {mesh.n_points} points, {mesh.n_cells} cells")
        if mesh.n_points == 0:
            raise ValueError("No mesh left after bbox filtering.")

    return proj.transform_mesh(mesh, mesh_units=mesh_units)


def _fix_extension(out_path: Path, mesh) -> Path:
    ext = out_path.suffix.lower()
    is_poly = isinstance(mesh, pv.PolyData)
    if is_poly and ext == ".vtu":
        new_path = out_path.with_suffix(".vtp")
        log.warning(f"  output is PolyData; writing {new_path.name} instead")
        return new_path
    if (not is_poly) and ext == ".vtp":
        new_path = out_path.with_suffix(".vtu")
        log.warning(f"  output is UnstructuredGrid; writing {new_path.name} instead")
        return new_path
    return out_path