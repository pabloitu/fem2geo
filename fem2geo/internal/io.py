import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv

from fem2geo.data import FractureData, FaultData, CatalogData
from fem2geo.internal.schema import ModelSchema

log = logging.getLogger("fem2geoLogger")


_PLANES_COLS = {"strike", "dip"}
_FAULTS_COLS = {"strike", "dip", "rake"}


def load_structural_csv(path) -> FractureData | FaultData:
    """Read structural measurements from a CSV file.

    Files with strike/dip/rake columns produce FaultData; strike/dip only gives
    FractureData.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structural data file not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    cols = set(df.columns)

    if _FAULTS_COLS <= cols:
        kind = "faults"
    elif _PLANES_COLS <= cols:
        kind = "fractures"
    else:
        raise ValueError(
            f"Unrecognised CSV columns: {list(df.columns)}. "
            f"Expected: {sorted(_FAULTS_COLS)} or {sorted(_PLANES_COLS)}."
        )

    if df.empty:
        raise ValueError(f"Structural data file is empty: {path}")

    planes = df[["strike", "dip"]].to_numpy(dtype=float)

    if kind == "faults":
        rakes = df["rake"].to_numpy(dtype=float)
        data = FaultData(planes=planes, rakes=rakes)
    else:
        data = FractureData(planes=planes)

    log.info(f"Loaded structural data: {path} ({kind}, {len(df)} measurements)")
    return data


def load_catalog_csv(path, columns) -> CatalogData:
    """
    Read a point catalog from a CSV file.

    The three coordinate columns become x, y, z; remaining numeric columns become
    attrs. Non-numeric columns are dropped with a warning. Empty cells in numeric
    columns are read as NaN.

    Parameters
    ----------
    path : str or Path
        CSV file with a header row.
    columns : tuple of str
        ``(x_col, y_col, z_col)`` column names.

    Returns
    -------
    CatalogData
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Catalog file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Catalog file is empty: {path}")

    x_col, y_col, z_col = columns
    for c in (x_col, y_col, z_col):
        if c not in df.columns:
            raise ValueError(
                f"Column '{c}' not found in {path}. " f"Available: {list(df.columns)}"
            )

    numeric = df.apply(pd.to_numeric, errors="coerce")
    dropped = [
        c for c in df.columns if numeric[c].isna().all() and not df[c].isna().all()
    ]
    if dropped:
        log.warning(f"  dropped non-numeric columns: {dropped}")

    coord_cols = (x_col, y_col, z_col)
    attrs = {
        c: numeric[c].to_numpy(dtype=float)
        for c in df.columns
        if c not in coord_cols and c not in dropped
    }

    log.info(
        f"Loaded catalog: {path} ({len(df)} points, " f"{len(attrs)} numeric attrs)"
    )
    return CatalogData(
        x=numeric[x_col].to_numpy(dtype=float),
        y=numeric[y_col].to_numpy(dtype=float),
        z=numeric[z_col].to_numpy(dtype=float),
        attrs=attrs,
    )


def load_mesh(path) -> pv.DataSet:
    """
    Read a PyVista mesh from a VTK file (.vtp, .vtu, .vtk).

    Raises if the file is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")
    mesh = pv.read(str(path))
    if mesh.n_points == 0:
        raise ValueError(f"Empty mesh: {path}")
    log.info(f"Loaded mesh: {path} ({mesh.n_points} points, {mesh.n_cells} cells)")
    return mesh


def load_raster(path, z_band=None, window=None) -> pv.PolyData:
    """
    Read a GeoTIFF as a triangulated PolyData surface.

    Each pixel center becomes a point in the raster's own CRS. Bands
    are attached as point_data arrays. If ``z_band`` is given, that
    band drives the Z coordinate; otherwise Z is zero everywhere. An
    optional rasterio ``window`` restricts the read.

    Parameters
    ----------
    path : str or Path
        GeoTIFF file.
    z_band : int, optional
        1-indexed band to use as elevation.
    window : rasterio.windows.Window, optional
        Sub-window to read. Defaults to the full raster.

    Returns
    -------
    pyvista.PolyData
        Triangulated surface with one point per pixel and point_data
        arrays for every band plus a ``valid`` mask.
    """
    import rasterio
    from rasterio.windows import Window

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raster file not found: {path}")

    with rasterio.open(str(path)) as ds:
        if window is None:
            window = Window(0, 0, ds.width, ds.height)

        w, h = int(window.width), int(window.height)
        if w < 2 or h < 2:
            raise ValueError(
                f"Raster window too small to triangulate: {w}x{h} (need >= 2x2)."
            )
        if z_band is not None and z_band > ds.count:
            raise ValueError(f"z_band={z_band} but raster has {ds.count} band(s).")

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
            desc = ds.descriptions[i - 1] if ds.descriptions else None
            name = str(desc) if desc else f"band_{i}"
            bands[name] = arr

        if z_band is not None:
            z_name = list(bands.keys())[z_band - 1]
            zs = bands[z_name]
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

    faces = quad_faces(valid, w)
    if faces.size == 0:
        raise ValueError("No valid faces built from raster (all pixels masked?).")

    poly = pv.PolyData(points, faces)
    for name, arr in bands.items():
        poly.point_data[name] = arr.ravel()
    poly.point_data["valid"] = valid.astype(np.uint8).ravel()

    log.info(
        f"Loaded raster: {path} ({poly.n_points} points, "
        f"{poly.n_cells} triangles, {len(bands)} bands)"
    )
    return poly


def quad_faces(valid, w):
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


def load_solver_output(
    path, schema: ModelSchema | str = "adeli"
) -> pv.UnstructuredGrid:
    """
    Load a FEM result file and rename arrays to canonical names.

    Scalar and vector fields are renamed one-to-one from the schema. Tensor arrays
    are renamed so they can be reassembled later by canonical key. Directional fields
    (dir_*) are normalized to unit vectors.
    """
    if isinstance(schema, str):
        schema = ModelSchema.builtin(schema)

    mesh = pv.read(path)

    if mesh.n_points == 0:
        raise ValueError(f"Empty mesh: {path}")

    loaded, skipped = [], []

    for canonical, entry in schema.fields.items():
        if rename_array(mesh, entry.solver_key, canonical):
            if canonical.startswith("dir_"):
                normalize_vector(mesh, canonical)
            loaded.append(canonical)
        else:
            skipped.append(entry.solver_key)

    for name, entry in schema.tensors.items():
        if entry.is_packed:
            array_ok = rename_array(mesh, entry.voigt6, name)
            (loaded if array_ok else skipped).append(name if array_ok else entry.voigt6)
        else:
            for comp, solver_key in entry.components.items():
                tag = f"_tensor_{name}_{comp}"
                array_ok = rename_array(mesh, solver_key, tag)
                (loaded if array_ok else skipped).append(
                    tag if array_ok else solver_key
                )

    log.info(f"  Fields: {', '.join(sorted(loaded))}")
    if skipped:
        log.warning(f"  Not found: {', '.join(sorted(skipped))}")

    return mesh


def rename_array(grid, solver_key, canonical) -> bool:
    """Rename an array in cell or point data. Returns True if found."""
    for store in (grid.cell_data, grid.point_data):
        if solver_key in store:
            if solver_key != canonical:
                store[canonical] = store[solver_key]
                del store[solver_key]
            return True
    return False


def normalize_vector(grid, canonical):
    """Normalize a directional array to unit vectors in place."""
    for store in (grid.cell_data, grid.point_data):
        if canonical in store:
            arr = np.asarray(store[canonical], dtype=float)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            store[canonical] = arr / np.where(norms < 1e-12, 1.0, norms)
            return
