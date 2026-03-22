import csv
import logging
from pathlib import Path

import numpy as np
import pyvista as pv

from fem2geo.data import FractureData, FaultData
from fem2geo.internal.schema import ModelSchema

log = logging.getLogger("fem2geoLogger")


_PLANES_COLS = {"strike", "dip"}
_FAULTS_COLS = {"strike", "dip", "rake"}


def load_structural_csv(path) -> FractureData | FaultData:
    """
    Read structural measurements from a CSV file.

    Column matching is case-insensitive. Files with strike/dip/rake
    columns produce FaultData; strike/dip only gives FractureData.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structural data file not found: {path}")

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = {name.strip().lower() for name in reader.fieldnames}

        if _FAULTS_COLS <= cols:
            kind = "faults"
        elif _PLANES_COLS <= cols:
            kind = "fractures"
        else:
            raise ValueError(
                f"Unrecognised CSV columns: {reader.fieldnames}. "
                f"Expected: {sorted(_FAULTS_COLS)} or {sorted(_PLANES_COLS)}."
            )

        col_map = {name.strip().lower(): name.strip() for name in reader.fieldnames}
        rows = list(reader)

    if not rows:
        raise ValueError(f"Structural data file is empty: {path}")

    strikes = np.array([float(r[col_map["strike"]]) for r in rows])
    dips    = np.array([float(r[col_map["dip"]])    for r in rows])
    planes  = np.column_stack([strikes, dips])

    if kind == "faults":
        rakes = np.array([float(r[col_map["rake"]]) for r in rows])
        data = FaultData(planes=planes, rakes=rakes)
    else:
        data = FractureData(planes=planes)

    log.info(f"Loaded structural data: {path} ({kind}, {len(rows)} measurements)")
    return data


def load_grid(path, schema: ModelSchema | str = "adeli") -> pv.UnstructuredGrid:
    """
    Load a FEM result file and rename arrays to canonical names.

    Scalar and vector fields are renamed one-to-one from the schema. Tensor arrays
    are renamed so they can be reassembled later by canonical key. Directional fields
    (dir_*) are normalized to unit vectors.
    """
    if isinstance(schema, str):
        schema = ModelSchema.builtin(schema)

    grid = pv.read(path)
    loaded, skipped = [], []

    for canonical, entry in schema.fields.items():
        if _rename(grid, entry.solver_key, canonical):
            if canonical.startswith("dir_"):
                _normalize(grid, canonical)
            loaded.append(canonical)
        else:
            skipped.append(entry.solver_key)

    for name, entry in schema.tensors.items():
        if entry.is_packed:
            ok = _rename(grid, entry.voigt6, name)
            (loaded if ok else skipped).append(name if ok else entry.voigt6)
        else:
            for comp, solver_key in entry.components.items():
                tag = f"_tensor_{name}_{comp}"
                ok = _rename(grid, solver_key, tag)
                (loaded if ok else skipped).append(tag if ok else solver_key)

    log.info(f"  Fields: {', '.join(sorted(loaded))}")
    if skipped:
        log.warning(f"  Not found: {', '.join(sorted(skipped))}")

    return grid


def _rename(grid, solver_key, canonical) -> bool:
    """Rename an array in cell or point data. Returns True if found."""
    for store in (grid.cell_data, grid.point_data):
        if solver_key in store:
            if solver_key != canonical:
                store[canonical] = store[solver_key]
                del store[solver_key]
            return True
    return False


def _normalize(grid, canonical):
    """Normalize a directional array to unit vectors in place."""
    for store in (grid.cell_data, grid.point_data):
        if canonical in store:
            arr = np.asarray(store[canonical], dtype=float)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            store[canonical] = arr / np.where(norms < 1e-12, 1.0, norms)
            return