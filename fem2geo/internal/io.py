import csv
import logging
from pathlib import Path

import numpy as np
import pyvista as pv

from fem2geo.data import FractureData, FaultData
from fem2geo.internal.schema import ModelSchema

log = logging.getLogger("fem2geoLogger")


# structural data

_PLANES_COLS = {"strike", "dip"}
_FAULTS_COLS = {"strike", "dip", "rake"}


def load_structural_csv(path) -> FractureData | FaultData:
    """
    Load structural geology measurements from a CSV file.

    Column detection is case-insensitive. Returns :class:`FractureData`
    for strike/dip or :class:`FaultData` for strike/dip/rake.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structural data file not found: {path}")

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        raw_fields = [name.strip().lower() for name in reader.fieldnames]

        cols = set(raw_fields)
        if _FAULTS_COLS <= cols:
            kind = "faults"
        elif _PLANES_COLS <= cols:
            kind = "fractures"
        else:
            raise ValueError(
                f"Unrecognised CSV columns: {reader.fieldnames}. "
                f"Expected: {sorted(_FAULTS_COLS)} or {sorted(_PLANES_COLS)}.")

        col_map = {name.strip().lower(): name.strip()
                   for name in reader.fieldnames}
        rows = list(reader)

    if not rows:
        raise ValueError(f"Structural data file is empty: {path}")

    strikes = np.array([float(r[col_map["strike"]]) for r in rows])
    dips = np.array([float(r[col_map["dip"]]) for r in rows])
    planes = np.column_stack([strikes, dips])

    if kind == "faults":
        rakes = np.array([float(r[col_map["rake"]]) for r in rows])
        data = FaultData(planes=planes, rakes=rakes)
    else:
        data = FractureData(planes=planes)

    log.info(f"Loaded structural data: {path} "
             f"({kind}, {len(rows)} measurements)")
    return data


# grid loading

def load_grid(path, schema: ModelSchema | str = "adeli") -> pv.UnstructuredGrid:
    """
    Load a FEM model and rename arrays to canonical names.

    Scalar/vector fields declared in ``schema.fields`` are renamed
    one-to-one. Tensor arrays declared in ``schema.tensors`` are renamed
    so that :meth:`Model._assemble_tensor` can find them by canonical key.

    Directional fields (``dir_*``) are normalized to unit vectors.
    """
    if isinstance(schema, str):
        schema = ModelSchema.builtin(schema)

    grid = pv.read(path)
    loaded, skipped = [], []

    # scalar/vector fields
    for canonical, entry in schema.fields.items():
        ok = _rename_array(grid, entry.solver_key, canonical)
        if ok:
            if canonical.startswith("dir_"):
                _normalize_dir(grid, canonical)
            loaded.append(canonical)
        else:
            skipped.append(entry.solver_key)

    # tensor arrays
    for name, entry in schema.tensors.items():
        if entry.is_packed:
            ok = _rename_array(grid, entry.voigt6, name)
            (loaded if ok else skipped).append(
                name if ok else entry.voigt6)
        else:
            for comp, solver_key in entry.components.items():
                tag = f"_tensor_{name}_{comp}"
                ok = _rename_array(grid, solver_key, tag)
                (loaded if ok else skipped).append(
                    tag if ok else solver_key)

    log.info(f"Loaded {path}")
    log.info(f"  fields: {', '.join(sorted(loaded))}")
    if skipped:
        log.warning(f"  not found in file: {', '.join(sorted(skipped))}")

    return grid


def _rename_array(grid, solver_key, canonical) -> bool:
    """Rename an array in cell or point data. Returns True if found."""
    if solver_key in grid.cell_data:
        if solver_key != canonical:
            grid.cell_data[canonical] = grid.cell_data[solver_key]
            del grid.cell_data[solver_key]
        return True
    if solver_key in grid.point_data:
        if solver_key != canonical:
            grid.point_data[canonical] = grid.point_data[solver_key]
            del grid.point_data[solver_key]
        return True
    return False


def _normalize_dir(grid, canonical):
    """Normalize a directional array to unit vectors in place."""
    for store in (grid.cell_data, grid.point_data):
        if canonical in store:
            arr = np.asarray(store[canonical], dtype=float)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            store[canonical] = arr / np.where(norms < 1e-12, 1.0, norms)
            return