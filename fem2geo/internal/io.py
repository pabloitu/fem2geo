import csv
import logging
from pathlib import Path

import numpy as np
import pyvista as pv

from fem2geo.data import FractureData, FaultData
from fem2geo.internal.schema import ModelSchema

log = logging.getLogger("fem2geoLogger")


# structural data column sets (lowercase, stripped)
_PLANES_COLS = {"strike", "dip"}
_FAULTS_COLS = {"strike", "dip", "rake"}


def load_structural_csv(path) -> FractureData | FaultData:
    """
    Load structural geology measurements from a CSV file.

    The file must have a header row. Column detection is based on the header
    names (case-insensitive, stripped). Two formats are supported:

    - **fractures**: columns ``strike, dip`` — plane orientations
      (fractures, joints, veins, dykes). Returns :class:`FractureData`.
    - **faults**: columns ``strike, dip, rake`` — fault planes with slip
      direction. Returns :class:`FaultData`.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file.

    Returns
    -------
    FractureData or FaultData

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the header does not match any supported column set.
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
                f"Expected one of: {sorted(_FAULTS_COLS)} or {sorted(_PLANES_COLS)}."
            )

        # build column name mapping (original header name → lowercase key)
        col_map = {name.strip().lower(): name.strip() for name in reader.fieldnames}
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

    log.info(f"Loaded structural data: {path} ({kind}, {len(rows)} measurements)")
    return data


def load_grid(path, schema: ModelSchema | str = "adeli") -> pv.UnstructuredGrid:
    """
    Load a FEM model and rename fields to canonical names.

    Directional fields (``dir_*``) are normalized to unit vectors.
    Original solver field names are removed; only canonical names are kept.
    Fields present in the schema but absent from cell or point data are
    skipped and logged at WARNING level.

    Parameters
    ----------
    path : str or Path
        Path to a VTK or VTU file.
    schema : ModelSchema or str
        A ``ModelSchema`` instance or the name of a built-in schema.

    Returns
    -------
    pv.UnstructuredGrid
        Grid with canonical field names. All directional fields are ENU
        unit vectors. All other fields retain their original solver units
        unless converted externally via ``schema.si_factor``.
    """
    if isinstance(schema, str):
        schema = ModelSchema.builtin(schema)

    model = pv.read(path)
    loaded, skipped = [], []

    for canonical, entry in schema.fields.items():
        in_cells  = entry.solver_key in model.cell_data
        in_points = entry.solver_key in model.point_data

        if not in_cells and not in_points:
            skipped.append(entry.solver_key)
            continue

        data = model.cell_data[entry.solver_key] if in_cells \
            else model.point_data[entry.solver_key]

        if canonical.startswith("dir_"):
            data = _unit_vectors(data)

        if in_cells:
            model.cell_data[canonical] = data
            del model.cell_data[entry.solver_key]
        else:
            model.point_data[canonical] = data
            del model.point_data[entry.solver_key]

        loaded.append(canonical)

    log.info(f"Loaded {path}")
    log.info(f"  fields: {', '.join(sorted(loaded))}")
    if skipped:
        log.warning(f"  not found in file: {', '.join(sorted(skipped))}")

    return model


def _unit_vectors(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return arr / norms