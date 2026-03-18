import logging

import numpy as np
import pyvista as pv

from fem2geo.internal.schema import ModelSchema

log = logging.getLogger("fem2geoLogger")


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