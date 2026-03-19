import logging
from functools import cached_property

import numpy as np
import pyvista as pv

from fem2geo.internal.io import load_grid
from fem2geo.internal.schema import ModelSchema

log = logging.getLogger("fem2geoLogger")

_STRESS_IDX = [
    ("s_xx", "s_xy", "s_zx"),
    ("s_xy", "s_yy", "s_yz"),
    ("s_zx", "s_yz", "s_zz"),
]


class Model:
    """
    A FEM model with canonical field names and ENU-normalized directions.

    Parameters
    ----------
    grid : pv.UnstructuredGrid
        Grid with canonical field names as produced by :func:`load_grid`.
    schema : ModelSchema
        Schema used to load the grid. Retained for unit conversion.

    Notes
    -----
    Construct via :meth:`from_file`, not directly.
    Field properties are assembled from the grid on first access and cached.
    ``volumes`` and ``cell_centers`` are also lazy as they require a PyVista
    computation pass.
    Stress-derived quantities (``dir_s2``, ``val_s1``, ``val_s2``, ``val_s3``)
    are computed from the stress tensor when not present in the file.
    """

    def __init__(self, grid: pv.UnstructuredGrid, schema: ModelSchema):
        self._grid = grid
        self.schema = schema

    @classmethod
    def from_file(cls, path, schema: ModelSchema | str = "adeli") -> "Model":
        """
        Load a FEM model file.

        Parameters
        ----------
        path : str or Path
        schema : ModelSchema or str
            A ``ModelSchema`` instance or the name of a built-in schema.

        Returns
        -------
        Model
        """
        if isinstance(schema, str):
            schema = ModelSchema.builtin(schema)
        grid = load_grid(path, schema)
        return cls(grid, schema)

    # Geometry
    @property
    def points(self) -> np.ndarray:
        return self._grid.points

    @property
    def n_cells(self) -> int:
        return self._grid.number_of_cells

    @cached_property
    def volumes(self) -> np.ndarray:
        return self._grid.compute_cell_sizes().cell_data["Volume"]

    @cached_property
    def cell_centers(self) -> np.ndarray:
        return self._grid.cell_centers().points

    # Stress tensor
    @cached_property
    def stress(self) -> np.ndarray:
        """
        Stress tensor for all cells.

        Returns
        -------
        numpy.ndarray, shape (N, 3, 3)
        """
        return self._assemble_stress()

    # Eigenvectors
    @cached_property
    def dir_s1(self) -> np.ndarray:
        return self._field("dir_s1")

    @cached_property
    def dir_s3(self) -> np.ndarray:
        return self._field("dir_s3")

    @cached_property
    def dir_s2(self) -> np.ndarray:
        """
        Intermediate principal stress direction, derived as dir_s1 × dir_s3.

        Returns
        -------
        numpy.ndarray, shape (N, 3)
        """
        if "dir_s2" in self._grid.array_names:
            return self._field("dir_s2")
        d = np.cross(self.dir_s1, self.dir_s3)
        norms = np.linalg.norm(d, axis=1, keepdims=True)
        return d / np.where(norms < 1e-12, 1.0, norms)

    def eigenvectors(self) -> np.ndarray:
        """
        Principal stress directions for all cells.

        Returns
        -------
        numpy.ndarray, shape (N, 3, 3)
            eigenvectors[:, :, i] is the unit vector for the i-th principal direction,
            ordered to match :meth:`eigenvalues`.
        """
        _, vecs = np.linalg.eigh(self.stress)
        return vecs

    # Principal values
    @cached_property
    def val_s1(self) -> np.ndarray:
        if "val_s1" in self._grid.array_names:
            return self._field("val_s1")
        return self._principal_values()[:, 0]

    @cached_property
    def val_s2(self) -> np.ndarray:
        """Derived from deviatoric trace identity: val_s1 + val_s2 + val_s3 = 0."""
        return -(self.val_s1 + self.val_s3)

    @cached_property
    def val_s3(self) -> np.ndarray:
        if "val_s3" in self._grid.array_names:
            return self._field("val_s3")
        return self._principal_values()[:, 2]

    def eigenvalues(self) -> np.ndarray:
        """
        Principal stress values for all cells, sorted ascending.

        Returns
        -------
        numpy.ndarray, shape (N, 3)
            Each row is [val_s1, val_s2, val_s3] (most compressive first).
        """
        return np.linalg.eigvalsh(self.stress)

    # Kinematics

    @cached_property
    def u(self) -> np.ndarray:
        return self._field("u")

    @cached_property
    def v(self) -> np.ndarray:
        return self._field("v")

    @cached_property
    def t(self) -> np.ndarray:
        return self._field("t")

    # Tensor scalar representations
    @cached_property
    def i1_strain(self) -> np.ndarray:
        return self._field("i1_strain")

    @cached_property
    def j2_strain(self) -> np.ndarray:
        return self._field("j2_strain")

    @cached_property
    def j2_stress(self) -> np.ndarray:
        return self._field("j2_stress")

    @cached_property
    def plastic_eff(self) -> np.ndarray:
        return self._field("plastic_eff")

    @cached_property
    def plastic_vol(self) -> np.ndarray:
        return self._field("plastic_vol")

    @cached_property
    def plastic_yield(self) -> np.ndarray:
        return self._field("plastic_yield")

    # Extraction
    def extract_sphere(self, center, radius) -> "Model":
        """
        Extract cells touched by a sphere.

        Parameters
        ----------
        center : array-like, shape (3,)
        radius : float

        Returns
        -------
        Model
        """
        center = np.asarray(center)
        mask = np.linalg.norm(self._grid.points - center, axis=1) < radius
        return Model(self._extract(mask), self.schema)

    def extract_box(self, center, dim) -> "Model":
        """
        Extract cells touched by an axis-aligned bounding box.

        Parameters
        ----------
        center : array-like, shape (3,)
        dim : array-like, shape (3,)
            Full extents in each direction.

        Returns
        -------
        Model
        """
        center, dim = np.asarray(center), np.asarray(dim)
        ll, ur = center - dim / 2.0, center + dim / 2.0
        mask = np.all((self._grid.points >= ll) & (self._grid.points <= ur), axis=1)
        return Model(self._extract(mask), self.schema)

    # Stress averages
    def avg_principal(self) -> tuple:
        """
        Volume-weighted average stress principal values and directions.

        Returns
        -------
        val : numpy.ndarray, shape (3,)
            Eigenvalues sorted ascending (most compressive first).
        vec : numpy.ndarray, shape (3, 3)
            Eigenvectors as columns in ENU coordinates.
        """
        avg = self.avg_dev_stress()
        val, vec = np.linalg.eigh(avg)
        order = np.argsort(val)
        return val[order], vec[:, order]

    def avg_dev_stress(self) -> np.ndarray:
        """
        Volume-weighted average of the deviatoric stress tensor.

        Returns
        -------
        numpy.ndarray, shape (3, 3)
        """
        return np.einsum("ijk,i->jk", self.stress, self.volumes) / self.volumes.sum()

    def avg_total_stress(self, rho: float = 2800.0, g: float = 9.81) -> np.ndarray:
        """
        Volume-weighted average of the total stress tensor.

        Adds lithostatic pressure (``rho * g * depth``) to the diagonal of
        each cell's deviatoric stress tensor before averaging. Use only when
        the model stores deviatoric stress — if lithostatic is already
        included, use :meth:`avg_dev_stress` instead.

        Parameters
        ----------
        rho : float
            Density in kg/m³.
        g : float
            Gravitational acceleration in m/s².

        Returns
        -------
        numpy.ndarray, shape (3, 3)
        """
        tensors = self.stress.copy()
        p = rho * g * self.cell_centers[:, 2] * -1000.0 / 1e6
        tensors[:, 0, 0] += p
        tensors[:, 1, 1] += p
        tensors[:, 2, 2] += p
        return np.einsum("ijk,i->jk", tensors, self.volumes) / self.volumes.sum()

    # Persistence
    def save(self, path) -> None:
        """
        Save the model to a VTK/VTU file.

        Parameters
        ----------
        path : str or Path
        """
        self._grid.save(str(path))
        log.info(f"Saved model to {path}")

    # Private
    def _principal_values(self) -> np.ndarray:
        return self.eigenvalues()

    def _field(self, canonical: str) -> np.ndarray:
        if canonical in self._grid.cell_data:
            return np.asarray(self._grid.cell_data[canonical])
        if canonical in self._grid.point_data:
            return np.asarray(self._grid.point_data[canonical])
        raise KeyError(
            f"Field '{canonical}' not found. "
            f"Check that the schema maps this field and it is present in the file."
        )

    def _assemble_stress(self) -> np.ndarray:
        n = self._grid.number_of_cells
        t = np.zeros((n, 3, 3))
        has_components = all(
            key in self._grid.cell_data or key in self._grid.point_data
            for row in _STRESS_IDX for key in row
        )
        if has_components:
            for i, row in enumerate(_STRESS_IDX):
                for j, key in enumerate(row):
                    t[:, i, j] = self._field(key)
            return t
        # Fallback: reconstruct from principal values and directions
        required = ("val_s1", "val_s3", "dir_s1", "dir_s3")
        missing = [k for k in required
                   if k not in self._grid.cell_data and k not in self._grid.point_data]
        if missing:
            raise KeyError(
                f"Stress components missing and reconstruction failed. "
                f"Need val_s1, val_s3, dir_s1, dir_s3 but missing: {missing}"
            )
        log.warning(
            "Stress components not found — reconstructing from principal values and directions.")
        v1 = self.val_s1
        v3 = self.val_s3
        v2 = -(v1 + v3)
        d1 = self.dir_s1
        d2 = self.dir_s2
        d3 = self.dir_s3

        t = (v1[:, None, None] * (d1[:, :, None] * d1[:, None, :])
             + v2[:, None, None] * (d2[:, :, None] * d2[:, None, :])
             + v3[:, None, None] * (d3[:, :, None] * d3[:, None, :]))
        return t

    def _extract(self, point_mask: np.ndarray) -> pv.UnstructuredGrid:
        """
        Extract cells that reference at least one flagged point.

        Parameters
        ----------
        point_mask : numpy.ndarray, shape (n_points,), dtype bool
            True for points inside the selection region.

        Returns
        -------
        pv.UnstructuredGrid
        """
        # VTK connectivity: flat array of point ids per cell
        conn = self._grid.cell_connectivity
        offsets = self._grid.offset

        # build per-cell flag: does any point in the cell satisfy the mask?
        # expand offsets to start/end pairs
        starts = offsets[:-1]
        ends = offsets[1:]

        # for each cell, check if any referenced point is flagged
        # use np.add.reduceat on the mask values looked up through connectivity
        flagged = point_mask[conn].astype(np.intp)
        counts = np.add.reduceat(flagged, starts)
        cell_ids = np.where(counts > 0)[0]

        log.info(f"Extracted {len(cell_ids)} cells")

        if len(cell_ids) == 0:
            return pv.UnstructuredGrid()

        return self._grid.extract_cells(cell_ids)