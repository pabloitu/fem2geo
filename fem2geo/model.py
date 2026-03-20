import logging
from functools import cached_property

import numpy as np
import pyvista as pv

from fem2geo.internal.io import load_grid
from fem2geo.internal.schema import ModelSchema
from fem2geo.utils.tensor import unpack_voigt6, unpack_components

log = logging.getLogger("fem2geoLogger")


# field descriptors

class ScalarField:
    """Descriptor for a scalar field looked up by canonical name."""

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        val = obj._field(self.name)
        setattr(obj, self.name, val)
        return val


class VectorField:
    """Descriptor for a vector field looked up by canonical name."""

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        val = obj._field(self.name)
        setattr(obj, self.name, val)
        return val


class TensorField:
    """Descriptor for a tensor assembled via the schema."""

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        val = obj._assemble_tensor(self.name)
        setattr(obj, self.name, val)
        return val


class Model:
    """
    A FEM model with canonical field names and ENU directions.

    Construct via :meth:`from_file`. Properties are lazy and cached.
    Tensor assembly is schema-driven: the schema declares whether a
    tensor is stored as a packed Voigt array or as individual
    components, and :meth:`_assemble_tensor` dispatches accordingly.

    Simple fields (scalars, vectors, tensors that need no special
    logic) are declared as class-level descriptors. Computed fields
    with fallback logic remain as ``cached_property``.
    """

    # tensors
    strain = TensorField("strain")
    strain_rate = TensorField("strain_rate")

    # kinematics
    u = VectorField("u")
    v = VectorField("v")
    t = ScalarField("t")

    # principal directions
    dir_s1 = VectorField("dir_s1")
    dir_s3 = VectorField("dir_s3")

    # scalar fields
    i1_strain = ScalarField("i1_strain")
    j2_strain = ScalarField("j2_strain")
    j2_stress = ScalarField("j2_stress")
    plastic_eff = ScalarField("plastic_eff")
    plastic_vol = ScalarField("plastic_vol")
    plastic_yield = ScalarField("plastic_yield")
    plastic_mode = ScalarField("plastic_mode")
    mean_stress = ScalarField("mean_stress")
    viscosity = ScalarField("viscosity")
    threshold_ratio = ScalarField("threshold_ratio")
    temperature = ScalarField("temperature")
    fluid_pressure = ScalarField("fluid_pressure")
    i1_strain_rate = ScalarField("i1_strain_rate")
    j2_strain_rate = ScalarField("j2_strain_rate")
    darcy_vel = VectorField("darcy_vel")
    heat_flux = VectorField("heat_flux")

    def __init__(self, grid: pv.UnstructuredGrid, schema: ModelSchema):
        self._grid = grid
        self.schema = schema

    @classmethod
    def from_file(
        cls, path, schema: ModelSchema | str = "adeli"
    ) -> "Model":
        if isinstance(schema, str):
            schema = ModelSchema.builtin(schema)
        return cls(load_grid(path, schema), schema)

    # geometry

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

    # tensor assembly

    def _assemble_tensor(self, name: str) -> np.ndarray:
        """
        Assemble (N, 3, 3) symmetric tensor from grid data.

        Reads the schema to determine format (voigt6 or components),
        then delegates to the appropriate unpacker.
        """
        if not self.schema.has_tensor(name):
            raise KeyError(
                f"No tensor '{name}' in schema "
                f"'{self.schema.name}'."
            )

        entry = self.schema.tensors[name]

        if entry.is_packed:
            return unpack_voigt6(self._array(name))

        arrays = {
            comp: self._array(f"_tensor_{name}_{comp}")
            for comp in entry.components
        }
        return unpack_components(arrays)

    def _array(self, key: str) -> np.ndarray:
        """Look up a renamed array by canonical key."""
        if key in self._grid.cell_data:
            return np.asarray(self._grid.cell_data[key])
        if key in self._grid.point_data:
            return np.asarray(self._grid.point_data[key])
        raise KeyError(f"Array '{key}' not found in grid.")

    # stress (computed, with fallback)

    @cached_property
    def stress(self) -> np.ndarray:
        """
        Stress tensor, shape (N, 3, 3).

        Tries schema tensor first, then principal reconstruction.
        """
        if self.schema.has_tensor("stress"):
            try:
                return self._assemble_tensor("stress")
            except KeyError:
                log.warning(
                    "Stress tensor declared but data missing "
                    "— falling back to reconstruction."
                )
        return self._reconstruct_stress()

    def _reconstruct_stress(self) -> np.ndarray:
        required = ("val_s1", "val_s3", "dir_s1", "dir_s3")
        missing = [
            k for k in required
            if k not in self._grid.cell_data
            and k not in self._grid.point_data
        ]
        if missing:
            raise KeyError(
                f"Cannot assemble stress — missing: {missing}"
            )
        log.warning("Reconstructing stress from principals.")
        v1, v3 = self.val_s1, self.val_s3
        v2 = -(v1 + v3)
        d1, d2, d3 = self.dir_s1, self.dir_s2, self.dir_s3
        return (
            v1[:, None, None] * (d1[:, :, None] * d1[:, None, :])
            + v2[:, None, None] * (d2[:, :, None] * d2[:, None, :])
            + v3[:, None, None] * (d3[:, :, None] * d3[:, None, :])
        )

    @cached_property
    def stress_dev(self) -> np.ndarray:
        """Deviatoric stress, shape (N, 3, 3)."""
        s = self.stress
        tr = np.trace(s, axis1=1, axis2=2)
        return s - (tr / 3.0)[:, None, None] * np.eye(3)

    # strain (computed, with fallback)

    @cached_property
    def strain_plastic(self) -> np.ndarray:
        """
        Plastic strain tensor, shape (N, 3, 3).

        Loaded from schema if available, otherwise reconstructed
        from scalar invariants and stress eigenvectors assuming
        coaxiality (isotropic flow rule).
        """
        if self.schema.has_tensor("strain_plastic"):
            try:
                return self._assemble_tensor("strain_plastic")
            except KeyError:
                log.warning(
                    "strain_plastic declared but data missing "
                    "— falling back to coaxiality."
                )
        return self._reconstruct_plastic_strain()

    @cached_property
    def strain_elastic(self) -> np.ndarray:
        """
        Elastic strain tensor, shape (N, 3, 3).

        Loaded from schema if available, otherwise computed as
        total strain minus plastic strain.
        """
        if self.schema.has_tensor("strain_elastic"):
            try:
                return self._assemble_tensor("strain_elastic")
            except KeyError:
                log.warning(
                    "strain_elastic declared but data missing "
                    "— computing as total - plastic."
                )
        return self.strain - self.strain_plastic

    def _reconstruct_plastic_strain(self) -> np.ndarray:
        """
        Reconstruct plastic strain from scalar invariants and
        stress eigenvectors (coaxiality assumption).
        """
        try:
            eff = self.plastic_eff
            vol = self.plastic_vol
        except KeyError:
            log.info("No plastic strain data — assuming elastic.")
            return np.zeros((self.n_cells, 3, 3))

        if np.all(eff == 0) and np.all(vol == 0):
            return np.zeros((self.n_cells, 3, 3))

        d1, d2, d3 = self.dir_s1, self.dir_s2, self.dir_s3

        # deviatoric stress eigenvalues for shape
        s1, s2, s3 = self.val_s1, self.val_s2, self.val_s3
        dev = np.column_stack([s1, s2, s3])

        # normalize deviatoric shape to unit J2
        j2 = np.sqrt(0.5 * np.sum(dev**2, axis=1, keepdims=True))
        j2 = np.where(j2 < 1e-30, 1.0, j2)
        shape = dev / j2

        # scale to plastic_eff and add volumetric part
        ep_dev = eff[:, None] * shape
        ep = ep_dev + (vol / 3.0)[:, None]

        return (
            ep[:, 0, None, None] * (d1[:, :, None] * d1[:, None, :])
            + ep[:, 1, None, None] * (d2[:, :, None] * d2[:, None, :])
            + ep[:, 2, None, None] * (d3[:, :, None] * d3[:, None, :])
        )

    # principal directions (computed)

    @cached_property
    def dir_s2(self) -> np.ndarray:
        if "dir_s2" in self._grid.array_names:
            return self._field("dir_s2")
        d = np.cross(self.dir_s1, self.dir_s3)
        norms = np.linalg.norm(d, axis=1, keepdims=True)
        return d / np.where(norms < 1e-12, 1.0, norms)

    def eigenvectors(self) -> np.ndarray:
        """Principal stress directions, shape (N, 3, 3)."""
        _, vecs = np.linalg.eigh(self.stress)
        return vecs

    # principal values (computed, with fallback)

    @cached_property
    def val_s1(self) -> np.ndarray:
        if "val_s1" in self._grid.array_names:
            return self._field("val_s1")
        return self.eigenvalues()[:, 0]

    @cached_property
    def val_s2(self) -> np.ndarray:
        if "val_s2" in self._grid.array_names:
            return self._field("val_s2")
        return -(self.val_s1 + self.val_s3)

    @cached_property
    def val_s3(self) -> np.ndarray:
        if "val_s3" in self._grid.array_names:
            return self._field("val_s3")
        return self.eigenvalues()[:, 2]

    def eigenvalues(self) -> np.ndarray:
        """Principal stress values, sorted ascending, shape (N, 3)."""
        return np.linalg.eigvalsh(self.stress)

    # extraction

    def extract(self, zone: dict) -> "Model":
        """
        Extract a sub-model from a zone config dict.

        Parameters
        ----------
        zone : dict
            Must contain ``type`` (``sphere`` or ``box``),
            ``center``, and ``radius`` or ``dim``.
        """
        kind = zone["type"]
        if kind == "sphere":
            return self.extract_sphere(zone["center"], zone["radius"])
        if kind == "box":
            return self.extract_box(zone["center"], zone["dim"])
        raise ValueError(f"Unknown zone type '{kind}'.")

    def extract_sphere(self, center, radius) -> "Model":
        """Extract cells touched by a sphere."""
        center = np.asarray(center)
        dist = np.linalg.norm(self._grid.points - center, axis=1)
        return Model(self._extract(dist < radius), self.schema)

    def extract_box(self, center, dim) -> "Model":
        """Extract cells touched by an axis-aligned box."""
        center, dim = np.asarray(center), np.asarray(dim)
        lo, hi = center - dim / 2.0, center + dim / 2.0
        pts = self._grid.points
        mask = np.all((pts >= lo) & (pts <= hi), axis=1)
        return Model(self._extract(mask), self.schema)

    # averages

    def avg_tensor(self, name: str) -> np.ndarray:
        """
        Volume-weighted average of a tensor field, shape (3, 3).

        Accepts schema tensor names and computed properties:
        'stress', 'stress_dev', 'strain', 'strain_rate',
        'strain_plastic', 'strain_elastic'.
        """
        _PROPS = {
            "stress": lambda: self.stress,
            "stress_dev": lambda: self.stress_dev,
            "strain": lambda: self.strain,
            "strain_rate": lambda: self.strain_rate,
            "strain_plastic": lambda: self.strain_plastic,
            "strain_elastic": lambda: self.strain_elastic,
        }
        if name in _PROPS:
            tensors = _PROPS[name]()
        else:
            tensors = self._assemble_tensor(name)
        w = self.volumes
        return np.einsum("ijk,i->jk", tensors, w) / w.sum()

    def avg_principal(self, name: str = "stress") -> tuple:
        """
        Volume-weighted average principal values and directions.

        Returns
        -------
        val : numpy.ndarray, shape (3,)
            Eigenvalues sorted ascending.
        vec : numpy.ndarray, shape (3, 3)
            Eigenvectors as columns in ENU.
        """
        avg = self.avg_tensor(name)
        val, vec = np.linalg.eigh(avg)
        order = np.argsort(val)
        return val[order], vec[:, order]

    # persistence

    def save(self, path) -> None:
        self._grid.save(str(path))
        log.info(f"Saved model to {path}")

    # private helpers

    def _field(self, canonical: str) -> np.ndarray:
        """Look up a scalar/vector field by canonical name."""
        if canonical in self._grid.cell_data:
            return np.asarray(self._grid.cell_data[canonical])
        if canonical in self._grid.point_data:
            return np.asarray(self._grid.point_data[canonical])
        raise KeyError(
            f"Field '{canonical}' not found in grid."
        )

    def _extract(self, point_mask: np.ndarray) -> pv.UnstructuredGrid:
        """Extract cells referencing at least one flagged point."""
        conn = self._grid.cell_connectivity
        starts = self._grid.offset[:-1]
        flagged = point_mask[conn].astype(np.intp)
        counts = np.add.reduceat(flagged, starts)
        cell_ids = np.where(counts > 0)[0]
        log.info(f"Extracted {len(cell_ids)} cells")
        if len(cell_ids) == 0:
            return pv.UnstructuredGrid()
        return self._grid.extract_cells(cell_ids)