import logging
from functools import cached_property

import numpy as np
import pyvista as pv

from fem2geo.internal.io import load_grid
from fem2geo.internal.schema import ModelSchema
from fem2geo.utils.tensor import unpack_voigt6, unpack_components

log = logging.getLogger("fem2geoLogger")


class Model:
    """
    A FEM model with canonical field names and ENU-normalized directions.

    Construct via :meth:`from_file`. Properties are lazy and cached.
    Tensor assembly is schema-driven: the schema declares whether a tensor
    is stored as a packed Voigt array or as individual components, and
    :meth:`_assemble_tensor` dispatches accordingly.
    """

    def __init__(self, grid: pv.UnstructuredGrid, schema: ModelSchema):
        self._grid = grid
        self.schema = schema

    @classmethod
    def from_file(cls, path, schema: ModelSchema | str = "adeli") -> "Model":
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
        then delegates to the appropriate unpacker in ``tensor.py``.
        Arrays are looked up by canonical name (as renamed by ``load_grid``).
        """
        if not self.schema.has_tensor(name):
            raise KeyError(
                f"No tensor '{name}' in schema '{self.schema.name}'.")

        entry = self.schema.tensors[name]

        if entry.is_packed:
            return unpack_voigt6(self._array(name))

        arrays = {comp: self._array(f"_tensor_{name}_{comp}")
                  for comp in entry.components}
        return unpack_components(arrays)

    def _array(self, key: str) -> np.ndarray:
        """Look up a renamed array by canonical key."""
        if key in self._grid.cell_data:
            return np.asarray(self._grid.cell_data[key])
        if key in self._grid.point_data:
            return np.asarray(self._grid.point_data[key])
        raise KeyError(f"Array '{key}' not found in grid.")

    # stress

    @cached_property
    def stress(self) -> np.ndarray:
        """
        Stress tensor, shape (N, 3, 3).

        Tries schema tensor first, falls back to principal reconstruction.
        """
        if self.schema.has_tensor("stress"):
            try:
                return self._assemble_tensor("stress")
            except KeyError:
                log.warning("Stress tensor declared in schema but data "
                            "missing — falling back to reconstruction.")
        return self._reconstruct_stress()

    def _reconstruct_stress(self) -> np.ndarray:
        required = ("val_s1", "val_s3", "dir_s1", "dir_s3")
        missing = [k for k in required
                   if k not in self._grid.cell_data
                   and k not in self._grid.point_data]
        if missing:
            raise KeyError(
                f"Cannot assemble stress — missing: {missing}")
        log.warning("Reconstructing stress from principals.")
        v1, v3 = self.val_s1, self.val_s3
        v2 = -(v1 + v3)
        d1, d2, d3 = self.dir_s1, self.dir_s2, self.dir_s3
        return (v1[:, None, None] * (d1[:, :, None] * d1[:, None, :])
                + v2[:, None, None] * (d2[:, :, None] * d2[:, None, :])
                + v3[:, None, None] * (d3[:, :, None] * d3[:, None, :]))

    @cached_property
    def stress_dev(self) -> np.ndarray:
        """Deviatoric stress tensor, shape (N, 3, 3). Removes ⅓tr(σ)I per cell."""
        s = self.stress
        tr = np.trace(s, axis1=1, axis2=2)
        return s - (tr / 3.0)[:, None, None] * np.eye(3)

    # strain

    @cached_property
    def strain(self) -> np.ndarray:
        """Total strain tensor, shape (N, 3, 3)."""
        return self._assemble_tensor("strain")

    @cached_property
    def strain_rate(self) -> np.ndarray:
        """Total strain rate tensor, shape (N, 3, 3)."""
        return self._assemble_tensor("strain_rate")

    @cached_property
    def strain_plastic(self) -> np.ndarray:
        """
        Plastic strain tensor, shape (N, 3, 3).

        If the schema declares a ``strain_plastic`` tensor, it is loaded
        directly. Otherwise, the tensor is reconstructed from the scalar
        invariants ``plastic_eff`` (deviatoric magnitude) and ``plastic_vol``
        (volumetric part) assuming **coaxiality** with the stress tensor:
        the plastic strain principal directions are taken from the stress
        eigenvectors, and the deviatoric shape is proportional to the
        stress deviator.

        This assumption holds exactly for isotropic associated flow rules
        and is a good approximation for non-associated Drucker-Prager /
        Mohr-Coulomb plasticity.
        """
        if self.schema.has_tensor("strain_plastic"):
            try:
                return self._assemble_tensor("strain_plastic")
            except KeyError:
                log.warning("strain_plastic declared but data missing — "
                            "falling back to coaxiality reconstruction.")
        return self._reconstruct_plastic_strain()

    @cached_property
    def strain_elastic(self) -> np.ndarray:
        """
        Elastic strain tensor, shape (N, 3, 3).

        If the schema declares a ``strain_elastic`` tensor, it is loaded
        directly. Otherwise computed as ``strain - strain_plastic``.
        """
        if self.schema.has_tensor("strain_elastic"):
            try:
                return self._assemble_tensor("strain_elastic")
            except KeyError:
                log.warning("strain_elastic declared but data missing — "
                            "computing as total - plastic.")
        return self.strain - self.strain_plastic

    def _reconstruct_plastic_strain(self) -> np.ndarray:
        """
        Reconstruct plastic strain tensor from scalar invariants and
        stress eigenvectors (coaxiality assumption).

        Falls back to a zero tensor if plastic invariants are not
        available (purely elastic model).
        """
        try:
            eff = self.plastic_eff
            vol = self.plastic_vol
        except KeyError:
            log.info("No plastic strain data — assuming purely elastic.")
            return np.zeros((self.n_cells, 3, 3))

        if np.all(eff == 0) and np.all(vol == 0):
            return np.zeros((self.n_cells, 3, 3))

        d1, d2, d3 = self.dir_s1, self.dir_s2, self.dir_s3

        # deviatoric stress eigenvalues for shape
        s1, s2, s3 = self.val_s1, self.val_s2, self.val_s3
        dev = np.column_stack([s1, s2, s3])

        # normalize deviatoric shape to unit J2
        j2_dev = np.sqrt(0.5 * np.sum(dev ** 2, axis=1, keepdims=True))
        j2_dev = np.where(j2_dev < 1e-30, 1.0, j2_dev)
        shape = dev / j2_dev

        # scale to plastic_eff magnitude and add volumetric part
        ep_dev = eff[:, None] * shape
        ep = ep_dev + (vol / 3.0)[:, None]

        # reconstruct tensor: ε_p = Σ ep_i * (d_i ⊗ d_i)
        return (ep[:, 0, None, None] * (d1[:, :, None] * d1[:, None, :])
                + ep[:, 1, None, None] * (d2[:, :, None] * d2[:, None, :])
                + ep[:, 2, None, None] * (d3[:, :, None] * d3[:, None, :]))

    # principal directions

    @cached_property
    def dir_s1(self) -> np.ndarray:
        return self._field("dir_s1")

    @cached_property
    def dir_s3(self) -> np.ndarray:
        return self._field("dir_s3")

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

    # principal values

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

    # kinematics

    @cached_property
    def u(self) -> np.ndarray:
        return self._field("u")

    @cached_property
    def v(self) -> np.ndarray:
        return self._field("v")

    @cached_property
    def t(self) -> np.ndarray:
        return self._field("t")

    # scalar fields

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

    @cached_property
    def mean_stress(self) -> np.ndarray:
        return self._field("mean_stress")

    @cached_property
    def plastic_mode(self) -> np.ndarray:
        return self._field("plastic_mode")

    @cached_property
    def viscosity(self) -> np.ndarray:
        return self._field("viscosity")

    @cached_property
    def threshold_ratio(self) -> np.ndarray:
        return self._field("threshold_ratio")

    @cached_property
    def temperature(self) -> np.ndarray:
        return self._field("temperature")

    @cached_property
    def fluid_pressure(self) -> np.ndarray:
        return self._field("fluid_pressure")

    @cached_property
    def i1_strain_rate(self) -> np.ndarray:
        return self._field("i1_strain_rate")

    @cached_property
    def j2_strain_rate(self) -> np.ndarray:
        return self._field("j2_strain_rate")

    @cached_property
    def darcy_vel(self) -> np.ndarray:
        return self._field("darcy_vel")

    @cached_property
    def heat_flux(self) -> np.ndarray:
        return self._field("heat_flux")

    # extraction

    def extract(self, zone: dict) -> "Model":
        """
        Extract a sub-model from a zone config dict.

        Parameters
        ----------
        zone : dict
            Must contain ``type`` (``sphere`` or ``box``), ``center``,
            and either ``radius`` (sphere) or ``dim`` (box).
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
        mask = np.linalg.norm(self._grid.points - center, axis=1) < radius
        return Model(self._extract(mask), self.schema)

    def extract_box(self, center, dim) -> "Model":
        """Extract cells touched by an axis-aligned bounding box."""
        center, dim = np.asarray(center), np.asarray(dim)
        ll, ur = center - dim / 2.0, center + dim / 2.0
        mask = np.all((self._grid.points >= ll) & (self._grid.points <= ur),
                      axis=1)
        return Model(self._extract(mask), self.schema)

    # averages

    def avg_tensor(self, name: str) -> np.ndarray:
        """
        Volume-weighted average of a tensor field, shape (3, 3).

        Parameters
        ----------
        name : str
            Any tensor: 'stress', 'stress_dev', 'strain', 'strain_rate',
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
        return np.einsum("ijk,i->jk", tensors,
                         self.volumes) / self.volumes.sum()

    def avg_principal(self, name: str = "stress") -> tuple:
        """
        Volume-weighted average principal values and directions.

        Parameters
        ----------
        name : str
            Tensor to decompose (default: 'stress').

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
        raise KeyError(f"Field '{canonical}' not found in grid.")

    def _extract(self, point_mask: np.ndarray) -> pv.UnstructuredGrid:
        """Extract cells that reference at least one flagged point."""
        conn = self._grid.cell_connectivity
        starts = self._grid.offset[:-1]
        flagged = point_mask[conn].astype(np.intp)
        counts = np.add.reduceat(flagged, starts)
        cell_ids = np.where(counts > 0)[0]
        log.info(f"Extracted {len(cell_ids)} cells")
        if len(cell_ids) == 0:
            return pv.UnstructuredGrid()
        return self._grid.extract_cells(cell_ids)