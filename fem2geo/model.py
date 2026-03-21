import logging
from functools import cached_property

import numpy as np
import pyvista as pv

from fem2geo.internal.io import load_grid
from fem2geo.internal.schema import ModelSchema
from fem2geo.utils import tensor

log = logging.getLogger("fem2geoLogger")


# field descriptors

class ScalarField:
    """Lazy descriptor for scalar fields."""

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        val = obj.get(self.name)
        setattr(obj, self.name, val)
        return val


class VectorField:
    """Lazy descriptor for vector fields."""

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        val = obj.get(self.name)
        setattr(obj, self.name, val)
        return val


class TensorField:
    """Lazy descriptor for tensor fields."""

    def __init__(self, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        val = obj._assemble_tensor(self.name)   # noqa
        setattr(obj, self.name, val)
        return val


class Model:
    """
    Interface to a FEM/BEM model for geomechanical post-processing.

    Wraps a mesh loaded from VTK/VTU files and provides named access to stress,
    strain, kinematic, and scalar fields in ENU coordinates. Fields are lazy-loaded
    and cached on first access.

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        Mesh with arrays renamed to canonical names by
        :func:`~fem2geo.internal.io.load_grid`.
    schema : ModelSchema
        Schema mapping canonical names to solver-specific array names and units.

    See Also
    --------
    from_file : Construct a Model directly from a VTK/VTU path.

    Attributes
    ----------
    stress : numpy.ndarray, shape (N, 3, 3)
        Full stress tensor.
    stress_dev : numpy.ndarray, shape (N, 3, 3)
        Deviatoric stress tensor.
    strain : numpy.ndarray, shape (N, 3, 3)
        Total strain tensor.
    strain_rate : numpy.ndarray, shape (N, 3, 3)
        Total strain rate tensor.
    strain_plastic : numpy.ndarray, shape (N, 3, 3)
        Plastic strain tensor.
    strain_elastic : numpy.ndarray, shape (N, 3, 3)
        Elastic strain tensor.
    u : numpy.ndarray, shape (N, 3)
        Displacement vectors.
    v : numpy.ndarray, shape (N, 3)
        Velocity vectors.
    t : numpy.ndarray, shape (N,)
        Time field (if available).
    dir_s1 : numpy.ndarray, shape (N, 3)
        Most compressive principal stress direction (ENU).
    dir_s3 : numpy.ndarray, shape (N, 3)
        Least compressive principal stress direction (ENU).
    i1_strain : numpy.ndarray, shape (N,)
        First strain invariant (volumetric strain).
    j2_strain : numpy.ndarray, shape (N,)
        Second deviatoric strain invariant.
    j2_stress : numpy.ndarray, shape (N,)
        Second deviatoric stress invariant.
    i1_strain_rate : numpy.ndarray, shape (N,)
        First strain rate invariant.
    j2_strain_rate : numpy.ndarray, shape (N,)
        Second deviatoric strain rate invariant.
    plastic_eff : numpy.ndarray, shape (N,)
        Effective plastic strain.
    plastic_vol : numpy.ndarray, shape (N,)
        Volumetric plastic strain.
    plastic_yield : numpy.ndarray, shape (N,)
        Plastic yield indicator.
    plastic_mode : numpy.ndarray, shape (N,)
        Plastic failure mode.
    mean_stress : numpy.ndarray, shape (N,)
        Mean stress (pressure).
    viscosity : numpy.ndarray, shape (N,)
        Effective viscosity.
    threshold_ratio : numpy.ndarray, shape (N,)
        Yield threshold ratio.
    temperature : numpy.ndarray, shape (N,)
        Temperature field.
    fluid_pressure : numpy.ndarray, shape (N,)
        Fluid pressure.
    darcy_vel : numpy.ndarray, shape (N, 3)
        Darcy velocity vectors.
    heat_flux : numpy.ndarray, shape (N, 3)
        Heat flux vectors.
    """

    # tensors
    strain      = TensorField("strain")
    strain_rate = TensorField("strain_rate")

    # kinematics
    u = VectorField("u")
    v = VectorField("v")
    t = ScalarField("t")

    # scalar fields
    i1_strain       = ScalarField("i1_strain")
    j2_strain       = ScalarField("j2_strain")
    j2_stress       = ScalarField("j2_stress")
    i1_strain_rate  = ScalarField("i1_strain_rate")
    j2_strain_rate  = ScalarField("j2_strain_rate")
    plastic_eff     = ScalarField("plastic_eff")
    plastic_vol     = ScalarField("plastic_vol")
    plastic_yield   = ScalarField("plastic_yield")
    plastic_mode    = ScalarField("plastic_mode")
    mean_stress     = ScalarField("mean_stress")
    viscosity       = ScalarField("viscosity")
    threshold_ratio = ScalarField("threshold_ratio")
    temperature     = ScalarField("temperature")
    fluid_pressure  = ScalarField("fluid_pressure")
    darcy_vel       = VectorField("darcy_vel")
    heat_flux       = VectorField("heat_flux")

    def __init__(self, grid: pv.UnstructuredGrid, schema: ModelSchema):
        self.grid = grid
        self.schema = schema

    @classmethod
    def from_file(cls, path, schema: ModelSchema | str = "adeli") -> "Model":
        """
        Load a model from a VTK/VTU file.

        Parameters
        ----------
        path : str or Path
            Path to the mesh file.
        schema : ModelSchema or str
            Schema instance or name of a built-in schema (e.g. `"adeli"``,
             ``"adeli2"``).

        Returns
        -------
        Model
        """
        if isinstance(schema, str):
            schema = ModelSchema.builtin(schema)
        return cls(load_grid(path, schema), schema)

    # geometry

    @property
    def points(self) -> np.ndarray:
        """Mesh node coordinates, shape (N_points, 3)."""
        return self.grid.points

    @property
    def n_cells(self) -> int:
        """Number of cells in the mesh."""
        return self.grid.number_of_cells

    @cached_property
    def volumes(self) -> np.ndarray:
        """Cell volumes, shape (N,)."""
        return self.grid.compute_cell_sizes().cell_data["Volume"]

    @cached_property
    def cell_centers(self) -> np.ndarray:
        """Cell center coordinates, shape (N, 3)."""
        return self.grid.cell_centers().points

    # stress

    @cached_property
    def stress(self) -> np.ndarray:
        """
        Full stress tensor, shape (N, 3, 3).

        Assembled from the schema tensor if available. Otherwise, reconstructed from
        principal values and directions.
        """
        if "stress" in self.schema.tensors:
            try:
                return self._assemble_tensor("stress")
            except KeyError:
                log.warning("Full stress tensor not found")
        return self._reconstruct_stress()

    @cached_property
    def stress_dev(self) -> np.ndarray:
        """
        Deviatoric stress tensor, shape (N, 3, 3).

        Computed as ``stress - (1/3) tr(stress) I`` per cell.
        """
        s = self.stress
        trace = np.trace(s, axis1=1, axis2=2)
        return s - (trace / 3.0)[:, None, None] * np.eye(3)

    def _reconstruct_stress(self) -> np.ndarray:
        required = ("val_s1", "val_s3", "dir_s1", "dir_s3")
        missing = [
            k for k in required
            if k not in self.grid.cell_data
            and k not in self.grid.point_data
        ]
        if missing:
            raise KeyError(f"Cannot assemble stress — missing: {missing}")
        log.warning("Reconstructing stress tensor from principals.")
        v1, v3 = self.val_s1, self.val_s3
        v2 = -(v1 + v3)
        d1, d2, d3 = self.dir_s1, self.dir_s2, self.dir_s3
        return (
            v1[:, None, None] * (d1[:, :, None] * d1[:, None, :])
            + v2[:, None, None] * (d2[:, :, None] * d2[:, None, :])
            + v3[:, None, None] * (d3[:, :, None] * d3[:, None, :])
        )

    # strain

    @cached_property
    def strain_plastic(self) -> np.ndarray:
        """
        Plastic strain tensor, shape (N, 3, 3).

        Loaded from schema if available. Otherwise, reconstructed from
        ``plastic_eff`` and ``plastic_vol`` assuming isotropic flow rules).
        """
        if "strain_plastic" in self.schema.tensors:
            try:
                return self._assemble_tensor("strain_plastic")
            except KeyError:
                log.warning(
                    "strain_plastic missing — reconstructing assuming coaxiality."
                )
        return self._reconstruct_plastic_strain()

    @cached_property
    def strain_elastic(self) -> np.ndarray:
        """
        Elastic strain tensor, shape (N, 3, 3).

        Loaded from schema if available, otherwise computed as ``strain -
        strain_plastic``.
        """
        if "strain_elastic" in self.schema.tensors:
            try:
                return self._assemble_tensor("strain_elastic")
            except KeyError:
                log.warning(
                    "strain_elastic missing. Computing as total - plastic."
                )
        return self.strain - self.strain_plastic

    def _reconstruct_plastic_strain(self) -> np.ndarray:
        try:
            eff = self.plastic_eff
            vol = self.plastic_vol
        except KeyError:
            log.info("No plastic strain data — assuming elastic model.")
            return np.zeros((self.n_cells, 3, 3))

        if np.all(eff == 0) and np.all(vol == 0):
            return np.zeros((self.n_cells, 3, 3))

        d1, d2, d3 = self.dir_s1, self.dir_s2, self.dir_s3
        s1, s2, s3 = self.val_s1, self.val_s2, self.val_s3
        dev = np.column_stack([s1, s2, s3])

        j2 = np.sqrt(0.5 * np.sum(dev**2, axis=1, keepdims=True))
        j2 = np.where(j2 < 1e-30, 1.0, j2)
        shape = dev / j2

        ep_dev = eff[:, None] * shape
        ep = ep_dev + (vol / 3.0)[:, None]

        return (
            ep[:, 0, None, None] * (d1[:, :, None] * d1[:, None, :])
            + ep[:, 1, None, None] * (d2[:, :, None] * d2[:, None, :])
            + ep[:, 2, None, None] * (d3[:, :, None] * d3[:, None, :])
        )


    # principal stress directions
    @cached_property
    def dir_s1(self) -> np.ndarray:
        """
        Maximum compressive principal stress direction, shape (N, 3).
        """
        if "dir_s1" in self.grid.array_names:
            return self.get("dir_s1")
        else:
            return self.eigenvectors("stress")[:, :, 2]

    @cached_property
    def dir_s2(self) -> np.ndarray:
        """
        Intermediate principal stress direction, shape (N, 3).
        """
        if "dir_s2" in self.grid.array_names:
            return self.get("dir_s2")
        elif "dir_s3" in self.grid.array_names:
            d = np.cross(self.dir_s1, self.dir_s3)
            norms = np.linalg.norm(d, axis=1, keepdims=True)
            return d / np.where(norms < 1e-12, 1.0, norms)
        else:
            return self.eigenvectors("stress")[:, :, 1]

    @cached_property
    def dir_s3(self) -> np.ndarray:
        """
        Minimum compressive principal stress direction, shape (N, 3).
        """
        if "dir_s3" in self.grid.array_names:
            return self.get("dir_s3")
        else:
            return self.eigenvectors("stress")[:, :, 0]


    # eigendecomposition

    def eigenvectors(self, name: str) -> np.ndarray:
        """
        Eigenvectors of a tensor field, sorted by ascending eigenvalue (in continuum
        mechanics convention: compression is negative)

        Parameters
        ----------
        name : str
            Canonical tensor name (e.g. ``'stress'``, ``'strain_plastic'``).

        Returns
        -------
        numpy.ndarray, shape (N, 3, 3)
            Eigenvectors as columns per cell.
        """
        return tensor.eigenvectors(getattr(self, name))

    # principal stress values

    def eigenvalues(self, name: str) -> np.ndarray:
        """
        Eigenvalues of a tensor field, sorted ascending (in continuum mechanics
         convention: compression is negative).

        Parameters
        ----------
        name : str
            Canonical tensor name (e.g. ``'stress'``, ``'strain_plastic'``).

        Returns
        -------
        numpy.ndarray, shape (N, 3)
            Eigenvalues per cell, column 0 smallest (most compressive).
        """
        return tensor.eigenvalues(getattr(self, name))

    @cached_property
    def val_s1(self) -> np.ndarray:
        """
        Most compressive principal stress value, shape (N,).

        Loaded from the grid if available, otherwise the smallest eigenvalue
        of the stress tensor.
        """
        if "val_s1" in self.grid.array_names:
            return self.get("val_s1")
        return self.eigenvalues("stress")[:, 0]

    @cached_property
    def val_s2(self) -> np.ndarray:
        """
        Intermediate principal stress value, shape (N,).

        Loaded from the grid if available, otherwise computed as
        ``-(val_s1 + val_s3)`` (deviatoric trace-free condition).
        """
        if "val_s2" in self.grid.array_names:
            return self.get("val_s2")
        return -(self.val_s1 + self.val_s3)

    @cached_property
    def val_s3(self) -> np.ndarray:
        """
        Least compressive principal stress value, shape (N,).

        Loaded from the grid if available, otherwise the largest eigenvalue
        of the stress tensor.
        """
        if "val_s3" in self.grid.array_names:
            return self.get("val_s3")
        return self.eigenvalues("stress")[:, 2]

    # extraction

    def extract(self, zone: dict) -> "Model":
        """
        Extract a sub-model from a zone config dict.

        Parameters
        ----------
        zone : dict
            Must contain ``type`` (``"sphere"`` or ``"box"``),
            ``center``, and ``radius`` (sphere) or ``dim`` (box).

        Returns
        -------
        Model
            A new Model containing only cells within the zone.
        """
        kind = zone["type"]
        if kind == "sphere":
            return self.extract_sphere(zone["center"], zone["radius"])
        if kind == "box":
            return self.extract_box(zone["center"], zone["dim"])
        raise ValueError(f"Unknown zone type '{kind}'.")

    def extract_sphere(self, center, radius) -> "Model":
        """
        Extract cells whose nodes fall within a sphere.

        Parameters
        ----------
        center : array-like, shape (3,)
            Sphere center in model coordinates.
        radius : float
            Sphere radius.

        Returns
        -------
        Model
        """
        center = np.asarray(center)
        dist = np.linalg.norm(self.grid.points - center, axis=1)
        return Model(self._extract_cells(dist < radius), self.schema)

    def extract_box(self, center, dim) -> "Model":
        """
        Extract cells whose nodes fall within an axis-aligned box.

        Parameters
        ----------
        center : array-like, shape (3,)
            Box center in model coordinates.
        dim : array-like, shape (3,)
            Box dimensions (full width in each direction).

        Returns
        -------
        Model
        """
        center, dim = np.asarray(center), np.asarray(dim)
        lo, hi = center - dim / 2.0, center + dim / 2.0
        pts = self.grid.points
        mask = np.all((pts >= lo) & (pts <= hi), axis=1)
        return Model(self._extract_cells(mask), self.schema)

    # averages

    def avg_tensor(self, name: str) -> np.ndarray:
        """
        Volume-weighted average of a tensor field.

        Parameters
        ----------
        name : str
            Tensor name. Accepts schema tensors and computed
            properties: ``'stress'``, ``'stress_dev'``,
            ``'strain'``, ``'strain_rate'``,
            ``'strain_plastic'``, ``'strain_elastic'``.

        Returns
        -------
        numpy.ndarray, shape (3, 3)
            Symmetric average tensor.
        """
        computed = {
            "stress", "stress_dev", "strain", "strain_rate",
            "strain_plastic", "strain_elastic",
        }
        tensors = (
            getattr(self, name) if name in computed
            else self._assemble_tensor(name)
        )
        w = self.volumes
        return np.einsum("ijk,i->jk", tensors, w) / w.sum()

    def avg_principals(self, name: str = "stress") -> tuple:
        """
        Eigendecompose the volume-weighted average of a tensor field.

        Parameters
        ----------
        name : str
            Canonical tensor name (default: ``'stress'``).

        Returns
        -------
        val : numpy.ndarray, shape (3,)
            Eigenvalues sorted ascending.
        vec : numpy.ndarray, shape (3, 3)
            Eigenvectors as columns in ENU coordinates.
        """
        avg = self.avg_tensor(name)
        val, vec = np.linalg.eigh(avg)
        order = np.argsort(val)
        return val[order], vec[:, order]

    # persistence

    def save(self, path) -> None:
        """
        Save the model grid to a VTK/VTU file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        self.grid.save(str(path))

    # internal

    def get(self, canonical: str) -> np.ndarray:
        """Look up a named array from cell or point data."""
        for store in (self.grid.cell_data, self.grid.point_data):
            if canonical in store:
                return np.asarray(store[canonical])
        raise KeyError(f"Field '{canonical}' not found in grid.")

    def _assemble_tensor(self, name: str) -> np.ndarray:
        if name not in self.schema.tensors:
            raise KeyError(f"No tensor '{name}' in schema '{self.schema.name}'.")
        entry = self.schema.tensors[name]
        if entry.is_packed:
            return tensor.unpack_voigt6(self.get(name))
        arrays = {comp: self.get(f"_tensor_{name}_{comp}") for comp in entry.components}
        return tensor.unpack_components(arrays)

    def _extract_cells(self, point_mask: np.ndarray) -> pv.UnstructuredGrid:
        conn = self.grid.cell_connectivity
        starts = self.grid.offset[:-1]
        flagged = point_mask[conn].astype(np.intp)
        counts = np.add.reduceat(flagged, starts)
        cell_ids = np.where(counts > 0)[0]
        log.info(f"Extracted {len(cell_ids)} cells")
        if len(cell_ids) == 0:
            return pv.UnstructuredGrid()
        return self.grid.extract_cells(cell_ids)