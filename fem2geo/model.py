import logging
from functools import cached_property

import numpy as np
import pyvista as pv

from fem2geo.internal.io import load_solver_output
from fem2geo.internal.schema import ModelSchema
from fem2geo.utils import tensor

__all__ = ["Model"]

log = logging.getLogger("fem2geoLogger")


class Model:
    """
    Interface to a FEM/BEM model for geomechanical post-processing.

    Wraps a mesh loaded from VTK/VTU files and provides named access to stress,
    strain, kinematic, and scalar fields in ENU coordinates.

    Parameters
    ----------
    grid : pyvista.UnstructuredGrid
        Mesh with arrays renamed to canonical names by
        :func:`~fem2geo.internal.io.load_solver_output`.
    schema : ModelSchema
        Schema mapping canonical names to solver-specific array names and units.

    See Also
    --------
    Model.from_file : Construct a Model directly from a VTK/VTU path.
    """

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
            Schema instance, name of a built-in schema (e.g. ``"adeli"``,
            ``"adeli2"``), or path to a custom schema YAML file.

        Returns
        -------
        Model
        """
        if isinstance(schema, str):
            schema = ModelSchema.load(schema)
        return cls(load_solver_output(path, schema), schema)

    # tensors

    @cached_property
    def strain(self) -> np.ndarray:
        """Total strain tensor :math:`\\varepsilon_{ij}`, shape (N, 3, 3)."""
        return self._assemble_tensor("strain")

    @cached_property
    def strain_rate(self) -> np.ndarray:
        """Total strain rate tensor :math:`\\dot{\\varepsilon}_{ij}`, shape (N, 3, 3)."""
        return self._assemble_tensor("strain_rate")

    # kinematics

    @cached_property
    def u(self) -> np.ndarray:
        """Displacement vector :math:`u_i`, shape (N, 3)."""
        return self.get("u")

    @cached_property
    def v(self) -> np.ndarray:
        """Velocity vector :math:`v_i`, shape (N, 3)."""
        return self.get("v")

    @cached_property
    def t(self) -> np.ndarray:
        """Time field :math:`t`, shape (N,)."""
        return self.get("t")

    # scalar fields

    @cached_property
    def i1_strain(self) -> np.ndarray:
        """First strain invariant :math:`I_1(\\varepsilon) = \\mathrm{tr}\\, \\varepsilon`, shape (N,)."""
        return self.get("i1_strain")

    @cached_property
    def j2_strain(self) -> np.ndarray:
        """Second deviatoric strain invariant :math:`J_2(\\varepsilon) = \\tfrac{1}{2} e_{ij} e_{ij}`, shape (N,)."""
        return self.get("j2_strain")

    @cached_property
    def j2_stress(self) -> np.ndarray:
        """Second deviatoric stress invariant :math:`J_2(\\sigma) = \\tfrac{1}{2} s_{ij} s_{ij}`, shape (N,)."""
        return self.get("j2_stress")

    @cached_property
    def i1_strain_rate(self) -> np.ndarray:
        """First strain rate invariant :math:`I_1(\\dot{\\varepsilon}) = \\mathrm{tr}\\,\\dot{\\varepsilon}`, shape (N,)."""
        return self.get("i1_strain_rate")

    @cached_property
    def j2_strain_rate(self) -> np.ndarray:
        """Second deviatoric strain rate invariant :math:`J_2(\\dot{\\varepsilon})`,
        shape (N,)."""
        return self.get("j2_strain_rate")

    @cached_property
    def plastic_eff(self) -> np.ndarray:
        """Effective plastic strain :math:`\\varepsilon^p_{\\mathrm{eff}}`, shape (N,
        )."""
        return self.get("plastic_eff")

    @cached_property
    def plastic_vol(self) -> np.ndarray:
        """Volumetric plastic strain :math:`\\varepsilon^p_{\\mathrm{vol}}`,
        shape (N,)."""
        return self.get("plastic_vol")

    @cached_property
    def plastic_yield(self) -> np.ndarray:
        """Plastic yield indicator, shape (N,)."""
        return self.get("plastic_yield")

    @cached_property
    def plastic_mode(self) -> np.ndarray:
        """Plastic failure mode, shape (N,)."""
        return self.get("plastic_mode")

    @cached_property
    def mean_stress(self) -> np.ndarray:
        """Mean stress (pressure) :math:`p = \\tfrac{1}{3}\\,\\mathrm{tr}\\,\\sigma`,
        shape (N,)."""
        return self.get("mean_stress")

    @cached_property
    def viscosity(self) -> np.ndarray:
        """Effective viscosity :math:`\\eta`, shape (N,)."""
        return self.get("viscosity")

    @cached_property
    def threshold_ratio(self) -> np.ndarray:
        """Yield threshold ratio, shape (N,)."""
        return self.get("threshold_ratio")

    @cached_property
    def temperature(self) -> np.ndarray:
        """Temperature field :math:`T`, shape (N,)."""
        return self.get("temperature")

    @cached_property
    def fluid_pressure(self) -> np.ndarray:
        """Fluid pressure :math:`p_f`, shape (N,)."""
        return self.get("fluid_pressure")

    @cached_property
    def darcy_vel(self) -> np.ndarray:
        """Darcy velocity vector :math:`q_i`, shape (N, 3)."""
        return self.get("darcy_vel")

    @cached_property
    def heat_flux(self) -> np.ndarray:
        """Heat flux vector :math:`q^T_i`, shape (N, 3)."""
        return self.get("heat_flux")

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
    def cell_volumes(self) -> np.ndarray:
        """Cell volumes :math:`V_c`, shape (N,)."""
        return self.grid.compute_cell_sizes().cell_data["Volume"]

    @cached_property
    def cell_centers(self) -> np.ndarray:
        """Cell center coordinates, shape (N, 3)."""
        return self.grid.cell_centers().points

    # stress

    @cached_property
    def stress(self) -> np.ndarray:
        """
        Full stress tensor :math:`\\sigma_{ij}`, shape (N, 3, 3).

        Assembled from the schema tensor if available. Otherwise, reconstructed from
        principal values and directions.
        """
        if "stress" in self.schema.tensors:
            try:
                return self._assemble_tensor("stress")
            except KeyError:
                log.warning("Full stress tensor not found")

        required = ("val_s1", "val_s3", "dir_s1", "dir_s3")
        missing = [
            k for k in required
            if k not in self.grid.cell_data
            and k not in self.grid.point_data
        ]
        if missing:
            raise KeyError(f"Cannot assemble stress — missing: {missing}")
        log.warning("Reconstructing stress tensor from principals.")
        vals = np.column_stack([self.val_s1, self.val_s2, self.val_s3])
        dirs = np.stack([self.dir_s1, self.dir_s2, self.dir_s3], axis=-1)
        return tensor.reconstruct_from_principals(vals, dirs)

    @cached_property
    def stress_dev(self) -> np.ndarray:
        r"""
        Deviatoric stress tensor, shape (N, 3, 3).

        .. math::

           s_{ij} = \sigma_{ij} - \tfrac{1}{3}\,\sigma_{kk}\,\delta_{ij}
        """
        s = self.stress
        trace = np.trace(s, axis1=1, axis2=2)
        return s - (trace / 3.0)[:, None, None] * np.eye(3)

    # strain

    @cached_property
    def strain_plastic(self) -> np.ndarray:
        """
        Plastic strain tensor :math:`\\varepsilon^p_{ij}`, shape (N, 3, 3).

        Loaded from schema if available. Otherwise, reconstructed from
        ``plastic_eff`` and ``plastic_vol`` assuming isotropic flow rules.
        """
        if "strain_plastic" in self.schema.tensors:
            try:
                return self._assemble_tensor("strain_plastic")
            except KeyError:
                log.warning(
                    "strain_plastic missing — reconstructing assuming coaxiality."
                )

        try:
            eff = self.plastic_eff
            vol = self.plastic_vol
        except KeyError:
            log.info("No plastic strain data — assuming elastic model.")
            return np.zeros((self.n_cells, 3, 3))

        if np.all(eff == 0) and np.all(vol == 0):
            return np.zeros((self.n_cells, 3, 3))

        dev = np.column_stack([self.val_s1, self.val_s2, self.val_s3])
        j2 = np.sqrt(0.5 * np.sum(dev**2, axis=1, keepdims=True))
        j2 = np.where(j2 < 1e-30, 1.0, j2)
        shape = dev / j2

        ep = eff[:, None] * shape + (vol / 3.0)[:, None]
        dirs = np.stack([self.dir_s1, self.dir_s2, self.dir_s3], axis=-1)
        return tensor.reconstruct_from_principals(ep, dirs)

    @cached_property
    def strain_elastic(self) -> np.ndarray:
        r"""
        Elastic strain tensor, shape (N, 3, 3).

        Loaded from schema if available, otherwise computed as

        .. math::

           \varepsilon^e_{ij} = \varepsilon_{ij} - \varepsilon^p_{ij}
        """
        if "strain_elastic" in self.schema.tensors:
            try:
                return self._assemble_tensor("strain_elastic")
            except KeyError:
                log.warning(
                    "strain_elastic missing. Computing as total - plastic."
                )
        return self.strain - self.strain_plastic

    # principal stress directions

    @cached_property
    def dir_s1(self) -> np.ndarray:
        """Maximum compressive principal stress direction :math:`\\mathbf{n}_1`,
        shape (N, 3)."""
        if "dir_s1" in self.grid.array_names:
            return self.get("dir_s1")
        else:
            return self.eigenvectors("stress")[:, :, 2]

    @cached_property
    def dir_s2(self) -> np.ndarray:
        """Intermediate principal stress direction :math:`\\mathbf{n}_2`, shape (N,
        3)."""
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
        """Minimum compressive principal stress direction :math:`\\mathbf{n}_3`,
        shape (N, 3)."""
        if "dir_s3" in self.grid.array_names:
            return self.get("dir_s3")
        else:
            return self.eigenvectors("stress")[:, :, 0]

    # eigendecomposition

    def eigenvectors(self, name: str) -> np.ndarray:
        """
        Eigenvectors of a tensor field, sorted by ascending eigenvalue (in continuum
        mechanics convention: compression is negative).

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

    # principal stress values

    @cached_property
    def val_s1(self) -> np.ndarray:
        """
        Most compressive principal stress :math:`\\sigma_1`, shape (N,).

        Loaded from the grid if available, otherwise the smallest eigenvalue
        of the stress tensor.
        """
        if "val_s1" in self.grid.array_names:
            return self.get("val_s1")
        return self.eigenvalues("stress")[:, 0]

    @cached_property
    def val_s2(self) -> np.ndarray:
        """
        Intermediate principal stress :math:`\\sigma_2`, shape (N,).

        Loaded from the grid if available, otherwise computed as
        ``-(val_s1 + val_s3)`` (deviatoric trace-free condition).
        """
        if "val_s2" in self.grid.array_names:
            return self.get("val_s2")
        return -(self.val_s1 + self.val_s3)

    @cached_property
    def val_s3(self) -> np.ndarray:
        """
        Least compressive principal stress :math:`\\sigma_3`, shape (N,).

        Loaded from the grid if available, otherwise the largest eigenvalue
        of the stress tensor.
        """
        if "val_s3" in self.grid.array_names:
            return self.get("val_s3")
        return self.eigenvalues("stress")[:, 2]

    # extraction

    def extract(self, center, radius) -> "Model":
        """
        Extract cells whose nodes fall within a sphere.

        Parameters
        ----------
        center : array-like, shape (3,)
            Sphere center in model coordinates.
        radius : float
            Sphere radius.
        """
        center = np.asarray(center)
        dist = np.linalg.norm(self.grid.points - center, axis=1)
        sub = Model(self._extract_cells(dist < radius), self.schema)
        if sub.n_cells == 0:
            raise ValueError(
                f"No cells found in sphere at {center.tolist()} with "
                f"radius {radius}"
            )
        return sub

    # averages

    def avg_tensor(self, name: str) -> np.ndarray:
        r"""
        Cell-volume-weighted average of a tensor field.

        .. math::

           \bar{T}_{ij} = \frac{\sum_c V_c\, T_{ij}^{(c)}}{\sum_c V_c}

        where :math:`V_c` is the volume of cell :math:`c` and
        :math:`T_{ij}^{(c)}` is the tensor value at that cell.

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
        t = getattr(self, name)
        w = self.cell_volumes
        avg = np.einsum("ijk,i->jk", t, w) / w.sum()
        return 0.5 * (avg + avg.T)

    def avg_principals(self, name: str = "stress") -> tuple:
        """
        Eigenvectors of the the volume-weighted average of a tensor field.

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