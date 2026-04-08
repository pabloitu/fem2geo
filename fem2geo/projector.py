"""
Project georeferenced data into a FEM coordinate frame.

A Projector handles three things: a CRS reprojection, unit and sign
conversion for XY and Z, and an optional anchor + rotation that pins a
chosen geographic point to a chosen FEM coordinate. It can be applied
to raw arrays, point catalogs, or PyVista meshes.
"""

import logging

import numpy as np
import pyvista as pv
from pyproj import CRS

from fem2geo.data import CatalogData
from fem2geo.utils.projections import (
    unit_factor, flip_z, reproject_xy, rotate_xy,
)

log = logging.getLogger("fem2geoLogger")


class Projector:
    """
    Transform georeferenced coordinates into a FEM frame.

    Two required arguments (the source and destination CRS) plus a set
    of unit/sign conventions and an optional alignment anchor. Once
    built, the same Projector can be applied to arrays, CatalogData
    objects, or PyVista meshes.

    Parameters
    ----------
    src_crs, dst_crs : str or pyproj.CRS
        Source and destination coordinate reference systems.
    src_xy_units, dst_xy_units : str
        ``"deg"`` for a geographic source, ``"m"`` or ``"km"`` for
        projected coordinates.
    src_z_units, dst_z_units : str
        ``"m"`` or ``"km"``.
    src_z_positive, dst_z_positive : str
        ``"up"`` or ``"down"``.
    anchor_geo : tuple, optional
        ``(lon, lat, depth_km)`` of a reference point. Depth is positive
        downward. Must be set together with ``anchor_fem``.
    anchor_fem : tuple, optional
        ``(x, y, z)`` of the same reference point in the FEM frame, in
        ``dst_xy_units`` and following ``dst_z_positive``.
    rotation_deg : float, optional
        Counter-clockwise rotation around the anchor, in degrees.
        Requires an anchor.
    """

    def __init__(
        self,
        src_crs, dst_crs,
        src_xy_units="deg", dst_xy_units="m",
        src_z_units="km", dst_z_units="m",
        src_z_positive="down", dst_z_positive="up",
        anchor_geo=None, anchor_fem=None, rotation_deg=None,
    ):
        self.src_crs = CRS.from_user_input(src_crs)
        self.dst_crs = CRS.from_user_input(dst_crs)
        self.src_xy_units = str(src_xy_units).strip().lower()
        self.dst_xy_units = str(dst_xy_units).strip().lower()
        self.src_z_units = str(src_z_units).strip().lower()
        self.dst_z_units = str(dst_z_units).strip().lower()
        self.src_z_positive = str(src_z_positive).strip().lower()
        self.dst_z_positive = str(dst_z_positive).strip().lower()
        self.rotation_deg = None if rotation_deg is None else float(rotation_deg)

        self._validate_units()
        self._validate_anchor(anchor_geo, anchor_fem)

        self.anchor_geo = tuple(anchor_geo) if anchor_geo is not None else None
        self.anchor_fem = tuple(anchor_fem) if anchor_fem is not None else None

        self.dx, self.dy, self.dz = self._anchor_offset()

    def _validate_units(self):
        if self.dst_xy_units not in ("m", "km"):
            raise ValueError("dst_xy_units must be 'm' or 'km'.")
        if self.src_crs.is_geographic:
            if self.src_xy_units != "deg":
                raise ValueError(
                    "src_xy_units must be 'deg' for a geographic source CRS."
                )
        else:
            if self.src_xy_units not in ("m", "km"):
                raise ValueError(
                    "src_xy_units must be 'm' or 'km' for a projected source CRS."
                )
        unit_factor(self.src_z_units)
        unit_factor(self.dst_z_units)
        for name, val in (("src_z_positive", self.src_z_positive),
                          ("dst_z_positive", self.dst_z_positive)):
            if val not in ("up", "down"):
                raise ValueError(f"{name} must be 'up' or 'down'.")

    def _validate_anchor(self, anchor_geo, anchor_fem):
        if (anchor_geo is None) != (anchor_fem is None):
            raise ValueError("anchor_geo and anchor_fem must be set together.")
        if self.rotation_deg is not None and anchor_geo is None:
            raise ValueError("rotation_deg requires an anchor.")
        for name, val in (("anchor_geo", anchor_geo), ("anchor_fem", anchor_fem)):
            if val is not None and len(val) != 3:
                raise ValueError(f"{name} must have length 3.")

    def _anchor_offset(self):
        if self.anchor_geo is None:
            return 0.0, 0.0, 0.0

        lon, lat, depth_km = self.anchor_geo
        ax, ay = reproject_xy([lon], [lat], "epsg:4326", self.dst_crs)
        ax = ax[0] / unit_factor(self.dst_xy_units)
        ay = ay[0] / unit_factor(self.dst_xy_units)

        depth_dst = depth_km * 1000.0 / unit_factor(self.dst_z_units)
        az = -depth_dst if self.dst_z_positive == "up" else depth_dst

        x0, y0, z0 = self.anchor_fem
        return x0 - ax, y0 - ay, z0 - az

    # core

    def transform(self, x, y, z):
        """
        Transform arrays of source coordinates into the FEM frame.

        Parameters
        ----------
        x, y : array-like
            XY in the source CRS, in ``src_xy_units``.
        z : array-like
            Z in ``src_z_units`` following ``src_z_positive``.

        Returns
        -------
        X, Y, Z : numpy.ndarray
            Coordinates in the FEM frame.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        z = np.asarray(z, dtype=float)

        if self.src_crs.is_geographic:
            X, Y = reproject_xy(x, y, self.src_crs, self.dst_crs)
        else:
            s = unit_factor(self.src_xy_units)
            X, Y = reproject_xy(x * s, y * s, self.src_crs, self.dst_crs)

        X = X / unit_factor(self.dst_xy_units)
        Y = Y / unit_factor(self.dst_xy_units)

        Z = z * unit_factor(self.src_z_units)
        Z = flip_z(Z, self.src_z_positive, self.dst_z_positive)
        Z = Z / unit_factor(self.dst_z_units)

        X, Y, Z = X + self.dx, Y + self.dy, Z + self.dz

        if self.rotation_deg is not None:
            x0, y0, _ = self.anchor_fem
            X, Y = rotate_xy(X, Y, x0, y0, self.rotation_deg)

        return X, Y, Z

    # catalog

    def transform_catalog(self, cat):
        """
        Project a CatalogData into the FEM frame.

        Returns a new CatalogData with transformed coordinates and the
        original ``attrs`` carried through unchanged.
        """
        X, Y, Z = self.transform(cat.x, cat.y, cat.z)
        return CatalogData(
            x=X, y=Y, z=Z,
            attrs={k: v.copy() for k, v in cat.attrs.items()},
        )

    # mesh

    def transform_mesh(self, mesh, mesh_units=None):
        """
        Project a PyVista mesh into the FEM frame.

        Mesh Z is assumed positive up. Cell connectivity and all data
        arrays are preserved. The input mesh is not modified.

        Parameters
        ----------
        mesh : pyvista.DataSet
        mesh_units : str, optional
            ``"m"`` or ``"km"``. If different from ``src_xy_units`` or
            ``src_z_units``, the mesh points are rescaled before
            projection. Defaults to ``src_xy_units``.

        Returns
        -------
        pyvista.DataSet
            A copy of the input mesh with transformed points.
        """
        out = mesh.copy()
        pts = np.asarray(out.points, dtype=float)

        u = self.src_xy_units if mesh_units is None else str(mesh_units).strip().lower()
        if u not in ("m", "km"):
            raise ValueError("mesh_units must be 'm' or 'km'.")

        xy_scale = unit_factor(u) / unit_factor(self.src_xy_units)
        z_scale = unit_factor(u) / unit_factor(self.src_z_units)

        x_in = pts[:, 0] * xy_scale
        y_in = pts[:, 1] * xy_scale
        z_in = pts[:, 2] * z_scale

        if self.src_z_positive == "down":
            z_in = -z_in

        X, Y, Z = self.transform(x_in, y_in, z_in)
        out.points = np.c_[X, Y, Z]
        return out