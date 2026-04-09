import numpy as np
from pyproj import CRS

from fem2geo.data import CatalogData
from fem2geo.utils.projections import (
    unit_factor, flip_z, reproject_xy, rotate_xy,
)


class Projector:
    """
    Transform georeferenced coordinates into a real-world or local cartesian frame.

    A Projector holds a source/destination CRS pair, unit and sign
    conventions for XY and Z, and an optional alignment anchor that
    pins a chosen geographic point to a chosen local coordinate, with
    optional rotation around that anchor. Projector can be used to transform raw
    arrays, CatalogData objects, or meshes.

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
        ``(lon, lat, depth_km)`` of a reference point. Depth is
        positive downward. Must be set together with ``anchor_local``.
    anchor_local : tuple, optional
        ``(x, y, z)`` of the same reference point in the local frame,
        in ``dst_xy_units`` and following ``dst_z_positive``.
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
        anchor_geo=None, anchor_local=None, rotation_deg=None,
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

        # validate units
        if self.dst_xy_units not in ("m", "km"):
            raise ValueError("dst_xy_units must be 'm' or 'km'.")
        if self.src_crs.is_geographic:
            if self.src_xy_units != "deg":
                raise ValueError(
                    "src_xy_units must be 'deg' for a geographic source CRS."
                )
        elif self.src_xy_units not in ("m", "km"):
            raise ValueError(
                "src_xy_units must be 'm' or 'km' for a projected source CRS."
            )
        unit_factor(self.src_z_units)
        unit_factor(self.dst_z_units)
        for name, val in (("src_z_positive", self.src_z_positive),
                          ("dst_z_positive", self.dst_z_positive)):
            if val not in ("up", "down"):
                raise ValueError(f"{name} must be 'up' or 'down'.")

        # validate anchor
        if (anchor_geo is None) != (anchor_local is None):
            raise ValueError("anchor_geo and anchor_local must be set together.")
        if self.rotation_deg is not None and anchor_geo is None:
            raise ValueError("rotation_deg requires an anchor.")
        for name, val in (("anchor_geo", anchor_geo),
                          ("anchor_local", anchor_local)):
            if val is not None and len(val) != 3:
                raise ValueError(f"{name} must have length 3.")

        self.anchor_geo = tuple(anchor_geo) if anchor_geo is not None else None
        self.anchor_local = (
            tuple(anchor_local) if anchor_local is not None else None
        )

        # compute anchor offset
        if self.anchor_geo is None:
            self.dx, self.dy, self.dz = 0.0, 0.0, 0.0
        else:
            lon, lat, depth_km = self.anchor_geo
            ax, ay = reproject_xy([lon], [lat], "epsg:4326", self.dst_crs)
            ax = ax[0] / unit_factor(self.dst_xy_units)
            ay = ay[0] / unit_factor(self.dst_xy_units)
            depth_dst = depth_km * 1000.0 / unit_factor(self.dst_z_units)
            az = -depth_dst if self.dst_z_positive == "up" else depth_dst
            x0, y0, z0 = self.anchor_local
            self.dx, self.dy, self.dz = x0 - ax, y0 - ay, z0 - az

    # core

    def transform(self, x, y, z):
        """
        Transform arrays of source coordinates into the local frame.

        Parameters
        ----------
        x, y : array-like
            XY in the source CRS, in ``src_xy_units``.
        z : array-like
            Z in ``src_z_units`` following ``src_z_positive``.

        Returns
        -------
        X, Y, Z : numpy.ndarray
            Coordinates in the local frame.
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
            x0, y0, _ = self.anchor_local
            X, Y = rotate_xy(X, Y, x0, y0, self.rotation_deg)

        return X, Y, Z

    def transform_points(self, points):
        """
        Transform an ``(N, 3)`` array of source coordinates.

        Column 0 is X in ``src_xy_units``, column 1 is Y in
        ``src_xy_units``, column 2 is Z in ``src_z_units`` following
        ``src_z_positive``. XY and Z units are independent, unlike in
        :meth:`transform_mesh`.

        Parameters
        ----------
        points : array-like, shape (N, 3)

        Returns
        -------
        numpy.ndarray, shape (N, 3)
            Transformed coordinates in the local frame.
        """
        pts = np.asarray(points, dtype=float)
        X, Y, Z = self.transform(pts[:, 0], pts[:, 1], pts[:, 2])
        return np.c_[X, Y, Z]

    # catalog

    def transform_catalog(self, cat):
        """
        Project a CatalogData into the local frame.

        Parameters
        ----------
        cat : CatalogData
            Catalog in source coordinates.

        Returns
        -------
        CatalogData
            A new catalog with transformed coordinates. The original
            ``attrs`` are copied through unchanged.
        """
        X, Y, Z = self.transform(cat.x, cat.y, cat.z)
        return CatalogData(
            x=X, y=Y, z=Z,
            attrs={k: v.copy() for k, v in cat.attrs.items()},
        )

    # mesh

    def transform_mesh(self, mesh):
        """
        Project a PyVista mesh into the local frame.

        Mesh points are assumed to be ENU (Z positive up) with all three
        components in ``src_xy_units``. Cell connectivity and data arrays
        are preserved; the input mesh is not modified.

        Parameters
        ----------
        mesh : pyvista.DataSet

        Returns
        -------
        pyvista.DataSet
            A copy of the input mesh with transformed points.
        """
        if self.src_xy_units != self.src_z_units:
            raise ValueError(
                f"Mesh projection requires src_xy_units == src_z_units; "
                f"got xy={self.src_xy_units!r}, z={self.src_z_units!r}."
            )
        if self.src_z_positive != "up":
            raise ValueError(
                "Mesh projection requires src_z_positive='up' (ENU)."
            )

        out = mesh.copy()
        pts = np.asarray(out.points, dtype=float)
        X, Y, Z = self.transform(pts[:, 0], pts[:, 1], pts[:, 2])
        out.points = np.c_[X, Y, Z]
        return out