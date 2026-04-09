import numpy as np
from pyproj import CRS, Transformer


UNIT_M = {"m": 1.0, "km": 1000.0}


def unit_factor(unit):
    """
    Meters per unit for ``"m"`` or ``"km"``.
    """
    u = str(unit).strip().lower()
    if u not in UNIT_M:
        raise ValueError(f"unit must be 'm' or 'km', got '{unit}'.")
    return UNIT_M[u]


def flip_z(z, src_positive, dst_positive):
    """
    Convert a Z array between up/down sign conventions. Negates if the
    conventions differ, returns the input unchanged otherwise.
    """
    sp = str(src_positive).strip().lower()
    dp = str(dst_positive).strip().lower()
    for name, val in (("src_positive", sp), ("dst_positive", dp)):
        if val not in ("up", "down"):
            raise ValueError(f"{name} must be 'up' or 'down', got '{val}'.")
    z = np.asarray(z, dtype=float)
    return -z if sp != dp else z


def reproject_xy(x, y, src_crs, dst_crs):
    """
    Reproject XY arrays from one CRS to another using pyproj.

    Parameters
    ----------
    x, y : array-like
        Coordinates in the source CRS, in its native units.
    src_crs, dst_crs : str or pyproj.CRS
        Source and destination CRSs (e.g. ``"epsg:4326"``).

    Returns
    -------
    X, Y : numpy.ndarray
        Coordinates in the destination CRS, in its native units.
    """
    tfm = Transformer.from_crs(
        CRS.from_user_input(src_crs), CRS.from_user_input(dst_crs), always_xy=True
    )
    X, Y = tfm.transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    return np.asarray(X, dtype=float), np.asarray(Y, dtype=float)


def rotate_xy(x, y, x0, y0, angle_deg):
    """
    Rotate XY arrays counter-clockwise around a pivot.
    """
    t = np.deg2rad(float(angle_deg))
    c, s = np.cos(t), np.sin(t)
    dx, dy = np.asarray(x, dtype=float) - x0, np.asarray(y, dtype=float) - y0
    return x0 + c * dx - s * dy, y0 + s * dx + c * dy


def to_lonlat(x, y, src_crs, src_xy_units):
    """
    Back-project source XY to lon/lat (EPSG:4326) for bbox filtering.

    If the source CRS is already geographic, returns the inputs unchanged
    (apart from array conversion). For projected sources, scales XY to
    meters using ``src_xy_units`` before reprojecting.

    Parameters
    ----------
    x, y : array-like
        Coordinates in the source CRS.
    src_crs : str or pyproj.CRS
    src_xy_units : str
        ``"deg"`` for geographic sources, ``"m"`` or ``"km"`` for projected.

    Returns
    -------
    lon, lat : numpy.ndarray
        Coordinates in degrees, EPSG:4326.
    """
    src = CRS.from_user_input(src_crs)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if src.is_geographic:
        return x, y
    s = unit_factor(src_xy_units)
    return reproject_xy(x * s, y * s, src, "epsg:4326")


def bbox_mask(lon, lat, depth_km, lon_range=None, lat_range=None, depth_range_km=None):
    """
    Boolean mask from optional lon/lat/depth ranges. Any range left as
    ``None`` is ignored.
    """
    lon = np.asarray(lon, dtype=float)
    m = np.ones(lon.shape, dtype=bool)
    if lon_range is not None:
        m &= (lon >= lon_range[0]) & (lon <= lon_range[1])
    if lat_range is not None:
        lat = np.asarray(lat, dtype=float)
        m &= (lat >= lat_range[0]) & (lat <= lat_range[1])
    if depth_range_km is not None:
        d = np.asarray(depth_km, dtype=float)
        m &= (d >= depth_range_km[0]) & (d <= depth_range_km[1])
    return m


def bbox_to_src_bounds(bbox, crs, xy_units, z_units, z_positive):
    """
    Convert a lon/lat/depth_km bbox into source-frame axis bounds.

    Parameters
    ----------
    bbox : dict
        Mapping with optional ``lon``, ``lat``, ``depth_km`` keys, each
        a ``[min, max]`` pair.
    crs : str or pyproj.CRS
        Source CRS.
    xy_units : str
        ``"deg"`` | ``"m"`` | ``"km"``.
    z_units : str
        ``"m"`` | ``"km"``.
    z_positive : str
        ``"up"`` | ``"down"``.

    Returns
    -------
    list
        ``[xmin, xmax, ymin, ymax, zmin, zmax]`` in source units and
        sign convention. Missing axes become ``-inf`` / ``inf``.
    """
    lon_r = bbox.get("lon")
    lat_r = bbox.get("lat")
    d_r = bbox.get("depth_km")

    if lon_r is None and lat_r is None:
        x_min = y_min = -np.inf
        x_max = y_max = np.inf
    else:
        lo0, lo1 = lon_r if lon_r is not None else (-180.0, 180.0)
        la0, la1 = lat_r if lat_r is not None else (-85.0, 85.0)
        clons = np.array([lo0, lo1, lo1, lo0])
        clats = np.array([la0, la0, la1, la1])
        cx, cy = reproject_xy(clons, clats, "epsg:4326", crs)
        if str(xy_units).lower() != "deg":
            cx = cx / unit_factor(xy_units)
            cy = cy / unit_factor(xy_units)
        x_min, x_max = (
            (cx.min(), cx.max()) if lon_r is not None else (-np.inf, np.inf)
        )
        y_min, y_max = (
            (cy.min(), cy.max()) if lat_r is not None else (-np.inf, np.inf)
        )

    if d_r is None:
        z_min, z_max = -np.inf, np.inf
    else:
        z_vals = flip_z(
            np.array(d_r, dtype=float) * 1000.0, "down", z_positive,
        ) / unit_factor(z_units)
        z_min, z_max = float(z_vals.min()), float(z_vals.max())

    return [x_min, x_max, y_min, y_max, z_min, z_max]


def bbox_to_crs_bounds(bbox, crs):
    """
    Convert a lon/lat bbox into ``(left, bottom, right, top)`` in a
    target CRS's native units.

    Parameters
    ----------
    bbox : dict
        Mapping with optional ``lon`` and ``lat`` keys.
    crs : str or pyproj.CRS
        Target CRS.

    Returns
    -------
    tuple or None
        ``(left, bottom, right, top)``, or ``None`` if ``bbox`` has
        neither ``lon`` nor ``lat`` set.
    """
    lon_r = bbox.get("lon")
    lat_r = bbox.get("lat")
    if lon_r is None and lat_r is None:
        return None

    lo0, lo1 = lon_r if lon_r is not None else (-180.0, 180.0)
    la0, la1 = lat_r if lat_r is not None else (-85.0, 85.0)
    clons = np.array([lo0, lo1, lo1, lo0])
    clats = np.array([la0, la0, la1, la1])
    cx, cy = reproject_xy(clons, clats, "epsg:4326", crs)
    return float(cx.min()), float(cy.min()), float(cx.max()), float(cy.max())