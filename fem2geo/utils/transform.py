import numpy as np


def unit(v):
    """
    Return the unit vector of ``v``.

    Parameters
    ----------
    v : array-like, shape (3,)
        Vector to normalize.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Unit-length vector.

    Raises
    ------
    ValueError
        If the vector has zero length.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector cannot be normalized.")
    return v / n


def enu_to_ned(v_enu):
    """
    Convert a vector from ENU (E, N, U) to NED (N, E, D).

    Parameters
    ----------
    v_enu : array-like, shape (3,)
        Vector components in ENU coordinates: [E, N, U].

    Returns
    -------
    numpy.ndarray, shape (3,)
        Vector components in NED coordinates: [N, E, D].

    Notes
    -----
    Coordinate definitions:
      - ENU: x=East, y=North, z=Up
      - NED: x=North, y=East, z=Down

    Mapping:
      - N = ENU_y
      - E = ENU_x
      - D = -ENU_z
    """
    v_enu = np.asarray(v_enu, dtype=float)
    return np.array([v_enu[1], v_enu[0], -v_enu[2]], dtype=float)


def ned_to_enu(v_ned):
    """
    Convert a vector from NED (N, E, D) to ENU (E, N, U).

    Parameters
    ----------
    v_ned : array-like, shape (3,)
        Vector components in NED coordinates: [N, E, D].

    Returns
    -------
    numpy.ndarray, shape (3,)
        Vector components in ENU coordinates: [E, N, U].

    Notes
    -----
    Mapping:
      - E = NED_y
      - N = NED_x
      - U = -NED_z
    """
    v_ned = np.asarray(v_ned, dtype=float)
    return np.array([v_ned[1], v_ned[0], -v_ned[2]], dtype=float)


def line_sphe2ned(sphe):
    """
    Transform a line element from spherical coordinates to cartesian NED.

    Parameters
    ----------
    sphe : array-like, shape (2,)
        Spherical line coordinates [plunge, azm] in degrees, where:
          - azm is azimuth clockwise from North.
          - plunge is positive downward from horizontal.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Unit line direction cosines in NED coordinates: [N, E, D].

    Notes
    -----
    Line elements are treated as axes (v and -v equivalent) and canonicalized
    to be down-directed (D >= 0).
    """
    sphe = np.asarray(sphe, dtype=float)
    plunge = np.deg2rad(sphe[0])
    azm = np.deg2rad(sphe[1])

    ned = np.array(
        [
            np.cos(azm) * np.cos(plunge),
            np.sin(azm) * np.cos(plunge),
            np.sin(plunge),
        ],
        dtype=float,
    )

    if ned[2] < 0:
        ned *= -1

    return ned


def line_ned2sphe(ned, eps=1e-12):
    """
    Transform a line element from cartesian NED coordinates to spherical.

    Parameters
    ----------
    ned : array-like, shape (3,)
        Line direction cosines in NED coordinates: [N, E, D]. The input may be
        non-unit; it will be normalized.
    eps : float
        Threshold to treat a line as near-vertical when computing azimuth.

    Returns
    -------
    numpy.ndarray, shape (2,)
        Spherical line coordinates [plunge, azm] in degrees, where:
          - azm is azimuth clockwise from North in [0, 360)
          - plunge is positive downward in [0, 90]

    Notes
    -----
    Line elements are treated as axes (v and -v equivalent) and canonicalized
    to be down-directed (D >= 0). Azimuth is undefined for vertical lines; a
    deterministic value (0) is returned when the horizontal component is below
    ``eps``.
    """
    ned = unit(ned)

    if ned[2] < 0:
        ned = -ned

    plunge = np.rad2deg(np.arcsin(np.clip(ned[2], -1.0, 1.0)))

    if np.hypot(ned[0], ned[1]) < eps:
        azm = 0.0
    else:
        azm = (np.rad2deg(np.arctan2(ned[1], ned[0])) + 360.0) % 360.0

    return np.array([plunge, azm], dtype=float)


def line_sphe2enu(sphe):
    """
    Transform a line element from spherical coordinates to cartesian ENU.

    Parameters
    ----------
    sphe : array-like, shape (2,)
        Spherical line coordinates [plunge, azm] in degrees, where:
          - azm is azimuth clockwise from North.
          - plunge is positive downward from horizontal.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Unit line direction cosines in ENU coordinates: [E, N, U].

    Notes
    -----
    This uses the NED spherical convention (plunge positive downward) and then
    converts NED -> ENU. Line elements are treated as axes and canonicalized
    to be down-directed in NED (D >= 0).
    """
    ned = line_sphe2ned(sphe)
    return ned_to_enu(ned)


def line_enu2sphe(enu, eps=1e-12):
    """
    Transform a line element from cartesian ENU coordinates to spherical.

    Parameters
    ----------
    enu : array-like, shape (3,)
        Line direction cosines in ENU coordinates: [E, N, U]. The input may be
        non-unit; it will be normalized internally via conversion to NED.
    eps : float
        Threshold to treat a line as near-vertical when computing azimuth.

    Returns
    -------
    numpy.ndarray, shape (2,)
        Spherical line coordinates [plunge, azm] in degrees (NED convention):
          - azm is azimuth clockwise from North in [0, 360)
          - plunge is positive downward in [0, 90]
    """
    ned = enu_to_ned(enu)
    return line_ned2sphe(ned, eps=eps)


def line_rake2sphe(rake):
    """
    Transform a line defined by plane strike/dip and rake into spherical line coordinates.

    Parameters
    ----------
    rake : array-like, shape (3,)
        [strike, dip, rake] in degrees. Strike/dip follow the right-hand rule
        convention. Rake is measured within the plane from the strike direction.

    Returns
    -------
    numpy.ndarray, shape (2,)
        [plunge, azm] in degrees.

    Raises
    ------
    ValueError
        If input shape is invalid or rake is not within [0, 180].
    """
    rake = np.asarray(rake, dtype=float)
    if rake.shape != (3,):
        raise ValueError("rake must be a length-3 array: [strike, dip, rake].")
    if rake[2] < 0 or rake[2] > 180:
        raise ValueError("Rake angle is not within 0 and 180 deg")

    strike = np.deg2rad(rake[0])
    dip = np.deg2rad(rake[1])
    r = np.deg2rad(rake[2])

    plunge = np.rad2deg(np.arcsin(np.clip(np.sin(r) * np.sin(dip), -1.0, 1.0)))
    azm = np.rad2deg(strike + np.arctan2(np.cos(dip) * np.sin(r), np.cos(r)))

    if plunge < 0.0:
        azm += 180.0

    azm = (azm + 360.0) % 360.0

    return np.array([abs(plunge), azm], dtype=float)


def plane_sphe2ned(sphe, eps=1e-12):
    """
    Convert a plane (strike/dip) to the NED unit normal vector.

    Parameters
    ----------
    sphe : array-like, shape (2,)
        Plane spherical coordinates [strike, dip] in degrees.
    eps : float
        Tolerance used to detect degenerate cases and near-horizontal normals.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Unit normal vector in NED coordinates: [N, E, D].

    Notes
    -----
    The normal is canonicalized to be down-directed (D >= 0). When D is near zero,
    a deterministic sign is chosen.
    """
    sphe = np.asarray(sphe, dtype=float)
    if sphe.shape != (2,):
        raise ValueError("sphe must be a length-2 array: [strike, dip].")

    v1 = line_sphe2ned([0.0, sphe[0]])
    v2 = line_sphe2ned([sphe[1], sphe[0] + 90.0])

    cr = np.cross(v1, v2)
    nrm = np.linalg.norm(cr)
    if nrm < eps:
        raise ValueError("Degenerate plane definition (normal is undefined).")

    n = cr / nrm

    if abs(n[2]) < eps and n[1] > 0:
        n *= -1.0

    if n[2] < 0:
        n *= -1.0

    return n


def plane_sphe2enu(sphe, eps=1e-12):
    """
    Convert a plane (strike/dip) to the ENU unit normal vector.

    Parameters
    ----------
    sphe : array-like, shape (2,)
        Plane spherical coordinates [strike, dip] in degrees.
    eps : float
        Tolerance used to detect degenerate cases and near-horizontal normals.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Unit normal vector in ENU coordinates: [E, N, U].

    Notes
    -----
    The normal is canonicalized to be up-directed (U >= 0). When U is near zero,
    a deterministic sign is chosen.
    """
    sphe = np.asarray(sphe, dtype=float)
    if sphe.shape != (2,):
        raise ValueError("sphe must be a length-2 array: [strike, dip].")

    v1 = line_sphe2enu([0.0, sphe[0]])
    v2 = line_sphe2enu([sphe[1], sphe[0] + 90.0])

    cr = np.cross(v1, v2)
    nrm = np.linalg.norm(cr)
    if nrm < eps:
        raise ValueError("Degenerate plane definition (normal is undefined).")

    n = cr / nrm

    if abs(n[2]) < eps and n[0] < 0:
        n *= -1.0

    if n[2] < 0:
        n *= -1.0

    return n


def plane_pole2sphe(sphe):
    """
    Convert a plane pole (line spherical coordinates) to plane strike/dip.

    Parameters
    ----------
    sphe : array-like, shape (2,)
        Pole spherical coordinates [plunge, azm] in degrees.

    Returns
    -------
    numpy.ndarray, shape (2,)
        Plane spherical coordinates [strike, dip] in degrees.
    """
    sphe = np.asarray(sphe, dtype=float)
    if sphe.shape != (2,):
        raise ValueError("sphe must be a length-2 array: [plunge, azm].")

    strike = (sphe[1] + 90.0) % 360.0
    dip = 90.0 - sphe[0]

    return np.array([strike, dip], dtype=float)


def lineplane2rake(enu, plane, tol=5e-3, eps=1e-12):
    """
    Transform a line (ENU) contained within a plane (strike/dip) into strike/dip/rake.

    Parameters
    ----------
    enu : array-like, shape (3,)
        Line direction in ENU coordinates [E, N, U]. Treated as an axis.
    plane : array-like, shape (2,)
        Plane spherical coordinates [strike, dip] in degrees.
    tol : float
        Tolerance for the scalar triple product containment test (unit vectors).
    eps : float
        Threshold for near-parallel/degenerate checks.

    Returns
    -------
    numpy.ndarray, shape (3,)
        [strike, dip, rake] in degrees, with rake in [0, 180].

    Raises
    ------
    ValueError
        If input shapes are invalid or the plane basis is degenerate.
    Exception
        If the line is not contained within the plane (within tolerance).
    """
    plane = np.asarray(plane, dtype=float)
    if plane.shape != (2,):
        raise ValueError("plane must be a length-2 array: [strike, dip].")

    enu = unit(enu)

    rho = unit(line_sphe2enu([0.0, plane[0]]))
    mu = unit(line_sphe2enu(line_rake2sphe(np.array([plane[0], plane[1], 90.0]))))

    n = np.cross(rho, mu)
    nn = np.linalg.norm(n)
    if nn < eps:
        raise ValueError("Degenerate plane basis (rho and mu nearly parallel).")
    n = n / nn

    trp = abs(np.linalg.det(np.vstack((enu, mu, rho))))
    if trp > tol:
        raise Exception(
            "Line is not contained within the plane.\n "
            "  scalar triple prod:  %.5e" % trp
        )

    c = float(np.clip(np.dot(rho, enu), -1.0, 1.0))
    r = float(np.rad2deg(np.arccos(c)))

    R_hat = np.cross(rho, enu)
    Rn = np.linalg.norm(R_hat)
    if Rn < eps:
        r = 0.0 if c >= 0.0 else 180.0
        return np.array([plane[0], plane[1], r], dtype=float)

    R_hat = R_hat / Rn
    if np.dot(R_hat, n) > 0.0:
        r = 180.0 - r

    return np.array([plane[0], plane[1], r], dtype=float)