import numpy as np

# Remove these if they are truly unused in this module.
# import matplotlib.pyplot as plt
# import mplstereonet as mpl


# =============================================================================
# Basis transform
# =============================================================================

def _unit(v):
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


# =============================================================================
# Line elements
# =============================================================================

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
            np.cos(azm) * np.cos(plunge),  # N
            np.sin(azm) * np.cos(plunge),  # E
            np.sin(plunge),                # D (positive down)
        ],
        dtype=float,
    )

    if ned[2] < 0:
        ned *= -1

    return ned


def line_ned2sphe(ned):
    """
    Transform a line element from cartesian NED coordinates to spherical.

    Parameters
    ----------
    ned : array-like, shape (3,)
        Line direction cosines in NED coordinates: [N, E, D]. The input may be
        non-unit; it will be normalized.

    Returns
    -------
    numpy.ndarray, shape (2,)
        Spherical line coordinates [plunge, azm] in degrees, where:
          - azm is azimuth clockwise from North in [0, 360)
          - plunge is positive downward in [0, 90]

    Notes
    -----
    Line elements are treated as axes (v and -v equivalent) and canonicalized
    to be down-directed (D >= 0).
    """
    ned = _unit(ned)

    if ned[2] < 0:
        ned = -ned

    plunge = np.rad2deg(np.arcsin(np.clip(ned[2], -1.0, 1.0)))

    # azimuth undefined for near-vertical lines; pick a deterministic value
    if np.hypot(ned[0], ned[1]) < 1e-12:
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


def line_enu2sphe(enu):
    """
    Transform a line element from cartesian ENU coordinates to spherical.

    Parameters
    ----------
    enu : array-like, shape (3,)
        Line direction cosines in ENU coordinates: [E, N, U]. The input may be
        non-unit; it will be normalized internally via conversion to NED.

    Returns
    -------
    numpy.ndarray, shape (2,)
        Spherical line coordinates [plunge, azm] in degrees (NED convention):
          - azm is azimuth clockwise from North in [0, 360)
          - plunge is positive downward in [0, 90]

    Notes
    -----
    Line elements are treated as axes (v and -v equivalent) and canonicalized
    to be down-directed (D >= 0) in the internal NED representation.
    """
    ned = enu_to_ned(enu)
    return line_ned2sphe(ned)


def line_rake2sphe(rake):
    """
    Transform a line defined by plane strike/dip and rake into spherical line coordinates.

    Parameters
    ----------
    rake : array-like, shape (3,)
        [strike, dip, rake] in degrees, where strike/dip follow the right-hand rule
        convention and rake is measured within the plane from the strike direction.

    Returns
    -------
    numpy.ndarray, shape (2,)
        Spherical line coordinates [plunge, azm] in degrees.

    Raises
    ------
    ValueError
        If rake angle is not within [0, 180].

    Notes
    -----
    Rake is restricted to [0, 180] to avoid sense ambiguity. Sense of movement, if
    needed, should be tracked separately.
    """
    rake = np.asarray(rake, dtype=float)
    if rake[2] < 0 or rake[2] > 180:
        raise ValueError("Rake angle is not within 0 and 180 deg")

    strike = np.deg2rad(rake[0])
    dip = np.deg2rad(rake[1])
    r = np.deg2rad(rake[2])

    plunge = np.rad2deg(np.arcsin(np.sin(r) * np.sin(dip)))
    azm = np.rad2deg(strike + np.arctan2(np.cos(dip) * np.sin(r), np.cos(r)))

    if plunge < 0.0:
        azm += 180.0

    azm = (azm + 360.0) % 360.0

    return np.array([np.abs(plunge), azm], dtype=float)


# =============================================================================
# Plane elements
# =============================================================================

def plane_sphe2ned(sphe):
    """
    Convert a plane (strike/dip) to the NED unit normal vector.

    Parameters
    ----------
    sphe : array-like, shape (2,)
        Plane spherical coordinates [strike, dip] in degrees.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Unit normal vector in NED coordinates: [N, E, D].

    Notes
    -----
    The normal is constructed from:
      - strike direction (horizontal line at azm=strike)
      - down-dip direction (plunge=dip at azm=strike+90)

    The result is then canonicalized to be down-directed (D >= 0).
    """
    sphe = np.asarray(sphe, dtype=float)

    v1 = line_sphe2ned([0.0, sphe[0]])
    v2 = line_sphe2ned([sphe[1], sphe[0] + 90.0])

    cr = np.cross(v1, v2)
    n = cr / np.linalg.norm(cr)

    if n[2] == 0:
        if n[1] > 0:
            n *= -1

    n *= np.sign(n[2]) + (n[2] == 0)
    return n


def plane_sphe2enu(sphe):
    """
    Convert a plane (strike/dip) to the ENU unit normal vector.

    Parameters
    ----------
    sphe : array-like, shape (2,)
        Plane spherical coordinates [strike, dip] in degrees.

    Returns
    -------
    numpy.ndarray, shape (3,)
        Unit normal vector in ENU coordinates: [E, N, U].

    Notes
    -----
    The normal is constructed from ENU versions of:
      - strike direction (horizontal line at azm=strike)
      - down-dip direction (plunge=dip at azm=strike+90)

    The result is canonicalized to be up-directed (U >= 0).
    """
    sphe = np.asarray(sphe, dtype=float)

    v1 = line_sphe2enu([0.0, sphe[0]])
    v2 = line_sphe2enu([sphe[1], sphe[0] + 90.0])

    cr = np.cross(v1, v2)
    n = cr / np.linalg.norm(cr)

    if n[2] == 0:
        if n[0] < 0:
            n *= -1

    n *= np.sign(n[2]) + (n[2] == 0)
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

    Notes
    -----
    Uses the standard relationship:
      - dip = 90 - pole_plunge
      - strike = pole_azm + 90 (mod 360)
    """
    sphe = np.asarray(sphe, dtype=float)

    strike = sphe[1] + 90.0
    strike = (strike + 360.0) % 360.0
    dip = 90.0 - sphe[0]

    return np.array([strike, dip], dtype=float)


def lineplane2rake(enu, plane, tol=5e-3):
    """
    Transform a line (ENU) contained within a plane (strike/dip) into strike/dip/rake.

    Parameters
    ----------
    enu : array-like, shape (3,)
        Line direction in ENU coordinates [E, N, U]. Treated as an axis (v and -v
        equivalent).
    plane : array-like, shape (2,)
        Plane spherical coordinates [strike, dip] in degrees.
    tol : float
        Tolerance for the scalar triple product containment test. This is applied
        after normalizing vectors to unit length.

    Returns
    -------
    numpy.ndarray, shape (3,)
        [strike, dip, rake] in degrees, with rake in [0, 180].

    Raises
    ------
    Exception
        If the line is not contained within the plane (within tolerance).
    """
    enu = _unit(enu)
    plane = np.asarray(plane, dtype=float)

    rho = _unit(line_sphe2enu([0.0, plane[0]]))
    mu = _unit(line_sphe2enu(line_rake2sphe(np.array([plane[0], plane[1], 90.0]))))
    n = np.cross(rho, mu)

    trp = np.abs(np.linalg.det(np.vstack((enu, mu, rho))))
    if trp > tol:
        raise Exception(
            "Line is not contained within the plane.\n "
            "  scalar triple prod:  %.5e" % trp
        )

    r = np.rad2deg(np.arccos(np.clip(np.dot(rho, enu), -1.0, 1.0)))

    R_hat = np.cross(rho, enu)
    R_hat = R_hat / np.linalg.norm(R_hat)

    if np.dot(R_hat, n) > 0:
        r = 180.0 - r

    return np.array([plane[0], plane[1], r], dtype=float)
