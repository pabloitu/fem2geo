import numpy as np



def unit(v):
    """
    Return the unit vector of ``v`` (single vector only).

    Parameters
    ----------
    v : array-like, shape (3,)

    Returns
    -------
    numpy.ndarray, shape (3,)

    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector cannot be normalized.")
    return v / n


# coordinate frame conversions

def enu2ned(v):
    """
    Convert vector(s) from ENU to NED.

    Swaps first two components and negates the third:
    ``[E, N, U] -> [N, E, D]``.

    Parameters
    ----------
    v : array-like, shape (..., 3)

    Returns
    -------
    numpy.ndarray, same shape as input
    """
    v = np.asarray(v, dtype=float)
    out = np.empty_like(v)
    out[..., 0] = v[..., 1]
    out[..., 1] = v[..., 0]
    out[..., 2] = -v[..., 2]
    return out


def ned2enu(v):
    """
    Convert vector(s) from NED to ENU.

    Same operation as :func:`enu2ned` (inverse mapping).

    Parameters
    ----------
    v : array-like, shape (..., 3)

    Returns
    -------
    numpy.ndarray, same shape as input
    """
    return enu2ned(v)


# line (axis) conversions — canonicalized to down-directed (D >= 0)

def line_sphe2ned(plunge, azimuth):
    """
    Spherical line coordinates to NED unit axis.

    Parameters
    ----------
    plunge : float or array-like
        Plunge in degrees (>= 0).
    azimuth : float or array-like
        Azimuth in degrees, clockwise from North.

    Returns
    -------
    numpy.ndarray, shape (3,) or (N, 3)
        NED direction cosines, canonicalized D >= 0.
    """
    plunge = np.asarray(plunge, dtype=float)
    azimuth = np.asarray(azimuth, dtype=float)
    scalar = plunge.ndim == 0 and azimuth.ndim == 0

    p = np.deg2rad(plunge)
    a = np.deg2rad(azimuth)

    cp = np.cos(p)
    ned = np.stack([np.cos(a) * cp, np.sin(a) * cp, np.sin(p)], axis=-1)

    # canonicalize: flip where D < 0
    mask = ned[..., 2] < 0
    if np.any(mask):
        ned[mask] *= -1

    return ned.squeeze() if scalar else ned


def line_ned2sphe(ned):
    """
    NED unit axis to spherical line coordinates.

    Parameters
    ----------
    ned : array-like, shape (3,) or (N, 3)

    Returns
    -------
    plunge : float or numpy.ndarray, shape (N,)
        Plunge in degrees [0, 90].
    azimuth : float or numpy.ndarray, shape (N,)
        Azimuth in degrees [0, 360). Returns 0 for vertical lines.
    """
    ned = np.asarray(ned, dtype=float)
    scalar = ned.ndim == 1

    if scalar:
        ned = ned[None, :]

    # normalize
    norms = np.linalg.norm(ned, axis=1, keepdims=True)
    ned = ned / np.where(norms < 1e-12, 1.0, norms)

    # canonicalize D >= 0
    mask = ned[:, 2] < 0
    ned[mask] *= -1

    plunge = np.rad2deg(np.arcsin(np.clip(ned[:, 2], -1.0, 1.0)))
    azimuth = np.rad2deg(np.arctan2(ned[:, 1], ned[:, 0])) % 360.0

    # near-vertical: azimuth undefined, return 0
    horiz = np.hypot(ned[:, 0], ned[:, 1])
    azimuth = np.where(horiz < 1e-12, 0.0, azimuth)

    if scalar:
        return float(plunge[0]), float(azimuth[0])
    return plunge, azimuth


def line_sphe2enu(plunge, azimuth):
    """
    Spherical line coordinates to ENU unit axis.

    Parameters
    ----------
    plunge : float or array-like
    azimuth : float or array-like

    Returns
    -------
    numpy.ndarray, shape (3,) or (N, 3)
    """
    return ned2enu(line_sphe2ned(plunge, azimuth))


def line_enu2sphe(enu):
    """
    ENU unit axis to spherical line coordinates.

    Parameters
    ----------
    enu : array-like, shape (3,) or (N, 3)

    Returns
    -------
    plunge : float or numpy.ndarray, shape (N,)
    azimuth : float or numpy.ndarray, shape (N,)
    """
    return line_ned2sphe(enu2ned(enu))


# plane basis

def plane_basis_enu(strike, dip):
    """
    Orthonormal basis vectors of a fault plane in ENU.

    Returns the strike direction, the dip vector (steepest descent line
    in the plane), and the plane normal_vec. All three are mutually
    orthogonal unit vectors forming a right-handed basis.

    The dip vector plunges at the dip angle toward ``strike + 90°``. The normal_vec
    points away from the dipping side (upward for non-vertical planes).

    Parameters
    ----------
    strike : float or array-like
    dip : float or array-like

    Returns
    -------
    strike_vec : numpy.ndarray, shape (3,) or (N, 3)
        Horizontal unit vector along the strike azimuth.
    dip_vec : numpy.ndarray, shape (3,) or (N, 3)
        Unit vector plunging at the dip angle toward the dip direction.
    normal_vec : numpy.ndarray, shape (3,) or (N, 3)
        Unit normal_vec to the plane (= strike_vec × dip_vec).
    """
    strike = np.asarray(strike, dtype=float)
    dip = np.asarray(dip, dtype=float)
    scalar = strike.ndim == 0 and dip.ndim == 0

    s = np.deg2rad(strike)
    d = np.deg2rad(dip)
    da = s + np.pi / 2.0

    strike_vec = np.stack([np.sin(s), np.cos(s), np.zeros_like(s)], axis=-1)
    dip_vec = np.stack([
        np.sin(da) * np.cos(d),
        np.cos(da) * np.cos(d),
        -np.sin(d),
    ], axis=-1)
    normal_vec = np.cross(strike_vec, dip_vec)

    if scalar:
        return strike_vec.squeeze(), dip_vec.squeeze(), normal_vec.squeeze()
    return strike_vec, dip_vec, normal_vec


# line-rake conversions (axis, unsigned rake [0, 180])

def line_rake2sphe(strike, dip, rake):
    """
    Unsigned rake on a plane to spherical line coordinates.

    Parameters
    ----------
    strike : float or array-like
    dip : float or array-like
    rake : float or array-like
        Unsigned rake in degrees [0, 180].

    Returns
    -------
    plunge : float or numpy.ndarray, shape (N,)
    azimuth : float or numpy.ndarray, shape (N,)
    """
    strike = np.asarray(strike, dtype=float)
    dip = np.asarray(dip, dtype=float)
    rake = np.asarray(rake, dtype=float)
    scalar = strike.ndim == 0 and dip.ndim == 0 and rake.ndim == 0

    s = np.deg2rad(strike)
    d = np.deg2rad(dip)
    r = np.deg2rad(rake)

    plunge = np.rad2deg(np.arcsin(np.clip(np.sin(r) * np.sin(d), -1.0, 1.0)))
    azimuth = np.rad2deg(s + np.arctan2(np.cos(d) * np.sin(r), np.cos(r)))

    # canonicalize: if plunge < 0, flip azimuth and take abs(plunge)
    flip = plunge < 0
    azimuth = np.where(flip, azimuth + 180.0, azimuth)
    plunge = np.abs(plunge)
    azimuth = azimuth % 360.0

    if scalar:
        return float(plunge), float(azimuth)
    return plunge, azimuth


def line_enu2rake(enu, strike, dip, check=False, tol=5e-3):
    """
    ENU axis on a plane to unsigned rake.

    Projects the vector onto the plane's strike and updip directions
    and returns the angle in [0, 180]. Since lines are axes (v and -v
    equivalent), the result is the same for both orientations.

    Parameters
    ----------
    enu : array-like, shape (3,) or (N, 3)
    strike : float or array-like
    dip : float or array-like
    check : bool
        If True, verify the line lies in the plane (scalar input only).
    tol : float
        Tolerance for the containment check.

    Returns
    -------
    rake : float or numpy.ndarray
        Unsigned rake in degrees [0, 180].
    """
    enu = np.asarray(enu, dtype=float)
    scalar = enu.ndim == 1
    if scalar:
        enu = enu[None, :]

    norms = np.linalg.norm(enu, axis=1, keepdims=True)
    enu = enu / np.where(norms < 1e-12, 1.0, norms)

    strike_dir, dip_vec, normal = plane_basis_enu(strike, dip)
    strike_dir = np.atleast_2d(strike_dir)
    dip_vec = np.atleast_2d(dip_vec)
    normal = np.atleast_2d(normal)

    if check and scalar:
        proj = np.abs(np.einsum("ij,ij->i", enu[:1], normal[:1])[0])
        if proj > tol:
            raise ValueError(
                f"Line is not contained in the plane "
                f"(normal projection = {proj:.5e}, tol = {tol})."
            )

    cs = np.einsum("ij,ij->i", enu, strike_dir)
    cu = np.einsum("ij,ij->i", enu, -dip_vec)
    rake = np.rad2deg(np.arctan2(cu, cs))

    rake = rake % 360.0
    rake = np.where(rake > 180.0, 360.0 - rake, rake)

    if scalar:
        return float(rake[0])
    return rake


# slip (directed) conversions — signed rake, Aki & Richards (-180, 180]

def slip_rake2enu(strike, dip, rake):
    """
    Signed rake to directed slip vector in ENU.

    Uses the Aki & Richards convention:

    - rake > 0: reverse/thrust (hanging wall up).
    - rake < 0: normal (hanging wall down).
    - rake = 0: pure left-lateral.
    - rake = +/-180: pure right-lateral.

    Parameters
    ----------
    strike : float or array-like
    dip : float or array-like
    rake : float or array-like
        Signed rake in degrees (-180, 180].

    Returns
    -------
    numpy.ndarray, shape (3,) or (N, 3)
        Directed unit slip vector in ENU.
    """
    strike = np.asarray(strike, dtype=float)
    dip = np.asarray(dip, dtype=float)
    rake = np.asarray(rake, dtype=float)
    scalar = strike.ndim == 0 and dip.ndim == 0 and rake.ndim == 0

    r = np.deg2rad(rake)
    strike_dir, dip_vec, _ = plane_basis_enu(strike, dip)

    slip = np.cos(r)[..., None] * strike_dir + np.sin(r)[..., None] * (-dip_vec)

    norms = np.linalg.norm(slip, axis=-1, keepdims=True)
    slip = slip / np.where(norms < 1e-12, 1.0, norms)

    return slip.squeeze() if scalar else slip


def slip_enu2rake(enu, strike, dip):
    """
    Directed ENU vector to signed rake (Aki & Richards).

    Projects the vector onto the fault plane's strike and updip directions
    and returns rake in degrees.

    Parameters
    ----------
    enu : array-like, shape (3,) or (N, 3)
    strike : float or array-like
    dip : float or array-like

    Returns
    -------
    rake : float or numpy.ndarray
        Signed rake in degrees (-180, 180].
    """
    enu = np.asarray(enu, dtype=float)
    scalar = enu.ndim == 1
    if scalar:
        enu = enu[None, :]

    strike_dir, dip_vec, _ = plane_basis_enu(strike, dip)
    strike_dir = np.atleast_2d(strike_dir)
    dip_vec = np.atleast_2d(dip_vec)

    cs = np.einsum("ij,ij->i", enu, strike_dir)
    cu = np.einsum("ij,ij->i", enu, -dip_vec)
    rake = np.rad2deg(np.arctan2(cu, cs))

    if scalar:
        return float(rake[0])
    return rake


# plane normal conversions

def plane_sphe2enu(strike, dip):
    """
    Plane (strike/dip) to ENU unit normal.

    Computes the normal as the cross product of the strike direction (horizontal,
    azimuth = strike) and the dip vector (plunging with dip angle toward strike + 90°).

    Parameters
    ----------
    strike : float or array-like
    dip : float or array-like

    Returns
    -------
    numpy.ndarray, shape (3,) or (N, 3)
    """
    strike = np.asarray(strike, dtype=float)
    dip = np.asarray(dip, dtype=float)
    scalar = strike.ndim == 0 and dip.ndim == 0

    # v1: strike direction in ENU (plunge=0, azimuth=strike)
    v1 = line_sphe2enu(np.zeros_like(strike), strike)

    # v2: dip vector in ENU (plunge=dip, azimuth=strike+90)
    v2 = line_sphe2enu(dip, strike + 90.0)

    n = np.cross(v1, v2)
    norms = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.where(norms < 1e-12, 1.0, norms)

    # canonicalize Up >= 0 (handle vertical planes where Up ~ 0)
    if n.ndim == 1:
        if abs(n[2]) < 1e-12 and n[0] < 0:
            n *= -1
        elif n[2] < 0:
            n *= -1
    else:
        vert = np.abs(n[..., 2]) < 1e-12
        flip_vert = vert & (n[..., 0] < 0)
        flip_down = ~vert & (n[..., 2] < 0)
        n[flip_vert] *= -1
        n[flip_down] *= -1

    return n.squeeze() if scalar else n


def plane_sphe2ned(strike, dip):
    """
    Plane (strike/dip) to NED unit normal, canonicalized Down >= 0.

    Computes the normal as the cross product of the strike direction
    and the dip vector, both in NED coordinates.

    Parameters
    ----------
    strike : float or array-like
    dip : float or array-like

    Returns
    -------
    numpy.ndarray, shape (3,) or (N, 3)
    """
    strike = np.asarray(strike, dtype=float)
    dip = np.asarray(dip, dtype=float)
    scalar = strike.ndim == 0 and dip.ndim == 0

    # v1: strike direction in NED (plunge=0, azimuth=strike)
    v1 = line_sphe2ned(np.zeros_like(strike), strike)

    # v2: dip vector in NED (plunge=dip, azimuth=strike+90)
    v2 = line_sphe2ned(dip, strike + 90.0)

    n = np.cross(v1, v2)
    norms = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.where(norms < 1e-12, 1.0, norms)

    # canonicalize D >= 0 (handle horizontal normals)
    if n.ndim == 1:
        if abs(n[2]) < 1e-12 and n[1] > 0:
            n *= -1
        elif n[2] < 0:
            n *= -1
    else:
        horiz = np.abs(n[..., 2]) < 1e-12
        flip_horiz = horiz & (n[..., 1] > 0)
        flip_up = ~horiz & (n[..., 2] < 0)
        n[flip_horiz] *= -1
        n[flip_up] *= -1

    return n.squeeze() if scalar else n


def plane_pole2sphe(plunge, azimuth):
    """
    Plane pole (plunge/azimuth) to strike/dip.

    Parameters
    ----------
    plunge : float or array-like
    azimuth : float or array-like

    Returns
    -------
    strike : float or numpy.ndarray
    dip : float or numpy.ndarray
    """
    plunge = np.asarray(plunge, dtype=float)
    azimuth = np.asarray(azimuth, dtype=float)
    scalar = plunge.ndim == 0 and azimuth.ndim == 0

    strike = (azimuth + 90.0) % 360.0
    dip = 90.0 - plunge

    if scalar:
        return float(strike), float(dip)
    return strike, dip


# stereonet grids

def grid_nodes(n_strikes, n_dips):
    """
    Create node grids for stereonet discretization.

    Parameters
    ----------
    n_strikes : int
        Number of strike bins. Nodes: n_strikes + 1 columns.
    n_dips : int
        Number of dip bins. Nodes: n_dips + 1 rows.

    Returns
    -------
    mesh_strikes, mesh_dips : numpy.ndarray
        Meshgrids of strike and dip nodes (degrees).
    """
    strikes = np.linspace(0.0, 360.0, n_strikes + 1, endpoint=True)
    dips = np.linspace(0.0, 90.0, n_dips + 1, endpoint=True)
    return np.meshgrid(strikes, dips)


def grid_centers(mesh_strikes, mesh_dips):
    """
    Compute cell-center strike/dip arrays from node grids.

    Parameters
    ----------
    mesh_strikes, mesh_dips : numpy.ndarray
        Node grids as returned by :func:`grid_nodes`.

    Returns
    -------
    strikes_c, dips_c : numpy.ndarray
        Cell-center strike and dip arrays.
    """
    s = (mesh_strikes[:-1, :-1] + mesh_strikes[:-1, 1:]) / 2.0
    d = (mesh_dips[:-1, :-1] + mesh_dips[1:, :-1]) / 2.0
    return s, d