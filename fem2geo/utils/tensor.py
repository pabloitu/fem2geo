import numpy as np

from fem2geo.utils import transform as tr


__all__ = [
    "unpack_voigt6",
    "unpack_components",
    "eigenvalues",
    "eigenvectors",
    "rot_matrix",
    "rot_tensor",
    "normals_from_planes",
    "ensure_normals",
    "resolved_shear_enu",
    "slip_tendency",
    "dilation_tendency",
    "combined_tendency",
    "kostrov_tensor",
]

# Voigt ordering: [xx, yy, zz, xy, yz, zx] -> (i, j) symmetric tensor
_VOIGT_MAP = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)]

# component label -> (i, j)
_COMP_IDX = {
    "xx": (0, 0), "yy": (1, 1), "zz": (2, 2),
    "xy": (0, 1), "yz": (1, 2), "zx": (0, 2),
}


def unpack_voigt6(packed):
    """
    Unpack (N, 6) Voigt-ordered array into (N, 3, 3) symmetric tensors.

    Voigt ordering: [xx, yy, zz, xy, yz, zx].

    Parameters
    ----------
    packed : array-like, shape (N, 6)

    Returns
    -------
    numpy.ndarray, shape (N, 3, 3)
    """
    packed = np.asarray(packed, dtype=float)
    n = packed.shape[0]
    t = np.zeros((n, 3, 3))
    for col, (i, j) in enumerate(_VOIGT_MAP):
        t[:, i, j] = packed[:, col]
        t[:, j, i] = packed[:, col]
    return t


def unpack_components(arrays):
    """
    Assemble (N, 3, 3) symmetric tensors from a dict of component arrays.

    Parameters
    ----------
    arrays : dict[str, array-like]
        Maps component labels (xx, yy, zz, xy, yz, zx) to (N,) arrays.

    Returns
    -------
    numpy.ndarray, shape (N, 3, 3)
    """
    first = next(iter(arrays.values()))
    n = len(np.asarray(first))
    t = np.zeros((n, 3, 3))
    for comp, arr in arrays.items():
        i, j = _COMP_IDX[comp]
        arr = np.asarray(arr, dtype=float)
        t[:, i, j] = arr
        t[:, j, i] = arr
    return t


def eigenvalues(tensors):
    """
    Eigenvalues of an array of symmetric 3x3 tensors, sorted ascending.

    Parameters
    ----------
    tensors : array-like, shape (N, 3, 3)

    Returns
    -------
    numpy.ndarray, shape (N, 3)
        Eigenvalues per cell, column 0 smallest (most compressive).
    """
    return np.linalg.eigvalsh(np.asarray(tensors, dtype=float))


def eigenvectors(tensors):
    """
    Eigenvectors of an array of symmetric 3x3 tensors.

    Parameters
    ----------
    tensors : array-like, shape (N, 3, 3)

    Returns
    -------
    numpy.ndarray, shape (N, 3, 3)
        Eigenvectors as columns, sorted by ascending eigenvalue.
    """
    _, vecs = np.linalg.eigh(np.asarray(tensors, dtype=float))
    return vecs


def rot_matrix(angle, axis):
    """
    Create a 3D right-handed rotation matrix.

    Parameters
    ----------
    angle : float
        Rotation angle in degrees.
    axis : int
        Axis index: 1=x(E), 2=y(N), 3=z(U).

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Rotation matrix.

    Raises
    ------
    ValueError
        If axis is not one of {1, 2, 3}.
    """
    a = np.deg2rad(angle)
    c, s = np.cos(a), np.sin(a)

    if axis == 3:
        return np.array([[c, -s, 0.0],
                         [s,  c, 0.0],
                         [0.0, 0.0, 1.0]], dtype=float)
    if axis == 2:
        return np.array([[c, 0.0,  s],
                         [0.0, 1.0, 0.0],
                         [-s, 0.0,  c]], dtype=float)
    if axis == 1:
        return np.array([[1.0, 0.0, 0.0],
                         [0.0,  c, -s],
                         [0.0,  s,  c]], dtype=float)

    raise ValueError("axis must be one of {1, 2, 3}.")


def rot_tensor(tensor, angle, axis):
    """
    Rotate a 2nd-order tensor by a given axis and angle.

    Parameters
    ----------
    tensor : array-like, shape (3, 3)
        Tensor to rotate.
    angle : float
        Rotation angle in degrees.
    axis : int
        Axis index: 1=x(E), 2=y(N), 3=z(U).

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Rotated tensor.

    Raises
    ------
    ValueError
        If tensor is not shape (3, 3).
    """
    T = np.asarray(tensor, dtype=float)
    if T.shape != (3, 3):
        raise ValueError("tensor must be shape (3, 3).")

    R = rot_matrix(angle, axis)
    return R @ T @ R.T


def normals_from_planes(planes):
    """
    Compute ENU unit normals from strike/dip planes.

    Parameters
    ----------
    planes : array-like, shape (2,) or (N, 2)
        Plane(s) [strike, dip] in degrees.

    Returns
    -------
    numpy.ndarray, shape (N, 3)
        Unit normals in ENU coordinates [E, N, U].

    Raises
    ------
    ValueError
        If input has invalid shape.
    """
    planes = np.asarray(planes, dtype=float)
    if planes.ndim == 1:
        if planes.shape != (2,):
            raise ValueError("plane must be length-2 [strike, dip].")
        return tr.plane_sphe2enu(planes[0], planes[1])[None, :]
    if planes.ndim == 2:
        if planes.shape[1] != 2:
            raise ValueError("planes must have shape (N, 2).")
        return tr.plane_sphe2enu(planes[:, 0], planes[:, 1])
    raise ValueError("planes must be shape (2,) or (N,2).")


def ensure_normals(normals):
    """
    Normalize ENU normals.

    Parameters
    ----------
    normals : array-like, shape (3,) or (N, 3)
        Normal vector(s) in ENU coordinates.

    Returns
    -------
    numpy.ndarray, shape (N, 3)
        Unit normal vector(s).

    Raises
    ------
    ValueError
        If any normal is zero or shape is invalid.
    """
    normals = np.asarray(normals, dtype=float)
    if normals.ndim == 1:
        if normals.shape != (3,):
            raise ValueError("normal must be length-3.")
        n = normals[None, :]
    elif normals.ndim == 2:
        if normals.shape[1] != 3:
            raise ValueError("normals must have shape (N, 3).")
        n = normals
    else:
        raise ValueError("normals must be shape (3,) or (N,3).")

    nn = np.linalg.norm(n, axis=1)
    if np.any(nn == 0):
        raise ValueError("normal vectors must be non-zero.")
    return n / nn[:, None]


def resolved_shear_enu(sigma, plane=None, normal=None, eps=1e-12):
    """
    Resolve shear traction of a stress tensor on a plane (ENU).

    Parameters
    ----------
    sigma : array-like, shape (3, 3)
        Stress tensor in ENU coordinates.
    plane : array-like, shape (2,), optional
        [strike, dip] in degrees (right-hand rule).
    normal : array-like, shape (3,), optional
        Plane unit normal in ENU coordinates [E, N, U].
    eps : float
        Threshold for treating shear magnitude as zero.

    Returns
    -------
    tau : float
        Shear traction magnitude (always non-negative).
    tau_hat : numpy.ndarray, shape (3,)
        Directed unit shear traction vector in ENU. This is the true physical
        direction of the resolved shear stress — no sign canonicalization is
        applied, so the vector carries kinematic sense.

    Raises
    ------
    ValueError
        If sigma has invalid shape or both/neither plane and normal are provided.

    Notes
    -----
    The returned ``tau_hat`` is the direction in which the material on the
    positive-normal side of the plane is being pushed by shear stress. For
    Wallace-Bott comparison, this can be compared directly with the observed
    slip vector (e.g. from a signed rake).
    """
    S = np.asarray(sigma, dtype=float)
    if S.shape != (3, 3):
        raise ValueError("sigma must be shape (3, 3).")

    if (plane is None) == (normal is None):
        raise ValueError("Provide exactly one of plane or normal.")

    if normal is None:
        plane = np.asarray(plane, dtype=float)
        n = tr.plane_sphe2enu(plane[0], plane[1])
    else:
        n = ensure_normals(normal)[0]

    t = S @ n
    t_n = np.dot(t, n) * n
    t_s = t - t_n

    mag = float(np.linalg.norm(t_s))
    if mag < eps:
        return 0.0, np.zeros(3, dtype=float)

    return mag, t_s / mag


def slip_tendency(sigma, planes=None, normals=None, eps=1e-12):
    """
    Normalized slip tendency Ts' for one or many planes.

    Computes the ratio |tau| / |sigma_n| and normalizes by the maximum
    slip tendency achievable in the given stress state, so the result is
    always in [0, 1]. A value of 1 corresponds to the optimally oriented
    plane for slip (Morris et al., 1996).

    The maximum slip tendency is found analytically from the three 2D
    Mohr circles defined by pairs of principal stresses.

    Parameters
    ----------
    sigma : array-like, shape (3, 3)
        Stress tensor in ENU coordinates.
    planes : array-like, shape (2,) or (N, 2), optional
        Plane(s) [strike, dip] in degrees.
    normals : array-like, shape (3,) or (N, 3), optional
        Plane normal(s) in ENU coordinates.
    eps : float
        Threshold for treating sigma_n or Ts_max as zero.

    Returns
    -------
    numpy.ndarray or float
        Normalized slip tendency in [0, 1]. Scalar if input is one
        plane/normal, array otherwise.

    Raises
    ------
    ValueError
        If sigma has invalid shape or both/neither planes and normals
        are provided.

    References
    ----------
    Morris, A., Ferrill, D. A., & Henderson, D. B. (1996). Slip-tendency
    analysis and fault reactivation. *Geology*, 24(3), 275-278.
    """
    S = np.asarray(sigma, dtype=float)
    if S.shape != (3, 3):
        raise ValueError("sigma must be shape (3, 3).")

    if (planes is None) == (normals is None):
        raise ValueError("Provide exactly one of planes or normals.")

    if normals is None:
        n = normals_from_planes(planes)
        scalar = np.asarray(planes).ndim == 1
    else:
        n = ensure_normals(normals)
        scalar = np.asarray(normals).ndim == 1

    t = n @ S.T
    sigma_n = np.einsum("ij,ij->i", t, n)

    t_n = sigma_n[:, None] * n
    t_s = t - t_n
    tau = np.linalg.norm(t_s, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        raw = tau / np.abs(sigma_n)
    raw[np.abs(sigma_n) < eps] = 0.0

    # Ts_max from the three 2D Mohr circles
    ts_max = _ts_max(S, eps)
    if ts_max < eps:
        out = np.zeros_like(raw)
    else:
        out = raw / ts_max

    np.clip(out, 0.0, 1.0, out=out)
    return float(out[0]) if scalar else out


def _ts_max(sigma, eps=1e-12):
    """
    Maximum slip tendency for a stress state.

    For each pair of principal stresses (si, sj), the maximum
    tau / |sigma_n| on the corresponding Mohr circle is
    |si - sj| / |si + sj| (when si + sj != 0). Returns the
    largest value across all three pairs.
    """
    eigs = np.linalg.eigvalsh(sigma)
    ts = 0.0
    for i in range(3):
        for j in range(i + 1, 3):
            si, sj = eigs[i], eigs[j]
            denom = abs(si + sj)
            if denom > eps:
                ts = max(ts, abs(si - sj) / denom)
    return ts


def dilation_tendency(sigma, planes=None, normals=None, eps=1e-12):
    """
    Compute dilation tendency Td = (s1 - sigma_n)/(s1 - s3) for one or many planes.

    Parameters
    ----------
    sigma : array-like, shape (3, 3)
        Stress tensor in ENU coordinates.
    planes : array-like, shape (2,) or (N, 2), optional
        Plane(s) [strike, dip] in degrees.
    normals : array-like, shape (3,) or (N, 3), optional
        Plane normal(s) in ENU coordinates.
    eps : float
        Threshold for treating (s1 - s3) as zero (near-isotropic stress).

    Returns
    -------
    numpy.ndarray or float
        Dilation tendency values. Scalar if input is one plane/normal, array otherwise.

    Raises
    ------
    ValueError
        If sigma has invalid shape or both/neither planes and normals are provided.

    Notes
    -----
    Uses eigenvalues of sigma to define s1=min(eigs), s3=max(eigs). For near-isotropic
    tensors where |s1 - s3| < eps, returns NaNs.
    """
    S = np.asarray(sigma, dtype=float)
    if S.shape != (3, 3):
        raise ValueError("sigma must be shape (3, 3).")

    if (planes is None) == (normals is None):
        raise ValueError("Provide exactly one of planes or normals.")

    if normals is None:
        n = normals_from_planes(planes)
        scalar = np.asarray(planes).ndim == 1
    else:
        n = ensure_normals(normals)
        scalar = np.asarray(normals).ndim == 1

    val = np.linalg.eigvalsh(S)
    s1 = float(np.min(val))
    s3 = float(np.max(val))

    denom = (s1 - s3)
    if abs(denom) < eps:
        out = np.full((n.shape[0],), np.nan, dtype=float)
        return float(out[0]) if scalar else out

    t = n @ S.T
    sigma_n = np.einsum("ij,ij->i", t, n)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = (s1 - sigma_n) / denom

    return float(out[0]) if scalar else out


def combined_tendency(sigma, planes=None, normals=None, eps=1e-12):
    """
    Combined reactivation tendency: Ts' + Td.

    Sums the normalized slip tendency (Morris et al., 1996) and the
    dilation tendency (Ferrill et al., 1999). Both components are in
    [0, 1], so the result ranges from 0 (stable) to 2 (maximum
    reactivation potential for both slip and dilation).

    Parameters
    ----------
    sigma : array-like, shape (3, 3)
        Stress tensor in ENU coordinates.
    planes : array-like, shape (2,) or (N, 2), optional
        Plane(s) [strike, dip] in degrees.
    normals : array-like, shape (3,) or (N, 3), optional
        Plane normal(s) in ENU coordinates.
    eps : float
        Threshold for near-zero denominators.

    Returns
    -------
    numpy.ndarray or float
        Combined tendency values in [0, 2].

    References
    ----------
    Ferrill, D. A., Smart, K. J., & Morris, A. P. (2020). Fault failure
    modes, deformation mechanisms, dilation tendency, slip tendency, and
    conduits v. seals. *Geol. Soc. Lond. Spec. Publ.*, 496, 75-98.
    """
    ts = slip_tendency(sigma, planes=planes, normals=normals, eps=eps)
    td = dilation_tendency(sigma, planes=planes, normals=normals, eps=eps)
    ts = np.asarray(ts, dtype=float)
    td = np.asarray(td, dtype=float)
    out = ts + td
    scalar = out.ndim == 0
    return float(out) if scalar else out


def kostrov_tensor(strikes, dips, rakes):
    """
    Compute the Kostrov (1974) summed moment tensor from a fault population.

    Each fault contributes a symmetric dyad ``½(s⊗n + n⊗s)`` where ``s``
    is the unit slip vector (directed, from signed rake) and ``n`` is the
    unit fault plane normal. The sum over all faults gives a symmetric
    tensor whose eigenvectors are the bulk kinematic axes of the fault
    population (shortening, intermediate, extension).

    All faults are weighted equally (unit potency).

    Parameters
    ----------
    strikes : array-like, shape (N,)
        Strike in degrees (right-hand rule).
    dips : array-like, shape (N,)
        Dip in degrees.
    rakes : array-like, shape (N,)
        Signed rake in degrees (Aki & Richards, (-180, 180]).

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Symmetric Kostrov tensor in ENU coordinates.
    """
    strikes = np.asarray(strikes, dtype=float)
    dips = np.asarray(dips, dtype=float)
    rakes = np.asarray(rakes, dtype=float)

    # slip vectors: directed ENU, shape (N, 3)
    slips = tr.slip_rake2enu(strikes, dips, rakes)
    if slips.ndim == 1:
        slips = slips[None, :]

    # fault normals: ENU, shape (N, 3)
    normals = tr.plane_sphe2enu(strikes, dips)
    if normals.ndim == 1:
        normals = normals[None, :]

    # Kostrov sum: ½ Σ (s⊗n + n⊗s)
    # s⊗n has components s_i * n_j, so s⊗n + n⊗s = s_i*n_j + n_i*s_j
    K = np.einsum("ki,kj->ij", slips, normals) + np.einsum("ki,kj->ij", normals, slips)
    K *= 0.5

    return K