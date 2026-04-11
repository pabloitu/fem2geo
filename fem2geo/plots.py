import mplstereonet
import numpy as np

from fem2geo.utils.transform import line_enu2sphe, line_rake2sphe


MODEL_COLORS = [
    "#E63946",  # red
    "#2196F3",  # blue
    "#4CAF50",  # green
    "#FF9800",  # orange
    "#9C27B0",  # purple
    "#00BCD4",  # cyan
    "#FF5722",  # deep orange
    "#607D8B",  # blue-grey
]


def get_style(default, *overrides, drop=("show", "style"), **kw):
    """
    Merge style dicts into a matplotlib-ready kwargs dict.

    Parameters
    ----------
    default : dict
        Base style.
    *overrides : dict
        Successive override dicts (typically user config).
    drop : tuple of str
        Keys to strip from the result.
    **kw
        Final per-call overrides (e.g. ``marker="o"``).
    """
    out = dict(default)
    for o in overrides:
        out.update(o)
    out.update(kw)
    for k in drop:
        out.pop(k, None)
    return out


def stereo_field(
    ax, mesh_strikes, mesh_dips, values,
    cmap="viridis", vmin=None, vmax=None, levels=None,
    cbar=True, cbar_label=None, cbar_kwargs=None,
):
    """
    Draw a pre-computed scalar field on a stereonet as a pcolormesh.

    Parameters
    ----------
    ax : mplstereonet axes
    mesh_strikes, mesh_dips : numpy.ndarray
        Node grids from :func:`~fem2geo.utils.transform.grid_nodes`.
    values : numpy.ndarray
        Scalar values at cell centers, shape (n_dips, n_strikes).
    cmap : str
        Colormap name.
    vmin, vmax : float, optional
        Color scaling bounds.
    levels : int or sequence of float, optional
        Discrete bins. If int, creates ``levels`` evenly-spaced bins
        between ``vmin`` and ``vmax``. If a sequence, uses its values
        as bin edges directly. If None, draws a continuous colormap.
    cbar : bool
        If True, attach a colorbar to the figure. If False, only draw
        the pcolormesh and return the mappable.
    cbar_label : str, optional
        Colorbar label.
    cbar_kwargs : dict, optional
        Extra kwargs forwarded to ``fig.colorbar``.

    Returns
    -------
    mappable
        The pcolormesh mappable.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm

    norm = None
    edges = None
    if levels is not None:
        if isinstance(levels, int):
            edges = np.linspace(vmin, vmax, levels + 1)
        else:
            edges = np.asarray(levels, dtype=float)
        norm = BoundaryNorm(edges, plt.get_cmap(cmap).N)

    lon, lat = mplstereonet.pole(mesh_strikes, mesh_dips)
    if norm is not None:
        m = ax.pcolormesh(lon, lat, values, cmap=cmap, shading="auto",
                          norm=norm)
    else:
        m = ax.pcolormesh(lon, lat, values, cmap=cmap, shading="auto",
                          vmin=vmin, vmax=vmax)
    if cbar:
        defaults = {"shrink": 0.6, "pad": 0.08}
        if edges is not None:
            defaults["ticks"] = edges
        defaults.update(cbar_kwargs or {})
        ax.get_figure().colorbar(m, ax=ax, label=cbar_label or "", **defaults)
    return m


def _unpack_args(first, second, kind):
    first = np.asarray(first, dtype=float)
    if second is None:
        if first.ndim == 0:
            raise ValueError(f"{kind[1]} required for scalar {kind[0]}.")
        packed = np.atleast_2d(first)
        return packed[:, 0], packed[:, 1]
    return np.atleast_1d(first), np.atleast_1d(np.asarray(second, dtype=float))


def stereo_line(ax, plunge, azimuth=None, label=None, **kwargs):
    """
    Plot line elements on a stereonet. Accepts separate arrays or
    a packed (N, 2) array.
    """
    plunge, azimuth = _unpack_args(plunge, azimuth, ("plunge", "azimuth"))
    ax.line(plunge, azimuth, label=label, **kwargs)


def stereo_pole(ax, strike, dip=None, label=None, **kwargs):
    """
    Plot poles to planes on a stereonet. Accepts separate arrays
    or a packed (N, 2) array.
    """
    strike, dip = _unpack_args(strike, dip, ("strike", "dip"))
    ax.pole(strike, dip, label=label, **kwargs)


def stereo_plane(ax, strike, dip=None, label=None, **kwargs):
    """
    Plot great circles on a stereonet. Accepts separate arrays or
    a packed (N, 2) array.
    """
    strike, dip = _unpack_args(strike, dip, ("strike", "dip"))
    for i in range(len(strike)):
        ax.plane(strike[i], dip[i], label=label if i == 0 else None, **kwargs)


def stereo_arrow(
    ax, from_xy, to_xy,
    color="k", arrowsize=1.0, linewidth=1.0, alpha=1.0,
    label=None, **kwargs,
):
    """Plot a directed arrow on a stereonet between two projected points."""
    ax.annotate(
        "",
        xy=to_xy,
        xytext=from_xy,
        arrowprops=dict(
            arrowstyle="->,head_length={0},head_width={1}".format(
                0.4 * arrowsize, 0.25 * arrowsize,
            ),
            color=color,
            lw=linewidth,
            alpha=alpha,
        ),
        label=label,
        **kwargs,
    )


def stereo_slip_arrow(
    ax, strike, dip, signed_rake,
    color="k", arrowsize=1.0, linewidth=1.0, length=0.08, label=None,
):
    """
    Draw slip direction arrows on a stereonet for fault planes.

    Each arrow is anchored at the slip line's stereonet position and points toward
    the pole for reverse sense (rake > 0) or away for normal sense (rake < 0),
    following the Aki & Richards convention.

    Parameters
    ----------
    ax : mplstereonet axes
    strike, dip : float or array-like
        Fault plane orientation(s) in degrees.
    signed_rake : float or array-like
        Signed rake in degrees (Aki & Richards, (-180, 180]).
    color : str
    arrowsize : float
        Scales arrow head size.
    linewidth : float
    length : float
        Arrow length in projected stereonet coordinates (radians).
    label : str, optional
        Applied only to the first arrow.
    """
    strike = np.atleast_1d(np.asarray(strike, dtype=float))
    dip = np.atleast_1d(np.asarray(dip, dtype=float))
    signed_rake = np.atleast_1d(np.asarray(signed_rake, dtype=float))

    for i in range(len(strike)):
        if np.isnan(signed_rake[i]):
            continue

        plunge, azm = line_rake2sphe(strike[i], dip[i], abs(signed_rake[i]))

        sx, sy = mplstereonet.line(plunge, azm)
        sx, sy = sx[0], sy[0]
        px, py = mplstereonet.pole(strike[i], dip[i])
        px, py = px[0], py[0]

        dist = np.hypot(px - sx, py - sy)
        if dist < 1e-12:
            continue

        dx, dy = (px - sx) / dist, (py - sy) / dist
        if signed_rake[i] < 0:
            dx, dy = -dx, -dy

        stereo_arrow(ax, (sx, sy), (sx + length * dx, sy + length * dy),
                     color=color, arrowsize=arrowsize, linewidth=linewidth,
                     label=label if i == 0 else None)


def stereo_contour(
    ax, plunge, azimuth=None,
    color="k", levels=4, sigma=2, linewidth=1.0, **kwargs,
):
    """
    Plot kernel density contour lines of line elements on a stereonet.
    Accepts separate arrays or a packed (N, 2) array.
    """
    plunge, azimuth = _unpack_args(plunge, azimuth, ("plunge", "azimuth"))
    ax.density_contour(
        plunge, azimuth, measurement="lines", colors=color,
        levels=levels, sigma=sigma, linewidths=linewidth, **kwargs,
    )


def stereo_axes(ax, vecs, style, labels=None, markers=("o", "s", "v")):
    """
    Plot a 3-axis frame on a stereonet as line markers.

    Parameters
    ----------
    ax : mplstereonet axes
    vecs : numpy.ndarray
        Either (3, 3) for a single frame or (N, 3, 3) for N frames.
        Axes are stored as columns: ``vecs[..., :, i]`` is the i-th axis.
    style : dict
        Base style. Marker is overridden per axis.
    labels : tuple of str, optional
        Three labels for the legend.
    markers : tuple of str
        Markers for axes 1, 2, 3. Defaults to circle, square, triangle.
    """
    vecs = np.asarray(vecs)
    if vecs.ndim == 2:
        vecs = vecs[None, :, :]
    for i in range(3):
        p, a = line_enu2sphe(vecs[:, :, i])
        label = labels[i] if labels is not None else None
        stereo_line(ax, p, a, label=label, **{**style, "marker": markers[i]})


def stereo_axes_contour(ax, vecs, style):
    """
    Plot a 3-axis frame on a stereonet as density contours.

    Parameters
    ----------
    ax : mplstereonet axes
    vecs : numpy.ndarray
        Either (3, 3) for a single frame or (N, 3, 3) for N frames.
        Axes are columns: ``vecs[..., :, i]`` is the i-th axis.
    style : dict
        Contour style (color, levels, sigma, linewidth).
    """
    vecs = np.asarray(vecs)
    if vecs.ndim == 2:
        vecs = vecs[None, :, :]
    for i in range(3):
        p, a = line_enu2sphe(vecs[:, :, i])
        stereo_contour(ax, p, a, **style)