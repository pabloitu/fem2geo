from dataclasses import dataclass, asdict

import mplstereonet
import numpy as np

from fem2geo.utils.transform import line_rake2sphe


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


@dataclass
class PlotConfig:
    """
    Bag of matplotlib style parameters. Fields left as None are omitted
    when passed to matplotlib, so the library defaults apply.

    Use :meth:`update` to override fields from a config dict or kwargs,
    and :meth:`kwargs` to get a clean dict for matplotlib calls.
    """
    color:           str   = None
    marker:          str   = None
    markersize:      float = None
    markeredgecolor: str   = None
    alpha:           float = None
    linewidth:       float = None
    levels:          int   = None
    sigma:           float = None

    def update(self, overrides=None, **kw):
        """Return a new PlotConfig with overrides applied."""
        if isinstance(overrides, dict):
            kw = {**overrides, **kw}
        valid = {
            k: v for k, v in kw.items()
            if k in self.__dataclass_fields__ and v is not None
        }
        return PlotConfig(**{**asdict(self), **valid})

    def kwargs(self):
        """Return non-None fields as a dict for matplotlib."""
        return {k: v for k, v in asdict(self).items() if v is not None}


def stereo_field(
    ax, mesh_strikes, mesh_dips, values,
    cmap="RdYlBu_r", vmin=None, vmax=None,
    cbar_label=None, cbar_kwargs=None,
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
    cbar_label : str, optional
        Colorbar label.
    cbar_kwargs : dict, optional
        Extra kwargs forwarded to ``fig.colorbar``.

    Returns
    -------
    mappable
        The pcolormesh mappable.
    """
    lon, lat = mplstereonet.pole(mesh_strikes, mesh_dips)
    m = ax.pcolormesh(lon, lat, values, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    defaults = {"shrink": 0.6, "pad": 0.08}
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
    a packed (N, 2) array for backward compatibility.
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
    for n in range(len(strike)):
        ax.plane(strike[n], dip[n], label=label if n == 0 else None, **kwargs)


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
    Draw a slip direction arrow on a stereonet for a fault plane.

    The arrow is anchored at the slip line's stereonet position and points
    toward the pole for reverse sense (rake > 0) or away for normal sense
    (rake < 0), following the Aki & Richards convention.

    Parameters
    ----------
    ax : mplstereonet axes
    strike, dip : float
        Fault plane orientation in degrees.
    signed_rake : float
        Signed rake in degrees (Aki & Richards convention, (-180, 180]).
    color : str
    arrowsize : float
        Scales arrow head size.
    linewidth : float
    length : float
        Arrow length in projected stereonet coordinates (radians).
    label : str, optional
    """
    plunge, azm = line_rake2sphe(strike, dip, abs(signed_rake))

    sx, sy = mplstereonet.line(plunge, azm)
    sx, sy = sx[0], sy[0]
    px, py = mplstereonet.pole(strike, dip)
    px, py = px[0], py[0]

    dist = np.hypot(px - sx, py - sy)
    if dist < 1e-12:
        return

    dx, dy = (px - sx) / dist, (py - sy) / dist
    if signed_rake < 0:
        dx, dy = -dx, -dy

    stereo_arrow(ax, (sx, sy), (sx + length * dx, sy + length * dy),
                 color=color, arrowsize=arrowsize, linewidth=linewidth, label=label)


def stereo_contour(
    ax, plunge, azimuth=None, label=None,
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