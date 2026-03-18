from dataclasses import dataclass, asdict

import mplstereonet as mpl
import numpy as np

from fem2geo.utils.tensor import grid_nodes, grid_centers, slip_tendency, dilation_tendency

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

_TENDENCY_FNS = {"slip": slip_tendency, "dilation": dilation_tendency}
_CBAR_LABELS = {
    "slip": r"Slip tendency $|\tau|/|\sigma_n|$",
    "dilation": r"Dilation tendency $(\sigma_1-\sigma_n)/(\sigma_1-\sigma_3)$",
}
_MARKERS = {"s1": "o", "s2": "s", "s3": "v"}


@dataclass
class PlotConfig:
    """
    Style for a plotted direction set (σ1, σ2, or σ3).

    All fields map directly to matplotlib kwargs. Marker shape is set
    per-axis by the job and should not be overridden in the config.

    Parameters
    ----------
    color : str
    markersize : float
    markeredgecolor : str
    alpha : float
    linewidth : float
        Contour line width (contour style only).
    levels : int
        Number of density contour lines (contour style only).
    sigma : float
        Maximum sigma level for density contours (contour style only).
    """
    color: str = "k"
    markersize: float = 8
    markeredgecolor: str = "k"
    alpha: float = 1.0
    linewidth: float = 1.0
    levels: int = 4
    sigma: float = 2.0

    @classmethod
    def avg(cls, color: str = "k") -> "PlotConfig":
        return cls(color=color, markersize=8, markeredgecolor="k", alpha=1.0)

    @classmethod
    def cell(cls, color: str = "k") -> "PlotConfig":
        return cls(color=color, markersize=3, markeredgecolor="none", alpha=0.4)

    @classmethod
    def density(cls, color: str = "k") -> "PlotConfig":
        return cls(color=color, levels=4, sigma=2.0, linewidth=1.0)

    @classmethod
    def from_cfg(cls, base: "PlotConfig", cfg: dict) -> "PlotConfig":
        """Override fields from a config dict, ignoring non-PlotConfig keys."""
        _skip = {"show", "style"}
        overrides = {k: v for k, v in cfg.items()
                     if k not in _skip and k in cls.__dataclass_fields__}
        return cls(**{**asdict(base), **overrides})

    def update(self, cfg: dict) -> "PlotConfig":
        """Return a new PlotConfig with fields from cfg applied, skipping show/style."""
        return PlotConfig.from_cfg(self, cfg)

    def scatter_kwargs(self, marker: str) -> dict:
        return {"color": self.color, "marker": marker,
                "markersize": self.markersize,
                "markeredgecolor": self.markeredgecolor,
                "alpha": self.alpha}

    def contour_kwargs(self) -> dict:
        return {"color": self.color, "levels": self.levels,
                "sigma": self.sigma, "linewidth": self.linewidth,
                "alpha": self.alpha}


def stereo_field(ax, sigma, kind="dilation", n_strikes=180, n_dips=45,
                 cmap="jet", vmin=None, vmax=None, cbar_label=None, cbar_kwargs=None):
    """
    Compute and draw a scalar tendency field on an existing stereonet axes.

    Parameters
    ----------
    ax : mplstereonet axes
    sigma : array-like, shape (3, 3)
        Stress tensor in ENU coordinates.
    kind : str
        ``slip`` or ``dilation``.
    n_strikes, n_dips : int
        Stereonet discretization (number of cells).
    cmap : str
        Colormap name.
    vmin, vmax : float, optional
        Color scaling.
    cbar_label : str, optional
        Overrides the default colorbar label.
    cbar_kwargs : dict, optional
        Extra kwargs forwarded to ``fig.colorbar``.

    Returns
    -------
    mappable
        The pcolormesh mappable.
    """
    if kind not in _TENDENCY_FNS:
        raise ValueError(f"kind must be 'slip' or 'dilation', got '{kind}'.")

    mesh_s, mesh_d = grid_nodes(n_strikes, n_dips)
    cs, cd = grid_centers(mesh_s, mesh_d)
    planes = np.column_stack([cs.ravel(), cd.ravel()])
    vals = _TENDENCY_FNS[kind](sigma, planes=planes).reshape(cs.shape)
    lon, lat = mpl.pole(mesh_s, mesh_d)

    m = ax.pcolormesh(lon, lat, vals, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
    label = cbar_label if cbar_label is not None else _CBAR_LABELS[kind]
    ax.get_figure().colorbar(m, ax=ax, label=label, shrink=0.5, **(cbar_kwargs or {}))
    return m


def stereo_line(ax, directions, label=None, **kwargs):
    """
    Plot one or many line elements on a stereonet axes.

    The label is assigned to the first point only.

    Parameters
    ----------
    ax : mplstereonet axes
    directions : array-like, shape (2,) or (N, 2)
        Plunge/azimuth pairs in degrees.
    label : str, optional
    **kwargs
        Forwarded to ``ax.line``.
    """
    directions = np.atleast_2d(directions)
    for n, (plunge, azimuth) in enumerate(directions):
        ax.line(plunge, azimuth, label=label if n == 0 else None, **kwargs)


def stereo_contour(ax, directions, label=None, color="k", levels=4, sigma=2,
                   linewidth=1.0, **kwargs):
    """
    Plot kernel density contour lines of line elements on a stereonet.

    Parameters
    ----------
    ax : mplstereonet axes
    directions : array-like, shape (N, 2)
        Plunge/azimuth pairs in degrees.
    label : str, optional
    color : str
    levels : int
    sigma : float
    linewidth : float
    **kwargs
        Extra kwargs forwarded to ``ax.density_contour``.
    """
    directions = np.atleast_2d(directions)
    ax.density_contour(directions[:, 0], directions[:, 1],
                       measurement="lines", colors=color,
                       levels=levels, sigma=sigma,
                       linewidths=linewidth, **kwargs)
