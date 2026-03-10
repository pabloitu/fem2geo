import numpy as np
import matplotlib.pyplot as plt
import mplstereonet as mpl

from fem2geo.tensor import slip_tendency, dilation_tendency, grid_nodes, grid_centers


__all__ = [
    "plot_slip_tendency",
    "plot_dilation_tendency",
    "plot_slip_dilation_tendency",
]


def _stereo_ax(fig, gs_cell, grid=True):
    """
    Create a stereonet axes in a GridSpec cell.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure handle.
    gs_cell : matplotlib.gridspec.SubplotSpec
        GridSpec cell.
    grid : bool
        Whether to show the stereonet grid.

    Returns
    -------
    matplotlib.axes.Axes
        Stereonet axes.
    """
    ax = fig.add_subplot(gs_cell, projection="stereonet")
    if grid:
        ax.grid(True)
    return ax


def _cbar_ax(fig, gs_cell):
    """
    Create a plain axes for a colorbar.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure handle.
    gs_cell : matplotlib.gridspec.SubplotSpec
        GridSpec cell.

    Returns
    -------
    matplotlib.axes.Axes
        Plain axes (no projection) suitable for colorbars.
    """
    return fig.add_subplot(gs_cell)


def _plane_grid(n_strikes, n_dips):
    """
    Build plane grid (centers) and stereonet node coordinates.

    Parameters
    ----------
    n_strikes : int
        Number of strike bins (cells).
    n_dips : int
        Number of dip bins (cells).

    Returns
    -------
    planes : numpy.ndarray, shape (n_strikes*n_dips, 2)
        Plane centers as [strike, dip] (degrees).
    lon, lat : numpy.ndarray
        Stereonet pcolormesh nodes as returned by mplstereonet.pole.
    shape : tuple
        (n_dips, n_strikes) output shape for reshaping computed values.
    meshes : tuple
        (mesh_strikes, mesh_dips) node meshes.
    """
    mesh_strikes, mesh_dips = grid_nodes(n_strikes, n_dips)
    cs, cd = grid_centers(mesh_strikes, mesh_dips)
    planes = np.column_stack([cs.ravel(), cd.ravel()])
    lon, lat = mpl.pole(mesh_strikes, mesh_dips)
    return planes, lon, lat, cs.shape, (mesh_strikes, mesh_dips)


def plot_slip_tendency(
    sigma,
    n_strikes=180,
    n_dips=45,
    cmap="jet",
    figsize=(8, 8),
    grid=True,
    vmin=None,
    vmax=None,
    cbar_label="",
    cbar_kwargs=None,
):
    """
    Plot slip tendency on a stereonet.

    Parameters
    ----------
    sigma : array-like, shape (3, 3)
        Stress tensor in ENU coordinates.
    n_strikes, n_dips : int
        Strike/dip discretization (number of cells).
    cmap : str
        Colormap name.
    figsize : tuple
        Figure size (inches).
    grid : bool
        Show stereonet grid.
    vmin, vmax : float, optional
        Color scaling for pcolormesh.
    cbar_label : str
        Colorbar label.
    cbar_kwargs : dict, optional
        Extra kwargs forwarded to fig.colorbar.

    Returns
    -------
    fig, ax, values, meshes, mappable, colorbar
    """
    planes, lon, lat, shp, meshes = _plane_grid(n_strikes, n_dips)
    vals = slip_tendency(sigma, planes=planes).reshape(shp)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.07], wspace=0.25)

    ax = _stereo_ax(fig, gs[0, 0], grid=grid)
    m = ax.pcolormesh(lon, lat, vals, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)

    cax = _cbar_ax(fig, gs[0, 1])
    cb = fig.colorbar(m, cax=cax, **(cbar_kwargs or {}))
    if cbar_label:
        cb.set_label(cbar_label)

    return fig, ax, vals, meshes, m, cb


def plot_dilation_tendency(
    sigma,
    n_strikes=180,
    n_dips=45,
    cmap="jet",
    figsize=(8, 8),
    grid=True,
    vmin=None,
    vmax=None,
    cbar_label=r"Dilation Tendency $(\sigma_1-\sigma_n)/(\sigma_1-\sigma_3)$",
    cbar_kwargs=None,
):
    """
    Plot dilation tendency on a stereonet.

    Parameters
    ----------
    sigma : array-like, shape (3, 3)
        Stress tensor in ENU coordinates.
    n_strikes, n_dips : int
        Strike/dip discretization (number of cells).
    cmap : str
        Colormap name.
    figsize : tuple
        Figure size (inches).
    grid : bool
        Show stereonet grid.
    vmin, vmax : float, optional
        Color scaling for pcolormesh.
    cbar_label : str
        Colorbar label.
    cbar_kwargs : dict, optional
        Extra kwargs forwarded to fig.colorbar.

    Returns
    -------
    fig, ax, values, meshes, mappable, colorbar
    """
    planes, lon, lat, shp, meshes = _plane_grid(n_strikes, n_dips)
    vals = dilation_tendency(sigma, planes=planes).reshape(shp)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.07], wspace=0.25)

    ax = _stereo_ax(fig, gs[0, 0], grid=grid)
    m = ax.pcolormesh(lon, lat, vals, cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)

    cax = _cbar_ax(fig, gs[0, 1])
    cb = fig.colorbar(m, cax=cax, **(cbar_kwargs or {}))
    if cbar_label:
        cb.set_label(cbar_label)

    return fig, ax, vals, meshes, m, cb


def plot_slip_dilation_tendency(
    sigma,
    n_strikes=180,
    n_dips=45,
    cmap_slip="rainbow",
    cmap_dil="jet",
    figsize=(18, 8),
    grid=True,
    vmin_slip=None,
    vmax_slip=None,
    vmin_dil=None,
    vmax_dil=None,
    cbar_label_slip="",
    cbar_label_dil="",
    cbar_kwargs_slip=None,
    cbar_kwargs_dil=None,
):
    """
    Plot slip and dilation tendency side-by-side on stereonets.

    Notes
    -----
    Slip and dilation use different scales, so this function always uses two colorbars.

    Returns
    -------
    fig, ax_slip, ax_dil, slip_vals, dil_vals, meshes, m1, cb1, m2, cb2
    """
    planes, lon, lat, shp, meshes = _plane_grid(n_strikes, n_dips)

    slip_vals = slip_tendency(sigma, planes=planes).reshape(shp)
    dil_vals = dilation_tendency(sigma, planes=planes).reshape(shp)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1.0, 0.07, 1.0, 0.07],
        wspace=0.35,
    )

    ax_s = _stereo_ax(fig, gs[0, 0], grid=grid)
    m1 = ax_s.pcolormesh(
        lon, lat, slip_vals, cmap=cmap_slip, shading="auto", vmin=vmin_slip, vmax=vmax_slip
    )
    cax1 = _cbar_ax(fig, gs[0, 1])
    cb1 = fig.colorbar(m1, cax=cax1, **(cbar_kwargs_slip or {}))
    if cbar_label_slip:
        cb1.set_label(cbar_label_slip)

    ax_d = _stereo_ax(fig, gs[0, 2], grid=grid)
    m2 = ax_d.pcolormesh(
        lon, lat, dil_vals, cmap=cmap_dil, shading="auto", vmin=vmin_dil, vmax=vmax_dil
    )
    cax2 = _cbar_ax(fig, gs[0, 3])
    cb2 = fig.colorbar(m2, cax=cax2, **(cbar_kwargs_dil or {}))
    if cbar_label_dil:
        cb2.set_label(cbar_label_dil)

    return fig, ax_s, ax_d, slip_vals, dil_vals, meshes, m1, cb1, m2, cb2