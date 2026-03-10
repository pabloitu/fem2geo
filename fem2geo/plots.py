import numpy as np
import matplotlib.pyplot as plt
import mplstereonet as mpl

from fem2geo.tensor import slip_tendency, dilation_tendency, grid_nodes, grid_centers


def plot_slip_tendency(sigma, n_strikes=180, n_dips=45, cmap="jet"):
    """
    Plot slip tendency on a stereonet.

    Parameters
    ----------
    sigma : array-like, shape (3, 3)
        Stress tensor in ENU coordinates.
    n_strikes : int
        Number of strike bins.
    n_dips : int
        Number of dip bins.
    cmap : str
        Matplotlib colormap name.

    Returns
    -------
    fig, ax, values, (mesh_strikes, mesh_dips)
    """
    mesh_strikes, mesh_dips = grid_nodes(n_strikes, n_dips)
    cs, cd = grid_centers(mesh_strikes, mesh_dips)

    planes = np.column_stack([cs.ravel(), cd.ravel()])
    vals = slip_tendency(sigma, planes=planes).reshape(cs.shape)

    lon, lat = mpl.pole(mesh_strikes, mesh_dips)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="stereonet")
    ax.grid()
    cax = ax.pcolormesh(lon, lat, vals, cmap=cmap, shading="auto")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    return fig, ax, vals, (mesh_strikes, mesh_dips)


def plot_dilation_tendency(sigma, n_strikes=180, n_dips=45, cmap="jet"):
    """
    Plot dilation tendency on a stereonet.

    Parameters
    ----------
    sigma : array-like, shape (3, 3)
        Stress tensor in ENU coordinates.
    n_strikes : int
        Number of strike bins.
    n_dips : int
        Number of dip bins.
    cmap : str
        Matplotlib colormap name.

    Returns
    -------
    fig, ax, values, (mesh_strikes, mesh_dips)
    """
    mesh_strikes, mesh_dips = grid_nodes(n_strikes, n_dips)
    cs, cd = grid_centers(mesh_strikes, mesh_dips)

    planes = np.column_stack([cs.ravel(), cd.ravel()])
    vals = dilation_tendency(sigma, planes=planes).reshape(cs.shape)

    lon, lat = mpl.pole(mesh_strikes, mesh_dips)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="stereonet")
    ax.grid()
    cax = ax.pcolormesh(lon, lat, vals, cmap=cmap, shading="auto")
    fig.colorbar(
        cax,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label=r"Dilation Tendency $(\sigma_1-\sigma_n)/(\sigma_1-\sigma_3)$",
    )

    return fig, ax, vals, (mesh_strikes, mesh_dips)


def plot_slip_dilation_tendency(
    sigma, n_strikes=180, n_dips=45, cmap_slip="rainbow", cmap_dil="jet"
):
    """
    Plot slip and dilation tendency side-by-side on stereonets.

    Parameters
    ----------
    sigma : array-like, shape (3, 3)
        Stress tensor in ENU coordinates.
    n_strikes : int
        Number of strike bins.
    n_dips : int
        Number of dip bins.
    cmap_slip : str
        Colormap for slip tendency.
    cmap_dil : str
        Colormap for dilation tendency.

    Returns
    -------
    fig, ax_slip, ax_dil, slip_vals, dil_vals, (mesh_strikes, mesh_dips)
    """
    mesh_strikes, mesh_dips = grid_nodes(n_strikes, n_dips)
    cs, cd = grid_centers(mesh_strikes, mesh_dips)
    planes = np.column_stack([cs.ravel(), cd.ravel()])

    slip_vals = slip_tendency(sigma, planes=planes).reshape(cs.shape)
    dil_vals = dilation_tendency(sigma, planes=planes).reshape(cs.shape)

    lon, lat = mpl.pole(mesh_strikes, mesh_dips)

    fig = plt.figure(figsize=(18, 8))
    ax_s = fig.add_subplot(121, projection="stereonet")
    ax_s.grid()
    ax_d = fig.add_subplot(122, projection="stereonet")
    ax_d.grid()

    c1 = ax_s.pcolormesh(lon, lat, slip_vals, cmap=cmap_slip, shading="auto")
    fig.colorbar(c1, ax=ax_s, fraction=0.046, pad=0.04)

    c2 = ax_d.pcolormesh(lon, lat, dil_vals, cmap=cmap_dil, shading="auto")
    fig.colorbar(c2, ax=ax_d, fraction=0.046, pad=0.04)

    return fig, ax_s, ax_d, slip_vals, dil_vals, (mesh_strikes, mesh_dips)