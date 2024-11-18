import matplotlib.pyplot as plt
import mplstereonet as mpl
import numpy as np

from fem2geo import transform_funcs as tr


def rotmatrix(angle, axis):
    """
    Creates a rotation matrix for a given angle and axis of rotation

    Arguments:
        angle: Angles in sexagesimal degrees
        axis: Axis upon which to do the rotation

    """

    if axis == 3:
        R = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
                      [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0],
                      [0, 0, 1]])
    elif axis == 2:
        R = np.array([[np.cos(np.deg2rad(angle)), 0, np.sin(np.deg2rad(angle))],
                      [0, 1, 0], [-np.sin(np.deg2rad(angle)), 0,
                                  np.cos(np.deg2rad(angle))]])
    elif axis == 1:
        R = np.array([[1, 0, 0], [0, np.cos(np.deg2rad(angle)),
                                  -np.sin(np.deg2rad(angle))], [0, np.sin(np.deg2rad(angle)),
                                                                np.cos(np.deg2rad(angle))]])
    return R


def rottensor(tensor, angle, axis):
    """
    Rotates a tensor by a given axis and angle

    Args:
        tensor: A 3x3 tensor
        angle: Angle in sexagesimal degrees
        axis: Axis upon which to do the rotation
    """

    R = rotmatrix(angle, axis)
    rotT = np.dot(R, np.dot(tensor, R.T))
    return rotT


def resshear_enu(plane, sigma):
    np.seterr(divide='ignore', invalid='ignore')
    """
    Get resolved shear of a stress tensor onto a plane.
    Input:
        - plane[str/dip]
        - tensor[3x3 np.ndarray] --> solid-mechanics sign convention
    
    Output:
    *Note: To avoid ambiguity between full/half-azimutal measures, vectors are 
    given in both a direction tau_hat (in the respective coordinate system), 
    and a scalar tau, to express both sense and magnitude of the stress vector.
    
        - tau:   Resolved shear stress magnitude. If tau is positive,
                 it represents the resolved stress onto the footwall 
                 ("related" to reverse-kinematics), elsewise is onto the 
                 hanging wall ("related" to normal-kinematics)
        - tau_hat: Resolver shear stress direction in ENU coordinates 
                   (physically pointing upwards)                 
    """
    n = tr.plane_sphe2enu(plane)
    t = np.dot(sigma, n)

    t_n = np.dot(t, n) * n
    t_s = t - t_n

    t_s_mag = np.linalg.norm(t_s)
    t_s_dir = t_s / t_s_mag
    if t_s_dir[2] < 0:
        t_s_mag *= -1
        t_s_dir *= -1

    return t_s_mag, t_s_dir


def get_slip_tendency(sigma, p_disc):
    """
    Gets the slip tendency value for a given stress tensor and plane.

    Args:
         sigma [3x3 np.ndarray]:  Full tensor
         p_disc [nx2 np.ndarray: str/dip]: Plane(s) upon which to calculate slip tendency value
    """

    D = []

    if p_disc.ndim == 1:
        n = tr.plane_sphe2enu(p_disc)
        t = np.dot(sigma, n)
        sigma_n = np.dot(t, n)
        sigma_s, sigma_s_dir = resshear_enu(p_disc, sigma)

        return abs(sigma_s / sigma_n)
    else:
        for plane in p_disc:
            n = tr.plane_sphe2enu(plane)
            t = np.dot(sigma, n)
            sigma_n = np.dot(t, n)
            sigma_s, sigma_s_dir = resshear_enu(plane, sigma)

            D.append(abs(sigma_s / sigma_n))

        return np.array(D)


def plot_slip_tendency(sigma, n_strikes=181, n_dips=46):
    """
    Calculates the slip tendency for tensor projected onto all planes in a polar discretization

    Args:
        sigma: The 3x3 tensor
        n_strikes: The amount of values to discretize the strikes
        n_dips: The amount of values to discretize the dip

    Returns:
        Returns, the figure, the axis, the slip tendency values and the discretized angles.
    """
    Val, Vec = np.linalg.eig(sigma)

    Val = np.sort(Val)  ## Sort list of eigen values: minimum is s1
    Vec = Vec[:, np.argsort(Val)]  ## Sort eigen vectors

    strikes = np.linspace(0, 360, n_strikes, endpoint=True)  ## every 2 angles
    dips = np.linspace(0, 90, n_dips, endpoint=True)  ## every 2 angles

    mesh_strikes, mesh_dips = np.meshgrid(strikes, dips)
    plane_data = np.array([([i[0], i[1]]) for i
                           in np.nditer((mesh_strikes, mesh_dips))])

    D = get_slip_tendency(sigma, plane_data)
    lon, lat = mpl.pole(mesh_strikes, mesh_dips)
    D_reshaped = D.reshape(mesh_strikes.shape)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='stereonet')
    ax.grid()
    cax = ax.pcolormesh(lon, lat, D_reshaped, cmap='jet', shading='auto')

    cbaxes = fig.add_axes([0.92, 0.1, 0.03, 0.3])  # Add additional axis for colorbar
    fig.colorbar(cax, cax=cbaxes, shrink=0.4)

    return fig, ax, D_reshaped, (mesh_strikes, mesh_dips)


def get_dilation_tendency(sigma, n_disc):
    """
    Gets the dilation tendency value for a given stress tensor and plane.

    Args:
         sigma [3x3 np.ndarray]:  Full tensor
         p_disc [nx2 np.ndarray: str/dip]: Plane(s) upon which to calculate dilation tendency
    """
    D = []
    val, vec = np.linalg.eig(sigma)
    s1 = np.min(val)
    s3 = np.max(val)

    for n_hat in n_disc:
        sn = np.dot(n_hat, np.dot(sigma, n_hat))
        D.append((s1 - sn) / (s1 - s3))
    return np.array(D)


def plot_dilation_tendency(sigma, n_strikes=181, n_dips=46):
    """
    Calculates the dilation tendency for tensor projected onto all planes from a polar
    discretization

    Args:
        sigma: The 3x3 tensor
        n_strikes: The amount of values to discretize the strikes
        n_dips: The amount of values to discretize the dip

    Returns:
        Returns, the figure, the axis, the dilation tendency values and the discretized angles.
    """
    Val, Vec = np.linalg.eig(sigma)

    Val = np.sort(Val)  # Sort list of eigen values: minimum is s1
    Vec = Vec[:, np.argsort(Val)]  # Sort eigen vectors

    strikes = np.linspace(0, 360, n_strikes, endpoint=True)  # every 2 angles
    dips = np.linspace(0, 90, n_dips, endpoint=True)  # every 2 angles

    mesh_strikes, mesh_dips = np.meshgrid(strikes, dips)
    norms = np.array([tr.plane_sphe2enu([i[0], i[1]]) for i
                      in np.nditer((mesh_strikes, mesh_dips))])
    a = norms.reshape((mesh_strikes.shape[0], mesh_strikes.shape[1], 3))

    D = get_dilation_tendency(sigma, norms)
    lon, lat = mpl.pole(mesh_strikes, mesh_dips)
    D_reshaped = D.reshape(mesh_strikes.shape)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='stereonet')
    ax.grid()
    cax = ax.pcolormesh(lon, lat, D_reshaped, cmap='jet', shading='auto')

    cbaxes = fig.add_axes([0.90, 0.02, 0.03, 0.3])  # Add additional axis for colorbar
    fig.colorbar(cax, cax=cbaxes, shrink=0.4,
                 label='Dilation Tendency $(\sigma_1 - \sigma_n)/(\sigma_1-\sigma_3)$')

    return fig, ax, D_reshaped, (mesh_strikes, mesh_dips)


def plot_slipndilation_tendency(sigma, n_strikes=181, n_dips=46):
    """
    Plots jointly the slip and dilation tendency for a tensor projected onto all planes from a
    polar discretization

    Args:
        sigma: The 3x3 tensor
        n_strikes: The amount of values to discretize the strikes
        n_dips: The amount of values to discretize the dip

    Returns:
        Returns, the figure, the axis, the tendency values and the discretized angles.
    """
    Val, Vec = np.linalg.eig(sigma)

    Val = np.sort(Val)  # Sort list of eigen values: minimum is s1
    Vec = Vec[:, np.argsort(Val)]  # Sort eigen vectors

    strikes = np.linspace(0, 360, n_strikes, endpoint=True)  # every 2 angles
    dips = np.linspace(0, 90, n_dips, endpoint=True)  # every 2 angles

    mesh_strikes, mesh_dips = np.meshgrid(strikes, dips)

    # Slip tendency
    plane_data = np.array([([i[0], i[1]]) for i
                           in np.nditer((mesh_strikes, mesh_dips))])

    D_s = get_slip_tendency(sigma, plane_data)
    lon_s, lat_s = mpl.pole(mesh_strikes, mesh_dips)
    D_reshaped_s = D_s.reshape(mesh_strikes.shape)

    # Dilation tendency
    norms = np.array([tr.plane_sphe2enu([i[0], i[1]]) for i
                      in np.nditer((mesh_strikes, mesh_dips))])
    a = norms.reshape((mesh_strikes.shape[0], mesh_strikes.shape[1], 3))

    D_d = get_dilation_tendency(sigma, norms)
    lon_d, lat_d = mpl.pole(mesh_strikes, mesh_dips)
    D_reshaped_d = D_d.reshape(mesh_strikes.shape)

    fig = plt.figure(figsize=(18, 8))

    ax_s = fig.add_subplot(121, projection='stereonet')
    ax_s.grid()
    ax_d = fig.add_subplot(122, projection='stereonet')
    ax_d.grid()

    cax = ax_s.pcolormesh(lon_s, lat_s, D_reshaped_s, cmap='rainbow', shading='auto')

    cbaxes_s = fig.add_axes([0.48, 0.1, 0.03, 0.3])  # Add additional axis for colorbar
    fig.colorbar(cax, cax=cbaxes_s, shrink=0.4)

    cax = ax_d.pcolormesh(lon_d, lat_d, D_reshaped_d, cmap='jet', shading='auto')

    cbaxes_d = fig.add_axes([0.92, 0.1, 0.03, 0.3])  # Add additional axis for colorbar
    fig.colorbar(cax, cax=cbaxes_d, shrink=0.4)

    return fig, ax_s, ax_d, D_reshaped_s, D_reshaped_d, (mesh_strikes, mesh_dips)
