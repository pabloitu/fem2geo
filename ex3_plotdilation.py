import numpy as np
import matplotlib.pyplot as plt
import mplstereonet as mpl

import fem2geo_lib
from fem2geo_lib import transform_funcs as tr
from fem2geo_lib import model_handler as mh
from fem2geo_lib import tensor_methods as tm

"""

In this example, I just define a random tensor (without accessing the models)
and create a contoured stereoplot of the Dilation tendency. Each point in the 
stereoplot represents the Pole of the plane, and its coloured by its tendency
to dilate:

    e.g. 
-A point in the middle of the stereoplot, represent a roughly Horizontal plane
-One straight to the left (W direction), represents a vertical NS plane


I plot the sigma1, sigma2 and sigma 3 directions for convenience.

*** Note:   The DT is a relative measure, between 0 and 1,
for all planes possible. So, I think its only interesting to see this variable
in locations of the model, where high absolute dilation is observed.


"""
# =============================================================================
# Example 3a :  Manual plotting
# =============================================================================

# Define tensor and get principal values  
# (Pure shear, compressive in EW direction)
Tensor1 = np.array([[-1, 0 , 0],
                    [0, 0, 0],
                    [0, 0, 0]])

Val, Vec = np.linalg.eig(Tensor1)


Vec = Vec[:, np.argsort(Val)]  ## Sort eigen vectors according to sorted values
Val = np.sort(Val)      ## Sort list of eigen values: minimum is s1

s1 = Val[0]
s1_dir = Vec[:,0]

s2 = Val[1]
s2_dir = Vec[:,1]

s3 = Val[2]
s3_dir = Vec[:,2]

phi = np.abs((s2 - s3)/(s1 - s3))


# Create discretization of planes

strikes = np.linspace(0, 360, 181, endpoint=True) ## every 2 angles
dips = np.linspace(0, 90, 46, endpoint=True)  ## every 2 angles

# Create a mesh grid (pair strikes=[0, 2, ... 360] with dips = [0, 2 , .. 90])
mesh_strikes, mesh_dips = np.meshgrid(strikes, dips)

# Now we must get the planes normals. 
####  ** Note that the iteration goes first through columns and then rows of 
#### the meshes mesh_strieks and mesh_dips.
#### e.g: (1,1), (1,2), (1,3) ... (i,j), (i,j+1)... (ni, nj-1),(ni,nj)

norms = np.array([tr.plane_sphe2enu([i[0], i[1]]) for i 
                                      in np.nditer((mesh_strikes, mesh_dips))])

a = norms.reshape((mesh_strikes.shape[0], mesh_strikes.shape[1],3))

# We get the dilation tendency
D = tm.get_dilation_tendency(Tensor1, norms)


# # Get the stereoplot coordinates (lat, lon) from the strikes and dips
lon, lat = mpl.pole(mesh_strikes, mesh_dips)
# # Reshape the Dilation tendency array into the mesh discretization
D_reshaped = D.reshape(mesh_strikes.shape)

# Get the Dilation tendency contour stereoplot

plt.close('all')
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='stereonet')
ax.grid()
cax = ax.pcolormesh(lon, lat, D_reshaped, cmap='jet', shading='auto')

cbaxes = fig.add_axes([0.92, 0.1, 0.03, 0.3]) #Add additional axis for colorbar
fig.colorbar(cax, cax = cbaxes, shrink = 0.4)


# Get and plot principal directions
# s1_dir = s1_dir*(np.sign(s1_dir[2]) + (s1_dir[2]==0))  # Point upwards. Not sure if needed, or transform funcs deal with it
s1_sphe = tr.line_enu2sphe(s1_dir)
s2_sphe = tr.line_enu2sphe(s2_dir)
s3_sphe = tr.line_enu2sphe(s3_dir)

ax.line(s1_sphe[0],s1_sphe[1] , c='w', marker='o',
            markeredgecolor='k', markersize=8, label= r'$\sigma_1$')
ax.line(s2_sphe[0],s2_sphe[1] , c='w', marker='s',
            markeredgecolor='k', markersize=8, label= r'$\sigma_2$')
ax.line(s3_sphe[0],s3_sphe[1] , c='w', marker='v',
            markeredgecolor='k', markersize=8, label= r'$\sigma_3$')
ax.legend()

ax.set_title('Dilation tendency plot \n' + 
              '$\sigma_1=%.3f$, $\sigma_3=%.3f$, $\phi=%.2f$' %
                                              (s1, s3, phi),  y=1.05)


# =============================================================================
# Example 3b :  Automatic plotting  ()
# =============================================================================


# Horizontal simple shear
Tensor2 = np.array([[-1, 0 , 0],
                    [0, 1, 0],
                    [0, 0, 0]])

Val, Vec = np.linalg.eig(Tensor2)


Vec = Vec[:, np.argsort(Val)]
Val = np.sort(Val)     

s1 = Val[0]
s1_dir = Vec[:,0]
s2 = Val[1]
s2_dir = Vec[:,1]
s3 = Val[2]
s3_dir = Vec[:,2]
phi = np.abs((s2 - s3)/(s1 - s3))


fig, ax, D, mesh_planes = tm.plot_dilation_tendency(Tensor2)

# Get and plot principal directions
s1_sphe = tr.line_enu2sphe(s1_dir)
s2_sphe = tr.line_enu2sphe(s2_dir)
s3_sphe = tr.line_enu2sphe(s3_dir)

ax.line(s1_sphe[0], s1_sphe[1] , c='w', marker='o',
            markeredgecolor='k', markersize=8, label= r'$\sigma_1$')
ax.line(s2_sphe[0], s2_sphe[1] , c='w', marker='s',
            markeredgecolor='k', markersize=8, label= r'$\sigma_2$')
ax.line(s3_sphe[0], s3_sphe[1] , c='w', marker='v',
            markeredgecolor='k', markersize=8, label= r'$\sigma_3$')
ax.legend()

ax.set_title('Dilation tendency plot \n' + 
              '$\sigma_1=%.3f$, $\sigma_3=%.3f$, $\phi=%.2f$' %
                                              (s1, s3, phi),  y=1.05)


# =============================================================================
# Example 3c :  Random tensor, no plot of stresses
# =============================================================================


# Oblique simple shear
Tensor3 = np.array([[-1, 0. , 0],
                    [0., 1, 0],
                    [0, 0, -0.2]])
Tensor3_rot = tm.rottensor(Tensor3, 30, 1)
Tensor3_rot = tm.rottensor(Tensor3_rot, -45, 2)
Tensor3_rot = tm.rottensor(Tensor3_rot, 10, 3)


#### This functions returns the figure and axes elements, as well as the D
#### values and the planes discretization.

fig, ax, D, planes = tm.plot_dilation_tendency(Tensor3_rot)

# Get and plot principal directions

ax.set_title('Dilation tendency plot \n' + 
              '$\sigma_1=%.3f$, $\sigma_3=%.3f$, $\phi=%.2f$' %
                                              (s1, s3, phi),  y=1.05)

plt.show()