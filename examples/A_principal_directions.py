import os
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

import fem2geo
from fem2geo import transform_funcs as tr
from fem2geo import model_handler


###############################################################################
# This example illustrates the following
#   1. Read a .vtk file corresponding the results of an ADELI
#      (https://code.google.com/archive/p/adeli/)-FEM model
#   2. Extract a subset of elements within a centered sphere of radios R.
#   3. Plot the principal stress directions for all elements within the sphere
###############################################################################

# =============================================================================
# Get Data
# =============================================================================

# Select file path
example_dir = fem2geo.dir_testdata  # examples folder of the fem2geo package
filename = os.path.join(example_dir, 'small_box.vtk')

# Read File with Pyvista
full_model = pv.read(filename)


# =============================================================================
# Process data
# =============================================================================
# Select a sphere within the model (instead of a point) to smooth fluctuations.
# Coordinates of sphere center and radius
center = (22, 22, -7)
radius = 0.8

# Get Sub-model
sub_model = model_handler.get_submodel_sphere(full_model, center, radius)

# Save sub_model for visualization as vtu
sub_model.save(os.path.join(example_dir, 'ex1_circle.vtu'))   # <<<<<<<< Visualize in Paraview

# Get sub_model Sigma1 direction
print('Present cell variables in the model')
print(sub_model.cell_data.keys())

# We select dir_DevStress_1 and dir_DevStress_3 as variables.
s1 = sub_model.cell_data['dir_DevStress_1']
s3 = sub_model.cell_data['dir_DevStress_3']

# We iterate over all elements and get the spherical coordinates of Sigma_1
s1_spherical = []
for i in s1:
    # force s1 direction to point always upwards, to avoid azimuthal ambiguity
    s1_up = i*np.sign(i[2])
    # Convert ENU cartesian coordinates to spherical (plunge/azimuth)
    s1_i = tr.line_enu2sphe(s1_up)
    # Save into the list
    s1_spherical.append(s1_i)

# Same for Sigma_3
s3_spherical = []
for i in s3:
    s3_up = i*np.sign(i[2])
    s3_i = tr.line_enu2sphe(s3_up)
    s3_spherical.append(s3_i)

# =============================================================================
# Plot Data
# =============================================================================

# Standard stereo-plot initialization using the library mplstereonet
plt.close('all')
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='stereonet')

# add grid
ax.grid()

# Plot each Sigma_1 direction
for n, i in enumerate(s1_spherical):
    mylabel = None
    if n==0:
        mylabel = r'$\sigma_1$ orientation'
    ax.line(i[0], i[1], c='r', marker='o', markeredgecolor='k', label=mylabel)

# Plot each Sigma_3 direction
for n, i in enumerate(s3_spherical):
    mylabel = None
    if n==0:
        mylabel = r'$\sigma_3$ orientation'
    ax.line(i[0],i[1] , c='b', marker='o', markeredgecolor='k', label=mylabel)

# Show plots
ax.legend()
ax.set_title('Stereoplot of $\sigma_1$ and $\sigma_3$ \n' +
             'n of elements: %i' % sub_model.number_of_cells, y=1.08)
plt.show()



