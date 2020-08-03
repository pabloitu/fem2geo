import numpy as np
import pyvista as pv
import meshio as mi
import mplstereonet as mpl
import matplotlib.pyplot as plt

import lib.transformations as tr
import lib.model_handler as mh

# =============================================================================
# Get Data
# =============================================================================

filename = './test_data/test_box.vtk'

### Read File with Pyvista
full_model = pv.read(filename)

### Select coordinates of circle center and radius
center = (22,22, -7)
radius = 0.7

### Get Submodel
Submodel = mh.get_submodel_sphere(full_model, center, radius )

### Save submodel for visualization as vtu
Submodel.save('./test_data/circle.vtu')   #<<<<<<<< Visualize in paraview

### Get Submodel Sigma1 direction

print('Present CELL variables in model')
print(Submodel.cell_arrays.keys())


## We select dir_DevStress_1 and dir_DevStress_3 as variables

s1 = Submodel.cell_arrays['dir_DevStress_1']
s3 = Submodel.cell_arrays['dir_DevStress_3']


### We iterate over all s1 directions, to get its spherical coordinate
s1_sphe = []
for i in s1:
    # force s1 direciton to point always upwards, to avoid azimuthal ambiguity
    s1_up = i*np.sign(i[2])
    # Convert ENU cartesian coordiantes to spherical (plunge/azimuth)
    s1_i = tr.line_enu2sphe(s1_up)
    # Save into the list
    s1_sphe.append(s1_i)
    
### Same for s3
s3_sphe = []
for i in s3:
    s3_up = i*np.sign(i[2])
    s3_i = tr.line_enu2sphe(s3_up)
    s3_sphe.append(s3_i)
 
    
    
# =============================================================================
# Plot Data    
# =============================================================================
    
    
### Generical initialization of Stereoplots using the library mplstereonet

plt.close('all')
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='stereonet')

### add grid
ax.grid()

for n,i in enumerate(s1_sphe):
    mylabel = None
    if n==0:
        mylabel = r'$\sigma_1$ orientation'
    ax.line(i[0],i[1] , c='r', marker='o', markeredgecolor='k', label=mylabel)


for n, i in enumerate(s3_sphe):
    mylabel = None
    if n==0:
        mylabel = r'$\sigma_3$ orientation'
    ax.line(i[0],i[1] , c='b', marker='o', markeredgecolor='k', label=mylabel)
    
    
ax.legend()
ax.set_title('Stereoplot of $\sigma_1$ and $\sigma_3$ \n' + 
             'n of elements: %i' % Submodel.number_of_cells, y=1.08)
    

    