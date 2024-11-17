import numpy as np
import pyvista as pv
import meshio as mi
import mplstereonet as mpl
import matplotlib.pyplot as plt

import fem2geo_lib
from fem2geo_lib import transform_funcs as tr
from fem2geo_lib import model_handler as mh
from fem2geo_lib import tensor_methods as tm

'''
Multiple plots
'''

# =============================================================================
# Get Data
# =============================================================================

filename = './test_data/P1_1_jvgp1.18.vtk'

# ### Read File with Pyvista
full_model = pv.read(filename)

# ### Select coordinates of circle center and radius
center = (24.75,28.5, -7.3)   ### dilation roughly between fault and chamber
radius = 1

# ### Get Submodel
Submodel = mh.get_submodel_sphere(full_model, center, radius )

# ### Save submodel for visualization as vtu
short_filename = './test_data/Dilation_zone.vtu'
Submodel.save(short_filename)   #<<<<<<<< To visualize in paraview

Submodel = pv.read(short_filename)

### Get Submodel principal direction, and transform to spherical coords
s1 = [tr.line_enu2sphe(i) for i in Submodel.cell_arrays['dir_DevStress_1']]
s2 = [tr.line_enu2sphe(i) for i in Submodel.cell_arrays['dir_DevStress_2']]
s3 = [tr.line_enu2sphe(i) for i in Submodel.cell_arrays['dir_DevStress_3']]


### Get average stress of the cells
avg_stress = mh.get_stress_weightedavg(Submodel)
'''
Line above may have to be changed to remove lithostatic load
'''

### Get principal directions
val, vec = np.linalg.eig(avg_stress)
### sort by maximum compressive
vec = vec[:,np.argsort(val)]
val = np.sort(val)

s1_avg = tr.line_enu2sphe(vec[:,0].T)
s2_avg = tr.line_enu2sphe(vec[:,1].T)
s3_avg = tr.line_enu2sphe(vec[:,2].T)


### Get Slip & Dilation tendency
plt.close('all')

# First and second axis are define as the slip tendency plot and the dilation tendency plot, respectively

fig,ax1,ax2,D1,D2,planes = tm.plot_slipndilation_tendency(avg_stress)

# Legend and extra directions (sigma1,2,3) for slip tendency plot

for n, i in enumerate(zip(s1,s2,s3)):
    mylabel = [None, None, None]
    if n==0:
        mylabel = [r'$\sigma_1$',r'$\sigma_2$',r'$\sigma_3$']
    ax1.line(i[0][0],i[0][1] , c='r', marker='o', markeredgecolor='k', label=mylabel[0])
    ax1.line(i[1][0],i[1][1] , c='g', marker='s', markeredgecolor='k', label=mylabel[1])
    ax1.line(i[2][0],i[2][1] , c='b', marker='v', markeredgecolor='k', label=mylabel[2])
    
ax1.line(s1_avg[0], s1_avg[1] , c='w', marker='o',
            markeredgecolor='k', markersize=8, label=r'Average $\sigma_1$')
ax1.line(s2_avg[0], s2_avg[1] , c='w', marker='s',
            markeredgecolor='k', markersize=8, label=r'Average $\sigma_2$')
ax1.line(s3_avg[0], s3_avg[1] , c='w', marker='v',
            markeredgecolor='k', markersize=8, label=r'Average $\sigma_3$')
ax1.legend()

ax1.set_title('Slip tendency plot \n' + 
              '$\sigma_1=%.3f$, $\sigma_3=%.3f$, $\phi=%.2f$' %
              (val[0], val[2], (val[1]-val[2])/(val[0]-val[2])),  y=1.05)

# Legend and extra directions (sigma1,2,3) for dilation tendency plot
   
for n, i in enumerate(zip(s1,s2,s3)):
    mylabel = [None, None, None]
    if n==0:
        mylabel = [r'$\sigma_1$',r'$\sigma_2$',r'$\sigma_3$']
    ax2.line(i[0][0],i[0][1] , c='r', marker='o', markeredgecolor='k', label=mylabel[0])
    ax2.line(i[1][0],i[1][1] , c='g', marker='s', markeredgecolor='k', label=mylabel[1])
    ax2.line(i[2][0],i[2][1] , c='b', marker='v', markeredgecolor='k', label=mylabel[2])
   
ax2.line(s1_avg[0], s1_avg[1] , c='w', marker='o',
            markeredgecolor='k', markersize=8, label=r'Average $\sigma_1$')
ax2.line(s2_avg[0], s2_avg[1] , c='w', marker='s',
            markeredgecolor='k', markersize=8, label=r'Average $\sigma_2$')
ax2.line(s3_avg[0], s3_avg[1] , c='w', marker='v',
            markeredgecolor='k', markersize=8, label=r'Average $\sigma_3$')
ax2.legend()

ax2.set_title('Dilation tendency plot \n' + 
              '$\sigma_1=%.3f$, $\sigma_3=%.3f$, $\phi=%.2f$' %
              (val[0], val[2], (val[1]-val[2])/(val[0]-val[2])),  y=1.05)

plt.show()