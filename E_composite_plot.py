import os

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

import fem2geo
from fem2geo import model_handler as mh
from fem2geo import tensor_methods as tm
from fem2geo import transform_funcs as tr

# Select file path
example_dir = '' #fem2geo.dir_testdata  # examples folder of the fem2geo package ('../test_data')
filename = os.path.join(example_dir, 'pM2F5R.2.vtu')

# Read File with Pyvista
full_model = pv.read(filename)

# Select coordinates of circle center and radius
center = (10000, 10000, -3000)
radius = 1000

# Get sub_model
sub_model = mh.get_submodel_sphere(full_model, center, radius)

# Save sub_model for visualization as vtu
short_filename = os.path.join(example_dir, 'dilation_zone2.vtu')
sub_model.save(short_filename)  # To visualize in paraview
sub_model = pv.read(short_filename)

print(sub_model.cell_data.keys())
# Get sub_model principal direction, and transform to spherical coords
s1 = [tr.line_enu2sphe(i) for i in sub_model.cell_data['dir_DevStress1_(-)']]
s2 = [tr.line_enu2sphe(i) for i in sub_model.cell_data['dir_DevStress2_(-)']]
s3 = [tr.line_enu2sphe(i) for i in sub_model.cell_data['dir_DevStress3_(-)']]

# Get average stress of the cells
avg_stress = mh.get_stress_weightedavg(sub_model)

# Get principal directions
val, vec = np.linalg.eig(avg_stress)

# Sort by maximum compressive
vec = vec[:, np.argsort(val)]
val = np.sort(val)

s1_avg = tr.line_enu2sphe(vec[:, 0].T)
s2_avg = tr.line_enu2sphe(vec[:, 1].T)
s3_avg = tr.line_enu2sphe(vec[:, 2].T)

# Get Slip & Dilation tendency plots simultaneously
fig, ax1, ax2, D1, D2, planes = tm.plot_slipndilation_tendency(avg_stress)

# Legend and extra directions (sigma1,2,3) for slip tendency plot
for n, i in enumerate(zip(s1, s2, s3)):
    mylabel = [None, None, None]
    if n == 0:
        mylabel = [r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$']
    ax1.line(i[0][0], i[0][1], c='r', marker='o', markeredgecolor='k', label=mylabel[0])
    ax1.line(i[1][0], i[1][1], c='g', marker='s', markeredgecolor='k', label=mylabel[1])
    ax1.line(i[2][0], i[2][1], c='b', marker='v', markeredgecolor='k', label=mylabel[2])

ax1.line(s1_avg[0], s1_avg[1], c='w', marker='o',
         markeredgecolor='k', markersize=8, label=r'Average $\sigma_1$')
ax1.line(s2_avg[0], s2_avg[1], c='w', marker='s',
         markeredgecolor='k', markersize=8, label=r'Average $\sigma_2$')
ax1.line(s3_avg[0], s3_avg[1], c='w', marker='v',
         markeredgecolor='k', markersize=8, label=r'Average $\sigma_3$')
ax1.legend()

ax1.set_title('Slip tendency plot \n' +
              '$\sigma_1=%.3f$, $\sigma_3=%.3f$, $\phi=%.2f$' %
              (val[0], val[2], (val[1] - val[2]) / (val[0] - val[2])), y=1.05)


# Legend and extra directions (sigma1,2,3) for dilation tendency plot
for n, i in enumerate(zip(s1, s2, s3)):
    mylabel = [None, None, None]
    if n == 0:
        mylabel = [r'$\sigma_1$', r'$\sigma_2$', r'$\sigma_3$']
    ax2.line(i[0][0], i[0][1], c='r', marker='o', markeredgecolor='k', label=mylabel[0])
    ax2.line(i[1][0], i[1][1], c='g', marker='s', markeredgecolor='k', label=mylabel[1])
    ax2.line(i[2][0], i[2][1], c='b', marker='v', markeredgecolor='k', label=mylabel[2])

ax2.line(s1_avg[0], s1_avg[1], c='w', marker='o',
         markeredgecolor='k', markersize=8, label=r'Average $\sigma_1$')
ax2.line(s2_avg[0], s2_avg[1], c='w', marker='s',
         markeredgecolor='k', markersize=8, label=r'Average $\sigma_2$')
ax2.line(s3_avg[0], s3_avg[1], c='w', marker='v',
         markeredgecolor='k', markersize=8, label=r'Average $\sigma_3$')
ax2.legend()

ax2.set_title('Dilation tendency plot \n' +
              '$\sigma_1=%.3f$, $\sigma_3=%.3f$, $\phi=%.2f$' %
              (val[0], val[2], (val[1] - val[2]) / (val[0] - val[2])), y=1.05)

plt.savefig('E_Slip_Dilatancy_2stereoplots_10_10_3.png', dpi=300)

plt.show()
