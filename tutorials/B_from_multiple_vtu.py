import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import fem2geo
from fem2geo import transform_funcs as tr
from fem2geo import model_handler as mh


###############################################################################
# This example illustrates the following
#   1. Read multiple .vtu files corresponding the results of multiple ADELI
#      (https://code.google.com/archive/p/adeli/)-FEM models
#   2. Extract a subset of elements within a centered sphere of radios R.
#   3. Plot the principal stress directions for all elements within the sphere
###############################################################################


# Select file paths
example_dir = fem2geo.dir_testdata  # examples folder of the fem2geo package
models = ['model_a', 'model_b', 'model_c', 'model_d']
filenames = [os.path.join(example_dir, 'model_a.vtu'),
             os.path.join(example_dir, 'model_b.vtu'),
             os.path.join(example_dir, 'model_c.vtu'),
             os.path.join(example_dir, 'model_d.vtu')]

# Initialize plot
plt.close('all')
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='stereonet')

# Creating a random color list, from which to color each model
colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())

# Initialize legend elements with circle for s1 and triangle for s3
legend_elements = [Line2D([0], [0], color='k', linewidth=0.001, marker='o',
                          label=r'$\sigma_1$'),
                   Line2D([0], [0], color='k', linewidth=0.001, marker='^',
                          label=r'$\sigma_3$')]

# Iterate through models
for model, filename in zip(models, filenames):

    # Same as example1
    full_model = pv.read(filename)

    center = np.array([22, 22, -7])
    radius = 0.4
    box_dim = np.array([0.8, 0.8, 0.8])

    # Two sub-model extractions are possible. A bounding sphere or a bounding box
    # sub_model = mh.get_submodel_sphere(full_model, center, radius)
    sub_model = mh.get_submodel_box(full_model, center, box_dim)

    # Here you can uncomment to actually save the sub_models paraview files
    # sub_model.save(os.path.join(example_dir, f'{model}_extract.vtu'))  # Visualize in Paraview

    # Extract principal directions
    s1 = sub_model.cell_data['dir_DevStress_1']
    s3 = sub_model.cell_data['dir_DevStress_3']

    s1_spherical = []
    for i in s1:
        s1_up = i * np.sign(i[2])
        s1_i = tr.line_enu2sphe(s1_up)
        s1_spherical.append(s1_i)

    # Same for s3
    s3_spherical = []
    for i in s3:
        s3_up = i * np.sign(i[2])
        s3_i = tr.line_enu2sphe(s3_up)
        s3_spherical.append(s3_i)

    # We pick a random color of the color list, and assign it to model (not rlly important)
    color = colors[np.random.randint(0, len(colors))]
    colors.pop(colors.index(color))
    legend_elements.append(Patch(facecolor=color, edgecolor=color, label=model))

    # Plot each direction onto a stereo-plot
    for n, i in enumerate(s1_spherical):
        ax.line(i[0], i[1], marker='o', color=color, markeredgecolor='k')

    for n, i in enumerate(s3_spherical):
        ax.line(i[0], i[1], marker='^', color=color, markeredgecolor='k')

# Customize and show plot
ax.grid()
ax.legend(handles=legend_elements, loc=1, fontsize=7)
ax.set_title('Stereoplot of $\\sigma_1$ and $\\sigma_3$', y=1.08)
plt.show()
