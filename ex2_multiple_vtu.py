#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 18:42:38 2020

@author: pciturri
"""
import numpy as np
import pyvista as pv
import meshio as mi
import mplstereonet as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import os


import time

start = time.time()
from fem2geo_lib import transform_funcs as tr
from fem2geo_lib import model_handler as mh




### Manually set folders of the models to analyze
models_dir = '../../full_runs/Model Runs'  ## Modify according to your own directory

model_folders_name = ['P4_1_DC5VF_F0C2B1E4E10',
                      'P4_3_DC5VF_F0B4EE10',
                      'P4_4_DC5VF_F1B0E10E20',
                      'P4_5_DC5VF_F1B4E9']

vtu_name = 'jvgp1.15.vtu' ## Must be identical for all models, but if not..., 
                            ## then it can be added to each of the elements of 
                            ## list model_folders_name, and removed from 
                            ## variable filename in line 70 (the iterable)





## Initialize plot

plt.close('all')
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='stereonet')


#### Creating color list, to color each model
colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())

#### Initialize legend elements with circle for s1 and triangle for s3
legend_elements = [Line2D([0], [0], color='k', linewidth=0.001, marker='o', 
                          label=r'$\sigma_1$'),
                   Line2D([0], [0], color='k', linewidth=0.001, marker='^',
                          label=r'$\sigma_3$')]


## Iterate through all of the folders
for model in model_folders_name:

    
    ### Here we get the model filename by combinining 
    ### models_dir + model (the iterable) + vtu_name   
    ### *** Note that vtu_name can also be part of the iterable

    filename = os.path.join(models_dir, model, vtu_name)
    
    
    ##### Here is the same from example1
    full_model = pv.read(filename)
    
    center = np.array([22,22, -7])
    radius = 0.4
    box_dim = np.array([1,1,1])
    
    
    ##### GET SUBMODEL
    
    Submodel = mh.get_submodel_sphere(full_model, center, radius )
    
    ### Here I created a new method, to find cells within a bounding box, rather
    ### than a sphere. It might be faster (or not xd)... comment the Submodel above
    ### and discomment the line below
    
    # Submodel = mh.get_submodel_box(full_model, center, box_dim)
 
    
 
    
 
    ##############
    #Here you can uncomment, and save the sphere as vtu in each model folder,    
    ##############
    
    #file_saved_sphere = os.path.join(models_dir, model, 'sphere.vtu')
    #Submodel.save(file_saved_sphere)
    

    ##############
    #Or create a new folder, save all spheres in there.
    ## This later can be opened, withouth creating the submodels, thus saving
    ## a lot of time
    ##############
    
    # folder_name = './spheres/'
    # os.makedirs('./spheres/', exist_ok=True)
    # file_saved_sphere = os.path.join(folder_name, 'sphere_' + model + '.vtu')
    # Submodel.save(file_saved_sphere)
    
    
    
    s1 = Submodel.cell_arrays['dir_DevStress_1']
    s3 = Submodel.cell_arrays['dir_DevStress_3']
    
    s1_sphe = []
    for i in s1:
        s1_up = i*np.sign(i[2])
        s1_i = tr.line_enu2sphe(s1_up)
        s1_sphe.append(s1_i)
        
    ### Same for s3
    s3_sphe = []
    for i in s3:
        s3_up = i*np.sign(i[2])
        s3_i = tr.line_enu2sphe(s3_up)
        s3_sphe.append(s3_i)
     
        
    #### We pick a random color of the color list, and assign it to model
    #### (not rlly important)
    color = colors[np.random.randint(0, len(colors))]
    colors.pop(colors.index(color))
    legend_elements.append(Patch(facecolor=color, edgecolor=color, label=model))
    
    
    for n,i in enumerate(s1_sphe):
        ax.line(i[0],i[1] , marker='o',color=color, markeredgecolor='k')
    
    for n, i in enumerate(s3_sphe):
        ax.line(i[0],i[1] ,  marker='^', color=color,markeredgecolor='k')
        
        
ax.grid()      
## asign legend elements.
ax.legend(handles=legend_elements, loc=1, fontsize=7)
ax.set_title(r'Stereoplot of $\sigma_1$ and $\sigma_3$', y=1.08)


end = time.time()
print('Time elapsed:', end - start)