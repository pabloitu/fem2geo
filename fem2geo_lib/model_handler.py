
import numpy as np
import pyvista as pv
import meshio as mi
import mplstereonet as mpl
import matplotlib.pyplot as plt
#import transformations as tr


def get_submodel_sphere(model, center, radius):

    count = 0
    
    connectivity = []
    pv_connectivity = []
    pv_offset = []
    
    for i in range(model.number_of_cells-1):
        nnodes = model.cells[count]
        connectivity.append(list(model.cells[count+1:count + nnodes+1]))
        pv_connectivity.append(list(model.cells[count:count + nnodes+1]))
        pv_offset.append(count)
        count += nnodes + 1
        
    points_id = set(np.where(np.linalg.norm(model.points - center, axis=1) 
                                                < radius )[0])
    cells_id = [i for i,j in enumerate(connectivity) if points_id.intersection(j)]
    
    
    ien_ext = []
    for i,j  in enumerate(connectivity):
        if i in cells_id:
            ien_ext.extend(j)
    points_list =  list(set(ien_ext))


    pv_cells = []
    for i, j in enumerate(pv_connectivity):
        if i in cells_id:
            cell = j[:1]
            cell.extend([points_list.index(k) for k in j[1:]])
            pv_cells.extend(cell)


    grid = pv.UnstructuredGrid(np.array(pv_offset), np.array(pv_cells),
                               model.celltypes[cells_id], model.points[points_list])
    
    for i,j in model.cell_arrays.items():
        grid.cell_arrays[i] = j[cells_id]
        
    for i,j in model.point_arrays.items():
        grid.point_arrays[i] = j[points_list]
    return grid


def get_submodel_box(model, center, dim):

    count = 0
    
    connectivity = []
    pv_connectivity = []
    pv_offset = []
    
    for i in range(model.number_of_cells-1):
        nnodes = model.cells[count]
        connectivity.append(list(model.cells[count+1:count + nnodes+1]))
        pv_connectivity.append(list(model.cells[count:count + nnodes+1]))
        pv_offset.append(count)
        count += nnodes + 1
        
        
    ll = center - dim/2.  # west,south,bottom corner
    ur = center + dim/2.  # east,north,upper corner

    inidx = np.all(np.logical_and(ll <= model.points, model.points<= ur), axis=1)
    
    points_id = set(np.where(inidx)[0])

    cells_id = [i for i,j in enumerate(connectivity) if points_id.intersection(j)]
    
    
    ien_ext = []
    for i,j  in enumerate(connectivity):
        if i in cells_id:
            ien_ext.extend(j)
    points_list =  list(set(ien_ext))


    pv_cells = []
    for i, j in enumerate(pv_connectivity):
        if i in cells_id:
            cell = j[:1]
            cell.extend([points_list.index(k) for k in j[1:]])
            pv_cells.extend(cell)


    grid = pv.UnstructuredGrid(np.array(pv_offset), np.array(pv_cells),
                               model.celltypes[cells_id], model.points[points_list])
    
    for i,j in model.cell_arrays.items():
        grid.cell_arrays[i] = j[cells_id]
        
    for i,j in model.point_arrays.items():
        grid.point_arrays[i] = j[points_list]
    return grid


