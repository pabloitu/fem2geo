import numpy as np
import pyvista as pv


def get_submodel_sphere(model, center, radius):
    """
    Extracts a sub-model from a full FEM vtk or vtu results file.
    It extracts the cells that are touched by a sphere, which is defined by its center and radius
    """
    connectivity = []
    for i in model.cells_dict.values():
        connectivity.extend(i.tolist())

    print('Creating sub-mesh')
    points_id = set(np.where(np.linalg.norm(model.points - center, axis=1)
                             < radius)[0])
    #print( np.linalg.norm(model.points - center, axis=1), radius)

    cells_id = [i for i, j in enumerate(connectivity) if points_id.intersection(j)]
    #print(cells_id)

    ien_ext = []
    for i, j in enumerate(connectivity):
        if i in cells_id:
            ien_ext.extend(j)
    points_list = list(set(ien_ext))

    pv_cells = []
    for i, j in enumerate(connectivity):
        if i in cells_id:
            cell = [len(j)]
            cell.extend([points_list.index(k) for k in j])
            pv_cells.extend(cell)

    print('Passing FEM data')
    #print(pv_cells)
    #print(cells_id)
    #print(points_list)
    extracted_model = pv.UnstructuredGrid(np.array(pv_cells),
                               model.celltypes[cells_id], model.points[points_list])
    for i, j in model.cell_data.items():
        extracted_model.cell_data[i] = j[cells_id]

    for i, j in model.point_data.items():
        extracted_model.point_data[i] = j[points_list]

    return extracted_model


def get_submodel_box(model, center, dim):
    """
    Extracts a sub-model from a full FEM vtk or vtu results file.
    It extracts the cells that are touched by a bounding box, which is defined by its center
    and extents
    """

    connectivity = []
    for i in model.cells_dict.values():
        connectivity.extend(i.tolist())

    print('Getting mesh data')
    ll = center - dim / 2.  # west,south,bottom corner
    ur = center + dim / 2.  # east,north,upper corner

    print('get subpoints')
    inidx = np.all(np.logical_and(ll <= model.points, model.points <= ur), axis=1)

    print('rebuilding mini mesh')
    points_id = set(np.where(inidx)[0])

    cells_id = [i for i, j in enumerate(connectivity) if points_id.intersection(j)]

    ien_ext = []
    for i, j in enumerate(connectivity):
        if i in cells_id:
            ien_ext.extend(j)
    points_list = list(set(ien_ext))

    pv_cells = []
    for i, j in enumerate(connectivity):
        if i in cells_id:
            cell = [len(j)]
            cell.extend([points_list.index(k) for k in j])
            pv_cells.extend(cell)

    grid = pv.UnstructuredGrid(np.array(pv_cells),
                               model.celltypes[cells_id], model.points[points_list])

    print('passing data')
    for i, j in model.cell_data.items():
        grid.cell_data[i] = j[cells_id]

    for i, j in model.point_data.items():
        grid.point_data[i] = j[points_list]
    return grid


def get_stress_weightedavg(model):
    """
    Weighted average of stress tensor by its relative cell size.
    """

    Model = model.compute_cell_sizes()

    avg_sigma = np.zeros((3, 3))

    print(Model.number_of_cells)

    for elem_id in range(Model.number_of_cells):
        #print(elem_id)
        #print(Model.cell_data['Volume'][elem_id])
        #print(Model.cell_data["stresses_(MPa)"].shape)
        tensor = Model.cell_data["stresses_(MPa)"]
        real_tensor = np.empty((3, 3))
        real_tensor[0][0] = tensor[elem_id][0]
        real_tensor[1][1] = tensor[elem_id][1]
        real_tensor[2][2] = tensor[elem_id][2]
        real_tensor[0][1] = tensor[elem_id][3]
        real_tensor[0][2] = tensor[elem_id][5]
        real_tensor[1][0] = tensor[elem_id][3]
        real_tensor[1][2] = tensor[elem_id][4]
        real_tensor[2][1] = tensor[elem_id][4]
        real_tensor[2][0] = tensor[elem_id][5]
        avg_sigma += real_tensor * Model.cell_data['Volume'][elem_id]
        #avg_sigma += np.array([np.array([Model.cell_data[key][elem_id]
        #                                 for key in key_row])
        #                       for key_row in tensor_order]) * \
        #             Model.cell_data['Volume'][elem_id]

    avg_sigma /= np.sum(Model.cell_data['Volume'])

    return avg_sigma


def get_stress_weightedavg_wolitho(model, ro=2800, g=9.81):
    """
    Weighted average of stress tensor by its relative cell size, accounting for lithospheric stress
    if given
    """

    Model = model.compute_cell_sizes()

    avg_sigma = np.zeros((3, 3))

    cells_centers = model.cell_centers()

    for elem_id in range(Model.number_of_cells):
        # multiply by 1000 if data is in km only
        cell_depth = cells_centers.extract_points(elem_id).bounds[5] * -1000

        #avg_sigma += (np.array([np.array([Model.cell_data[key][elem_id]
        #                                  for key in key_row])
        #                        for key_row in tensor_order]) + np.array(
        avg_sigma += (np.array([real_tensor]) + np.array( 
             [[ro * g * cell_depth / (10 ** 6), 0, 0] \
              , [0, ro * g * cell_depth / (10 ** 6), 0], \
             [0, 0, ro * g * cell_depth / (10 ** 6)]])) * \
                     Model.cell_data['Volume'][elem_id]

    avg_sigma /= np.sum(Model.cell_data['Volume'])

    return avg_sigma
