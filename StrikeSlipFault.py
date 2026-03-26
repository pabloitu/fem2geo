# -*- coding: utf-8 -*-

import numpy as np

def get_boxpoints(dim, offset, rot_angle=0):
    """
    Get the 8 points defining a box. 
    Points are ordered in RHR - counterclockwise, from top to botton:
        (x_min,y_min,z_max)
        (x_max,y_min,z_max)
        (x_min,y_max,z_max)
        (x_max,y_max,z_max)
        (x_min,y_min,z_min)
        (x_max,y_min,z_min)
        (x_min,y_max,z_min)
        (x_max,y_max,z_min)
    Input:      - dim: Dimensions of the box
                - offset: Offset, or position of the first point
                - rot_angle (default 0): Once the box is created, and the offset
                is placed, a rotation of the box can be performed in the X-Y 
                plane, counterclockwise from the X axis.
    """
    
    ### Rotation matrix in the horizontal plane
    R = np.array([[np.cos(np.deg2rad(rot_angle)), 
                   -np.sin(np.deg2rad(rot_angle)), 0.],
                  [np.sin(np.deg2rad(rot_angle)),
                   np.cos(np.deg2rad(rot_angle)), 0.],
                   [0.,0.,1.]])
    
    ### Initial box, define with the first point in the origin
    box_points = np.array([[0.       , 0        , 0],
                           [dim[0]   , 0        , 0],
                           [dim[0]   , dim[1]   , 0],
                           [0        , dim[1]   , 0],
                           [0.       , 0        , -dim[2]],
                           [dim[0]   , 0        , -dim[2]],
                           [dim[0]   , dim[1]   , -dim[2]],
                           [0        , dim[1]   , -dim[2]]]) 

    ### Offset the points, and then rotates                 
    box_points = np.array([np.dot(R,np.array(i).T) + np.array(offset).T
                                            for i in box_points])

    return box_points

def get_faultpoints(rho, mu, trace_length, depth_length, thickness, offset,
                    depth_threshold=0.0):
    """
    Get the 8 points defining a box. It creates a NS fault with dimensions =
    (thickness, trace_length, depth_length). Afterwards, it gets rotated in the
    X-Z plane to provide the dip mu, and then rotated in the X-Y plane to 
    provide the strike.

    Input:      - rho: strike of the fault
                - mu: dip of the fault
                - trace_length: Fault length in the horizontal plane
                - depth_length: Vertical extent in the z plane (z_max-z_min) --
                  (careful, it does not define the dimension of the fault box, 
                  but rather up to what depth the fault extends)
                - thickness: Measure of the thickness in the plane normal 
                  direction
                - offset: Once the fault is created, it can be displaced by the
                  offset, with the origin set in the first point
                - depth_threshold: It defines up to what superior depth the 
                  fault can reach: it gets cropped upwards. Useful when the
                  fault extend up to the surface
    Output: numpy array with the points are ordered in RHR - counterclockwise,
            from top to botton:
                (x_min,y_min,z_max)
                (x_max,y_min,z_max)
                (x_min,y_max,z_max)
                (x_max,y_max,z_max)
                (x_min,y_min,z_min)
                (x_max,y_min,z_min)
                (x_min,y_max,z_min)
                (x_max,y_max,z_min)
    """    
    Rmu = np.array([[np.cos(np.deg2rad(90-mu)), 0,
                    -np.sin(np.deg2rad(90-mu))],
                    [0.,1.,0.],
                    [np.sin(np.deg2rad(90-mu)), 0.,
                     np.cos(np.deg2rad(90-mu))]])
    
    Rrho = np.array([[np.cos(np.deg2rad(-rho)), 
                      -np.sin(np.deg2rad(-rho)), 0.],
                      [np.sin(np.deg2rad(-rho)),
                       np.cos(np.deg2rad(-rho)), 0.],
                       [0.,0.,1.]])

    z = depth_length*np.cos(np.deg2rad(90-mu))
    
    fault_points = np.array([[0.            , 0             , 0],
                             [thickness     , 0             , 0],
                             [thickness     , trace_length  , 0],
                             [0             , trace_length  , 0],
                             [0.            , 0             , -z],
                             [thickness     , 0             , -z],
                             [thickness     , trace_length  , -z],
                             [0             , trace_length  , -z]])     
    

    fault_points = np.array([np.dot(Rmu,np.array(i)).T  for i in fault_points]) 
    fault_points = np.array([np.dot(Rrho,np.array(i)).T for i in fault_points])

    fault_points += offset 
    
    assertion = None
    for i,j in enumerate(fault_points):
        if j[2] > depth_threshold:
            assertion = True
            fault_points[i,2] = depth_threshold
    if assertion:
        print('One or more fault points are found above the surface. \
              Projected to the horizontal plane')
    return fault_points

def get_ellipsoidpoints(dim, offset, xy_rot=0.0, yz_rot=0.0, xz_rot=0.0):
    """
    Get the 7 points to define an ellipsoid. The first point defines the center
    of the ellipsoid, and the next 6 defines the start-and-end points of the
    Z, Y, X ellipsoid axes, respectively. A rotation in the three planes can be
    provided, first xz, then yz and finally xy planes.

    Input:      - dim: the dimensions of the ellipsoid's demiaxes
                - offset: the ellipsoid center position
                - xy_rot: rotation in the X-Y plane, angles in sex. degrees,
                          counterclockwise viewed from top (-Z)
                - yz_rot: rotation in the Y-Z plane, angles in sex. degrees,
                          counterclockwise, from the East  (-Y)
                - xz_rot: rotation in the X-Z plane, angles in sex. degrees,
                          counterclockwise, from the North (-X)

    Output: numpy array with the points are ordered in RHR - counterclockwise,
            from top to botton:
                (x_min,y_min,z_max)
                (x_max,y_min,z_max)
                (x_min,y_max,z_max)
                (x_max,y_max,z_max)
                (x_min,y_min,z_min)
                (x_max,y_min,z_min)
                (x_min,y_max,z_min)
                (x_max,y_max,z_min)
    """        
    
    Rxy = np.array([[np.cos(np.deg2rad(xy_rot)),
                    -np.sin(np.deg2rad(xy_rot)), 0],
                    [np.sin(np.deg2rad(xy_rot)), 
                     np.cos(np.deg2rad(xy_rot)), 0.],
                    [0.,0.,1.]])
    
    Ryz = np.array([[1.,0.,0.],
                    [0., np.cos(np.deg2rad(yz_rot)),
                     -np.sin(np.deg2rad(yz_rot))],
                    [0., np.sin(np.deg2rad(yz_rot)),
                     np.cos(np.deg2rad(yz_rot))]]) 

    Rxz = np.array([[np.cos(np.deg2rad(xz_rot)), 0., 
                     np.sin(np.deg2rad(xz_rot))],
                    [0.,1.,0.],
                    [-np.sin(np.deg2rad(xz_rot)), 0.,
                     np.cos(np.deg2rad(xz_rot))]]) 
    
    ellipsoid_points = np.array([[0.            , 0             , 0.],
                                 [0.            , 0             , dim[2]],
                                 [0.            , 0.            , -dim[2]],
                                 [0             , dim[1]        , 0.],
                                 [0.            , -dim[1]       , 0.],
                                 [dim[0]        , 0             , 0.],
                                 [-dim[0]       , 0.            , 0.]])  
    
    ellipsoid_points = np.array([np.dot(Rxy,(np.dot(Ryz,np.dot(Rxz,
                                  np.array(i).T)))) + np.array(offset).T
                                                    for i in ellipsoid_points])
    return ellipsoid_points


def box_geom_gen(file_name,
                 smallbox_points,
                 cl_small,
                 largebox_points=None,
                 cl_large=None,
                 faults_points=None,
                 cl_faults=None,
                 ellipsoids_points=None,
                 cl_ellipsoids=None):
    
    ### Set generic point and line ordering
      # For a random rectangular box, defines the point ordering to build the 
      # required lines
    rhr_pointarray = np.array([[1,2], [2,3], [3,4], [4,1],    ### top lines
                               [5,6], [6,7] ,[7,8], [8,5],    ### bot lines
                               [1,5], [2,6], [3,7], [4,8]])   ### side lines
    # Line ordering to build the rectangular box surfaces
    rhr_linearray = np.array([[1,2,3,4],                      ## top face
                              [-5,-8,-7,-6],                      ## bot face
                              [-1,9,5,-10],                   ## S face
                              [-2,10,6,-11],                  ## E face
                              [-3,11,7, -12],
                              [-4,12,8,-9]])

     # For a random ellipsoid, defines the point ordering to build the requi-
     # red Ellipses.
    ellipsoid_pointarray = np.array([[2, 1, 2, 6],  
                                     [6, 1, 3, 3],
                                     [2, 1, 2, 4],
                                     [4, 1, 3, 3],
                                     [2, 1, 2, 5], #
                                     [5, 1, 3, 3],
                                     [2, 1, 2, 7],
                                     [7, 1, 3, 3],  
                                     [6, 1, 6, 4], #
                                     [4, 1, 4, 7],
                                     [7, 1, 7, 5],
                                     [5, 1, 5, 6]])
     # Define the Ellipses ordering to build the Ellipsoid
    ellipsoid_linearray = np.array([[1, 9, -3],
                                    [3, 10, -7],
                                    [7, 11, -5],
                                    [5, 12, -1],
                                    [2, -4, -9],
                                    [4, -8, -10],
                                    [8, -6, -11],
                                    [6, -2, -12]])

    ### Build the problem's Points, Lines and Line Loops for each element
    ### (i.e. large box, small box, faults and ellipsoids)
    
    with open(file_name,'w') as f_:
        ## Characteristic Lengths 
        f_.write('cl_small = %.2f;\n' % cl_small)
        if largebox_points.__class__ == np.ndarray:
            f_.write('cl_large = %.2f;\n' % cl_large)  
        if faults_points:
            for i,j in enumerate(cl_faults):
                f_.write('cl_fault%i = %.2f;\n' % (i+1,j))
        if ellipsoids_points:
            for i,j in enumerate(cl_ellipsoids):
                f_.write('cl_ellipsoid%i = %.2f;\n' % (i+1,j))
                
        ## Starting point, line, surf and volunmes numbers (n-1)
        n_point = 0
        n_line = 0
        n_lineloop = 0
        n_surf = 0
        n_surfloop = 0
        n_vol = 0
        
        ## Initializing dictionaries to save interesting stuff throughout the 
        ## loop
        PointArrays = {'faults':[], 'ellipsoids':[],
                       'smallbox': None, 'largebox': None}
        LineArrays = {'faults':[], 'ellipsoids':[],
                      'smallbox': None, 'largebox': None}
        SurfArrays = {'faults':{}, 'ellipsoids':{},
                      'smallbox': [], 'largebox': []}
        
        ### Ellipsoids
        if ellipsoids_points:  
            for ellipsoid_i, ellipsoid_i_points in enumerate(ellipsoids_points):
                if ellipsoid_i == 0:
                    PointArrays['ellipsoids'].append(
                            {'id': np.arange(n_line + 1,
                                             n_line + 13),
                             'values': ellipsoid_pointarray + n_point})
                    LineArrays['ellipsoids'].append(
                            {'id': np.arange(n_lineloop + 1, n_lineloop + 9),
                             'values': np.sign(ellipsoid_linearray) *
                                       (np.abs(ellipsoid_linearray) + n_line)})
                else:
                    PointArrays['ellipsoids'].append(
                            {'id': np.arange(n_line + 1, n_line + 13),
                             'values': PointArrays['ellipsoids'][-1]['values']
                                       + 7})
                    
                    LineArrays['ellipsoids'].append(
                            {'id': np.arange(n_lineloop + 1, n_lineloop + 9),
                             'values': np.sign(LineArrays['ellipsoids'][-1]
                                       ['values'])*(np.abs(LineArrays
                                       ['ellipsoids'][-1]['values']) + 12)})
                
                for i in ellipsoid_i_points:
                    n_point += 1
                    f_.write(
                        'Point(%i) = {%.2f, %.2f, %.2f, cl_ellipsoid%i};\n' % 
                        (n_point, i[0], i[1], i[2], ellipsoid_i+1))
        
                for n,i in zip(
                        PointArrays['ellipsoids'][ellipsoid_i]['id'],
                        PointArrays['ellipsoids'][ellipsoid_i]['values']):
                    f_.write('Ellipse(%i) = {%i, %i, %i, %i};\n' % 
                             (n, i[0], i[1], i[2], i[3]))       
                n_line += 12
        
                for n,i in zip(
                        LineArrays['ellipsoids'][ellipsoid_i]['id'],
                        LineArrays['ellipsoids'][ellipsoid_i]['values']):   
                    f_.write('Line Loop(%i) = {%i, %i, %i};\n' % 
                             (n, i[0], i[1], i[2]))
                n_lineloop += 8
#                n_surf += 8                        
                            
                
        ### Faults
        if faults_points:
            for fault_i, fault_i_points in enumerate(faults_points):
                if fault_i == 0:
                    PointArrays['faults'].append({'id': np.arange(n_line + 1,
                                                                  n_line + 13),
                                                  'values': rhr_pointarray +
                                                            n_point})
                    
                    LineArrays['faults'].append({'id': np.arange(n_lineloop + 1, 
                                                                 n_lineloop + 7),
                                                 'values': np.sign(rhr_linearray)*
                                                           (np.abs(rhr_linearray)+
                                                            n_line)})
                else:
                    PointArrays['faults'].append({'id': np.arange(n_line + 1,
                                                                  n_line + 13),
                                                  'values': PointArrays['faults']
                                                            [-1]['values'] + 8})
                    
                    LineArrays['faults'].append({'id': np.arange(n_lineloop + 1,
                                                                 n_lineloop + 7),
                                                 'values': np.sign(LineArrays
                                                           ['faults'][-1]['values']
                                                           )*(np.abs(LineArrays
                                                           ['faults'][-1]
                                                           ['values']) + 12)})
    
                for i in fault_i_points:
                    n_point += 1
                    f_.write('Point(%i) = {%.2f, %.2f, %.2f, cl_fault%i};\n' % 
                             (n_point, i[0], i[1], i[2], fault_i+1))
        
                for n,i in zip(PointArrays['faults'][fault_i]['id'],
                               PointArrays['faults'][fault_i]['values']):
                    f_.write('Line(%i) = {%i, %i};\n' % (n, i[0], i[1]))       
                n_line += 12
        
                for n,i in zip(LineArrays['faults'][fault_i]['id'],
                               LineArrays['faults'][fault_i]['values']):   
        
                    f_.write('Line Loop(%i) = {%i, %i, %i, %i};\n' % 
                             (n, i[0], i[1], i[2], i[3]))
                n_lineloop +=6
#                n_surf += 6                
            
     
    
        ### Small Box
        PointArrays['smallbox'] = {'id': np.arange(n_line+1,n_line+13),
                                   'values': rhr_pointarray + n_point }
        
        LineArrays['smallbox'] = {'id': np.arange(n_lineloop + 1, n_lineloop + 7),
                                  'values': np.sign(rhr_linearray)*(np.abs(
                                              rhr_linearray)+ n_line)}        
        for i in smallbox_points:
            n_point += 1
            f_.write('Point(%i) = {%.2f, %.2f, %.2f, cl_small};\n' % 
                     (n_point, i[0], i[1], i[2]))

            
        for n,i in zip(PointArrays['smallbox']['id'],
                     PointArrays['smallbox']['values']):
            f_.write('Line(%i) = {%i, %i};\n' % (n, i[0], i[1]))       
        n_line += 12

        for n,i in zip(LineArrays['smallbox']['id'],
                     LineArrays['smallbox']['values']):   

            f_.write('Line Loop(%i) = {%i, %i, %i, %i};\n' % (n, i[0],
                                                             i[1],i[2],i[3]))
        n_lineloop += 6
#        n_surf += 6    

        if largebox_points.__class__ == np.ndarray:
        ## Large box gen
            PointArrays['largebox'] = {
                    'id': np.arange(n_line + 1 ,n_line + 13),
                    'values': PointArrays['smallbox']['values'] + 8 }
            LineArrays['largebox'] = {
                    'id': np.arange(n_lineloop+1, n_lineloop+7),
                    'values': np.sign(LineArrays['smallbox']['values'])*\
                              (np.abs(LineArrays['smallbox']['values'])+ 12)  }          
    
            for i in largebox_points:
                n_point += 1
                f_.write('Point(%i) = {%.2f, %.2f, %.2f, cl_large};\n' % 
                         (n_point, i[0], i[1], i[2]))        
                
            for n,i in zip(PointArrays['largebox']['id'],
                          PointArrays['largebox']['values']):
                f_.write('Line(%i) = {%i, %i};\n' % (n, i[0], i[1]))       
            n_line += 12
    
            for n,i in zip(LineArrays['largebox']['id'],
                         LineArrays['largebox']['values']):   
                f_.write('Line Loop(%i) = {%i, %i, %i, %i};\n' % 
                         (n, i[0], i[1],i[2],i[3]))
            n_lineloop += 6 
#            n_surf += 6
        
        ## Find faults coplanar to the smallbox. Identifies the surface numbers
        ## from the fault face contained within the smallbox face
        tol = 1e-3      ## Tolerance to find points contained within a plane
        id_coplanar_fs = [] 
        if faults_points:
    
            for n,i in enumerate(LineArrays['smallbox']['id']):
                ## For each smallbox surfaces, find the vectors n and d that 
                ## define the plane vector equation: n_hat . x = d
                points_small = smallbox_points[np.unique(np.ravel(
                               rhr_pointarray[list(np.abs(rhr_linearray[n])-1)]))
                               - 1]
                n_hat = np.cross(points_small[0] - points_small[1],
                                 points_small[0] - points_small[2])
                n_hat /= np.linalg.norm(n_hat)
                d = np.dot(n_hat, points_small[2])
                
                for fault_i, fault_points in enumerate(faults_points):
                    for m,j in enumerate(LineArrays['faults'][fault_i]['id']):
                        points_fault = fault_points[np.unique(np.ravel(
                                       rhr_pointarray[list(np.abs(
                                       rhr_linearray[m])-1)])) -1]
                        id_inside = 0
                                            
                        for p in points_fault:
                            if np.abs(np.dot(p,n_hat)-d)< tol:
                                id_inside +=1
                        if id_inside == 4: # Check if 4 points are contained
                            ### Saves LineLoop ids (LargeBox, SmallBox)                      
                            id_coplanar_fs.append([fault_i,i,j])   
        
    
        ## Find small box faces which are coplanar to the largebox.
        ## Identifies the surface numbers from the smallbox faces contained 
        ## within the largebox face
        tol = 1e-3      ## Tolerance to find points contained within a plane
        id_coplanar = [] 
        if largebox_points.__class__ == np.ndarray:
            for n,i in enumerate(LineArrays['largebox']['id']):
                points_large = largebox_points[np.unique(np.ravel(rhr_pointarray[list(np.abs(rhr_linearray[n])-1)])) -1]
                
                n_hat = np.cross(points_large[0]-points_large[1],points_large[0]-points_large[2])
                n_hat /= np.linalg.norm(n_hat)
                d = np.dot(n_hat, points_large[2])
                
                for m,j in enumerate(LineArrays['smallbox']['id']):
                    points_small = smallbox_points[np.unique(np.ravel(rhr_pointarray[list(np.abs(rhr_linearray[m])-1)])) -1]
                    id_inside = 0
                    for p in points_small:
                        if np.abs(np.dot(p,n_hat)-d)< tol:
                            id_inside +=1
                    if id_inside == 4:
                        ### Saves LineLoop faces ids (LargeBox, SmallBox)
                        id_coplanar.append([i,j])   
        
        
        ### Write final domains
        
        ####### SURFACES
        ## Ellipsoids surfaces
        if ellipsoids_points:
            for n, ellipsoid in enumerate(LineArrays['ellipsoids']):
                SurfArrays['ellipsoids'][n] = []
                for i in ellipsoid['id']:
                    n_surf += 1
                    f_.write('Surface(%i) = {%i};\n' % (n_surf, i))
                    SurfArrays['ellipsoids'][n].append(n_surf)        
        
        
        ## Faults surfaces  
        smallbox_miss = []    # Surface not included in the non-convex small 
                              # box volume (i.e. fault face in the ground surf.)
        if faults_points:
            for n, fault in enumerate(LineArrays['faults']):
                SurfArrays['faults'][n] = []
                for i in fault['id']:
                    n_surf += 1
                    f_.write('Surface(%i) = {-%i};\n' % (n_surf, i))
                    for j in id_coplanar_fs:
                        if j[2] == i:
                            smallbox_miss.append(n_surf)
                    SurfArrays['faults'][n].append(n_surf)


        ## Small box surfaces
        
        largebox_miss = []    ## Surfaces not included in the non-convex large
                              # box volume (i.e. smallbox_surf in the ground .surf.)
        for i in LineArrays['smallbox']['id']:
            n_surf += 1
            surf_ids = [i]
            for j in id_coplanar_fs:

                if j[1] == i:
                    surf_ids.append(j[2])
            f_.write('Surface(%i) = {' % n_surf + ', '.join(
                                [str(k) for k in surf_ids]) + '};\n')
            for j in id_coplanar:
                if j[1] == i:
                    largebox_miss.append(n_surf)
            SurfArrays['smallbox'].append(n_surf)
            
        ## Large box surfaces
        if largebox_points.__class__ == np.ndarray:
            for i in LineArrays['largebox']['id']:
                n_surf += 1
                surf_ids = [i]
                for j in id_coplanar:
                    if j[0] == i:
                        surf_ids.append(j[1])
                f_.write('Surface(%i) = {' % n_surf + ', '.join(
                                    [str(k) for k in surf_ids]) + '};\n' )             
                SurfArrays['largebox'].append(n_surf)
        
        ### VOLUMES
        ## Ellipsoids volumes
        aux_ellipsoid = []  
        if ellipsoids_points:            
            for i in range(len(ellipsoids_points)):
                n_surfloop += 1
                f_.write('Surface Loop(%i) = {' % n_surfloop + ', '.join(
                        [str(i) for i in SurfArrays['ellipsoids'][i]]) 
                                                                + '};\n')

                aux_ellipsoid.extend(SurfArrays['ellipsoids'][i])            
        
        
        ## Fault volumes        
        aux_faults = []
        if faults_points:
            for i in range(len(faults_points)):
                n_surfloop += 1
                f_.write('Surface Loop(%i) = {' % n_surfloop + ', '.join([str(i) for
                                          i in SurfArrays['faults'][i]]) + '};\n')
                n_vol += 1
                f_.write('Volume(%i) = {%i};\n' % (n_vol, n_surfloop))   

                
                aux_faults.extend(SurfArrays['faults'][i])
            
        for j in smallbox_miss:
            if faults_points:
                if j in aux_faults:
                    aux_faults.remove(j)
           

        surfs_small = [str(i) for i in SurfArrays['smallbox']]
        if faults_points:
            surfs_small.extend([-i for i in aux_faults])
        if ellipsoids_points:
            surfs_small.extend([-i for i in aux_ellipsoid])
        n_surfloop += 1        
        f_.write('Surface Loop(%i) = {' % n_surfloop + ', '.join([str(i) for
                                      i in surfs_small]) + '};\n')
     
        n_vol += 1
        f_.write('Volume(%i) = {%i};\n' % (n_vol, n_surfloop))   

        
        
        if largebox_points.__class__ == np.ndarray:
            aux_small = SurfArrays['smallbox']
            for i in largebox_miss:
                if i in aux_small:
                    aux_small.remove(i)
            
            surfs_large = [str(i) for i in SurfArrays['largebox']]
            surfs_large.extend([-i for i in aux_small])
            n_surfloop += 1 
            f_.write('Surface Loop(%i) = {' % n_surfloop + ', '.join([str(k) for k in surfs_large]) + '};\n')  
            n_vol += 1
            f_.write('Volume(%i) = {%i};\n' % (n_vol, n_surfloop))    

    f_.close()

    
if __name__ == '__main__':
    

    
    # Defining geometry
    # rectangle long axis oriented in dip direction of normal fault
    smallbox = get_boxpoints(dim=(15000,10000,6000),
                             offset= (0,0,0),
                             rot_angle=0.0)
    largebox = get_boxpoints(dim=(20000,10000,6000),
                             offset= (0,0,0),
                             rot_angle= 0.0)
    
    #Normal fault
    fault1 = get_faultpoints(rho=50, mu=90, trace_length=7000, depth_length=4350,
                             thickness=600, offset=(5000,3000, 0),
                             depth_threshold=-0.0) 

     
    #NOT USED
    #other fault -- E-W fault (Te Mihi Fault)
    fault2 = get_faultpoints(rho=30, mu=70, trace_length=24000, depth_length=7030,
                             thickness=600, offset=(18000,10000, 0),
                             depth_threshold=-0.0) 
                             
    #NOT USED
    #other fault -- NE-SW fault on western side - dip West (Whakaipo Fault)
    fault4 = get_faultpoints(rho=0, mu=-70, trace_length=24000, depth_length=7030,
                             thickness=600, offset=(11000,8000, 0),
                             depth_threshold=-0.0) 

   
    #NOT USED
    #other fault -- E-W fault on east quadrant (Te Mihi Fault)                        
    fault3 = get_faultpoints(rho=35, mu=70, trace_length=10000, depth_length=6500,
                             thickness=600, offset=(34200,34000, 0),
                             depth_threshold=-0.0) 
                             
    #NOT USED
    ellipsoid1 = get_ellipsoidpoints(dim= (17000, 17000, 2000),
                                     offset=(25000, 25000,-10000),
                                     xy_rot= -15., yz_rot=0.0, xz_rot=0)

    ### FULL TEST
    mesh_filename = './StrikeSlipFault_0.geo'
    box_geom_gen(file_name=mesh_filename,     ## mandatory params
                 smallbox_points=smallbox,    ## mandatory params             
                 cl_small=750,              ## mandatory params
                 largebox_points=[],    ## If no largebox is desired, set this value to None, or remove this line
                 cl_large=[],              ## If no largebox is desired, set this value to None, or remove this line
                 faults_points=[fault1],  ## If no fault is desired, set this value to [] or None or remove this line
                 cl_faults=[200],            ## If no fault is desired, set this value to [] or None or remove this line
                 ellipsoids_points=[],       ## If no ellipsoid is desired, set this value to [] or None or remove this line
                 cl_ellipsoids=[])           ## If no ellipsoid is desired, set this value to [] or None or remove this line
    print('Mesh file: ' + mesh_filename + ' created')   
