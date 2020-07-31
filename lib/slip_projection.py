
import numpy as np
import matplotlib.pyplot as plt
import mplstereonet as mpl
import struct_functions_v2005 as sf


def rotmatrix(angle,eje):
    #Input: Angulo de rotación en grados y eje en torno al que se gira
    #Output: Matriz de rotación en 3D, para rotaciones en el plano x-y
    if eje == 3:
        R = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)),0],\
                   [np.sin(np.deg2rad(angle)),np.cos(np.deg2rad(angle)),0],\
                   [0,0,1]])
    elif eje == 2:
        R = np.array([[np.cos(np.deg2rad(angle)), 0, np.sin(np.deg2rad(angle))],\
                       [0, 1, 0],[-np.sin(np.deg2rad(angle)),0,\
                       np.cos(np.deg2rad(angle))]])    
    elif eje == 1:
        R = np.array([[1, 0, 0],[0, np.cos(np.deg2rad(angle)),\
                      -np.sin(np.deg2rad(angle))],[0, np.sin(np.deg2rad(angle))\
                             ,np.cos(np.deg2rad(angle))]])
    return R


def rottensor(tensor,angle,eje):
    #Rota un tensor de stress en el eje que se desee
    #Input: Tensor (array), ángulo de rotación en grados y eje 
    #Output: Tensor rotado
    R = rotmatrix(angle,eje)
    rotT = np.dot(np.transpose(R),np.dot(tensor,R))
    return rotT


def resshear_enu(plane, sigma):
    
    """
    Get resolved shear of a stress tensor onto a plane.
    Input:
        - plane[str/dip]
        - tensor[3x3 np.ndarray] --> solid-mechanics sign convention
    
    Output:
    *Note: To avoid ambiguity between full/half-azimutal measures, vectors are 
    given in both a direction tau_hat (in the respective coordinate system), 
    and a scalar tau, to express both sense and magnitude of the stress vector.
    
        - tau:   Resolved shear stress magnitude. If tau is positive,
                 it represents the resolved stress onto the footwall 
                 ("related" to reverse-kinematics), elsewise is onto the 
                 hanging wall ("related" to normal-kinematics)
        - tau_hat: Resolver shear stress direction in ENU coordinates 
                   (physically pointing upwards)                 
    """
    n = sf.plane_sphe2enu(plane)
    t = np.dot(sigma,n)

    t_n = np.dot(t,n)*n
    t_s = t - t_n  
    
    t_s_mag = np.linalg.norm(t_s)
    t_s_dir = t_s/t_s_mag
    if t_s_dir[2] < 0:
        t_s_mag *= -1
        t_s_dir *= -1
        
    return t_s_mag, t_s_dir


def slip_tendency(plane, sigma):
    
    n = sf.plane_sphe2enu(plane)
    t = np.dot(sigma,n)

    t_n = np.dot(t,n)*n
    t_s = t - t_n  
    
    sigma_n = np.dot(t,n)
    sigma_s = np.linalg.norm(t_s)

    Ts = sigma_s/sigma_n
        
    return Ts
    
    




if __name__ == '__main__':
    
      
    T = np.array([[-2.0,0,0.],[0,-2.1,0],[0.,0,-2]])
    Trot = rottensor(T,45,3)
    
    val, vec = np.linalg.eig(Trot)
    s1 = sf.line_enu2sphe(vec[:,np.argmax(val)])
    s2 = sf.line_enu2sphe(vec[:,[k for k in [0,1,2] 
                    if k not in [np.argmax(val),np.argmin(val)]][0]])
    s3 = sf.line_enu2sphe(vec[:,np.argmin(val)])
    
    
    plt.close('all')
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='stereonet')

#    ax.density_contourf([2,13,55], [3,13,10], measurement='poles', cmap='Reds')
    
    ax.line(s1[0], s1[1], color='r')
    ax.line(s2[0], s2[1], color='y')
    ax.line(s3[0], s3[1], color='b')
    

#    i = np.lina    
    strikes = np.arange(1,359,5)
    dips = np.arange(5, 90, 5)
    strikes, dips = np.meshgrid(strikes, dips)
    strikes = strikes.ravel()
    dips = dips.ravel()
    
    Ts = [slip_tendency([i, j], Trot) for i,j in zip(strikes,dips)]
    cax = ax.density_contourf(strikes, dips, measurement='poles',weights=Ts )
    fig.colorbar(cax)
#    a = resshear_enu(plane,Trot)    
#    slip_sphe = sf.line_enu2sphe(a[1])

#    ax.plane(plane[0], plane[1])
#    ax.line(slip_sphe[0], slip_sphe[1])


#    plane = [255, 88]
#    a = slip_enu(plane,Trot)
#    
#    slip_sphe = sf.line_enu2sphe(a[1])
#    ax.plane(plane[0], plane[1])
#    ax.line(slip_sphe[0], slip_sphe[1])
    
