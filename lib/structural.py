
import numpy as np
import matplotlib.pyplot as plt
import mplstereonet as mpl



# =============================================================================
# Line elements
# =============================================================================


def line_sphe2ned(sphe):
    """
    [buz/azm] ~deg -> [cos_a, cos_b, cos_c]
    """
    ned = np.array([np.cos(np.deg2rad(sphe[1]))*np.cos(np.deg2rad(sphe[0])),
                    np.sin(np.deg2rad(sphe[1]))*np.cos(np.deg2rad(sphe[0])),
                    np.sin(np.deg2rad(sphe[0]))])
    if ned[2] < 0:
        ned *= -1
    return ned

def line_ned2sphe(ned):
    """
    [cos_a, cos_b, cos_c] -> [buz,azm] ~deg
    """ 
    buz = np.rad2deg(np.arcsin(ned[2]))     
    if ned[0] != 0:
        if ned[0] > 0 :
            dir_ = 0.
        else:
            dir_ = 180.
        azm = np.rad2deg(np.arctan(ned[1]/ned[0])) + dir_
    else:
        if ned[1] > 0:
            dir_ = 0.
        else:
            dir_ = 180.
        azm = 90. + dir_
    
    if buz < 0:        
        azm += 180.
        
    if azm > 360. or azm < 0.:
        azm -= 360.*np.sign(azm)
    
    return np.array([np.abs(buz), azm ])

def line_sphe2enu(sphe):
    """
    [buz, azm] ~deg -> [cos_A, cos_B, cos_C]
    """
    ned = line_sphe2ned(sphe)
    if ned[2] > 0:
        enu = np.array([-ned[1], -ned[0], ned[2]])
        
    else:  #### Check, then with vertical planes this is going to be an issue....
        enu = np.array([ned[1], ned[0], np.abs(ned[2])])
        
    return enu
    
def line_enu2sphe(enu):
    """
    [cos_A, cos_B, cos_C] -> [buz, azm] ~deg
    """ 
    return line_ned2sphe([-enu[1], -enu[0], enu[2]])


def rake2sphe(rake):
    """
    [str, dip, r] ~deg -> [buz, azm] ~deg
    """ 
 
    if rake[2] < 0 or rake[2] > 180:
        raise Exception("Rake angle is not within 0 and 180 deg")
    buz = np.rad2deg(np.arcsin(np.sin(np.deg2rad(rake[2]))*
                               np.sin(np.deg2rad(rake[1]))))
    azm = rake[0] + np.rad2deg(np.arctan2(np.cos(np.deg2rad(rake[1]))*
                                              np.sin(np.deg2rad(rake[2])),
                                          np.cos(np.deg2rad(rake[2]))))
    if buz < 0.:
        azm += 180.
    if azm > 360. or azm < 0.:
        azm -= 360.*np.sign(azm)
        
    return np.array([np.abs(buz),azm])


# =============================================================================
# Plane elements
# =============================================================================


def plane_sphe2ned(sphe):
    """
    From plane spherical coordinates to ned vector of the plane normal
    (pointing downwards)
    [strike, dip] -> [cos_a,cos_b,cos_c]
    """

    v1 = line_sphe2ned([0, sphe[0]])
    v2 = line_sphe2ned([sphe[1], sphe[0] + 90.])
    n = np.cross(v1, v2)/np.linalg.norm(np.cross(v1, v2))


    if n[2] == 0:
        if n[1] > 0:
            n *= -1
    return n

def plane_sphe2enu(sphe):
    """
    From plane spherical coordinates to enu vector of the plane normal
    (pointing upwards)
    [strike, dip] -> [cos_A,cos_B,cos_C]
    """
    v1 = line_sphe2enu([0, sphe[0]])
    v2 = line_sphe2enu([sphe[1], sphe[0] + 90.])  
    n = np.cross(v1, v2)/np.linalg.norm(np.cross(v1, v2))
    if n[2] == 0:
        if n[0] < 0:
            n *= -1
    return n



def pole2plane(sphe):
    ## Entrega un plano en base a su polo
    #Input: buz - azm  (esféricas)
    #Output: rho, mu   (esféricas) 

    rho = sphe[1] + 90.0
    if rho > 360.0:
        rho = rho - 360.0
    if rho < 0.0:
        rho = 360. + rho
    mu = 90 - sphe[0]
    return np.array([rho, mu])


def line2rake_enu(enu, plane):
    #Convierte una linea contenida en un plano a notacion rake
    #Inputs: Linea en ENU y plano en coordenadas esféricas
    #Output: Plano con rake
    rho = line_sphe2enu(rake2sphe(np.array([plane[0], plane[1], 0])))
    mu = line_sphe2enu(rake2sphe(np.array([plane[0], plane[1], 90])))  
    if np.dot(rho,enu) > 0:
        if np.dot(mu,enu) > 0:
            r = np.rad2deg(np.arccos(np.dot(rho,enu)))
        else:
            r = 360 - np.rad2deg(np.arccos(np.dot(rho,enu)))
    elif np.dot(rho,enu) < 0:
        if np.dot(mu,enu) > 0:
            r = np.rad2deg(np.arccos(np.dot(rho,enu)))
        else:
            r = 360 - np.rad2deg(np.arccos(np.dot(rho,enu)))
    return np.array([plane[0],plane[1],r])


if __name__ == '__main__':
    
    plt.close('all')
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='stereonet')
    
    
    a = plane_sphe2enu([225, 90])
    
    b = line_enu2sphe(a)
    ax.plane(235, 60)
    ax.line(b[0], b[1])
    ax.pole(235,60)
    
    
    
