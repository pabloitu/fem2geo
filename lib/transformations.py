
import numpy as np
import matplotlib.pyplot as plt
import mplstereonet as mpl




# =============================================================================
# Line elements
# =============================================================================

#HOLAAA

print('Hola mundo')

def line_sphe2ned(sphe):
    """
    Transforms line element from spherical coordinates
    to cartesian N-E-D.
    
    [plunge/azm] ~deg -> [cos_a, cos_b, cos_c]
    """
    ned = np.array([np.cos(np.deg2rad(sphe[1]))*np.cos(np.deg2rad(sphe[0])),
                    np.sin(np.deg2rad(sphe[1]))*np.cos(np.deg2rad(sphe[0])),
                    np.sin(np.deg2rad(sphe[0]))])
    if ned[2] < 0:
        ned *= -1
    return ned

def line_ned2sphe(ned):
    """
    Transforms line element from cartesian N-E-D coordinates
    to spherical.
    
    [cos_a, cos_b, cos_c] -> [plunge,azm] ~deg
    """ 
    plunge = np.rad2deg(np.arcsin(ned[2]))     
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
    
    if plunge < 0:        
        azm += 180.
        
    if azm >= 360. or azm < 0.:
        azm -= 360.*np.sign(azm)
    
    return np.array([np.abs(plunge), azm ])

def line_sphe2enu(sphe):
    """
    Transforms line element from spherical coordinates
    to cartesian E-N-U.
    [plunge, azm] ~deg -> [cos_A, cos_B, cos_C]
    """
    ned = line_sphe2ned(sphe)
    if ned[2] > 0:
        enu = np.array([-ned[1], -ned[0], ned[2]])
        
    else:  #### Check, then with vertical planes this is going to be an issue....
        enu = np.array([ned[1], ned[0], np.abs(ned[2])])
        
    return enu
    
def line_enu2sphe(enu):
    """
    Transforms line element from cartesian E-N-U coordinates
    to spherical.
    [cos_A, cos_B, cos_C] -> [plunge, azm] ~deg
    """ 
    return line_ned2sphe([-enu[1], -enu[0], enu[2]])


def line_rake2sphe(rake):
    """
    Transforms line defined from rake and plane to spherical coordinates.
    [str, dip, r] ~deg -> [plunge, azm] ~deg
    
    *Note: Definition of line element is strict, thus no ambiguity is allowed 
    in the sense direction: rake must lie between 0 and 180. Sense of movement 
    must be treated separatedly.
    """ 
 
    if rake[2] < 0 or rake[2] > 180:
        raise Exception("Rake angle is not within 0 and 180 deg")
        
    plunge = np.rad2deg(np.arcsin(np.sin(np.deg2rad(rake[2]))*
                               np.sin(np.deg2rad(rake[1]))))
    azm = rake[0] + np.rad2deg(np.arctan2(np.cos(np.deg2rad(rake[1]))*
                                              np.sin(np.deg2rad(rake[2])),
                                          np.cos(np.deg2rad(rake[2]))))
    if plunge < 0.:
        azm += 180.
    if azm >= 360. or azm < 0.:
        azm -= 360.*np.sign(azm)
        
    return np.array([np.abs(plunge),azm])


# =============================================================================
# Plane elements
# =============================================================================


def plane_sphe2ned(sphe):
    """
    From plane spherical coordinates to ned vector of the plane normal
    (pointing downwards)
    [strike, dip] ~deg -> [cos_a,cos_b,cos_c]
    """

    v1 = line_sphe2ned([0, sphe[0]])
    v2 = line_sphe2ned([sphe[1], sphe[0] + 90.])
    n = np.cross(v1, v2)/np.linalg.norm(np.cross(v1, v2))


    ## Force the limit of the azimutal cosines (cos_a,cos_b)
    ## to be constant when dip tends to 0, instead of pointing +180.
    ## e.g     cos_a(dip=0.0001) ~= cos_a(dip=0)
    ##         cos_b(dip=0.0001) ~= cos_b(dip=0)
    if n[2] == 0:   
        if n[1] > 0:
            
            n *= -1
    return n

def plane_sphe2enu(sphe):
    """
    From plane spherical coordinates to enu vector of the plane normal
    (pointing upwards)
    [strike, dip] ~deg -> [cos_A,cos_B,cos_C]
    """
    v1 = line_sphe2enu([0, sphe[0]])
    v2 = line_sphe2enu([sphe[1], sphe[0] + 90.])  
    n = np.cross(v1, v2)/np.linalg.norm(np.cross(v1, v2))
    
    
    if n[2] == 0:
        if n[0] < 0:
            n *= -1
    return n



def plane_pole2sphe(sphe):
    """
    From line spherical coordinates representing a plane normal, to a 
    plane in spherical coords 
    [plunge, azm] ~deg -> [strike, dip]
    """

    strike = sphe[1] + 90.0
    if strike >= 360.0:
        strike = strike - 360.0
    if strike < 0.0:
        strike = 360. + strike
    dip = 90 - sphe[0]
    return np.array([strike, dip])

def transft(asdsa):
    print('asdsaq')

def lineplane2rake(enu, plane, tol=5e-3):
    """
    Tranforms a line (in enu) contained within a plane (in sphe) into a
    strike/dip/rake measure.
    [cos_A, cos_B, cos_C] & [str, dip]~deg [ -> [str, dip, rake]~deg
    
    """
    #Inputs: Linea en ENU y plano en coordenadas esféricas
    #Output: Plano con rake
    
    
    rho = line_sphe2enu([0, plane[0]])
    mu = line_sphe2enu(line_rake2sphe(np.array([plane[0], plane[1], 90])))
    n = np.cross(rho, mu)

    
    trp = np.abs(np.linalg.det(np.vstack((enu,mu,rho))))
    if trp > tol:
        raise Exception("Line is not contained within the plane.\n " + 
                        "  scalar triple prod:  %.5e" % trp)

    r = np.rad2deg(np.arccos(np.dot(rho,enu)))
    R_hat = np.cross(rho, enu)
    R_hat = R_hat/np.linalg.norm(R_hat)
    if np.dot(R_hat, n) > 0:
        r = 180 - r
        
    return np.array([plane[0],plane[1],r])




if __name__ == '__main__':
    
    enu = line_sphe2enu([80,135])
    plane = [289,20]
    rake = 179.
    Line_sphe = line_rake2sphe([plane[0], plane[1], rake])

    Line_enu = line_sphe2enu(Line_sphe)

    r = lineplane2rake(Line_enu,plane)
    print(r)

