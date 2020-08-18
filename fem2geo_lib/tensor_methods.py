#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 17:12:08 2020

@author: pciturri
"""


import numpy as np
import mplstereonet as mpl
import matplotlib.pyplot as plt

import fem2geo_lib
from fem2geo_lib import transform_funcs as tr

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



def get_dilation_tendency(sigma, n_disc):
    

    D = []
    val, vec = np.linalg.eig(sigma)
    s1 = np.min(val)
    s3 = np.max(val)
    
    for n_hat in n_disc:
      sn = np.dot(n_hat, np.dot(sigma, n_hat))
      D.append((s1-sn)/(s1-s3))
    return np.array(D)
    

def plot_dilation_tendency(sigma, n_strikes=181, n_dips=46, plot_eigenvec=False):
    

    Val, Vec = np.linalg.eig(sigma)
    
    Val = np.sort(Val)      ## Sort list of eigen values: minimum is s1
    Vec = Vec[:, np.argsort(Val)]  ## Sort eigen vectors
    
    strikes = np.linspace(0, 360, n_strikes, endpoint=True) ## every 2 angles
    dips = np.linspace(0, 90, n_dips, endpoint=True)  ## every 2 angles
    
    mesh_strikes, mesh_dips = np.meshgrid(strikes, dips)
    norms = np.array([tr.plane_sphe2enu([i[0], i[1]]) for i 
                                         in np.nditer((mesh_strikes, mesh_dips))])
    a = norms.reshape((mesh_strikes.shape[0], mesh_strikes.shape[1],3))
    
    D = get_dilation_tendency(sigma, norms)
    lon, lat = mpl.pole(mesh_strikes, mesh_dips)
    D_reshaped = D.reshape(mesh_strikes.shape)
    

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='stereonet')
    ax.grid()
    cax = ax.pcolormesh(lon, lat, D_reshaped, cmap='jet', shading='auto')
    
    cbaxes = fig.add_axes([0.92, 0.1, 0.03, 0.3]) #Add additional axis for colorbar
    fig.colorbar(cax, cax = cbaxes, shrink = 0.4)
    
    return fig, ax, D_reshaped, (mesh_strikes, mesh_dips)


if __name__ == '__main__':
    
    print('hi')
    
    # b = [tf.line_enu2sphe(i) for i in a]
    
    # plt.close('all')
    # fig = plt.figure(figsize=(8,8))
    # ax = fig.add_subplot(111, projection='stereonet')  
    
    # for n, i in enumerate(b):

    #     ax.grid()
    #     mylabel = None
        
    #     if n==0:
    #         mylabel = r'$\sigma_3$ orientation'
    #     ax.line(i[0],i[1] , c='b', marker='o', markeredgecolor='k', label=mylabel)
        
        
        
        
        
    # azimuths = np.deg2rad(np.linspace(-90, 90, 90))
    # zeniths = np.deg2rad(np.arange(-90, 90, 1))
    
    # r, theta = np.meshgrid(zeniths, azimuths)
    # values = 1-2*np.random.random((azimuths.size, zeniths.size))
    
    # #-- Plot... ------------------------------------------------
    # strikes = np.linspace(0, 360, 130, endpoint=True)
    # dips = np.linspace(0, 90, 100, endpoint=True)
    
    # mesh_strikes, mesh_dips = np.meshgrid(strikes, dips)
    
    
    
    # lon, lat = mpl.pole(mesh_strikes, mesh_dips)
    # ax.pcolormesh(lon, lat, 1-lat**2, cmap='jet', shading='auto')





    # a
    # c = ax.contourf(shat[:,0], shat[:,1])