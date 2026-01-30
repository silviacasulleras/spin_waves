# -*- coding: utf-8 -*-
"""
Definition of the waveguide parameters

@author: Silvia Casulleras
"""

import numpy as np


def define_waveguide():
    #Define world size
    Xsize=80e-6
    Ysize=50e-9
    Zsize=50e-9

    #Define number of cells
    Nx=8000
    Ny=8
    Nz=8


    # MATERIAL/SYSTEM PARAMETERS
    By    = 0.270       # Bias field along the x direction
    Aex   = 4.2e-12     # exchange constant
    Ms    = 1.407e5     # saturation magnetization
    alpha = 1.75e-4     # damping parameter
    gamma = 1.76e11     # gyromagnetic ratio
    mu0   = 1.256e-6    # vacuum permeability
    alpha_x = 2*Aex/(mu0*(Ms**2)) #exchange constant in m**(-2)
    
    print("alpha_x=")
    print(alpha_x)
    l_ex    = np.sqrt(alpha_x) #exchange length
    
    print("Exchange length")
    print( l_ex*1e9)
    #Define CellSize
    Cx=Xsize/Nx
    Cy=Ysize/Ny
    Cz=Zsize/Nz
    
    #Define the parameters of the Gaussian antenna
    #A_antenna     = 1.6*1e-2    
    A_antenna     = 2.0053e-9  # Amplitude of the field created by the antenna 
    sigma_antenna = 30*1e-9    # Spatial extension of the antenna
    
    dk = 2*np.pi/Xsize
    k_min = -2*np.pi/(2*Cx)
    k_max = 2*np.pi/(2*Cx)
    N_k  = Nx
     
    print("gamma*mu0*M_s:")
    print(gamma*mu0*Ms/(2*np.pi*1e9))
    
    print("gamma*B0:")
    print(gamma*By/(2*np.pi*1e9))
    
    print("Factor: ")
    print(mu0*Ms/By)
    
    print("Kmin and kmax:")
    print(k_min*1e-6,k_max*1e-6)
      
    return {'Xsize' : Xsize,
            'Ysize' : Ysize,
            'Zsize' : Zsize,
            'Nx' : Nx,
            'Ny' : Ny,          
            'Nz' : Nz,        
            'By' : By,   
            'Aex': Aex,
            'Ms' : Ms,
            'alpha' : alpha,
            'gamma' : gamma, 
            'mu0'   : mu0,
            "A_antenna":A_antenna,
            "sigma_antenna":sigma_antenna,
            "dk":dk,
            "k_min":k_min,
            "k_max":k_max,
            "N_k":N_k
            }









