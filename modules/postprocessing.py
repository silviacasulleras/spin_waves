# -*- coding: utf-8 -*-
"""
Fourier transform implementation
27.09.21

@author: Silvia Casulleras
"""

import numpy as np

#Calculate the FT of a general function
def FT_function(omega,function):
    
    #omega is the frequency at which we evaluate the FT
    F_t  = function["values"]
    time = function["t"]
    time_0 = time[0]
    dt   = function["dt"]
    N_t  = len(time)
    
    suma=0
    for i in np.arange(0,N_t): #sum from t=0 to t=(N_t-1)*dt = Tfindal
        element = 1/(np.sqrt(2*np.pi))*dt*F_t[i]*np.exp(-1j*omega*(time_0+dt*i))
        suma+= element
    F_omega = suma
    return F_omega


#Calculate the FT of a general function
def IFT_function(t,function):
    
    #omega is the frequency at which we evaluate the FT
    F_omega    = function["values"]
    omega      = function["omega"]
    omega_min0 = omega[0]
    domega     = function["domega"]
    N_omega_1    = len(omega)
    
    suma=0
    for i in np.arange(0,N_omega_1):   #sum from omega=omega_min to omega=omega_min+N_omega*domega
        element = 1/(np.sqrt(2*np.pi))*domega*F_omega[i]*np.exp(1j*t*(omega_min0+domega*i))
        suma+= element
    F_time = suma
    return F_time

def IFT_function_pos(t,function):     #Fourier transform of a function f(t) when the input is given only in the positive range
    
    #omega is the frequency at which we evaluate the FT
    F_omega    = function["values"]
    omega      = function["omega"]
    omega_min0 = omega[0]
    domega     = function["domega"]
    N_omega    = len(omega)
    
    suma=0
    for i in np.arange(0,N_omega):
        element = 1/(np.sqrt(2*np.pi))*domega*(F_omega[i]*np.exp(1j*t*(omega_min0+domega*i))+ np.conjugate(F_omega[i])*np.exp(-1j*t*(omega_min0+domega*i)))
        suma+= element
    F_time = suma
    return F_time

