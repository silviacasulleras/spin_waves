"""
Created on Thu Feb  4 11:20:49 2021
Simulation of the evolution of a chirped spin wave in a trapezoidal waveguide (Following code from Qi Wang)
@author: Silvia Casulleras
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from glob import glob
from os import path
from numpy import load
from matplotlib.colors import LogNorm
import scipy.special

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#n=4
c = {"blue":'tab:blue', "orange": '#D95319', "green":"#5cc863", "green2":"#27ad81", "yellow": 'gold', "purple":'#472c7a', "red":"#A2142F", "light-blue":"4DBEEE" }
  

#from modules.postprocessing import show_m_comp, show_m_vs_z, show_m_vs_y, show_m_abs, show_beff, read_mumax3_ovffiles_custom, show_mz_vs_x, extractdispersion, correctdispersion, find_polynomial_fit, FT_function, FT_function_2D

from modules.mumax3_functions import create_mumax_script, read_npy_files

def plot_dispersion_rel():
    dispersion = {  "m(k,w)"       : np.load("simulations/dispersion_relation/m(k,omega).npy") ,
                    "pol_6th"      : np.load("simulations/dispersion_relation/pol_fit_6th.npy") , 
                    "omega_cutoff" : np.load("simulations/dispersion_relation/omega_cutoff.npy"),
                    "k1"           : np.load("simulations/dispersion_relation/k1.npy"),
                    "k2"           : np.load("simulations/dispersion_relation/k2.npy"),
                    "omega1"       : np.load("simulations/dispersion_relation/omega1.npy"),
                    "omega2"       : np.load("simulations/dispersion_relation/omega2.npy"),
                    "dk"           : np.load("simulations/dispersion_relation/dk.npy"),
                    "domega"       : np.load("simulations/dispersion_relation/domega.npy"),
                    "degree_pol_fit" : np.load("simulations/dispersion_relation/degree_pol_fit.npy"), 
                    "N_k_fit"      : np.load("simulations/dispersion_relation/N_k_fit.npy"),
                    "N_omega_fit"  : np.load("simulations/dispersion_relation/N_omega_fit.npy"),
                    "k_min"        : np.load("simulations/dispersion_relation/k_min.npy"),
                    "k_max"        : np.load("simulations/dispersion_relation/k_max.npy"),
                    "omega_min"    : np.load("simulations/dispersion_relation/omega_min.npy"),
                    "omega_max"    : np.load("simulations/dispersion_relation/omega_max.npy"),
                    "N_omega"      : np.load("simulations/dispersion_relation/N_omega.npy")
                   }
    
    mz0_fft         = dispersion["m(k,w)"]
    polcoeffs       = dispersion["pol_6th"]
    omega_c         = dispersion["omega_cutoff"]
    k1              = dispersion["k1"]
    k2              = dispersion["k2"]
    omega1          = dispersion["omega1"]
    omega2          = dispersion["omega2"]
    N_k_fit         = dispersion["N_k_fit"]
    N_omega_fit     = dispersion["N_omega_fit"]
    degree_pol_fit  = dispersion["degree_pol_fit"]
    domega          = dispersion["domega"]
    k_min           = dispersion["k_min"]
    k_max           = dispersion["k_max"]
    omega_min       = dispersion["omega_min"]
    omega_max       = dispersion["omega_max"]
    N_omega         = dispersion["N_omega"]
    
    def disp_fit(k,polcoeffs,degree_pol_fit):
        d = degree_pol_fit
        s = 0
        for i in range(0,d+1,1):
            s+= polcoeffs[i]*k**(d-i)
        return s
    
    klist = np.linspace(k1,k2,N_k_fit+1)  
    
    dispersion_list = disp_fit(klist,polcoeffs,degree_pol_fit)

    
    #Plot the dispersion relation and its quadratic approximation
    
    extent = [ 1e-6*k_min, 1e-6*k_max, 1e-9*omega_min/(2*np.pi),  1e-9*omega_max/(2*np.pi)]  # extent of k values and frequencies
 
    
    #Plot the dispersion relation and its quadratic approximation with the legend inside
    
    cm = 1/2.54
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    
    fig = plt.figure(figsize=(2.67*cm, 2*cm))
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.imshow(np.log10(np.transpose(np.abs(mz0_fft)**2,(1,0))), extent=extent, aspect='auto', origin='lower', cmap="viridis", vmin=0, vmax=5)
    plt.plot(klist/1e6,dispersion_list/(2*np.pi*1e9),color=c["orange"], linestyle= "dashed",label="pol. fit",linewidth=1.2,dashes=[2, 1])
    plt.ylim(6.1,7.7)
    plt.xlim(0,25)
    plt.ylabel("$\omega/(2\pi) $ (GHz)",labelpad=0)
    plt.xlabel("$k\,\, [\mu m^{-1}] $",labelpad=0)
    plt.tick_params(direction='in',color="black")
    cbaxes = fig.add_axes([0.1, 0.87, 0.5, 0.07]) 
    cbar = plt.colorbar(aspect=20,cax=cbaxes,orientation="horizontal")
    cbar.set_label('           log($|m_z(x,0,0,t)/M_s|^2$)', rotation=0, labelpad=-34,color='black')
    cbar.outline.set_edgecolor('white')
    cbaxes.tick_params(axis='both', colors='white')
    plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0)
    
    plt.savefig("plots/plots_article/dispersion_relation_spectrum.pdf", bbox_inches = 'tight',transparent=True,dpi=300)
    plt.savefig("plots/plots_article/dispersion_relation_spectrum.svg")
    
    plt.show()



def plot_transfer_function(xpoint):
    transfer_function={"omega"    : np.load(f"""simulations/transfer_function/F(omega)_freqs.npy"""), 
                       "F(omega)" : np.load(f"""simulations/transfer_function/F(omega)_values.npy"""),
                       "t"        : np.load(f"""simulations/transfer_function/F(t)_time.npy"""),
                       "F(t)"     : np.load(f"""simulations/transfer_function/F(t)_values.npy"""),
                       "domega"   : np.load("simulations/transfer_function/domega_T.npy"),
                       "dt"       : np.load("simulations/transfer_function/dt.npy"),
                       "sigma_antenna"   : np.load("simulations/transfer_function/sigma_antenna.npy"),
                       "A_antenna"       : np.load("simulations/transfer_function/A_antenna.npy"),                       
                       "dt_s"    : np.load("simulations/transfer_function/dt_s.npy"),
                       #"dt_simulation"    : np.load("simulations/transfer_function/dt000000.npy"),
                       "omega_max": np.load("simulations/transfer_function/omega_max_T.npy"),
                       "omega_min": np.load("simulations/transfer_function/omega_min_T.npy")}

    #Plot F(omega)
     
    omega_c = np.load(f"""simulations/dispersion_relation/omega_cutoff.npy""")

    cm = 1/2.54
    SMALL_SIZE = 8
    
    fig, ax1 = plt.subplots()
    #fig.set_figheight(1.9*cm)
    #fig.set_figwidth(6.4*cm)
    
    fig.set_figheight(2*cm)
    fig.set_figwidth(2.67*cm)
    
    
    color = c["blue"]
    ax1.set_xlabel("$\omega/(2\pi)$ [GHz]",size=SMALL_SIZE,labelpad=0)
    ax1.set_ylabel("$|f(\omega)| $ [A$\cdot \mu$m$^{-1}$]",color=color,size=SMALL_SIZE,labelpad=2)
    
    ax1.vlines(omega_c/(2*np.pi*1e9), -0.1, 0.7, linestyle="dashed",color=c["purple"],linewidth=1)
    ax1.plot(transfer_function["omega"]/(2*np.pi*1e9) ,np.abs(1e-6*transfer_function["F(omega)"]),color=c["blue"],label="abs")
    ax1.set_ylim(-0.1,0.7)
    ax1.set_xlim(6,7)
    #ax1.set_xticks(fontsize=SMALL_SIZE)
    ax1.tick_params(axis='x',direction='in')
 
    ax1.tick_params(axis='y', labelcolor=color,direction='in')
    #ax1.yaxis.grid(True,linestyle="dotted")
    #ax1.xaxis.grid(True,which="both",linestyle="dotted")
    plt.xticks(fontsize=SMALL_SIZE)
    

    color = c["orange"]
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(transfer_function["omega"]/(2*np.pi*1e9) ,np.angle(1e-6*transfer_function["F(omega)"])/np.pi,color=color,label="$\phi/\pi$")
    ax2.set_ylim(-1.15,0.35)
    ax2.set_yticks([-1,0])
    ax2.tick_params(axis='y', labelcolor=color,direction='in')
    #ax2.set_ylim(ax1.get_ylim())
    ax2.set_ylabel("$\\theta(\omega)/\pi$ ",color=color,size=SMALL_SIZE,rotation=270,labelpad=10)
   

    plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0)
  
    plt.savefig("plots/plots_article/transfer_function.pdf", bbox_inches = 'tight',transparent=True)
    plt.savefig("plots/plots_article/transfer_function.svg")    
    plt.show()



def plot_dispersion_rel_and_square_pulse():
    dispersion = {  "pol_6th"      : np.load("simulations/dispersion_relation/pol_fit_6th.npy") , 
                    "omega_cutoff" : np.load("simulations/dispersion_relation/omega_cutoff.npy"),
                    "k1"           : np.load("simulations/dispersion_relation/k1.npy"),
                    "k2"           : np.load("simulations/dispersion_relation/k2.npy"),
                    "omega1"       : np.load("simulations/dispersion_relation/omega1.npy"),
                    "omega2"       : np.load("simulations/dispersion_relation/omega2.npy"),
                    "dk"           : np.load("simulations/dispersion_relation/dk.npy"),
                    "domega"       : np.load("simulations/dispersion_relation/domega.npy"),
                    "degree_pol_fit" : np.load("simulations/dispersion_relation/degree_pol_fit.npy"), 
                    "N_k_fit"      : np.load("simulations/dispersion_relation/N_k_fit.npy"),
                    "N_omega_fit"  : np.load("simulations/dispersion_relation/N_omega_fit.npy"),
                    "k_min"        : np.load("simulations/dispersion_relation/k_min.npy"),
                    "k_max"        : np.load("simulations/dispersion_relation/k_max.npy"),
                    "omega_min"    : np.load("simulations/dispersion_relation/omega_min.npy"),
                    "omega_max"    : np.load("simulations/dispersion_relation/omega_max.npy"),
                    "N_omega"      : np.load("simulations/dispersion_relation/N_omega.npy")
                   }
    
    polcoeffs       = dispersion["pol_6th"]
    omega_c         = dispersion["omega_cutoff"]
    k1              = dispersion["k1"]
    k2              = dispersion["k2"]
    omega1          = dispersion["omega1"]
    omega2          = dispersion["omega2"]
    N_k_fit         = dispersion["N_k_fit"]
    N_omega_fit     = dispersion["N_omega_fit"]
    degree_pol_fit  = dispersion["degree_pol_fit"]
    domega          = dispersion["domega"]
    k_min           = dispersion["k_min"]
    k_max           = dispersion["k_max"]
    omega_min       = dispersion["omega_min"]
    omega_max       = dispersion["omega_max"]
    N_omega         = dispersion["N_omega"]
    
    Ck_list         = np.load("simulations/square_pulse/Ck_list.npy") 
    
    def disp_fit(k,polcoeffs,degree_pol_fit):
        d = degree_pol_fit
        s = 0
        for i in range(0,d+1,1):
            s+= polcoeffs[i]*k**(d-i)
        return s
    
    klist = np.linspace(k1,k2,N_k_fit+1)  
    
    dispersion_list = disp_fit(klist,polcoeffs,degree_pol_fit)

    
    #Plot the dispersion relation and its quadratic approximation
    
    
    #Plot the dispersion relation and its quadratic approximation with the legend inside
    
    cm = 1/2.54
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    
    fig = plt.figure(figsize=(2.55*cm, 2*cm))
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    
    plt.plot(klist/1e6,dispersion_list/(2*np.pi*1e9),color=c["orange"], label="pol. fit",linewidth=1.2)
    #plt.plot(klist/1e6,omega_c/(2*np.pi*1e9)+0.5*np.abs(Ck_list)*1e9,label="FT pulse",color=c['blue'])
    ck = omega_c/(2*np.pi*1e9)+0.85*np.abs(Ck_list)*1e9
    plt.fill_between(klist/1e6, omega_c/(2*np.pi*1e9), ck, alpha=0.5,color=c["blue"],linewidth=0)  
    plt.ylim(omega_c/(2*np.pi*1e9),7.75)
    plt.xlim(0,25)
    plt.tick_params(direction='in')
    plt.text(8,7.10,"  target \n pulse",ha='center',color=c["blue"])
    plt.ylabel("$\omega/(2\pi) $ [GHz]",labelpad=0)
    plt.xlabel("$k\,\, [\mu m^{-1}] $",labelpad=0)
    plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0)
    
    plt.savefig("plots/plots_article/dispersion_relation_+square_pulse.pdf", bbox_inches = 'tight',transparent=True,dpi=300)
    plt.savefig("plots/plots_article/dispersion_relation_+square_pulse.svg")
    plt.show()
 


    
def plot_backwards_evolved_square_pulse(waveguide,square_pulse_initial):
    # Backwards evolved pulse
    
    Amplitude_pulse = np.load("simulations/square_pulse/Amplitude_pulse.npy")
    Xsize = waveguide["Xsize"]
    Nx    = waveguide["Nx"]
    xlist = np.linspace(-Xsize/2,Xsize/2,Nx)
    x_f   = square_pulse_initial["df"]-5*1e-6
    sigma_f = square_pulse_initial["sigmaf"]
    
    initial_pulse = np.load("simulations/square_pulse/initial_pulse.npy")
    target_pulse  = np.load("simulations/square_pulse/target_pulse_analytical.npy")
    target_envelope = np.load("simulations/square_pulse/envelope_target_pulse_analytical.npy")
    
    
    cm = 1/2.54
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    
    #plt.figure(figsize=(8.6*cm, 4.3*cm))
    plt.figure(figsize=(7.15*cm, 2*cm))
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    
    
    plt.plot(xlist*1e6, initial_pulse, label="$t=0$",color=c['blue'],linewidth=1)
    plt.plot(xlist*1e6, target_pulse, label="$t=t_f$",color=c['green'],linewidth=1)
    plt.plot(xlist*1e6, target_envelope,color=c['green'], linestyle='dashed',linewidth=0.75)   
    
    plt.vlines(0,-0.007,0.007,color="black",linestyle="dashed",linewidth=0.5)
    #plt.vlines(x_f*1e6,-0.05,0.05,color="black",linestyle="dashed",linewidth=0.75)

    #plt.fill_between(xlist*1e6, 0, envelope,alpha=0.4,color=c["blue"],linewidth=0)

    #plt.hlines(0.003,(x_f -sigma_f)*1e6,(x_f-4*sigma_f)*1e6,color="black",linewidth=0.5)
    #plt.hlines(0.003,(x_f +sigma_f)*1e6,(x_f+4*sigma_f)*1e6,color="black",linewidth=0.5)
    plt.ylabel("$m_T(x,t)/M_s$",labelpad=-3)
    plt.xlabel("$x\,\, [\mu m] $",labelpad=0)
    plt.tick_params(direction='in')
    #plt.yticks([-0.05,0,0.05])
    plt.ylim(-0.005,0.005)
    plt.xlim(-30,10)
    
    #plt.text(0,0.026,"$2\sqrt{2} \sigma_f$",fontsize=SMALL_SIZE)
    #plt.text(6.4,-0.065,"$x_f$",fontsize=SMALL_SIZE)
    plt.text(1,0.003,"$m_T(x,t_f)$",fontsize=SMALL_SIZE,color= c["green"])
    plt.text(-22,0.003,"$m_T(x,0)$",fontsize=SMALL_SIZE,color= c["blue"])
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
   
    plt.savefig("plots/plots_article/initial+target_analytical.pdf", bbox_inches = 'tight',transparent=True)
    plt.savefig("plots/plots_article/initial+target_analytical.svg")    
    plt.show()
    
def plot_evolved_square(waveguide,square_pulse_initial):
     
    Xsize = waveguide["Xsize"]

    #xlist = np.linspace(-Xsize/2,Xsize/2,Nx)
    
    m_pulse = np.load("simulations/square_pulse/m(x,t)_alone.npy")
    Treal   = np.load("simulations/square_pulse/T.npy")
    t_max_fidelity = np.load("simulations/square_pulse/time_max_fidelity.npy")
   
    cm = 1/2.54
    SMALL_SIZE = 8
 
    #Plot the dispersion relation and its quadratic approximation
    fig =  plt.figure(figsize=(6*cm, 3.7*cm))
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    


    extent = [- 1e6*Xsize/2, 1e6*Xsize/2, 0, 1e9*Treal ]
    plt.imshow(np.transpose(np.abs(m_pulse)**2), extent=extent, aspect='auto', origin='lower', cmap="viridis",vmin=0,vmax=5*1e-7)
    plt.ylabel("t [ns]",labelpad=0)
    plt.xlabel("x [$\mu$m]",labelpad=0)
    plt.xlim(-25,25)
    plt.ylim(0,140)
    plt.tick_params(direction='in',color="white")
    #plt.ylim(t_origin*1e9,50)
    plt.vlines(0,0,1e9*Treal,color="white", linestyle="dashed",linewidth=0.8)    
    plt.hlines((t_max_fidelity)*1e9,-30,30,color=c["green"], linestyle="dashed",linewidth=0.8)    
    cbaxes = fig.add_axes([0.9, 0.3, 0.03, 0.6]) 
    cbar = plt.colorbar(aspect=20,cax=cbaxes)
    cbar.set_label('$|\\tilde{m}_z(x,0,0,t)/M_s|^2$', rotation=90, labelpad=-28, color='white')
    cbar.outline.set_edgecolor('white')
    plt.tick_params(direction='in',color="white")
    #cbar.outline.set_linewidth(1)
    cbaxes.tick_params(axis='both', colors='white')
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig("plots/plots_article/evolution_square.pdf", bbox_inches = 'tight',transparent=True,dpi=300)
    plt.savefig("plots/plots_article/evolution_square.svg", bbox_inches = 'tight')
    plt.show()
    

    

def plot_driving_square_freq(waveguide,dispersion):   
    
    sigma_a = waveguide["sigma_antenna"]
    A_a     = waveguide["A_antenna"]
    Ysize   = waveguide["Ysize"]
    By      = waveguide["By"]
    
    omega_c = dispersion["omega_cutoff"]
    
    driving={          "omega"    :   np.load("simulations/pulse_generation_square/V(omega)_freqs.npy"), 
                       "V_spectrum":  np.load("simulations/pulse_generation_square/V(omega)_values.npy"),
                       "t"        :   np.load("simulations/pulse_generation_square/V(t)_time.npy"),
                       "V_time"   :   np.load("simulations/pulse_generation_square/V(t)_values.npy"),
                       "domega"   :   np.load("simulations/pulse_generation_square/domega.npy"),
                       "dt_driving":  np.load("simulations/pulse_generation_square/dt_driving.npy"),
                       "T_driving":   np.load("simulations/pulse_generation_square/T_driving.npy"),
                       "T_driving_initial":   np.load("simulations/pulse_generation_square/T_driving_initial.npy")}
    
    driving_new_units = A_a/(Ysize*By)*driving["V_spectrum"]
    driving_angle = np.angle(driving_new_units)
    driving_abs   = np.abs(driving_new_units) 

    #Plot V(omega)

    cm = 1/2.54
    SMALL_SIZE = 8
    
    fig, ax1 = plt.subplots()
    fig.set_figheight(2*cm)
    fig.set_figwidth(2.55*cm)
    
    color = c["blue"]
    ax1.set_xlabel("$\omega/(2\pi)$ [GHz]",size=SMALL_SIZE,labelpad=0)
    ax1.set_ylabel("$|V(\omega)| $ [ps]",color=color,size=SMALL_SIZE,labelpad=-0.6)
    
    
    #ax1.vlines(omega_c/(2*np.pi*1e9), -0.01, 2.5, linestyle="dashed",color=c["purple"],linewidth=1)
    ax1.plot(driving["omega"]/(2*np.pi*1e9) , 1e12*driving_abs ,color=c["blue"],label="abs")
    ax1.set_ylim(-0.01,0.65)
    ax1.set_xlim(6.18,7.65)
    ax1.set_yticks([0,0.3,0.6])
    ax1.tick_params(axis='x',direction="in")
    ax1.tick_params(axis='y', labelcolor=color,direction="in")
    #ax1.yaxis.grid(True,linestyle="dotted")
    #ax1.xaxis.grid(True,which="both",linestyle="dotted")
    


    color = c["orange"]
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.scatter(driving["omega"]/(2*np.pi*1e9) , 1e9*(1/np.pi)*np.gradient(driving_angle, driving["domega"]),color=color,label="$\partial \\phi/\pi$",s=0.8)
    #ax2.scatter(driving["omega"]/(2*np.pi*1e9) , 1/(np.pi)*driving_angle,color=c["green"],label="$\\phi/\pi$",s=1)
    ax2.set_ylim(-23,3)
    ax2.tick_params(axis='y', labelcolor=color,direction="in")
    #ax2.set_ylim(ax1.get_ylim())
    ax2.set_ylabel("$\partial_{\omega}\\varphi(\omega)/\pi$ [ns]",color=color,size=SMALL_SIZE,rotation=270,labelpad=8)
    plt.xticks(fontsize=SMALL_SIZE)

    plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0)
  
    plt.savefig("plots/plots_article/driving_spectrum_square.pdf", bbox_inches = 'tight',transparent=True)
    plt.savefig("plots/plots_article/driving_spectrum_square.svg")    
    plt.show()
    
 

def plot_generated_square_pulse(waveguide,square_pulse_initial):
   
    Xsize      = waveguide["Xsize"]
    

    # Import the generated magnetization as a function of position x and time at y=z=0
    m_pulse_generated  = np.load("simulations/pulse_generation_square/m(x,t)_alone.npy")
    dt      = np.load("simulations/pulse_generation_square/dt.npy")  
    t0      = square_pulse_initial["t0"]
    Treal   = np.load("simulations/square_pulse/T.npy")
    t_max_fidelity = np.load("simulations/square_pulse/time_max_fidelity.npy")
    
    N_treal = np.shape(m_pulse_generated)[1]
    Treal   = dt*(N_treal-1)
    tlist_real = np.linspace(0,Treal,N_treal)    
    
    cm = 1/2.54
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    fig =  plt.figure(figsize=(7.15*cm, 3.75*cm))
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
  
    
    extent = [- 1e6*Xsize/2, 1e6*Xsize/2, 0, 1e9*Treal ]
    plt.imshow(np.transpose(np.abs(m_pulse_generated)**2), extent=extent, aspect='auto', origin='lower', cmap="viridis",vmin=0,vmax=1*1e-6)
    plt.ylabel("t [ns]",labelpad=0)
    plt.xlabel("x [$\mu$m]",labelpad=0)
    plt.xlim(-20,20)
    plt.ylim(40,100)
    plt.title("Generated pulse",size=8)
    plt.tick_params(direction='in',color="white")
    #plt.ylim(t_origin*1e9,50)
    plt.vlines(0,0,1e9*Treal,color="white", linestyle="dashed",linewidth=0.8)    
    plt.hlines((t_max_fidelity)*1e9,-30,30,color=c["green"], linestyle="dashed",linewidth=0.8)    
   # cbaxes = fig.add_axes([0.87, 0.1, 0.04, 0.6])
    #cbar = plt.colorbar(aspect=20,cax=cbaxes)
    cbar = plt.colorbar(aspect=18,pad=0.02)
   # cbar.set_label('$|\\tilde{m}_z(x,0,0,t)/M_s|^2$', rotation=90, labelpad=-28, color='white')
    cbar.set_label('$|m_z(x,0,0,t)/M_s|^2$', rotation=-90, labelpad=10, color='black')
    cbar.outline.set_edgecolor('black')
    plt.tick_params(direction='in',color="white")
    #cbar.outline.set_linewidth(1)
    #cbaxes.tick_params(axis='both', colors='white')
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig("plots/plots_article/generated_square.pdf", bbox_inches = 'tight', transparent="True",dpi=300)
    plt.savefig("plots/plots_article/generated_square.svg", bbox_inches = 'tight')
    plt.show()
        
 
def plot_comparison_square(waveguide):
    # Show m_z(x,t) at different times compared to the target state    

    target_pulse    = np.load("simulations/square_pulse/target_pulse_normalized.npy")
    target_envelope    = np.load("simulations/square_pulse/envelope_target_pulse_normalized.npy")
    generated_pulse = np.load("simulations/pulse_generation_square/m(x,time_max_fidelity)_normalized.npy")
    xlist_half      = np.load("simulations/square_pulse/xlist_half.npy")
    fidelity        = np.load("simulations/pulse_generation_square/fidelity_protocol_sigma=30nm.npy")
    Zsize = waveguide["Zsize"]
    
    cm = 1/2.54
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    #Plot the dispersion relation and its quadratic approximation
    plt.figure(figsize=(6.3*cm, 2*cm))
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=7)    # legend fontsize
 
    plt.plot(1e6*xlist_half,-generated_pulse*np.sqrt(Zsize),label="generated",color=c["green"])        
    plt.plot(1e6*xlist_half,target_pulse*np.sqrt(Zsize),label="target",color=c["blue"],linestyle="dashed")    
    env = target_envelope*np.sqrt(Zsize)
    plt.fill_between(1e6*xlist_half, - env, env, alpha=0.2,color=c["blue"], linewidth=0.0)    
    plt.ylim(-0.38,0.38)
    plt.yticks([-0.3,0,0.3])
    plt.xlim(8,12)
    plt.tick_params(direction='in')
    plt.text(11.9,0.27,f"""Fidelity={fidelity:.3f}""",ha='right', fontsize=7)
    plt.legend(loc="upper left",frameon=False, labelspacing=0, borderpad=-0.2)
    #plt.legend(loc="upper left",frameon=False, labelspacing=5.5, borderpad=-0.2)
    plt.ylabel("$m_z(x,0,0,t_f)\sqrt{L_z}/N$",labelpad=-2)
    plt.xlabel("x [$\mu$m]",labelpad=0)
    
    plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0)
    
    plt.savefig("plots/plots_article/comparison_square.pdf", bbox_inches = 'tight',transparent=True)
    plt.savefig("plots/plots_article/comparison_square.svg")    
    plt.show()    
    
def plot_sketch(waveguide):
    
    Xsize      = waveguide["Xsize"]
    Nx         = waveguide["Nx"]
    Cx         = Xsize/Nx
    
    xlist = np.linspace(-Xsize/2,Xsize/2,Nx)
    xlist_pos = np.linspace(-2,20,int(Nx/2))
    sigma_g =1.5*1e-6
    gaussian = np.exp( -(xlist+30*1e-6)**2/(2*sigma_g**2) )
    lambda_poisson = 2
    poisson = 1/60*(lambda_poisson**xlist_pos*5**xlist_pos/scipy.special.factorial(xlist_pos + 1))*np.exp( -lambda_poisson)
    poisson_2 = np.zeros(int(Nx/2))
    poisson_right = np.append(poisson_2,poisson)
    #Plot target pulse
    
    cm = 1/2.54
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    #Plot the dispersion relation and its quadratic approximation
    plt.figure(figsize=(8.6*cm, 2.3*cm))
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
 
    plt.plot(xlist*1e6 ,gaussian,color=c["green"],label="$t= \\tau $")
    plt.plot(xlist*1e6 ,poisson_right,color=c["green"],label="$t= \\tau_0 $",linestyle="dashed")
    plt.ylim(0,1.2)
    plt.xticks([]),plt.yticks([])
    #plt.vlines(0, -0.01,1.21, color=c["blue"],linestyle="dashed",linewidth=1)
    #plt.legend(loc="upper left")
    plt.ylabel("$m(x,t)$" ,labelpad=2)
    plt.xlabel("$x$ ",labelpad=2)
    
    plt.subplots_adjust(left=0.0, right=1, top=1, bottom=0.0)
    
    plt.savefig("plots/plots_article/sketch_target.pdf", bbox_inches = 'tight',transparent=True)
    plt.savefig("plots/plots_article/sketch_target.svg")    
    plt.show()
    
    #Plot generated pulse
    x2 = np.linspace(-40,40,Nx)
    gaussian_modified = 0.2*np.exp( -(x2-30)**2/(2*2**2) *np.log( np.abs(0.2*(x2-30)) )) 
    
    plt.figure(figsize=(8.6*cm, 2.3*cm))
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
 
    plt.plot(xlist*1e6 ,gaussian,color=c["blue"],label="$t= \\tau $")
    plt.plot(xlist*1e6 ,gaussian_modified,color=c["blue"],label="$t= \\tau_0 $")
    plt.ylim(0,1.2)
    plt.xticks([]),plt.yticks([])
    #plt.vlines(0, -0.01,1.21, color=c["blue"],linestyle="dashed",linewidth=1)
    #plt.legend(loc="upper left")
    plt.ylabel("$m(x,\\tau)$" ,labelpad=2)
    plt.xlabel("$x$ ",labelpad=2)
    
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    
    plt.savefig("plots/plots_article/sketch_generated.pdf", bbox_inches = 'tight',transparent=True)
    plt.savefig("plots/plots_article/sketch_generated.svg")    
    plt.show()
    
    
    x3 = np.linspace(-8,12,Nx)
    gaussian_modified_2 = np.exp( -(x3-np.sin(3*x3))**2/(2*2**2) ) 
    
    
    plt.figure(figsize=(4*cm, 2*cm))
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
 
    plt.plot(x3 ,gaussian_modified_2,color=c["purple"])
    plt.ylim(0,1.2)
    plt.xticks([]),plt.yticks([])
    #plt.vlines(-8, -0.01,1.21, color=c["purple"],linestyle="dashed",linewidth=0.7)
    #plt.vlines(12, -0.01,1.21, color=c["purple"],linestyle="dashed",linewidth=0.7)    
    #plt.legend(loc="upper left")
    plt.ylabel("$m(x_0,t)$" ,labelpad=2)
    plt.xlabel("$t$ ",labelpad=2)
    
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    
    plt.savefig("plots/plots_article/sketch_time_recording.pdf", bbox_inches = 'tight',transparent=True)
    plt.savefig("plots/plots_article/sketch_time_recording.svg")    
    plt.show()