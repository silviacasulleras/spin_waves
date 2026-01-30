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

#n=4
#colors = plt.cm.viridis(np.linspace(0,1,n))
c = {"blue":'#0072BD', "orange": '#D95319', "green":"#77AC30", "yellow": '#EDB120', "purple":'#7E2F8E', "red":"#A2142F", "light-blue":"4DBEEE" }
   

#from modules.postprocessing import show_m_comp, show_m_vs_z, show_m_vs_y, show_m_abs, show_beff, read_mumax3_ovffiles_custom, show_mz_vs_x, extractdispersion, correctdispersion, find_polynomial_fit, FT_function, FT_function_2D

from modules.mumax3_functions import create_mumax_script, read_npy_files

def define_parameters_sim(waveguide):
    # Parameters of the simulation of the dispersion relation
    T     = 35e-9        # simulation time for determining the dispersion relation (longer -> better frequency resolution)
    fmax  = 20e9         # maximum frequency (in Hz) of the sinc pulse
    dt    = 1/(2*fmax)   # the sample time is dt = 0.025 ns
    N_t   = int(T/dt)
    


    A_antenna     = waveguide["A_antenna"]         # Amplitude of the field created by the antenna 
    sigma_antenna = waveguide["sigma_antenna"]     # Spatial extension of the antenna
    
    Xsize = waveguide["Xsize"]
    dk    = waveguide["dk"]
    N_k   = waveguide["N_k"]
    k_min = waveguide["k_min"]
    k_max    = waveguide["k_max"]
    
    domega    = 2*np.pi/T
    omega_min = -2*np.pi/(2*dt)
    omega_max = 2*np.pi/(2*dt)
    N_omega   = N_t
    
    omega1  = 6*2*np.pi*1e9
    omega2  = 8*2*np.pi*1e9
        
    k1 = 0*1e6
    k2 = 25*1e6
    
    degree_pol_fit = 6
    

    return {"T":T, "fmax":fmax, "dt":dt,"N_t":N_t, "k_min":k_min, "k_max":k_max, "k1":k1, "k2":k2 , 
            "omega_min":omega_min, "omega_max":omega_max, "N_omega":N_omega,
            "omega1":omega1, "omega2":omega2, 'dk':dk,
            "N_k":N_k, "domega":domega, "N_omega":N_omega, "A_antenna":A_antenna,
            "sigma_antenna": sigma_antenna, "degree_pol_fit":degree_pol_fit}

def simulate_dispersion_relation(waveguide):
    
    params = define_parameters_sim(waveguide)
    
    T    = params["T"]
    fmax = params["fmax"]
    dt   = params["dt"]
    A_antenna     = params["A_antenna"]
    sigma_antenna = params["sigma_antenna"]    
    
    #Define world size
    Xsize = waveguide["Xsize"]
    Ysize = waveguide["Ysize"]
    Zsize = waveguide["Zsize"]

    #Define number of cells
    Nx = waveguide["Nx"]
    Ny = waveguide["Ny"]
    Nz = waveguide["Nz"]

    # MATERIAL/SYSTEM PARAMETERS
    By    = waveguide["By"]      # Bias field along the x direction
    Aex   = waveguide["Aex"]     # exchange constant
    Ms    = waveguide["Ms"]    # saturation magnetization
    alpha = waveguide["alpha"]     # damping parameter
    gamma = waveguide["gamma"]    # gyromagnetic ratio
    mu0   = waveguide["mu0"]   # vacuum permeability

    #Define CellSize
    Cx=Xsize/Nx
    Cy=Ysize/Ny
    Cz=Zsize/Nz
    
  
    
    #_________________________________________________________________________________________________________
    #Simulation of the dispersion relation of the spin waves
    #_________________________________________________________________________________________________________
    
    script = f"""
Maxerr = 1e-8

//Define resolution
Nx:={Nx}
Ny:={Ny}
Nz:={Nz}

//Define world size
Xsize:={Xsize}
Ysize:={Ysize}
Zsize:={Zsize}

//Define CellSize
Cx:=Xsize/Nx
Cy:=Ysize/Ny
Cz:=Zsize/Nz

//Set GridSize and CellSize automatically
SetGridSize(Nx, Ny, Nz)
SetCellSize(Cx, Cy, Cz)

//Set material parameters

Msat = {Ms}          //Saturation magnetization
Aex  = {Aex}         //Exchange constant
alpha = {alpha}      //global alpha
m = uniform(1,0,0)

number_of_steps:=90
width_of_step:=20e-9

for i:=1; i<number_of_steps; i++{{
	//Define the regions in which the damping should be increased
	DefRegion( i, cuboid(width_of_step, Ysize, Zsize).Transl(-Xsize*0.5+(i-1)*width_of_step,0,0) )
	DefRegion( i+number_of_steps, cuboid(width_of_step, Ysize, Zsize).Transl(Xsize*0.5-(i-1)*width_of_step,0,0) )
	
	//The damping should decrease exponentially from a start value alpha_0 to a selected value alpha_end
	//alpha(region)=alpha_0*exp(-beta*x)
	alpha_end:={alpha}
	alpha_0:=0.5
	beta:=-log(alpha_end/alpha_0)/number_of_steps
	
	//Define the damping values in each region
	alpha.setRegion(i,alpha_0*exp(-beta*i))
	alpha.setRegion(i+number_of_steps,alpha_0*exp(-beta*i))
	Msat.setRegion(i,{Ms})
	Msat.setRegion(i+number_of_steps,{Ms})
	Aex.setRegion(i,{Aex})
	Aex.setRegion(i+number_of_steps,{Aex})
	m.setregion(i,uniform(0,1,0))
	m.setregion(i+number_of_steps,uniform(0,1,0))
}}

snapshot(regions)
snapshot(alpha)
snapshot(Aex)
snapshot(Msat)

B_ext = vector(0,{By},0) //Sets a external field

m.LoadFile("./ground.ovf")

//mask for excitation field
//An antenna field is implemented via a so called VectorMask.
//The command NewVectorMask(x,y,z) sets the size of the mask
CPW_field := newVectorMask(Nx, 1, 1)
//In this case the antenna field is only one dimensional, since we only have a dependency on the x-position of the antenna field.

for i:=0; i<Nx; i++{{
    x := -{Xsize/2}+{Cx}*i    
    Bz_antenna := {A_antenna/(np.sqrt(2*np.pi)*sigma_antenna)}*exp(-pow(x,2)/(2*pow({sigma_antenna},2)))
    CPW_field.setVector(i, 0, 0, vector(0, 0, Bz_antenna))
}}

//Now only the frequency of the excitation has to be defined.
frequency:=10e9
//Furthermore, we have to add the antenna field to the external field.
//Note that you can add a time modulation of the field after the comma.
B_ext.Add(CPW_field, sinc(2*pi*frequency*t)) //modulation of z-component


Total_time:={T}
tableadd(B_ext)
tableadd(E_total)
MaxFre:={fmax}
tableautosave(1/(2*MaxFre))
mz1:= Croplayer(m.Comp(2),0)
mz2:= Croplayer(m.Comp(2),1)
mz3:= Croplayer(m.Comp(2),2)
mz4:= Croplayer(m.Comp(2),3)
mz5:= Croplayer(m.Comp(2),4)
mzgeneral:= m.Comp(2)

//AutoSave(mz4,1/(2*MaxFre))

AutoSave(mz5,1/(2*MaxFre))

save(mz4)
run(Total_time)


"""

    create_mumax_script(script,"simulations/dispersion_relation/","dispersion")
    print("Script dispersion.txt created")
    

def extract_dispersion_relation(waveguide):
    
    params = define_parameters_sim(waveguide)
    
    T    = params["T"]
    fmax = params["fmax"]
    dt   = params["dt"]
    N_t    = params["N_t"]
    
    #World size
    Xsize = waveguide["Xsize"]
    Ysize = waveguide["Ysize"]
    Zsize = waveguide["Zsize"]

    #Number of cells
    Nx = waveguide["Nx"]
    Ny = waveguide["Ny"]
    Nz = waveguide["Nz"]
    
    #Define CellSize
    Cx=Xsize/Nx
    Cy=Ysize/Ny
    Cz=Zsize/Nz

    dk = params["dk"]
    N_k = params["N_k"]
    k_min = params["k_min"]
    k_max = params["k_max"]
    
    domega = params["domega"]
    omega_min = params["omega_min"]
    omega_max = params["omega_max"]
    N_omega = params["N_omega"] 
  
    # Time and position 
    #xlist = np.arange(-Xsize/2,Xsize/2+Cx,Cx)   # list of positions along the wire
    xlist = np.linspace(-Xsize/2, Xsize/2, Nx+1, endpoint=True) 

    #tlist = np.arange(0,T,dt)          # list of time steps


    print("Importing files from dispersion_relation.out")
    
    fields = read_npy_files("m_z_yrange4_zrange4", "simulations/dispersion_relation/dispersion.out") #The last simulations of the square and chirped pulse have been done using this
   # fields = read_npy_files("m", "simulations/dispersion_relation/dispersion.out")
    

    
    # Stack all snapshots of the magnetization on top of each other
    mz = np.stack([fields[key] for key in sorted(fields.keys())])  #we obtain an array m_z[t,z=0,y=0,x]
    
    # Select the z component of m as a function of position x and time at y=z=0
    mz0 = mz[:,0,0,0, :]
    mz0 = np.transpose(mz0, (1,0)) #Now we have m_z[x,t]

    # Show the intensity plot of m(x,t)
    plt.figure(figsize=(10, 6))
    extent = [- 1e6*Xsize/2, 1e6*Xsize/2, 0, 1e9*T]  # extent of k values and frequencies

    
    #plt.imshow(np.transpose(np.abs(mz0)**2), extent=extent, aspect='auto', origin='lower', cmap="viridis",vmin=0.000,vmax=0.0005)
    plt.imshow(np.transpose(np.abs(mz0)**2), extent=extent, aspect='auto', origin='lower', cmap="viridis",vmin=0.000,vmax=0.002)
    plt.ylabel("T [ns]")
    plt.xlabel("x [$\mu$m]")
    plt.ylim(0,1)
    plt.hlines(dt*1e9, -2, 2, color="white",linestyle='dashed')
    plt.xlim(-2,2)
    plt.colorbar()
    plt.savefig("plots/dispersion_relation/mz_vs_x,t_v2.pdf",dpi=300)
    plt.show()


    # Apply the two dimensional FFT
    mz0_fft = np.fft.fft2(mz0)
    mz0_fft = np.fft.fftshift(mz0_fft)

    # Show the intensity plot of the 2D FFT in logarithmic scale 
    plt.figure(figsize=(10, 6))
    extent = [ 1e-6*k_min, 1e-6*k_max, 1e-9*omega_min/(2*np.pi),  1e-9*omega_max/(2*np.pi)]  # extent of k values and frequencies

    

    # Show the intensity plot of the 2D FFT in logarithmic scale 
    plt.figure(figsize=(10, 6))
    plt.imshow(np.log10(np.transpose(np.abs(mz0_fft)**2,(1,0))), extent=extent, aspect='auto', origin='lower', cmap="viridis", vmin=0, vmax=6)
    plt.ylabel("$\omega/(2\pi) $ (GHz)")
    plt.xlabel("$k\,\, [\mu m^{-1}] $")
    cbar=plt.colorbar()
    cbar.set_label('log(Intensity)', rotation=270, labelpad=10)
    plt.ylim(6,8)
    plt.xlim(-45,45)
    plt.savefig("plots/dispersion_relation/dispersion_v2.pdf",dpi=300)
    #plt.hlines(6.5,-25,25)
    plt.show()
    
    
    np.save("simulations/dispersion_relation/m(x,t)_v2", mz0)
    np.save("simulations/dispersion_relation/m(k,omega)_v2", mz0_fft) 
    
    return {"m_xt":mz0, "m_komega":mz0_fft}



# DISPERSION RELATION - POLYNOMIAL FIT

#Extract the points of maximum intensity from the spectrum
def extract_maxima(params,waveguide,dispersion_simulation):
    mz0_fft = dispersion_simulation["m_komega"]
    
    params = define_parameters_sim(waveguide)
    
    T    = params["T"]
    fmax = params["fmax"]
    dt   = params["dt"]
    N_t    = params["N_t"]
    
    #World size
    Xsize = waveguide["Xsize"]
    Ysize = waveguide["Ysize"]
    Zsize = waveguide["Zsize"]

    #Number of cells
    Nx = waveguide["Nx"]
    Ny = waveguide["Ny"]
    Nz = waveguide["Nz"]
    
    #Define CellSize
    Cx=Xsize/Nx
    Cy=Ysize/Ny
    Cz=Zsize/Nz

    dk = params["dk"]
    N_k = params["N_k"]
    k_min = params["k_min"]
    k_max = params["k_max"]
    
    domega = params["domega"]
    omega_min = params["omega_min"]
    omega_max = params["omega_max"]
    N_omega = params["N_omega"]
    
    
    # Define smaller range of frequencies and ks
    k1 = params["k1"]
    k2 = params["k2"]  

    omega1 = params["omega1"]
    omega2 = params["omega2"]
    
    klist = np.linspace(k_min, k_max, N_k+1, endpoint=True) 
    omegalist = np.linspace(omega_min, omega_max, N_omega+1, endpoint=True) 

    
    def find_nearest(a, a0):
        #"Element in nd array `a` closest to the scalar value `a0`"
        idx = np.abs(a - a0).argmin()
        return {"value":a.flat[idx],"index":idx}
    
    k1_in_list = find_nearest(klist, k1)["value"]
    k2_in_list = find_nearest(klist, k2)["value"]
    
    ik1 = find_nearest(klist, k1)["index"]
    ik2 = find_nearest(klist, k2)["index"]

    
    omega1_in_list = find_nearest(omegalist,omega1)["value"]
    omega2_in_list = find_nearest(omegalist,omega2)["value"]

    
    iomega1 = find_nearest(omegalist,omega1)["index"]
    iomega2 = find_nearest(omegalist,omega2)["index"]

    
    k_extracted=[]
    omega_extracted=[]
    
    for i in range(ik1,ik2):
        k = k_min+i*dk
        intensity =  np.log10(np.abs( mz0_fft[i,iomega1:iomega2] ) **2)
        i_freq = np.where(intensity == np.amax(intensity))[0][0]
        freq = omega1_in_list+i_freq*domega
        omega_extracted=np.append(omega_extracted,freq)
        k_extracted=np.append(k_extracted,k)
        
    # Show the intensity plot of the 2D FFT in logarithmic scale 
    plt.figure(figsize=(10, 6))
    plt.scatter(k_extracted/1e6,omega_extracted/(2*np.pi)/1e9,color='white')
    extent = [-1e-6*(2*np.pi)/(2*Cx), 1e-6*(2*np.pi)/(2*Cx), -1e-9*1/(2*dt),  1e-9*1/(2*dt)]  # extent of k values and frequencies
    plt.imshow(np.log10(np.transpose(np.abs(mz0_fft)**2,(1,0))), extent=extent, aspect='auto', origin='lower', cmap="viridis", vmin=0, vmax=8)
    plt.ylabel("$\omega/(2\pi) $ (GHz)")
    plt.xlabel("$k\,\, [\mu m^{-1}] $")
    plt.xlim(k1_in_list/1e6,k2_in_list/1e6)
    plt.ylim(omega1_in_list/(2*np.pi*1e9),omega2_in_list/(2*np.pi*1e9))
    plt.show()
    
    

    y = np.array(np.abs(mz0_fft[ik1:ik2,iomega1:iomega2])**2)
    
    np.savetxt("simulations/dispersion_relation/spectrum.CSV",y,delimiter=',')
    np.savetxt("simulations/dispersion_relation/omega.CSV",omega_extracted,delimiter=',')
    np.savetxt("simulations/dispersion_relation/k.CSV",k_extracted,delimiter=',')
    
    my_dict = {"k":k_extracted,"omega":omega_extracted}
    return my_dict

def fit_dispersion_relation(waveguide,dispersion_simulation):
    params = define_parameters_sim(waveguide)
    polcoeffs=find_polynomial_fit(params,waveguide,dispersion_simulation)  
    
    return polcoeffs


def disp_fit(k,polcoeffs,degree_pol_fit):
    d = degree_pol_fit
    s = 0
    for i in range(0,d+1,1):
        s+= polcoeffs[i]*k**(d-i)
    return s

def find_polynomial_fit(params,waveguide,dispersion_simulation):
    mz0_fft = dispersion_simulation["m_komega"]
    degree_pol_fit = params["degree_pol_fit"]
    k_min = params["k_min"]
    k_max = params["k_max"]
    N_k = params["N_k"]
    N_omega = params["N_omega"]
    
    #Extract the points of maximum intensity from the spectrum
    dispersion1=extract_maxima(params,waveguide,dispersion_simulation)


    #Remove the points that don't satisfy the monotony condition
    #dispersion2=correctdispersion(params,waveguide,dispersion_simulation,dispersion1)

    dispersion_corrected_k=dispersion1["k"]
    dispersion_corrected_w=dispersion1["omega"]

    #Find the sixth-order polynomial aproximation of the dispersion relation
    polcoeffs = np.polyfit(dispersion_corrected_k, dispersion_corrected_w, degree_pol_fit)
    omega_c   = polcoeffs[degree_pol_fit]
    omega_fit = disp_fit(dispersion_corrected_k,polcoeffs,degree_pol_fit)
    
    dt     = params["dt"]
    k_min  = params["k_min"]
    k_max  = params["k_max"]
    omega_min  = params["omega_min"]
    omega_max  = params["omega_max"]
    mz0_fft = dispersion_simulation["m_komega"]
    Xsize = waveguide["Xsize"]
    Nx = waveguide["Nx"]
    Cx=Xsize/Nx
    k1_fit = dispersion_corrected_k[0]
    k2_fit = dispersion_corrected_k[-1]
    dk = params["dk"] 
    N_k_fit = len(dispersion_corrected_k)
    omega1_fit  = dispersion_corrected_w[0]
    omega2_fit  = dispersion_corrected_w[-1]
    domega  = params["domega"]
    N_omega_fit = len(dispersion_corrected_w)
    

    #Loretnzian_freqs = np.fromfile("simulations/dispersion_relation/dispersion_maxima_Lorentzian",np.complex64)
    #Loretnzian_ks    = np.fromfile("simulations/dispersion_relation/dispersion_ks_Lorentzian",np.complex64)

    extent = [ 1e-6*k_min, 1e-6*k_max, 1e-9*omega_min/(2*np.pi),  1e-9*omega_max/(2*np.pi)]  # extent of k values and frequencies
 
    cm = 1/2.54
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    #Plot the dispersion relation and its quadratic approximation
    plt.figure(figsize=(6.8*cm, 4*cm))
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
     
    #plt.hlines(omega_c/(2*np.pi*1e9),0, 25, color="white", linestyle= "dashed",linewidth=1)
    plt.imshow(np.log10(np.transpose(np.abs(mz0_fft)**2,(1,0))), extent=extent, aspect='auto', origin='lower', cmap="viridis", vmin=0, vmax=5)
    #plt.scatter(Loretnzian_ks/1e6,Loretnzian_freqs/(2*np.pi*1e9),label="Lorentzian fit",color="white",s=0.25,alpha=0.5)
    plt.plot(dispersion_corrected_k/1e6,omega_fit/(2*np.pi*1e9),label="pol. fit",color=c["orange"],linestyle="dashed",linewidth=1)
    plt.ylim(6,8)
    plt.xlim(0,25)
    
    plt.ylabel("$\omega/(2\pi) $ (GHz)",labelpad=0)
    plt.xlabel("$k\,\, [\mu m^{-1}] $",labelpad=0)

    cbar=plt.colorbar(aspect=12)
    cbar.set_label('log($|m_z(k,0,0,\omega)|^2$)', rotation=270, labelpad=15)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    
    plt.savefig("plots/plots_article/S1_dispersion.pdf", bbox_inches = 'tight',transparent=True)
    plt.savefig("plots/plots_article/S1_dispersion.svg")    
    plt.show()
    
    
    
    np.save("simulations/dispersion_relation/klist_maxima", dispersion_corrected_k)   
    np.save("simulations/dispersion_relation/omegalist_maxima", dispersion_corrected_w)   
    np.save("simulations/dispersion_relation/pol_fit_6th", polcoeffs)
    np.save("simulations/dispersion_relation/degree_pol_fit", degree_pol_fit)    
    np.save("simulations/dispersion_relation/omega_cutoff", omega_c)    
    np.save("simulations/dispersion_relation/k1", k1_fit)    
    np.save("simulations/dispersion_relation/k2", k2_fit)       
    np.save("simulations/dispersion_relation/omega1", omega1_fit)   
    np.save("simulations/dispersion_relation/omega2", omega2_fit) 
    np.save("simulations/dispersion_relation/dk", dk)      
    np.save("simulations/dispersion_relation/N_k_fit", N_k_fit)   
    np.save("simulations/dispersion_relation/domega", domega)
    np.save("simulations/dispersion_relation/N_omega_fit", N_omega_fit)   
    np.save("simulations/dispersion_relation/k_min", k_min)    
    np.save("simulations/dispersion_relation/k_max", k_max)        
    np.save("simulations/dispersion_relation/omega_min", omega_min)    
    np.save("simulations/dispersion_relation/omega_max", omega_max)     
    np.save("simulations/dispersion_relation/N_k", N_k)     
    np.save("simulations/dispersion_relation/N_omega", N_omega)    


    
    #Error in the polynomial fit with respect to the numerical dispersion
    error_1 = np.abs((dispersion_corrected_w-omega_fit)/dispersion_corrected_w)
    print("Dispersion relation error numerical")
    print(np.amax(error_1))
    
    return {"6th":polcoeffs, "omega_cutoff":omega_c,"k1":k1_fit,
            "k2":k2_fit,"omega1":omega1_fit, "omega2":omega2_fit, "dk":dk, "domega":domega}





def import_dispersion_relation(waveguide):
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
                    #"numerical_k"  : np.load("simulations/dispersion_relation/numerical_k.npy"),                    
                    "N_k_fit"      : np.load("simulations/dispersion_relation/N_k_fit.npy"),
                    #"numerical_omega"  : np.load("simulations/dispersion_relation/numerical_omega.npy"),
                    "N_omega_fit"  : np.load("simulations/dispersion_relation/N_omega_fit.npy"),
                    #"Loretnzian_freqs" : np.fromfile("simulations/dispersion_relation/dispersion_maxima_Lorentzian",np.complex64),
                    #"Loretnzian_ks"    : np.fromfile("simulations/dispersion_relation/dispersion_ks_Lorentzian",np.complex64),
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
    
    params = define_parameters_sim(waveguide)
    

    dk = params["dk"]
    N_k = params["N_k"]
    k_min = params["k_min"]
    k_max = params["k_max"]
    
    
    omega_min = params["omega_min"]
    omega_max = params["omega_max"]
    N_omega = params["N_omega"]
    
    klist = np.linspace(k1,k2,N_k_fit+1)  
    dispersion_list = disp_fit(klist,polcoeffs,degree_pol_fit)

    
    #Plot the dispersion relation and its quadratic approximation
    
    extent = [ 1e-6*k_min, 1e-6*k_max, 1e-9*omega_min/(2*np.pi),  1e-9*omega_max/(2*np.pi)]  # extent of k values and frequencies
 
    cm = 1/2.54
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    #Plot the dispersion relation and its quadratic approximation
    plt.figure(figsize=(6.8*cm, 4*cm))
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
     
    plt.imshow(np.log10(np.transpose(np.abs(mz0_fft)**2,(1,0))), extent=extent, aspect='auto', origin='lower', cmap="viridis", vmin=0, vmax=5)
    plt.plot(klist/1e6,dispersion_list/(2*np.pi*1e9),label="pol. fit",color=c["orange"],linestyle="dashed",linewidth=1)
    plt.ylim(6.1,8)
    plt.xlim(0,25)
    
    plt.ylabel("$\omega/(2\pi) $ (GHz)",labelpad=0)
    plt.xlabel("$k\,\, [\mu m^{-1}] $",labelpad=0)

    cbar=plt.colorbar(aspect=12)
    cbar.set_label('log($|m_z(k,0,0,\omega)|^2$)', rotation=270, labelpad=15)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    
    plt.savefig("plots/plots_article/S1_dispersion.pdf", bbox_inches = 'tight',transparent=True)
    plt.savefig("plots/plots_article/S1_dispersion.svg")    
    plt.savefig("plots/dispersion_relation/dispersion_relation+fit.pdf", bbox_inches = 'tight',transparent=True)
   
    plt.show()
    
    #print(polcoeffs)
    
    
    
    #Error in the polynomial fit with respect to the numerical Lorentzian
    #error_2 = np.abs((dispersion["numerical_omega"]-Loretnzian_freqs[1:])/dispersion["numerical_omega"])
    #error_2 = np.abs((dispersion["numerical_k"]-Loretnzian_ks[:-1])/dispersion["numerical_k"])
    #print("Dispersion relation error numerical")
    #print(np.amax(error_2))
    
    
    
    return dispersion





#Not needed functions

#Remove the points that don't satisfy the monotony condition
def correctdispersion(params,waveguide,dispersion_simulation,dispersion1):
    mz0_fft = dispersion_simulation["m_komega"]
    
    params = define_parameters_sim(waveguide)
    
    T    = params["T"]
    fmax = params["fmax"]
    dt   = params["dt"]
    
    #World size
    Xsize = waveguide["Xsize"]
    Ysize = waveguide["Ysize"]
    Zsize = waveguide["Zsize"]

    #Number of cells
    Nx = waveguide["Nx"]
    Ny = waveguide["Ny"]
    Nz = waveguide["Nz"]
    
    #Define CellSize
    Cx=Xsize/Nx
    Cy=Ysize/Ny
    Cz=Zsize/Nz

    dk = params["dk"]
    domega = params["domega"]

    k1 = params["k1"]
    k2 = params["k2"]  
    
    omega1 = params["omega1"]
    omega2 = params["omega2"]
    
    ik1 = round(( 2*np.pi/(2*Cx)+k1)/dk)
    ik2 = round(( 2*np.pi/(2*Cx)+k2)/dk)
    
    iomega1 = round( (-2*np.pi/(2*dt)+omega1)/domega )
    iomega2 = round ( (-2*np.pi/(2*dt)+omega2)/domega )

    dispersion=dispersion1["omega"]
    dispersionk = dispersion1["k"]
    dispersion_corrected=dispersion[0]
    k_corrected=dispersionk[0]
    
    for i in range(1,int(np.size(dispersionk)/2)-1) :
        if(dispersion[i]<dispersion[i-1] or abs(dispersion[i]-dispersion[i-1])<0.015*2*np.pi*1e9) :
            dispersion_corrected=np.append(dispersion_corrected,dispersion[i])
            k_corrected=np.append(k_corrected,dispersionk[i])
            
    for i in range(int(np.size(dispersionk)/2),int(np.size(dispersionk))) :
        if(dispersion[i]>dispersion[i-1] or abs(dispersion[i]-dispersion[i-1])<0.015*2*np.pi*1e9) :
            dispersion_corrected=np.append(dispersion_corrected,dispersion[i])
            k_corrected=np.append(k_corrected,dispersionk[i])        
    my_dict = {"k":k_corrected,"omega":dispersion_corrected}
    
    np.save("simulations/dispersion_relation/numerical_k", k_corrected)
    np.save("simulations/dispersion_relation/numerical_omega", dispersion_corrected)
    
        # Show the intensity plot of the corrected dispersion on top of the spectrum in logarithmic scale 
    plt.figure(figsize=(10, 6))
    extent = [-1e-6*(2*np.pi)/(2*Cx), 1e-6*(2*np.pi)/(2*Cx), -1e-9*1/(2*dt),  1e-9*1/(2*dt)]  # extent of k values and frequencies
    plt.imshow(np.log10(np.transpose(np.abs(mz0_fft)**2,(1,0))), extent=extent, aspect='auto', origin='lower', cmap="viridis", vmin=0, vmax=8)
    plt.scatter(k_corrected/1e6,dispersion_corrected/(2*np.pi)/1e9, color="white",s=6)
    plt.ylabel("$\omega/(2\pi) $ (GHz)")
    plt.xlabel("$k\,\, [\mu m^{-1}] $")
    cbar=plt.colorbar()
    cbar.set_label('log(Intensity)', rotation=270, labelpad=10)
    plt.ylim(2.5,4.5)
    plt.xlim(0,15)
    #plt.savefig("dispersion_fit_corrected.pdf")
    plt.show() 
    
    return my_dict



    