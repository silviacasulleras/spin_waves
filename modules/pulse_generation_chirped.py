"""
Created on Thu Feb  4 11:20:49 2021
Simulation of the evolution of a chirped spin wave in a rectangular waveguide 
@author: Silvia Casulleras
"""

import numpy as np
import matplotlib.pyplot as plt
from modules.mumax3_functions import create_mumax_script, read_npy_files
from modules.postprocessing import FT_function, IFT_function, IFT_function_pos

#n=4
#colors = plt.cm.viridis(np.linspace(0,1,n))
c = {"blue":'#0072BD', "orange": '#D95319', "green":"#77AC30", "yellow": '#EDB120', "purple":'#7E2F8E', "red":"#A2142F", "light-blue":"4DBEEE" }
   

def params_simulation_generation_pulse(waveguide):
    T           = 150e-9                                     # Total simulation time
    T_driving   = 150e-9                                     # Total simulation time
    #T_driving_initial = -10*1e-9
    T_driving_initial = 0
    dt          = 1e-11                                      # Time step for the snapshots of the magnetization 
    dt_driving  = 1e-12                                      # Time step for the snapshots of the magnetization 
    N_t         = round(T/dt)+1                              # Number of snapshots of the magnetization
    N_t_driving = round((T_driving-T_driving_initial)/dt_driving)+1                      # Number of times we update the driving
    
    A_antenna     = waveguide["A_antenna"]    # Amplitude of the field created by the antenna 
    sigma_antenna = waveguide["sigma_antenna"]   # Spatial extension of the antenna
       
    Xsize       = waveguide["Xsize"]
    Nx          = waveguide["Nx"]
    Cx          = Xsize/Nx
    dk          = 2*np.pi/Xsize
       
    number_of_steps  = 90
    width_of_step    = 20e-9
    Delta_x          = number_of_steps*width_of_step
    x_i              = -Xsize/2+Delta_x
    x_f              = Xsize/2-Delta_x
    i_i              = round((x_i+Xsize/2)/Cx)
    i_f              = round((x_f+Xsize/2)/Cx)

    return {"T":T, "T_driving":T_driving, "T_driving_initial":T_driving_initial, 
            "dt":dt,"dt_driving":dt_driving, "N_t":N_t, "N_t_driving":N_t_driving, 
            'dk':dk, "Delta_x":Delta_x,
            "x_i":x_i, "x_f":x_f,"i_i":i_i, "i_f":i_f, "A_antenna":A_antenna, "sigma_antenna": sigma_antenna}


def target_chirped_pulse_spectrum(waveguide,  ground_state, dispersion, evolved_pulse, transfer_func):
    
    omega_1_S = 6*2*np.pi*1e9
    omega_2_S = 8*2*np.pi*1e9
    N_omega_S = 2000
    domega_S  = (omega_2_S-omega_1_S)/N_omega_S
    
    Ny     = waveguide["Ny"]
    Xsize  = waveguide["Xsize"]
    Nx     = waveguide["Nx"]
    Cx     = Xsize/Nx
    Ms     = waveguide["Ms"]
    Zsize  = waveguide["Zsize"]
    Ny     = waveguide["Ny"]    
    dt     = evolved_pulse["dt"]   
    N_t    = evolved_pulse["N_t"]
    omega_c = dispersion["omega_cutoff"]
    
    #Import the magnetization of the chirped pulse in temporal domain 
    mz_t_alone          = evolved_pulse["m(x0,t)_alone"]
    mz_t                = evolved_pulse["m(x0,t)"]  
    #mz_ground_t         = [mz_ground] * N_t
    mz_ground_t         = evolved_pulse["m_ground(x0,t)"] 
    m_pulse_alone_t     = {"t": evolved_pulse["t"], "values": mz_t_alone, "dt": evolved_pulse["dt"] }
    #m_pulse_t           = {"t": evolved_pulse["t"], "values": mz_t, "dt": evolved_pulse["dt"] }     
    m_pulse_ground_t    = {"t": evolved_pulse["t"], "values": mz_ground_t, "dt": evolved_pulse["dt"] }       
    
    cm = 1/2.54
    plt.figure(figsize=(8*cm, 5*cm))
    plt.plot(1e9*m_pulse_alone_t["t"],m_pulse_alone_t["values"],c="b")
    plt.ylabel("$m_z(x_0,t)$")
    plt.xlabel("t [ns]")
    plt.ylim(-0.002,0.002)
    #plt.vlines(-x0*1e6,-0.04,0.04,color="green",linestyle="dashed")
    plt.savefig("plots/chirped_pulse/m_vs_t_x=0.pdf",bbox_inches='tight')
    plt.show()
    
    #Calculate the spectrum of the pulse (without the ground state)
    
    omega_list_S  =  np.linspace(omega_1_S,omega_2_S,N_omega_S+1,endpoint=True) 
    
    #Limits of frequency for plotting

    m_pulse_spectrum_alone = {"omega":omega_list_S,"values": FT_function(omega_list_S,m_pulse_alone_t)}
    

    plt.figure(figsize=(9*cm, 5*cm))
    plt.plot(m_pulse_spectrum_alone["omega"]/(2*np.pi*1e9) ,1e9*np.real(m_pulse_spectrum_alone["values"]),color="b",label="re")
    plt.plot(m_pulse_spectrum_alone["omega"]/(2*np.pi*1e9) ,1e9*np.imag(m_pulse_spectrum_alone["values"]),color="orange",label="im" )
    plt.legend()
    plt.ylabel("$m_z(x=x_0,\omega)$ [ns]")
    plt.xlabel("$\omega/(2\pi)$ [GHz]")
    plt.xlim(1.1*omega_1_S/(2*np.pi*1e9),0.78*omega_2_S/(2*np.pi*1e9))
    plt.savefig("plots/chirped_pulse/spectrum_pulse.pdf")
    plt.show()
    
    mz_t_inc_ground   = evolved_pulse["m(x0,t)"]
    m_pulse_t_inc_ground   = {"t": evolved_pulse["t"], "values": mz_t_inc_ground, "dt": evolved_pulse["dt"] }
    m_pulse_spectrum_inc_ground= {"omega":omega_list_S,"values": FT_function(omega_list_S,m_pulse_t_inc_ground)}
        

    return {"omega": omega_list_S, "values_spectrum_alone": m_pulse_spectrum_alone["values"], 
            "values_spectrum_inc_ground": m_pulse_spectrum_inc_ground["values"], "omega_1_S":omega_1_S, "omega_2_S":omega_2_S, "N_omega_S":N_omega_S}

def find_nearest(a, a0):
    #"Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return {"value":a.flat[idx],"index":idx}   

def determine_driving_chirped(waveguide,dispersion,transfer_func,pulse_spectrum):
    
    #Determine the driving that generates the target magnetization
    omega_list = transfer_func["omega"]
    domega_T   = transfer_func["domega_T"]
    Ms         = waveguide["Ms"]
    omega_c    = dispersion["omega_cutoff"]
    
    omega_1_S = pulse_spectrum["omega_1_S"]
    omega_2_S = pulse_spectrum["omega_2_S"]
    N_omega_S = pulse_spectrum["N_omega_S"]
    domega_S  = (omega_2_S-omega_1_S)/N_omega_S
    omega_list_S  =  np.linspace(omega_1_S,omega_2_S,N_omega_S+1,endpoint=True) 

    #Parameters of the simulation
    params_sim  = params_simulation_generation_pulse(waveguide)
    T           = params_sim["T"]
    dt          = params_sim["dt"]
    N_t         = params_sim["N_t"]  
    T_driving   = params_sim["T_driving"]
    T_driving_initial = params_sim["T_driving_initial"]
    dt_driving  = params_sim["dt_driving"]
    N_t_driving = params_sim["N_t_driving"]   
    
    #Interpolate transfer function
    omega_list_T = transfer_func["omega"]
    transfer_function_list = transfer_func["F(omega)"]
   
    transfer_function_list_S =  np.interp(omega_list_S, omega_list_T, transfer_function_list)

    V_spectrum_values =  Ms*pulse_spectrum["values_spectrum_alone"]/transfer_function_list_S
    V_spectrum = {"omega":omega_list_S, "values":V_spectrum_values, 'domega':domega_S}

    cm = 1/2.54
    plt.figure(figsize=(9*cm, 5*cm))
    plt.plot(V_spectrum["omega"]/(2*np.pi*1e9) ,1e9*np.real(V_spectrum["values"]),color="b",label="re")
    plt.plot(V_spectrum["omega"]/(2*np.pi*1e9) ,1e9*np.imag(V_spectrum["values"]),color="orange",label="im")
    plt.legend()
    plt.ylabel("$V(\omega)$ [ns]")
    plt.xlabel("$\omega/(2\pi)$ [GHz]")
    plt.xlim(1.1*omega_1_S/(2*np.pi*1e9),0.78*omega_2_S/(2*np.pi*1e9))
    plt.show()
    

    plt.figure(figsize=(9*cm, 5*cm))
    plt.plot(V_spectrum["omega"]/(2*np.pi*1e9) ,1e9*np.real(V_spectrum["values"]),color="b",label="re")
    plt.plot(V_spectrum["omega"]/(2*np.pi*1e9) ,1e9*np.imag(V_spectrum["values"]),color="orange",label="im")
    plt.legend()
    plt.ylabel("$V(\omega)$ [ns]")
    plt.xlabel("$\omega/(2\pi)$ [GHz]")
    plt.show()    

    
    tlist_driving = np.linspace(T_driving_initial,T_driving,N_t_driving,endpoint=True)
    driving_time  = IFT_function_pos(tlist_driving,V_spectrum)
    V_time        = {"t":tlist_driving, "values":driving_time ,"dt_driving":dt_driving ,
                         "T_driving":T_driving, "T_driving_initial":T_driving_initial}
    

    plt.figure(figsize=(9*cm, 5*cm))
    plt.plot(V_time["t"]*1e9 ,np.real(V_time["values"]),color="b",label="re")
    plt.plot(V_time["t"]*1e9 ,np.imag(V_time["values"]),color="orange",label="im")
    plt.legend()
    plt.ylabel("$V(t)$")
    plt.xlabel("$t$ [ns]")
    plt.savefig("plots/pulse_generation_chirped/V_time.pdf")    
    plt.show()


    #Export the values of the driving in time and spectral domain

    np.save("simulations/pulse_generation_chirped/V(omega)_freqs", V_spectrum["omega"])
    np.save("simulations/pulse_generation_chirped/V(omega)_values", V_spectrum["values"])
    np.save("simulations/pulse_generation_chirped/domega", V_spectrum["domega"])


    np.save("simulations/pulse_generation_chirped/V(t)_time", V_time["t"])
    np.save("simulations/pulse_generation_chirped/V(t)_values", V_time["values"])
    np.save("simulations/pulse_generation_chirped/T_driving",  V_time["T_driving"])    
    np.save("simulations/pulse_generation_chirped/T_driving_initial",  V_time["T_driving_initial"])
    np.save("simulations/pulse_generation_chirped/dt_driving", V_time["dt_driving"])

    
    return {"omega": omega_list, "V_spectrum": V_spectrum["values"], "time": V_time["t"], "V_time": V_time["values"] , 
            "dt_driving":dt_driving, "T_driving": T_driving, "T_driving_initial": T_driving_initial, "domega":domega_T}




def import_driving_chirped(waveguide):   
    
    sigma_a = waveguide["sigma_antenna"]
    A_a     = waveguide["A_antenna"]
    Lz      = waveguide["Zsize"]
    Ly      = waveguide["Ysize"]
    mu0     = waveguide["mu0"]

    
    driving={          "omega"    :   np.load("simulations/pulse_generation_chirped/V(omega)_freqs.npy"), 
                       "V_spectrum":  np.load("simulations/pulse_generation_chirped/V(omega)_values.npy"),
                       "t"        :   np.load("simulations/pulse_generation_chirped/V(t)_time.npy"),
                       "V_time"   :   np.load("simulations/pulse_generation_chirped/V(t)_values.npy"),
                       "domega"   :   np.load("simulations/pulse_generation_chirped/domega.npy"),
                       "dt_driving":  np.load("simulations/pulse_generation_chirped/dt_driving.npy"),
                       "T_driving":   np.load("simulations/pulse_generation_chirped/T_driving.npy"),
                       "T_driving_initial":   np.load("simulations/pulse_generation_chirped/T_driving_initial.npy")}
    
    #Show V(omega)
    cm = 1/2.54
    plt.figure(figsize=(9*cm, 5*cm))
    plt.plot(driving["omega"]/(2*np.pi*1e9) ,1e9*np.real(driving["V_spectrum"]),color=c["green"],label="re")
    plt.plot(driving["omega"]/(2*np.pi*1e9) ,1e9*np.imag(driving["V_spectrum"]),color=c["blue"],label="im")
    plt.legend()
    plt.ylabel("$\\tilde V(\omega)$ [ns]" )        
    plt.xlabel("$\omega/(2\pi)$ [GHz]")
    plt.xlim(6,8)
    plt.savefig("plots/pulse_generation_chirped/V(omega).pdf")
    plt.show()

    #Delta_t_driving = np.load(f"simulations/pulse_generation_chirped/Delta_t_driving.npy")  
    V_t = driving["V_time"] 
    Delta_t_driving = 66.9*1e-9
    np.save("simulations/pulse_generation_chirped/Delta_t_driving_0.995",Delta_t_driving)  
    
    dt_driving = driving['dt_driving']
    i_time_end_driving = int(  Delta_t_driving/ dt_driving)
    
    percentage_end_driving = sum(np.abs(V_t[:i_time_end_driving])**2) / sum(np.abs(V_t)**2)
    
    print("End of driving (%):")
    print( percentage_end_driving)
    
    #Show V(t) 
    plt.figure(figsize=(9*cm, 5*cm))
    plt.plot(driving["t"]*1e9 ,np.real(driving["V_time"]),color="b")
    plt.ylabel("$V(t)$")
    plt.xlabel("$t$ [ns]")
    plt.xlim(0,70)
    plt.vlines(Delta_t_driving*1e9,-0.001,0.001,linestyle='dashed',color='orange')
    plt.savefig("plots/pulse_generation_chirped/V_time_all.pdf")
    plt.show()
    
    #Show V(t) 
    plt.figure(figsize=(9*cm, 5*cm))
    plt.plot(driving["t"]*1e9 ,np.real(driving["V_time"])**2,color="g")
    plt.ylabel("$V^2(t)$")
    plt.xlabel("$t$ [ns]")
    plt.vlines(i_time_end_driving*dt_driving*1e9,0,0.001**2,linestyle='dashed',color='orange')
    #plt.vlines(Delta_t_driving*1e9,0,0.001**2,linestyle='dashed',color='orange')

    V_t_max = np.amax(np.real(driving["V_time"]))
    B_max   = A_a/(np.sqrt(2*np.pi)*sigma_a)*V_t_max
    
    print("Vt max:")
    print(V_t_max)
        

    V_squared_total = sum(V_t[:i_time_end_driving]**2)*dt_driving/Delta_t_driving
    
    print("V squared total:")
    print(V_squared_total)
    
    U_total =  A_a**2/(4*np.sqrt(np.pi)*mu0*sigma_a)*Lz*Ly*V_squared_total
    print(U_total)
    
    #Energy of the homogeneous field:
    By = waveguide["By"]
    Lx = waveguide["Xsize"
                   ]    
    U_0 = 1/(2*mu0)*Lx*Ly*Lz*By**2
    print("U_0: [fJ]")
    print(U_0*1e15)
    
    print("U_tot/U_0:")
    print(U_total/U_0)
    
    np.save(f"""simulations/pulse_generation_chirped/B_max_sigma={int(sigma_a*1e9)}nm""",B_max)  
    np.save(f"""simulations/pulse_generation_chirped/U_total_sigma={int(sigma_a*1e9)}nm""",U_total) 
    np.save(f"""simulations/pulse_generation_chirped/U_0={int(sigma_a*1e9)}nm""",U_0)  
           


    
    return driving    


def simulate_generation_chirped_pulse(waveguide, driving_N_chirped):
    
    #Import waveguide parameters
    Xsize = waveguide["Xsize"]
    Ysize = waveguide["Ysize"]
    Zsize = waveguide["Zsize"]
    Nx = waveguide["Nx"]
    Ny = waveguide["Ny"]
    Nz = waveguide["Nz"]
    Cx=Xsize/Nx
    Cy=Ysize/Ny
    Cz=Zsize/Nz

    By    = waveguide["By"]      # Bias field along the z direction
    Aex   = waveguide["Aex"]     # exchange constant
    Ms    = waveguide["Ms"]      # saturation magnetization
    alpha = waveguide["alpha"]   # damping parameter
    gamma = waveguide["gamma"]   # gyromagnetic ratio
    mu0   = waveguide["mu0"]     # vacuum permeability
    
    params_sim  = params_simulation_generation_pulse(waveguide)
    dt          = params_sim["dt"]
    dt_driving  = params_sim["dt_driving"]
    N_t_driving = params_sim["N_t_driving"]
    T_driving   = params_sim["T_driving"] 
    T_driving_initial   = params_sim["T_driving_initial"]    
    T           = params_sim["T"]       
    
    V_t         = np.real(driving_N_chirped["V_time"])

    A_antenna = params_sim["A_antenna"]
    sigma_antenna = params_sim["sigma_antenna"]
    
    script=f"""
Maxerr = 1e-8 
//Maxerr = 1e-7

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
m = uniform(0,1,0)
  

//Define the regions in which the damping should be increased
	    
number_of_steps:=90
width_of_step:=20e-9

for i:=1; i<number_of_steps; i++{{
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

B_ext = vector(0,{By},0) //Sets a external field   

//m0:=loadfile("./ground.ovf")
m.loadfile("./ground.ovf")


//mask for excitation field
//An antenna field is implemented via a so called VectorMask.
//The command NewVectorMask(x,y,z) sets the size of the mask
CPW_field := newVectorMask(Nx, 1, 1)
//In this case the antenna field is only one dimensional, since we only have a dependency on the x-position of the antenna field.
  
//We also have to set the dimensions of the CPW:
        
a:=300e-9 //half width of antenna
b:=80e-9	//half height of antenna
gap:=0
ts:=Zsize
y:=-(ts/2+gap+b)	//y-position of the antenna field

d:=1200e-9 //middle-to-middle distance of the center line and ground
current:=(10*1e-3)
pos:=Xsize/2 //CPW in middle of waveguide

for i:=0; i<Nx; i++{{
    x := -{Xsize/2}+{Cx}*i    
    Bz_antenna := {A_antenna/(np.sqrt(2*np.pi)*sigma_antenna)}*exp(-pow(x,2)/(2*pow({sigma_antenna},2)))
    CPW_field.setVector(i, 0, 0, vector(0, 0, Bz_antenna))
}}


mz:= Crop(m.Comp(2),0,{Nx},{int(Ny/2)},{int(Ny/2+1)},{int(Nz/2)},{int(Nz/2+1)}) //Save m_z(x,0,0)

Bz:= Crop(B_ext.Comp(2),0,{Nx},{int(Ny/2)},{int(Ny/2+1)},{int(Nz/2)},{int(Nz/2+1)}) //Save m_z(x,0,0)

//DefRegion( 201, cuboid(width_of_step, Ysize, Zsize).Transl(-Xsize*0.5+(i-1)*width_of_step,0,0) )

save(mz)   //Save the magnetization at t=0
save(Bz)

AutoSave(mz,{dt}) //Save the magnetization every dt including t=0
AutoSave(Bz,{dt})

//We add the antenna field to the external field.
//We add a time modulation of the field after the comma.
f_i:= {V_t[0]}

B_ext.removeextraterms()
B_ext.Add(CPW_field, f_i ) //modulation of z-component


run({dt_driving})

"""
    for i in np.arange(1,N_t_driving):
        f_i  = V_t[i]
        script = script + f"""
B_ext.removeextraterms()
B_ext.Add(CPW_field, {f_i} ) //modulation of z-component
run({dt_driving})

"""
    script = script + f""" 
run({T}-{T_driving-T_driving_initial})
"""
   
    create_mumax_script(script,"simulations/pulse_generation_chirped/","pulse_generation_chirped")
    print("Mumax3 script pulse_generation_chirped.txt created")



def import_generated_chirped_pulse(waveguide, ground_state, driving, xpoint, evolved_pulse, target_pulse_spectrum):
   
    Ny         = waveguide["Ny"]
    Xsize      = waveguide["Xsize"]
    Nx         = waveguide["Nx"]
    Cx         = Xsize/Nx
    Ms         = waveguide["Ms"]
    Nz         = waveguide["Nz"]
    
    params_sim = params_simulation_generation_pulse(waveguide)
    T          = params_sim["T"]
    dt         = params_sim["dt"]
     
    domega     = driving["domega"]
    omega_min  = 6*2*np.pi*1e9
    omega_max  = driving["omega"][-1]

    # Import the magnetization files    
    mzfields = read_npy_files("m","simulations/pulse_generation_chirped/pulse_generation_chirped.out")

    # Stack all snapshots of the magnetization on top of each other
    mzfield = np.stack([mzfields[key] for key in sorted(mzfields.keys())])  #we obtain an array m_z[t,i=2,z=0,y,x]
    mzfield = np.transpose(mzfield, (1,4,3,2,0)) #Now we have m_z[i_z,x,y,z,t]
    
    # Select the components of m as a function of position x and time at y=z=0
    mz0     = mzfield[0,:,0,0, :]

    #Import the magnetization of the ground state
    ground = ground_state["ground_all"]
    mz_ground  = ground[2,int(Nz/2),int(Ny/2),int(Nz/2)]    # Obtain the z-component of the ground state at x=y=z=0
    mground_x  = ground[2,int(Nz/2),int(Ny/2),:]            # Select the z-component of the ground state vs x  at y=z=0
    
    #Build an array for the ground state magnetization as a function of x and time
    mground_xt = np.stack([mground_x for key in sorted(mzfields.keys())]) 
    mground_xt = np.transpose(mground_xt, (1,0)) 

    #np.save("simulations/pulse_generation_chirped/m(x,t)", mz0)
    np.save("simulations/pulse_generation_chirped/m(x,t)_alone", mz0 - mground_xt)
    np.save("simulations/pulse_generation_chirped/dt", dt)    
    np.save("simulations/pulse_generation_chirped/mzground_all", mground_xt)
 

    return {"m(x,t)":mz0 , "m_ground(x,t)":mground_xt}



def show_generated_chirped_pulse(waveguide, ground_state, driving, xpoint, evolved_pulse, target_pulse_spectrum):
   
    Ny         = waveguide["Ny"]
    Xsize      = waveguide["Xsize"]
    Nx         = waveguide["Nx"]
    Cx         = Xsize/Nx
    Ms         = waveguide["Ms"]
    Nz         = waveguide["Nz"]
    
    params_sim = params_simulation_generation_pulse(waveguide)
    T          = params_sim["T"]
    dt         = params_sim["dt"]
     
    domega     = driving["domega"]
    omega_min  = 6*2*np.pi*1e9
    omega_max  = driving["omega"][-1]
    
    sigma_a    = waveguide["sigma_antenna"]
    time_fidelity_max   = np.load("simulations/chirped_pulse/time_max_fidelity.npy") 

    # Import the generated magnetization as a function of position x and time at y=z=0
    m_pulse_generated     = np.load("simulations/pulse_generation_chirped/m(x,t)_alone.npy")
    N_treal = np.shape(m_pulse_generated)[1]
    Treal   = dt*(N_treal-1)
    tlist_real = np.linspace(0,Treal,N_treal)    
    
    # Time and position for calculating the fidelity at (x1,+inf)
    t_half = 50*1e-9
    x_half = 0*1e-6


     
    # Calculate the overlap between the generated and the target states

    xlist = np.linspace(-Xsize/2,Xsize/2,Nx)
    
    t_half_in_list = find_nearest(tlist_real, t_half)["value"]
    i_t_half   = find_nearest(tlist_real, t_half)["index"]
    time_half   = tlist_real[i_t_half:] 
    

    x_half_in_list = find_nearest(xlist, x_half)["value"]
    i_x_half   = find_nearest(xlist, x_half)["index"]
    xlist_half  = xlist[i_x_half:] 

    i_time_fidelity_max = find_nearest(tlist_real, time_fidelity_max)["index"]  
    fidelity_evolution  = np.load("simulations/chirped_pulse/fidelity_evolution.npy") 
    
    target_analytical   = np.load("simulations/chirped_pulse/target_pulse_analytical.npy") #Analytical target pulse
    envelope_analytical = np.load("simulations/chirped_pulse/envelope_target_pulse_analytical.npy") #Analytical target pulse
    
    
    #Plot the generated pulse in time domain
  
    # Show the intensity plot of m_z(x,t) at y=z=0
    cm = 1/2.54
    plt.figure(figsize=(9*cm, 5*cm))
    extent = [- 1e6*Xsize/2, 1e6*Xsize/2, 0, 1e9*Treal]  # extent of k values and frequencies
    plt.imshow(np.transpose(np.abs(m_pulse_generated)**2), extent=extent, aspect='auto', origin='lower', cmap="viridis",vmin=0,vmax=5*1e-7)
    plt.ylabel("t [ns]")
    plt.xlabel("x [$\mu$m]")
    plt.colorbar()
    plt.title("$|m_z(x,y=0,z=0,t)|^2$" )
    plt.vlines(0,0,150,color="white", linestyle="dashed",linewidth=0.5)  
    plt.hlines(time_fidelity_max*1e9,-30,30,color=c["green"], linestyle="dashed",linewidth=0.5)  
    plt.hlines(64.96,-30,30,color=c["green"], linestyle="dashed",linewidth=0.5)  
    plt.xlim(-20,20)
    plt.ylim(0,120)
    #plt.hlines(20,-40,40,color="white", linestyle="dashed")
    plt.savefig("plots/pulse_generation_chirped/m_abs_created+lines.pdf",bbox_inches='tight',dpi=300)
    plt.show()
    

   
    # Calculate the overlap between the generated and the analytical target states

    target_analytical_half_space  = target_analytical[i_x_half:]
    envelope_analytical_half_space  = envelope_analytical[i_x_half:]
    

    fidelity_list = []
    fidelity_tlist = []
    
    #Calculate the fidelity between the generated and analytical target vs time
    
    for i_t in range(find_nearest(tlist_real, time_fidelity_max-25*dt)["index"],find_nearest(tlist_real,time_fidelity_max+25*dt)["index"]+1,1):
        t = i_t*dt
        pulse_generated = m_pulse_generated[i_x_half:,i_t]
        pulse_target    = target_analytical[i_x_half:] 
        
        overlap_gg = Cx * np.dot(np.conjugate(pulse_generated),pulse_generated)
        overlap_tt = Cx * np.dot(np.conjugate(pulse_target),pulse_target)
        overlap_gt = Cx * np.dot(np.conjugate(pulse_generated),pulse_target)
        overlap = overlap_gt/np.sqrt(overlap_gg*overlap_tt)
        fidelity = overlap**2
        
        fidelity_list.append(fidelity)
        fidelity_tlist.append(t)

    fidelity_list = np.array(fidelity_list)
    fidelity_tlist = np.array(fidelity_tlist)
    
    # Show the fidelity vs time
    
    plt.figure(figsize=(9*cm, 5*cm))
    plt.scatter(1e9*fidelity_tlist,fidelity_list,color=c["blue"],s=8)
    plt.plot(1e9*fidelity_tlist,fidelity_list,color=c["blue"])
    plt.ylabel("$\mathcal{F}(t)$")
    plt.xlabel("t [ns]")
    plt.ylim(0,1)       
    plt.xlim((time_fidelity_max-15*dt)*1e9,(time_fidelity_max+15*dt)*1e9)
    plt.vlines(time_fidelity_max*1e9,0,5,color=c["green"],linestyle='dashed')
    #plt.legend()    
    plt.savefig("plots/pulse_generation_chirped/fidelity_vs_time.pdf")
    plt.show()
    
    #Time of maximum fidelity
    fidelity_max = np.amax(fidelity_list)
    i_fidelity_max = find_nearest(fidelity_list, fidelity_max)["index"]
    time_fidelity_max = fidelity_tlist[i_fidelity_max]
    i_time_fidelity_max = find_nearest(tlist_real, time_fidelity_max)["index"]
    

    print("Maximum fidelity of the protocol")
    print(fidelity_max)
    
    print("Time of maximum generation fidelity")
    print(time_fidelity_max*1e9)
    
    # Show m_z(x,t_f) at t=0 and t=tf
    
    mpulse_max_fidelity = m_pulse_generated[i_x_half:,i_time_fidelity_max]

    plt.figure(figsize=(9*cm, 5*cm))
    plt.plot(1e6*xlist_half,mpulse_max_fidelity,label="generated",color=c["blue"]) 
    plt.plot(1e6*xlist_half,target_analytical_half_space,label="target",color=c["green"])       
    plt.ylabel("$m_z(x,t)$")
    plt.xlabel("x [$\mu$m]")
    plt.xlim(5,8)
    plt.legend()    
    plt.savefig("plots/pulse_generation_chirped/pulses_at_maximum_fidelity.pdf",bbox_inches='tight')
    plt.show() 
    
    N_normalize_generated  = np.sqrt( Cx * sum( np.abs(mpulse_max_fidelity)**2 ) )
    N_normalize_target     = np.sqrt( Cx * sum( np.abs(target_analytical_half_space)**2 ) )
    N_normalize_env        = np.sqrt( Cx * sum( np.abs(envelope_analytical_half_space)**2 ) )
    
    plt.figure(figsize=(9*cm, 5*cm))
    plt.plot(1e6*xlist_half,mpulse_max_fidelity/N_normalize_generated *1e-3,label="generated",color=c["blue"]) 
    plt.plot(1e6*xlist_half,target_analytical_half_space/N_normalize_target*1e-3,label="target",color=c["green"],linestyle='dashed')        
    env = envelope_analytical_half_space/N_normalize_target*1e-3
    plt.fill_between(1e6*xlist_half, - env, env, alpha=0.4,color=c["green"])
    plt.ylabel("$m_z(x,t)/N$ [$\mu$m]$^{-1/2}$")
    plt.xlabel("x [$\mu$m]")
    plt.xlim(5,8)
    plt.legend()    
    plt.savefig("plots/pulse_generation_chirped/pulses_at_maximum_fidelity_N.pdf",bbox_inches='tight')
    plt.show() 
    
    np.save(f"""simulations/pulse_generation_chirped/fidelity_protocol_sigma={int(sigma_a*1e9)}nm""",fidelity_max)   
    np.save("simulations/pulse_generation_chirped/m(x,time_max_fidelity)", mpulse_max_fidelity)    
    
    np.save("simulations/pulse_generation_chirped/m(x,time_max_fidelity)_normalized", mpulse_max_fidelity/N_normalize_generated)   
    np.save("simulations/chirped_pulse/target_pulse_normalized",target_analytical_half_space/N_normalize_target)   
    np.save("simulations/chirped_pulse/envelope_target_pulse_normalized",envelope_analytical_half_space/N_normalize_target)   
    np.save("simulations/chirped_pulse/xlist_half",xlist_half)   
  
    
    # Determine the variance of the pulse
    
    def Exp(t,xlist,pulse):    #Calculate the expected value of the simulated pulse using |m(x,t)|^2 
        i_t = int(t/dt)
        P   = Cx * sum( np.abs(pulse[:,i_t])**2 )
        E   = Cx * np.dot( xlist, np.abs(pulse[:,i_t])**2) / P 
        return E
    Exp_func = np.vectorize(Exp)
    Exp_func.excluded.add(1)
    Exp_func.excluded.add(2)

    def var(t,xlist,pulse):     #Calculate the variance of the pulse   
        i_t = int(t/dt)
        P   = Cx * sum( np.abs(pulse[:,i_t])**2 )
        E   = Cx * np.dot( xlist, np.abs(pulse[:,i_t])**2) / P 
        var = Cx * np.dot( (xlist-E)**2 , np.abs(pulse[:,i_t])**2 ) / P 
        return var
    var_func = np.vectorize(var)
    var_func.excluded.add(1)
    var_func.excluded.add(2)
    
    
    #Calculate the variance 
    m_pulse_generated_half = m_pulse_generated[i_x_half:,:]
    
    
    def var_time(x,tlist,pulse):     #Calculate the variance of the pulse  
        x_nearest = find_nearest(xlist_half, x)["value"]
        i_x = int( (x_nearest - x_half)/Cx)
        P   = dt * sum( np.abs(pulse[i_x,:])**2 )
        E   = dt * np.dot( tlist, np.abs(pulse[i_x,:])**2) / P 
        
        var = dt * np.dot( (tlist-E)**2 , np.abs(pulse[i_x,:])**2 ) / P 
        return var
    var_time_func = np.vectorize(var_time)
    var_time_func.excluded.add(1)
    var_time_func.excluded.add(2)
    
    var_time_list = var_time_func(xlist_half,tlist_real,m_pulse_generated_half)    
    
    
    
    pos_max_fidelity = Exp(time_fidelity_max,xlist_half,m_pulse_generated_half) 
    var_max_fidelity = var(time_fidelity_max,xlist_half,m_pulse_generated_half) 
    
    print("time [ns], position [mu m] and variance [mu m] at maximum fidelity:")
    print(time_fidelity_max*1e9,1e6*pos_max_fidelity,np.sqrt(var_max_fidelity)*1e6)
    
    
    var_time_max_F = var_time_func(pos_max_fidelity, tlist_real, m_pulse_generated_half)    
    
    print("time variance [ns] at the central position of the pulse of maximum fidelity:")
    print(np.sqrt(var_time_max_F)*1e9)

    

    x_f = 6.8*1e-6   
    pulse_evolved       = np.load("simulations/chirped_pulse/m(x,t)_alone.npy")
    pulse_evolved_half  = pulse_evolved[i_x_half:,:]
    var_time_xf_generated = var_time_func(x_f, tlist_real, m_pulse_generated_half)    
    var_time_xf_evolved = var_time_func(x_f, tlist_real, pulse_evolved_half)   
    
    x_nearest = find_nearest(xlist_half, pos_max_fidelity)["value"]
    i_x = int( (x_nearest - x_half)/Cx)
    
    
    print("time variance [ns] of the generated pulse at x = x_f = 10mum:")
    print(np.sqrt(var_time_xf_generated)*1e9)
    

    print("time variance [ns] of the evolved pulse at x = x_f = 10mum:")
    print(np.sqrt(var_time_xf_evolved)*1e9)
    
    plt.figure(figsize=(8*cm, 5*cm))
    plt.plot(1e9*tlist_real,np.abs(m_pulse_generated_half[i_x,:])**2,label="generated",color=c["red"])        
    plt.plot(1e9*tlist_real,np.abs(pulse_evolved_half[i_x,:])**2,label="evolved",color=c["green"],alpha=0.5)   
    plt.ylabel("$m_z(x_f,t)$")
    plt.xlabel("t [ns]")
    #plt.vlines(pos_max_fidelity*1e6,0,10, color=c["blue"],linestyle="dashed")
    #plt.ylim(0,10)
    plt.xlim(78,90)
    plt.legend()
    #plt.hlines(np.sqrt(var_time_max_F)*1e9,0,30, color=c["green"],linestyle="dashed")
    plt.savefig("plots/pulse_generation_chirped/pulse_xf_vs_time.pdf",bbox_inches='tight')
    plt.show() 
    
    plt.figure(figsize=(8*cm, 5*cm))
    plt.plot(1e9*tlist_real,np.abs(m_pulse_generated_half[i_x,:])**2,label="generated",color=c["red"])   
    plt.plot(1e9*tlist_real,np.abs(pulse_evolved_half[i_x,:])**2,label="evolved",color=c["green"],alpha=0.5)       
    plt.ylabel("$m_z(x_f,t)$")
    plt.xlabel("t [ns]")
    #plt.vlines(pos_max_fidelity*1e6,0,10, color=c["blue"],linestyle="dashed")
    plt.ylim(0,0.1*1e-7)
    plt.legend()
    #plt.xlim(65,75)
    #plt.hlines(np.sqrt(var_time_max_F)*1e9,0,30, color=c["green"],linestyle="dashed")
    plt.savefig("plots/pulse_generation_chirped/pulse_xf_vs_time_all.pdf",bbox_inches='tight')
    plt.show()
    
    
    
    #Calculate the variance 

    
    var_list = var_func(time_half,xlist_half,m_pulse_generated_half)       
    
    #Minimum variance:    
    var_min = np.amin(var_list)
    time_var_min = i_t_half*dt + dt*np.argmin(np.sqrt(var_list))
    i_t_var_min = find_nearest(tlist_real, time_var_min)["index"]
      
    #print('Minimum std dev generated pulse:')
    #print(np.sqrt(var_min)*1e6)
    
    var_time_max_fidelity = var_list[i_time_fidelity_max]
    
    plt.figure(figsize=(8*cm, 5*cm))
    plt.plot(1e6*xlist_half,1e9*np.sqrt(var_time_list),label="generated",color=c["red"])        
    plt.ylabel("std. dev. in time $(x)$ [$\mu$m]")
    plt.xlabel("x [$\mu$m]")
    plt.vlines(pos_max_fidelity*1e6,0,10, color=c["blue"],linestyle="dashed")
    plt.ylim(0,10)
    plt.xlim(0,30)
    plt.hlines(np.sqrt(var_time_max_F)*1e9,0,30, color=c["green"],linestyle="dashed")
    plt.savefig("plots/pulse_generation_chirped/stardard_deviation_time(t).pdf",bbox_inches='tight')
    plt.show()  
    
      
    #print('Std dev generated pulse at maximum fidelity:')
    #print(np.sqrt(var_time_max_fidelity)*1e6)
    
    #i_time_end_driving = np.argmax( np.sqrt(var_list[:int(len(var_list)/2)] ))

    #time_end_driving = i_t_half*dt+dt*i_time_end_driving
    #var_end_driving = np.amax(var_list[:int(len(var_list)/2)] )
    
    time_end_driving = np.load(f"""simulations/pulse_generation_chirped/Delta_t_driving_0.995.npy""") 
    i_time_end_driving = find_nearest(tlist_real, time_end_driving)["index"]
    
    i_time_end_driving_half_list =   find_nearest(time_half, time_end_driving)["index"]
    var_end_driving = var_list[i_time_end_driving_half_list]
    
    print("Time of the end of the driving")
    print(time_end_driving*1e9)
    print(i_time_end_driving*1e9*dt)
    
    #np.save(f"""simulations/pulse_generation_chirped/Delta_t_driving""",time_end_driving)  
    
    print("Std dev at the end of the driving")
    print(np.sqrt(var_end_driving)*1e6)
    
    # Plot the variance of the pulse
    plt.figure(figsize=(8*cm, 5*cm))
    plt.plot(1e9*time_half,1e6*np.sqrt(var_list),label="generated",color=c["red"])        
    plt.ylabel("$\sigma(t)$ [$\mu$m]")
    plt.xlabel("t [ns]")
    plt.xlim(t_half*1e9,120)
    plt.ylim(0,2)
    plt.vlines(time_end_driving*1e9,0,2, color=c["blue"],linestyle="dashed")
    plt.vlines(time_var_min*1e9,0,2, color=c["blue"],linestyle="dashed")
    plt.vlines(time_fidelity_max*1e9,0,2, color=c["purple"],linestyle="dashed")
    plt.savefig("plots/pulse_generation_chirped/stardard_deviation(t).pdf",bbox_inches='tight')
    plt.show()  
    
    #Maximum intensity of the focused pulse:
    max_intensity = np.amax(np.abs(m_pulse_generated[:,i_time_fidelity_max])**2)
    
    plt.figure(figsize=(8*cm, 5*cm))
    plt.plot(1e6*xlist,np.abs(m_pulse_generated[:,i_time_fidelity_max])**2,label="generated",color=c["red"])        
    plt.ylabel("$|m_z(x,0,0,t_d)/M_s|^2$")
    plt.xlabel("x [\mu m]")
    plt.xlim(0,10)
    #plt.ylim(0,2)
    #plt.vlines(time_end_driving*1e9,0,2, color=c["blue"],linestyle="dashed")
    #plt.vlines(time_var_min*1e9,0,2, color=c["blue"],linestyle="dashed")
    #plt.vlines(time_fidelity_max*1e9,0,2, color=c["purple"],linestyle="dashed")
    plt.savefig("plots/pulse_generation_chirped/intensity_focused_pulse.pdf",bbox_inches='tight')
    plt.show()  
    
    print("Maximum intensity:")
    print(max_intensity)
    
    #Maximum intensity of the pulse just after the generation:
    max_intensity_end_driving = np.amax(np.abs(m_pulse_generated[:,i_time_end_driving])**2)
    
    #np.save(f"""simulations/pulse_generation_chirped/i_time_end_driving_full_list""",i_time_end_driving_full_list)  
    
    plt.figure(figsize=(8*cm, 5*cm))
    plt.plot(1e6*xlist,np.abs(m_pulse_generated[:,i_time_end_driving])**2,label="generated",color=c["red"])        
    plt.ylabel("$|m_z(x,0,0,t_d)/M_s|^2$")
    plt.xlabel("x [\mu m]")
    plt.xlim(0,10)
    #plt.ylim(0,2)
    #plt.vlines(time_end_driving*1e9,0,2, color=c["blue"],linestyle="dashed")
    #plt.vlines(time_var_min*1e9,0,2, color=c["blue"],linestyle="dashed")
    #plt.vlines(time_fidelity_max*1e9,0,2, color=c["purple"],linestyle="dashed")
    plt.savefig("plots/pulse_generation_chirped/intensity_end_driving_pulse.pdf",bbox_inches='tight')
    plt.show()  
    
    print("Maximum intensity just after the driving has finished:")
    print(max_intensity_end_driving)
    
    print("Increase in intensity:")
    print(max_intensity/max_intensity_end_driving)
    
    
    print("Reduction in variance:")
    print( np.sqrt(var_end_driving/var_time_max_fidelity))
    
    np.save(f"""simulations/pulse_generation_chirped/variance_at_max_fidelity_sigma={int(sigma_a*1e9)}nm""",var_time_max_fidelity)  
    #np.save(f"""simulations/pulse_generation_chirped/min_variance_sigma={sigma_a}""",var_min)    
    
    
    
