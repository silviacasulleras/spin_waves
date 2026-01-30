"""
Created on Thu Feb  4 11:20:49 2021
Determination of the transfer function of a square waveguide
@author: Silvia Casulleras
"""

import numpy as np
import matplotlib.pyplot as plt

from modules.mumax3_functions import read_npy_files, create_mumax_script
from modules.postprocessing import FT_function, IFT_function

#n=4
#colors = plt.cm.viridis(np.linspace(0,1,n))
c = {"blue":'#0072BD', "orange": '#D95319', "green":"#77AC30", "yellow": '#EDB120', "purple":'#7E2F8E', "red":"#A2142F", "light-blue":"4DBEEE" }
   

def params_sim_transfer(waveguide,dispersion):
    T    = 200e-9        # Total simulation time
    dt   = 0.01*1e-9     # Time step for the snapshots of the magnetization 
    N_t  = round(T/dt)   # Number of snapshots of the magnetization (including t=0 and but not t=T)
    dt_s = 1e-12         # Time step for the duration of the driving (similar to the timestep of the simulation)
    
    Amp_V = 1            # Amplitude of the driving 
    
    #Parameters of the determination of spectrum of the transfer function
    omega_c          = dispersion["omega_cutoff"]
    
    k1               = dispersion["k1"]
    k2               = dispersion["k2"] 
    N_k              = dispersion["N_k_fit"]
    dk               = dispersion["dk"] 
    #domega           = dispersion["domega"]  
    #N_omega          = dispersion['N_omega_fit']
    
    domega_T    = 2*np.pi/T
    omega_min_T = -2*np.pi/(2*dt)
    omega_max_T = 2*np.pi/(2*dt)
    N_omega_T   = N_t
    
    omega1           = dispersion["omega1"] 
    omega2           = dispersion["omega2"] 
    polcoeffs        = dispersion["pol_6th"]
    degree_pol_fit   = dispersion["degree_pol_fit"]
    
    return {"T":T, "dt":dt, "N_t":N_t, "omega1": omega1, "omega2": omega2, "domega_T":domega_T, "Amp_V":Amp_V, "dt_s":dt_s, 
            "N_omega_T": N_omega_T, "dk":dk,
            "k1":k1, "k2":k2, "N_k":N_k, "omega_min_T":omega_min_T, "omega_max_T":omega_max_T}


def simulate_transfer_function(waveguide,dispersion):
    
    params = params_sim_transfer(waveguide,dispersion)
    T      = params["T"]
    dt     = params["dt"]   
    Amp_V  = params["Amp_V"]
    dt_s   = params["dt_s"]
    
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
    A_antenna     = waveguide["A_antenna"]
    sigma_antenna = waveguide["sigma_antenna"]
    
    
    xlist = np.linspace(-Xsize/2,Xsize/2,Nx+1) 

    plt.plot(xlist*1e9,A_antenna/(np.sqrt(2*np.pi)*sigma_antenna) * np.exp(-xlist**2/(2*sigma_antenna**2)))
    plt.scatter(xlist*1e9,A_antenna/(np.sqrt(2*np.pi)*sigma_antenna)*np.exp(-xlist**2/(2*sigma_antenna**2)), s=10)    
    plt.xlim(-20*Cx*1e9,20*Cx*1e9)
    plt.xlabel("$x$ [nm]")
    plt.ylabel("$B_z(x)$ [T]")
    plt.axvspan(0,Cx*1e9,color="green",alpha=0.5)
    plt.savefig("plots/transfer_function/B_antenna_profile.pdf")
    plt.show()


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
    
//Set the magnetization in the ground state

m.loadfile("./ground.ovf")        //This is equivalent to relaxing the state

//An antenna field is implemented via a so called VectorMask.

//The command NewVectorMask(x,y,z) sets the size of the mask
CPW_field := newVectorMask(Nx, 1, 1)
//In this case the antenna field is only one dimensional, since we only have a dependency on the x-position of the antenna field.


for i:=0; i<Nx; i++{{
    x := -{Xsize/2}+{Cx}*i    
    Bz_antenna := {A_antenna/(np.sqrt(2*np.pi)*sigma_antenna)}*exp(-pow(x,2)/(2*pow({sigma_antenna},2)))
    CPW_field.setVector(i, 0, 0, vector(0, 0, Bz_antenna))
}}



mz:= Crop(m.Comp(2),0,{Nx},{int(Ny/2)},{int(Ny/2+1)},{int(Nz/2)},{int(Nz/2+1)}) //Save m_z(x,0,0)
Bz:= Crop(B_ext.Comp(2),0,{Nx},{int(Ny/2)},{int(Ny/2+1)},{int(Nz/2)},{int(Nz/2+1)}) //Save m_z(x,0,0)


tableadd(MaxAngle)
tableadd(dt)
tableautosave({dt})


//AutoSave(Bz,{dt})


AutoSave(mz,{dt}) //Save the magnetization every dt
save(mz)   //Save the magnetization at t=T (maybe repeated twice...)
SaveAs(mz, "single_saved_mz")

//We add the antenna field to the external field.
//Note that you can add a time modulation of the field after the comma.

B_ext.Add(CPW_field, {Amp_V} ) //modulation of z-component: delta function 
run({dt_s})

B_ext.RemoveExtraTerms()   
run({T-dt_s})
    

"""
    create_mumax_script(script,"simulations/transfer_function/","transfer_function")
    print("Script transfer_function.txt created")
    
    
    
def extract_transfer_function(waveguide, ground_state, dispersion, xpoint):
    
    params = params_sim_transfer(waveguide,dispersion)
    T      = params["T"]
    dt     = params["dt"]
    N_t    = params["N_t"]     
    Amp_V  = params["Amp_V"]
    dt_s   = params["dt_s"]
    T      = params["T"]
        
    Ny     = waveguide["Ny"]
    Nz     = waveguide["Nz"]    
    Xsize  = waveguide["Xsize"]
    Nx     = waveguide["Nx"]
    Cx     = Xsize/Nx
    Ms     = waveguide["Ms"]
    A_antenna     = waveguide["A_antenna"]
    sigma_antenna = waveguide["sigma_antenna"]
    
    #Plot the time dependence of the driving 
    #tlist      = np.linspace(0,T,N_t+1,endpoint=True)          # list of time steps when the simulation also saves m at t=0    


    #Import the evolution of the magnetization 

    mzfields=read_npy_files("m","simulations/transfer_function/transfer_function.out")

    # Stack all snapshots of the magnetization on top of each other
    mzfield = np.stack([mzfields[key] for key in sorted(mzfields.keys())])  #we obtain an array m_z[t,i=2,z=0,y=0,x]
    mzfield = np.transpose(mzfield, (1,4,3,2,0)) #Now we have m_z[i_z,x,y,z,t]
    
    # Select the components of m as a function of position x and time at y=z=0
    mz0 = mzfield[0,:,0,0,:]
    
    
    #Import the magnetization of the ground state
    ground = ground_state["ground_all"]
    mz_ground  = ground[2,int(Nz/2),int(Ny/2),int(Nz/2)]    # Obtain the z-component of the ground state at x=y=z=0
    mground_x  = ground[2,int(Nz/2),int(Ny/2),:]            # Select the z-component of the ground state vs x  at y=z=0
    
    #Build an array for the ground state magnetization as a function of x and time
    mground_xt = np.stack([mground_x for key in sorted(mzfields.keys())]) 
    mground_xt = np.transpose(mground_xt, (1,0)) 

    N_treal = np.shape(mz0)[1]
    Treal   = dt*(N_treal-1)
    #tlist_real = np.linspace(0,Treal,N_treal,endpoint=False)
    tlist_real = np.linspace(0,Treal,N_treal)  
    
    # Show the intensity plot of m_z(x,t) at y=z=0
    plt.figure(figsize=(10, 6))
    extent = [- 1e6*Xsize/2, 1e6*Xsize/2, 0, 1e9*Treal]  # extent of k values and frequencies
    plt.imshow(np.transpose(np.abs(mz0-mground_xt)**2), extent=extent, aspect='auto', origin='lower', cmap="viridis",vmin=0, vmax=1e-5)
    plt.ylabel("t [ns]")
    plt.xlabel("x [$\mu$m]")
    plt.colorbar()
    plt.title("$|m_z(x,y=0,z=0,t)|^2$" )
    plt.ylim(0,1)
    plt.xlim(-1,1)
    plt.savefig("plots/transfer_function/m_abs_generated.pdf")
    plt.show()
    
    '''
    # Show the intensity plot of m_z(x,t) at y=z=0
    plt.figure(figsize=(10, 6))
    extent = [- 1e6*Xsize/2, 1e6*Xsize/2, 0, 1e9*T]  # extent of k values and frequencies
    plt.imshow(np.transpose(np.abs(mz0-mground_xt)**2), extent=extent, aspect='auto', origin='lower', cmap="viridis")
    plt.ylabel("t [ns]")
    plt.xlabel("x [$\mu$m]")
    plt.colorbar()
    plt.title("$|m_z(x,y=0,z=0,t)|^2$" )
    plt.xlim(-10,10)
    plt.savefig("plots/transfer_function/m_abs_generated.pdf")
    plt.show()
    '''
    
    #Calculate the transfer function in temporal domain
    F_xt = np.sqrt(2*np.pi)*(mz0-mground_xt)*Ms/dt_s
    
    np.save("simulations/transfer_function/F_xt",F_xt)
    np.save("simulations/transfer_function/F_xt_tlist", tlist_real)
    np.save("simulations/transfer_function/Treal", Treal)
    np.save("simulations/transfer_function/dt", dt)
    
    return 
    #return {"t":tlist, "F(x,t)": F_xt,"dt": dt, "A_antenna":A_antenna, "sigma_antenna":sigma_antenna}

def show_transfer_function(waveguide, ground_state, dispersion, xpoint):
    
    params = params_sim_transfer(waveguide,dispersion)
    T      = params["T"]
    dt     = params["dt"]    
    Amp_V  = params["Amp_V"]
    dt_s   = params["dt_s"]

    # Frequencies at which we calculate the transfer function
    domega_T       = params["domega_T"]
    omega_max_T    = params["omega_max_T"]
    omega_min_T    = params["omega_min_T"]    
    N_omega_T      = params["N_omega_T"]
    
    #Limits of frequency for plotting
    omega1         = params["omega1"]     
    omega2         = params["omega2"]     
    
    Ny     = waveguide["Ny"]
    Nz     = waveguide["Nz"]    
    Xsize  = waveguide["Xsize"]
    Nx     = waveguide["Nx"]
    Cx     = Xsize/Nx
    Ms     = waveguide["Ms"]
    A_antenna     = waveguide["A_antenna"]
    sigma_antenna = waveguide["sigma_antenna"]    
    
    #Plot the time dependence of the driving 
    #tlist  = transfer_funct["t"]          # list of time steps when the simulation also saves m at t=0    
    tlist  = np.load("simulations/transfer_function/F_xt_tlist.npy")
    Treal  = np.load("simulations/transfer_function/Treal.npy")

    #Calculate the transfer function in temporal domain
    #F_xt = transfer_funct["F(x,t)"]
    F_xt = np.load("simulations/transfer_function/F_xt.npy")

    # Show the intensity plot of F(x,t) at y=z=0
    plt.figure(figsize=(10, 6))
    extent = [- 1e6*Xsize/2, 1e6*Xsize/2, 0, 1e9*Treal]  # extent of k values and frequencies
    plt.imshow(np.transpose(F_xt*1e-6*1e-9), extent=extent, aspect='auto', origin='lower', cmap="viridis",vmin=-0.5,vmax=0.5)
    plt.ylabel("t [ns]")
    plt.xlabel("x [$\mu$m]")
    plt.colorbar()
    plt.title("$F_z(x,y=0,z=0,t)$ [A$\cdot \mu$m$^{-1}\cdot$ns$^{-1}$]" )
    plt.xlim(-5,5)
    plt.ylim(0,5)
    plt.savefig("plots/transfer_function/F(x,t).pdf")
    plt.show()

    #Show F(t) at x=xpoint
    i_point = int(round((xpoint + Xsize/2 )/Cx))
    F_x = F_xt[i_point, :]
    F_function_t = {"t":tlist, "values":F_x , "dt":dt}

    plt.plot(F_function_t["t"]*1e9,F_function_t["values"]*1e-6*1e-9, label="re",color="b")
    plt.legend()
    plt.ylabel("$F_z(\\vec{r}_0,t)$ [A$\cdot \mu$m$^{-1}\cdot$ns$^{-1}$]" )
    plt.xlabel("$t$ [ns]")
    plt.savefig("plots/transfer_function/F(t).pdf")
    plt.xlim(-1,40)
    plt.show()
    
    
    
    #Calculate the transfer function in frequency domain
    #omega_list   = np.linspace(-omega_max_T,omega_max_T,N_omega_T,endpoint=True) 
    omega_list   = np.linspace(-omega_max_T,omega_max_T,N_omega_T+1,endpoint=True) 
    F_omega      = FT_function(omega_list,F_function_t)                    #Fourier transform of the function F(x=xpoint,t)
    F_function_omega = {"omega":omega_list, "values":F_omega}


    #Show F(x,omega)
    plt.plot(F_function_omega["omega"]/(2*np.pi*1e9) ,np.real(1e-6*F_function_omega["values"]),color="b",label="re")
    plt.plot(F_function_omega["omega"]/(2*np.pi*1e9) ,np.imag(1e-6*F_function_omega["values"]),color="orange",label="im")
    plt.legend()
    plt.ylabel("$\\tilde F_z(\\vec{r}_0,\omega)$ [A$\cdot \mu$m$^{-1}$]" )        
    plt.xlabel("$\omega/(2\pi)$ [GHz]")
    #plt.ylim(-0.75,0.75)
    plt.xlim(6,omega2/(2*np.pi*1e9))
    plt.savefig("plots/transfer_function/F(omega)")
    plt.show()

    
    #Export the values of the function F in time and spectral domain

    np.save("simulations/transfer_function/F(omega)_freqs",  F_function_omega["omega"])
    np.save("simulations/transfer_function/F(omega)_values", F_function_omega["values"])
    np.save("simulations/transfer_function/domega_T", domega_T)
    np.save("simulations/transfer_function/omega_max_T", omega_max_T)
    np.save("simulations/transfer_function/omega_min_T", omega_min_T)
    np.save("simulations/transfer_function/omega1", omega1)
    np.save("simulations/transfer_function/omega2", omega2)
        
    np.save("simulations/transfer_function/F(t)_time",   F_function_t["t"])
    np.save("simulations/transfer_function/F(t)_values", F_function_t["values"])
    np.save("simulations/transfer_function/dt", dt)
    np.save("simulations/transfer_function/dt_s", dt_s)
    np.save("simulations/transfer_function/A_antenna", A_antenna)
    np.save("simulations/transfer_function/sigma_antenna", sigma_antenna)

    return {"t":tlist, "F(t)": F_x, "omega":omega_list, "F(omega)": F_omega, "dt": dt, "domega_T":domega_T,
            "omega_max_T":omega_max_T, "omega_min_T":omega_min_T,  "A_antenna":A_antenna, "sigma_antenna":sigma_antenna}
   
    
def import_transfer_function(xpoint):
    transfer_function={"omega"            : np.load("simulations/transfer_function/F(omega)_freqs.npy"), 
                       "F(omega)"         : np.load("simulations/transfer_function/F(omega)_values.npy"),
                       "t"                : np.load("simulations/transfer_function/F(t)_time.npy"),
                       "F(t)"             : np.load("simulations/transfer_function/F(t)_values.npy"),
                       "domega_T"         : np.load("simulations/transfer_function/domega_T.npy"),
                       "dt"               : np.load("simulations/transfer_function/dt.npy"),                   
                       "dt_s"             : np.load("simulations/transfer_function/dt_s.npy"),
                       "omega_max_T"      : np.load("simulations/transfer_function/omega_max_T.npy"),
                       "omega_min_T"      : np.load("simulations/transfer_function/omega_min_T.npy"),
                       "omega1"           : np.load("simulations/transfer_function/omega1.npy"),
                       "omega2"           : np.load("simulations/transfer_function/omega2.npy"),
                       "omega_cutoff"     : np.load("simulations/dispersion_relation/omega_cutoff.npy")}
    
    
    #Show F(t)
    plt.figure(figsize=(10,4))    
    plt.plot(transfer_function["t"]*1e9, transfer_function["F(t)"]*1e-6*1e-9, label="re",color="b")
    plt.legend()
    plt.ylabel("$F_z(\\vec{r}_0,t)$ [A$\cdot \mu$m$^{-1}\cdot$ns$^{-1}$]" )
    plt.xlabel("$t$ [ns]")
    plt.xlim(-0.1,10)
    plt.savefig(f"""plots/transfer_function/F(t)_x={xpoint*1e6}.pdf""")
    plt.show()
     
    #Show F(omega)

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
    plt.figure(figsize=(6.8*cm, 4*cm))
    plt.plot(transfer_function["omega"]/(2*np.pi*1e9) ,np.real(1e-6*transfer_function["F(omega)"]),color=c["blue"],label="re")
    plt.plot(transfer_function["omega"]/(2*np.pi*1e9) ,np.imag(1e-6*transfer_function["F(omega)"]),color=c["orange"],label="im")
    plt.legend()
    plt.ylabel("$\\tilde F_z(\\vec{r}_0,\omega)$ [A$\cdot \mu$m$^{-1}$]",labelpad=0)        
    plt.xlabel("$\omega/(2\pi)$ [GHz]",labelpad=0)
    plt.ylim(-0.6,0.6)
    plt.xlim(6.1,7)
    plt.vlines(transfer_function['omega_cutoff']/(2*np.pi*1e9), -1, 2, linestyle="dashed",color=c["green"])
    plt.savefig("plots/transfer_function/F(omega).pdf", bbox_inches = 'tight',transparent=True)
    plt.show()
    
    #Show absolute value and phase of F(omega)
    
    
  
    
    return transfer_function
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    