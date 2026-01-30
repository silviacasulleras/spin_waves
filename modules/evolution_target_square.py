"""
Created on Thu Feb  4 11:20:49 2021
Simulation of the evolution of a square spin wave in a rectangular waveguide 
@author: Silvia Casulleras
"""

import numpy as np
import matplotlib.pyplot as plt
from modules.mumax3_functions import create_mumax_script, read_npy_files
from modules.postprocessing import FT_function, IFT_function
from scipy import special 
from scipy import optimize 
#n=4
#colors = plt.cm.viridis(np.linspace(0,1,n))
c = {"blue":'#0072BD', "orange": '#D95319', "green":"#77AC30", "yellow": '#EDB120', "purple":'#7E2F8E', "red":"#A2142F", "light-blue":"4DBEEE" }
  

def parameters_target_pulse(dispersion):
    
    #omega_0 = 6.416742146959894                              # carrier wavenumber of the target pulse
    k0      = 15*1e6  
    l0      = 2*np.pi/np.abs(k0)                              # carrier wavelength
    df      = 10*1e-6                                         # focal distance     
    sigmaf  = 2*1e-6                                          # spot size 
    sigma_conv  = 0.05*1e-6                                   # spot size 
    A           = 8*1e6
    #Amplitude_pulse = 0.04
    Amplitude_pulse = 0.004                # (to study the effect of nonlinearity)
    t0      = -70*1e-9
    #kmax = (k0+2/sigmaf)
    #lmin = 2*np.pi/kmax
    
    return {"k0":k0,  "df": df, "sigmaf":sigmaf, "sigma_conv":sigma_conv, "A":A,"Amplitude_pulse": Amplitude_pulse,"t0":t0}


def params_simulation_pulse(waveguide):
    T      = 150e-9         # Total simulation time
    dt     = 1e-11     # Time step for the snapshots of the magnetization 
    N_t    = round(T/dt)+1                              # Number of snapshots of the magnetization
   
    Xsize = waveguide["Xsize"]
    Nx    = waveguide["Nx"]
    Cx    = Xsize/Nx
    dk = 2*np.pi/Xsize
    
    number_of_steps  = 90
    width_of_step    = 20e-9
    Delta_x          = number_of_steps*width_of_step
    x_i              = -Xsize/2+Delta_x
    x_f              = Xsize/2-Delta_x
    i_i              = round((x_i+Xsize/2)/Cx)
    i_f              = round((x_f+Xsize/2)/Cx)

    return {"T":T, "N_t":N_t, "dt":dt, 'dk':dk,  "Delta_x":Delta_x, "x_i":x_i, "x_f":x_f,"i_i":i_i, "i_f":i_f}


def target_square_pulse(x,k0,df,sigma_f,sigma_conv,A,Amplitude_pulse): #smooth square pulse obtained as the convolution of a square and a gaussian pulse
    p      = Amplitude_pulse * A/2 * np.sqrt(2*np.pi)*sigma_conv * (special.erf( (2*x-2*df+sigma_f)/ (2*np.sqrt(2)*sigma_conv) ) + special.erf( (-2*x+2*df+sigma_f)/ (2*np.sqrt(2)*sigma_conv) )) * np.cos( x* k0)
    p_plus = Amplitude_pulse * A/2 * np.sqrt(2*np.pi)*sigma_conv * (special.erf( (2*x-2*df+sigma_f)/ (2*np.sqrt(2)*sigma_conv) ) + special.erf( (-2*x+2*df+sigma_f)/ (2*np.sqrt(2)*sigma_conv) )) * np.exp( 1j*x* k0)/2
    env    = Amplitude_pulse * A/2 * np.sqrt(2*np.pi)*sigma_conv * (special.erf( (2*x-2*df+sigma_f)/ (2*np.sqrt(2)*sigma_conv) ) + special.erf( (-2*x+2*df+sigma_f)/ (2*np.sqrt(2)*sigma_conv) ))
    return {"p":p, "p_plus":p_plus, "env":env}

def FT_pulse(k,k0,df,sigma_f,sigma_conv,A,Amplitude_pulse): #Analytical calculation of the FT of a square pulse
    C_k = Amplitude_pulse * A * sigma_conv * np.exp(-1j* (k-k0)*df-(k-k0)**2*sigma_conv**2/2)*np.sin((k-k0)*sigma_f/2)/(k-k0)
    return C_k


def disp_fit(k,polcoeffs,degree_pol_fit):
    d = degree_pol_fit
    s = 0
    for i in range(0,d+1,1):
        s+= polcoeffs[i]*k**(d-i)
    return s

def find_nearest(a, a0):
    #"Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return {"value":a.flat[idx],"index":idx}    

def back_evolved_pulse(x, t, klist, dk, Ck_list, disp_list):
    p_plus_arg = 1/np.sqrt(2*np.pi) * Ck_list * np.exp(-1j*disp_list*t)*np.exp(1j*klist*x)
    p_plus = np.sum(p_plus_arg)*dk
    p = 2*np.real(p_plus)
    return p

    


def define_target_pulse(waveguide, dispersion):
    params_target   = parameters_target_pulse(dispersion)
    k0               = params_target["k0"]
    df               = params_target["df"]
    sigmaf           = params_target["sigmaf"]
    sigma_conv       = params_target["sigma_conv"]
    A                = params_target["A"]
    Amplitude_pulse  = params_target["Amplitude_pulse"]
    omega_c          = dispersion["omega_cutoff"]
    k1               = dispersion["k1"]
    k2               = dispersion["k2"] 
    N_k              = dispersion["N_k_fit"]
    polcoeffs        = dispersion["pol_6th"]
    degree_pol_fit   = dispersion["degree_pol_fit"]

    #World size
    Xsize = waveguide["Xsize"]
    Nx = waveguide["Nx"]
   

    xlist = np.linspace(-Xsize/2,Xsize/2,Nx)      # list of positions along the wire
    klist = np.linspace(k1,k2,N_k+1)  
    dispersion_list = disp_fit(klist,polcoeffs,degree_pol_fit)
    
    target_pulse = target_square_pulse(xlist,k0,df,sigmaf,sigma_conv,A,Amplitude_pulse)['p']
    target_envelope = target_square_pulse(xlist,k0,df,sigmaf,sigma_conv,A,Amplitude_pulse)['env']
    
    #Plot the target square pulse
    cm = 1/2.54
    plt.figure(figsize=(8*cm, 5*cm))
    plt.plot(xlist*1e6, target_pulse)
    plt.plot(xlist*1e6, target_envelope,color=c['green'], linestyle='dashed')    
    plt.ylabel("$m(x,t)$")
    plt.xlabel("$x\,\, [\mu m] $")
    #plt.legend()
    plt.xlim(df*1e6-1*sigmaf*1e6,df*1e6+1*sigmaf*1e6)
    plt.savefig("plots/square_pulse/target_pulse.pdf",bbox_inches='tight')
    plt.show()
    
    Ck_list = FT_pulse(klist,k0,df,sigmaf,sigma_conv,A,Amplitude_pulse) 
    
    estim_max_k_pulse = k0 + 19/sigmaf
    
    i_k_max = find_nearest(klist, estim_max_k_pulse)["index"]
     
    P1 =   sum( np.abs(Ck_list[0:i_k_max])) /  sum( np.abs(Ck_list)) 
    print("Percentage of k's below k_max:")
    print(P1)     
    print("kmax")
    print(estim_max_k_pulse*1e-6)
    print("lambda:")
    print(2*np.pi/estim_max_k_pulse*1e9)
    
    #Plot the dispersion relation and the pulse
    plt.plot(klist/1e6, dispersion_list/(2*np.pi*1e9),label="dispersion",color=c['blue'])
    plt.plot(klist/1e6,omega_c/(2*np.pi*1e9)+0.5*Ck_list*1e9,label="FT pulse",color=c['orange'])
    plt.ylabel("$\omega/(2\pi) $ (GHz)")
    plt.xlabel("$k\,\, [\mu m^{-1}] $")
    plt.legend()
    plt.savefig("plots/square_pulse/dispersion+pulse.pdf",bbox_inches='tight')
    plt.show()
    
    np.save("simulations/square_pulse/target_pulse_analytical", target_pulse)
    np.save("simulations/square_pulse/envelope_target_pulse_analytical", target_envelope)
    np.save("simulations/square_pulse/max_k_pulse", estim_max_k_pulse)
    np.save("simulations/square_pulse/Ck_list", Ck_list)
    
    return {"target_pulse":target_pulse, "target_envelope":target_envelope, "Ck":Ck_list}


def backwards_evolution_target_pulse(waveguide, dispersion,target_pulse):
    params_target    = parameters_target_pulse(dispersion)
    k0               = params_target["k0"]
    df               = params_target["df"]
    sigmaf           = params_target["sigmaf"]
    sigma_conv       = params_target["sigma_conv"]
    A                = params_target["A"]
    Amplitude_pulse  = params_target["Amplitude_pulse"]
    omega_c          = dispersion["omega_cutoff"]
    k1               = dispersion["k1"]
    k2               = dispersion["k2"] 
    N_k              = dispersion["N_k_fit"]
    dk               = dispersion["dk"] 
    polcoeffs        = dispersion["pol_6th"]
    degree_pol_fit   = dispersion["degree_pol_fit"]
    t0               = params_target["t0"]

    #World size
    Xsize = waveguide["Xsize"]
    Nx = waveguide["Nx"]
    Cx = Xsize/Nx

    xlist = np.linspace(-Xsize/2,Xsize/2,Nx)      # list of positions along the wire
    klist = np.linspace(k1,k2,N_k+1)  
    dispersion_list = disp_fit(klist,polcoeffs,degree_pol_fit)
    
    target_pulse_vs_x = target_pulse['target_pulse']
    target_envelope   = target_pulse['target_envelope']
    Ck_list = target_pulse['Ck']
    
    '''
    #Plot the target square pulse
    plt.plot(xlist*1e6, target_pulse_vs_x,label="$t=t_f$")
    plt.ylabel("$m(x,t)$")
    plt.xlabel("$x\,\, [\mu m] $")
    plt.legend()
    plt.xlim(df*1e6-4*sigmaf*1e6,df*1e6+4*sigmaf*1e6)
    plt.savefig("plots/square_pulse/target_pulse.pdf")
    plt.show()
    


    #Plot the dispersion relation and the pulse
    plt.plot(klist/1e6, dispersion_list/(2*np.pi*1e9),label="dispersion",color=c['blue'])
    plt.plot(klist/1e6,omega_c/(2*np.pi*1e9)+0.5*Ck_list*1e8,label="FT pulse",color=c['orange'])
    plt.ylabel("$\omega/(2\pi) $ (GHz)")
    plt.xlabel("$k\,\, [\mu m^{-1}] $")
    plt.legend()
    plt.savefig("plots/square_pulse/dispersion+pulse.pdf")
    plt.show()
    '''
    
    # Calculate backwards evolved pulse

    back_evolved_pulse_func = np.vectorize(back_evolved_pulse) 
    back_evolved_pulse_func.excluded.add(2)
    back_evolved_pulse_func.excluded.add(4)
    back_evolved_pulse_func.excluded.add(5)
    
    initial_pulse = back_evolved_pulse_func(xlist, t0, klist, dk, Ck_list, dispersion_list)
    
    #Plot the initial_pulse and target pulses
    cm = 1/2.54
    plt.figure(figsize=(8*cm, 5*cm))
    plt.plot(xlist*1e6, target_pulse_vs_x,label="$t=t_f$",color=c['blue'],linewidth=1)
    plt.plot(xlist*1e6, target_envelope,color=c['green'], linestyle='dashed',linewidth=1)    
    plt.plot(xlist*1e6, initial_pulse, label="$t=0$",color=c['orange'],linewidth=1)
    plt.ylabel("$m(x,t)$")
    plt.xlabel("$x\,\, [\mu m] $")
    plt.legend()
    plt.xlim(-Xsize*1e6/2,df*1e6+4*sigmaf*1e6)
    plt.savefig("plots/square_pulse/initial+target_pulse.pdf",transparent=True,bbox_inches='tight')
    plt.show()
    
    # Calculate the amount of pulse to the left of the antenna:
    P1 =   sum( np.abs(initial_pulse[0:int(Nx/2)])**2) /  sum( np.abs(initial_pulse)**2) 
    print("Part of the pulse to the left of the antenna:")
    print(P1)    
    
        
    # Calculate the variance of the focused pulse 
   
    P    = Cx * sum( np.abs(target_pulse_vs_x)**2 )
    E    = Cx * np.dot( xlist , np.abs(target_pulse_vs_x)**2 ) / P 
    var  = Cx * np.dot( (xlist-E)**2 , np.abs(target_pulse_vs_x)**2 ) / P 
    
    #print("Variance of the target pulse: [mu m]")
    #print(np.sqrt(var)*1e6)
    
    np.save("simulations/square_pulse/initial_pulse", initial_pulse)
    
    return {"df": df, "sigmaf":sigmaf,  "initial_pulse": initial_pulse, 
            "target_pulse": target_pulse_vs_x, "target_envelope":target_envelope , 't0':t0}
    

    
def simulate_evolution_target(waveguide, dispersion,target_pulse_initial):
    # Parameters of the pulse
    params_square   = parameters_target_pulse(dispersion)
    df               = params_square["df"]
    sigmaf           = params_square["sigmaf"]
    Amplitude_pulse  = params_square["Amplitude_pulse"]
    
    # Parameters of the material
    By    = waveguide["By"]      # Bias field along the z direction
    Aex   = waveguide["Aex"]     # exchange constant
    Ms    = waveguide["Ms"]    # saturation magnetization
    alpha = waveguide["alpha"]     # damping parameter
    gamma = waveguide["gamma"]    # gyromagnetic ratio
    mu0   = waveguide["mu0"]   # vacuum permeability

    #Parameters of the simulation
    params_sim       = params_simulation_pulse(waveguide)
    Delta_x = params_sim["Delta_x"]    
    x_i = params_sim["x_i"]
    x_f = params_sim["x_f"]
    i_i = params_sim["i_i"]
    i_f = params_sim["i_f"]    
    
    T    = params_sim["T"]
    dt   = params_sim["dt"]
    
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
        
    initial_pulse = target_pulse_initial["initial_pulse"]

    # Create a Mumax3 script for the simulation of the evolution of the pulse
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
m = uniform(0,1,0)

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

//Range of positions along x with no damping:	
Delta_x:={Delta_x}
x_i:={x_i}
x_f:={x_f}
i_i:={i_i}
i_f:={i_f}

snapshot(regions)
snapshot(alpha)
snapshot(Aex)
snapshot(Msat)

B_ext = vector(0,{By},0) //Sets a external field

m0:=loadfile("./ground.ovf")
m.loadfile("./ground.ovf")

mz:= Crop(m.Comp(2),0,{Nx},{int(Ny/2)},{int(Ny/2+1)},{int(Nz/2)},{int(Nz/2+1)}) //Save m_z(x,0,0)


//mzgeneral:= m.Comp(2)
//mygeneral:= m.Comp(1)
//mxgeneral:= m.Comp(0)


"""
#Add a square pulse at the center of the waveguide pointing along z (homogeneous along the transverse plane)
    for i in np.arange(i_i+1,i_f):
        m_z_pulse = initial_pulse[i]
        script    = script + f"""
for l:=0; l<{Nz}; l++{{
        for p:=0; p<{Ny}; p++{{
                n:={i}
                mzpulse:={m_z_pulse}
                mypulse:=0
                mxpulse:=0
                m.SetCell(n, p, l, vector( mxpulse+m0.get(0,n,p,l) , mypulse+m0.get(1,n,p,l) , mzpulse+ m0.get(2,n,p,l)))
        }}
}}

"""
    script = script + f""" 
    
save(mz)   //Save the magnetization at t=0
AutoSave(mz,{dt}) //Save the magnetization every dt starting at t=0

run({T})

"""
    create_mumax_script(script,"simulations/square_pulse/","evolution")
    print("Script evolution.txt created")

def import_simulation_target(waveguide, ground_state, dispersion):
    Nz     = waveguide["Nz"]
    Xsize  = waveguide["Xsize"]
    Nx     = waveguide["Nx"]
    Cx     = Xsize/Nx
    Ms     = waveguide["Ms"]
    Zsize  = waveguide["Zsize"]
    Ny     = waveguide["Ny"]    
    
    params_pulse = parameters_target_pulse(dispersion)
    df     = params_pulse["df"]
    Amplitude_pulse = params_pulse["Amplitude_pulse"]
   
    params_sim = params_simulation_pulse(waveguide)
    T          = params_sim["T"]
    dt         = params_sim["dt"]
         
    mzfields=read_npy_files("m","simulations/square_pulse/evolution.out")

    # Stack all snapshots of the magnetization on top of each other
    mzfield = np.stack([mzfields[key] for key in sorted(mzfields.keys())])  #we obtain an array m_z[t,i=2,z=0,y=0,x]
    mzfield = np.transpose(mzfield, (1,4,3,2,0)) #Now we have m_z[i_z,x,y,z,t]
    
    # Select the components of m as a function of position x and time at y=z=0
    mz0     = mzfield[0,:,0,0,:]
    
    #Import the magnetization of the ground state
    ground = ground_state["ground_all"]
    mz_ground  = ground[2,int(Nz/2),int(Ny/2),int(Nz/2)]    # Obtain the z-component of the ground state at x=y=z=0
    mground_x  = ground[2,int(Nz/2),int(Ny/2),:]            # Select the z-component of the ground state vs x  at y=z=0
    
    #Build an array for the ground state magnetization as a function of x and time
    mground_xt = np.stack([mground_x for key in sorted(mzfields.keys())]) 
    mground_xt = np.transpose(mground_xt, (1,0)) 


    N_treal = np.shape(mz0)[1]
    Treal   = dt*(N_treal-1)
    tlist_real = np.linspace(0,Treal,N_treal)    
    xlist = np.linspace(-Xsize/2,Xsize/2,Nx)   # list of positions along the wire
    

 
    np.save("simulations/square_pulse/m(x,t)", mz0)
    np.save("simulations/square_pulse/tlist", tlist_real)
    np.save("simulations/square_pulse/T",  Treal)    
    np.save("simulations/square_pulse/dt", dt)
    np.save("simulations/square_pulse/mzground_all", mground_xt)
    np.save("simulations/square_pulse/Amplitude_pulse", Amplitude_pulse)
      
    
    return {"t":tlist_real, "m(x,t)": mz0, "dt":dt, "T":Treal, "N_t":N_treal,
            "mzground_all":mground_xt}    


def show_evolution_target(waveguide, ground_state, dispersion, target_pulse_initial):
    Nz     = waveguide["Nz"]
    Xsize  = waveguide["Xsize"]
    Nx     = waveguide["Nx"]
    Cx     = Xsize/Nx
    Ms     = waveguide["Ms"]
    Zsize  = waveguide["Zsize"]
    Ny     = waveguide["Ny"]    
    sigma_a    = waveguide["sigma_antenna"]
    
    params_pulse = parameters_target_pulse(dispersion)
    k0     = params_pulse["k0"]
    df     = params_pulse["df"]
    sigmaf = params_pulse["sigmaf"]
    A      = params_pulse["A"]
    sigma_conv  = params_pulse["sigma_conv"]
    Amplitude_pulse = params_pulse["Amplitude_pulse"]
    t0     = params_pulse['t0']
   
    params_sim = params_simulation_pulse(waveguide)
    T          = params_sim["T"]
    dt         = params_sim["dt"]
    
    #Target pulse and backwards evolved pulse
    initial_pulse   = target_pulse_initial["initial_pulse"]
    target_pulse    = target_pulse_initial["target_pulse"]
    target_envelope = target_pulse_initial["target_envelope"]
    
    # Import the generated magnetization
    mz0     = np.load("simulations/square_pulse/m(x,t).npy")
    mground_xt  = np.load("simulations/square_pulse/mzground_all.npy")
    
    N_treal = np.shape(mz0)[1]
    Treal   = dt*(N_treal-1)
    tlist = np.linspace(0, Treal, N_treal, endpoint = True)
    xlist = np.linspace(-Xsize/2,Xsize/2,Nx)   # list of positions along the wire    
    x0_point = 0
    #i_point = round((0 + Xsize/2 )/Cx)
    x0_in_list = find_nearest(xlist, x0_point)["value"]
    i_point   = find_nearest(xlist, x0_point)["index"]
    
    # Time of backwards evolution
    t0_in_list = find_nearest(tlist, -t0)["value"]
    i_t0       = find_nearest(tlist, -t0)["index"]
    
    # Time and position for calculating the fidelity at (x1,+inf)
    t1 = 50*1e-9
    x1 = 0*1e-6

    #Initial magnetization

    # Show the pulse at t=0 at the plane x=0
    cm = 1/2.54
    plt.figure(figsize=(8*cm, 5*cm))
    plt.plot(1e6*xlist,mz0[:,0],label="i=x")
    plt.ylabel("$m_z(x,0)$")
    plt.xlabel("x [$\mu$m]")
    plt.ylim(-Amplitude_pulse,1.1*Amplitude_pulse)
    plt.savefig("plots/square_pulse/m(x)_initial.pdf",bbox_inches='tight')
    plt.show()


    #Plot the generated pulse in time domain
    
    m_pulse = mz0-mground_xt
  
    # Show the intensity plot of m_z(x,t) at y=z=0
    cm = 1/2.54
    plt.figure(figsize=(9*cm, 5*cm))
    extent = [- 1e6*Xsize/2, 1e6*Xsize/2, 0, 1e9*Treal]  # extent of k values and frequencies
    #plt.imshow(np.transpose(np.abs(m_pulse)**2), extent=extent, aspect='auto', origin='lower', cmap="viridis")
    plt.imshow(np.transpose(np.abs(m_pulse)**2), extent=extent, aspect='auto', origin='lower', cmap="viridis",vmin=0,vmax=0.0000015)
    plt.ylabel("t [ns]")
    plt.xlabel("x [$\mu$m]")
    plt.colorbar()
    plt.title("$|m_z(x,y=0,z=0,t)|^2$" )
    plt.xlim(-20,20)
    plt.ylim(0,120)  
    plt.vlines(0,0,150,color="white", linestyle="dashed",linewidth=0.8) 
    plt.hlines(-t0*1e9,-30,30,color=c["green"],linestyle="dashed",linewidth=0.8) 
    plt.savefig("plots/square_pulse/evolution+lines_t0.pdf",dpi=600,bbox_inches='tight')
    plt.show()
    
    
    
    # Show m_z(x,t) at x=0 
    
    cm = 1/2.54
    plt.figure(figsize=(8*cm, 5*cm))
    plt.plot(1e9*tlist,m_pulse[i_point,:])
    plt.ylabel("$m_z(x,t)$")
    plt.xlabel("t [ns]")
    #plt.ylim(-0.03,0.03)    
    #plt.xlim(-35,20)
    #plt.legend()    
    plt.savefig("plots/square_pulse/m(0,t).pdf")
    plt.show()    
    
   
    
    t1_in_list = find_nearest(tlist, t1)["value"]
    i_t_half   = find_nearest(tlist, t1)["index"]
    time_pos   = tlist[i_t_half:] 
    

    x1_in_list = find_nearest(xlist, x1)["value"]
    i_x_half   = find_nearest(xlist, x1)["index"]
    xlist_pos  = xlist[i_x_half:] 
    
    m_pulse_pos = m_pulse[i_x_half:,:]
    
      
    
    
    '''
    #Calculate the gaussian fit of the target pulse
    
    target_pulse_pos = target_pulse[i_x_half:]
    parameters_fit_gaussian, covariance_gaussian  = optimize.curve_fit(fit_gaussian_function, xlist_pos*1e6 , target_pulse_pos,  p0=(0.005, 10, 2,  15) )

    A_fit = parameters_fit_gaussian[0]
    df_fit = parameters_fit_gaussian[1]
    sigma_f_fit = parameters_fit_gaussian[2]
    k0_fit         = parameters_fit_gaussian[3]
    pulse_gaussian_fit = fit_gaussian_function( xlist_pos*1e6, A_fit, df_fit, sigma_f_fit, k0_fit)
    envelope_gaussian_fit = gaussian_envelope_fitted( xlist_pos*1e6, A_fit, df_fit, sigma_f_fit, k0_fit)
    relative_error_gaussian_2_fit = sum( (target_pulse_pos-pulse_gaussian_fit)**2 )/ sum( target_pulse**2 )
    
    print("relative error gaussian fit to target pulse:")
    print(relative_error_gaussian_2_fit)   
    
    #plt.plot(xlist_pos*1e6, np.real(pulse_gaussian_fit),color=c['orange'])
    plt.plot(xlist_pos*1e6, np.real(target_pulse_pos),color=c['green'])
    plt.plot(xlist_pos*1e6, np.real(envelope_gaussian_fit),color=c['blue'],linestyle='dashed')
    plt.ylabel("$M(x,t)/M_s$")
    plt.xlabel("$x\,\, [\mu m] $")
    plt.xlim(5,15)
    plt.ylim(-Amplitude_pulse, Amplitude_pulse)
    plt.savefig("plots/square_pulse/envelope_fit_gaussian_target.pdf")
    plt.show()
    '''
 
    
           
    fidelity_list = []
    fidelity_tlist = []
    
    #Calculate the fidelity vs time
    
    for i_t in range(find_nearest(tlist, -t0-25*dt)["index"],find_nearest(tlist,-t0+25*dt)["index"]+1,1):
        t = i_t*dt
        pulse_simulated = m_pulse_pos[:,i_t]
        pulse_target    = target_pulse[i_x_half:] 
        
        overlap_gg = Cx * np.dot(np.conjugate(pulse_simulated),pulse_simulated)
        overlap_tt = Cx * np.dot(np.conjugate(pulse_target),pulse_target)
        overlap_gt = Cx * np.dot(np.conjugate(pulse_simulated),pulse_target)
        overlap = overlap_gt/np.sqrt(overlap_gg*overlap_tt)
        fidelity = overlap**2
        
        fidelity_list.append(fidelity)
        fidelity_tlist.append(t)

    fidelity_list = np.array(fidelity_list)
    fidelity_tlist = np.array(fidelity_tlist)
    
   
    
    #Time of maximum fidelity
    fidelity_max = np.amax(fidelity_list)
    i_fidelity_max = find_nearest(fidelity_list, fidelity_max)["index"]
    time_fidelity_max = fidelity_tlist[i_fidelity_max]
    i_time_fidelity_max = find_nearest(tlist, time_fidelity_max)["index"]
    
  
    
    # Show the fidelity vs time
    
    plt.figure(figsize=(8*cm, 5*cm))
    plt.scatter(1e9*fidelity_tlist,fidelity_list,color=c["blue"],s=8)
    plt.plot(1e9*fidelity_tlist,fidelity_list,color=c["blue"])
    plt.ylabel("$\mathcal{F}(t)$")
    plt.xlabel("t [ns]")
    plt.ylim(0,1)       
    plt.xlim((-t0-15*dt)*1e9,(-t0+15*dt)*1e9)
    plt.vlines(t0_in_list*1e9,0,5,color=c["green"],linestyle='dashed')
    plt.vlines(time_fidelity_max*1e9,0,5,color=c["orange"],linestyle='dashed')
    #plt.legend()    
    plt.savefig("plots/square_pulse/fidelity_vs_time.pdf",bbox_inches='tight')
    plt.show()
    

    print("Maximum fidelity:")
    print(fidelity_max)
    
    # Show m_z(x,t_f) at t=0 and t=tf
    
    mpulse_max_fidelity = m_pulse[i_x_half:,i_time_fidelity_max]
    target_pulse_half   = target_pulse[i_x_half:]

    plt.figure(figsize=(8*cm, 5*cm))
    plt.plot(1e6*xlist_pos,mpulse_max_fidelity,label="generated",color=c["blue"]) 
    plt.plot(1e6*xlist_pos,target_pulse_half,label="target",color=c["green"])       
    plt.ylabel("$m_z(x,t)$")
    plt.xlabel("x [$\mu$m]")
    plt.ylim(-Amplitude_pulse/2 , Amplitude_pulse/2)  
    plt.xlim(8,12)
    plt.legend()    
    plt.savefig("plots/square_pulse/pulses_at_maximum_fidelity.pdf",bbox_inches='tight')
    plt.show() 
    
    #Calculate the mean position and variance at maximum fidelity
    
    #__________________Determine the time of maximum compression 
    
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
    var_list = var_func(time_pos,xlist_pos,m_pulse_pos)        
    
    
        
    plt.figure(figsize=(8*cm, 5*cm))
    plt.plot(1e9*time_pos,1e6*np.sqrt(var_list),label="generated",color=c["red"])        
    plt.ylabel("$\sigma(t)$ [$\mu$m]")
    plt.xlabel("t [ns]")
    #plt.vlines(time_var_min*1e9,0.5,2.6, color=c["blue"],linestyle="dashed")
    plt.vlines(-t0*1e9,0.5,2.6, color=c["green"],linestyle="dashed")
    plt.xlim(t1*1e9,130)
    plt.ylim(0.5,3)
    #plt.hlines(np.sqrt(var_target_value)*1e6,60,120, color=c["green"],linestyle="dashed")
    plt.savefig("plots/square_pulse/stardard_deviation(t).pdf",bbox_inches='tight')
    plt.show()   

    
    pos_max_fidelity = Exp(time_fidelity_max,xlist_pos,m_pulse_pos) 
    var_max_fidelity = var(time_fidelity_max,xlist_pos,m_pulse_pos) 
    
    print("time [ns], position [mu m] and variance [mu m] at maximum fidelity:")
    print(time_fidelity_max*1e9,1e6*pos_max_fidelity,np.sqrt(var_max_fidelity)*1e6)
    
    x_nearest = find_nearest(xlist_pos, pos_max_fidelity)["value"]
    i_x = int( x_nearest/Cx)  
    
    plt.figure(figsize=(8*cm, 5*cm))
    plt.plot(1e9*tlist,np.abs(m_pulse_pos[i_x,:])**2,label="generated",color=c["red"])        
    plt.ylabel("$m_z(x_f,t)$")
    plt.xlabel("t [ns]")
    #plt.vlines(pos_max_fidelity*1e6,0,10, color=c["blue"],linestyle="dashed")
    #plt.ylim(0,10)
    plt.xlim(65,75)
    #plt.hlines(np.sqrt(var_time_max_F)*1e9,0,30, color=c["green"],linestyle="dashed")
    plt.savefig("plots/square_pulse/pulse_xf_vs_time.pdf",bbox_inches='tight')
    plt.show() 
    
    plt.figure(figsize=(8*cm, 5*cm))
    plt.plot(1e9*tlist,np.abs(m_pulse_pos[i_x,:])**2,label="generated",color=c["red"])        
    plt.ylabel("$m_z(x_f,t)$")
    plt.xlabel("t [ns]")
    #plt.vlines(pos_max_fidelity*1e6,0,10, color=c["blue"],linestyle="dashed")
    plt.ylim(0,0.2*1e-7)
    #plt.xlim(65,75)
    #plt.hlines(np.sqrt(var_time_max_F)*1e9,0,30, color=c["green"],linestyle="dashed")
    plt.savefig("plots/square_pulse/pulse_xf_vs_time_all.pdf",bbox_inches='tight')
    plt.show()
    
    cm = 1/2.54
    plt.figure(figsize=(9*cm, 5*cm))
    extent = [- 1e6*Xsize/2, 1e6*Xsize/2, 0, 1e9*Treal]  # extent of k values and frequencies
    plt.imshow(np.transpose(np.abs(m_pulse)**2), extent=extent, aspect='auto', origin='lower', cmap="viridis",vmin=0,vmax=0.0000015)
    plt.ylabel("t [ns]")
    plt.xlabel("x [$\mu$m]")
    plt.colorbar()
    plt.title("$|m_z(x,y=0,z=0,t)|^2$" )
    plt.vlines(pos_max_fidelity*1e6,0,150,color="white", linestyle="dashed",linewidth=0.5)  
    plt.hlines(time_fidelity_max*1e9,-30,30,color="white", linestyle="dashed",linewidth=0.5)  
    plt.xlim(7,13)
    plt.ylim(66,74)
    #plt.hlines(20,-40,40,color="white", linestyle="dashed")
    plt.savefig("plots/square_pulse/m_abs_created+lines_detail.pdf",bbox_inches='tight',dpi=300)
    plt.show()
    
    def var_time(x,tlist,pulse):     #Calculate the variance of the pulse  
        x_nearest = find_nearest(xlist_pos, x)["value"]
        i_x = int( x_nearest/Cx)
        P   = dt * sum( np.abs(pulse[i_x,:])**2 )
        E   = dt * np.dot( tlist, np.abs(pulse[i_x,:])**2) / P 
        
        #print("Expectation value of time:")
        #print(E*1e9)
        var = dt * np.dot( (tlist-E)**2 , np.abs(pulse[i_x,:])**2 ) / P 
        return var
    var_time_func = np.vectorize(var_time)
    var_time_func.excluded.add(1)
    var_time_func.excluded.add(2)
    var_time_list = var_time_func(xlist_pos,tlist,m_pulse_pos)    
    
    var_time_max_F = var_time_func(pos_max_fidelity, tlist, m_pulse_pos)    
   
    
    plt.figure(figsize=(8*cm, 5*cm))
    plt.plot(1e6*xlist_pos,1e9*np.sqrt(var_time_list),label="generated",color=c["red"])        
    plt.ylabel("std. dev. in time $(x)$ [$\mu$m]")
    plt.xlabel("x [$\mu$m]")
    plt.vlines(pos_max_fidelity*1e6,0,10, color=c["blue"],linestyle="dashed")
    plt.ylim(0,10)
    plt.xlim(0,30)
    plt.hlines(np.sqrt(var_time_max_F)*1e9,0,30, color=c["green"],linestyle="dashed")
    plt.savefig("plots/square_pulse/stardard_deviation_time(t).pdf",bbox_inches='tight')
    plt.show()  
    
     
    print("time variance [ns] at the central position of the pulse of maximum fidelity:")
    print(np.sqrt(var_time_max_F)*1e9)
    
    
    
    #Find the square fit of the pulse at the time of maximum fidelity
    
    #__________________Calculate the smooth square fit to the pulse at a given time t
    def fit_function(x, A_fit,df_fit,sigma_f_fit,  k0_fit, sigma_conv_fit):
        y = A_fit*np.sqrt( np.pi/2) * (special.erf( (2*x-2*df_fit+sigma_f_fit)/ (2*np.sqrt(2)*sigma_conv_fit) ) + special.erf( (-2*x+2*df_fit+sigma_f_fit)/ (2*np.sqrt(2)*sigma_conv_fit) )) *np.cos( x* k0_fit)
        return y
   
    def envelope_fitted(x, A_fit,df_fit,sigma_f_fit, k0_fit, sigma_conv_fit):
        y = A_fit*np.sqrt( np.pi/2) * (special.erf( (2*x-2*df_fit+sigma_f_fit)/ (2*np.sqrt(2)*sigma_conv_fit) ) + special.erf( (-2*x+2*df_fit+sigma_f_fit)/ (2*np.sqrt(2)*sigma_conv_fit) ))
        return y
    
    def find_square_fit(t, fit_function, xlist_pos, m_pulse_pos, tlist):
        
        t_in_list = find_nearest(tlist, t)["value"]
        i_t       = find_nearest(tlist, t)["index"]
        pulse_simulated = m_pulse_pos[:,i_t]
        
        #Lower bounds for the fit (in m[mu m])
        A_fit_l          = Amplitude_pulse/100
        df_fit_l         = df*1e6 - 2
        sigma_f_fit_l    = 0
        k0_fit_l         = k0*1e-6 - 2
        sigma_conv_fit_l = 0
        
        #Upper bounds for the fit (in m[mu m])
        A_fit_u          = 2*Amplitude_pulse
        df_fit_u         = df*1e6 + 2
        sigma_f_fit_u    = 5*sigmaf*1e6
        k0_fit_u         = k0*1e-6 + 2
        sigma_conv_fit_u = 1
        
        parameters_fit, covariance  = optimize.curve_fit(fit_function, xlist_pos*1e6 , pulse_simulated, bounds=((A_fit_l, df_fit_l, sigma_f_fit_l,  k0_fit_l, sigma_conv_fit_l), (A_fit_u, df_fit_u, sigma_f_fit_u,  k0_fit_u, sigma_conv_fit_u)))
        p_err  = np.sqrt(np.diag(covariance))
        A_fit = parameters_fit[0]
        df_fit = parameters_fit[1]
        sigma_f_fit = parameters_fit[2]
        k0_fit         = parameters_fit[3]
        sigma_conv_fit = parameters_fit[4]
        pulse_fit = fit_function( xlist_pos*1e6, A_fit, df_fit, sigma_f_fit,  k0_fit, sigma_conv_fit)
        envelope_fit = envelope_fitted( xlist_pos*1e6, A_fit, df_fit, sigma_f_fit,   k0_fit, sigma_conv_fit)
    
        # Do the fit starting with the values obtained above
        
        parameters_fit, covariance  = optimize.curve_fit(fit_function, xlist_pos*1e6 , pulse_simulated, p0=(A_fit, df_fit, sigma_f_fit,  k0_fit, sigma_conv_fit) )
        p_err  = np.sqrt(np.diag(covariance))
        A_fit = parameters_fit[0]
        df_fit = parameters_fit[1]
        sigma_f_fit = parameters_fit[2]
        k0_fit         = parameters_fit[3]
        sigma_conv_fit = parameters_fit[4]
        pulse_fit = fit_function( xlist_pos*1e6, A_fit, df_fit, sigma_f_fit,  k0_fit, sigma_conv_fit)
        envelope_fit = envelope_fitted( xlist_pos*1e6, A_fit, df_fit, sigma_f_fit,   k0_fit, sigma_conv_fit)
        relative_error_fit = sum( (pulse_simulated-pulse_fit)**2 )/ sum( pulse_simulated**2 )
         
        '''
        #plt.plot(xlist_pos*1e6, np.real(pulse_fit),color=c['orange'])
        plt.figure(figsize=(8*cm, 5*cm))
        plt.plot(xlist_pos*1e6, np.real(pulse_simulated),color=c['green'])
        plt.plot(xlist_pos*1e6, np.real(envelope_fit),color=c['blue'],linestyle='dashed')
        plt.ylabel("$M(x,t)/M_s$")
        plt.xlabel("$x\,\, [\mu m] $")
        plt.ylim(-Amplitude_pulse/2, Amplitude_pulse/2)
        plt.xlim(5,15)
      #  plt.savefig(f"""plots/square_pulse/envelope_fit_t={round(t*1e9)}ns.pdf""")
        plt.show()
        '''
        return {"pulse":pulse_simulated, "pulse_fit":pulse_fit, "envelope_fit":envelope_fit, "rel_error":relative_error_fit}
    
   
    
    # Plot the pulse at maximum fidelity together with its fit
    N_normalize_simulated  = np.sqrt( Cx * sum( np.abs(mpulse_max_fidelity)**2 ) )
    N_normalize_target     = np.sqrt( Cx * sum( np.abs(target_pulse_half )**2 ) )

    
    square_fit_t_max_F = find_square_fit(time_fidelity_max, fit_function, xlist_pos, m_pulse_pos, tlist)  
    envelope_square_fit   = square_fit_t_max_F["envelope_fit"]
    N_normalize_envelope  = np.sqrt( Cx * sum( np.abs(envelope_square_fit )**2 ) )

    
    
          
    plt.figure(figsize=(9*cm, 5*cm))
    plt.plot(1e6*xlist_pos,mpulse_max_fidelity/N_normalize_simulated *1e-3,label="generated",color=c["blue"]) 
    plt.plot(1e6*xlist_pos,target_pulse_half/N_normalize_target*1e-3,label="target",color=c["green"],linestyle='dashed')       
    #plt.plot(1e6*xlist_pos,envelope_square_fit/N_normalize_envelope*1e-3,color=c["green"],linestyle='dashed')       
    plt.ylabel("$m_z(x,t)/N$ [$\mu$m]$^{-1/2}$")
    plt.xlabel("x [$\mu$m]")
    plt.xlim(8,12)
    plt.legend()    
    plt.savefig("plots/square_pulse/pulses_at_maximum_fidelity_N.pdf",bbox_inches='tight')
    plt.show() 
    
      
   # Save the spectrum of the magnetization as a function of x and t

    np.save("simulations/square_pulse/m(x,t)", mz0)
    np.save("simulations/square_pulse/m(x,t)_alone", m_pulse)
    np.save("simulations/square_pulse/m(x0,t)", mz0[i_point,:])
    np.save("simulations/square_pulse/m(x0,t)_alone", m_pulse[i_point,:])    
    np.save("simulations/square_pulse/tlist", tlist)

    np.save("simulations/square_pulse/T",  Treal)    
    np.save("simulations/square_pulse/dt", dt)
       
    np.save("simulations/square_pulse/time_max_fidelity", time_fidelity_max)     
    np.save("simulations/square_pulse/m(x,time_max_fidelity)", m_pulse[:,i_time_fidelity_max])     
    np.save("simulations/square_pulse/pos_at_time_max_fidelity", pos_max_fidelity)   
    np.save("simulations/square_pulse/var_at_time_max_fidelity", var_max_fidelity)   
    np.save("simulations/square_pulse/fidelity_evolution", fidelity_max)   
    np.save("simulations/square_pulse/pulse_vs_time_at_xf", m_pulse_pos[i_x,:])   
    np.save("simulations/square_pulse/xlist",xlist)     
          
    np.save("simulations/square_pulse/envelope_square_fit",envelope_square_fit)   
    
    
    
    return {"t":tlist, "m(x,t)": mz0, "m(x0,t)": mz0[i_point,:], "m(x0,t)_alone":mz0[i_point,:]-mground_xt[i_point,:], 
            "m_ground(x0,t)":mground_xt[i_point,:], "dt":dt, "T":Treal, "N_t":N_treal, 'x1':x1, 't1':t1, 'Amplitude_pulse':Amplitude_pulse}    






