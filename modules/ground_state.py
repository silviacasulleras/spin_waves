"""
Created on Thu Feb  4 11:20:49 2021
Simulation of the ground state of a rectangular waveguide
@author: Silvia Casulleras
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

from modules.mumax3_functions import create_mumax_script, read_npy_files

#n=4
#colors = plt.cm.viridis(np.linspace(0,1,n))
c = {"blue":'#0072BD', "orange": '#D95319', "green":"#77AC30", "yellow": '#EDB120', "purple":'#7E2F8E', "red":"#A2142F", "light-blue":"4DBEEE" }
   

def simulate_ground_state(waveguide):
   
    #Define world size
    Xsize = waveguide["Xsize"]
    Ysize = waveguide["Ysize"]
    Zsize = waveguide["Zsize"]

    #Define number of cells
    Nx = waveguide["Nx"]
    Ny = waveguide["Ny"]
    Nz = waveguide["Nz"]

    # MATERIAL/SYSTEM PARAMETERS
    By    = waveguide["By"]      # Bias field along the z direction
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
    # Write a Mumax3 script to find the stable configuration of the waveguide
    #_________________________________________________________________________________________________________
    
    script=f"""
    OutputFormat = OVF1_TEXT
    
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
        
    B_ext=vector(0,{By},0)
    
    relax() //brings magnetization to the groundstate
    save(m)
    save(B_eff)
    
    //tableautosave(10e-12)

    """

    create_mumax_script(script,"simulations/ground_state/","ground_state")
    print("Script ground_state.txt created")

def show_m_comp_2(m,Cy,Cz,Nx,Ny,Nz,key):
    m = np.transpose(m, (0,3,2,1)) #Now we have m[mcomponent,x,y,z]
    mx = m[0,int(Nx/2),:,:] #component x of the magnetization at the plane x=0
    my = m[1,int(Nx/2),:,:] #component y of the magnetization at the plane x=0
    mz = m[2,int(Nx/2),:,:] #component z of the magnetization at the plane x=0
    
    Lcelly = Cy
    Lcellz = Cz
    GridSizey = Ny
    GridSizez = Nz
    Ly = Lcelly * GridSizey
    Lz = Lcellz * GridSizez
    # Extent=[0,Lx,0,Lz]
    plt.imshow(mx.transpose(), cmap="viridis",origin = "lower", extent=[-1e9*Ly/2,1e9*Ly/2,-1e9*Lz/2,1e9*Lz/2])
    plt.title('$m_x$')
    plt.xlabel('y [nm]')
    plt.ylabel('z [nm]')
    #ax2.set_ylabel('y (cell index)')
    #ax3.set_ylabel('y (cell index)') 
    plt.colorbar()
    plt.savefig('plots/ground_state/colorplot_'+key+'_x_planex=0.pdf')
    plt.show()    

    plt.imshow(my.transpose(), cmap="viridis",origin = "lower", extent=[-1e9*Ly/2,1e9*Ly/2,-1e9*Lz/2,1e9*Lz/2],vmin=0.995,vmax=1)
    plt.title('$m_y$')
    plt.xlabel('y [nm]')
    plt.ylabel('z [nm]')
    #ax2.set_ylabel('y (cell index)')
    #ax3.set_ylabel('y (cell index)') 
    plt.colorbar()
    plt.savefig('plots/ground_state/colorplot_'+key+'_y_planex=0.pdf')
    plt.show()    

    plt.imshow(mz.transpose(), cmap="viridis",origin = "lower", extent=[-1e9*Ly/2,1e9*Ly/2,-1e9*Lz/2,1e9*Lz/2])
    plt.title('$m_z$')
    plt.xlabel('y [nm]')
    plt.ylabel('z [nm]')
    #ax2.set_ylabel('y (cell index)')
    #ax3.set_ylabel('y (cell index)') 
    plt.colorbar()
    plt.savefig('plots/ground_state/colorplot_'+key+'_z_planex=0.pdf')
    plt.show()    



def show_m_comp_3(m,Cx,Cz,Nx,Ny,Nz,key):
    m = np.transpose(m, (0,3,2,1)) #Now we have m[mcomponent,x,y,z]
    mx = m[0,:,int(Ny/2),:] #component x of the magnetization at the plane y=0
    my = m[1,:,int(Ny/2),:] #component y of the magnetization at the plane y=0
    mz = m[2,:,int(Ny/2),:] #component z of the magnetization at the plane y=0
    
    Lcellx = Cx
    Lcellz = Cz
    GridSizex = Nx
    GridSizez = Nz
    Lx = Lcellx * GridSizex
    Lz = Lcellz * GridSizez

    # Extent=[0,Lx,0,Lz]
    plt.imshow(mx.transpose(), cmap="viridis",origin = "lower", extent=[-1e6*Lx/2,1e6*Lx/2,-1e9*Lz/2,1e9*Lz/2], aspect=1/8)
    plt.title('$m_x$')
    plt.xlabel('x [$\mu$m]')
    plt.ylabel('z [nm]')
    #ax2.set_ylabel('y (cell index)')
    #ax3.set_ylabel('y (cell index)') 
    plt.colorbar()
    plt.savefig('plots/ground_state/colorplot_'+key+'_x_planey=0.pdf')
    plt.show()    

    plt.imshow(my.transpose(), cmap="viridis",origin = "lower", extent=[-1e6*Lx/2,1e6*Lx/2,-1e9*Lz/2,1e9*Lz/2], aspect=1/8)
    plt.title('$m_y$')
    plt.xlabel('x [$\mu$m]')
    plt.ylabel('z [nm]')
    #ax2.set_ylabel('y (cell index)')
    #ax3.set_ylabel('y (cell index)') 
    plt.colorbar()
    plt.savefig('plots/ground_state/colorplot_'+key+'_y_planey=0.pdf')
    plt.show()  

    plt.imshow(mz.transpose(), cmap="viridis",origin = "lower", extent=[-1e6*Lx/2,1e6*Lx/2,-1e9*Lz/2,1e9*Lz/2], aspect=1/8)
    plt.title('$m_z$')
    plt.xlabel('x [$\mu$m]')
    plt.ylabel('z [nm]')
    #ax2.set_ylabel('y (cell index)')
    #ax3.set_ylabel('y (cell index)') 
    plt.colorbar()
    plt.savefig('plots/ground_state/colorplot_'+key+'_z_planey=0.pdf')
    plt.show()  
    

def show_m_comp_4(m,Cx,Cy,Nx,Ny,Nz,key):
    m = np.transpose(m, (0,3,2,1)) #Now we have m[mcomponent,x,y,z]
    mx = m[0,:,:,int(Nz/2)] #component x of the magnetization at the plane z=0
    my = m[1,:,:,int(Nz/2)] #component y of the magnetization at the plane z=0
    mz = m[2,:,:,int(Nz/2)] #component z of the magnetization at the plane z=0
    
    Lcellx = Cx
    Lcelly = Cy
    GridSizex = Nx
    GridSizey = Ny
    Lx = Lcellx * GridSizex
    Ly = Lcelly * GridSizey
    # Extent=[0,Lx,0,Lz]
    plt.imshow(mx.transpose(), cmap="viridis",origin = "lower", extent=[-1e6*Lx/2,1e6*Lx/2,-1e9*Ly/2,1e9*Ly/2], aspect=1/8)
    plt.title('$m_x$')
    plt.xlabel('x [$\mu$m]')
    plt.ylabel('y [nm]')
    #ax2.set_ylabel('y (cell index)')
    #ax3.set_ylabel('y (cell index)') 
    plt.colorbar()
    plt.savefig('plots/ground_state/colorplot_'+key+'_x_planez=0.pdf')
    plt.show()    

    plt.imshow(my.transpose(), cmap="viridis",origin = "lower", extent=[-1e6*Lx/2,1e6*Lx/2,-1e9*Ly/2,1e9*Ly/2], aspect=1/8)
    plt.title('$m_y$')
    plt.xlabel('x [$\mu$m]')
    plt.ylabel('y [nm]')
    #ax2.set_ylabel('y (cell index)')
    #ax3.set_ylabel('y (cell index)') 
    plt.colorbar()
    plt.savefig('plots/ground_state/colorplot_'+key+'_y_planez=0.pdf')
    plt.show() 
    
    plt.imshow(mz.transpose(), cmap="viridis",origin = "lower", extent=[-1e6*Lx/2,1e6*Lx/2,-1e9*Ly/2,1e9*Ly/2], aspect=1/8)
    plt.title('$m_z$')
    plt.xlabel('x [$\mu$m]')
    plt.ylabel('y [nm]')
    #ax2.set_ylabel('y (cell index)')
    #ax3.set_ylabel('y (cell index)') 
    plt.colorbar()
    plt.savefig('plots/ground_state/colorplot_'+key+'_z_planez=0.pdf')
    plt.show()   




def show_ground_state(waveguide):
    
    fields = read_npy_files("m", "simulations/ground_state/ground_state.out")
    
    m0=fields["m000000"]
   
    #Define world size
    Xsize = waveguide["Xsize"]
    Ysize = waveguide["Ysize"]
    Zsize = waveguide["Zsize"]

    #Define number of cells
    Nx = waveguide["Nx"]
    Ny = waveguide["Ny"]
    Nz = waveguide["Nz"]
    
    Cx=Xsize/Nx
    Cy=Ysize/Ny
    Cz=Zsize/Nz

    #Plots of the ground state magnetization
    
     #Plot the magnetization at the plane x=0
    show_m_comp_2(m0, Cy, Cz, Nx , Ny , Nz, "m000000")
    #Plot the magnetization at the plane y=0
    show_m_comp_3(m0, Cx, Cz, Nx , Ny , Nz, "m000000")
    #Plot the magnetization at the plane z=0
    show_m_comp_4(m0,Cx,Cy,Nx,Ny,Nz, "m000000")
    
    ground     = np.load("simulations/ground_state/ground_state.out/m000000.npy")  #Import the ground state of the waveguide 
    mz_ground  = ground[2,int(Nz/2),int(Ny/2),int(Nz/2)]    # Obtain the z-component of the ground state at x=y=z=0
    mground_x  = ground[2,int(Nz/2),int(Ny/2),:]            # Select the z-component of the ground state vs x  at y=z=0
 

    return {"m_ground_x":mground_x, "ground_all": ground}
