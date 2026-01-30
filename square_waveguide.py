"""
Created on 19.05.2022

- Simulation of the dispersion relation and the transfer function of a rectangular waveguide.
- Simultation of the evolution of a square pulse
- Determination of the driving
- Generation of a square pulse
@author: Silvia Casulleras
"""

import numpy as np
import matplotlib.pyplot as plt

from modules.mumax3_functions         import create_mumax_script, read_npy_files
from modules.waveguide_params         import define_waveguide
from modules.ground_state             import simulate_ground_state, show_ground_state
from modules.dispersion               import simulate_dispersion_relation, extract_dispersion_relation, fit_dispersion_relation, import_dispersion_relation
from modules.transfer_function        import simulate_transfer_function, extract_transfer_function, show_transfer_function, import_transfer_function
from modules.evolution_target_square  import define_target_pulse, backwards_evolution_target_pulse, simulate_evolution_target, import_simulation_target, show_evolution_target
from modules.evolution_target_chirped import define_target_chirped_pulse, backwards_evolution_chirped_pulse, simulate_evolution_chirped, import_simulation_target_chirped, show_evolution_target_chirped
from modules.pulse_generation_square  import target_square_pulse_spectrum, determine_driving_squared, import_driving_squared, simulate_generation_squared_pulse, show_generated_squared_pulse, import_generated_squared_pulse
from modules.pulse_generation_chirped import target_chirped_pulse_spectrum, determine_driving_chirped, import_driving_chirped, simulate_generation_chirped_pulse, show_generated_chirped_pulse, import_generated_chirped_pulse

from modules.postprocessing           import FT_function, IFT_function
from modules.plots_article_chirped    import plot_dispersion_rel, plot_transfer_function, plot_sketch
from modules.plots_article_square     import plot_dispersion_rel_and_square_pulse, plot_backwards_evolved_square_pulse, plot_evolved_square, plot_driving_square_freq, plot_generated_square_pulse, plot_comparison_square
from modules.plots_article_chirped    import plot_dispersion_rel_and_chirped_pulse, plot_backwards_evolved_chirped_pulse, plot_evolved_chirped, plot_driving_chirped_freq, plot_generated_chirped_pulse, plot_comparison_chirped, plot_evolved_chirped_2


#PART 1: Characterization of the waveguide 

#Define the parameters of the waveguide
waveguide = define_waveguide()
 
#Determine the ground state
simulate_ground_state(waveguide)                                                   # create a mumax3 script to find the ground state
ground_state = show_ground_state(waveguide)                                        # show the ground state 

#Simulate the dispersion relation
simulate_dispersion_relation(waveguide)                                            # create a mumax3 script with the excitation of the spin waves
#dispersion_simulation = extract_dispersion_relation(waveguide)                    # calculate the dispersion relation by applying the 2D FT to the magnetization
#pol_dispersion        = fit_dispersion_relation(waveguide,dispersion_simulation)  # calculate the 6th order polynomial fit to the dispersion relation
dispersion            = import_dispersion_relation(waveguide)

#Simulate the transfer function at x=0
xa = 0
simulate_transfer_function(waveguide,dispersion)                                              # create a mumax3 script to simulate the transfer function
#transfer_func    = extract_transfer_function(waveguide,ground_state,dispersion,xa)            # calculate the transfer function at x=x0
#show_transfer_function(waveguide,ground_state,dispersion,xa)                    # show the transfer function in general
transfer_func_2  = import_transfer_function(xa)                                    # import the transfer function at x=x0

#PART 2: Pulse evolution

    # 1) Square pulse

target_square_pulse = define_target_pulse(waveguide, dispersion) 
target_square_pulse_initial = backwards_evolution_target_pulse(waveguide, dispersion,target_square_pulse)                                         # define a chirped pulse
simulate_evolution_target(waveguide, dispersion, target_square_pulse_initial) 
#import_simulation_target(waveguide, ground_state, dispersion)                                 # create a mumax3 script that simulates the evolution of the pulse 
evolved_square_pulse = show_evolution_target(waveguide, ground_state, dispersion,target_square_pulse_initial)                      # show the evolution of the state

    # 2) Self-compressing pulse
    
target_chirped_pulse = define_target_chirped_pulse(waveguide, dispersion) 
target_chirped_pulse_initial = backwards_evolution_chirped_pulse(waveguide, dispersion,target_chirped_pulse)                                         # define a chirped pulse
#simulate_evolution_chirped(waveguide, dispersion, target_chirped_pulse_initial) 
#import_simulation_target_chirped(waveguide, ground_state, dispersion)                                 # create a mumax3 script that simulates the evolution of the pulse 
evolved_chirped_pulse = show_evolution_target_chirped(waveguide, ground_state, dispersion,target_chirped_pulse_initial)                      # show the evolution of the state

#PART 3: Pulse generation

    # 1) Square pulse generation

pulse_square_spectrum   = target_square_pulse_spectrum(waveguide, ground_state,dispersion, evolved_square_pulse, transfer_func_2) 
#driving_squared  = determine_driving_squared(waveguide,dispersion,transfer_func_2,pulse_square_spectrum)    # calculate the driving that generates the pulse
driving_squared  = import_driving_squared(waveguide)
simulate_generation_squared_pulse(waveguide, driving_squared)                      # create a mumax3 script that generates the pulse
#generated_pulse = import_generated_squared_pulse(waveguide, ground_state, driving_squared, xa, evolved_square_pulse, pulse_square_spectrum)
show_generated_squared_pulse(waveguide, ground_state, driving_squared, xa, evolved_square_pulse, pulse_square_spectrum)

    # 2) Chirped pulse generation
    
pulse_chirped_spectrum   = target_chirped_pulse_spectrum(waveguide, ground_state,dispersion, evolved_chirped_pulse, transfer_func_2) 
#driving_chirped  = determine_driving_chirped(waveguide,dispersion,transfer_func_2,pulse_chirped_spectrum)    # calculate the driving that generates the pulse
driving_chirped  = import_driving_chirped(waveguide)
#simulate_generation_chirped_pulse(waveguide, driving_chirped)                      # create a mumax3 script that generates the pulse
#generated_pulse = import_generated_chirped_pulse(waveguide, ground_state, driving_chirped, xa, evolved_chirped_pulse, pulse_chirped_spectrum)
show_generated_chirped_pulse(waveguide, ground_state, driving_chirped, xa, evolved_chirped_pulse, pulse_chirped_spectrum)


#Generate plots for the paper

#plot_sketch(waveguide)
#plot_dispersion_rel()
#plot_transfer_function(xa)

#plot_backwards_evolved_square_pulse(waveguide,target_square_pulse_initial)
#plot_dispersion_rel_and_square_pulse()
#plot_evolved_square(waveguide,target_chirped_pulse_initial)
#plot_driving_square_freq(waveguide,dispersion)
#plot_driving_square_time(waveguide,dispersion)
#plot_generated_square_pulse(waveguide,target_square_pulse_initial)
#plot_comparison_square(waveguide)

#plot_backwards_evolved_chirped_pulse(waveguide,target_chirped_pulse_initial)
#plot_dispersion_rel_and_chirped_pulse()
#plot_evolved_chirped(waveguide,target_chirped_pulse_initial)
plot_evolved_chirped_2(waveguide,target_chirped_pulse_initial)
#plot_driving_chirped_freq(waveguide,dispersion)
#plot_driving_chirped_time(waveguide,dispersion)
#plot_generated_chirped_pulse(waveguide,target_chirped_pulse_initial)
#plot_comparison_chirped(waveguide)






'''
#Generation of a Gaussian pulse at x=0
gaussian_pulse   = define_gaussian_pulse_inc_ground(waveguide,transfer_func_2)  
#driving          = determine_driving_analytical(waveguide,transfer_func_2,gaussian_pulse) # calculate the driving that generates the pulse
driving_2        = import_driving()
simulate_generation_pulse(waveguide, driving_2)                                    # create a mumax3 script that generates the pulse
m_pulse_created  = import_generated_pulse(waveguide,driving_2,gaussian_pulse,xa)    # import the generated pulse and compare it with the gaussian pulse
show_generated_pulse(waveguide,driving_2,transfer_func_2,gaussian_pulse,xa,m_pulse_created)
'''


