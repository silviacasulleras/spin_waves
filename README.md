# Pulse Design and Propagation in Magnonic Waveguides

## Description

This project implements a numerical framework to model and control spin-wave dynamics in rectangular magnonic waveguides. The code combines micromagnetic simulations with analytical postprocessing to design and generate tailored spin-wave pulses, including square and chirped (self-compressing) pulses.

The workflow covers waveguide characterization, pulse propagation, and inverse design of experimentally feasible driving fields, with simulations performed using **MuMax3** (GPU-accelerated program).

---

## Methodology

The main steps of the workflow are:

- Definition of waveguide geometry and material parameters  
- Ground-state micromagnetic simulation  
- Simulation and analysis of dispersion relations and transfer functions  
- Backward propagation of target pulses  
- Forward simulation of pulse generation and propagation  

Both square and chirped pulses are supported.

---

## Code Structure

- **`main.py`**  
  Runs the full pipeline: waveguide characterization, pulse evolution, pulse generation, and visualization.

- **`modules/`**  
  Modular components for simulation setup, pulse design, postprocessing, and plotting.

---

## Tech Stack

- Python  
- NumPy, SciPy  
- Matplotlib  
- MuMax3 (micromagnetic solver)  
- GPU-accelerated micromagnetic simulations

---

## Usage

1. Install **MuMax3** and required Python packages.
2. Set waveguide parameters in `modules/waveguide_params.py`.
3. Run:
   ```bash
   python main.py

---
## Related Publication

If you use this code for research purposes, please refer to:

S. Casulleras et al., "Generation of Spin-Wave Pulses by Inverse Design", Phys. Rev. Applied 19, 064085 (2023) at https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.19.064085


---
## Author
This code was developed by Silvia Casulleras at the Institute for Quantum Optics and Quantum Information (IQOQI), Austrian Academy of Sciences and University of Innsbruck, Austria.  
