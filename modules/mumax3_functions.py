# -*- coding: utf-8 -*-
"""
Post-processing of Mumax3 simulations.
Created following https://colab.research.google.com/github/JeroenMulkers/mumax3-tutorial/blob/master/postprocessing.ipynb .

Created on Mon Jan 25 11:04:36 2021

@author: Silvia Casulleras
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.fftpack as sfft
from scipy import interpolate
import math

def read_mumax3_table(filename):
    """Puts the mumax3 output table in a pandas dataframe"""

    from pandas import read_table
    
    table = read_table(filename)
    table.columns = ' '.join(table.columns).split()[1::2]
    
    return table

def read_mumax3_ovffiles(outputdir):
    """Load all ovffiles in outputdir into a dictionary of numpy arrays 
    with the ovffilename (without extension) as key"""
    
    from subprocess import run, PIPE, STDOUT
    from glob import glob
    from os import path
    from numpy import load

    # convert all ovf files in the output directory to numpy files
    p = run(["mumax3-convert","-numpy",outputdir+"/*.ovf"], stdout=PIPE, stderr=STDOUT)
    if p.returncode != 0:
        print(p.stdout.decode('UTF-8'))

    # read the numpy files (the converted ovf files)
    fields = {}
    for npyfile in glob(outputdir+"/*.npy"):
        key = path.splitext(path.basename(npyfile))[0]
        fields[key] = load(npyfile)
    
    return fields


def read_npy_files(key,outputdir):
    from glob import glob
    from os import path
    from numpy import load
    fields = {}
    for npyfile in glob(outputdir+f"""/{key}*.npy"""):
        key = path.splitext(path.basename(npyfile))[0]
        print(key)
        fields[key] = load(npyfile)

    return fields


def read_mumax3_ovffiles_ground(key):
    """Load all ovffiles in outputdir into a dictionary of numpy arrays 
    with the ovffilename (without extension) as key"""
    
    from subprocess import run, PIPE, STDOUT
    from glob import glob
    from os import path
    from numpy import load 

    # convert all ovf files in the output directory to numpy files
    p = run(["mumax3-convert","-numpy","ground.ovf"], stdout=PIPE, stderr=STDOUT)
    if p.returncode != 0:
        print(p.stdout.decode('UTF-8'))

    # read the numpy files (the converted ovf files)
    fields = {}
    for npyfile in glob("ground.npy"):
        key = path.splitext(path.basename(npyfile))[0]
        print(key)
        fields[key] = load(npyfile)

    return fields

'''
def create_mumax_script(script,name):
    scriptfile = name + ".txt" 
        # write the input script in scriptfile
    with open(scriptfile, 'w' ) as f:
        f.write(script)
'''      

def create_mumax_script(script,outdir,name):
    scriptfile = outdir + name + ".txt" 
        # write the input script in scriptfile
    with open(scriptfile, 'w' ) as f:
        f.write(script)        
        
    
def run_mumax3(script, name, verbose=False):
    """ Executes a mumax3 script and convert ovf files to numpy files
    
    Parameters
    ----------
      script:  string containing the mumax3 input script
      name:    name of the simulation (this will be the name of the script and output dir)
      verbose: print stdout of mumax3 when it is finished
    """
    
    from subprocess import run, PIPE, STDOUT
    from os import path

    scriptfile = name + ".txt" 
    outputdir  = name + ".out"

    # write the input script in scriptfile
    with open(scriptfile, 'w' ) as f:
        f.write(script)
    
    # call mumax3 to execute this script
    print("Running Mumax3...")
    p = run(["mumax3","-f",scriptfile], stdout=PIPE, stderr=STDOUT)
    if verbose or p.returncode != 0:
        print(p.stdout.decode('UTF-8'))
        
    if path.exists(outputdir + "/table.txt"):
        table = read_mumax3_table(outputdir + "/table.txt")
    else:
        table = None
        
    fields = read_mumax3_ovffiles(outputdir)
    
    return table, fields