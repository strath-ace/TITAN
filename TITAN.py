import sys
import os
import pandas as pd
import numpy as np
import configparser
import traceback  
from argparse import ArgumentParser, RawTextHelpFormatter
from Configuration import configuration
from Output import output, dynamic_plots
from Dynamics import dynamics, propagation
from Fragmentation import fragmentation
from Postprocess import postprocess as pp
from Postprocess import postprocess_emissions as pp_emissions
from Thermal import thermal
from Structural import structural
from pathlib import Path

# --- FEniCSx Import ---
try:
    from cubesat_stress_module import CubeSatStressModule
    FENICS_AVAILABLE = True
except ImportError:
    print("Warning: Could not import 'cubesat_stress_module'. FEniCS features disabled.")
    FENICS_AVAILABLE = False

def loop(options = [], titan = []):
    """Simulation loop for time propagation"""

    options.current_iter = titan.iter
    options.user_time    = options.dynamics.time_step

    if hasattr(options, 'vehicle') and options.vehicle and titan.assembly:
        titan.assembly[0].mass = options.vehicle.mass   

    # ==================================================
    # 1. INITIALIZE FENICS SOLVER
    # ==================================================
    stress_solver = None
    if FENICS_AVAILABLE and getattr(options, 'fenics_active', True):
        print("--- Initializing Coupled FEniCSx Solver ---")
        try:
            stress_solver = CubeSatStressModule("cubesat_volumetric_60deg_up.xdmf")
            print("   [FEniCSx] CubeSat Structural Solver Ready.")
        except Exception as e:
            print(f"   [Error] FEniCS Initialization failed: {e}")
            stress_solver = None

    print(f"\n--- Simulation Start ---")
    print(f"   Target Iterations: {options.iters}")
    print(f"   Objects Loaded:    {len(titan.assembly)}")
    print("------------------------\n")

    while titan.iter < options.iters:
        options.high_fidelity_flag = False

        fragmentation.fragmentation(titan = titan, options = options)

        if not titan.assembly: 
            print("Empty assembly. Aborting.")
            return      

        # Dynamics Propagation
        if 'legacy' in options.dynamics.propagator: 
            dynamics.integrate(titan = titan, options = options)
        else:
            propagation.propagate(titan = titan, options = options)

        if hasattr(titan,'end_trigger'): return
        
        # Thermal Computation
        if options.thermal.ablation:
            thermal.compute_thermal(titan = titan, options = options)

        # Output Generation
        if options.current_iter % options.output_freq == 0:
            output.generate_surface_solution(titan = titan, options = options, iter_value = titan.iter)         
        
        output.iteration(titan = titan, options = options)
        
        # ==================================================
        # 2. RUN FENICS STRESS STEP
        # ==================================================
        if stress_solver and titan.assembly:
            coupling_freq = 50
            
            if titan.iter == 0 or (titan.iter + 1) % coupling_freq == 0:
                try:
                    # 1. Safely extract VELOCITY
                    try:
                        v_raw = titan.assembly[0].velocity
                        v = float(np.linalg.norm(v_raw)) 
                    except:
                        v = 6500.0 
                        
                    # 2. Safely extract DENSITY (rho)
                    try:
                        rho_raw = titan.freestream.rho
                        if isinstance(rho_raw, np.ndarray):
                            rho = float(rho_raw.flatten()[0]) 
                        else:
                            rho = float(rho_raw)
                    except:
                        rho = 0.001 

                    # 3. Ensure TIME is a pure float
                    try:
                        current_t = float(titan.time)
                    except:
                        current_t = float(titan.iter * options.dynamics.time_step)

                    # 4. Calculate pressure 
                    current_p = float(0.5 * rho * (v**2) * 1.84)
                    
                    # 5. Hand the clean numbers to FEniCS
                    stress_solver.solve_step(current_p, current_t)
                    
                    print(f"      [FEniCS] Stress calculated for P = {current_p:.0f} Pa at t = {current_t:.2f}s")
                    
                except Exception as e:
                    print(f"      [FEniCS Error] {e}")
                    # --- NEW: Print exactly where the error is happening ---
                    print("      --- Full Traceback ---")
                    traceback.print_exc()
                    print("      ----------------------")

        titan.iter += 1
        options.current_iter = titan.iter
            
    if stress_solver:
        stress_solver.close()
        print("Coupled Simulation Complete.")

def main(filename = "", postprocess = "", filter_name = None, emissions = ""):
    configParser = configparser.RawConfigParser()   
    configParser.read(filename.strip())

    options, titan = configuration.read_config_file(configParser, postprocess, emissions)
    options.filepath = filename

    try:
        options.fenics_active = configParser.getboolean('Options', 'FENICS')
    except:
        options.fenics_active = False

    if (not postprocess) and (not emissions):
        loop(options, titan)
        print("Finished simulation")
    
if __name__ == "__main__":
    output.TITAN_information()
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("-c", "--config", dest="configfilename", type=str, help="input config file")
    args=parser.parse_args()

    if not args.configfilename:
        raise Exception('The user needs to provide a file!.\n')

    main(filename = args.configfilename)