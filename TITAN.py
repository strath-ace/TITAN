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

    #The mass input in the options file is given for one vehicle/assembly
    if options.vehicle:
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

    # Apply commands at t=0 so initial output/visualisation starts consistent
    for ass in titan.assembly:
        ass.update_geometry()

    if options.dynamic_plots:
        plot = dynamic_plots.initialise_figs(titan, options)

    if options.postproc_in_loop is not None:
        import numpy as np
        import pandas as pd
        import os
        pp_existing = np.array([])
        i_time = 0

    while titan.iter < options.iters:

        options.current_iter = titan.iter
        options.high_fidelity_flag = False

        # 1) Control (optional)
        if hasattr(titan, "controlsystem") and titan.controlsystem is not None:
            titan.controlsystem.step(titan, options)

        # 2) Geometry update BEFORE fragmentation
        for ass in titan.assembly:
            ass.update_geometry()

        # 3) Fragmentation
        fragmentation.fragmentation(titan=titan, options=options)

        if not titan.assembly:
            print("Empty assembly. Aborting.")
            return

        # 2) Adaptive timestep
        if options.time_counter > 0:
            options.dynamics.time_step = options.collision.post_fragmentation_timestep
            options.time_counter -= 1
        else:
            options.dynamics.time_step = options.user_time

        # 3) Dynamics
        if 'legacy' in options.dynamics.propagator:
            dynamics.integrate(titan=titan, options=options)
        else:
            propagation.propagate(titan=titan, options=options)

        if hasattr(titan, 'end_trigger'):
            return


        # Thermal Computation
        if options.thermal.ablation:
            thermal.compute_thermal(titan=titan, options=options)
        # ==================================================
        # Structural dynamics (FEniCS / FEM) — FIXED
        # ==================================================
        coupling_freq = 50

        if titan.iter == 0 or (titan.iter + 1) % coupling_freq == 0:

            if stress_solver:
                try:
                    # --- 1. Velocity ---
                    try:
                        v_raw = titan.assembly[0].velocity
                        v = float(np.linalg.norm(v_raw))
                    except:
                        v = 6500.0

                    # --- 2. Density ---
                    try:
                        rho_raw = titan.freestream.rho
                        if isinstance(rho_raw, np.ndarray):
                            rho = float(rho_raw.flatten()[0])
                        else:
                            rho = float(rho_raw)
                    except:
                        rho = 0.001

                    # --- 3. Time ---
                    try:
                        current_t = float(titan.time)
                    except:
                        current_t = float(titan.iter * options.dynamics.time_step)

                    # --- 4. Pressure ---
                    current_p = float(0.5 * rho * (v**2) * 1.84)

                    # --- 5. Solve ---
                    stress_solver.solve_step(current_p, current_t)

                    print(f"[FEniCS] Stress: P={current_p:.0f} Pa at t={current_t:.2f}s")

                except Exception as e:
                    print(f"[FEniCS ERROR] {e}")
                    traceback.print_exc()

            else:
                structural.run_FENICS(titan=titan, options=options)

            # keep your merged feature
            output.generate_volume_solution(titan=titan, options=options)

        # Output Generation
        if options.current_iter % options.output_freq == 0:
            output.generate_surface_solution(titan=titan, options=options, iter_value=titan.iter)

        output.iteration(titan=titan, options=options)

        if options.dynamic_plots:
            for _assembly in titan.assembly:
                plot = dynamic_plots.update_plot(_assembly, plot, titan.time)

        if options.postproc_in_loop is not None:
            if not os.path.exists(options.output_folder + '/Dense_surface_solution') and os.path.exists(options.output_folder + '/Data/data.csv'):
                data = pd.read_csv(options.output_folder + '/Data/data.csv', index_col=False)
                data_obj = pd.read_csv(options.output_folder + '/Data/data_assembly.csv', index_col=False)
                iter_interval = np.unique(data['Iter'].to_numpy())
                iters_to_run = iter_interval[~np.isin(iter_interval, pp_existing)]
                for iter_value in range(min(iters_to_run), max(iters_to_run) + 1, options.output_freq):
                    pp.generate_visualization(options, data, iter_value, options.postproc_in_loop, None, data_obj)
                pp_existing = np.hstack((pp_existing, iters_to_run))

            if os.path.exists(options.output_folder + '/Data/data_smooth.csv'):
                data_smooth = pd.read_csv(options.output_folder + '/Data/data_smooth.csv')
                times = np.unique(data_smooth['Time'].to_numpy())
                times_to_run = times[~np.isin(times, pp_existing)]
                for time in times_to_run:
                    pp.generate_visualization(options, data_smooth, np.round(time, 6), options.postproc_in_loop, is_dense=True, iter_override=i_time)
                    i_time += 1
                pp_existing = np.hstack((pp_existing, times_to_run))



        titan.iter += 1
        titan.post_event_iter += 1
        options.current_iter = titan.iter

        # Save state periodically
        if options.current_iter % options.save_freq == 0 or options.high_fidelity_flag:
            options.save_state(titan, options.current_iter)
            
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
        print("Finished simulation\n")
        #print(titan.nfeval)
        return options, titan

    #Postprocess of the simulated solution to pass from Body-frame
    #to ECEF-Frame or Wind-Frame
    if postprocess:
        Path(options.output_folder+'/Postprocess/').mkdir(parents=True, exist_ok=True)
        pp.postprocess(options, postprocess, filter_name)
    if emissions:
        Path(options.output_folder+'/Postprocess_emissions/').mkdir(parents=True, exist_ok=True)
        pp_emissions.postprocess_emissions(options)
    
if __name__ == "__main__":
    output.TITAN_information()
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)

    parser.add_argument("-c", "--config",
                        dest="configfilename",
                        type=str,
                        help="input config file",
                        metavar="configfile")
    parser.add_argument("-pp", "--postprocess",
                        dest="postprocess",
                        type=str,
                        help="simulation postprocess (ECEF, WIND)",
                        metavar="postprocess")
    parser.add_argument("-MC", "--montecarlo",
                        dest="n_samples",
                        type=int,
                        help = "run a Monte Carlo campaign of N simulations",
                        metavar="n_samples")
    parser.add_argument("-flt", "--filter",
                        dest="filtername",
                        type=str,
                        help="filter postprocess (name of the object)",
                        metavar="filtername")
    parser.add_argument("-em", "--emissions",
                        dest="emissions",
                        action="store_true")
    
    args=parser.parse_args()

    if not args.configfilename:
        raise Exception('The user needs to provide a file!.\n')

    filename = args.configfilename
    postprocess = args.postprocess
    filter_name = args.filtername
    emissions = args.emissions

    if args.n_samples is not None:
        from Uncertainty import MC_wrapper
        MC_wrapper.run(filename,args.n_samples)
        exit()

    if postprocess and (postprocess.lower()!="wind" and postprocess.lower()!="ecef" and postprocess.lower()!="int"):
        raise Exception("Postprocess can only be WIND, ECEF or INT")

    main(filename = filename, postprocess = postprocess, filter_name = filter_name, emissions = emissions)
