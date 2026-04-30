#
# Copyright (c) 2023 TITAN Contributors (cf. AUTHORS.md).
#
# This file is part of TITAN 
# (see https://github.com/strath-ace/TITAN).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
import configparser
from argparse import ArgumentParser, RawTextHelpFormatter
from .Configuration import configuration
from .Output import output, dynamic_plots
from .Dynamics import dynamics, propagation
from .Fragmentation import fragmentation
from .Postprocess import postprocess as pp
from .Postprocess import postprocess_emissions as pp_emissions
from .Thermal import thermal
from .Structural import structural
from pathlib import Path


def loop(options = [], titan = []):
    """Simulation loop for time propagation

    The function calls the different modules to perform
    dynamics propagation, thermal ablation, fragmentation
    assessment and structural dynamics for each time iteration.
    The loop finishes when the iteration number is higher than
    the one the user specified.

    Parameters
    ----------
    options : Options
        object of class :class:`configuration.Options`
    titan : Assembly_list
        object of class Assembly_list
    """

    #For collision testing purposes
    if "sphere-sphere.txt" in options.filepath:
        titan.assembly[0].mass = 1
        titan.assembly[1].mass = 2
        titan.assembly[0].velocity[2] = 5

    if options.structural_dynamics:
        print("Structural dynamics selected: still requiring further validation")
    #    exit("Structural dynamics is currently under development")

    options.current_iter = titan.iter
    options.user_time    = options.dynamics.time_step

    #The mass input in the options file is given for one vehicle/assembly
    if options.vehicle:
        titan.assembly[0].mass = options.vehicle.mass   

    if options.dynamic_plots: plot = dynamic_plots.initialise_figs(titan, options)
    if options.postproc_in_loop is not None: 
        import numpy as np
        import pandas as pd
        import os
        pp_existing = np.array([])
        i_time=0

    # Run main TITAN loop
    while titan.iter < options.iters:
        options.high_fidelity_flag = False
        
        # Check fragmentation case
        fragmentation.fragmentation(titan = titan, options = options)

        if not titan.assembly: return      

        # Adjust post-event counter
        if options.time_counter>0:
            options.dynamics.time_step = options.collision.post_fragmentation_timestep
            options.time_counter-=1
        else:
            options.dynamics.time_step = options.user_time

        # Perform dynamics integrations
        if 'legacy' in options.dynamics.propagator: dynamics.integrate(titan = titan, options = options)
        else:
            propagation.propagate(titan = titan, options = options)

        # Finish if integrator signals ending
        if hasattr(titan,'end_trigger'): return
        
        # Perform thermal step
        if options.thermal.ablation and not options.dynamics.augmented_state:
            thermal.compute_thermal(titan = titan, options = options)

        # Structural mechanics (not well-implemented here)
        if options.structural_dynamics and (titan.iter+1)%options.fenics.FE_freq == 0:
            #TODO
            structural.run_FENICS(titan = titan, options = options)
            output.generate_volume_solution(titan = titan, options = options)
        
        # Generate
        if options.current_iter%options.output_freq == 0:
            output.generate_surface_solution(titan = titan, options = options, iter_value = titan.iter)         
        
        output.iteration(titan = titan, options = options, show_flow_solves=options.verbose)
        
        if options.dynamic_plots:
            for _assembly in titan.assembly: plot = dynamic_plots.update_plot(_assembly, plot, titan.time)

        if options.postproc_in_loop is not None:
            if not os.path.exists(options.output_folder+'/Dense_surface_solution') and os.path.exists(options.output_folder+'/Data/data.csv'):
                data = pd.read_csv(options.output_folder+'/Data/data.csv', index_col = False)
                data_obj = pd.read_csv(options.output_folder+'/Data/data_assembly.csv', index_col = False)
                iter_interval = np.unique(data['Iter'].to_numpy())
                iters_to_run = iter_interval[~np.isin(iter_interval,pp_existing)]
                for iter_value in range(min(iters_to_run), max(iters_to_run)+1, options.output_freq):
                    pp.generate_visualization(options, data, iter_value, options.postproc_in_loop, None, data_obj)
                pp_existing = np.hstack((pp_existing,iters_to_run))
            elif os.path.exists(options.output_folder+'/Data/data_smooth.csv'):
                data_smooth = pd.read_csv(options.output_folder+'/Data/data_smooth.csv')
                times = np.unique(data_smooth['Time'].to_numpy())
                times_to_run = times[~np.isin(times, pp_existing)]
                for time in times_to_run:
                    pp.generate_visualization(options, data_smooth, np.round(time,6), options.postproc_in_loop,is_dense=True, iter_override=i_time)
                    i_time+=1
                pp_existing = np.hstack((pp_existing,times_to_run))


        titan.iter += 1
        titan.post_event_iter +=1
        options.current_iter = titan.iter
        if options.current_iter%options.save_freq == 0 or options.high_fidelity_flag == True:
            options.save_state(titan, options.current_iter)


def main(filename = "", postprocess = "", filter_name = None, emissions = ""):
    """TITAN main function

    Parameters
    ----------
    filename : str
        Name of the configuration file
    postprocess : str
        Postprocess method. If specified, TITAN will only perform the postprocess of the already obtained solution in the specified output folder.
        The config fille still needs to be specified.
    """

    configParser = configparser.RawConfigParser()   
    configFilePath = filename.strip()
    configParser.read(configFilePath)

    #Pre-processing phase: Creates the options and titan class
    options, titan = configuration.read_config_file(configParser, postprocess, emissions)
    options.filepath = filename

    #Initialization of the simulation
    if (not postprocess) and (not emissions):
        loop(options, titan)
        print("Finished simulation")
        if options.verbose: print('Total of {} flow solves'.format(titan.nfeval))
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

    # To run TITAN, it requires the user to specify a configuration 
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
        from .Uncertainty import MC_wrapper
        MC_wrapper.run(filename,args.n_samples)
        exit()

    if postprocess and (postprocess.lower()!="wind" and postprocess.lower()!="ecef" and postprocess.lower()!="int"):
        raise Exception("Postprocess can only be WIND, ECEF or INT")

    main(filename = filename, postprocess = postprocess, filter_name = filter_name, emissions = emissions)
