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
import sys
import configparser
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
import numpy as np
from matplotlib import pyplot as plt

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
    #if "sphere-sphere.txt" in options.filepath:
        #titan.assembly[0].mass = 1
        #titan.assembly[1].mass = 2
        #titan.assembly[0].velocity[2] = 5

    if options.structural_dynamics:
        print("Structural dynamics selected: still requiring further validation")
    #    exit("Structural dynamics is currently under development")

    options.current_iter = titan.iter
    options.user_time    = options.dynamics.time_step

    #The mass input in the options file is given for one vehicle/assembly
    if options.vehicle:
        titan.assembly[0].mass = options.vehicle.mass   

    if options.dynamic_plots: plot = dynamic_plots.initialise_figs(titan)
    while titan.iter < options.iters:
        options.high_fidelity_flag = False

        #if options.current_iter%options.output_freq == 0:
        #    output.generate_surface_solution(titan = titan, options = options)

        fragmentation.fragmentation(titan = titan, options = options)

        if not titan.assembly: return      

        if options.time_counter>0:
            options.dynamics.time_step = options.collision.post_fragmentation_timestep
            options.time_counter-=1
        else:
            options.dynamics.time_step = options.user_time

        if 'legacy' in options.dynamics.propagator: dynamics.integrate(titan = titan, options = options)
        else:
            propagation.propagate(titan = titan, options = options)

        #output.generate_surface_solution(titan = titan, options = options, iter_value = titan.iter)
        if hasattr(titan,'end_trigger'): return
        
        if options.thermal.ablation:
            thermal.compute_thermal(titan = titan, options = options)

        if options.structural_dynamics and (titan.iter+1)%options.fenics.FE_freq == 0:
            #TODO
            structural.run_FENICS(titan = titan, options = options)
            output.generate_volume_solution(titan = titan, options = options)
            
        if options.current_iter%options.output_freq == 0:
            output.generate_surface_solution(titan = titan, options = options, iter_value = titan.iter)         
        
        output.iteration(titan = titan, options = options)
        if titan.iter>0: print('Total of {} flow solves'.format(titan.nfeval))

        if options.dynamic_plots:
            for _assembly in titan.assembly: plot = dynamic_plots.update_plot(_assembly, plot, titan.time)

        titan.iter += 1
        titan.post_event_iter +=1
        options.current_iter = titan.iter
        if options.current_iter%options.save_freq == 0 or options.high_fidelity_flag == True:
            options.save_state(titan, options.current_iter)

   # options.save_state(titan)

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
    configFilePath = filename.lstrip()
    configParser.read(configFilePath)

    # This function is for easy modification of config objects, called here to prevent code repeats
    load_and_run_cfg(configParser, postprocess, filter_name)

def load_and_run_cfg(configParser,postprocess,filter_name):
    emissions = False
    #Pre-processing phase: Creates the options and titan class
    options, titan = configuration.read_config_file(configParser, postprocess)

    #Initialization of the simulation
    if (not postprocess) and (not emissions):
        loop(options, titan)
        print("Finished simulation")
        print(titan.nfeval)
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
    parser.add_argument("-flt", "--filter",
                        dest="filtername",
                        type=str,
                        help="filter postprocess (name of the object)",
                        metavar="filtername")
    parser.add_argument("-em", "--emissions",
                        dest="emissions",
                        action="store_true")
    parser.add_argument("-MC", "--montecarlo",
                        dest="n_samples",
                        help = "run a Monte Carlo campaign of N simulations",
                        metavar="n_samples")
    
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

    if postprocess and (postprocess.lower()!="wind" and postprocess.lower()!="ecef"):
        raise Exception("Postprocess can only be WIND or ECEF")

    main(filename = filename, postprocess = postprocess, filter_name = filter_name, emissions = emissions)