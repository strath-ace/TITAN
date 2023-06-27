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
from Output import output
from Dynamics import dynamics
from Fragmentation import fragmentation
from Postprocess import postprocess as pp
from Thermal import thermal
from Structural import structural
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
    
    #The mass input in the options file is given for one vehicle/assembly
    if options.vehicle:
        titan.assembly[0].mass = options.vehicle.mass

    while titan.iter < options.iters:
        
        fragmentation.fragmentation(titan = titan, options = options)
        if not titan.assembly: return

        dynamics.integrate(titan = titan, options = options)
        
        if options.ablation:
            if options.ablation_mode == "tetra":
                thermal.compute_thermal_tetra(titan = titan, options = options)
            elif options.ablation_mode == "0d":
                thermal.compute_thermal_0D(titan = titan, options = options)
            else:
                raise ValueError("Ablation Mode can only be 0D or Tetra")

        if options.structural_dynamics:
            #TODO
            structural.run_FENICS(titan = titan, options = options)
            output.generate_volume_solution(titan = titan, options = options)

        output.generate_surface_solution(titan = titan, options = options)
        
        output.iteration(titan = titan, options = options)

        titan.iter += 1
        options.current_iter = titan.iter
        if options.current_iter%options.save_freq == 0:
            options.save_state(titan)

    options.save_state(titan)

def main(filename = "", postprocess = ""):
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
    configFilePath = filename
    configParser.read(configFilePath)

    #Pre-processing phase: Creates the options and titan class
    options, titan = configuration.read_config_file(configParser, postprocess)
    options.filepath = filename

    #Initialization of the simulation
    if not postprocess:
        loop(options, titan)
        print("Finished simulation")
        return options, titan

    #Postprocess of the simulated solution to pass from Body-frame
    #to ECEF-Frame or Wind-Frame
    if postprocess:
        Path(options.output_folder+'/Postprocess/').mkdir(parents=True, exist_ok=True)
        pp.postprocess(options, postprocess)
    
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
    
    args=parser.parse_args()

    if not args.configfilename:
        raise Exception('The user needs to provide a file!.\n')

    filename = args.configfilename
    postprocess = args.postprocess
    if postprocess and (postprocess.lower()!="wind" and postprocess.lower()!="ecef"):
        raise Exception("Postprocess can only be WIND or ECEF")

    main(filename = filename, postprocess = postprocess)