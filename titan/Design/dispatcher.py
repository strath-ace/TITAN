#
# Copyright (c) 2026 TITAN Contributors (cf. AUTHORS.md).
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
"""Dispatcher for full TITAN sim experimental designs"""

from ..__main__ import loop
from ..Configuration.configuration import read_config_file
from ..Design.doe import full_factorial
from ..Dynamics.propagation import collect_state_vectors
from ..Uncertainty.utils import UQMapper as Mapper, report_outputs_from_csv, collate_QoI
from ..Uncertainty.MC_wrapper import write_datafile, update_datafile, log_error
import numpy as np
import configparser
import pathlib
import concurrent.futures
import copy

import yaml
try: from yaml import CLoader as Loader
except: from yaml import Loader


def run(filename:str, design : callable = full_factorial, design_kwargs : dict = {'levels' : 6}):
    """Configures the TITAN case before dispatching the specified design

    :param filename: Config file path
    :type filename: str
    :param design: Specification of experimental design, default is full_factorial
    :type design: callable
    :
    """
    configParser = configparser.RawConfigParser()   
    configFilePath = filename.lstrip()
    configParser.read(configFilePath)
    base_options, base_titan = read_config_file(configParser,'')
    base_folder = base_options.output_folder

    base_titan.uq_mapper = Mapper(base_titan, base_options)
    n_parameters = len(base_titan.uq_mapper.parameters)
    vector_list = design(n_parameters, **design_kwargs)
    base_options.uncertainty.prime_seed = 0
    if base_options.collision.flag: # Can't serialise collision data
        for _assembly in base_titan.assembly: _assembly.collision = None

    if base_options.uncertainty.n_procs>1:
        print('Beginning Parallel Campaign...') 

        with concurrent.futures.ProcessPoolExecutor(base_options.uncertainty.n_procs) as executor:
            output_futures = [executor.submit(wrapper,base_titan,base_options,i_sample,vector) for i_sample, vector in enumerate(vector_list)]
            concurrent.futures.wait(output_futures)
    else:

        for i_sample, vector in enumerate(vector_list):
            print('Beginning Serial Campaign...') 
            new_titan = copy.deepcopy(base_titan)
            new_options = copy.deepcopy(base_options)
            wrapper(new_titan, new_options, i_sample, vector)
            del new_titan, new_options
    
    #collate_QoI(vector_list, base_folder, 0)
    exit()

def wrapper(titan, options, i_sample : int, vector : np.ndarray):
    """Maps the given vector to a TITAN simulation and dispatches it

    :param titan: Base TITAN object
    :type titan: Assembly_list
    :param options: Base Options object
    :type options: Options
    :param i_sample: Sample number
    :type i_sample: int
    :param vector: Domain vector
    :type vector: np.ndarray
    :return: Output names, output values
    :rtype: list, list
    """

    print('Starting run {} vec {}'.format(i_sample, vector), flush=True)
    options.output_folder += '/Campaign_'+str(options.uncertainty.prime_seed)+'/DoE_' + str(i_sample)
    print(options.output_folder)
    options.clean_up_folders()
    options.create_output_folders()
    component_list = []
    collect_state_vectors(titan, options)
    for _assembly in titan.assembly: 
        [component_list.append(comp.name) for comp in _assembly.objects]
    state_info = titan.uq_mapper.map_domain(titan, options, vector)
    write_datafile(options.output_folder, i_sample, state_info)
    if i_sample % round(2*options.uncertainty.n_procs) == 0:
        options.dynamic_plots  = False
        options.write_surface_solutions = True
    else:
        options.dynamic_plots  = False
        options.write_surface_solutions = False
    options.filepath = ''
    try:
        loop(options,titan)

        with open(str(pathlib.Path(options.uncertainty.yaml).resolve()),'r') as f: 
            output_dict = yaml.load(f,Loader)['outputs']
        output_names, output_values = report_outputs_from_csv(options.output_folder, output_dict, component_list)

    except Exception as e: 
        log_error(options.output_folder, str(vector).strip('[]'), e)
        raise e
    
    update_datafile(options.output_folder, str(vector).strip('[]'), output_names, output_values)

    return output_names, output_values