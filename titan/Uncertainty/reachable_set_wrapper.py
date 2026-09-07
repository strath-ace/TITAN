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

"""Alternate propagator to estimate feasible re-entry envelope"""

import configparser
import numpy as np
import pandas as pd
import pathlib
import copy
import concurrent.futures

from ..Configuration.configuration import read_config_file
from ..Dynamics.propagation import collect_state_vectors, update_dynamic_attributes

from ..Uncertainty import reachable_sets as reachable

attitude_free_param = True
N_rk = 3

def run(filename : str):
    configParser = configparser.RawConfigParser()   
    configFilePath = filename.lstrip()
    configParser.read(configFilePath)
    options, titan = read_config_file(configParser,'')
    if not options.dynamics.augmented_state: raise Exception('Augmented state required for reachable state propagation!')

    collect_state_vectors(titan, options)
    for _assembly in titan.assembly:
        update_dynamic_attributes(_assembly,_assembly.state_vector, options, force=True)
        reachable_set_propagate(_assembly, options)

def reachable_set_propagate(assembly, options):
    assem_iter = 0
    time = 0
    num_angles = 16


    config_names = ['max_transverse','max_drag','min_drag', 'max_flux', 'min_flux']
    num_configs = len(config_names)
    ## If we have attitude as a free parameter we can reduce our state to 3DoF
    if attitude_free_param:
        aero_configs = reachable.get_aero_configs(assembly, options)
        state = np.hstack([assembly.state_vector[:6],assembly.state_vector[13:]])

    else: 
        state = assembly.state_equation

    angles = np.linspace(0., 2*np.pi, num_angles, endpoint = False)
    num_states = num_angles * num_configs
    angles = np.hstack([angles for _ in range(num_configs)])

    configs = np.hstack([[cfg for _ in range(num_angles)] for cfg in config_names])

    assert len(angles)==len(configs)==num_states
    configs = [aero_configs[cfg] for cfg in configs]
    #states = [np.array(state, copy=True) for _ in range(num_states)]
    #valid_states = list(range(num_states))
    
    output_dir = pathlib.Path(options.output_folder+'/reachable_{}/'.format(assembly.id)).resolve()
    if not output_dir.exists(): output_dir.mkdir()
    columns = ['Iter','Time','Assembly_id','Mass','Altitude','Velocity','Flight_path_angle','Heading_angle','Latitude','Longitude',
                        'ECEF_X','ECEF_Y','ECEF_Z','ECEF_U','ECEF_V','ECEF_W','T','Phi','Config_id','State_id']
    data = np.zeros([1,len(columns)], dtype = np.float64)

    num_workers = num_states
    if num_workers>1:
        with concurrent.futures.ProcessPoolExecutor(num_workers) as executor:
            fut = [executor.submit(
                propagate_configuration,assembly, options, 
                {'state' : state, 
                'aero_config' : configs[i_state], 
                'phi' : angles[i_state], 
                'id' : configs[i_state].id,
                'index' : i_state}
                ) for i_state in range(num_states)]

            concurrent.futures.wait(fut)

        for future in fut:
            if not future._exception:
                data = np.vstack([data, future.result()])
    else:
        data = np.vstack([data, [propagate_configuration(assembly, options,
                {'state' : state, 
                'aero_config' : configs[i_state], 
                'phi' : angles[i_state], 
                'id' : configs[i_state].id,
                'index' : i_state}
                ) for i_state in range(num_states)]])


    output_data = pd.DataFrame(data = data[1:,:], columns = columns)
    output_data = output_data.sort_values(by=['Iter','State_id'])
    output_data.to_csv(str(output_dir)+'/reachable_points.csv', index = False, header=True, mode='w')
    reachable.get_envelope_from_csv(str(output_dir)+'/reachable_points.csv',str(output_dir))
    
    exit()

def propagate_configuration(assembly, options, configuration : dict) -> np.ndarray:
    assem_iter = 0
    time = 0.
    data = np.zeros([1,20], dtype=np.float64)
    state = configuration['state']
    aero_config = configuration['aero_config']
    angle = configuration['phi']
    config_id = configuration['id']
    index = configuration['index']

    while assem_iter<options.iters:
        assembly.state_vector[:6]  = state[:6]
        assembly.state_vector[13:] = state[6:]
        update_dynamic_attributes(assembly, assembly.state_vector, options, force=True)

        state_data = np.zeros(data.shape[1])
        state_data[0] = assem_iter
        state_data[1] = time
        state_data[2] = assembly.id
        state_data[3] = assembly.mass
        state_data[4] = assembly.trajectory.altitude
        state_data[5] = assembly.trajectory.velocity
        state_data[6] = assembly.trajectory.gamma*180/np.pi
        state_data[7] = assembly.trajectory.chi*180/np.pi
        state_data[8] = assembly.trajectory.latitude*180/np.pi
        state_data[9] = assembly.trajectory.longitude*180/np.pi
        state_data[10] = assembly.position[0]
        state_data[11] = assembly.position[1]
        state_data[12] = assembly.position[2]
        state_data[13] = assembly.velocity[0]
        state_data[14] = assembly.velocity[1]
        state_data[15] = assembly.velocity[2]
        state_data[16] = np.mean(assembly.aerothermo.temperature)
        state_data[17] = angle*180/np.pi
        state_data[18] = config_id
        state_data[19] = index

        state = reachable.rk_N(N_rk, state, options.dynamics.time_step, assembly, options, aero_config, angle)

        if assembly.mass<=options.dynamics.ignore_mass or assembly.trajectory.altitude<0: break

        if assem_iter % 10 == 0: print('State {}: Config {} Phi {} @ Iteration {}'.format(index, config_id, angle*180/np.pi, assem_iter))
        data = np.vstack([data,state_data])
        assem_iter+=1
        time += options.dynamics.time_step
    return data[1:,:]



if __name__=='__main__':
    configFilePath = '/home/tommy/reachable_sets/sat.cfg'
    run(configFilePath)
