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
        reachable_set_propagate(_assembly, options)

def reachable_set_propagate(assembly, options):
    assem_iter = 0
    time = 0
    num_states = 8
    configs = ['max_transverse','min_transverse','max_drag','min_drag']
    ## If we have attitude as a free parameter we can reduce our state to 3DoF
    if attitude_free_param:
        aero_configs = reachable.get_aero_configs(assembly, options)
        state = np.hstack([assembly.state_vector[:6],assembly.state_vector[13:]])

    else: 
        state = assembly.state_equation
    angles = np.linspace(0., 2*np.pi, num_states)
    num_states*=len(configs)
    angles = np.hstack([angles for _ in configs])
    states = [copy.copy(state) for _ in range(num_states)]
    valid_states = list(range(num_states))
    
    output_file = pathlib.Path(options.output_folder+'/reachable.csv').resolve()
    columns = [['Iter','Time','Assembly_id','Mass','Altitude','Velocity','Flight_path_angle','Heading_angle','Latitude','Longitude',
                        'ECEF_X','ECEF_Y','ECEF_Z','ECEF_U','ECEF_V','ECEF_W','T','Phi','Config_id','Valid']]
    while assem_iter < options.iters:
        ## Need a way to check for fragmentation
        data = np.zeros_like(columns, dtype=np.float64)
        i_config = 0
        for i_state in range(num_states):
            state_data = np.zeros(data.shape[1])
            if i_state in valid_states: # This is a concession to visualisation, letting invalid states still have an output as the previous state enables convenient meshing of the envelope
                state_data[-1] = 1 # State is valid
                states[i_state] = reachable.rk_N(N_rk, states[i_state], options.dynamics.time_step, assembly, options, aero_configs[configs[i_config]], angles[i_state])
                assembly.state_vector[:6]  = states[i_state][:6]
                assembly.state_vector[13:] = states[i_state][6:]
                update_dynamic_attributes(assembly, assembly.state_vector, options, force=True)
                if assembly.mass<=options.dynamics.ignore_mass: 
                    valid_states.pop(valid_states.index(i_state))
            else: state_data[-1] = 0 # State is invalid

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
            state_data[17] = angles[i_state]*180/np.pi
            state_data[18] = i_config
            print('Iter: ', assem_iter, ' ',i_state,'/',num_states)
            data = np.vstack([data,state_data])
            if angles[i_state] == angles[-1]: i_config+=1



        output_data = pd.DataFrame(data = data[1:,:], columns = columns)
        header = not output_file.exists()
        output_data.to_csv(output_file, index=False, header=header, mode='a')
        assem_iter+=1
        time += options.dynamics.time_step
        if len(valid_states)<1: break
        ## Integrate
    exit()

if __name__=='__main__':
    configFilePath = '/home/tommy/reachable_sets/sat.cfg'
    run(configFilePath)
