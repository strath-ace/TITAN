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

import numpy as np
import yaml
import pandas as pd
from collections.abc import MutableSequence
import pathlib
try: from yaml import CLoader as Loader
except: from yaml import Loader

from Uncertainty import mappings as ma
class UQMapper(MutableSequence):
    ## Once approprUQ Mapper
    def __init__(self, titan, options):
        self.parameters = []
        self.samplers = []
        self.parse_uq_yaml(titan, options.uncertainty.yaml)
        self.callback_flags   = {}
        self.output_dict     = None
        for key in ma.callbacks.keys(): self.callback_flags[key] = False

    def __getitem__(self, idx):
        return self.parameters[idx]

    def __setitem__(self, idx, parameter_data):
        try: 
            self.parameters[idx] = UQMapObject(*parameter_data)
            self.parameters[idx].id = idx
        except Exception as e: 
            raise Exception('Error setting UQ parameter {} with error: {}'.format(idx,e))

    def __delitem__(self, idx):
        del self.parameters[idx]

    def __len__(self):
        return len(self.parameters)

    def insert(self, idx, parameter_data):
        try: 
            self.parameters.insert(idx,UQMapObject(*parameter_data))
            for i_param, param in enumerate(self.parameters): param.id = i_param
        except Exception as e: 
            raise Exception('Error setting UQ parameter {} with error: {}'.format(idx,e))
        
    def parse_uq_yaml(self, titan, filepath):
        with open(str(pathlib.Path(filepath).resolve()),'r') as f: uq_yaml = yaml.load(f,Loader)

        parameters = uq_yaml['parameters']
        distributions_dict = uq_yaml['distributions']
        self.output_dict = uq_yaml['outputs']
        for param, code in parameters.items():
            comp_check = [comp_param in param for comp_param in ma.library_addresses[ma.component_address]]
            if np.any(comp_check):
                parameter_type = ma.library_addresses[ma.component_address][comp_check.index(True)]
                component_name = param.split(parameter_type)[-1]
                component_index = ma.get_component_index_from_name(component_name,titan,code[0])
                address = ma.component_address(code[0],component_index)
                assignment = ma.library_assignments[parameter_type]
            else:
                ma.library_check(param)
                for section, sec_attrs in ma.library_addresses.items():
                    if param in sec_attrs: 
                        address = section(code[0])
                        assignment = ma.library_assignments[param]
                        break
            callback_flags = [param in cb for cb in list(ma.callbacks.values())]
            self.append([param, address, assignment, code, callback_flags])

        for i_distri, distri in distributions_dict.items():
            distri_name = list(distri.keys())[0]
            if distri_name not in ma.available_distris.keys(): 
                raise Exception('Could not find distribution {}'.format(distri_name))
            self.samplers.append(ma.available_distris[distri_name](**distri[distri_name]))


    def map_from_seed(self, seed, titan, options):
        self.state_info = {}
        self.state_info['seed'] = seed
        titan.rng = np.random.RandomState(seed)
        sample_out = []
        assem_ids = [[] for _ in range(len(self.callback_flags))]

        for sam in self.samplers:
            sam.random_state = titan.rng
            sample_out.append(np.atleast_1d(sam.rvs()))

        for param in self.parameters:

            param.assign(sample_out[param.code[1]][param.code[2]], titan, options)
            self.state_info[param.name] = sample_out[param.code[1]][param.code[2]]

            i_cb = 0
            for do_cb, cb in zip(param.callback, list(self.callback_flags.keys())): 
                if do_cb: 
                    self.callback_flags[cb] = True
                    assem_ids[i_cb].append(param.code[0])
                i_cb += 1
        i_cb = 0
        for func, flag in self.callback_flags.items():
            if flag: func(titan, options, np.unique(assem_ids[i_cb]))
            i_cb += 1

        return self.state_info
            
class UQMapObject():
    def __init__(self, param_name, param_object, param_assign_loc, code, callback = None):
        self.name = param_name
        self.assign_obj = param_object
        self.assign_location = param_assign_loc
        self.id = -1
        self.callback = callback
        self.code = code
    
    def resolve_object(self, titan, options):
        if isinstance(self.assign_obj, str):
            address_steps = self.assign_obj.split(',')
            return_obj = titan if 'titan' in address_steps[0].lower() else options
            for step in address_steps[1:]:
                instructions = step.split(';')
                match instructions[0]:
                    case 'attr': return_obj = getattr(return_obj, instructions[1])
                    case 'index': return_obj = return_obj[int(instructions[1])]
                    case 'key': return_obj = return_obj[instructions[1]]
            self.assign_obj =  return_obj
    
    def assign(self, value, titan, options):
        self.resolve_object(titan,options)
        if len(self.assign_location)>1: 
            new_value = getattr(self.assign_obj,self.assign_location[0])
            new_value[self.assign_location[1]] = value
        else: new_value = value
        setattr(self.assign_obj,self.assign_location[0], new_value)

def report_outputs_from_csv(output_folder, output_dict, all_components, write=True):
    data = pd.read_csv(output_folder+'/Data/data.csv')
    assembly_data = pd.read_csv(output_folder+'/Data/data_assembly.csv')
    geo_path = all_components[0].rsplit('/', maxsplit=1)[0]+'/'
    name_list = []
    quantity_map = {}

    if output_dict == 'demise':
        quantities = ['Altitude','Mass','Tmax']
        component_map = get_component_map(assembly_data, all_components)
        for quant in quantities: quantity_map[quant] = component_map

    elif output_dict == 'risk':
        quantities = ['Altitude','Latitude','Longitude','Velocity','Mass', 'Kinetic_energy','Lref']
        component_map = get_survivors_map(data, assembly_data)
        for quant in quantities: quantity_map[quant] = component_map

    else: 
        for quant, component_list in output_dict.items():
            if isinstance(component_list, str):
                if component_list=='all':
                    component_map = get_component_map(assembly_data, all_components)

                elif component_list=='survivors':
                    component_map = get_survivors_map(data, assembly_data)

                else: 
                    component_map = get_component_map(assembly_data, [geo_path+component_list])

            elif isinstance(component_list, list): 
                comp_paths = [geo_path+comp for comp in component_list]
                component_map = get_component_map(assembly_data, comp_paths)

            else: 
                raise Exception('Could not process {} as a component list'.format(component_list))
            
            quantity_map[quant] = component_map
    
    last_data = data.sort_values('Iter',ascending=False)[['Assembly_ID']+list(quantity_map.keys())]
    last_data = last_data.drop_duplicates(subset='Assembly_ID')

    header = []
    values = []
    for quant, component_map in quantity_map.items():
        for component_path, assembly_id in component_map.items():
            comp_name = component_path.rsplit('/',maxsplit=1)[1].rsplit('.',maxsplit=1)[0]
            value = float(last_data[last_data['Assembly_ID']==assembly_id][quant].iloc[0])
            header.append(quant + '_' + comp_name)
            values.append(str(value))
    if write:
        str_header = ','.join(header)+'\n'
        str_values = ','.join(values)+'\n'
        with open(output_folder+'/QoI.csv','w') as f:
            f.write(str_header)
            f.write(str_values)
    return header, values

def get_survivors_map(data, assembly_data, filter_alt=1000):
    last_obj_data = assembly_data.sort_values('Iter',ascending=False).drop_duplicates(subset='Obj_name')
    last_obj_data = last_obj_data[['Assembly_ID','Obj_name']]
    alt_data = data[['Iter', 'Assembly_ID', 'Altitude']]

    survivor_assembly_ids = alt_data[alt_data['Altitude']<filter_alt]['Assembly_ID']
    survivor_assembly_ids = pd.unique(survivor_assembly_ids)

    survivors = last_obj_data[last_obj_data['Assembly_ID'].isin(survivor_assembly_ids)]
    #survivors_list = list(pd.unique(survivors['Obj_name']))

    survivor_map = survivors.set_index('Obj_name')['Assembly_ID'].to_dict()
    #if len(survivor_map.keys())==0: raise Exception('No surviving components')
    return survivor_map

def get_component_map(assembly_data, component_list):
    last_obj_data = assembly_data.sort_values('Iter',ascending=False).drop_duplicates(subset='Obj_name')
    last_obj_data = last_obj_data[last_obj_data['Obj_name'].isin(component_list)]

    return last_obj_data.set_index('Obj_name')['Assembly_ID'].to_dict()

def collate_QoI(seedlist, base_folder, campaign_seed = 0):
    root_df = pd.DataFrame()
    for i_sample, seed in enumerate(seedlist):
        try: 
            csv_path = base_folder+'/Campaign_{}/MC_{}/QoI.csv'.format(campaign_seed, i_sample)
            df = pd.read_csv(str(pathlib.Path(csv_path).resolve()))
            root_df = pd.concat([root_df,df])
        except Exception as e: print('Error retrieving result {}: {}'.format(i_sample,e))
    output_path = base_folder+'/Campaign_{}/QoI.csv'.format(campaign_seed)
    root_df.to_csv(str(pathlib.Path(output_path).resolve()))
