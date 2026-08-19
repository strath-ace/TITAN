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
"""Optimisation of Aerodynamic Parameters"""

import numpy as np
from functools import partial
from scipy.optimize import dual_annealing

from ..Aerothermo.aerothermo import ray_trace, compute_aerodynamics
from ..Forces.forces import
def attitude_function(assembly, options, weights : list, conditions : dict, compute_coefficients : bool attitude_vector : np.ndarray) -> float:
    flow_dir = np.array([np.cos(attitude_vector[0])*np.cos(attitude_vector[1]), 
                np.cos(attitude_vector[0])*np.sin(attitude_vector[1]), 
                np.sin(attitude_vector[0])])
    flow_dir /= np.linalg.norm(flow_dir)
    assembly.velocity = conditions['velocity_magnitude'] * flow_dir/
    ray_trace([assembly],options.aerothermo.subdivision_triangle, options)
    compute_aerodynamics(assembly, assembly.aero_index, flow_dir, options)

    if not compute_coefficients: return np.sum(weights[0] * assembly.aerothermo.pressure) + np.sum(weights[1] * assembly.aerothermo.shear)
    
    force_facets = -assembly.aerothermo.pressure[:,None]*assembly.mesh.facet_normal+assembly.aerothermo.shear*np.linalg.norm(assembly.mesh.facet_normal, axis=1)[:,None]
    force = np.sum(force_facets, axis = 0)

    drag = np.dot(force, flow_dir)
    slip = np.dot(force, np.cross(assembly.state_vector[:3]/np.linalg.norm(assembly.state_vector[:3]), flow_dir))
    lift = np.dot(force, np.cross(slip/np.linalg.norm(slip),flow_dir))

    # p_dyn = 0.5 * assembly.freesteam.density * conditions['velocity_magnitude']**2
    # Cd = drag / (p_dyn * assembly.Aref)
    # Cl = lift / (p_dyn * assembly.Aref)
    # Cs = slip / (p_dyn * assembly.Aref)
    
    return lift * weights[0] + drag * weights[1] + slip * weights[2]

class AeroOptimiser():
    '''Class for managing the the construction and solving of an optimisation problem in terms of aerodynamics'''
    def __init__(assembly, conditions, options, problem_kind = 'attitude', objective = 'coefficients', objective_weights = [1,1,1]):
        self.kind = problem_kind
        self.assembly = assembly
    
    def setup_obj_func(self, conditions, options):
        compute_coefficients = True if self.objective == 'coefficients' else False
        match self.kind:
            case 'attitude':
                self.obj_func = partial(attitude_function, 
                                        self.assembly, 
                                        options, 
                                        self.objective_weights, 
                                        conditions, 
                                        compute_coefficients)
            case 'freestream': raise NotImpementedError
    
    def solve(self):
        