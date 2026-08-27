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
from scipy.optimize import dual_annealing, basinhopping, shgo, brute, direct, differential_evolution, minimize
from scipy.spatial.transform import Rotation, RigidTransform
import configparser
import pathlib

from ..Aerothermo.aerothermo import ray_trace, compute_aerodynamics, compute_aerothermodynamics, write_rays_to_vtk
from ..Configuration.configuration import read_config_file
from ..Dynamics.propagation import collect_state_vectors, update_dynamic_attributes
from ..Freestream.mix_properties import compute_freestream, compute_stagnation
from ..Output.output import create_surface_solution, update_surface_solution, write_surface_solution


def attitude_function(assembly, options, debug_visuals : bool, weights : np.ndarray, conditions : dict, integrated : bool, attitude_vector : np.ndarray, info : dict) -> float:
    """Attitude objective function, inputs an attitude vector in MRP and returns a scalar output

    :param assembly: Target assembly
    :type assembly: geometry.Assembly
    :param options: TITAN options
    :type options: configuration.Options
    :param debug_visuals: enable to generate visualisations of the optimisation process
    :type debug_visuals: bool
    :param weights: Weights applied to facet pressure and shear if integrated is false, otherwise applied to either lift, drag and crosswind or L/D 
    :type weights: np.ndarray
    :param conditions: Aerodynamic conditions passed to the obj_func, currently inert 
    :type conditions: dict
    :param integrated: Whether to integrate the pressure and shear fields over the body
    :type integrated: bool
    :param attitude_vector: Attitude as defined by Modified Rodrigues Parameters
    :type attitude_vector: np.ndarray
    :param info: Dictionary containing an 'n_feval' entry, used to track function evalutations
    :type info: dict
    :return: Objective function output
    :rtype: float
    """

    assembly.aerothermo.pressure.fill(assembly.freestream.pressure)
    assembly.aerothermo.shear.fill(0.0)

    if debug_visuals: 
        visual_folder = pathlib.Path(options.output_folder+'/Opt_{}/'.format(assembly.id)).resolve()
        if not visual_folder.exists(): visual_folder.mkdir()
    # #R = Rotation.from_euler('ZYX', attitude_vector)
    # if np.linalg.norm(attitude_vector)==0.: attitude_vector[-1]+=1
    # else: attitude_vector/=np.linalg.norm(attitude_vector)

    R_ECEF_from_B = Rotation.from_mrp(attitude_vector)
    Rig_ECEF_from_B = RigidTransform.from_rotation(R_ECEF_from_B)* RigidTransform.from_translation(-assembly.COG)
    assembly.state_vector[6:10] = R_ECEF_from_B.as_quat()
    update_dynamic_attributes(assembly, assembly.state_vector, options, force=True)
    flow_dir = -assembly.velocity/np.linalg.norm(assembly.velocity)


    output_rays = 'body' if debug_visuals and info['n_feval'] % 10 == 0 else None
    ray_trace([assembly],options.aerothermo.subdivision_triangle, options, output_rays=output_rays)
    compute_aerodynamics(assembly, assembly.aero_index, flow_dir, options)
    if debug_visuals and info['n_feval'] % 10 == 0:
        compute_aerothermodynamics(assembly, assembly.aero_index, flow_dir, options)
        solution = update_surface_solution(assembly, options, info['sol'])
        write_surface_solution(options,solution, 'Solutions', int(info['n_feval']/10), folder='Opt_{}'.format(assembly.id))
    if isinstance(info, list): info = info[0]
    


    if not integrated: 
        roll_pitch_yaw = R_ECEF_from_B.as_euler('ZYX',degrees=True)
        obj_func = np.sum(weights[0] * assembly.aerothermo.pressure) + np.sum(weights[1] * assembly.aerothermo.shear)
        if info['n_feval'] % 25 == 0:
                print('n={} | Roll {}° | Pitch {}° | Yaw {}° | Obj_func = {}'.format(info['n_feval'], 
                                                                                     round(roll_pitch_yaw[0],2), 
                                                                                     round(roll_pitch_yaw[1],2), 
                                                                                     round(roll_pitch_yaw[2],2), 
                                                                                     round(obj_func,6)))

        return obj_func

    # Force in the body frame
    force_facets = -assembly.aerothermo.pressure[:,None]*assembly.mesh.facet_normal+assembly.aerothermo.shear*np.linalg.norm(assembly.mesh.facet_normal, axis=1)[:,None]
    force = np.sum(force_facets, axis = 0)


    # Force in ECEF frame -> need to convert to wind frame
    F_ECEF = R_ECEF_from_B.apply(force)
    drag = np.dot(F_ECEF, flow_dir)
    #drag *= -1
    xwind_hat  = np.cross(flow_dir, assembly.position/np.linalg.norm(assembly.position))
    xwind_hat /= np.linalg.norm(xwind_hat)
    xwind = np.dot(F_ECEF, xwind_hat)
    lift = np.dot(F_ECEF, np.cross(xwind_hat,flow_dir))
    if debug_visuals and info['n_feval'] % 10 == 0:
        body_basis = np.array([R_ECEF_from_B.inv().apply(flow_dir),
                               R_ECEF_from_B.inv().apply(xwind_hat),
                               R_ECEF_from_B.inv().apply(np.cross(xwind_hat,flow_dir))])
        write_rays_to_vtk(str(visual_folder)+'/basis_'+str(int(info['n_feval']/10))+'.vtk',np.zeros([3,3]),body_basis)
        write_rays_to_vtk(str(visual_folder)+'/forces_'+str(int(info['n_feval']/10))+'.vtk',np.zeros([3,3]),1e-3*body_basis*np.array([[drag],[xwind],[lift]]))
        write_rays_to_vtk(str(visual_folder)+'/facets_'+str(int(info['n_feval']/10))+'.vtk',assembly.mesh.facet_COG,assembly.mesh.facet_COG-1e-2*force_facets)
    assert drag>-0.5
    
    
    # p_dyn = 0.5 * assembly.freesteam.density * conditions['velocity_magnitude']**2
    # Cd = drag / (p_dyn * assembly.Aref)
    # Cl = lift / (p_dyn * assembly.Aref)
    # Cs = xwind / (p_dyn * assembly.Aref)
    #obj_func = float((lift ** weights[0]) * (drag ** weights[1]) * (abs(xwind) ** weights[2]))
    if len(weights)>1:
        obj_func = float((abs(lift) * weights[0]) + (drag * weights[1]) + (abs(xwind) * weights[2]))
    else: 
        transverse_vector = lift*np.cross(xwind_hat,flow_dir) + xwind * xwind_hat
        obj_func = abs((drag/np.linalg.norm(transverse_vector))**weights[0])
    if info['n_feval'] % 25 == 0:
        roll_pitch_yaw = R_ECEF_from_B.as_euler('ZYX',degrees=True)
        print('n={} | Roll {}° | Pitch {}° | Yaw {}° | Lift {}N | Drag {}N | xwind {}N | Obj_func = {}'.format(info['n_feval'], 
                                                                                                              round(roll_pitch_yaw[0],2), 
                                                                                                              round(roll_pitch_yaw[1],2), 
                                                                                                              round(roll_pitch_yaw[2],2),
                                                                                                              round(lift,4), 
                                                                                                              round(drag,4), 
                                                                                                              round(xwind,4), 
                                                                                                              round(obj_func,6)))

    info['n_feval'] +=1
    return obj_func
valid_solvers = ['dual_annealing','basinhopping','shgo','brute', 'direct', 'differential_evolution']
class AeroOptimiser():
    '''Class for managing the the construction and solving of an optimisation problem in terms of aerodynamics'''
    def __init__(self, assembly, conditions : dict, options, problem_kind : str = 'attitude', objective : str = 'integrated', objective_weights : np.ndarray = [1,1,1], solver : str = 'direct', budget : float = 5e2, visualise : bool = False):
        """Create an optimiser to solve an aerodynamic problem

        :param assembly: Target assembly
        :type assembly: geometry.Assembly
        :param conditions: Specified freestream conditions
        :type conditions: dict
        :param options: TITAN options
        :type options: configuration.Options
        :param problem_kind: Define parameter space to optimise over, currently only attitude is implemented, defaults to 'attitude'
        :type problem_kind: str, optional
        :param objective: Define output space to optimise, selecting anything other than integrated or transverse means specifying weights for individual facets. 
        Integrated means specifying weights for lift drag and crosswind respectively, transverse means specifying a ratio direction (+ve maximise L/D, -ve minimise L/D). Defaults to 'integrated'
        :type objective: str, optional
        :param objective_weights: Weights to use for the objective function. if using integrated these correspond to Lift, Drag and Crosswind respectively, 
        otherwise this should be an N_facets x 2 array specifying the weights for pressure and shear for each facet respectively. Defaults to [1,1,1]
        :type objective_weights: np.ndarray, optional
        :param solver: Solver selection, any scipy global optimiser can be used here but DiRECT has been found to give best results. Defaults to 'direct'
        :type solver: str, optional
        :param budget: Solver-dependent budget usually tuned to be approximately equal to be number of function evals, defaults to 5e2
        :type budget: float, optional
        :param visualise: Enable to output optimisation visualisation, defaults to False
        :type visualise: bool, optional
        """
        self.kind = problem_kind
        self.assembly = assembly
        self.objective = objective
        self.objective_weights = objective_weights
        self.budget = budget
        #: The solver to use for optimisation, DiRECT is highly recommended
        self.solver = solver
        self.visualise = visualise
        if self.objective=='transverse': assert len(objective_weights)==1
        if self.objective=='integrated': assert len(objective_weights)==3

        self.setup_obj_func(conditions, options)
        self.result = None
        self.n_feval = 0
        # Useful for checking output
        self.solution = create_surface_solution(self.assembly, options)
    
    def setup_obj_func(self, conditions : dict, options):
        """Initialise the objective function based upon problem description

        :param conditions: Dict specifying problem conditions, inert at present
        :type conditions: dict
        :param options: TITAN options
        :type options: configuration.Options
        """
        integrated = True if self.objective == 'integrated' or self.objective=='transverse' else False
        match self.kind:
            case 'attitude':
                compute_freestream(options.freestream.model, self.assembly.trajectory.altitude, self.assembly.trajectory.velocity, self.assembly.Lref, self.assembly.freestream, self.assembly, options)
                compute_stagnation(self.assembly.freestream, options.freestream)
                conditions['velocity_magnitude'] = np.linalg.norm(self.assembly.trajectory.velocity)
                self.obj_func = partial(attitude_function, 
                                        self.assembly, 
                                        options,
                                        self.visualise, 
                                        self.objective_weights, 
                                        conditions, 
                                        integrated)
            case 'freestream': raise NotImplementedError
    
    def solve(self):
        """Run the optimiser
        """
        if self.kind=='attitude':
            
            match self.solver:
                case 'dual_annealing':
                    self.result = dual_annealing(self.obj_func, [(-1,1),(-1,1),(-1,1)], maxfun=int(self.budget),no_local_search=False, args=[{'n_feval':self.n_feval, 'sol' : self.solution}], minimizer_kwargs={'options' : {'maxiter' : 100}})
                case 'basinhopping':
                    n_hops = int(self.budget/400)
                    print(n_hops)
                    self.result = basinhopping(self.obj_func, [0,0,1], niter=n_hops, T = np.pi/2, minimizer_kwargs={'options' : {'maxiter' : 100}, 'args' : [{'n_feval':self.n_feval, 'sol' : self.solution}]})
                case 'shgo':
                    n_points = int(self.budget/100)
                    self.result = shgo(self.obj_func, [(-1,1),(-1,1),(-1,1)],iters=6, n=n_points, args=[{'n_feval':self.n_feval, 'sol' : self.solution}],minimizer_kwargs={'options' : {'maxiter' : 100}})
                case 'brute':
                    self.result = brute(self.obj_func, [(-1,1),(-1,1),(-1,1)], Ns = 5, finish=minimize, args=[{'n_feval':self.n_feval, 'sol' : self.solution}])
                case 'direct':
                    self.result = direct(self.obj_func, [(-1,1),(-1,1),(-1,1)], args=[{'n_feval':self.n_feval, 'sol' : self.solution}],maxfun=int(self.budget))
                case 'differential_evolution':
                    n_iters = int(self.budget/100)
                    self.result = differential_evolution(self.obj_func,  [(-1,1),(-1,1),(-1,1)], args=[{'n_feval':self.n_feval, 'sol' : self.solution}], popsize = 20, maxiter=n_iters)

                case _: raise Exception('Did not recognise optimiser {}, available options are...{}'.format(self.solver, valid_solvers))
            if hasattr(self.result, 'message'): print(self.result.message)

    def collect_theta_set(self, options) -> tuple[np.ndarray]:
        """Collect set of optimal angles theta for the body, calls the optimiser if no solution yet exists

        :param options: TITAN options
        :type options: configuration.Options
        :return: Set of angles, set of partial factors and set of hit indices for the optimal attitude
        :rtype: tuple[np.ndarray]
        """
        if not self.kind=='attitude': print('Note: Attitude is not a free parameter in this optimisation')
        if self.result is None: self.solve()
        if self.solver == 'brute': attitude = self.result
        else: attitude = self.result['x']

        self.assembly.aerothermo.pressure.fill(self.assembly.freestream.pressure)
        self.assembly.aerothermo.shear.fill(0.0)
        
        R_ECEF_from_B = Rotation.from_mrp(attitude)
        self.assembly.state_vector[6:10] = R_ECEF_from_B.as_quat()
        update_dynamic_attributes(self.assembly, self.assembly.state_vector, options, force=True)

        ray_trace([self.assembly],options.aerothermo.subdivision_triangle, options)#, output_rays='leading')

        if self.objective == 'transverse':
            flow_dir = -self.assembly.velocity/np.linalg.norm(self.assembly.velocity)
            compute_aerodynamics(self.assembly, self.assembly.aero_index, flow_dir, options)
            # Force in the body frame
            force_facets = -self.assembly.aerothermo.pressure[:,None]*self.assembly.mesh.facet_normal+self.assembly.aerothermo.shear*np.linalg.norm(self.assembly.mesh.facet_normal, axis=1)[:,None]
            force = np.sum(force_facets, axis = 0)
            
            # Force in ECEF frame -> need to convert to wind frame
            F_ECEF = R_ECEF_from_B.apply(force)

            # drag = np.dot(F_ECEF, flow_dir)
            # xwind_hat = np.cross(flow_dir, self.assembly.position/np.linalg.norm(self.assembly.position))
            # xwind = np.dot(F_ECEF, xwind_hat)
            # lift_hat = np.cross(xwind_hat,flow_dir)
            # lift = np.dot(F_ECEF, lift_hat)
            # This is nonsense, what we need is as follows: Flow direction in body frame (from R_ECEF_from_B quat), then dot F_B with 
            # flow_dir to get drag magnitude, subtract drag vector to get transverse magnitude
            # Then we can define the transverse locus (do we want a locus?)

            flow_dir_body = R_ECEF_from_B.inv().apply(flow_dir)
            self.flow_dir_body = flow_dir_body


        self.theta_set = self.assembly.aerothermo.theta
        self.pf_set = self.assembly.aerothermo.partial_factor
        self.index_set = self.assembly.aero_index
       

        return self.theta_set, self.pf_set, self.index_set
    
# if __name__=='__main__':
#     configParser = configparser.RawConfigParser()   
#     configFilePath = '/home/tommy/reachable_sets/sat.cfg'
#     configParser.read(configFilePath)

#     #Pre-processing phase: Creates the options and titan class
#     options, titan = read_config_file(configParser, '','')
#     collect_state_vectors(titan, options)
#     optimiser = AeroOptimiser(titan.assembly[0], {'velocity_magnitude' : 7800}, options, objective='integrated', objective_weights=[1,1,1])#.1])
#     optimiser.budget = 450
#     optimiser.solver = 'direct'
#     optimiser.solve()
#     optimiser.collect_theta_set(options)
#     print('w')