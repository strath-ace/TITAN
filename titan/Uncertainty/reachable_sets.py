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
"""Functionality for reachable set propagation"""
import numpy as np
import pymap3d
import copy

from ..Aerothermo.aerothermo import compute_aerodynamics, compute_aerothermodynamics
from ..Design.aero import AeroOptimiser
from ..Dynamics.propagation import update_dynamic_attributes, RK_N_actual, RK_k_factors, RK_tableaus
from ..Freestream.mix_properties import compute_freestream, compute_stagnation

base_opt_state = np.array([6458137.33, 0., 0., 0., 0., 7400., 0., 0., 0., 1., 0., 0., 0.])

class AeroAttitude():
    """Data class for holding an aerodynamic attitude of an assembly"""
    def __init__(self, theta_set : np.ndarray, hit_set : np.ndarray, pf_set : np.ndarray, flow_dir_body : np.ndarray):    

        self.theta_set = theta_set
        self.hit_set = hit_set
        self.pf_set = pf_set
        self.flow_dir_body = flow_dir_body

    def get_forces(self, assembly, options):
        assembly.aerothermo.pressure *= 0
        assembly.aerothermo.pressure += assembly.freestream.pressure
        assembly.aerothermo.shear    *= 0

        assembly.aero_index = self.hit_set
        assembly.aerothermo.theta = self.theta_set
        assembly.aerothermo.partial_factor = self.pf_set


        flow_dir = -assembly.velocity/np.linalg.norm(assembly.velocity)
        xwind_hat = np.cross(flow_dir, assembly.position/np.linalg.norm(assembly.position))
        

        compute_aerodynamics(assembly, self.hit_set, flow_dir, options)
        force_facets = -assembly.aerothermo.pressure[:,None]*assembly.mesh.facet_normal+assembly.aerothermo.shear*np.linalg.norm(assembly.mesh.facet_normal, axis=1)[:,None]
        force = np.sum(force_facets, axis = 0)

        self.drag = np.dot(force, self.flow_dir_body)
        self.transverse = np.linalg.norm(force - self.drag*self.flow_dir_body)
        wind_basis = np.array([
            flow_dir,
            xwind_hat,
            np.cross(xwind_hat,flow_dir)
        ])

        return self.drag, self.transverse, wind_basis

    def get_flux(self, assembly, options, dt=1.0):
        assembly.aerothermo.heatflux *= 0
        assembly.aerothermo.he       *= 0
        assembly.aerothermo.hw       *= 0
        assembly.aerothermo.Te       *= 0
        assembly.aerothermo.rhoe     *= 0
        assembly.aerothermo.ue       *= 0
        assembly.aerothermo.ce_i     *= 0

        assembly.aero_index = self.hit_set
        assembly.aerothermo.theta = self.theta_set
        assembly.aerothermo.partial_factor = self.pf_set

        compute_aerothermodynamics(assembly, self.hit_set, -assembly.velocity/np.linalg.norm(assembly.velocity), options)


        ## Quite an ugly code duplication here, sorry :/
        # TODO reorganise thermal code structure to be methods of the objects
        Tref = 273

        #if assembly.ablation_mode != '0d': continue
        d_temperatures = []
        d_masses = []
        for obj in assembly.objects:
            facet_area = np.linalg.norm(obj.mesh.facet_normal, ord = 2, axis = 1)
            heatflux = assembly.aerothermo.heatflux[obj.facet_index]
            Qin = np.sum(heatflux*facet_area)
            
            cp  = obj.material.specificHeatCapacity(obj.temperature)
            emissivity = obj.material.emissivity(obj.temperature)

            Atot = np.sum(facet_area)

            # Estimating the radiation heat-flux
            Qrad = 5.670373e-8*emissivity*(obj.temperature**4 - Tref**4)*Atot

            # Computing temperature change
            if obj.mass>0:
                dT = (Qin-Qrad)*dt/(obj.mass*cp)
            else: dT = 0.0


            if obj.temperature+dT > obj.material.meltingTemperature:
                dT_melt = obj.material.meltingTemperature - obj.temperature
                melt_Q = (obj.mass*cp)*(dT-dT_melt)
                dm = -melt_Q/(obj.material.meltingHeat)
                dT = dT_melt
            else:
                dm = 0

            obj.mdot = dm
            obj.Tdot = dT

            #obj.photons = compute_radiance(obj.temperature, Atot, emissivity)
            d_temperatures.append(dT)
            d_masses.append(dm)

        return d_temperatures, d_masses

def get_aero_configs(assembly, options) -> dict[AeroAttitude]:
    aero_configs = {}
    assembly_state = copy.copy(assembly.state_vector)
    assembly.state_vector[:13] = base_opt_state
    opt = AeroOptimiser(assembly, {}, options, objective='transverse', objective_weights=[1], visualise=False)

    opt.solve()
    theta, pf, hits = opt.collect_theta_set(options)
    aero_configs['max_transverse'] = AeroAttitude(theta_set=theta, hit_set=hits, pf_set=pf, flow_dir_body=opt.flow_dir_body)
    options.n_debug = 0
    opt.objective_weights=[-1]
    opt.setup_obj_func({}, options)
    opt.solve()
    theta, pf, hits = opt.collect_theta_set(options)
    aero_configs['min_transverse'] = AeroAttitude(theta_set=theta, hit_set=hits, pf_set=pf, flow_dir_body=opt.flow_dir_body)

    opt.objective='integrated'
    opt.objective_weights=[0., -1., 0.]
    opt.setup_obj_func({}, options)
    opt.solve()
    theta, pf, hits = opt.collect_theta_set(options)
    aero_configs['max_drag'] = AeroAttitude(theta_set=theta, hit_set=hits, pf_set=pf, flow_dir_body=opt.flow_dir_body)
    
    opt.objective_weights=[1., 1., 1.]
    opt.setup_obj_func({}, options)
    opt.solve()
    theta, pf, hits = opt.collect_theta_set(options)
    aero_configs['min_drag'] = AeroAttitude(theta_set=theta, hit_set=hits, pf_set=pf, flow_dir_body=opt.flow_dir_body)

    assembly.state_vector = assembly_state
    return aero_configs

def rk_N(N,state,dt,assembly,options,aero_config,phi):
    """Documentation for the function.
:param N: Integer value for n.
:type N: int
:param state_vectors: Value for state vectors.
:type state_vectors: Any
:param state_vectors_prior: Value for state vectors prior.
:type state_vectors_prior: Any
:param derivatives_prior: Value for derivatives prior.
:type derivatives_prior: Any
:param dt: Numeric value for dt.
:type dt: float
:param titan: TITAN simulation object.
:type titan: object
:param options: Options or configuration object.
:type options: object
:return: Return value.
:rtype: Any"""
    k_n = []
    for i_k in range(RK_N_actual[str(N)]):
        k_state_vectors = state
        for i_coeff in range(i_k): 
            delta_tableau = k_n[i_coeff]*RK_tableaus[str(N)][i_k][i_coeff]
            k_state_vectors += delta_tableau
        if i_k==0:
            d_dt = state_equation(assembly, options, aero_config, phi,k_state_vectors)
        else: d_dt = state_equation(assembly, options, aero_config, phi ,k_state_vectors)
        k_n.append(d_dt)
    new_state = state
    for i_k in range(N): 
        delta_factors = k_n[i_k]*RK_k_factors[str(N)][i_k] * dt
        new_state += delta_factors
        
    return new_state

def state_equation(assembly,options,aero_config, phi, state):
    assembly.state_vector[:6] = state[:6]
    assembly.state_vector[13:] = state[6:]
    update_dynamic_attributes(assembly,assembly.state_vector, options, force=True)

    compute_freestream(options.freestream.model, assembly.trajectory.altitude, assembly.trajectory.velocity, assembly.Lref, assembly.freestream, assembly, options)
    compute_stagnation(assembly.freestream, options.freestream)


    drag, transverse, basis = aero_config.get_forces(assembly, options)

    F_aero = drag*basis[0] + transverse*basis[1]*np.cos(phi) + transverse*basis[2]*np.sin(phi)
    a_aero = F_aero/assembly.mass if assembly.mass>0 else np.zeros(3)
    wE = options.planet.omega()    
    r = np.linalg.norm(assembly.position)
    gr,gt = options.planet.gravitationalAcceleration(r, phi = np.pi/2 - assembly.trajectory.latitude)
    a_grav = pymap3d.enu2uvw(0,0, gr,assembly.trajectory.latitude, assembly.trajectory.longitude,deg = False)
    a_centrif = -np.cross(np.array([0,0,wE]), np.cross(np.array([0,0,wE]), assembly.position))
    a_coriolis = -2*np.cross(np.array([0,0,wE]), assembly.velocity)

    dx = state[3:6]
    
    dv = a_aero + a_centrif + a_coriolis + a_grav
    
    dT, dm = aero_config.get_flux(assembly, options)
    dObj = []
    for T, m in zip(dT, dm):
        dObj.append(T)
        dObj.append(m)
    return np.hstack([dx,dv,dObj])