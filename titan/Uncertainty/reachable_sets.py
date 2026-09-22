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
import pandas as pd
import open3d as o3d
from scipy.interpolate import PchipInterpolator

from ..Aerothermo.aerothermo import aerodynamics_module_continuum, aerodynamics_module_freemolecular, bridging, aerothermodynamics_module_continuum, aerothermodynamics_module_freemolecular, bridging_altitudes, create_thermal_bridge
from ..Design.aero import AeroOptimiser, AeroSurrogate
from ..Dynamics.propagation import update_dynamic_attributes, RK_N_actual, RK_k_factors, RK_tableaus
from ..Freestream.mix_properties import compute_freestream, compute_stagnation
from ..Uncertainty.utils import UQMapObject, UQMapper

base_opt_state_c = np.array([6448137.33, 0., 0., 0., 0., 7400., 0., 0., 0., 1., 0., 0., 0.])
base_opt_state_fmf = np.array([6478137.33, 0., 0., 0., 0., 7700., 0., 0., 0., 1., 0., 0., 0.])

class AeroAttitude():
    """Data class for holding a pair of aerodynamic attitudes (one for each regime) of an assembly"""
    def __init__(self, theta_set_fmf : np.ndarray, hit_set_fmf : np.ndarray, pf_set_fmf : np.ndarray, flow_dir_body_fmf : np.ndarray, tangents_fmf : np.ndarray, theta_set_c : np.ndarray, hit_set_c : np.ndarray, pf_set_c : np.ndarray, flow_dir_body_c : np.ndarray):

        self.theta_set_fmf = theta_set_fmf
        self.hit_set_fmf = hit_set_fmf
        self.pf_set_fmf = np.atleast_2d(pf_set_fmf)
        self.flow_dir_body = flow_dir_body_fmf
        self.tangents_fmf = tangents_fmf


        self.theta_set_c = theta_set_c
        self.hit_set_c = hit_set_c
        self.pf_set_c = pf_set_c
        self.flow_dir_body_c = flow_dir_body_c
        self.id = 0

    def get_forces(self, assembly, options):
        if not hasattr(assembly.aerothermo, 'tangent_vector'): assembly.aerothermo.tangent_vector = self.tangents_fmf
        assembly.aerothermo.pressure *= 0
        assembly.aerothermo.pressure += assembly.freestream.pressure
        assembly.aerothermo.shear    *= 0




        flow_dir = -assembly.velocity/np.linalg.norm(assembly.velocity)
        xwind_hat = np.cross(flow_dir, assembly.position/np.linalg.norm(assembly.position))
        

        Kn_cont_pressure = options.aerothermo.knc_pressure
        Kn_free = options.aerothermo.knf

        #Pressure calculation only if Drag model is False
        if (not options.vehicle) or (options.vehicle and not options.vehicle.Cd):
            if  (assembly.freestream.knudsen <= Kn_cont_pressure):
                assembly.aero_index = self.hit_set_c
                assembly.aerothermo.theta = self.theta_set_c
                assembly.aerothermo.partial_factor = self.pf_set_c
                assembly.aerothermo.pressure[self.hit_set_c] += aerodynamics_module_continuum(assembly, self.hit_set_c, flow_dir) \
                * self.pf_set_c[self.hit_set_c] * options.aerothermo.CP_mult

            elif (assembly.freestream.knudsen >= Kn_free):
                assembly.aero_index = self.hit_set_fmf
                assembly.aerothermo.theta = self.theta_set_fmf
                assembly.aerothermo.partial_factor = self.pf_set_fmf
                pressure, shear = aerodynamics_module_freemolecular(assembly, self.hit_set_fmf, flow_dir)
                assembly.aerothermo.pressure[self.hit_set_fmf] += pressure * self.pf_set_fmf[self.hit_set_fmf,:]  * options.aerothermo.CP_mult
                assembly.aerothermo.shear[self.hit_set_fmf] += shear * self.pf_set_fmf[self.hit_set_fmf,:] * options.aerothermo.CTau_mult

            else: 
                aerobridge = bridging(assembly.freestream, Kn_cont_pressure, Kn_free )
                pressures, shears = self.aerodynamics_module_bridging(assembly, aerobridge)
                assembly.aerothermo.pressure += pressures * options.aerothermo.CP_mult
                assembly.aerothermo.shear += shears * options.aerothermo.CTau_mult
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

        Kn_cont_heatflux = options.aerothermo.knc_heatflux       
        Kn_free = options.aerothermo.knf
        
        StConst = assembly.freestream.density*assembly.freestream.velocity**3 / 2.0
        if StConst<0.05: StConst = 0.05 # Neglect Cooling effect    
    
        # Heatflux calculation for Earth
        if options.planet.name == "earth":
            if  (assembly.freestream.knudsen <= Kn_cont_heatflux):
                assembly.aero_index = self.hit_set_c
                assembly.aerothermo.theta = self.theta_set_c
                assembly.aerothermo.partial_factor = self.pf_set_c
                assembly.aerothermo.heatflux[self.hit_set_c] = aerothermodynamics_module_continuum(assembly, self.hit_set_c, options)*StConst
                assembly.aerothermo.heatflux[self.hit_set_c] *= self.pf_set_c[self.hit_set_c] * options.aerothermo.CH_mult 
    
            elif (assembly.freestream.knudsen >= Kn_free):
                assembly.aero_index = self.hit_set_fmf
                assembly.aerothermo.theta = self.theta_set_fmf
                assembly.aerothermo.partial_factor = self.pf_set_fmf
                assembly.aerothermo.heatflux[self.hit_set_fmf] = aerothermodynamics_module_freemolecular(assembly, self.hit_set_fmf)*StConst
                assembly.aerothermo.heatflux[self.hit_set_fmf] *= assembly.aerothermo.partial_factor[self.hit_set_fmf] * options.aerothermo.CH_mult 
    
            else: 
                #atmospheric model for the aerothermodynamics bridging needs to be the NRLSMSISE00
                assembly.aerothermo.heatflux = self.aerothermodynamics_module_bridging(assembly, options)*StConst * options.aerothermo.CH_mult 
                
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

    def aerodynamics_module_bridging(self, assembly, aerobridge : float):
        """Pressure computation for Transitional regime
    :param assembly: Assembly object to process.
    :type assembly: object
    :param p: Value for p.
    :type p: Any
    :param aerobridge: Value for aerobridge.
    :type aerobridge: Any
    :param flow_direction: Value for flow direction.
    :type flow_direction: str
    :return: Return value.
    :rtype: Any"""

        assembly.aero_index = self.hit_set_c
        assembly.aerothermo.theta = self.theta_set_c
        assembly.aerothermo.partial_factor = self.pf_set_c

        Pcont = np.zeros_like(assembly.aerothermo.pressure)
        Pcont[self.hit_set_c] = aerodynamics_module_continuum(assembly, self.hit_set_c, None)
        
        assembly.aero_index = self.hit_set_fmf
        assembly.aerothermo.theta = self.theta_set_fmf
        assembly.aerothermo.partial_factor = self.pf_set_fmf
        Pfree = np.zeros_like(assembly.aerothermo.pressure)
        Sfree = np.zeros_like(assembly.aerothermo.shear)
        Pfree[self.hit_set_fmf], Sfree[self.hit_set_fmf] = aerodynamics_module_freemolecular(assembly, self.hit_set_fmf, -np.linalg.norm(assembly.velocity))

        Pressure = Pcont + (Pfree - Pcont)* aerobridge
        Shear = 0 + (Sfree - 0)* aerobridge

        return Pressure, Shear

    
    def aerothermodynamics_module_bridging(self, assembly, options):
        """Heatflux computation for the heat-flux regime
        :param assembly: Assembly object to process.
        :type assembly: object
        :param p: Value for p.
        :type p: Any
        :param flow_direction: Value for flow direction.
        :type flow_direction: str
        :param atm_data: Value for atm data.
        :type atm_data: Any
        :param Kn_cont: Value for kn cont.
        :type Kn_cont: Any
        :param Kn_free: Value for kn free.
        :type Kn_free: Any
        :param options: Options or configuration object.
        :type options: object
        :return: Return value.
        :rtype: Any"""

        lref = assembly.Lref
        free = assembly.freestream
        facet_radius = assembly.mesh.facet_radius
        facet_normal = assembly.mesh.facet_normal
        Kn_cont = options.aerothermo.knc_heatflux       
        Kn_free = options.aerothermo.knf
        #Computes the altitude of which the transition between flow regimes occur
        alt_cont, alt_free = bridging_altitudes("NRLMSISE00", Kn_cont, Kn_free, lref, options)
        
        free_cont = copy.copy(free)
        free_free = copy.copy(free)

        #Computes the freestream properties for the transition altitudes
        compute_freestream("NRLMSISE00", alt_cont, free.velocity, lref, free_cont, assembly, options)
        compute_freestream("NRLMSISE00", alt_free, free.velocity, lref, free_free, assembly, options)
        
        #HFcont = aerothermodynamics_module_continuum(nodes_normal,nodes_radius, free,p, wall_temperature, flow_direction, hf_model)
        #HFfree = aerothermodynamics_module_freemolecular(nodes_normal,free,p, flow_direction, wall_temperature)
        if not hasattr(options,'interp_bridge'): options.interp_bridge = create_thermal_bridge()

        #Interpolates the data according to experimental values and local radius to obtain a more accurate bridging factor

        Rmodels = np.array([0.0875,   #Mars Micro
                            0.664,    #Pathfinder
                            3.0,      #Average Rn
                            5.3])     #Orion CEV

        Thermal_bridge = np.array([interpolator(free.knudsen)[3] for interpolator in options.interp_bridge])


        Thermal_bridge[Thermal_bridge<0] = 0
        Thermal_bridge[Thermal_bridge>1] = 1 

        rN_bridge = np.copy(facet_radius)

        rN_bridge[rN_bridge > 5.3] = 5.3; # The maximum calibrated radius is 5.3m.
        rN_bridge[rN_bridge < 0.0875] = 0.0875; # The minimum calibrated radius is 0.0875m. (Mars Micro Probe)

        fBridge2 = PchipInterpolator(Rmodels, Thermal_bridge)
        BridgeReq = fBridge2(rN_bridge)

        compute_stagnation(free_cont, options.freestream)
        compute_stagnation(free_free, options.freestream)

        #Compute the Stanton number for both regimes, in the transition altitudes
        Stc = np.zeros_like(assembly.aerothermo.heatflux)
        assembly.aero_index = self.hit_set_c
        assembly.aerothermo.theta = self.theta_set_c
        assembly.aerothermo.partial_factor = self.pf_set_c
        Stc[self.hit_set_c] = aerothermodynamics_module_continuum(assembly, self.hit_set_c, options)
        

        Stfm = np.zeros_like(assembly.aerothermo.heatflux)
        assembly.aero_index = self.hit_set_fmf
        assembly.aerothermo.theta = self.theta_set_fmf
        assembly.aerothermo.partial_factor = self.pf_set_fmf
        Stfm[self.hit_set_fmf] = aerothermodynamics_module_freemolecular(assembly, self.hit_set_fmf)

        St = Stc + (Stfm - Stc) * BridgeReq

        St.shape = (-1)
        return St

def get_aero_configs(assembly, options) -> dict[AeroAttitude]:
    aero_configs = {}
    assembly_state = np.array(assembly.state_vector, copy=True)

    ## Continuum case...
    assembly.state_vector[:13] = base_opt_state_c
    update_dynamic_attributes(assembly, assembly.state_vector, options, force=True)

    continuum_surr = AeroSurrogate(training_iters=100, num_workers=10)
    continuum_surr.create_ground_truth_func(assembly, options)
    continuum_surr.sample(samples=500)
    continuum_surr.fit()

    assembly.state_vector[:13] = base_opt_state_fmf
    update_dynamic_attributes(assembly, assembly.state_vector, options, force=True)

    rarefied_surr = AeroSurrogate(training_iters=100, num_workers=10)
    rarefied_surr.create_ground_truth_func(assembly, options)
    rarefied_surr.sample(samples=500)
    rarefied_surr.fit()

    configurations = ['max_transverse','max_drag','min_drag']
    weights = [[-1.,1.],[1.,0.],[-1.,0.]]
    for i in range(len(assembly.objects)): 
        configurations.append('maxheat_'+str(i))
        weights.append([1])
        configurations.append('minheat_'+str(i))
        weights.append([-1])
    i_cfg = 0
    
    for configuration, weight in zip(configurations, weights):
        objective = configuration[3:] if 'heat' in configuration else 'ratio'
        continuum_opt = AeroOptimiser(assembly, {}, options, objective=objective, objective_weights=weight, visualise=False, surrogate=continuum_surr)
        continuum_opt.solve()

        rarefied_opt = AeroOptimiser(assembly, {}, options, objective=objective, objective_weights=weight, visualise=False, surrogate=rarefied_surr)
        rarefied_opt.solve()

        theta_c, pf_c, hits_c = continuum_opt.collect_theta_set(options)
        theta_f, pf_f, hits_f = rarefied_opt.collect_theta_set(options)
        aero_configs[configuration] = AeroAttitude(
            theta_set_c=theta_c, hit_set_c=hits_c, pf_set_c=pf_c, flow_dir_body_c=continuum_opt.flow_dir_body,
            theta_set_fmf=theta_f, hit_set_fmf=hits_f, pf_set_fmf=pf_f, flow_dir_body_fmf=rarefied_opt.flow_dir_body, tangents_fmf=rarefied_opt.tangent_vector
                )

        aero_configs[configuration].id = i_cfg
        i_cfg+=1
    assembly.state_vector = assembly_state
    update_dynamic_attributes(assembly, assembly.state_vector, options, force=True)

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
        k_state = np.array(state, copy=True)
        for i_coeff in range(i_k): 
            delta_tableau = k_n[i_coeff]*RK_tableaus[str(N)][i_k][i_coeff]*dt
            k_state += delta_tableau
        if i_k==0:
            d_dt = state_equation(assembly, options, aero_config, phi,k_state)
        else: d_dt = state_equation(assembly, options, aero_config, phi ,k_state)
        k_n.append(d_dt)
    new_state = np.array(state, copy=True)
    for i_k in range(RK_N_actual[str(N)]): 
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

def get_envelope_from_csv(filepath, output_directory):
    csv = pd.read_csv(filepath)

    points_ECEF_pos = csv[['ECEF_X','ECEF_Y','ECEF_Z']].to_numpy()
    points_ECEF_vel = csv[['ECEF_U','ECEF_V','ECEF_W']].to_numpy()
    
    points_GEO_pos = csv[['Longitude','Latitude','Altitude']].to_numpy()
    points_GEO_pos[:,-1] = 1e-3*points_GEO_pos[:,-1]

    points_GEO_vel = csv[['Velocity','Flight_path_angle','Heading_angle']].to_numpy()
    points_GEO_vel[:,-1] = 1e-3*points_GEO_pos[:,-1]
    
    points_DEMISE = csv[['Time','Mass','T']].to_numpy()

    points_list = [points_ECEF_pos, points_ECEF_vel, points_GEO_pos, points_GEO_vel, points_DEMISE]
    namelist = ['env_ECEF.stl','env_vECEF.stl','env_GEO.stl','env_vGEO.stl','env_DEMISE.stl']
    alpha_list = [2.5e6, 1e3, 100, 50.0, 100]

    pcd = o3d.geometry.PointCloud()
    
    for points, filename, alpha in zip(points_list, namelist, alpha_list):
        pcd.points = o3d.utility.Vector3dVector(points)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(output_directory+'/'+filename,mesh)
        exit()

