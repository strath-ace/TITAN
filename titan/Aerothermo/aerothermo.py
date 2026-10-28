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
"""aerothermo module."""
import numpy as np
from ..Freestream import mix_properties
from ..Dynamics.frames import *
from scipy import special
from copy import copy
from datetime import timedelta
from ..Aerothermo import su2, switch, sparta
from ..Aerothermo import thermo_equations as eqs
from ..Aerothermo import raytracer as ray
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import RigidTransform as Trans
from scipy.spatial import KDTree



def compute_aerothermo(titan, options):
    """Fidelity selection for aerothermo computation

    :param titan: TITAN simulation object.
    :type titan: AssemblyList
    :param options: Options object.
    :type options: Options"""

    atmo_model = options.freestream.model
    
    for assembly in titan.assembly:
        #Compute the freestream properties and stagnation quantities
        mix_properties.compute_freestream(atmo_model, assembly.trajectory.altitude, assembly.trajectory.velocity, assembly.Lref, 
                                          assembly.freestream, assembly, options, titan.epoch + timedelta(seconds=titan.time))
        mix_properties.compute_stagnation(assembly.freestream, options.freestream)

    if options.fidelity.lower() == 'low':
        titan.groups = compute_low_fidelity_aerothermo(titan.assembly, options, titan.epoch + timedelta(seconds=titan.time))
    elif options.fidelity.lower() == 'high':

        if  (assembly.freestream.knudsen <= options.aerothermo.knc_pressure):
            if options.cfd.cfd_restart: su2.restart_cfd_aerothermo(titan, options)
            else: su2.compute_cfd_aerothermo(titan, options)
        else:
            sparta.compute_dsmc_aerothermo(titan, options)

    elif options.fidelity.lower() == 'multi':
        switch.compute_aerothermo(titan, options)
    else:
        raise Exception("Select the correct fidelity options : (Low, High, Multi)")

def compute_aerodynamics(assembly, index : list, flow_direction : np.ndarray, options):
    """Low-fidelity computation of the aerodynamics (pressure, friction)

    :param assembly: Assembly object to process.
    :type assembly: object
    :param index: Integer value for index.
    :type index: int
    :param flow_direction: Unit vector pointing in the flow direction.
    :type flow_direction: np.ndarray
    :param options: Options object.
    :type options: Options"""

    Kn_cont_pressure = options.aerothermo.knc_pressure
    Kn_free = options.aerothermo.knf

    #Pressure calculation only if Drag model is False
    if (not options.vehicle) or (options.vehicle and not options.vehicle.Cd):
        if  (assembly.freestream.knudsen <= Kn_cont_pressure):
            assembly.aerothermo.pressure[index] += aerodynamics_module_continuum(assembly, index, flow_direction) \
                * assembly.aerothermo.partial_factor[index] * options.aerothermo.CP_mult

        elif (assembly.freestream.knudsen >= Kn_free): 
            pressure, shear = aerodynamics_module_freemolecular(assembly, index, flow_direction)
            assembly.aerothermo.pressure[index] += pressure * assembly.aerothermo.partial_factor[index]  * options.aerothermo.CP_mult
            assembly.aerothermo.shear[index] += shear * assembly.aerothermo.partial_factor[index,None] * options.aerothermo.CTau_mult

        else: 
            aerobridge = bridging(assembly.freestream, Kn_cont_pressure, Kn_free )
            pressures, shears = aerodynamics_module_bridging(assembly, index, aerobridge, flow_direction)
            assembly.aerothermo.pressure[index] += pressures * assembly.aerothermo.partial_factor[index] * options.aerothermo.CP_mult
            assembly.aerothermo.shear[index] += shears * assembly.aerothermo.partial_factor[index,None] * options.aerothermo.CTau_mult


def compute_aerothermodynamics(assembly, index : list, flow_direction : np.ndarray, options, epoch = None):
    """Low-fidelity computation of the aerothermodynamics (heat-flux)

    :param assembly: Assembly object to process.
    :type assembly: object
    :param index: Integer value for index.
    :type index: int
    :param flow_direction: Unit vector pointing in the flow direction.
    :type flow_direction: np.ndarray
    :param options: Options or configuration object.
    :type options: Options"""

    Kn_cont_heatflux = options.aerothermo.knc_heatflux       
    Kn_free = options.aerothermo.knf

    StConst = assembly.freestream.density*assembly.freestream.velocity**3 / 2.0
    if StConst<0.05: StConst = 0.05 # Neglect Cooling effect    

    # Heatflux calculation for Earth
    if options.planet.name == "earth":
        if  (assembly.freestream.knudsen <= Kn_cont_heatflux):
            assembly.aerothermo.heatflux[index] = aerothermodynamics_module_continuum(assembly, index, options)*StConst
            assembly.aerothermo.heatflux[index] *= assembly.aerothermo.partial_factor[index] * options.aerothermo.CH_mult 

        elif (assembly.freestream.knudsen >= Kn_free): 
            assembly.aerothermo.heatflux[index] = aerothermodynamics_module_freemolecular(assembly, index)*StConst
            assembly.aerothermo.heatflux[index] *= assembly.aerothermo.partial_factor[index] * options.aerothermo.CH_mult 

        else: 
            #atmospheric model for the aerothermodynamics bridging needs to be the NRLSMSISE00
            atmo_model = "NRLMSISE00" if not options.freestream.model.upper() == "PYMSIS_ONLINE" else options.freestream.model
            aerobridge = bridging(assembly.freestream, Kn_cont_heatflux, Kn_free )
            assembly.aerothermo.heatflux[index] = aerothermodynamics_module_bridging(assembly, index, flow_direction, atmo_model, 
                                                                                     Kn_cont_heatflux, Kn_free, options, epoch)*StConst
            assembly.aerothermo.heatflux[index] *= assembly.aerothermo.partial_factor[index] * options.aerothermo.CH_mult 


    elif options.planet.name == "neptune" or options.planet.name == "uranus":
        #https://sci.esa.int/documents/34923/36148/1567260384517-Ice_Giants_CDF_study_report.pdf        
        assembly.aerothermo.heatflux[index] = aerothermodynamics_module_ice_giants(assembly, index, options)


def compute_low_fidelity_aerothermo(assemblies, options, epoch = None) -> list:
    """Low-fidelity aerothermo computation
    
    :param assemblies: List of assembly objects
    :type assemblies: list[object]
    :param options: Options object.
    :type options: Options
    :return: Groups of assemblies as defined by Sphere of Influence.
    :rtype: list[list]"""
    for _assembly in assemblies: del _assembly.aero_index

    #Number of subdivisions
    n = options.aerothermo.subdivision_triangle
    flow_directions, groups, group_map = sphere_of_influence_assembly_groups(assemblies, options.aerothermo.SoI_rad)
    
    for it, _assembly in enumerate(assemblies):
        _assembly.aerothermo.heatflux *= 0
        _assembly.aerothermo.pressure *= 0
        _assembly.aerothermo.pressure += _assembly.freestream.pressure
        _assembly.aerothermo.shear    *= 0
        _assembly.aerothermo.he       *= 0
        _assembly.aerothermo.hw       *= 0
        _assembly.aerothermo.Te       *= 0
        _assembly.aerothermo.rhoe     *= 0
        _assembly.aerothermo.ue       *= 0
        _assembly.aerothermo.ce_i     *= 0

        _assembly.quaternion_prev = _assembly.quaternion #to be used in thermal model
        flow_direction = flow_directions[group_map[it]]
        if not hasattr(_assembly, 'aero_index'):
            ray.trace(groups[group_map[it]],n, options)
        else: 
            pass

        index = _assembly.aero_index
        compute_aerothermodynamics(_assembly, index, flow_direction, options, epoch = epoch)
        compute_aerodynamics(_assembly, index, flow_direction, options)
        #if options.pato.flag and options.pato.Ta_bc == "ablation": compute_equilibrium_chemistry(_assembly, options.aerothermo.mixture, index)
        #if options.pato: compute_frozen_chemistry(_assembly, options.aerothermo.mixture)
    return groups




def aerodynamics_module_continuum(assembly, p : np.ndarray, flow_direction : np.ndarray) -> np.ndarray:
    """Pressure computation for continuum regime

    :param assembly: Assembly object to process.
    :type assembly: Assembly
    :param p: Array of hit indices
    :type p: np.ndarray
    :param flow_direction: Value for flow direction.
    :type flow_direction: np.ndarray
    :return: Pressure.
    :rtype: np.ndarray"""
    facet_normal = assembly.mesh.facet_normal
    free = assembly.freestream

    length_normal = np.linalg.norm(facet_normal, axis = 1, ord = 2)

    p = p*(length_normal[p] != 0)

    #assembly.aerothermo.theta[p] =np.pi/2 - np.arccos(np.clip(np.sum(- flow_direction * facet_normal[p]/length_normal[p,None] , axis = 1), -1.0, 1.0))
    P0_s = free.P1_s
    Cpmax= (2.0/(free.gamma*free.per_facet_mach[p]**2.0))*((P0_s/free.pressure-1.0))

    #TODO
    if free.mach <= 1.0: Cpmax = 1

    Theta = assembly.aerothermo.theta[p]

    Cp = Cpmax*np.sin(Theta)**2
    Cp[Theta < 0] = 0

    Pressure = Cp[:,None] * 0.5  *free.density * free.velocity**2

    Pressure[np.isnan(Pressure)] = 0
    Pressure.shape = (-1)

    return Pressure


def aerothermodynamics_module_ice_giants(assembly, p : np.ndarray, options) -> np.ndarray:
    """Low-fidelity computation of the aerothermodynamics in the Ice giants planet

    :param assembly: Assembly object to process.
    :type assembly: Assembly
    :param p: Array of hit indices.
    :type p: np.ndarray
    :param options: Options or configuration object.
    :type options: Options
    :return: Return value.
    :rtype: Any"""

    length_normal = np.linalg.norm(assembly.mesh.facet_normal, axis = 1, ord = 2)
    p = p*(length_normal[p] != 0)
    
    Theta = assembly.aerothermo.theta[p]

    if options.vehicle:
        nose_radius = options.vehicle.noseRadius
        assembly.aerothermo.qconvstag = 10000*9.08 * np.sqrt(1/(2*nose_radius)) * assembly.freestream.density ** (0.419778) * (assembly.freestream.velocity/1000) ** (2.67892)
        assembly.aerothermo.qradstag = 10000*0.091 * nose_radius * assembly.freestream.density ** (1.3344555) * (assembly.freestream.velocity/1000) ** (6.75706138)
        assembly.aerothermo.qstag = assembly.aerothermo.qconvstag+assembly.aerothermo.qradstag

    facet_radius = assembly.mesh.facet_radius[p]
    Qstag = 10000*9.08 * np.sqrt(1/(2*facet_radius)) * assembly.freestream.density ** (0.419778) * (assembly.freestream.velocity/1000) ** (2.67892)
    Qradstag = 10000*0.091 * facet_radius * assembly.freestream.density ** (1.3344555) * (assembly.freestream.velocity/1000) ** (6.75706138)

    K = 0.1
    Q = Qstag + Qradstag
    Q = Q*(K + (1-K)* np.sin(Theta)) #Lees laminar heat transfer distribution
    Q[Q<0] = 0

    return Q

def aerothermodynamics_module_continuum(assembly, p : np.ndarray, options) -> np.ndarray:
    """Heatflux computation for continuum regime

    :param assembly: Assembly object to process.
    :type assembly: Assembly
    :param p: Array of hit indices
    :type p: np.ndarray
    :param options: Options or configuration object.
    :type options: Options
    :return: Array of Stanton numbers 
    :rtype: np.ndarray"""

    facet_normal = assembly.mesh.facet_normal
    facet_radius = assembly.mesh.facet_radius
    free = assembly.freestream
    body_temperature = assembly.aerothermo.temperature

    hf_model = options.aerothermo.heat_model

    if options.aerothermo.cat_method.lower() == 'constant':
        cat_rate = options.aerothermo.cat_rate
    elif options.aerothermo.cat_method.lower() == 'material':
        cat_rate = np.ones(len(facet_normal))
        for obj in assembly.objects:
            if obj.material.catalycity != None:
                cat_rate[obj.facet_index] = obj.material.catalycity

        cat_rate = cat_rate[p]
    else:
        raise ValueError("Error in catalicity method (constant or material)")


    length_normal = np.linalg.norm(facet_normal, ord = 2, axis = 1)
    p = p*(length_normal[p] != 0)

    Theta = assembly.aerothermo.theta[p]

    T0s  = free.T1_s
    P02  = free.P1_s
    h0s  = free.h1_s
    rhos = free.rho_s

    Pr = free.prandtl
    mu_T0s = free.mu_s

    dudx = 1.0/facet_radius* np.sqrt(2*(P02-free.pressure)/rhos)
    flow_ble = None

    StConst = free.density*free.velocity**3 / 2.0
    if StConst<0.05: StConst = 0.05 # Neglect Cooling effect (as in Fostrad)

    if free.mach < 1: hf_model = 'vd'

    match hf_model:

        case 'sc': #Scarab formulation and Lees distribution
            # (OLD Fostrad equation)
            Re0norm = free.density * free.velocity / (free.mu *(T0s/free.temperature)**free.omega)
            Re0 = 2.0*facet_radius[p]*Re0norm
            Stc = 2.1/np.sqrt(Re0)

        case 'vd': #Van Driest
            # (Old Fostrad equation)
            #This Van Driest formula is considering non-reacting flow, thus not accounting for changes in the mixture for the BLE
            Stc = 0.763*(Pr**(-0.6))*(rhos*mu_T0s)**0.5*np.sqrt(dudx[p])*(h0s-free.cp*body_temperature[p])/StConst 

        case 'fr': #Fay Riddell
            mix = eqs.mixture_mpp(options.aerothermo.mixture)
            flow_ble = eqs.stagnation_line(Tfree = free.temperature, Pfree = free.pressure, Mfree = free.mach, Twall = body_temperature[p], mix = mix)
            vel_grad = eqs.velocity_gradient(options.aerothermo.vel_grad, facet_radius[p], flow_ble, options.aerothermo.standoff)
            q = eqs.general_eq(flow_ble, vel_grad, 'fr')
            Stc = q/StConst

        case 'fr_noncat': #Fay Riddell
            mix = eqs.mixture_mpp(options.aerothermo.mixture)
            flow_ble = eqs.stagnation_line(Tfree = free.temperature, Pfree = free.pressure, Mfree = free.mach, Twall = body_temperature[p], mix = mix)
            vel_grad = eqs.velocity_gradient(options.aerothermo.vel_grad, facet_radius[p], flow_ble, options.aerothermo.standoff)
            q = eqs.general_eq(flow_ble, vel_grad, 'fr_noncat')        
            Stc = q/StConst

        case 'fr_parcat': #Fay Riddell
            do_pfm = False
            if do_pfm:
                Mfree = free.mach * np.sin(Theta)
                Mfree[Mfree<1] = 1
                flow_ble = eqs.stagnation_line_per_facet(Tfree = free.temperature, Pfree = free.pressure, Mfree = Mfree, Twall = body_temperature[p], mixname = options.aerothermo.mixture)
            else: 
                mix = eqs.mixture_mpp(options.aerothermo.mixture)
                flow_ble = eqs.stagnation_line(Tfree = free.temperature, Pfree = free.pressure, Mfree = free.mach, Twall = body_temperature[p], mix = mix)
            vel_grad = eqs.velocity_gradient(options.aerothermo.vel_grad, facet_radius[p], flow_ble, options.aerothermo.standoff)
            q = eqs.general_eq(flow_ble, vel_grad, 'fr_parcat', cat_rate)       
            Stc = q/StConst

        case 'sg': #Sutton_graves
            mix = eqs.mixture_mpp(options.aerothermo.mixture)
            flow_ble = eqs.stagnation_line(Tfree = free.temperature, Pfree = free.pressure, Mfree = free.mach, Twall = body_temperature[p], mix = mix)
            vel_grad = eqs.velocity_gradient(options.aerothermo.vel_grad, facet_radius[p], flow_ble, options.aerothermo.standoff)
            q = eqs.general_eq(flow_ble, vel_grad, 'sg')
            Stc = q/StConst

    K = 0.1
    Stc = Stc*(K + (1-K)* np.sin(Theta)) #Lees laminar heat transfer distribution 

    Stc[Stc < 0] = 0
    Stc.shape = (-1)
    if options.thermal.ablation_mode=='byproducts':
        if flow_ble is not None:
            assembly.byproducts.column_height_mix[p] = 0.99*flow_ble.u_post/vel_grad
            assembly.byproducts.rho_mix = flow_ble.rhofree#flow_ble.rhoe
            assembly.byproducts.c_i_mix = flow_ble.c_i_free#flow_ble.ce_i
            assembly.byproducts.T_mix = flow_ble.Twall#flow_ble.Te
            assembly.byproducts.P_mix = flow_ble.Pfree#flow_ble.Pe
            assembly.byproducts.oxy_content = flow_ble.oxygen_mf
        else:
            assembly.byproducts.column_height_mix[p] = np.zeros_like(assembly.aerothermo.theta)
    # if not pathlib.Path('./stagline_stations.csv').resolve().exists():
    #     header = True
    # else: header = False
    # ble_data_columns = ['Altitude']
    # for station in ['free', 'post', 'BLE']:
    #     for data_type in ['Ma_','P_','T_','rho_','%N_','%O_','%NO_','%N2_','%O2_']
    #         ble_data_columns.append(data_type+station)
    #ble_data = None
    # ble_data = [assembly.trajectory.altitude, flow_ble.Mfree, flow_ble.Pfree, flow_ble.Tfree, flow_ble.rhofree]
    # [ble_data.append(c_i) for c_i in flow_ble.c_i_free]
    # [ble_data.append(dat) for dat in [flow_ble.M_post, flow_ble.P_post, flow_ble.T_post, flow_ble.rho_post]]
    # [ble_data.append(c_i) for c_i in energy_loop(.c_pos).Y()]
    # for i_facet, f_id in enumerate(p):
    #     if ble_data is None: ble_data = [[assembly.trajectory.altitude,f_id, flow_ble.Mfree, flow_ble.Pfree, flow_ble.Tfree, rho_free]]
    #with open(str(pathlib.Path('./stagline_stations.csv').resolve()),'w') as f:

    return Stc

def aerothermodynamics_module_freemolecular(assembly, p : np.ndarray) -> np.ndarray:
    """Heatflux computation for free-molecular regime

    :param assembly: Assembly object to process.
    :type assembly: Assembly
    :param p: Array of hit indices
    :type p: np.ndarray
    :return: Array of Stanton numbers
    :rtype: np.ndarray"""

    facet_normal = assembly.mesh.facet_normal
    free = assembly.freestream
    Wall_Temperature = assembly.aerothermo.temperature

    StConst = free.density*free.velocity**3 / 2.0
    if StConst<0.05: StConst = 0.05 # Neglect Cooling effect (as in Fostrad)

    length_normal = np.linalg.norm(facet_normal, ord = 2, axis = 1)
    p = p*(length_normal[p] != 0)

    #assembly.aerothermo.theta[p] =np.pi/2 - np.arccos(np.clip(np.sum(- flow_direction * facet_normal[p]/length_normal[p,None] , axis = 1), -1.0, 1.0))

    Theta = assembly.aerothermo.theta[p]

    AccCoeff = 1.0 #TODO Wall molecular diffusive accomodation coefficient
    SR = np.sqrt(0.5*free.gamma)*free.per_facet_mach[p]
    
    Q_fm = AccCoeff * free.pressure*np.sqrt(0.5*free.R*free.temperature/np.pi) * \
           ((SR**2 + free.gamma/(free.gamma - 1.0) - (free.gamma + 1.0)/(2 * (free.gamma - 1)) * Wall_Temperature[p] / free.temperature ) * \
           (np.exp(-(SR*np.sin(Theta))**2) + np.sqrt(np.pi) * (SR * np.sin(Theta)) * (1 + special.erf(SR*np.sin(Theta)))) - 0.5 * np.exp(-(SR*np.sin(Theta))**2))


    Stfm = Q_fm/StConst
    Stfm.shape = (-1)

    return Stfm

def aerodynamics_module_freemolecular(assembly, p : np.ndarray, flow_direction : np.ndarray):
    """Pressure computation for Free-molecular regime

    :param assembly: Assembly object to process.
    :type assembly: Assembly
    :param p: Array of hit indices
    :type p: np.ndarray
    :param flow_direction: Unit vector in flow direction
    :type flow_direction: np.ndarray
    :return: Array of pressures, array of shears.
    :rtype: tuple[np.ndarray]"""

    facet_normal = assembly.mesh.facet_normal
    free = assembly.freestream
    body_temperature = assembly.aerothermo.temperature


    length_normal = np.linalg.norm(facet_normal, ord = 2, axis = 1)
    #assembly.aerothermo.theta[p] =np.pi/2 - np.arccos(np.clip(np.sum(- flow_direction * facet_normal[p]/length_normal[p,None] , axis = 1), -1.0, 1.0))

    Theta = assembly.aerothermo.theta[p]

    SR = np.sqrt(0.5*free.gamma)*free.per_facet_mach[p]
    # According to https://doi.org/10.1016/0032-0633(91)90020-B
    SN = 1.11-0.17/np.cos(np.abs(Theta-np.pi/2))
    ST = 0.93
    SN = np.clip(ST, 0.0, 1.0)

    pfm1 = ((2 - SN)/np.sqrt(np.pi)*(SR*np.sin(Theta)) + 0.5*SN*np.sqrt(body_temperature[p]/free.temperature))*np.exp(-(SR*np.sin(Theta))**2.0)
    pfm2 = ((2 - SN)*(SR**2*np.sin(Theta)**2 + 0.5) + 0.5 * SN * np.sqrt(np.pi) * np.sqrt(body_temperature[p]/free.temperature) * (SR*np.sin(Theta)))*(1 + special.erf(SR*np.sin(Theta)))
    pfm = (1/SR**2)*(pfm1+pfm2)
    
    Pressure = pfm[:,None]*(0.5*free.density*free.velocity**2 )
    Pressure[np.isnan(Pressure)] = 0

    tfm = (ST*np.cos(Theta)/SR/np.sqrt(np.pi)) * (np.exp(-(SR*np.sin(Theta))**2.0) + np.sqrt(np.pi) * SR * np.sin(Theta) * (1 + special.erf(SR*np.sin(Theta))))
    Shear = tfm[:,None]*(0.5*free.density*free.velocity**2 )
    Shear[np.isnan(Shear)] = 0

    if not hasattr(assembly.aerothermo,'tangent_vector'):

        direction = np.copy(flow_direction)
        direction[1] += 1e-8 # To prevent "bang-on" zero-norm tangent vectors
        direction /= np.linalg.norm(direction)
        direction.shape = (-1)
        direction=np.tile(direction,(len(facet_normal[p]),1))


        tangent_vector = direction - ((direction*facet_normal[p]).sum(axis = 1))[:,None]*facet_normal[p]/(facet_normal[p]*facet_normal[p]).sum(axis=1)[:,None]
        tangent_vector = tangent_vector/np.sqrt((tangent_vector*tangent_vector).sum(axis=1)[:,None])
    else: tangent_vector = assembly.aerothermo.tangent_vector
    Pressure.shape = (-1)
    Shear.shape = (-1)

    Shear = Shear[:,None]*np.atleast_2d(tangent_vector)
    return Pressure, Shear

def bridging(free, Kn_cont : float, Kn_free: float) -> float:
    """Computation of the bridging factor for the aeordynamic computation
    :param free: Freestream conditions
    :type free: Freestream
    :param Kn_cont: Continuum knudsen.
    :type Kn_cont: float
    :param Kn_free: Free molecular knudsen number.
    :type Kn_free: float
    :return: Bridging factor t ∈ [0,1].
    :rtype: float"""

    CF_ratiolow  = 0.1508
    CF_ratiohigh = 1e-6
    Kn_trans_R = (np.log(free.knudsen)-np.log(Kn_cont))/(np.log(Kn_free)- np.log(Kn_cont))

    BridgeCF = Kn_trans_R/((1+special.erf(Kn_trans_R*4.0-2.0))/2.0)
    if   (BridgeCF > 1): BridgeCF=(BridgeCF-1)*CF_ratiolow   + 1
    elif (BridgeCF < 1): BridgeCF=(BridgeCF-1)*CF_ratiohigh  + 1

    AeroBridge = (1+special.erf(Kn_trans_R*4-2.0))/2.0*BridgeCF
    return AeroBridge

def aerodynamics_module_bridging(assembly, p : np.ndarray, aerobridge : float, flow_direction : np.ndarray):
    """Pressure computation for Transitional regime

    :param assembly: Assembly object to process.
    :type assembly: Assembly
    :param p: Array of hit indices.
    :type p: np.ndarray
    :param aerobridge: Bridging factor t ∈ [0,1].
    :type aerobridge: float
    :param flow_direction: Unit vector in flow direction.
    :type flow_direction: np.ndarray
    :return: Pressure array, shear array.
    :rtype: tuple[np.ndarray]"""

    facet_normal = assembly.mesh.facet_normal
    free = assembly.freestream
    wall_temperature = assembly.aerothermo.temperature



    Pcont = aerodynamics_module_continuum(assembly, p, flow_direction)
    Pfree, Sfree = aerodynamics_module_freemolecular(assembly, p, flow_direction)

    Pressure = Pcont + (Pfree - Pcont)* aerobridge
    Shear = 0 + (Sfree - 0)* aerobridge

    return Pressure, Shear

def create_thermal_bridge():
    """Interpolates the data according to experimental values and local radius to obtain a more accurate bridging factor.
    
    Based upon the Mars Micro, Pathfinder and Orion CEV capsules along with an averaged nose radius

    :return: list of interpolators for the 4 cases
    :rtype: list[scipy.interpolation.PchipInterpolator]
    """
    #
    interpolators = []
    Rmodels = np.array([0.0875,   #Mars Micro
                        0.664,    #Pathfinder
                        3.0,      #Average Rn
                        5.3])     #Orion CEV

    Thermal_bridge = np.zeros((4))

    Micro_breaks = np.array([0.001, 0.0017, 0.0063, 0.0261, 0.0583, 0.2903, 0.9300, 2.2, 9.3, 79.8])
    Micro_coeffs = np.array([[-3491400.76967448, 1301.36167961469, 41.2299369761223, 0],
                             [-495178.115347958, 1498.63483389429, 37.9194841961614, 0.0283010726422985],
                             [6337.16885119717, -520.292748574922, 20.4908498052555, 0.185825845944000],
                             [-578.660943327858, -56.7472736230651, 7.34097688561876, 0.436749240415114],
                             [5.85844037619246, -5.49517312716001, 1.89134968638308, 0.594931001233851],
                             [0.0933853668135608, -0.278250671898889, 0.287410511387558, 0.811122770199370],
                             [0.00419312892561080, -0.0220736733955154, 0.0460577235912695, 0.905561385099685],
                             [1.11170515521551e-05, -0.000780200690684111, 0.0102802145472998, 0.937040923399790],
                             [3.30532319579131e-08, -9.75492854066848e-06, 0.000882596445817803, 0.974679444906316],
                             [3.30532319579131e-08, -9.75492854066848e-06, 0.000882596445817803, 0.974679444906316]])


    f_micro = PchipInterpolator(Micro_breaks, Micro_coeffs)
    interpolators.append(f_micro)

    MarsPath_breaks = np.array([0.00103, 0.00357, 0.014, 0.0271, 0.0547, 0.109, 0.206, 0.404, 1.54, 5.03, 24.1, 100])
    MarsPath_coeffs = np.array([[-434289.992260872, 409.141588399997, 27.0574112949722, 0],
                                [8968.42553338165, -518.860453182897, 20.7302546218335, 0.0642487046632124],
                                [-2524.01267973610, -149.524331276682, 12.8337133928574, 0.234196891191710],
                                [-705.575250716640, -51.0802316051753, 7.61673846549975, 0.370984455958549],
                                [78.6842497491937, -24.9647150334351, 3.18467267193635, 0.527461139896373],
                                [12.9643407989090, -5.39482234943265, 1.16950378993431, 0.639378238341969],
                                [-0.633675217101030, -0.649309891750675, 0.488852701875176, 0.713892426289748],
                                [0.0176076589687376, -0.0838525503839340, 0.157198175108222, 0.780310880829016],
                                [9.04305262329855e-05, -0.00484113006748949, 0.0348530210414839, 0.876489515791929],
                                [2.73454165324125e-06, -0.000178526372138471, 0.00436629172811849, 0.943005181347150],
                                [3.79269163767899e-09, -3.99349982287506e-06, 0.000540666345372661, 0.980310880829016],
                                [3.79269163767899e-09, -3.99349982287506e-06, 0.000540666345372661, 0.980310880829016]])
    
    f_mars = PchipInterpolator(MarsPath_breaks, MarsPath_coeffs)
    interpolators.append(f_mars)

    MeanR_breaks = np.array([0.001, 0.0033, 0.0073, 0.0161, 0.0456, 0.0788, 0.3857, 0.8532, 2.5, 7, 20, 100])
    MeanR_coeffs = np.array([[-25014.0060084375, -130.324225043622, 14.8316002530129, 0],
                             [-93038.9726637940, 216.378730476698, 13.8409792135817, 0.0329853075035311],
                             [2188.18942066116, -233.172651488835, 11.1266074737498, 0.0857301662534536],
                             [-235.155283061758, -46.8133661621039, 7.52283025568612, 0.167256748445935],
                             [-504.650095030751, -13.0877422038533, 4.14422385489454, 0.342486239744378],
                             [2.06975409022096, -3.04277995511635, 1.60872244924224, 0.447152590756571],
                             [0.181093568436558, -0.376124205150528, 0.325996551248476, 0.714092136992955],
                             [0.00289934055336248, -0.0288490000566234, 0.0930563937550737, 0.802795013423228],
                             [9.76192752829730e-05, -0.00259432494971926, 0.0216269952125893, 0.890752495187731],
                             [2.82868265276213e-06, -0.000189872684323298, 0.00420844163855657, 0.944434449872729],
                             [5.88313029489772e-09, -5.11781282977032e-06, 0.000705893951101214, 0.973270323311445],
                             [5.88313029489772e-09, -5.11781282977032e-06, 0.000705893951101214, 0.973270323311445]])

    f_mean = PchipInterpolator(MeanR_breaks, MeanR_coeffs)
    interpolators.append(f_mean)

    Orion_breaks = np.array([0.001, 0.0033, 0.0073, 0.0161, 0.04562, 0.0788, 0.3857, 0.8532 ,2.5,7,20,100])
    Orion_coeffs = np.array([[-16779.2722274906, -82.7046724094568, 8.06917824107994, 0],
                            [-87282.7376790222, 249.069257202799, 7.42627785930088, 0.0178457633428624],
                            [16724.0889896745, -294.011209950247, 5.24741349233841, 0.0458902191980685],
                            [-476.210482460133, 14.5441763778172, 3.96516647443931, 0.0807899864845479],
                            [-886.021627665031, 19.1806629801446, 3.57890958610900, 0.198265096011358],
                            [1.04505577211155, -2.91121839021878, 1.92514512255928, 0.305768843456316],
                            [0.186937795116975, -0.471887181374021, 0.433699200216903, 0.652585280865702],
                            [0.00444761436152593, -0.0383048048069601, 0.115051519182560, 0.771306810652743],
                            [0.000111056135577246, -0.00302891238480069, 0.0250748905331701, 0.876756956076944],
                            [3.68304211401379e-06, -0.000216439606659083, 0.00456133930628160, 0.938378478038472],
                            [4.83482194239522e-09, -5.58775291402762e-06, 0.000801211884950431, 0.969189239019236],
                            [4.83482194239522e-09, -5.58775291402762e-06, 0.000801211884950431, 0.969189239019236]])

    f_orion = PchipInterpolator(Orion_breaks, Orion_coeffs)
    interpolators.append(f_orion)
    return interpolators
def aerothermodynamics_module_bridging(assembly, p, flow_direction, atm_model, Kn_cont, Kn_free, options, epoch = None):
    """Heatflux computation for the transitional regime

    :param assembly: Assembly object to process.
    :type assembly: Assembly
    :param p: Array of hit indices
    :type p: np.ndarray
    :param flow_direction: Unit vector in flow direction.
    :type flow_direction: np.ndarray
    :param atm_model: Atmospheric model.
    :type atm_data: str
    :param Kn_cont: Continuum Knudsen number.
    :type Kn_cont: float
    :param Kn_free: Free molecular Knudsen number.
    :type Kn_free: float
    :param options: Options or configuration object.
    :type options: Options
    :return: Array of Stanton numbers.
    :rtype: np.ndarray"""

    lref = assembly.Lref
    free = assembly.freestream
    facet_radius = assembly.mesh.facet_radius
    facet_normal = assembly.mesh.facet_normal

    #Computes the altitude of which the transition between flow regimes occur
    alt_cont, alt_free = bridging_altitudes(atm_model, Kn_cont, Kn_free, lref, options, 
                                            [assembly.trajectory.latitude,assembly.trajectory.longitude], epoch)
    
    free_cont = copy(free)
    free_free = copy(free)

    #Computes the freestream properties for the transition altitudes
    mix_properties.compute_freestream(atm_model, alt_cont, free.velocity, lref, free_cont, assembly, options,epoch)
    mix_properties.compute_freestream(atm_model, alt_free, free.velocity, lref, free_free, assembly, options, epoch)
    
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
    
    length_normal = np.linalg.norm(facet_normal, ord = 2, axis = 1)
    p = p*(length_normal[p] != 0)

    mix_properties.compute_stagnation(free_cont, options.freestream)
    mix_properties.compute_stagnation(free_free, options.freestream)

    #Compute the Stanton number for both regimes, in the transition altitudes
    Stc = aerothermodynamics_module_continuum(assembly, p, options)
    Stfm = aerothermodynamics_module_freemolecular(assembly, p)

    St = Stc + (Stfm - Stc) * BridgeReq[p]

    St.shape = (-1)
    return St

def bridging_altitudes(model : str, Kn_cont : float, Kn_free : float, lref : float, options, latlon = None, epoch = None):
    """Bridging function for returning equivalent altitudes for continuum and free molecular flow

    :param model: Atmospheric model.
    :type model: str
    :param Kn_cont: Continuum Knudsen Number.
    :type Kn_cont: float
    :param Kn_free: Free molecular Knudsen Number.
    :type Kn_free: float
    :param lref: Reference length (m).
    :type lref: float
    :param options: Options object
    :type options: Options
    :param latlon: Latitude and longitude
    :type latlon: tuple, optional
    :param epoch: Atmospheric reference time
    :type epoch: datetime.datetime, optional 
    :return: Continuum and Free-Molecular Altitudes (m).
    :rtype: tuple[float]"""
    if not hasattr(options, 'kn_interp'):
        h_interval = np.linspace(1000,300000,25000)
        altitude_knudsen = mix_properties.interpolate_atmosphere_knudsen(model, lref, h_interval, options, latlon, epoch)
        options.kn_interp = altitude_knudsen
    alt_cont = options.kn_interp(Kn_cont)
    alt_free = options.kn_interp(Kn_free)

    return alt_cont, alt_free


def compute_per_facet_flow_dir(assembly,flow_direction : np.ndarray, do_pfm=False) -> np.ndarray:
    """Recalculates flow direction on a per-facet basis, returns array of flow directions and array of local Machs
    
    :param assembly: Assembly object to process.
    :type assembly: Assembly
    :param flow_direction: Unit vector pointing in flow direction.
    :type flow_direction: str
    :param do_pfm: Flag to calculate .
    :type do_pfm: Any
    :return: Array of Mach numbers
    :rtype: np.ndarray
    """

    free = assembly.freestream
    velocity_resultant = free.mach*free.sound*np.tile(flow_direction,[len(assembly.mesh.facet_area),1])
    if do_pfm:
        angular_velocity_vector = np.array([assembly.roll_vel,assembly.pitch_vel,assembly.yaw_vel])
        centroid_radii =  assembly.mesh.facet_COG - assembly.mesh.COG
        tangential_velocity = np.cross(angular_velocity_vector,centroid_radii)
        velocity_resultant -= tangential_velocity
        
    mach_resultant = np.linalg.norm(velocity_resultant,axis=1)/free.sound
    velocity_norm = np.linalg.norm(velocity_resultant, axis=1, keepdims=True)
    velocity_resultant = velocity_resultant/velocity_norm
    return velocity_resultant, mach_resultant


def sphere_of_influence_assembly_groups(assembly_list : list, sphere_radius : float):
    """Takes a list of assemblies and returns list of assembly groups by sphere of influence as well as mean flow direction and mapping dict

    :param assembly_list: List of assembly.
    :type assembly_list: list
    :param sphere_radius: Value for sphere radius.
    :type sphere_radius: Float
    :return: List of mean flow dirs, list of assembly groups,  mapping dictionary.
    :rtype: tuple"""

    # Firstly convert assembly positions into kd-tree and build base groupings
    positions = np.array([_assembly.position for _assembly in assembly_list])
    tree = KDTree(positions)
    groupings =[[i] for i in range(len(assembly_list))]

    # Then query pairs to find "constellations" of assemblies in space
    assembly_pairs = tree.query_pairs(sphere_radius)
    

    # Which can be assigned back to the "home" assembly
    for pair in assembly_pairs:
        groupings[pair[0]].append(pair[1])


    # Then iterate through groups in descending order (largest groups first)
    group_lengths = [len(group) for group in groupings]
    assembly_in_group = [False for _ in assembly_list]

    group_ids = []
    group_mapping = {}

    for i_group in np.argsort(group_lengths)[::-1]:
        if np.all(assembly_in_group): break # If we've finished assigning assemblies to groups
        # Otherwise assign the assemblies from the biggest group and remove them from other groups
        if len(groupings[i_group])>0:
            group_ids.append(groupings[i_group])
            for assem_id in groupings[i_group]:
                group_mapping[assem_id] = len(group_ids)-1
                assembly_in_group[assem_id] = True
                for group in groupings:
                    if len(group)>0:
                        if (not group==groupings[i_group]) and (assem_id in group):
                            group.remove(assem_id)

    # Finally collect the actual assembly reference and compute a mean flow vector for the group
    assembly_groups = []
    mean_flow_dirs = []

    for assembly_ids in group_ids: 
        assembly_groups.append([])
        [assembly_groups[-1].append(assembly_list[i_assem]) for i_assem in assembly_ids]
        flow_dirs = []
        for _assembly in assembly_groups[-1]:
            flow_dirs.append(-Rot.from_quat(_assembly.quaternion).inv().apply(_assembly.velocity))
            flow_dirs[-1] /= np.linalg.norm(flow_dirs[-1])
            assert np.isclose(np.linalg.norm(flow_dirs[-1]), 1.0)
        mean_flow_dirs.append(np.mean(flow_dirs,axis=0))

    return mean_flow_dirs, assembly_groups, group_mapping


