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
import numpy as np
from Freestream import mix_properties
from Dynamics.frames import *
from scipy import special
from copy import copy
from Aerothermo import su2, switch
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.spatial.transform import Rotation as Rot
import trimesh


#Not using it anymore -> switched to use Rays
def backfaceculling(body, nodes, nodes_normal, free_vector, npix):
    """
    Backface culling function

    This function detects the facets that are impinged by the flow

    Parameters
    ----------
    body: Assembly
        Object of Assembly class
    free_vector: np.array
        Array with the freestream direction with respect to the Body frame
    npix: int
        Resolution of the matrix used for the facet projection methodology (pixels)

    Returns
    -------
    node_points: np.array
        Array of IDs of the visible nodes
    """

    # Matrix of BOdy to ECEF frame
    R_B_ECEF_0 = Rot.from_quat(body.quaternion)
   
    #Matrix of ECEF to NED frame
    R_ECEF_NED_0 = R_NED_ECEF(lat = body.trajectory.latitude, lon = body.trajectory.longitude).inv()
    
    #Matrix of NED to Wind frame
    R_NED_W_0 = R_W_NED(ha = body.trajectory.chi, fpa = body.trajectory.gamma).inv()
    
    #Compute the rotation matrix from Body to Wind
    R_B_W_0 = R_NED_W_0*R_ECEF_NED_0*R_B_ECEF_0

    #number of pixels along each direction
    p_y = npix
    p_z = npix

    normals = np.copy(body.mesh.facet_normal)

    #vector of facets with chance to be wet
    p1 = np.dot(normals, free_vector)
    p1 = p1<0 

    x_elem_COG = R_B_W_0.apply(np.copy(body.mesh.facet_COG[p1]))[:,0]
    p = np.argsort(x_elem_COG)[::-1]

    facets = np.copy(body.mesh.facets[p1][p])
    elem_COG = R_B_W_0.apply(np.copy(body.mesh.facet_COG[p1]))[p,1:]

    # y,z 3D coordinates of each vertex of each triangular facet
    v0 = R_B_W_0.apply(np.copy(body.mesh.v0[p1]))[p,1:]
    v1 = R_B_W_0.apply(np.copy(body.mesh.v1[p1]))[p,1:]
    v2 = R_B_W_0.apply(np.copy(body.mesh.v2[p1]))[p,1:]

    v = np.stack([v0,v1,v2], axis = 0)
    v.shape = (-1,2)

    # image = np.zeros((p_z+1,p_y+1)).astype(np.uint8)
    image = np.zeros((p_z + 1, p_y + 1), dtype = bool)

    #bounding box in y,z for 3D points
    Start = np.min(v, axis = 0)#-body.mesh.min[1:]*0.01
    End   = np.max(v, axis = 0)#+body.mesh.max[1:]*0.01

    #Turn v0,v1,v2 and elem_COG into index, i.e., they become 2D coordinates in the pixel space
    v0 = (((v0 - Start)/(End - Start))*np.array([p_y,p_z])).astype(int)
    v1 = (((v1 - Start)/(End - Start))*np.array([p_y,p_z])).astype(int)
    v2 = (((v2 - Start)/(End - Start))*np.array([p_y,p_z])).astype(int)
    elem_COG = (((elem_COG - Start)/(End - Start))*np.array([p_y,p_z])).astype(int)

    node_points=[]#np.array([]).astype(int)

    #for each triangle, the bounding box in pixel space
    row_min = np.minimum(np.minimum(v0[:,1],v1[:,1]),v2[:,1])
    row_max = np.maximum(np.maximum(v0[:,1],v1[:,1]),v2[:,1])

    col_min = np.minimum(np.minimum(v0[:,0],v1[:,0]),v2[:,0])
    col_max = np.maximum(np.maximum(v0[:,0],v1[:,0]),v2[:,0])

    Area = 0.5 * (-v1[:,1]*v2[:,0] + v0[:,1]*(-v1[:,0]+v2[:,0])+ v0[:,0]*(v1[:,1] - v2[:,1])+v1[:,0]*v2[:,1])


    #Loop for each vertex
    #Check if the Center of Geometry of each facet is already inside any projected facet
    #If not, project the facet in the 2D matrix pixel space
    for i in range(len(v0)):
        if Area[i] == 0: continue
        if (image[elem_COG[i,1],elem_COG[i,0]]) == 0:
            node_points.append(facets[i,:])

            rows = np.arange(row_min[i],row_max[i]+1)
            cols = np.arange(col_min[i],col_max[i]+1)

            p = np.zeros((row_max[i]-row_min[i]+1,col_max[i]-col_min[i]+1,2)).astype(int)
            p[:,:,1] = rows[:,None]
            p[:,:,0] = cols[:,None].transpose()

            s = ( v0[i,1]*v2[i,0] - v0[i,0]*v2[i,1] + (v2[i,1] - v0[i,1])*p[:,:,0] + (v0[i,0] - v2[i,0])*p[:,:,1])/(2*Area[i])
            t = ( v0[i,0]*v1[i,1] - v0[i,1]*v1[i,0] + (v0[i,1] - v1[i,1])*p[:,:,0] + (v1[i,0] - v0[i,0])*p[:,:,1])/(2*Area[i])

            flag = (s>=0)*(t>=0)*(s+t<=1)

            image[row_min[i]:row_max[i] + 1, col_min[i]:col_max[i] + 1] += flag
            image[elem_COG[i,1],elem_COG[i,0]] = True

    node_points = np.array(node_points)
    node_points=np.sort(np.unique(node_points))

    return node_points

def compute_aerothermo(titan, options):
    """
    Fidelity selection for aerothermo computation

    Parameters
    ----------
    titan: Assembly_list
        Object of class Assembly_list
    options: Options
        Object of class Options
    """

    atmo_model = options.freestream.model
    
    for assembly in titan.assembly:
        #Compute the freestream properties and stagnation quantities
        mix_properties.compute_freestream(atmo_model, assembly.trajectory.altitude, assembly.trajectory.velocity, assembly.Lref, assembly.freestream, assembly, options)
        if assembly.freestream.mach >= 1: mix_properties.compute_stagnation(assembly.freestream, options.freestream)

    if options.fidelity.lower() == 'low':
        compute_low_fidelity_aerothermo(titan.assembly, options)
    elif options.fidelity.lower() == 'high':
        su2.compute_cfd_aerothermo(titan.assembly, options)
    elif options.fidelity.lower() == 'multi':
        switch.compute_aerothermo(titan, options)
    else:
        raise Exception("Select the correct fidelity options : (Low, High, Multi)")

def compute_aerodynamics(assembly, obj, index, flow_direction, options):
    """
    Low-fidelity computation of the aerodynamics (pressure, friction)

    Parameters
    ----------
    assembly: Assembly_list
        Object of class Assembly_list
    obj: Component
        Object of class Component
    index: np.array(int)
        Indexing list indicating nodes facing the flow (backface culling)
    flow_direction: np.array(float)
        Array indicating direction of the flow in the body frame
    options: Options
        Object of class Options
    """

    Kn_cont_pressure = options.aerothermo.knc_pressure
    Kn_free = options.aerothermo.knf

    #Pressure calculation only if Drag model is False
    if (not options.vehicle) or (options.vehicle and not options.vehicle.Cd):
        if  (assembly.freestream.knudsen <= Kn_cont_pressure):
            assembly.aerothermo.pressure[index] = aerodynamics_module_continuum(assembly.mesh.facet_normal, assembly.freestream, index, flow_direction)
    
        elif (assembly.freestream.knudsen >= Kn_free): 
            assembly.aerothermo.pressure[index], assembly.aerothermo.shear[index] = aerodynamics_module_freemolecular(assembly.mesh.facet_normal, assembly.freestream , index, flow_direction, assembly.aerothermo.temperature)
    
        else: 
            aerobridge = bridging(assembly.freestream, Kn_cont_pressure, Kn_free )
            assembly.aerothermo.pressure[index], assembly.aerothermo.shear[index] = aerodynamics_module_bridging(assembly.mesh.facet_normal, assembly.freestream, index, aerobridge, flow_direction, assembly.aerothermo.temperature)

def compute_aerothermodynamics(assembly, obj, index, flow_direction, options):
    """
    Low-fidelity computation of the aerothermodynamics (heat-flux)

    Parameters
    ----------
    assembly: Assembly_list
        Object of class Assembly_list
    obj: Component
        Object of class Component
    index: np.array(int)
        Indexing list indicating nodes facing the flow (backface culling)
    flow_direction: np.array(float)
        Array indicating direction of the flow in the body frame
    options: Options
        Object of class Options

    """

    Kn_cont_heatflux = options.aerothermo.knc_heatflux       
    Kn_free = options.aerothermo.knf

    StConst = assembly.freestream.density*assembly.freestream.velocity**3 / 2.0
    if StConst<0.05: StConst = 0.05 # Neglect Cooling effect    

    # Heatflux calculation for Earth
    if options.planet.name == "earth":
        if  (assembly.freestream.knudsen <= Kn_cont_heatflux):
            assembly.aerothermo.heatflux[index] = aerothermodynamics_module_continuum(assembly.mesh.facet_normal, assembly.mesh.facet_radius, assembly.freestream, index, assembly.aerothermo.temperature, flow_direction, options.aerothermo.heat_model)*StConst
        
        elif (assembly.freestream.knudsen >= Kn_free): 
            assembly.aerothermo.heatflux[index] = aerothermodynamics_module_freemolecular(assembly.mesh.facet_normal, assembly.freestream, index, flow_direction, assembly.aerothermo.temperature)*StConst
        
        else: 
            #atmospheric model for the aerothermodynamics bridging needs to be the NRLSMSISE00
            atmo_model = "NRLMSISE00"
            aerobridge = bridging(assembly.freestream, Kn_cont_heatflux, Kn_free )
            assembly.aerothermo.heatflux[index] = aerothermodynamics_module_bridging(assembly.mesh.facet_normal, assembly.mesh.facet_radius, assembly.freestream, index, assembly.aerothermo.temperature, flow_direction, atmo_model, options.aerothermo.heat_model, Kn_cont_heatflux, Kn_free, assembly.Lref, assembly, options)*StConst

    elif options.planet.name == "neptune" or options.planet.name == "uranus":
        #https://sci.esa.int/documents/34923/36148/1567260384517-Ice_Giants_CDF_study_report.pdf        
        assembly.aerothermo.heatflux[index] = aerothermodynamics_module_ice_giants(assembly, index, flow_direction, options)


def compute_low_fidelity_aerothermo(assembly, options) :
    """
    Low-fidelity aerothermo computation

    Function to compute the aerodynamic and aerothermodynamic using low-fidelity methods.
    It can compute from free-molecular to continuum regime. For the transitional regime, it uses a bridging methodology.

    Parameters
    ----------
    assembly: Assembly_list
        Object of class Assembly_list
    options: Options
        Object of class Options
    """

    heat_model = options.aerothermo.heat_model

    for it, _assembly in enumerate(assembly):
        _assembly.aerothermo.heatflux *= 0
        _assembly.aerothermo.pressure *= 0
        _assembly.aerothermo.shear    *= 0

        #Turning flow direction to ECEF -> Body to be used to the Backface culling algorithm
        flow_direction = -Rot.from_quat(_assembly.quaternion).inv().apply(_assembly.velocity)/np.linalg.norm(_assembly.velocity)

        mesh = trimesh.Trimesh(vertices=_assembly.mesh.nodes, faces=_assembly.mesh.facets)
        ray = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

        ray_list = _assembly.mesh.facet_COG - flow_direction*3*_assembly.Lref

        ray_directions = np.tile(flow_direction,len(ray_list))
        ray_directions.shape = (-1,3)

        index = np.unique(ray.intersects_first(ray_origins = ray_list, ray_directions = ray_directions))
        index = index[index != -1] 

        compute_aerothermodynamics(_assembly, [], index, flow_direction, options)
        compute_aerodynamics(_assembly, [], index, flow_direction, options)

def aerodynamics_module_continuum(facet_normal,free, p, flow_direction):
    """
    Pressure computation for continuum regime

    Function uses the Modified Newtonian Theory

    Parameters
    ----------
    nodes_normal: np.array
        List of the normals of each vertex on the surface
    free: Assembly.Freestream
        Freestream object
    p: np.array
        List of vertex IDs that are visible to the flow
    flow_direction: np.array
        Vector containing the flow_direction in the Body frame

    Returns
    -------
    Pressure: np.array
        Vector with pressure values
    """

    length_normal = np.linalg.norm(facet_normal, axis = 1, ord = 2)

    p = p*(length_normal[p] != 0)

    Theta =np.pi/2 - np.arccos(np.clip(np.sum(- flow_direction * facet_normal[p]/length_normal[p,None] , axis = 1), -1.0, 1.0))

    P0_s = free.P1_s
    Cpmax= (2.0/(free.gamma*free.mach**2.0))*((P0_s/free.pressure-1.0))

    #TODO
    if free.mach <= 1.1: Cpmax = 1

    Cp = Cpmax*np.sin(Theta)**2
    Cp[Theta < 0] = 0

    Pressure = Cp[:,None] * 0.5  *free.density * free.velocity**2

    Pressure[np.isnan(Pressure)] = 0
    Pressure.shape = (-1)

    return Pressure


def aerothermodynamics_module_ice_giants(assembly, index, flow_direction, options):
    """
    Low-fidelity computation of the aerothermodynamics in the Ice giants planet

    Parameters
    ----------
    assembly: Assembly_list
        Object of class Assembly_list
    index: np.array(int)
        Indexing list indicating nodes facing the flow (backface culling)
    flow_direction: np.array(float)
        Array indicating direction of the flow in the body frame
    options: Options
        Object of class Options

    Returns
    -------

    Q: np.array()
        Array of Heatflux values
    """


    length_normal = np.linalg.norm(assembly.mesh.nodes_normal, axis = 1, ord = 2)
    index = index*(length_normal[index] != 0)
    Theta =np.pi/2 - np.arccos(np.clip(np.sum(- flow_direction * assembly.mesh.nodes_normal[index]/length_normal[index,None] , axis = 1), -1.0, 1.0))

    if options.vehicle:
        nose_radius = options.vehicle.noseRadius
        assembly.aerothermo.qconvstag = 10000*9.08 * np.sqrt(1/(2*nose_radius)) * assembly.freestream.density ** (0.419778) * (assembly.freestream.velocity/1000) ** (2.67892)
        assembly.aerothermo.qradstag = 10000*0.091 * nose_radius * assembly.freestream.density ** (1.3344555) * (assembly.freestream.velocity/1000) ** (6.75706138)
        assembly.aerothermo.qstag = assembly.aerothermo.qconvstag+assembly.aerothermo.qradstag

    nodes_radius = assembly.mesh.nodes_radius[index]
    Qstag = 10000*9.08 * np.sqrt(1/(2*nodes_radius)) * assembly.freestream.density ** (0.419778) * (assembly.freestream.velocity/1000) ** (2.67892)
    Qradstag = 10000*0.091 * nodes_radius * assembly.freestream.density ** (1.3344555) * (assembly.freestream.velocity/1000) ** (6.75706138)

    K = 0.1
    Q = Qstag + Qradstag
    Q = Q*(K + (1-K)* np.sin(Theta)) #Lees laminar heat transfer distribution
    Q[Q<0] = 0

    return Q

def aerothermodynamics_module_continuum(facet_normal,facet_radius, free,p,body_temperature, flow_direction, hf_model):
    """
    Heatflux computation for continuum regime

    Function uses the Scarab equation (sc) or the Van Driest equation (vd)

    Parameters
    ----------
    nodes_normal: np.array
        List of the normals of each vertex on the surface
    nodes_radius: np.array
        Local radius of each vertex
    free: Assembly.Freestream
        Freestream object
    p: np.array
        List of vertex IDs that are visible to the flow
    body_temperature: float
        Temperature of the body
    flow_direction: np.array
        Vector containing the flow_direction in the Body frame
    hf_model: str
        Heatflux model to be used (default = ??, sc = Scarab, vd = Van Driest)

    Returns
    -------
    Stc: np.array
        Vector with Stanton number
    """

    length_normal = np.linalg.norm(facet_normal, ord = 2, axis = 1)
    p = p*(length_normal[p] != 0)

    Theta =np.pi/2 - np.arccos(np.clip(np.sum(- flow_direction * facet_normal[p]/length_normal[p,None] , axis = 1), -1.0, 1.0))

    T0s  = free.T1_s
    P02  = free.P1_s
    h0s  = free.h1_s
    rhos = free.rho_s

    Pr = free.prandtl
    mu_T0s = free.mu_s

    dudx = 1.0/facet_radius* np.sqrt(2*(P02-free.pressure)/rhos)

    StConst = free.density*free.velocity**3 / 2.0
    if StConst<0.05: StConst = 0.05 # Neglect Cooling effect (as in Fostrad)

    if hf_model == 'sc': #Scarab formulation and Lees distribution
        Re0norm = free.density * free.velocity / (free.mu *(T0s/free.temperature)**free.omega)
        Re0 = 2.0*facet_radius[p]*Re0norm
        Stc = 2.1/np.sqrt(Re0)
    
    if hf_model == 'vd': #Van Driest
        Stc = 0.763*(Pr**(-0.6))*(rhos*mu_T0s)**0.5*np.sqrt(dudx[p])*(h0s-free.cp*body_temperature[p])/StConst 

    K = 0.1

    Stc = Stc*(K + (1-K)* np.sin(Theta)) #Lees laminar heat transfer distribution 

    Stc[Stc < 0] = 0
    Stc.shape = (-1)

    return Stc

def aerothermodynamics_module_freemolecular(facet_normal, free, p, flow_direction, Wall_Temperature):
    """
    Heatflux computation for free-molecular regime

    Function uses the Schaaf and Chambre Theory
    Based on book of Wallace Hayes - Hypersonic Flow Theory

    Parameters
    ----------
    nodes_normal: np.array
        List of the normals of each vertex on the surface
    free: Assembly.Freestream
        Freestream object
    p: np.array
        List of vertex IDs that are visible to the flow
    Wall_temperature: float
        Temperature of the body
    flow_direction: np.array
        Vector containing the flow_direction in the Body frame


    Returns
    -------
    Stfm: np.array
        Vector with Stanton number
    """

    StConst = free.density*free.velocity**3 / 2.0
    if StConst<0.05: StConst = 0.05 # Neglect Cooling effect (as in Fostrad)

    length_normal = np.linalg.norm(facet_normal, ord = 2, axis = 1)
    p = p*(length_normal[p] != 0)

    Theta =np.pi/2 - np.arccos(np.clip(np.sum(- flow_direction * facet_normal[p]/length_normal[p,None] , axis = 1), -1.0, 1.0))

    AccCoeff = 1.0 #TODO Wall molecular diffusive accomodation coefficient
    SR = np.sqrt(0.5*free.gamma)*free.mach
    
    Q_fm = AccCoeff * free.pressure*np.sqrt(0.5*free.R*free.temperature/np.pi) * \
           ((SR**2 + free.gamma/(free.gamma - 1.0) - (free.gamma + 1.0)/(2 * (free.gamma - 1)) * Wall_Temperature[p] / free.temperature ) * \
           (np.exp(-(SR*np.sin(Theta))**2) + np.sqrt(np.pi) * (SR * np.sin(Theta)) * (1 + special.erf(SR*np.sin(Theta)))) - 0.5 * np.exp(-(SR*np.sin(Theta))**2))


    Stfm = Q_fm/StConst
    Stfm.shape = (-1)

    return Stfm

def aerodynamics_module_freemolecular(facet_normal,free,p, flow_direction, body_temperature):
    """
    Pressure computation for Free-molecular regime

    Function uses the Schaaf and Chambre theory

    Parameters
    ----------
    nodes_normal: np.array
        List of the normals of each vertex on the surface
    free: Assembly.Freestream
        Freestream object
    p: np.array
        List of vertex IDs that are visible to the flow
    flow_direction: np.array
        Vector containing the flow_direction in the Body frame
    body_temperature: float
        Temperature of the body
    Returns
    -------
    Pressure: np.array
        Vector with pressure values
    Shear: np.array
        Vector with skin friction values
    """

    length_normal = np.linalg.norm(facet_normal, ord = 2, axis = 1)
    Theta =np.pi/2 - np.arccos(np.clip(np.sum(- flow_direction * facet_normal[p]/length_normal[p,None] , axis = 1), -1.0, 1.0))

    SR = np.sqrt(0.5*free.gamma)*free.mach
    SN = 1.0 #TODO 0.93
    ST = 1.0 #TODO

    pfm1 = ((2 - SN)/np.sqrt(np.pi)*(SR*np.sin(Theta)) + 0.5*SN*np.sqrt(body_temperature[p]/free.temperature))*np.exp(-(SR*np.sin(Theta))**2.0)
    pfm2 = ((2 - SN)*(SR**2*np.sin(Theta)**2 + 0.5) + 0.5 * SN * np.sqrt(np.pi) * np.sqrt(body_temperature[p]/free.temperature) * (SR*np.sin(Theta)))*(1 + special.erf(SR*np.sin(Theta)))
    pfm = (1/SR**2)*(pfm1+pfm2)
    
    Pressure = pfm[:,None]*(0.5*free.density*free.velocity**2 )
    Pressure[np.isnan(Pressure)] = 0

    tfm = (ST*np.cos(Theta)/SR/np.sqrt(np.pi)) * (np.exp(-(SR*np.sin(Theta))**2.0) + np.sqrt(np.pi) * SR * np.sin(Theta) * (1 + special.erf(SR*np.sin(Theta))))
    Shear = tfm[:,None]*(0.5*free.density*free.velocity**2 )
    Shear[np.isnan(Shear)] = 0

    direction = np.copy(flow_direction)
    direction.shape = (-1)
    direction=np.tile(direction,(len(facet_normal[p]),1))

    tangent_vector = direction - ((direction*facet_normal[p]).sum(axis = 1))[:,None]*facet_normal[p]/(facet_normal[p]*facet_normal[p]).sum(axis=1)[:,None]
    tangent_vector = tangent_vector/np.sqrt((tangent_vector*tangent_vector).sum(axis=1)[:,None])
    
    Pressure.shape = (-1)
    Shear.shape = (-1)

    Shear = Shear[:,None]*tangent_vector
    return Pressure, Shear

def bridging(free, Kn_cont, Kn_free):

    """
    Computation of the bridging factor for the aeordynamic computation

    Parameters
    ----------
    free: Assembly.Freestream
        Freestream object
    Kn_cont: float
        Knudsen limit for the continuum regime
    Kn_free: float
        Knudsen limit for the free-molecular regime

    Returns
    -------
    AeroBridge: float
        Bridging factor
    """

    CF_ratiolow  = 0.1508
    CF_ratiohigh = 1e-6
    Kn_trans_R = (np.log(free.knudsen)-np.log(Kn_cont))/(np.log(Kn_free)- np.log(Kn_cont))

    BridgeCF = Kn_trans_R/((1+special.erf(Kn_trans_R*4.0-2.0))/2.0)
    if   (BridgeCF > 1): BridgeCF=(BridgeCF-1)*CF_ratiolow   + 1
    elif (BridgeCF < 1): BridgeCF=(BridgeCF-1)*CF_ratiohigh  + 1

    AeroBridge = (1+special.erf(Kn_trans_R*4-2.0))/2.0*BridgeCF
    return AeroBridge

def aerodynamics_module_bridging(facet_normal,free,p,aerobridge, flow_direction, wall_temperature):
    """
    Pressure computation for Transitional regime

    Parameters
    ----------
    nodes_normal: np.array
        List of the normals of each vertex on the surface
    free: Assembly.Freestream
        Freestream object
    p: np.array
        List of vertex IDs that are visible to the flow
    aerobridge: float
        Bridging value between 0 and 1
    flow_direction: np.array
        Vector containing the flow_direction in the Body frame
    body_temperature: float
        Temperature of the body
    Returns
    -------
    Pressure: np.array
        Vector with pressure values
    Shear: np.array
        Vector with skin friction values
    """

    Pcont = aerodynamics_module_continuum(facet_normal,free,p,flow_direction)
    Pfree, Sfree = aerodynamics_module_freemolecular(facet_normal,free, p, flow_direction, wall_temperature)

    Pressure = Pcont + (Pfree - Pcont)* aerobridge
    Shear = 0 + (Sfree - 0)* aerobridge

    return Pressure, Shear

def aerothermodynamics_module_bridging(facet_normal, facet_radius,free,p, wall_temperature, flow_direction, atm_data, hf_model, Kn_cont, Kn_free, lref, assembly, options):
    """
    Heatflux computation for the heat-flux regime

    Parameters
    ----------
    nodes_normal: np.array
        List of the normals of each vertex on the surface
    nodes_radius: np.array
        Local radius of each vertex
    free: Assembly.Freestream
        Freestream object
    p: np.array
        List of vertex IDs that are visible to the flow
    wall_temperature: float
        Temperature of the body
    flow_direction: np.array
        Vector containing the flow_direction in the Body frame
    atm_data: str
        Atmospheric model
    hf_model: str
        Heatflux model to be used (default = ??, sc = Scarab, vd = Van Driest)
    Kn_cont: float
        Knudsen limit for the continuum regime
    Kn_free: float
        Knudsen limit for the free-molecular regime
    lref: float
        Reference length
    options: Options
        Object of class Options

    Returns
    -------
    St: np.array
        Vector with Stanton number
    """

    #Computes the altitude of which the transition between flow regimes occur
    alt_cont, alt_free = bridging_altitudes(atm_data, Kn_cont, Kn_free, lref)
    
    free_cont = copy(free)
    free_free = copy(free)

    #Computes the freestream properties for the transition altitudes
    mix_properties.compute_freestream(atm_data, alt_cont, free.velocity, lref, free_cont, assembly, options)
    mix_properties.compute_freestream(atm_data, alt_free, free.velocity, lref, free_free, assembly, options)
    
    #HFcont = aerothermodynamics_module_continuum(nodes_normal,nodes_radius, free,p, wall_temperature, flow_direction, hf_model)
    #HFfree = aerothermodynamics_module_freemolecular(nodes_normal,free,p, flow_direction, wall_temperature)


    #Interpolates the data according to experimental values and local radius to obtain a more accurate bridging factor

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


    f2 = PchipInterpolator(Micro_breaks, Micro_coeffs)
    Thermal_bridge[0] = f2(free.knudsen)[3]

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

    MarsPath_bridge = MarsPath_coeffs[:,3]+MarsPath_breaks*MarsPath_coeffs[:,2]+MarsPath_breaks**2*MarsPath_coeffs[:,1]+MarsPath_breaks**3*MarsPath_coeffs[:,0]
    
    f2 = PchipInterpolator(MarsPath_breaks, MarsPath_coeffs)
    Thermal_bridge[1] = f2(free.knudsen)[3]

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

    MeanR_bridge = MeanR_coeffs[:,3]+MeanR_breaks*MeanR_coeffs[:,2]+MeanR_breaks**2*MeanR_coeffs[:,1]+MeanR_breaks**3*MeanR_coeffs[:,0]

    f2 = PchipInterpolator(MeanR_breaks, MeanR_coeffs)
    Thermal_bridge[2] = f2(free.knudsen)[3]

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

    Orion_bridge = Orion_coeffs[:,3]+Orion_breaks*Orion_coeffs[:,2]+Orion_breaks**2*Orion_coeffs[:,1]+Orion_breaks**3*Orion_coeffs[:,0]

    f2 = PchipInterpolator(Orion_breaks, Orion_coeffs)
    Thermal_bridge[3] = f2(free.knudsen)[3]

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
    Stc = aerothermodynamics_module_continuum(facet_normal, facet_radius,free_cont,p, wall_temperature, flow_direction, hf_model)
    Stfm = aerothermodynamics_module_freemolecular(facet_normal,free_free,p, flow_direction, wall_temperature)

    St = Stc + (Stfm - Stc) * BridgeReq[p]

    St.shape = (-1)
    return St

def bridging_altitudes(model, Kn_cont,Kn_free, lref):

    h_interval = np.linspace(1000,300000,25000)
    altitude_knudsen = mix_properties.interpolate_atmosphere_knudsen(model, lref, h_interval)

    alt_cont = altitude_knudsen(Kn_cont)
    alt_free = altitude_knudsen(Kn_free)

    return alt_cont, alt_free