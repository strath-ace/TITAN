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
from Uncertainty.atmosphere import add_wind
import trimesh
try:
    import mutationpp as mpp
except:
    exit("Mutationpp library not set up")

def mixture_mpp():
    """
    Retrieve the mixture object of the Mutation++ library
    With the chemical reactions for air5
    """

    opts = mpp.MixtureOptions("air_5")
    opts.setThermodynamicDatabase("RRHO")
    opts.setStateModel("ChemNonEq1T")
    opts.setViscosityAlgorithm("Gupta-Yos")

    mix = mpp.Mixture(opts)
    
    return mix

### Stagnation Equations
def stagnation_P(P, gamma, M):
    P_0 = P * (1 + ((gamma - 1.0)/2.0)*(M**2))**(gamma / (gamma - 1))
    return P_0

def stagnation_T(T, gamma, M):
    T_0 = T * (1 + ((gamma - 1.0)/2.0)*(M**2))
    return T_0

### Normal Shock Equations
def normal_shock_P(P, gamma, M):
    P_post = P*((2.0 * gamma * (M**2)) - (gamma - 1.0)) / (gamma + 1.0)
    return P_post

def normal_shock_T(T, gamma, M):
    T_post = T*(((2.0 * gamma * (M**2.0)) - (gamma - 1.0)) * (((gamma - 1.0) * (M**2.0)) + 2.0)) / (((gamma + 1.0)**2.0) * (M**2.0))
    return T_post
    

def normal_shock_M(gamma, M):
    M_post = np.sqrt((((gamma - 1.0) * (M**2.0)) + 2.0) / ((2.0 * gamma * (M**2.0)) - (gamma - 1.0)))
    return M_post
    

def normal_shock_rho(rho, gamma, M):
    rho_post = rho*(((gamma + 1.0) * (M**2.0)) / (((gamma - 1.0) * (M**2.0)) + 2.0))
    return rho_post

### Loop to match total enthalpy (conserved)
def energy_loop(mix, T_eq, P_eq, h_ref):
    tol = 1
    h_eq = 0
    dT = 1

    while abs(h_ref-h_eq)>tol:
        mix.equilibrate(T_eq, P_eq)

        h_eq = mix.mixtureHMass()
        cp_eq = mix.mixtureFrozenCpMass()

        dT = (h_eq-h_ref)/cp_eq
        T_eq = T_eq - dT*0.1

    return mix

class flow_helper():
    """
    Class to store the flow conditions at freestream, stagnation, BLE and wall
    """

    def __init__(self, Tfree, Pfree, Mfree, Twall, mix = None):

        if mix == None: self.mix = mixture_mpp()
        else: self.mix = mix

        self.Tfree = Tfree
        self.Pfree = Pfree
        self.Mfree = Mfree
        self.Twall = Twall

        #Equilibrate the mix with the freesteam conditions:
        self.mix.equilibrate(self.Tfree, self.Pfree)        
        self.gammafree = self.mix.mixtureFrozenGamma()
        self.ufree = self.Mfree*self.mix.frozenSoundSpeed()
        self.mufree = self.mix.viscosity()
        self.rhofree = self.mix.density()
        self.c_i_free = self.mix.Y()

        #molecular weight
        self.MW_free = self.mix.mixtureMw()
        
        self.T0_free = stagnation_T(self.Tfree, self.gammafree, self.Mfree)
        self.P0_free = stagnation_P(self.Pfree, self.gammafree, self.Mfree)
        self.H0_free = self.mix.mixtureHMass() + (self.Mfree*self.mix.frozenSoundSpeed())**2/2.0

        #Post-shock conditions:
        self.T_post = normal_shock_T(self.Tfree, self.gammafree, self.Mfree)
        self.P_post = normal_shock_P(self.Pfree, self.gammafree, self.Mfree)
        self.rho_post = normal_shock_rho(self.rhofree, self.gammafree, self.Mfree)
        self.M_post = normal_shock_M(self.gammafree, self.Mfree)
        self.u_post = self.M_post*np.sqrt((self.gammafree*self.P_post)/self.rho_post)

        self.T0_post = stagnation_T(self.T_post, self.gammafree, self.M_post)
        self.P0_post = stagnation_P(self.P_post, self.gammafree, self.M_post)
        self.rho0_post = self.rho_post*(1+(self.gammafree - 1) / 2.0 * self.M_post**2)**(1/(self.gammafree - 1))

        #Boundary layer edge conditions
        #Assuming mixture at equilibrium
        self.Te = self.T0_post
        self.Pe = self.P0_post

        #Energy loop (Need to match the Total enthalpy)
        self.mix = energy_loop(self.mix, self.Te, self.Pe, self.H0_free)

        self.Te = self.mix.T()
        self.Pe = self.mix.P()
        self.rhoe = self.mix.density()
        self.mue = self.mix.viscosity()
        self.He = self.mix.mixtureHMass()

        #N O NO N2 O2 according to air_5 from Mutationpp
        self.ce_i = self.mix.Y()
        self.xe_i = self.mix.X()
        self.MWe = self.mix.mixtureMw()

        self.mix.setState(self.mix.densities(), self.Te, 1)
        self.mu_orig_e = self.mix.viscosity()

        #N - 33867025.2 J/Kg heat of formation
        #O - 15432544.8 J/Kg Heat of formation

        #Heat of dissociation                 
        self.Hd = 33867025.2*self.ce_i[0] + 15432544.8 *self.ce_i[1]

        #Wall conditions
        #Assuming mixture at equilibrium
        self.Pwall = self.Pe
        self.rhow = np.zeros(len(Twall))
        self.muw = np.zeros(len(Twall))
        self.Hw = np.zeros(len(Twall))

        for index, T in enumerate(Twall):
            self.mix.equilibrate(T, self.Pwall)
            self.rhow[index] = self.mix.density()
            self.muw[index] = self.mix.viscosity()
            self.Hw[index] = self.mix.mixtureHMass()

        #Adimensional numbers
        #At the moment these values are hardcoded according to several literature sources
        self.Pr = 0.71
        self.Le = 1.0

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
        if options.freestream.model=='GRAM':
            add_wind(assembly,options)

        #Compute the freestream properties and stagnation quantities
        mix_properties.compute_freestream(atmo_model, assembly.trajectory.altitude, assembly.trajectory.velocity, assembly.Lref, assembly.freestream, assembly, options)
        mix_properties.compute_stagnation(assembly.freestream, options.freestream)

    if options.fidelity.lower() == 'low':
        compute_low_fidelity_aerothermo(titan.assembly, options)
    elif options.fidelity.lower() == 'high':
        if options.cfd.cfd_restart: su2.restart_cfd_aerothermo(titan, options)
        else: su2.compute_cfd_aerothermo(titan, options)
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
            assembly.aerothermo.pressure[index] *= assembly.aerothermo.partial_factor[index]

        elif (assembly.freestream.knudsen >= Kn_free): 
            assembly.aerothermo.pressure[index], assembly.aerothermo.shear[index] = aerodynamics_module_freemolecular(assembly.mesh.facet_normal, assembly.freestream , index, flow_direction, assembly.aerothermo.temperature)
            assembly.aerothermo.pressure[index] *= assembly.aerothermo.partial_factor[index]
            assembly.aerothermo.shear[index] *= assembly.aerothermo.partial_factor[index,None]

        else: 
            aerobridge = bridging(assembly.freestream, Kn_cont_pressure, Kn_free )
            assembly.aerothermo.pressure[index], assembly.aerothermo.shear[index] = aerodynamics_module_bridging(assembly.mesh.facet_normal, assembly.freestream, index, aerobridge, flow_direction, assembly.aerothermo.temperature)
            assembly.aerothermo.pressure[index] *= assembly.aerothermo.partial_factor[index]
            assembly.aerothermo.shear[index] *= assembly.aerothermo.partial_factor[index,None]

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
            assembly.aerothermo.heatflux[index] = aerothermodynamics_module_continuum(assembly.mesh.facet_normal, assembly.mesh.facet_radius, assembly.freestream, index, assembly.aerothermo.temperature, flow_direction, options, assembly)*StConst
            assembly.aerothermo.heatflux[index] *= assembly.aerothermo.partial_factor[index] 

        elif (assembly.freestream.knudsen >= Kn_free): 
            assembly.aerothermo.heatflux[index] = aerothermodynamics_module_freemolecular(assembly.mesh.facet_normal, assembly.freestream, index, flow_direction, assembly.aerothermo.temperature)*StConst
            assembly.aerothermo.heatflux[index] *= assembly.aerothermo.partial_factor[index]

        else: 
            #atmospheric model for the aerothermodynamics bridging needs to be the NRLSMSISE00
            atmo_model = "NRLMSISE00"
            aerobridge = bridging(assembly.freestream, Kn_cont_heatflux, Kn_free )
            assembly.aerothermo.heatflux[index] = aerothermodynamics_module_bridging(assembly.mesh.facet_normal, assembly.mesh.facet_radius, assembly.freestream, index, assembly.aerothermo.temperature, flow_direction, atmo_model, Kn_cont_heatflux, Kn_free, assembly.Lref, assembly, options)*StConst
            assembly.aerothermo.heatflux[index] *= assembly.aerothermo.partial_factor[index] 


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

    def COG_subdivision(v0,v1,v2, COG, start, n, i = 1):

        v0v1 = (v0 + v1) / 2.0
        v0v2 = (v0 + v2) / 2.0
        v1v2 = (v1 + v2) / 2.0

        if i == n:

            COG[start+0::4**n,:] = (v0v1 + v0v2 + v0)/3.0
            COG[start+1::4**n,:] = (v0v1 + v1v2 + v1)/3.0
            COG[start+2::4**n,:] = (v0v2 + v1v2 + v2)/3.0
            COG[start+3::4**n,:] = (v0v1 + v0v2 + v1v2)/3.0

            return start + 4

        else:
            start = COG_subdivision(v0v1,v0v2, v0, COG, start, n, i+1)
            start = COG_subdivision(v0v1,v1, v1v2, COG, start, n, i+1)
            start = COG_subdivision(v0v2,v1v2, v2, COG, start, n, i+1)
            start = COG_subdivision(v0v1,v1v2, v0v2, COG, start, n, i+1)


    def edge_subdivision(v0,v1,v2, n):
    # Each subdivision level divides the triangle into 4 parts with equal areas
    # Function returns the number of triangles and the geometrical center of each generated triangle

        if n == 0:
            COG = (v0+v1+v2)/3.0

        else:
            COG = np.zeros((len(v0)*4**n,3))
            COG_subdivision(v0,v1,v2,COG, 0, n)

        return COG

    #Number of subdivisions
    n = options.aerothermo.subdivision_triangle

    for it, _assembly in enumerate(assembly):
        _assembly.aerothermo.heatflux *= 0
        _assembly.aerothermo.pressure *= 0
        _assembly.aerothermo.shear    *= 0

        #Turning flow direction to ECEF -> Body to be used to the Backface culling algorithm
        flow_direction = -Rot.from_quat(_assembly.quaternion).inv().apply(_assembly.velocity)/np.linalg.norm(_assembly.velocity)

        mesh = trimesh.Trimesh(vertices=_assembly.mesh.nodes, faces=_assembly.mesh.facets)
        ray = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)

        COG = edge_subdivision(_assembly.mesh.v0, _assembly.mesh.v1, _assembly.mesh.v2, n)

        ray_list = COG - 1E-4*flow_direction  #flow_direction*3*_assembly.Lref

        ray_directions = np.tile(-flow_direction,len(ray_list))
        ray_directions.shape = (-1,3)

        index = ~ray.intersects_any(ray_origins = ray_list, ray_directions = ray_directions)
        index.shape = (-1, 4**n)
        index = np.sum(index, axis = 1)

        _assembly.aerothermo.partial_factor = np.zeros(len(_assembly.mesh.facets)) + index/(4**n)

        index = np.arange(len(_assembly.mesh.facets))[index != 0]

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
    if free.mach <= 1.0: Cpmax = 1

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


    length_normal = np.linalg.norm(assembly.mesh.facet_normal, axis = 1, ord = 2)
    index = index*(length_normal[index] != 0)
    Theta =np.pi/2 - np.arccos(np.clip(np.sum(- flow_direction * assembly.mesh.facet_normal[index]/length_normal[index,None] , axis = 1), -1.0, 1.0))

    if options.vehicle:
        nose_radius = options.vehicle.noseRadius
        assembly.aerothermo.qconvstag = 10000*9.08 * np.sqrt(1/(2*nose_radius)) * assembly.freestream.density ** (0.419778) * (assembly.freestream.velocity/1000) ** (2.67892)
        assembly.aerothermo.qradstag = 10000*0.091 * nose_radius * assembly.freestream.density ** (1.3344555) * (assembly.freestream.velocity/1000) ** (6.75706138)
        assembly.aerothermo.qstag = assembly.aerothermo.qconvstag+assembly.aerothermo.qradstag

    facet_radius = assembly.mesh.facet_radius[index]
    Qstag = 10000*9.08 * np.sqrt(1/(2*facet_radius)) * assembly.freestream.density ** (0.419778) * (assembly.freestream.velocity/1000) ** (2.67892)
    Qradstag = 10000*0.091 * facet_radius * assembly.freestream.density ** (1.3344555) * (assembly.freestream.velocity/1000) ** (6.75706138)

    K = 0.1
    Q = Qstag + Qradstag
    Q = Q*(K + (1-K)* np.sin(Theta)) #Lees laminar heat transfer distribution
    Q[Q<0] = 0

    return Q

def aerothermodynamics_module_continuum(facet_normal,facet_radius, free,p,body_temperature, flow_direction, options, assembly):
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
        Heatflux model to be used (default = ??, sc = Scarab, vd = Van Driest, fr = Fay-Riddell, sg = Sutton-Graves)

    Returns
    -------
    Stc: np.array
        Vector with Stanton number
    """

    def FR(flow, vel_grad):
        q = 0.94*(flow.rhow*flow.muw)**0.1*(flow.rhoe*flow.mue)**0.4*(flow.He - flow.Hw)*np.sqrt(vel_grad)
        return q

    def FR_non_cat(flow, vel_grad):
        q = 0.94*(flow.rhow*flow.muw)**0.1*(flow.rhoe*flow.mue)**0.4*(flow.He - flow.Hw)*np.sqrt(vel_grad)*(1-flow.Hd/flow.He)
        return q

    def VD(flow, vel_grad):
        q = 0.94*(flow.rhoe*flow.mue)**0.5*(flow.He - flow.Hw)*np.sqrt(vel_grad)
        return q

    def SCARAB(flow, radius):

        ## In TITAN is 2*radius because fostrad assumes SCARAB uses diameter ?
        # In addition, Scarab uses that viscosity at stagnation point is given by the power law
        # And chemistry is not accounted for in this scarab formulation 

        # The equation
        #    Re = flow.rhofree * flow.ufree/flow.mue * (2* radius)
        # is replaced by
        Re = flow.rhofree * flow.ufree/(flow.mufree*(flow.T0_post/flow.Tfree)**0.75)
        Re0 = Re * (2* radius)
        St = 2.1/np.sqrt(Re0)
        q = St * 0.5*flow.rhofree*flow.ufree**3
        return q

    def SG(flow, radius):
        #K retrieved from Sutton graves paper
        q =  0.1117*np.sqrt(flow.Pe/radius)*(1/np.sqrt(101325))*(flow.He - flow.Hw)
        return q

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

    if free.mach < 1: hf_model = 'vd'

    if hf_model == 'sc': #Scarab formulation and Lees distribution
        # (OLD Fostrad equation)
        Re0norm = free.density * free.velocity / (free.mu *(T0s/free.temperature)**free.omega)
        Re0 = 2.0*facet_radius[p]*Re0norm
        Stc = 2.1/np.sqrt(Re0)
    
    if hf_model == 'vd': #Van Driest
        # (Old Fostrad equation)
        #This Van Driest formula is considering non-reacting flow, thus not accounting for changes in the mixture for the BLE
        Stc = 0.763*(Pr**(-0.6))*(rhos*mu_T0s)**0.5*np.sqrt(dudx[p])*(h0s-free.cp*body_temperature[p])/StConst 

    if hf_model == 'fr': #Fay Riddell
        mix = mixture_mpp()
        flow_ble = flow_helper(Tfree = free.temperature, Pfree = free.pressure, Mfree = free.mach, Twall = body_temperature[p], mix = mix)
        vel_grad = velocity_gradient(options.aerothermo.vel_grad, facet_radius[p], flow_ble, options.aerothermo.standoff)
        q = general_eq(flow_ble, vel_grad, 'fr')
        Stc = q/StConst

    if hf_model == 'fr_noncat': #Fay Riddell
        mix = mixture_mpp()
        flow_ble = flow_helper(Tfree = free.temperature, Pfree = free.pressure, Mfree = free.mach, Twall = body_temperature[p], mix = mix)
        vel_grad = velocity_gradient(options.aerothermo.vel_grad, facet_radius[p], flow_ble, options.aerothermo.standoff)
        q = general_eq(flow_ble, vel_grad, 'fr_noncat')        
        Stc = q/StConst

    if hf_model == 'fr_parcat': #Fay Riddell
        mix = mixture_mpp()
        flow_ble = flow_helper(Tfree = free.temperature, Pfree = free.pressure, Mfree = free.mach, Twall = body_temperature[p], mix = mix)
        vel_grad = velocity_gradient(options.aerothermo.vel_grad, facet_radius[p], flow_ble, options.aerothermo.standoff)
        q = general_eq(flow_ble, vel_grad, 'fr_parcat', cat_rate)       
        Stc = q/StConst

    if hf_model == 'sg': #Sutton_graves
        mix = mixture_mpp()
        flow_ble = flow_helper(Tfree = free.temperature, Pfree = free.pressure, Mfree = free.mach, Twall = body_temperature[p], mix = mix)
        vel_grad = velocity_gradient(options.aerothermo.vel_grad, facet_radius[p], flow_ble, options.aerothermo.standoff)
        q = general_eq(flow_ble, vel_grad, 'sg')
        Stc = q/StConst

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
def aerothermodynamics_module_bridging(facet_normal, facet_radius,free,p, wall_temperature, flow_direction, atm_data, Kn_cont, Kn_free, lref, assembly, options):
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
    Stc = aerothermodynamics_module_continuum(facet_normal, facet_radius,free_cont,p, wall_temperature, flow_direction, options, assembly)
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

### Standoff Distance:
def compute_delta(flow, method_delta):
    if method_delta.lower() == 'billig':
        return 0.143*np.exp(3.24/flow.Mfree**2)
    
    if method_delta.lower() == 'lobb':
        return 0.82*flow.rhofree/flow.rho_post

    if method_delta.lower() == 'serbin':
        M = flow.Mfree
        g = flow.gammafree
        return 2.0 / (3.0 * ((((g+1.0)**2*M**2)/(4*g*M**2-2*(g-1.0)))**(1/(g-1))*((g+1)*M**2)/(2+(g-1)*M**2)-1)) 
    
    if method_delta.lower() == 'probstein':
        ratio = flow.rhofree/flow.rho_post
        return ratio/(1-ratio+np.sqrt(8.0/3.0*ratio))
    
    if method_delta.lower() == "freeman":
        return flow.rhofree/flow.rho_post


### Velocity Gradient:
def velocity_gradient(method, radius, flow, method_delta = 'billig'):
    if method.lower() == "fr":
        return 1/radius*(np.sqrt(2*(flow.Pe - flow.Pfree)/flow.rhoe))

    if method.lower() == "linnell":
        k = flow.rhofree/flow.rho_post
        return flow.ufree/radius*np.sqrt(flow.rho_post/flow.rhoe*k*(2-k))

    if method.lower() == "newton":
        return flow.ufree/radius

    if method.lower() == "stokes":
        delta = compute_delta(flow, method_delta)
        return 3.0/2.0*flow.u_post/radius*(((1+delta)**3)/((1+delta)**3-1))

    if method.lower() == "olivier":
        delta = compute_delta(flow, method_delta)
        return flow.ufree/radius*(1+delta)/delta*(flow.Pe-flow.P_post)/(flow.rhofree*flow.ufree**2)*(flow.rho_post/flow.rho0_post)# or low.rhoe?? Do I need to equilibrate right after shock?

### Heatflux_equations:
def general_eq(flow, vel_grad, method = "FR", cat_rate = 0):
    q = flow.muw/flow.Pr* \
        detady(flow, vel_grad, method) * \
        dhdeta(flow, method) * (flow.He - flow.Hw) *\
        LAF(flow, method, cat_rate, vel_grad)
    
    return q

#Distance used in heat equations:
def detady(flow, vel_grad, method):
    return np.sqrt(2)*flow.rhow*np.sqrt(vel_grad)/(flow.rhoe*flow.mue)**0.5

#Approximation dh/dη
def dhdeta(flow, method):
    if method.lower() == 'fr' or method.lower() == 'fr_noncat' or method.lower() == 'fr_parcat':
        return 0.54*(flow.rhoe*flow.mue/flow.rhow/flow.muw)**0.9*flow.Pr**0.4
    if method.lower() == 'vd':
        return 0.54*(flow.rhoe*flow.mue/flow.rhow/flow.muw)**1.0*flow.Pr**0.4
    if method.lower() == 'sg':
        return 0.58*(flow.MW_free/flow.MWe)**(1.0/8.0)*(flow.rhoe*flow.mue/flow.rhow/flow.muw)*flow.Pr**0.4*np.sqrt(flow.mu_orig_e/flow.mue)


def coeff_goulard(flow, vel_grad, rate):
    #TODO. Not sure what would be the Sc number here, leaving to be approximatly one
    Sc = 1.0
    coeff =  1.0 / (1 + (0.47 * Sc **(-2/3.0) * (2*vel_grad*flow.mue * flow.rhoe) ** 0.5) / (flow.rhow * rate / (2*np.pi * 28.96 / (8.314)/ flow.Twall)) )

    return coeff

#Lewis augmentation factor:
def LAF(flow, method, cat_rate = 0, vel_grad = 0):
    if method == 'fr_noncat': return (1 - flow.Hd/flow.He)
    if method == 'fr_parcat': return (1+(flow.Le*coeff_goulard(flow, vel_grad, cat_rate) -1)*flow.Hd/flow.He)
    return 1