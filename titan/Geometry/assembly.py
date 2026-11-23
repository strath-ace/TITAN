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
#
"""The assembly module is used to create collections of components that fly together, they are connected
using connectivity lists specified in the configuration template, the format is as [[A,B,C],[D,E,0]] where
A is connect to B through C and D is connected to E through \"a virtual joint\" represented by 0 which has 
no mass or or volume.

The assembly class stores simulation data on trajectory, freestream, and aerothermodynamics as well as a list
of component objects as children. 

Assemblies are combined into the AssemblyList class which governs the global TITAN simulation of a population
of assemblies, global simulation parameters such as time, iteration and epoch are exposed here. 
This AssemblyList is often aliased as "titan" in other parts of the code as it is the highest level representation 
of a TITAN simulation
"""
from ..Geometry import mesh as Mesh
from ..Geometry import gmsh_api as GMSH
from ..Geometry.tetra import inertia_tetra, vol_tetra

import numpy as np
from copy import deepcopy
import subprocess
import os
from datetime import datetime as dt

def create_assembly_flag(list_bodies : list, Flags : np.ndarray):
    """Build connected-component membership from bodies and joint links.

    :param list_bodies: Components considered when forming assemblies.
    :type list_bodies: list
    :param Flags: Component-index pairs that represent connectivity links.
    :type Flags: np.ndarray
    :return: Boolean rows indicating component membership for each assembly.
    :rtype: numpy.ndarray
    """

    #Generates an empty (boolean set to False) 2D squared matrix with the size of the components list length
    assembly_flag = np.zeros((len(list_bodies),len(list_bodies)), dtype = bool)

    #Loop the components list
    #If they are Primitive or the mass is larger or equal than 0, the correspondent diagonal entry is set to true
    for i in range(len(list_bodies)):
        if not list_bodies[i].type == 'Joint' and list_bodies[i].mass >= 0:
            assembly_flag[i,i] = True
    
    #Loops the Flags variable
    #Sets the assembly_flag entries to True whenever the component are linked together
    for i in range(len(Flags)):
        assembly_flag[Flags[i,0]-1,Flags[i,1]-1] = True 
        assembly_flag[Flags[i,1]-1,Flags[i,0]-1] = True 

    #Loops the assembly flag
    #Sums the rows whenever a column has more than one True value
    for i in range(len(assembly_flag)):
        lines = np.nonzero(assembly_flag[:,i])[0]
        if len(lines) > 1:
            for line in lines[1:]:
                assembly_flag[lines[0]] += assembly_flag[line]
                assembly_flag[line] += assembly_flag[lines[0]]

    #Deletes repeated rows
    assembly_flag = np.unique(assembly_flag, axis = 0)
    assembly_flag = assembly_flag[np.sum(assembly_flag, axis = 1) != 0]

    return assembly_flag

class Assembly_list():
    """ Class Assembly list
    
        A class to store a list of assemblies and respective information, as well as the number of iterations and simulation time
    """

    def __init__(self, components : list =None, objects : list =None):
        """Initialize a simulation container from component objects.

        :param components: Components used to form the initial assemblies.
        :type components: list
        :param objects: Identical to components, kept for backwards compatibility.
        :type objects: list
        """

        if components is not None and objects is not None:
            raise TypeError("Use components or objects, not both")
        if components is None:
            components = [] if objects is None else objects

        #: [array] List of components
        self.components = np.array(components)
        
        #: [array] List of assemblies
        self.assembly = []

        #: [array] Number ID to identify the assembly. Whenever an assembly is generated (i.e. due to fragmentation/ablation or during the preprocessing phase), it will have this number ID.
        self.id = 1

        #: [float] simulation physical time 
        self.time = 0

        #: [float] nominal time step
        self.delta_t = 1.0

        #: [int] Iteration
        self.iter = 0

        #: [int] Iterations in reference to a fragmentation event, necessary for backward difference-style propagators
        self.post_event_iter = self.iter

        #: [array] List of the linkage information between the different components
        self.connectivity = np.array([], dtype = int)

        #: [int] Number of collisions that have ocurred
        self.n_collisions = 0

        #: [datetime] Epoch time of the simulation start
        self.epoch = dt.now()

    @property
    def objects(self):
        """Backward-compatible alias for components."""
        return self.components

    @objects.setter
    def objects(self, value):
        self.components = value

    def create_assembly(self, connectivity : list, aoa : float = 0.0, slip : float = 0.0, roll : float = 0.0, options = None):            
        """Group connected components into assemblies and initialize their state.

        :param connectivity: Component-link table describing joints and connections.
        :type connectivity: Any
        :param aoa: Initial angle of attack in radians.
        :type aoa: float
        :param slip: Initial sideslip angle in radians.
        :type slip: float
        :param roll: Initial roll angle in radians.
        :type roll: float
        :param options: Options or configuration object.
        :type options: Options
    """

        self.connectivity = connectivity
        Flags = np.array([], dtype = int)

        #Loops the user defined connectivity specified in the Config file
        #and generates a 2D numpy array with two columns.
        #
        #Each row contains two components that are linked 
        for i in range(len(connectivity)):
            if connectivity[i,2] == 0:
                Flags = np.append(Flags, [connectivity[i,0],connectivity[i,1]])
            else:
                Flags = np.append(Flags, np.array([connectivity[i,0],connectivity[i,2]]))
                Flags = np.append(Flags, np.array([connectivity[i,2],connectivity[i,1]]))

        Flags.shape = (int(len(Flags)/2) ,2)

        #Generates the Assembly connectivity matrix, used to combine the different components into assemblies
        assembly_flag = create_assembly_flag(self.components, Flags)

        #loops the Assembly connectivity matrix in order to generate the different assemblies and append
        #them into a list.
        for i in range(len(assembly_flag)):
            self.assembly.append(Assembly(self.components[assembly_flag[i]], self.id, aoa = aoa, slip = slip, roll = roll, options = options))
            self.id += 1
            connectivity_assembly = np.zeros(connectivity.shape, dtype = bool)
            id_objs = np.array(range(1,len(assembly_flag[i])+1))[assembly_flag[i]]
            for id in id_objs:
                connectivity_assembly += (connectivity == id)
            if len(connectivity_assembly) != 0:
                self.assembly[-1].connectivity = connectivity[(np.sum(connectivity_assembly, axis = 1) >= 2)]
            else: self.assembly[-1].connectivity = []
        if options is not None: self.delta_t = options.dynamics.time_step

class Dynamics():
    """ Class Dynamics
    
        A class to store the dynamics information of the assembly
    """

    def __init__ (self, roll : float = 0, pitch : float = 0, yaw : float = 0, vel_roll : float = 0, vel_pitch : float = 0, vel_yaw : float = 0):
        """Initialize body attitude and angular-velocity components.

        :param roll: Roll angle in radians.
        :type roll: float
        :param pitch: Pitch angle in radians.
        :type pitch: float
        :param yaw: Yaw angle in radians.
        :type yaw: float
        :param vel_roll: Roll angular velocity in radians per second.
        :type vel_roll: float
        :param vel_pitch: Pitch angular velocity in radians per second.
        :type vel_pitch: float
        :param vel_yaw: Yaw angular velocity in radians per second.
        :type vel_yaw: float
        """
        
        #: [float] Roll angle in radians
        self.roll = roll

        #: [float] Pitch angle in radians
        self.pitch = pitch

        #: [float] Yaw angle in radians
        self.yaw = yaw

        #: [float] Roll angular velocity in rad/s
        self.vel_roll = vel_roll

        #: [float] Pitch angular velocity in rad/s
        self.vel_pitch= vel_pitch

        #: [float] Yaw angular velocity in rad/s
        self.vel_yaw = vel_yaw

        #: [float] Instantaneous acceleration vector in m/s^2
        self.acceleration = np.zeros(3)

class Body_force():
    """ Class Body_force
    
        A class to store the force and moment information that the assembly experiences at each iteration in the body frame
    """

    def __init__(self, force : np.ndarray = np.zeros((3,1)), moment : np.ndarray = np.zeros((3,1))):
        """Initialise the force and moment vectors in the body frame

        :param force: 3x1 array describing the force vector in the body frame in Newtons
        :type force: np.ndarray
        :param moment: 3x1 array describing the moment vector in the body frame in Newton-metres
        :type moment: np.ndarray"""

        #: [np.array] Force array (3x1)
        self.force = force

        #: [np.array] Moment array (3x1)
        self.moment = moment

class Wind_force():
    """ Class Wind_force
    
        A class to store the force information that the assembly experiences at each iteration in the wind frame
    """
    def __init__(self, lift : float = 0.0, drag : float = 0.0, crosswind : float = 0.0):
        """Initialise the lift drag and crosswind terms (in the wind frame)

        :param lift: Scalar value for lift in Newtons
        :type lift: float
        :param drag: Scalar value for drag in Newtons
        :type drag: float
        :param crosswind: Scalar value for crosswind in Newtons
        :type crosswind: float"""

        #: [float] Lift force
        self.lift = lift

        #: [float] Drag force
        self.drag = drag

        #: [float] Crosswind force
        self.crosswind = crosswind

class Freestream():
    """ Class Freestream
    
        A class to store freestream information with respect to the position and velocity of each assembly
    """

    def __init__(self, pressure : float = 0, mach : float = 0, gamma : float = 0, knudsen : float = 0, prandtl : float = 0, 
                 temperature : float = 0, density : float = 0, velocity : float = 0, R : float = 0, mfp : float = 0, 
                 omega : float = 0, diameter : float = 0, mu : float = 0, cp : float = 0, cv : float = 0):
        """ Initialise the freestream conditions of the assembly


        :param pressure: Pressure (Pa).
        :type pressure: float
        :param mach: Mach number.
        :type mach: float
        :param gamma: Ratio of specific heats.
        :type gamma: float
        :param knudsen: Knudsen number.
        :type knudsen: float
        :param prandtl: Prandtl number.
        :type prandtl: float
        :param temperature: Temperature (K).
        :type temperature: float
        :param density: Density (kg/m3).
        :type density: float
        :param velocity: Velocity (m/s).
        :type velocity: float
        :param R: Specific Gas Constant (J/kg⋅K).
        :type R: float
        :param mfp: Mean free path (m).
        :type mfp: float
        :param omega: Mean viscosity coefficient.
        :type omega: float
        :param diameter: Mean diameter (m).
        :type diameter: float
        :param mu: Mean dynamic viscosity (Pa⋅s).
        :type mu: float
        :param cp: Specific heat capacity at constant pressure(J/kg⋅K).
        :type cp: float
        :param cv: Specific heat capacity at constant volume (J/kg⋅K).
        :type cv: float"""


        #:[float] Freestream pressure [Pa]
        self.pressure    = pressure  

        #:[float] Freestream mach 
        self.mach        = mach  

        #:[float] Freestream specific heat ratio
        self.gamma       = gamma      

        #:[float] Freestream knudsen
        self.knudsen     = knudsen  

        #:[float] Freestream prandtl  
        self.prandtl     = prandtl  

        #:[float] Freestream temperature [K]
        self.temperature = temperature

        #:[float] Freestream density [kg/m^3]
        self.density     = density  

        #:[float] Freestream velocity [m/s]
        self.velocity    = velocity 

        #:[float] Heat capacity at constant pressure
        self.cp          = cp

        #:[float] Heat capacity at constant volume
        self.cv          = cv
        
        #:[float] Gas constant
        self.R           = R  

        #:[float] mean free path in meters  
        self.mfp         = mfp 

        #:[float] Mean viscosity coefficient
        self.omega       = omega

        #:[float] Mean diameter
        self.diameter    = diameter

        #:[float] Mean viscosity
        self.mu          = mu
        #self.muEC        = muEC 
        #self.muSu        = muSu 

        self.kb          = 1.38064852e-23  #Boltzmann constant: J/K
        self.ninf = 0   # number density
        
        #:[array (float)] percentage of species in the mixture with respect to moles
        self.percent_gas = None

        #:[array (float)] percentage of species in the mixture with respect to mass
        self.mass_fraction = None

        #:[array (str)] list of species in the mixture
        self.species_index = None

        #:[float] Pressure at the stagnation point
        self.P1_s = 0

        #:[float] Temperature at the stagnation point
        self.T1_s = 0

        #:[float] Viscosity at the stagnation point
        self.mu_s = 0

        #:[float] Density at the stagnation point
        self.rho_s = 0

        #:[?float?] Specific enthalpy at the stagnation point 
        self.h1_s = 0

class Aerothermo():


    """ Class Aerothermo
    
        A class to store the surface quantities
    """

    def __init__(self,n_faces : int, wall_temperature : float = 300):
        """Initialise the surface quantities.

        :param n_faces: Number of surface facets on the mesh.
        :type n_faces: int
        :param wall_temperature: Wall temperature to populate surfaces with.
        :type wall_temperature: float"""

        self.density = np.zeros((n_faces))      
        self.temperature = np.zeros((n_faces))

        #: [np.array] Pressure [Pa] 
        self.pressure = np.zeros((n_faces))
        self.momentum = np.zeros((n_faces,3))

        #: [np.array] Skin friction [Pa]
        self.shear = np.zeros((n_faces,3))

        #: [np.array] Heatflux [W]
        self.heatflux = np.zeros((n_faces))
        self.wall_temperature = wall_temperature

        self.theta = np.zeros((n_faces))
        self.he = np.zeros((n_faces))
        self.hw = np.zeros((n_faces))
        self.Te = np.zeros((n_faces))
        self.rhoe = np.zeros((n_faces))
        self.ue = np.zeros((n_faces))
        self.debug_alpha = np.zeros((n_faces))

        #Air-5 species + material element
        self.nSpecies = 6
        self.ce_i = np.zeros((n_faces, self.nSpecies))

    def append(self, n_faces : int = 0, temperature : float = 300):
        """Add additional n_faces to the surface set

        :param n_faces: Number of new surface facets to append.
        :type n_faces: int
        :param temperature: Temperature to populate new facets with.
        :type temperature: float"""

        self.temperature = np.append(self.temperature, np.ones(n_faces)*temperature)
        self.pressure = np.append(self.pressure, np.zeros(n_faces))
        self.heatflux = np.append(self.heatflux, np.zeros(n_faces))
        self.shear = np.append(self.shear, np.zeros((n_faces,3)), axis = 0)
        self.debug_alpha = np.append(self.debug_alpha, np.zeros(n_faces))
        self.theta = np.append(self.theta, np.zeros(n_faces))
        self.Te = np.append(self.Te, np.zeros(n_faces))
        self.he = np.append(self.he, np.zeros(n_faces))
        self.hw = np.append(self.hw, np.zeros(n_faces))
        self.rhoe = np.append(self.rhoe, np.zeros(n_faces))
        self.ue = np.append(self.ue, np.zeros(n_faces))
        self.ce_i = np.append(self.ce_i, np.zeros((n_faces, self.nSpecies)))

    def delete(self, index : int):
        """Remove face from surface set by index
        :param index: Index to remove.
        :type index: int"""

        self.temperature = np.delete(self.temperature, index)
        self.pressure = np.delete(self.pressure, index)
        self.heatflux = np.delete(self.heatflux, index)
        self.shear = np.delete(self.shear, index, axis = 0)
        self.debug_alpha = np.delete(self.debug_alpha, index)
        self.theta = np.delete(self.theta, index)
        self.Te = np.delete(self.Te, index)
        self.he = np.delete(self.he, index)
        self.hw = np.delete(self.hw, index)
        self.rhoe = np.delete(self.rhoe, index)
        self.ue = np.delete(self.ue, index)
        self.ce_i = np.delete(self.ce_i, index)


class Assembly():
    """ Class Assembly
    
        A class to store the information respective to each assemly at every time iteration
    """

    @property
    def objects(self):
        """Backward-compatible alias for components."""
        return self.components

    @objects.setter
    def objects(self, value):
        self.components = value

    def __init__(self, components : list = None, id : int = 0, aoa : float = 0.0, slip : float = 0.0, roll : float = 0.0, options = None, objects : list = None):
        """Create an assembly from a list of components, an id and an attitude state.

        :param components: Components that are part of the assembly.
        :type components: Any
        :param objects: Backward-compatible alias for components.
        :type objects: Any
        :param id: Integer value for id.
        :type id: int
        :param aoa: Angle of attack in radians.
        :type aoa: float
        :param slip: Angle of sideslip in radians.
        :type slip: float
        :param roll: Roll angle in radians.
        :type roll: float
        :param options: Options or configuration object.
        :type options: Options"""

        if components is not None and objects is not None:
            raise TypeError("Use components or objects, not both")
        if components is None:
            components = [] if objects is None else objects

        #: [int] ID of the assembly
        self.id = id

        #: [array] List of the components that are part of the assembly 
        self.components = []

        #: [Mesh] Object of class Mesh containing the grid information
        self.mesh = Mesh.Mesh([])
        self.cfd_mesh = Mesh.Mesh([])
        self.trajectory = None 
        self.loads = None 
        self.fenics = None

        #: [float] Mass of the assembly [kg]
        self.mass = 0

        #: [array] Inertia matrix in the body frame [kg/m²]
        self.inertia = np.zeros((3,3))

        #: [array] CYZ coordinates of the center of mass in the body frame [meters]
        self.COG = np.array([0.,0.,0.])

        #: [float] Area of reference [meters^2]
        self.Aref = None

        #: [float] Length of reference [meters]
        self.Lref = 1.0

        #: [float] Angle of attack [radians]
        self.aoa = aoa

        #: [float] Slip angle [radians]
        self.slip = slip

        #: [Dynamics] Object of class Dynamics to store the dynamics information
        self.dynamics = Dynamics()

        #: [Body_force] Object of class Body_force to store the force and moment information in the body frame
        self.body_force = Body_force()

        #: [Body_force] Object of class Wind_force to store the force information in the wind frame
        self.wind_force = Wind_force()

        #: [Freestream] Object of class Freestream to store the freestream information
        self.freestream = Freestream()

        #: [Aerothermo] Object of class Aerothermo to store the surface quantities
        self.aerothermo = None

        # TODO
        # Need to check if these are used. They are repeated in the Dynamics function
        self.roll = roll
        self.pitch = 0  #Change this
        self.yaw = 0    #Change this
        self.roll_vel = 0
        self.pitch_vel = 0
        self.yaw_vel = 0

        if len(components) != 0:
            self.components=components

            #Loop the components that belong to the assembly, and append the surface mesh
            for component in components:
                self.mesh = Mesh.append(self.mesh, component.mesh)
                component.parent_id = self.id

            #Create the mapping between the facets and the vertex coordinates
            ___, self.mesh.facets = Mesh.map_facets_connectivity(self.mesh.v0, self.mesh.v1, self.mesh.v2) 
            self.cfd_mesh.facets = np.copy(self.mesh.facets)

            #Remove the repeated facets
            self.cfd_mesh.idx = Mesh.remove_repeated_facets(self.mesh) 

            #Volumetric meshing
            self.mesh.COG = Mesh.compute_geometrical_COG(self.mesh.facet_COG, self.mesh.facet_area)
            self.mesh.nodes, self.mesh.facets = Mesh.map_facets_connectivity(self.mesh.v0, self.mesh.v1, self.mesh.v2)
            self.mesh.min, self.mesh.max = Mesh.compute_min_max(self.mesh.nodes)
            self.mesh.edges, self.mesh.facet_edges = Mesh.map_edges_connectivity(self.mesh.facets)
            self.mesh.nodes_normal = Mesh.compute_nodes_normals(len(self.mesh.nodes), self.mesh.facets ,self.mesh.facet_COG, self.mesh.v0,self.mesh.v1,self.mesh.v2)
            self.mesh.xmin, self.mesh.xmax = Mesh.compute_min_max(self.mesh.nodes)
            self.mesh.nodes_radius, self.mesh.facet_radius, self.mesh.Avertex, self.mesh.Acorner = Mesh.compute_curvature(self.mesh.nodes, self.mesh.facets, self.mesh.nodes_normal, self.mesh.facet_normal, self.mesh.facet_area, self.mesh.v0, self.mesh.v1, self.mesh.v2)
            #self.mesh.facet_radius = np.ones((len(self.mesh.facets)))

            self.mesh.surface_displacement = np.zeros((len(self.mesh.nodes),3))

            #Create mapping between the nodes and facets of the singular component and the assembly
            for component in components:
                #obj.node_index, obj.node_mask = Mesh.create_index(self.mesh.nodes, obj.mesh.nodes)
                #obj.facet_index, obj.facet_mask = Mesh.create_index_facet(self.mesh.facet_COG, obj.mesh.facet_COG)
                component.node_index  = Mesh.create_index_mapping(self.mesh.nodes, component.mesh.nodes)
                component.facet_index = Mesh.create_index_mapping(self.mesh.facet_COG, component.mesh.facet_COG)

            #self.mesh.original_nodes = np.copy(self.mesh.nodes)
            self.inside_shock = np.zeros(len(self.mesh.nodes))

        # self.Lref = np.max(self.mesh.xmax-self.mesh.xmin)
        self.Lref = (self.mesh.xmax-self.mesh.xmin)[0]


        self.aerothermo = Aerothermo(len(self.mesh.facets))
        self.aerothermo_cfd = Aerothermo(len(self.mesh.nodes))

        if options.thermal.ablation_mode=='byproducts':
            from ..Thermal.byproducts import Byproducts
            self.byproducts = Byproducts(len(self.mesh.facets), cutoff=options.thermal.mf_cutoff)
            self.byproducts.get_species_list(self)
        #Initialize surface temperature of the assembly
        for component in self.components:
            self.aerothermo.temperature[component.facet_index] = component.temperature
            self.aerothermo.debug_alpha[component.facet_index] = component.debug_alpha

        self.collision = None

        self.emissivity = np.zeros(len(self.mesh.facets))
        self.material_density = np.zeros(len(self.mesh.facets))
        self.emissive_power = np.zeros(len(self.mesh.facets))
        self.total_emissive_power = 0
        self.hf_cond = np.zeros(len(self.mesh.facets))

        if options.thermal.ablation_mode.lower() == '0d':
            if options.thermal.post_fragment_tetra_ablation:
                if len(self.components) > 1:
                    self.ablation_mode = '0d'
                else:
                    self.ablation_mode = 'tetra'
            else:
                self.ablation_mode = '0d'
        elif options.thermal.ablation_mode.lower() == 'byproducts':
            self.ablation_mode = 'byproducts'
            self.mDotMelt = np.zeros(len(self.mesh.facets))

        elif options.thermal.ablation_mode.lower() == 'tetra':
            self.ablation_mode = 'tetra'

        elif options.thermal.ablation_mode.lower() == 'pato':
            self.ablation_mode = 'PATO'  
            if options.pato.Ta_bc == 'ablation':
                self.mDotVapor = np.zeros(len(self.mesh.facets))
                self.mVapor = np.zeros(len(self.mesh.facets))
                self.mDotMelt = np.zeros(len(self.mesh.facets))
                self.mMelt = np.zeros(len(self.mesh.facets))
                self.updated_gas_density = np.zeros(len(self.mesh.facets))
                self.LOS = np.zeros(len(self.mesh.facets))

        else: raise ValueError("Ablation mode has to be Tetra, 0D or PATO")

        self.distance_travelled = 0

        self.quaternion_prev = np.array([])

        self.aero_index = np.array([])

        self.blackbody_emissions_OI_surf  = np.zeros(len(self.mesh.facets))
        self.blackbody_emissions_AlI_surf = np.zeros(len(self.mesh.facets))
        self.atomic_emissions_OI_surf     = np.zeros(len(self.mesh.facets))
        self.atomic_emissions_AlI_surf    = np.zeros(len(self.mesh.facets))

        self.index_blackbody = np.array([])
        self.index_atomic    = np.array([])

        self.angle_blackbody = np.zeros(len(self.mesh.facets))
        self.angle_atomic    = np.zeros(len(self.mesh.facets))

        self.RIC = np.zeros(6)



    def generate_inner_domain(self, write : bool = False, output_folder : str = '', output_filename : str= '', 
                              bc_ids : list[int] = [], size_override : str =None, min_size : float = None):
        """Generates the 3D internal mesh

        :param write: Whether to write the mesh to a .vtk file, defaults to False
        :type write: bool
        :param output_folder: Path to the output folder
        :type output_folder: str
        :param output_filename: Output filename.
        :type output_filename: str
        :param bc_ids: List of boundary condition ids.
        :type bc_ids: list
        :param size_override: Set as either "joint" or "panel" to use a smaller cell size for thin parts, defaults to None.
        :type size_override: str
        :param min_size: Passed to gmsh as the minimum size option.
        :type min_size: float"""


        print("Generating volumetric volume ... ")

        #Saves the 3D volumetric information
        self.mesh.vol_coords, self.mesh.vol_elements, self.mesh.vol_density, self.mesh.vol_tag = GMSH.generate_inner_domain(self.mesh, 
                                                                                                                            self, 
                                                                                                                            write = write, 
                                                                                                                            output_folder = output_folder, 
                                                                                                                            output_filename = output_filename, 
                                                                                                                            bc_ids = bc_ids,
                                                                                                                            ref_obj_override = size_override,
                                                                                                                            min_size=min_size)

        self.mesh.volume_displacement = np.zeros((len(self.mesh.vol_coords),3))
        #self.mesh.original_vol_coords = np.copy(self.mesh.vol_coords)
        self.mesh.vol_T = np.ones(len(self.mesh.vol_elements))
        self.mesh.vol_orig_index = np.arange(len(self.mesh.vol_elements))

        coords = self.mesh.vol_coords
        elements = self.mesh.vol_elements

        #Computes the volume of every single tetrahedral
        vol = vol_tetra(coords[elements[:,0]],coords[elements[:,1]],coords[elements[:,2]], coords[elements[:,3]])
        self.mesh.vol_volume = vol

        #Copy original coords to use when fragmenting an object
        self.mesh.original_vol_coords = deepcopy(self.mesh.vol_coords)

        print("Volume Grid Completed")

        print("Passing Volume information to objects..")

        for component in self.components:
            index = (self.mesh.vol_tag == component.id)
            self.mesh.vol_T[index] = component.temperature            
            component.mesh.vol_elements = np.copy(self.mesh.vol_elements[index])

        print("Create mapping between surface facets and tetras")
        self.mesh.index_surf_tetra = Mesh.map_surf_to_tetra(self.mesh.vol_coords, self.mesh.vol_elements)

        print("Done")

    def compute_mass_properties(self):
        """Computes the inertial properties"""

        coords = self.mesh.vol_coords
        elements = self.mesh.vol_elements
        density = self.mesh.vol_density
        tag = self.mesh.vol_tag
        vol = self.mesh.vol_volume

        #Computes the mass of every single tetrahedral
        self.mesh.vol_mass  = vol*density
        self.mass = np.sum(self.mesh.vol_mass)


        #Computes the Center of Mass
        if self.mass <= 0:
            self.COG = np.array([0,0,0])
        else:
            self.COG = np.sum(0.25*(coords[elements[:,0]] + coords[elements[:,1]] + coords[elements[:,2]] + coords[elements[:,3]])*self.mesh.vol_mass[:,None], axis = 0)/self.mass

        #Computes the inertia matrix
        self.inertia = inertia_tetra(coords[elements[:,0]],coords[elements[:,1]],coords[elements[:,2]], coords[elements[:,3]], vol, self.COG, density)

        #Loop over the components to compute each individual inertial properties
        for component in self.components:
            index = (tag == component.id)
            component.compute_mass_properties(coords, elements[index], density[index])

    def rearrange_ids(self):
        """Rearrange the objects ids and connectivity not to break in the fragmentation section"""

        list_of_ids = [obj.id for obj in self.components]
        new_ids = np.arange(1, len(list_of_ids)+1)
        
        #copy_connectivity = np.copy(self.connectivity)
        copy_vol_tag = np.copy(self.mesh.vol_tag)

        #print(self.components, self.mesh.vol_tag, self.connectivity, list_of_ids, new_ids)
        #Organize into dictionary to easy access:
        d = {}
        for key, value in zip(list_of_ids, new_ids):
            d[key] = value

        #Change the values in the object
        for component in self.components:
            component.id = d[component.id]

        #Change the values in the connectivity matrix
        #Change the values in vol_tag
        for id in list_of_ids:
            #if len(self.connectivity):
            #    self.connectivity[copy_connectivity == id] = d[id]
            self.mesh.vol_tag[copy_vol_tag == id] = d[id]


def copy_assembly(list_assemblies : list[Assembly], options) -> list[Assembly]:
    """Creates a duplicate of a list of assemblies.

    :param list_assemblies: List of assemblies to copy.
    :type list_assemblies: list[Assembly]
    :param options: Options or configuration object.
    :type options: Options
    :return: New assembly list.
    :rtype: list[Assembly]"""
    from copy import deepcopy

    if options.collision.flag:
        for assembly in list_assemblies:
            assembly.collision = None

    copy_list_assemblies = deepcopy(list_assemblies)

    return copy_list_assemblies
