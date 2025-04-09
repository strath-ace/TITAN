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
from Geometry import mesh as Mesh
from Geometry import gmsh_api as GMSH
from Geometry.tetra import inertia_tetra, vol_tetra
import numpy as np
from copy import deepcopy
import subprocess
import os

def create_assembly_flag(list_bodies, Flags):
    """
    Generates the assembly connectivity matrix

    Creates a flag m*n where m is the number of assemblies and n is the sum of all components used in the simulation.
    For every component belonging to a Body, the flag is True on that position.
    The assemblies are created according to the generated matrix

    Parameters
    ----------
    list_bodies: array of components
        array containing the used-defined components
    Flags: np.array
        numpy array containing the linkage information of each component

    Returns
    -------
    assembly_flag: np.array
        numpy array containing information on how to generate the assemblies with respect to the components introduced in the simulation
    """

    #Generates an empty (boolean set to False) 2D squared matrix with the size of the components list length
    assembly_flag = np.zeros((len(list_bodies),len(list_bodies)), dtype = bool)

    #Loop the components list
    #If they are Primitive or the mass is larger or equal than 0, the correspondent diagonal entry is set to true
    for i in range(len(list_bodies)):
        if list_bodies[i].type == 'Primitive' and list_bodies[i].mass >= 0:
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

    def __init__(self, objects):

        #: [array] List of components
        self.objects = np.array(objects)
        
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

    def create_assembly(self, connectivity, aoa = 0.0, slip = 0.0, roll = 0.0, options = None):            
        """
        Creates the assembly list

        Parameters
        ----------
        connectivty: array
            array containing the user-defined connectivity between the different components
        aoa: float
            Angle of attack in degrees
        slip: float
            Slip angle in degrees
        roll: float
            Roll angle in degrees
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
        assembly_flag = create_assembly_flag(self.objects, Flags)

        #loops the Assembly connectivity matrix in order to generate the different assemblies and append
        #them into a list.
        for i in range(len(assembly_flag)):
            self.assembly.append(Assembly(self.objects[assembly_flag[i]], self.id, aoa = aoa, slip = slip, roll = roll, options = options))
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

    def __init__ (self, roll = 0, pitch = 0, yaw = 0, vel_roll = 0, vel_pitch = 0, vel_yaw = 0):
        
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

class Body_force():
    """ Class Body_force
    
        A class to store the force and moment information that the assembly experiences at each iteration in the body frame
    """

    def __init__(self, force = np.zeros((3,1)), moment = np.zeros((3,1))):

        #: [np.array] Force array (3x1)
        self.force = force

        #: [np.array] Moment array (3x1)
        self.moment = moment

class Wind_force():
    """ Class Wind_force
    
        A class to store the force information that the assembly experiences at each iteration in the wind frame
    """
    def __init__(self, lift = 0, drag = 0, crosswind = 0):
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

    def __init__(self, pressure = 0, mach = 0, gamma = 0, knudsen = 0, prandtl = 0, temperature = 0, density = 0,
                 velocity = 0, R = 0, mfp = 0, omega = 0, diameter = 0, mu = 0, cp = 0, cv = 0):

#        self.fixed_freestream = False

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
        self.percent_mass = None

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

    def __init__(self,n_points, wall_temperature = 300):

        self.density = np.zeros((n_points))      
        self.temperature = np.zeros((n_points))

        #: [np.array] Pressure [Pa] 
        self.pressure = np.zeros((n_points))
        self.momentum = np.zeros((n_points,3))

        #: [np.array] Skin friction [Pa]
        self.shear = np.zeros((n_points,3))

        #: [np.array] Heatflux [W]
        self.heatflux = np.zeros((n_points))
        self.wall_temperature = wall_temperature

        self.theta = np.zeros((n_points))
        self.he = np.zeros((n_points))
        self.hw = np.zeros((n_points))
        self.Te = np.zeros((n_points))
        self.rhoe = np.zeros((n_points))
        self.ue = np.zeros((n_points))

        #Air-5 species + material element
        self.nSpecies = 6
        self.ce_i = np.zeros((n_points, self.nSpecies))

    def append(self, n_points = 0, temperature = 300):
        self.temperature = np.append(self.temperature, np.ones(n_points)*temperature)
        self.pressure = np.append(self.pressure, np.zeros(n_points))
        self.heatflux = np.append(self.heatflux, np.zeros(n_points))
        self.shear = np.append(self.shear, np.zeros((n_points,3)), axis = 0)
        self.theta = np.append(self.theta, np.zeros(n_points))
        self.Te = np.append(self.Te, np.zeros(n_points))
        self.he = np.append(self.he, np.zeros(n_points))
        self.hw = np.append(self.hw, np.zeros(n_points))
        self.rhoe = np.append(self.rhoe, np.zeros(n_points))
        self.ue = np.append(self.ue, np.zeros(n_points))
        self.ce_i = np.append(self.ce_i, np.zeros((n_points, self.nSpecies)))

    def delete(self, index):
        self.temperature = np.delete(self.temperature, index)
        self.pressure = np.delete(self.pressure, index)
        self.heatflux = np.delete(self.heatflux, index)
        self.shear = np.delete(self.shear, index, axis = 0)
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

    def __init__(self, objects = [], id = 0, aoa = 0.0, slip = 0.0, roll = 0.0, options = None):

        #: [int] ID of the assembly
        self.id = id

        #: [array] List of the components that are part of the assembly 
        self.objects = []

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
        self.Aref = 1.0

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

        if len(objects) != 0:
            self.objects=objects

            #Loop the components that belong to the assembly, and append the surface mesh
            for obj in objects:
                self.mesh = Mesh.append(self.mesh, obj.mesh)
                obj.parent_id = self.id

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
            for obj in objects:
                #obj.node_index, obj.node_mask = Mesh.create_index(self.mesh.nodes, obj.mesh.nodes)
                #obj.facet_index, obj.facet_mask = Mesh.create_index_facet(self.mesh.facet_COG, obj.mesh.facet_COG)
                obj.node_index  = Mesh.create_index_mapping(self.mesh.nodes, obj.mesh.nodes)
                obj.facet_index = Mesh.create_index_mapping(self.mesh.facet_COG, obj.mesh.facet_COG)

            #self.mesh.original_nodes = np.copy(self.mesh.nodes)
            self.inside_shock = np.zeros(len(self.mesh.nodes))

        self.Lref = np.max(self.mesh.xmax-self.mesh.xmin)

        self.aerothermo = Aerothermo(len(self.mesh.facets))
        self.aerothermo_cfd = Aerothermo(len(self.mesh.nodes))

        #Initialize surface temperature of the assembly
        for obj in self.objects:
            self.aerothermo.temperature[obj.facet_index] = obj.temperature

        self.collision = None

        self.emissivity = np.zeros(len(self.mesh.facets))
        self.material_density = np.zeros(len(self.mesh.facets))
        self.emissive_power = np.zeros(len(self.mesh.facets))
        self.total_emissive_power = 0
        self.hf_cond = np.zeros(len(self.mesh.facets))

        if options.thermal.ablation_mode.lower() == '0d':
            if options.thermal.post_fragment_tetra_ablation:
                if len(self.objects) > 1:
                    self.ablation_mode = '0d'
                else:
                    self.ablation_mode = 'tetra'
            else:
                self.ablation_mode = '0d'

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


    def generate_inner_domain(self, write = False, output_folder = '', output_filename = '', bc_ids = []):
        """
        Generates the 3D structural mesh

        Generates the tetrahedral inner domain using the GMSH software

        Parameters
        ----------
        write: bool
            Flag to output the 3D domain
        output_folder: str
            Directory of the output folder when writing the 3D domain
        output_filename: str
            Name of the file
        """


        print("Generating volumetric volume ... ")

        #Saves the 3D volumetric information
        self.mesh.vol_coords, self.mesh.vol_elements, self.mesh.vol_density, self.mesh.vol_tag = GMSH.generate_inner_domain(self.mesh, self, write = write, output_folder = output_folder, output_filename = output_filename, bc_ids = bc_ids)

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

        for obj in self.objects:
            index = (self.mesh.vol_tag == obj.id)
            self.mesh.vol_T[index] = obj.temperature            
            obj.mesh.vol_elements = np.copy(self.mesh.vol_elements[index])

        print("Create mapping between surface facets and tetras")
        self.mesh.index_surf_tetra = Mesh.map_surf_to_tetra(self.mesh.vol_coords, self.mesh.vol_elements)

        print("Done")

    def compute_mass_properties(self):
        """
        Computes the inertial properties

        Function to compute the inertial properties using the 3D domain information. 
        """

        coords = self.mesh.vol_coords
        elements = self.mesh.vol_elements
        density = self.mesh.vol_density
        tag = self.mesh.vol_tag
        vol = self.mesh.vol_volume

        #Computes the mass of every single tetrahedral
        self.mesh.vol_mass  = vol*density
        self.mass = np.sum(self.mesh.vol_mass)

        print('density:', density[0])

        print('volume:', np.sum(vol))

        print('mass:', self.mass)

        #Computes the Center of Mass
        if self.mass <= 0:
            self.COG = np.array([0,0,0])
        else:
            self.COG = np.sum(0.25*(coords[elements[:,0]] + coords[elements[:,1]] + coords[elements[:,2]] + coords[elements[:,3]])*self.mesh.vol_mass[:,None], axis = 0)/self.mass

        #Computes the inertia matrix
        self.inertia = inertia_tetra(coords[elements[:,0]],coords[elements[:,1]],coords[elements[:,2]], coords[elements[:,3]], vol, self.COG, density)

        #Loop over the components to compute each individual inertial properties
        for obj in self.objects:
            index = (tag == obj.id)
            obj.compute_mass_properties(coords, elements[index], density[index])

    def rearrange_ids(self):
        """
        Rearrange the objects ids and connectivity not to break in the fragmentation section
        """

        list_of_ids = [obj.id for obj in self.objects]
        new_ids = np.arange(1, len(list_of_ids)+1)
        
        #copy_connectivity = np.copy(self.connectivity)
        copy_vol_tag = np.copy(self.mesh.vol_tag)

        #print(self.objects, self.mesh.vol_tag, self.connectivity, list_of_ids, new_ids)
        #Organize into dictionary to easy access:
        d = {}
        for key, value in zip(list_of_ids, new_ids):
            d[key] = value

        #Change the values in the object
        for obj in self.objects:
            obj.id = d[obj.id]

        #Change the values in the connectivity matrix
        #Change the values in vol_tag
        for id in list_of_ids:
            #if len(self.connectivity):
            #    self.connectivity[copy_connectivity == id] = d[id]
            self.mesh.vol_tag[copy_vol_tag == id] = d[id]


def copy_assembly(list_assemblies, options):
    from copy import deepcopy

    if options.collision.flag:
        for assembly in list_assemblies:
            assembly.collision = None

    copy_list_assemblies = deepcopy(list_assemblies)

    return copy_list_assemblies