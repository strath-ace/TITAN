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
from Geometry.tetra import inertia_tetra, inertia_tetra_new, vol_tetra
from Material.material import Material
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from Geometry.tetra import volume_from_convex_hull

class Component_list():
    # A class with the purpose of storing the different components in a list
    def __init__(self):
        self.object = []
        self.id = 1
        
    def insert_component(self,filename, file_type, inner_stl = '', id = 0, binary = True, trigger_type = 'Indestructible', trigger_value = 0,fenics_bc_id = -1, material = 'Unittest', temperature = 300, options = None, global_ID = 0, bloom_config = [False, 0, 0, 0]):

        self.object.append(Component(filename, file_type, inner_stl = inner_stl, id = self.id, 
                           binary = binary, temperature = temperature, trigger_type = trigger_type,
                           trigger_value = trigger_value, fenics_bc_id = fenics_bc_id, material = material, options = options, global_ID = global_ID, bloom_config = bloom_config))
        self.id += 1

class Component():
    """ Component class

        Class to store the information of a singular component.
    """
    
    def __init__(self,filename, file_type, inner_stl = '', id = 0, binary = True, temperature = 300,
                 trigger_type = 'Indestructible', trigger_value = 0, fenics_bc_id = -1, material = 'Unittest',
                 v0 = [], v1 = [], v2 = [], parent_id = None, parent_part = None, options = None, global_ID = 0, bloom_config = [False, 0, 0, 0]):

        print("Generating Body: ", filename)
        
        #: [str] Name of the file where the mesh is stores
        self.name = filename

        #: [str] Type of the component (joint, primitive). Several sub-components can be used to form a larger component
        self.type = file_type
    
        #if self.type == "Joint":
        
        #: [str] Type of trigger for type joint (Altitude, Temperature, Stress)
        self.trigger_type = trigger_type

        #: [float] Value of the trigger criteria
        self.trigger_value = trigger_value

        #: [int] ID of the component
        self.id = id
        self.global_ID = global_ID
        self.inner_mesh = False

        mesh = Mesh.Mesh(filename)

        if filename == None:
            self.name = "New_component"
        
        if len(v0) != 0:# and v1 and v2:
            mesh.v0 = v0
            mesh.v1 = v1
            mesh.v2 = v2

        mesh = Mesh.compute_mesh(mesh, compute_radius = True) #TODO
        
        #: [Mesh] Object of class mesh that stores the mesh information
        self.mesh = mesh

        #: [kg] Mass of the component
        self.mass = 0

        #: [Material] Object of class Material to store the material properties
        self.material = Material(material, options)

        self.material_name = material

        #: [K] Temperature
        self.temperature = temperature

        #: [meters] Center of mass in XYZ coordinates
        self.COG = np.array([0.,0.,0.])


        #: [kg/m^2] Inertia matrix
        self.inertia = np.zeros((3,3))
        
        if inner_stl:
            #self.inner_mesh = True
            
            inner_mesh = Mesh.Mesh(inner_stl)
            self.inner_mesh = Mesh.compute_mesh(inner_mesh, compute_radius = False)
            #self.mesh.inner_nodes  = inner_mesh.nodes
            #self.mesh.inner_edges  = inner_mesh.edges
            #self.mesh.inner_facets = inner_mesh.facets
            #self.mesh.inner_facet_edges = inner_mesh.facet_edges
        
        self.fenics_bc_id = fenics_bc_id
        self.vol_id = -1

        self.max_stress = 0
        self.yield_stress = 0

        self.parent_id = 0
        self.parent_part = self.name #None
        
        if parent_id: 
            self.parent_id = parent_id
            self.parent_part = parent_part

        self.photons = 0

        #if options.thermal.ablation and options.thermal.ablation_mode.lower() == 'pato' and (not ("_joint" in self.name)):
        if options.thermal.ablation and options.thermal.ablation_mode.lower() == 'pato':      
            self.pato = PATO(options, len(mesh.facets), bloom_config, self.global_ID, self.temperature)
            self.bloom = bloom(bloom_config)        

    def compute_mass_properties(self, coords, elements, density):
        """
        Compute the inertia properties

        Uses the volumetric grid information, along with the material density to compute the mass,
        Center of mass and inertia matrix using tetras

        Parameters
        ----------
        coords: np.array
            numpy array containing the XYZ coordinates of the vertex of each tetrahedral element
        elements: np.array
            numpy array containing the connectivity information of each tetrahedral element
        """

        vol = vol_tetra(coords[elements[:,0]],coords[elements[:,1]],coords[elements[:,2]], coords[elements[:,3]])

        mass = vol*density
        self.mass = np.sum(mass)

        if self.mass <= 0:
            self.COG = np.array([0,0,0])
        else:
            self.COG = np.sum(0.25*(coords[elements[:,0]] + coords[elements[:,1]] + coords[elements[:,2]] + coords[elements[:,3]])*mass[:,None], axis = 0)/self.mass
        
        self.inertia = inertia_tetra(coords[elements[:,0]],coords[elements[:,1]],coords[elements[:,2]], coords[elements[:,3]], vol, self.COG, density)


    def compute_hybrid_mass_properties(self):

        print('\nHYBRID')

        density = self.material.density

        #Computes the inertia properties of the unstructured region

        coords = self.mesh.vol_coords

        #Computes the volume of every single tetrahedral
        elements = self.mesh.vol_elements_tetra

        vol_uns = vol_tetra(coords[elements[:,0]],coords[elements[:,1]],coords[elements[:,2]], coords[elements[:,3]])

        mass_uns = vol_uns*density
        self.mass_uns = np.sum(mass_uns)


        self.COG_uns = np.sum((1/4)*(coords[elements[:,0]] + coords[elements[:,1]] + coords[elements[:,2]] + coords[elements[:,3]])*mass_uns[:,None], axis = 0)/self.mass_uns

        self.local_COG_uns = (1/4)*(coords[elements[:,0]] + coords[elements[:,1]] + coords[elements[:,2]] + coords[elements[:,3]])*mass_uns[:,None]
        
        self.inertia_uns = inertia_tetra(coords[elements[:,0]],coords[elements[:,1]],coords[elements[:,2]], coords[elements[:,3]], vol_uns, self.COG_uns, density)
        #print('volume:', vol_uns);exit()
        #print('COG:', self.COG_uns)
        print('inertia_uns new:', self.inertia_uns)
        
        self.inertia_uns, self.mass_uns, self.COG_uns, vol_uns = inertia_prisms(coords[elements[:,0]],coords[elements[:,1]],coords[elements[:,2]], coords[elements[:,3]], density)
        #print('volume:', np.sum(vol_uns))
        #print('COG:', self.COG_uns)
        print('\ninertia_prisms:', self.inertia_uns)
        exit()
        
        print('v0:', coords[elements[:,0]])
        print('v1:', coords[elements[:,1]])
        print('v2:', coords[elements[:,2]])
        print('v3:', coords[elements[:,3]])


        print('volume:', np.sum(vol_uns))
        print('mass:', self.mass_uns)
        print('density:', density)
        print('COG:', self.COG_uns)
        print('inertia_uns:', self.inertia_uns)

        exit()
        #Computes the inertia properties of the structured region composed of triangular prisms

        elements = self.mesh.vol_elements_prism
        #vol_str = vol_triangular_prism(coords[elements[:,0]],coords[elements[:,1]],coords[elements[:,2]], coords[elements[:,3]], coords[elements[:,4]], coords[elements[:,5]])

        #total_volume = np.sum(vol_uns) + np.sum(vol_prism)

        self.inertia_str, self.mass_str, self.COG_str, vol_prism = inertia_prisms(coords[elements[:,0]],coords[elements[:,1]],coords[elements[:,2]], coords[elements[:,3]], coords[elements[:,4]], coords[elements[:,5]], density)

        print('inertia_str:', self.inertia_str)


        #Combine unstructured and structured regions

        self.mesh.vol_volume = np.concatenate((vol_uns, vol_prism), axis=0)        

        self.inertia, self.mass, self.COG = combine_inertia_tensors(self.inertia_uns, self.mass_uns, self.COG_uns, self.inertia_str, self.mass_str, self.COG_str)

        
        print('volume:', np.sum(vol_uns) + np.sum(vol_prism))
        print('mass:', self.mass)
        print('COG:', self.COG)
        print('inertia:', self.inertia)
        exit()


# Function to apply the parallel axis theorem to translate inertia tensor
def translate_inertia(inertia_tensor, mass, region_com, total_com):
    d = region_com - total_com  # Vector from region COM to total COM
    d_squared = np.dot(d, d)
    translation_inertia = mass * (d_squared * np.eye(3) - np.outer(d, d))
    return inertia_tensor + translation_inertia

# Function to calculate the total inertia tensor of the body
def combine_inertia_tensors(inertia1, mass1, com1, inertia2, mass2, com2):
    # Calculate total mass
    total_mass = mass1 + mass2
    
    # Calculate total center of mass
    total_com = (mass1 * com1 + mass2 * com2) / total_mass
    
    # Translate each region's inertia tensor to the total center of mass
    translated_inertia1 = translate_inertia(inertia1, mass1, com1, total_com)
    translated_inertia2 = translate_inertia(inertia2, mass2, com2, total_com)
    
    # Sum the translated inertia tensors to get the total inertia tensor
    total_inertia_tensor = translated_inertia1 + translated_inertia2
    
    return total_inertia_tensor, total_mass, total_com

# Function to calculate the center of mass of a polyhedron
def center_of_mass(points):
    return np.mean(points, axis=0)

# Function to compute inertia tensor for a single polyhedron relative to its center of mass
def moment_of_inertia(points, mass, center_of_mass):
    inertia_tensor = np.zeros((3, 3))
    for p in points:
        r = p - center_of_mass
        inertia_tensor[0, 0] += mass * (r[1]**2 + r[2]**2)
        inertia_tensor[1, 1] += mass * (r[0]**2 + r[2]**2)
        inertia_tensor[2, 2] += mass * (r[0]**2 + r[1]**2)
        inertia_tensor[0, 1] -= mass * r[0] * r[1]
        inertia_tensor[0, 2] -= mass * r[0] * r[2]
        inertia_tensor[1, 2] -= mass * r[1] * r[2]

    inertia_tensor[1, 0] = inertia_tensor[0, 1]
    inertia_tensor[2, 0] = inertia_tensor[0, 2]
    inertia_tensor[2, 1] = inertia_tensor[1, 2]

    return inertia_tensor

# Function to calculate total inertia tensor using parallel processing
def inertia_prisms(v0, v1, v2, v3, density):
    n = len(v0)
    all_points = np.stack([v0, v1, v2, v3], axis=1)  # Shape (n, 6, 3)

    # Function to process each polyhedron
    def process_polyhedron(i):
        points = all_points[i]
        volume = volume_from_convex_hull(points)
        mass = volume * density
        com = center_of_mass(points)
        inertia_tensor = moment_of_inertia(points, mass, com)
        #print(f"Tetra {i}: inertia_local =\n{inertia_tensor}")
        return volume, mass, com, inertia_tensor

    # Use ThreadPoolExecutor to parallelize the computation of volume, mass, com, and inertia tensor
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_polyhedron, range(n)))

    # Unpack the results
    volumes, masses, centers_of_mass, inertia_tensors = zip(*results)
    volumes = np.array(volumes)
    #print('vol inertia_prisms:', volumes)
    masses = np.array(masses)
    centers_of_mass = np.array(centers_of_mass)
    #print('COG_local inertia_prisms:', centers_of_mass)
    total_mass = np.sum(masses)

    # Compute total center of mass
    total_com = np.sum(masses[:, np.newaxis] * centers_of_mass, axis=0) / total_mass

    # Compute total inertia tensor, including parallel axis theorem correction
    total_inertia_tensor = np.zeros((3, 3))
    for i in range(n):
        # Inertia relative to global center of mass
        translation_vector = centers_of_mass[i] - total_com
        #print(f"Tetra {i}: d (COG_local - global_com) = {translation_vector}")
        translation_inertia = masses[i] * (
            np.dot(translation_vector, translation_vector) * np.eye(3) - np.outer(translation_vector, translation_vector)
        )
        total_inertia_tensor += inertia_tensors[i] + translation_inertia
        #print(f"Tetra {i}: inertia_global after translation =\n{inertia_tensors[i] + translation_inertia}")  

    return total_inertia_tensor, total_mass, total_com, volumes

class PATO():
    """ Class PATO
    
        A class to store the PATO simulation
    """

    def __init__(self, options, len_facets, bloom_config, object_id = 0, temperature = 300):

        self.initial_temperature = temperature
        
        self.temperature = np.empty(len_facets); self.temperature.fill(temperature)

        self.hf_cond = np.zeros(len_facets)

        #: [bool] Flag value indicating the use of PATO for the thermal model
        self.flag = bloom_config[0]

        Path(options.output_folder+'/PATO_'+str(object_id)+'/').mkdir(parents=True, exist_ok=True)
        Path(options.output_folder+'/PATO_'+str(object_id)+'/verification/').mkdir(parents=True, exist_ok=True)
        Path(options.output_folder+'/PATO_'+str(object_id)+'/verification/unstructured_gmsh/').mkdir(parents=True, exist_ok=True)
        Path(options.output_folder+'/PATO_'+str(object_id)+'/constant/').mkdir(parents=True, exist_ok=True)
        Path(options.output_folder+'/PATO_'+str(object_id)+'/constant/subMat1/').mkdir(parents=True, exist_ok=True)
        Path(options.output_folder+'/PATO_'+str(object_id)+'/origin.0/').mkdir(parents=True, exist_ok=True)
        Path(options.output_folder+'/PATO_'+str(object_id)+'/origin.0/subMat1').mkdir(parents=True, exist_ok=True)
        Path(options.output_folder+'/PATO_'+str(object_id)+'/system/').mkdir(parents=True, exist_ok=True)
        Path(options.output_folder+'/PATO_'+str(object_id)+'/system/subMat1').mkdir(parents=True, exist_ok=True)
        Path(options.output_folder+'/PATO_'+str(object_id)+'/qconv').mkdir(parents=True, exist_ok=True)
        Path(options.output_folder+'/PATO_'+str(object_id)+'/qconv-bkp').mkdir(parents=True, exist_ok=True)
        Path(options.output_folder+'/PATO_'+str(object_id)+'/mesh').mkdir(parents=True, exist_ok=True)
        Path(options.output_folder+'/PATO_'+str(object_id)+'/data').mkdir(parents=True, exist_ok=True)        

class bloom():
    def __init__(self, bloom_config):

        #: [bool] Flag value indicating the use of Bloom to generate the boundary layer mesh
        self.flag = bloom_config[0]

        #: [int] Number of Layers in the boundary layer
        self.layers = int(bloom_config[1])

        #: [float] Value of spacing of the first element in the boundary layer
        self.spacing = bloom_config[2]

        #: [float] Value of the growth rate, starting at the first element
        self.growth_rate = bloom_config[3]
