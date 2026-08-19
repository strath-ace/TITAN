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
"""component module."""
from ..Geometry import mesh as Mesh
from ..Geometry.tetra import inertia_tetra, vol_tetra
from ..Material.material import Material
import numpy as np
from pathlib import Path

class Component_list():
    """Component_list."""
    # A class with the purpose of storing the different components in a list
    def __init__(self):
        """__init__."""
        self.object = []
        self.id = 1
        
    def insert_component(self,filename, file_type, inner_stl = '', id = 0, binary = True, trigger_type = 'Indestructible', 
                         trigger_value = 0,fenics_bc_id = -1, material = 'Unittest', temperature = 300, options = None, 
                         global_ID = 0, bloom_config = [False, 0, 0, 0], alpha = 1.0, mixture=None,
                         mass_fractions=None, species=None, explosive_parameters=None):

        self.object.append(Component(filename, file_type, inner_stl = inner_stl, id = self.id, 
                           binary = binary, temperature = temperature, trigger_type = trigger_type,
                           trigger_value = trigger_value, fenics_bc_id = fenics_bc_id, material = material, 
                           options = options, global_ID = global_ID, bloom_config = bloom_config, 
                           alpha=alpha, mixture=mixture, mass_fractions=mass_fractions, species=species, explosive_parameters=explosive_parameters))
        self.id += 1

class Component():
    """Documentation for the function.
    :param filename: Path to the relevant file.
    :type filename: str
    :param file_type: Value for file type.
    :type file_type: str
    :param inner_stl: Value for inner stl.
    :type inner_stl: Any
    :param id: Integer value for id.
    :type id: int
    :param binary: Value for binary.
    :type binary: Any
    :param trigger_type: Value for trigger type.
    :type trigger_type: Any
    :param trigger_value: Value for trigger value.
    :type trigger_value: Any
    :param fenics_bc_id: Integer value for fenics bc id.
    :type fenics_bc_id: int
    :param material: Value for material.
    :type material: Any
    :param temperature: Numeric value for temperature.
    :type temperature: float
    :param options: Options or configuration object.
    :type options: object
    :param global_ID: Integer value for global id.
    :type global_ID: int
    :param bloom_config: Value for bloom config.
    :type bloom_config: Any
    :param alpha: Numeric value for alpha.
    :type alpha: float
    :param mixture: Value for mixture.
    :type mixture: Any
    :param mass_fractions: Value for mass fractions.
    :type mass_fractions: Any
    :param species: Value for species.
    :type species: Any
    :param explosive_parameters: Value for explosive parameters.
    :type explosive_parameters: Any"""
    """ Component class

        Class to store the information of a singular component.
    """
    
    def __init__(self,filename, file_type, inner_stl = '', id = 0, binary = True, temperature = 300,
                 trigger_type = 'Indestructible', trigger_value = 0, fenics_bc_id = -1, material = 'Unittest',
                 v0 = [], v1 = [], v2 = [], parent_id = None, parent_part = None, options = None, global_ID = 0, 
                 bloom_config = [False, 0, 0, 0], alpha = 1.0, mixture=None, mass_fractions=None, species=None, explosive_parameters=None):

        print("Generating Body: ", filename)
        
        #: [str] Name of the file where the mesh is stores
        self.name = filename

        #: [str] Type of the component (joint, primitive). Several sub-components can be used to form a larger component
        self.type = file_type
    
        #if self.type == "Joint":
        """Documentation for the function.
:param filename: Path to the relevant file.
:type filename: str
:param file_type: Value for file type.
:type file_type: str
:param inner_stl: Value for inner stl.
:type inner_stl: Any
:param id: Integer value for id.
:type id: int
:param binary: Value for binary.
:type binary: Any
:param temperature: Numeric value for temperature.
:type temperature: float
:param trigger_type: Value for trigger type.
:type trigger_type: Any
:param trigger_value: Value for trigger value.
:type trigger_value: Any
:param fenics_bc_id: Integer value for fenics bc id.
:type fenics_bc_id: int
:param material: Value for material.
:type material: Any
:param v0: Value for v0.
:type v0: Any
:param v1: Value for v1.
:type v1: Any
:param v2: Value for v2.
:type v2: Any
:param parent_id: Integer value for parent id.
:type parent_id: int
:param parent_part: Value for parent part.
:type parent_part: Any
:param options: Options or configuration object.
:type options: object
:param global_ID: Integer value for global id.
:type global_ID: int
:param bloom_config: Value for bloom config.
:type bloom_config: Any
:param alpha: Numeric value for alpha.
:type alpha: float
:param mixture: Value for mixture.
:type mixture: Any
:param mass_fractions: Value for mass fractions.
:type mass_fractions: Any
:param species: Value for species.
:type species: Any
:param explosive_parameters: Value for explosive parameters.
:type explosive_parameters: Any"""
        if self.type == 'Explosive':
            self.explosive = Explosive(explosive_parameters)
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

        self.density_ratio = 1


        ## For use with byproducts
        # Name of a mutation mixture
        self.mixture = mixture
        # Array of mass fractions for the object
        self.mass_fraction = mass_fractions
        # Names of species corresponding to the relevant mass fractions
        self.species = species
        self.debug_alpha = alpha
        self.volume = 0


    def compute_mass_properties(self, coords, elements, density):
        """Compute the inertia properties
:param coords: Value for coords.
:type coords: Any
:param elements: Value for elements.
:type elements: Any
:param density: Numeric value for density.
:type density: float"""

        vol = vol_tetra(coords[elements[:,0]],coords[elements[:,1]],coords[elements[:,2]], coords[elements[:,3]])
        
        mass = vol*density
        self.mass = np.sum(mass)
        self.volume = np.sum(vol)
        if self.mass <= 0:
            self.COG = np.array([0,0,0])
        else:
            self.COG = np.sum(0.25*(coords[elements[:,0]] + coords[elements[:,1]] + coords[elements[:,2]] + coords[elements[:,3]])*mass[:,None], axis = 0)/self.mass
        
        self.inertia = inertia_tetra(coords[elements[:,0]],coords[elements[:,1]],coords[elements[:,2]], coords[elements[:,3]], vol, self.COG, density)


class PATO():
    """ Class PATO
    
        A class to store the PATO simulation
    """

    def __init__(self, options, len_facets, bloom_config, object_id = 0, temperature = 300):
        """Documentation for the function.
:param options: Options or configuration object.
:type options: object
:param len_facets: Value for len facets.
:type len_facets: Any
:param bloom_config: Value for bloom config.
:type bloom_config: Any
:param object_id: Integer value for object id.
:type object_id: int
:param temperature: Numeric value for temperature.
:type temperature: float"""

        self.initial_temperature = temperature
        
        self.temperature = np.empty(len_facets); self.temperature.fill(temperature)

        self.hf_cond = np.zeros(len_facets)

        self.mDotVapor = np.zeros(len_facets)

        self.mDotMelt = np.zeros(len_facets)

        #: [bool] Flag value indicating the use of PATO for the thermal model
        self.flag = bloom_config[0]

        #: [float] Value of mass loss due to ablation
        self.mass_loss = 0        

        self.molten = np.zeros(len_facets)

        # NB: Tommy has not very much idea about Bprime at all so its possible this is nonsense
        # Enthalpy of recovery
        self.h_r = 2e5 * np.ones(len_facets)

        # Conductivity BC
        self.rhoeUeCH = 0.3 * np.ones(len_facets)

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
    """bloom."""
    def __init__(self, bloom_config):
        """Documentation for the function.
:param bloom_config: Value for bloom config.
:type bloom_config: Any"""

        #: [bool] Flag value indicating the use of Bloom to generate the boundary layer mesh
        self.flag = bloom_config[0]

        #: [int] Number of Layers in the boundary layer
        self.layers = int(bloom_config[1])

        #: [float] Value of spacing of the first element in the boundary layer
        self.spacing = bloom_config[2]

        #: [float] Value of the growth rate, starting at the first element
        self.growth_rate = bloom_config[3]

class Explosive():
    """Explosive."""
    def __init__(self, explosive_parameters=None):
        """Documentation for the function.
:param explosive_parameters: Value for explosive parameters.
:type explosive_parameters: Any"""


        defaults = [24, 10.0, 1e6, 1.0, 5e-3]
        if explosive_parameters is None: explosive_parameters = defaults
        else:
            for i_param in range(len(explosive_parameters)):
                if explosive_parameters[i_param] is None: 
                    explosive_parameters[i_param] = defaults[i_param]

        #: [int] Number of fragments to explode into
        self.n_fragments = int(explosive_parameters[0])

        #: [float] Characteristic velocity of explosion, overwritten by energy if using nasa_conservation
        self.char_velocity = float(explosive_parameters[1])

        #: [float] Explosive Energy (J)
        self.energy = float(explosive_parameters[2])

        #: [float] Amount of energy converted into in kinetic energy if using nasa_conservation
        self.kinetic_factor = float(explosive_parameters[3])

        #: [float] Wider fractures remove more of an objects initial volume (and mass) but make the fracture algorithm more robust
        self.crack_width = float(explosive_parameters[4])
