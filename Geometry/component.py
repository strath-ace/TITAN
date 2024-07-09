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
from Geometry.tetra import inertia_tetra, vol_tetra
from Material.material import Material
import numpy as np
from pathlib import Path

class Component_list():
    # A class with the purpose of storing the different components in a list
    def __init__(self):
        self.object = []
        self.id = 1

    def insert_component(self,filename, file_type, inner_stl = '', id = 0, binary = True, trigger_type = 'Indestructible', trigger_value = 0,fenics_bc_id = -1, material = 'Unittest', temperature = 300, options = None, global_ID = 0):
        
        self.object.append(Component(filename, file_type, inner_stl = inner_stl, id = self.id, 
                           binary = binary, temperature = temperature, trigger_type = trigger_type,
                           trigger_value = trigger_value, fenics_bc_id = fenics_bc_id, material = material, options = options, global_ID = global_ID))
        self.id += 1

class Component():
    """ Component class

        Class to store the information of a singular component.
    """
    
    def __init__(self,filename, file_type, inner_stl = '', id = 0, binary = True, temperature = 300,
                 trigger_type = 'Indestructible', trigger_value = 0, fenics_bc_id = -1, material = 'Unittest',
                 v0 = [], v1 = [], v2 = [], parent_id = None, parent_part = None, options = None, global_ID = 0):

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


        if options.thermal.ablation and options.thermal.ablation_mode.lower() == 'pato':      
            self.pato = PATO(options, len(mesh.facets), self.global_ID, self.temperature)        
            self.connectivity = np.array([], dtype = int)

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


class PATO():
    """ Class PATO
    
        A class to store the PATO simulation
    """

    def __init__(self, options, len_facets, object_id = 0, temperature = 300):

        self.initial_temperature = temperature
        
        self.temperature = np.empty(len_facets); self.temperature.fill(temperature)

        self.hf_cond = np.zeros(len_facets)

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
        Path(options.output_folder+'/PATO_'+str(object_id)+'/mesh').mkdir(parents=True, exist_ok=True)
        Path(options.output_folder+'/PATO_'+str(object_id)+'/data').mkdir(parents=True, exist_ok=True)        
