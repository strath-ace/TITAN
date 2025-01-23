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
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
try: # Joblib is a dropin replacement for pickle with better behaviour for large files
    import joblib as pickle
except:
    import pickle
import copy
from Geometry import component as Component
from Geometry import assembly as Assembly
from Dynamics import dynamics
from Dynamics import collision
from Output import output
from Model import planet, vehicle, drag_model
import pathlib
from Aerothermo import bloom
from Thermal import pato
from Geometry import gmsh_api as GMSH


class Collision_options():

    def __init__(self, post_fragmentation_iters = 0,
                 post_fragmentation_timestep = 0.0,
                 elastic_factor = 1.0,
                 max_depth = 1E-6,
                 mesh_factor = 1E-3):

        self.flag = False
        self.post_fragmentation_iters = post_fragmentation_iters
        self.post_fragmentation_timestep = post_fragmentation_timestep
        self.elastic_factor = elastic_factor
        self.max_depth = max_depth
        self.mesh_factor = mesh_factor

class Trajectory():
    """ Class Trajectory
    
        A class to store the user-defined trajectory information
    """

    def __init__(self, altitude = 0, gamma = 0, chi = 0, velocity = 0, 
              latitude = 0, longitude = 0):
        
        #: [meters] Altitude value.
        self.altitude = altitude

        #: [radians] Flight Path Angle value.
        self.gamma = gamma
        
        #: [radians] Heading Angle value.
        self.chi = chi  

        #: [meters/second] Velocity value.
        self.velocity = velocity

        #: [radians] Latitude value.
        self.latitude = latitude

        #: [radians] Longitude value.
        self.longitude = longitude

class Meshing():

    def __init__(self):
        self.far_size = 0.5
        self.surf_size = 0.5


#FENICS class
class Fenics():
    """ FEniCS class

        Class to store the user-defined information for the structural dynamics simulation using FEniCS
    """

    def __init__ (self, E = 68e9, FENICS = False, FE_MPI = False, FE_MPI_cores = 12, FE_verbose = False):

        #: [Pa] Young Modulus
        self.E = E

        #: [bool] Flag value indicating if MPI is to be used for FEniCS
        self.FE_MPI = FE_MPI # uses MPI for FE if True

        #: [int] Flag value indicating the number of cores if **FE_MPI=True**
        self.FE_MPI_cores = FE_MPI_cores # Number of MPI cores

        #: [bool] Flag value indicating the verbosity of FEniCS solver
        self.FE_verbose = FE_verbose # printing FE progress (for debugging)

        self.FE_freq = 1

class Dynamics():
    """ Dynamics class

        A class to store the user-defined dynamics options for the simulation
    """

    def __init__(self, time_step = 0, time = 0, propagator = 'EULER', adapt_propagator = False, manifold_correction = True):

        #: [seconds] Physical time of the simulation.
        self.time = time

        #: [seconds] Value of the time-step.
        self.time_step = time_step

        #: [str] Name of the propagator to be used in the dynamics (options - EULER).
        self.propagator = propagator

        #: [bool] Flag value indicating time-step adaptation
        self.adapt_propagator = adapt_propagator

        #: [bool] Flag value indicating manifold correction
        self.manifold_correction = manifold_correction

class CFD():
    def __init__(self, solver = 'NAVIER_STOKES', cfl = 0.5, iters= 1, muscl = 'NO', conv_method = 'AUSM', adapt_iter = 2, cores = 1, cfd_restart = False, restart_grid = 0, restart_iter = 0):
        
        #: [str] Name of the CFD solver
        self.solver = solver

        #: [float] Value of CFL
        self.cfl = cfl

        #: [int] Maximum number of CFD iterations
        self.iters = iters

        #: [bool] Flag value indicating MUSCL reconstruction
        self.muscl = muscl

        #: [str] Name of the convective scheme
        self.conv_method = conv_method

        #: [int] Number of mesh adaptations
        self.adapt_iter = adapt_iter

        #: [int] Number of cores to run the CFD simulation
        self.cores = cores

        #: [bool] Flag to restart from a CFD solution
        self.cfd_restart = cfd_restart

        #: [bool] Iteration of adaptated grid to be used at restart of a CFD simulation
        self.restart_grid = restart_grid

        #: [bool] TITAN iteration to be used for the restart of a CFD simulation
        self.restart_iter = restart_iter                

class Bloom():
    def __init__(self, flag = False, layers = 20, spacing = 0.0006, growth_rate = 1.075):

        #: [bool] Flag value indicating the use of Bloom to generate the boundary layer mesh
        self.flag = flag

        #: [int] Number of Layers in the boundary layer
        self.layers = layers

        #: [float] Value of spacing of the first element in the boundary layer
        self.spacing = spacing

        #: [float] Value of the growth rate, starting at the first element
        self.growth_rate = growth_rate

class Amg():
    def __init__(self, flag = False, p=4,c = 100000, hgrad = 1.6, sensor = 'Mach'):

        #: [bool] Flag value indicating the use of Bloom to adapt the mesh
        self.flag = flag

        #: [int] Value of the p-norm (options - 1, 2, 4).
        self.p = p

        #: [int] Value of the complex number for mesh adaptation:
        #: an higher complex number increases the size of the vertex
        self.c = c

        #: [float] Value of the hgrad
        self.hgrad = hgrad

        #: [str] Name of the sensor field to be used in the adaptation
        self.sensor = sensor


class Thermal():
    def __init__(self, ablation = False, ablation_mode = "0D", post_fragment_tetra_ablation = False):

        #: [boolean] Flag to perform ablation
        self.ablation = False

        #: [str] Ablation Model (0D, tetra)
        self.ablation_mode = "0D"

        self.post_fragment_tetra_ablation = False


class PATO():
    def __init__(self, flag = False, time_step = 0.1, n_cores = 6, pato_mode = 'qconv', fstrip = 1):


        #: [boolean] Flag to perform PATO simulation
        self.flag = False

        #: [boolean] Flag to perform PATO simulation
        self.time_step = time_step  

        #: [int] Number of cores to perform PATO simulation
        self.n_cores = n_cores    

        #: [int] String to define type of boundary condition used in the PATO simulation
        self.Ta_bc = pato_mode #'fixed', 'qconv' or 'ablation' 

        #Fraction of the melted material that is stripped away
        self.fstrip = fstrip


class Radiation():
    def __init__(self, particle_emissions = False,
                 spectral = False, spectral_freq = 10000, phi = 0, theta = 0, wavelengths = [0]):

        #: [boolean] Flag to compute particle emissions
        self.particle_emissions = particle_emissions

        #: [boolean] Flag to compute spectral emissions
        self.spectral = spectral  

        #: [int] Frequency to compute spectral emissions
        self.spectral_freq = spectral_freq                

        self.phi = phi
        self.theta = theta
        self.wavelengths = wavelengths             

class Aerothermo():
    """ Aerothermo class

        A class to store the user-defined aerothemo model options
    """

    def __init__(self, heat_model = 'vd', knc_pressure = 1E-3, knc_heatflux = 1E-3, knf = 100, mixture = "air5"):

        #: [str] Name of the heatflux model to be used
        self.heat_model = heat_model

        #: [float] Value of the continuum knudsen for the pressure computation
        self.knc_pressure = knc_pressure

        #: [float] Value of the continuum knudsen for the heatflux computation
        self.knc_heatflux = knc_heatflux

        #: [float] Value of the free-molecular knudsen
        self.knf = knf

        #: [str] Mixture file name
        self.mixture = mixture

class Freestream():
    """ Freestream class

        A class to store the user-defined freestream properties per time iteration
    """

    def __init__(self):

        #: [Pa] Value of the freestream Pressure
        self.pressure = None

        #: [float] Value of the freestream Mach number
        self.mach = None

        #: [float] Value of the freestream specific heat ratio
        self.gamma = None
        
        #: [float] Value of the freestream knudsen
        self.knudsen = None

        #: [float] Value of the freestream prandtl  
        self.prandtl = None

        #: [kelvin] Value of the freestream temperature
        self.Temperature = None

        #: [kg/m^3] Value of the freestream density
        self.rho = None

        #: [m/s] Value of the freestream velocity
        self.Velocity = None

        #self.cp = None

        #: [ ?? ] Value of the Gas constant
        self.R = None

        #: [ ?? ]
        self.mfp = None

        #: [ ?? ]
        self.omega = None

        #: [ ?? ]
        self.muEC = None

        #: [ ?? ]
        self.muSu = None 

        #self.kb = 1.38064852e-23  #Boltzmann constant: J/K

        #: [ ?? ]
        self.ninf = 0   # number density

        #: [ ?? ]
        self.percent_gas = 0

        #:[ ?? ]
        self.model = "NRLMSISE00"

        #: Selection of freestream calculation method (Mutationpp, default = Standard)
        self.method = "Standard"

class GRAM():
    def __init__(self):
        self.gramPath = ''
        self.spicePath = ''
        self.MinMaxFactor = 0.0
        self.ComputeMinMaxFactor  = 1
class Uncertainty():
    def __init__(self,qoi_filepath = 'QoI.pkl',yaml_path='UQ.yaml',flag = 0):
        # Objects as st
        self.objects = ''
        self.outputs = ''
        self.stls = []
        self.quantities = {}
        self.qoi_filepath = qoi_filepath
        self.qoi_flag = flag
        self.yaml_path = yaml_path
    
    def build_quantities(self, path):
        self.outputs = [name.strip() for name in self.outputs.split(',')] if not self.outputs=='demise_points' else ['Latitude','Longitude','Altitude']
        self.objects = [name.strip() for name in self.objects.split(',')]
        for i_obj, obj in enumerate(self.objects):
            self.stls.append(path+obj)
            self.objects[i_obj] = obj.rsplit( ".", 1 )[ 0 ]
            self.quantities[self.stls[i_obj]] = {output : [] for output in self.outputs}
        filepath = self.qoi_filepath
        if not os.path.exists(filepath):
            with open(filepath,'wb') as file: 
                pickle.dump(self.quantities, file)

            

class Options():
    """ Options class

        A class that keeps the information of the selected user-defined options for all the disciplinary
        areas and methods required to run the simulation
    """

    def __init__(self, iters = 1, time_step = 0.1, fidelity = 'Low',
                 SPARTA = False, SP_NUM = 1, sp_iters = 0, SPARTA_MPI_cores = 4, Opti_Flag = 'OFF',
                 fenics = False, FE_MPI = False, FE_MPI_cores = 12, FE_verbose = False,
                 case = 'benchmark', E = 68e9, output_folder = 'TITAN_sol', propagator = 'Euler', adapt_propagator=False,
                 assembly_rotation = [], manifold_correction = True, adapt_time_step = False, rerr_tol=1e-3, 
                 num_joints= 0, frame_for_writing = 'W', max_time_step=0.5, save_displacement = False, save_vonMises = False):

        #: [:class:`.Fenics`] Object of class Fenics
        self.fenics = Fenics(fenics)

        #: [:class:`.Dynamics`] Object of class Dynamics
        self.dynamics = Dynamics()
        self.thermal = Thermal()
        self.pato = PATO()
        self.radiation = Radiation()
        self.cfd = CFD()
        self.bloom = Bloom()
        self.amg = Amg()

        self.meshing = Meshing()

        self.collision = Collision_options()

        #: [:class:`.Aerothermo`] Object of class Aerothermo
        self.aerothermo = Aerothermo()

        #: [:class:`.Freestream`] Object of class Freestream
        self.freestream = Freestream()

        self.planet = planet.ModelPlanet("Earth")

        self.uncertainty = Uncertainty()

        self.vehicle = None
        
        #: [int] Number of dynamic iterations
        self.iters = iters

        #:[int] Frequency of generating a restart file [per number of iterations]
        self.save_freq = 500

        #:[int] Frequency of generating the output surface solution [per number of iterations]
        self.output_freq = 500        

        #: [int] Current iteration
        self.current_iter = 0
                
        self.output_folder = output_folder + '/'

        ### self.SPARTA = SPARTA
        ### self.SP_NUM = SP_NUM
        ### self.sp_iters = sp_iters
        ### self.SPARTA_MPI_cores = SPARTA_MPI_cores
        ### self.Opti_Flag = Opti_Flag

        ### self.save_displacement = save_displacement    
        ### self.save_vonMises = save_vonMises
        ### self.num_joints = num_joints

        #: [boolean] Flag to perform structural dynamics
        self.structural_dynamics = False

        #: [str] Fidelity of the aerothermo calculation (Low - Default, High, Hybrid)
        self.fidelity = fidelity

        self.assembly_path = ""

        
    def clean_up_folders(self):
        """
        Cleans the simulation output folder specified in the configuration file
        """

        if os.path.isdir(self.output_folder):
            shutil.rmtree(self.output_folder)

    def create_output_folders(self):
        """
        Creates the folder structure to save the soluions
        """

        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        Path(self.output_folder+'/Data/').mkdir(parents=True, exist_ok=True)
        Path(self.output_folder+'/Restart/').mkdir(parents=True, exist_ok=True)
        Path(self.output_folder+'/Surface_solution/').mkdir(parents=True, exist_ok=True)
        Path(self.output_folder+'/Volume/').mkdir(parents=True, exist_ok=True)
    
        if self.freestream.model.lower() == "gram": 
            Path(self.output_folder+'/GRAM/').mkdir(parents=True, exist_ok=True)

        if self.fidelity.lower() == 'multi' or self.fidelity.lower() == 'high':
            Path(self.output_folder+'/CFD_sol/').mkdir(parents=True, exist_ok=True)
            Path(self.output_folder+'/CFD_Grid/').mkdir(parents=True, exist_ok=True)
            if self.bloom.flag:
                Path(self.output_folder+'/CFD_Grid/Bloom/').mkdir(parents=True, exist_ok=True)
            if self.amg.flag:
                Path(self.output_folder+'/CFD_Grid/Amg/').mkdir(parents=True, exist_ok=True)

    def save_mesh(self,titan):
        outfile = open(self.output_folder + '/Restart/'+'Mesh.p','wb')
        is_saved = False
        recursion_limit = sys.getrecursionlimit()
        while not is_saved:
            try:
                pickle.dump(titan, outfile)
                is_saved=True
                print('Mesh saving succeeded at recursion limit: {}'.format(recursion_limit))
            except:
                print('Mesh saving failed at recursion limit: {}'.format(recursion_limit))
                is_saved=False
                recursion_limit=int(np.ceil(1.1*recursion_limit))
                sys.setrecursionlimit(recursion_limit)
            
        outfile.close() 

    def save_state(self, titan, i = 0, CFD = False):
        """
        Saves the TITAN object state

        Parameters
        ----------
        titan : Assembly_list
            Object of class Assebly_list
        i: int
            Iteration number.
        """

        #titan.high_fidelity_model = None 
        # titan.low_fidelity_model = None 
        
        print("Saving state")

        if self.collision.flag:
            for assembly in titan.assembly:
                assembly.collision = None

        outfile = open(self.output_folder + '/Restart/'+ 'Assembly_State.p','wb')
        pickle.dump(titan, outfile)
        outfile.close()
        outfile = open(self.output_folder + '/Restart/'+ 'Assembly_State_'+str(i)+'_.p','wb')
        pickle.dump(titan, outfile)
        outfile.close()

        if CFD:
            outfile = open(self.output_folder + '/Restart/'+ 'Assembly_State_CFD_'+str(i)+'.p','wb')
            pickle.dump(titan, outfile)
            outfile.close()

        if self.collision.flag:
            for assembly in titan.assembly: collision.generate_collision_mesh(assembly, self)
            collision.generate_collision_handler(titan, self)

    def read_mesh(self):
        infile = open(self.output_folder + '/Restart/'+ 'Mesh.p','rb')
        titan = pickle.load(infile)
        infile.close()
        return titan

    def read_state(self, i = 0):
        """
        Load last state of the TITAN object

        Returns
        -------
        titan: Assembly_list
            Object of class Assembly_list
        """

        if self.fidelity.lower() == 'high' and self.cfd.cfd_restart:
            infile = open(self.output_folder + '/Restart/'+ 'Assembly_State_CFD_'+str(i)+'.p','rb')
            titan = pickle.load(infile)
            infile.close()
        else: 
            infile = open(self.output_folder + '/Restart/'+ 'Assembly_State.p','rb')
            titan = pickle.load(infile)
            infile.close()

        return titan


def get_config_value(configParser, variable, section, field, var_type, list_type = None):
    
    if configParser.has_option(section, field):

        if var_type == 'boolean' or var_type == 'bool':
            try:        
                variable = configParser.getboolean(section, field)
            except ValueError:
                print(f"Error reading the value of field {field} in section {section}. Returning to default values!")
                pass

        elif var_type == 'int':
            try:        
                variable = configParser.getint(section, field)
            except ValueError:
                print(f"Error reading the value of field {field} in section {section}. Returning to default values!")
                pass

        elif var_type == 'float':
            try:        
                variable = configParser.getfloat(section, field)
            except ValueError:
                print(f"Error reading the value of field {field} in section {section}. Returning to default values!")
                pass

        elif var_type == 'str' or var_type == 'string':
            try:        
                variable = configParser.get(section, field)
            except ValueError:
                print(f"Error reading the value of field {field} in section {section}. Returning to default values!")
                pass

        if var_type == 'custom':
            try:        
                if list_type == 'angle': variable = check_angle(configParser.get(section, field))
                if list_type == 'fidelity': variable = check_fidelity(configParser.get(section, field))
                if list_type == 'connectivity': variable = check_connectivity(configParser.get(section, field))
                if list_type == 'wavelengths': variable = check_wavelengths(configParser.get(section, field))
                if list_type == 'initial_condition':
                    ids, variable = check_initial_condition_array(configParser.get(section, field))
                    return ids, variable
            except ValueError:
                print(f"Error reading the value of field {field} in section {section}. Returning to default values!")
                pass

    return variable

def check_angle(keywords):

    angle = float(keywords)
    angle *=np.pi/180.0

    return angle

def check_fidelity(fidelity):
    return fidelity


def check_connectivity(connectivity):

    connectivity = connectivity.replace('[','').replace(']','').replace(' ','').split(',')
    connectivity = [int(i) for i in connectivity]
    connectivity = np.array(connectivity)
    connectivity.shape = (-1,3)

    return connectivity

def check_wavelengths(wavelengths):

    wavelengths = wavelengths.replace('[','').replace(']','').replace(' ','').split(',')
    wavelengths = [float(i) for i in wavelengths]
    wavelengths = np.array(wavelengths)

    return wavelengths

def check_initial_condition_array(initial_condition):
    
    ids = []
    condition = []

    array_cond = initial_condition.replace(' ', '').replace('(', '').replace(')', '').split(';')
    
    for cond in array_cond:
        a = cond.replace('[','').replace(']','').split(':')
        ids.append(int(a[0]))

        condition.append([float(i) for i in a[1].split(',')])

    return np.array(ids), np.array(condition)

def read_trajectory(configParser):
    """
    Read the Trajectory specified in the config file

    Parameters
    ----------
    configParser: configParser
        Object of Config Parser

    Returns
    -------
    trajectory: Trajectory
        Object of class Trajectory
    """

    trajectory = Trajectory()

    trajectory.altitude = get_config_value(configParser, trajectory.altitude, 'Trajectory', 'Altitude', 'float')
    trajectory.velocity = get_config_value(configParser, trajectory.velocity, 'Trajectory', 'Velocity', 'float')
    trajectory.gamma =    get_config_value(configParser, trajectory.gamma, 'Trajectory', 'Flight_path_angle', 'custom', 'angle')
    trajectory.chi =      get_config_value(configParser, trajectory.chi, 'Trajectory', 'Heading_angle', 'custom', 'angle')
    trajectory.latitude = get_config_value(configParser, trajectory.latitude, 'Trajectory', 'Latitude', 'custom', 'angle')
    trajectory.longitude =get_config_value(configParser, trajectory.longitude, 'Trajectory', 'Longitude', 'custom', 'angle')
    
    return trajectory

def read_geometry(configParser, options):
    """
    Geometry pre-processing

    Reads the specified configuration file and creates a list with the information of the objects and assemblies

    Parameters
    ----------
    configParser: configParser
        Object of Config Parser

    Returns
    -------
    titan: Assembly_list
        Object of class Assembly_list
    """

    #Reads the path to the geometrical files
    path = get_config_value(configParser, '', 'Assembly', 'Path', 'str')
    options.assembly_path = path

    #Initialization of the object of class Component_list to store the user-defined compoents
    objects = Component.Component_list()

    #Loops through the user-defined components, checks if they are either Primitives or Joints 
    #and creates the object according to the specified parameters in the config file
    obj_global_ID = 0

    for section in configParser.sections():
        if section == 'Objects':
            for name, value in configParser.items(section):
                value = value.replace('[','').replace(']','').replace(' ','').split(",")
                object_type = [s for s in value if "type=" in s.lower()][0].split("=")[1]

                if object_type == 'Primitive':
                    object_path = path+[s for s in value if "name=" in s.lower()][0].split("=")[1]
                    material= [s for s in value if "material=" in s.lower()][0].split("=")[1]
                    

                    try:
                        inner_stl_file = [s for s in value if "inner_stl=" in s.lower()][0].split("=")[1]
                    except:
                        inner_stl_file = 'none'

                    if inner_stl_file != 'None' and inner_stl_file != 'none':
                        inner_path = path+inner_stl_file
                    else:
                        inner_path = ''

                    try:
                        trigger_type = [s for s in value if "trigger_type=" in s.lower()][0].split("=")[1]
                        trigger_value = [s for s in value if "trigger_value=" in s.lower()][0].split("=")[1]
                    except:
                        trigger_type = ""
                        trigger_value = 0

                    try:
                        fenics_bc_id = [s for s in value if "fenics_id=" in s.lower()][0].split("=")[1]
                    except:
                        fenics_bc_id = None

                    try:
                        temperature = float([s for s in value if "temperature=" in s.lower()][0].split("=")[1])
                    except:
                        temperature = 300

                    bloom = [False, 0, 0, 0]
                    try:                        
                        for s in value:
                            if 'bloom' in s.lower():
                                bloom = s.split('=')[1].strip('()').split(';')  
                                bloom = [eval(bloom[0]), float(bloom[1]), float(bloom[2]), float(bloom[3])]       
                    except:
                        bloom = [False, 0, 0, 0]               
                    
                    objects.insert_component(filename = object_path, file_type = object_type, trigger_type = trigger_type, trigger_value = float(trigger_value), 
                        fenics_bc_id = fenics_bc_id, inner_stl = inner_path, material = material, temperature = temperature, options = options, global_ID = obj_global_ID, bloom_config = bloom)

                if object_type == 'Joint':
                    object_path = path+[s for s in value if "name=" in s.lower()][0].split("=")[1]
                    material= [s for s in value if "material=" in s.lower()][0].split("=")[1]

                    try:
                        inner_stl_file = [s for s in value if "inner_stl=" in s.lower()][0].split("=")[1]
                    except:
                        inner_stl_file = 'none'

                    if inner_stl_file != 'None' and inner_stl_file != 'none':
                        inner_path = path+inner_stl_file
                    else:
                        inner_path = ''
                    try:
                        trigger_type = [s for s in value if "trigger_type=" in s.lower()][0].split("=")[1]
                        trigger_value = [s for s in value if "trigger_value=" in s.lower()][0].split("=")[1]
                    except:
                        trigger_type = ""
                        trigger_value = 0

                    try:
                        fenics_bc_id = [s for s in value if "fenics_id=" in s.lower()][0].split("=")[1]
                    except:
                        fenics_bc_id = None

                    try:
                        temperature = float([s for s in value if "temperature=" in s.lower()][0].split("=")[1])
                    except:
                        temperature = 300   

                    bloom = [False, 0, 0, 0]

                    try:                        
                        for s in value:
                            if 'bloom' in s.lower():
                                bloom = s.split('=')[1].strip('()').split(';')  
                                bloom = [eval(bloom[0]), float(bloom[1]), float(bloom[2]), float(bloom[3])]              
                    except:
                        bloom = [False, 0, 0, 0]              

                    objects.insert_component(filename = object_path, file_type = object_type, inner_stl = inner_path,
                                             trigger_type = trigger_type, trigger_value = float(trigger_value), fenics_bc_id = fenics_bc_id, material = material, temperature = temperature, options = options, global_ID = obj_global_ID, bloom_config = bloom) 

                obj_global_ID += 1


    # Creates a list of the different assemblies, where each assembly is a object with several linked components
    titan = Assembly.Assembly_list(objects.object)
    connectivity = get_config_value(configParser, np.array([]), 'Assembly', 'Connectivity', 'custom', 'connectivity')

    # Stores the information regarding the initial attitude of the assemblies
    aoa = get_config_value(configParser, 0.0, 'Assembly', 'Angle_of_attack', 'custom','angle')
    slip = get_config_value(configParser, 0.0, 'Assembly', 'Sideslip', 'custom','angle')
    roll = get_config_value(configParser, 0.0, 'Assembly', 'Roll', 'custom','angle')

    titan.create_assembly(connectivity = connectivity, aoa = aoa, slip = slip, roll = roll, options = options)

    return titan

def read_initial_conditions(titan, options, configParser):

    ids, variable = get_config_value(configParser, (None,np.array([0,0,0])), 'Initial Conditions', 'Angular Velocity', 'custom', 'initial_condition')
    if ids != None:
        for i,value in zip(ids,variable):
            titan.assembly[i-1].roll_vel  = value[0]*np.pi/180.0
            titan.assembly[i-1].pitch_vel = value[1]*np.pi/180.0
            titan.assembly[i-1].yaw_vel   = value[2]*np.pi/180.0
    return


def read_config_file(configParser, postprocess = "", emissions = ""):
    """
    Read the config file

    Parameters
    ----------
    configParser: configParser
        Object of Config Parser
    postprocess: str
        Postprocess method of the solution. If not None, only returns output_folder

    Returns
    -------
    options: Options
        Object of class Options
    titan: Assembly_list
        List of objects of class Assembly_list

    """

    options = Options()
    
    #Read Options Conditions
    options.output_folder = get_config_value(configParser, options.output_folder, 'Options', 'output_folder', 'str')
    options.output_freq     = get_config_value(configParser, options.output_freq, 'Options', 'Output_freq', 'int')
    if postprocess: return options, None

    options.iters         = get_config_value(configParser, options.iters, 'Options', 'Num_iters', 'int')
    options.save_freq     = get_config_value(configParser, options.save_freq, 'Options', 'Save_freq', 'int')
    options.load_state    = get_config_value(configParser, False, 'Options', 'Load_state', 'boolean')
    options.load_mesh     = get_config_value(configParser, False, 'Options', 'Load_mesh', 'boolean')
    options.fidelity      = get_config_value(configParser, options.fidelity, 'Options', 'Fidelity', 'custom', 'fidelity')
    #options.SPARTA =       get_config_value(configParser, options.SPARTA, 'Options', 'SPARTA', 'boolean')
    options.structural_dynamics  = get_config_value(configParser, False, 'Options', 'Structural_dynamics', 'boolean')

    options.collision.flag = get_config_value(configParser, False, 'Options', 'Collision', 'boolean')
    options.material_file  = get_config_value(configParser, 'database_material.xml', 'Options', 'Material_file', 'str')
    options.time_counter   = 0

    options.write_solutions     = get_config_value(configParser, True, 'Options', 'Write_solutions', 'boolean')

    #Read FENICS options
    if options.structural_dynamics:
        options.fenics.E            = get_config_value(configParser, options.fenics.E, 'FENICS', 'E', 'float')
        options.fenics.FE_MPI       = get_config_value(configParser, options.fenics.FE_MPI, 'FENICS', 'FENICS_MPI', 'bool')
        options.fenics.FE_MPI_cores = get_config_value(configParser, options.fenics.FE_MPI_cores, 'FENICS', 'FENICS_cores', 'int')
        options.fenics.FE_verbose   = get_config_value(configParser, options.fenics.FE_verbose, 'FENICS', 'FENICS_verbose', 'boolean')
        options.fenics.FE_freq      = get_config_value(configParser, options.fenics.FE_freq, 'FENICS', 'FENICS_freq', 'int')

    #Read Dynamics options
    options.dynamics.time = 0
    options.dynamics.time_step           = get_config_value(configParser, options.dynamics.time_step, 'Time', 'Time_step', 'float')
    options.dynamics.use_bwd_diff        = get_config_value(configParser, False, 'Time', 'Backward_difference', 'boolean')
    #options.dynamics.propagator          = get_config_value(configParser, options.dynamics.propagator, 'Time', 'Propagator', 'str')
    #options.dynamics.adapt_propagator    = get_config_value(configParser, options.dynamics.adapt_propagator, 'Time', 'Adapt_propagator', 'boolean')
    #options.dynamics.manifold_correction = get_config_value(configParser, options.dynamics.manifold_correction, 'Time', 'Manifold_correction', 'boolean')

    #Read Thermal options
    options.thermal.ablation       = get_config_value(configParser, False, 'Thermal', 'Ablation', 'boolean')
    if options.thermal.ablation:
        options.thermal.ablation_mode  = get_config_value(configParser, "0D",  'Thermal', 'Ablation_mode', 'str').lower()
        
        if (options.thermal.ablation_mode == "pato"):
            options.pato.flag = True
            options.pato.Ta_bc  = get_config_value(configParser, options.pato.Ta_bc,  'PATO', 'PATO_mode', 'str').lower()
            #PATO and TITAN time-step need to be the same for the consistency of the heat conduction and density change algorithm
            options.pato.time_step = options.dynamics.time_step#get_config_value(configParser, 0.1, 'PATO', 'Time_step', 'float')
            options.pato.n_cores = get_config_value(configParser, 6, 'PATO', 'N_cores', 'int')
            options.pato.fstrip = get_config_value(configParser, options.pato.fstrip, 'PATO', 'fStrip', 'float')
            if options.pato.n_cores < 2: print('Error: PATO run on 2 cores minimum.'); exit()
            options.radiation.particle_emissions  = get_config_value(configParser, False,  'Radiation', 'Particle_emissions', 'boolean')
            
            options.pato.solution_type = get_config_value(configParser, 'surface', 'PATO', 'Solution_type', 'str').lower()
            
        options.radiation.spectral               = get_config_value(configParser, options.radiation.spectral, 'Radiation', 'Spectral_emissions', 'boolean')

        if(options.radiation.spectral):
            options.radiation.spectral_freq      = get_config_value(configParser, options.radiation.spectral_freq, 'Radiation', 'Frequency', 'int')
            options.radiation.particle_emissions = get_config_value(configParser, options.radiation.particle_emissions, 'Radiation', 'Particle_emissions', 'boolean')
            options.radiation.phi                = get_config_value(configParser, options.radiation.phi, 'Radiation', 'Phi', 'custom', 'angle')
            options.radiation.theta              = get_config_value(configParser, options.radiation.theta, 'Radiation', 'Theta', 'custom', 'angle')
            options.radiation.wavelengths        = get_config_value(configParser, options.radiation.wavelengths, 'Radiation', 'Wavelengths', 'custom', 'wavelengths')

    if emissions: return options, None

    #Read Low-fidelity aerothermo options
    options.aerothermo.heat_model = get_config_value(configParser, options.aerothermo.heat_model, 'Aerothermo', 'Heat_model', 'str')
    options.aerothermo.vel_grad   = get_config_value(configParser, 'fr', 'Aerothermo', 'Vel_grad', 'str')
    options.aerothermo.standoff   = get_config_value(configParser, 'freeman', 'Aerothermo', 'Standoff', 'str')
    options.aerothermo.cat_method = get_config_value(configParser, 'constant', 'Aerothermo', 'Catalicity_method', 'str')
    options.aerothermo.cat_rate   = get_config_value(configParser, 1.0, 'Aerothermo', 'Catalicity_rate', 'float')
    options.aerothermo.subdivision_triangle = get_config_value(configParser, 0, 'Aerothermo', 'Level_division', 'int')
    options.aerothermo.mixture = get_config_value(configParser, options.aerothermo.mixture, 'Aerothermo', 'Mixture', 'str')

    #Read meshing options
    options.meshing.far_size  = get_config_value(configParser, 0.5, 'Mesh', 'Far_size', 'float')
    options.meshing.surf_size = get_config_value(configParser, 100, 'Mesh', 'Surf_size', 'float')

    #Read Freestream options
    options.freestream.model =  get_config_value(configParser, options.freestream.model, 'Freestream', 'Model', 'str')
    options.freestream.method =  get_config_value(configParser, options.freestream.method, 'Freestream', 'Method', 'str')

    if options.freestream.model.lower() == "gram":
        options.gram = GRAM()
        options.gram.gramPath = get_config_value(configParser, options.gram.MinMaxFactor, 'GRAM', 'GRAM_Path', 'str')
        options.gram.spicePath = get_config_value(configParser, options.gram.spicePath, 'GRAM', 'SPICE_Path', 'str') 
        options.gram.MinMaxFactor = get_config_value(configParser, options.gram.MinMaxFactor, 'GRAM', 'MinMaxFactor', 'str')
        options.gram.ComputeMinMaxFactor = get_config_value(configParser, options.gram.ComputeMinMaxFactor, 'GRAM', 'ComputeMinMaxFactor', 'str')
        options.gram.Uncertain = get_config_value(configParser, False, 'GRAM','Uncertain', 'boolean')
        options.gram.Seed = get_config_value(configParser, 'Auto', 'GRAM','Seed', 'str')
        options.gram.reference = get_config_value(configParser, False, 'GRAM','Use_reference_traj', 'boolean')
        options.gram.wind = get_config_value(configParser, False, 'GRAM','Use_wind', 'boolean')
        if os.path.exists(options.output_folder+'/GRAM/gramSpecies.pkl'):
            pathlib.Path(options.output_folder+'/GRAM/gramSpecies.pkl').unlink()
        if os.path.exists(options.output_folder+'/GRAM/gramWind.pkl'):
            pathlib.Path(options.output_folder+'/GRAM/gramWind.pkl').unlink()

    #Read Planet
    options.planet = planet.ModelPlanet(get_config_value(configParser, "Earth", 'Model', 'Planet', 'str'))

    #Read Vehicle
    vehicleFlag = get_config_value(configParser, False, 'Model', 'Vehicle', 'boolean')
    if vehicleFlag:
        options.vehicle = vehicle.ModelVehicle()
        options.vehicle.mass = get_config_value(configParser, 0, 'Vehicle', 'Mass', 'float')
        options.vehicle.Aref = get_config_value(configParser, 0, 'Vehicle', 'Area_reference', 'float')
        options.vehicle.noseRadius = get_config_value(configParser, 0, 'Vehicle', 'Nose_radius', 'float')

    #Reads a optional Drag model for the vehicle
    dragFlag = get_config_value(configParser, False, 'Model', 'Drag_model', 'boolean')
    if dragFlag:
        dragFile = get_config_value(configParser,"", 'Vehicle', 'Drag_file', 'str')
        if not dragFile:
            raise Exception("A drag file needs to be specified")
        else:
            options.vehicle.Cd = drag_model.read_csv(dragFile)

    if options.fidelity:

        #Read DSMC conditions
        #options.sparta.obj_path = get_config_value(configParser, options.sparta.obj_path, 'SPARTA', 'Obj_path', 'str')
        #options.sparta.surf_name = get_config_value(configParser, options.sparta.surf_name, 'SPARTA', 'Surf_name', 'str')
        #options.SP_NUM = get_config_value(configParser, options.SP_NUM, 'SPARTA', 'NUM_Iters', 'int')
        #options.sparta.lref = get_config_value(configParser, options.sparta.lref, 'SPARTA', 'LREF', 'float')
        #options.sparta.ppc =      get_config_value(configParser, options.sparta.ppc, 'SPARTA', 'PPC', 'int')
        #options.sparta.acc =      get_config_value(configParser, options.sparta.acc, 'SPARTA', 'acc', 'float')
        #options.sparta.cores =    get_config_value(configParser, options.sparta.cores, 'SPARTA', 'Num_cores', 'int')
        #options.sparta.run_steady = get_config_value(configParser, options.sparta.run_steady, 'SPARTA', 'RUN_Steady', 'int')
        #options.sparta.run_sample = get_config_value(configParser, options.sparta.run_sample, 'SPARTA', 'RUN_Sample', 'int')
        #options.sparta.Nadapt = get_config_value(configParser, options.sparta.Nadapt, 'SPARTA', 'Nadapt', 'int')
        #options.sparta.Nevery = get_config_value(configParser, options.sparta.Nevery, 'SPARTA', 'Nevery', 'int')
        #options.sparta.Nrepeat = get_config_value(configParser, options.sparta.Nrepeat, 'SPARTA', 'Nrepeat', 'int')

        #Read SU2 conditions
        options.cfd.solver =      get_config_value(configParser, options.cfd.solver, 'SU2', 'Solver', 'str')
        options.cfd.conv_method = get_config_value(configParser, options.cfd.conv_method, 'SU2', 'Conv_method', 'str')
        options.cfd.adapt_iter =  get_config_value(configParser, options.cfd.adapt_iter, 'SU2', 'Adapt_iter', 'int')
        options.cfd.cores =       get_config_value(configParser, options.cfd.cores, 'SU2', 'Num_cores', 'int')
        options.cfd.iters =       get_config_value(configParser, options.cfd.iters, 'SU2', 'Num_iters', 'int')
        options.cfd.muscl =       get_config_value(configParser, options.cfd.muscl, 'SU2', 'Muscl', 'str')
        options.cfd.cfl =         get_config_value(configParser, options.cfd.cfl, 'SU2', 'Cfl', 'float')
        options.cfd.cfd_restart =     get_config_value(configParser, options.cfd.cfd_restart, 'SU2', 'Restart', 'boolean')
        options.cfd.restart_grid=     get_config_value(configParser, options.cfd.restart_grid, 'SU2', 'Restart_grid', 'int')
        options.cfd.restart_iter=     get_config_value(configParser, options.cfd.restart_iter, 'SU2', 'TITAN_iter', 'int')        

        #Read Bloom conditions
        options.bloom.flag =        get_config_value(configParser,options.bloom.flag,'Bloom', 'Flag', 'boolean')
        options.bloom.layers =      get_config_value(configParser,options.bloom.layers,'Bloom', 'Layers', 'int')
        options.bloom.spacing =     get_config_value(configParser,options.bloom.spacing,'Bloom', 'Spacing', 'float')
        options.bloom.growth_rate = get_config_value(configParser,options.bloom.growth_rate,'Bloom', 'Growth_Rate', 'float')

        #Read AMG conditions
        options.amg.flag = get_config_value(configParser,options.amg.flag,'AMG', 'Flag', 'boolean')
        options.amg.p = get_config_value(configParser,options.amg.p,'AMG', 'P', 'int')
        options.amg.c = get_config_value(configParser,options.amg.c,'AMG', 'C', 'int')
        options.amg.sensor = get_config_value(configParser,options.amg.sensor,'AMG', 'Sensor', 'str')

    if options.collision.flag:
        options.collision.post_fragmentation_iters = get_config_value(configParser, options.collision.post_fragmentation_iters, 'Collision', 'Post_fragmentation_iters', 'int')
        options.collision.post_fragmentation_timestep = get_config_value(configParser, options.collision.post_fragmentation_timestep, 'Collision', 'Post_fragmentation_timestep', 'float')
        options.collision.max_depth = get_config_value(configParser, options.collision.max_depth, 'Collision', 'Max_depth', 'float')
        options.collision.mesh_factor = get_config_value(configParser, options.collision.mesh_factor, 'Collision', 'Mesh_factor', 'float')
        options.collision.elastic_factor = get_config_value(configParser, options.collision.elastic_factor, 'Collision', 'Elastic_factor', 'float')

    output.options_information(options)

    if options.load_mesh != True and options.load_state != True:
        options.clean_up_folders()

    options.create_output_folders()

    # Quantities of interest for uncertainty propagation
    if configParser.has_section('Uncertainty'):
        options.uncertainty.yaml_path = get_config_value(configParser, options.uncertainty.yaml_path, 'Uncertainty', 'Yaml_path', 'str')
        options.uncertainty.qoi_flag = get_config_value(configParser, False, 'Uncertainty', 'Extract_QoI', 'boolean')
        if options.uncertainty.qoi_flag:
            options.uncertainty.objects = get_config_value(configParser, options.uncertainty.objects, 'Uncertainty', 'Objects', 'str')
            options.uncertainty.outputs = get_config_value(configParser, options.uncertainty.outputs, 'Uncertainty', 'Outputs', 'str')
            options.uncertainty.qoi_filepath = options.output_folder +'/Data/'+ options.uncertainty.qoi_filepath
            options.uncertainty.build_quantities(get_config_value(configParser, '', 'Assembly', 'Path', 'str'))

    options.wrap_propagator = get_config_value(configParser,False,'Time','Wrap_propagator','boolean')

    if options.wrap_propagator:
        options.uncertainty.ut_DoF = get_config_value(configParser,6,'Uncertainty','UT_DoF','int')
        options.uncertainty.cov_UT = get_config_value(configParser,False,'Uncertainty','UT_use_covariance','boolean')
        from Uncertainty.UT import setupUT
        options=setupUT(options)
        
    
    if options.load_state:
        titan = options.read_state()

    else:
        #Read the initial trajectory details
        trajectory = read_trajectory(configParser)

        if options.load_mesh:
            titan = options.read_mesh()

        else:
            #Reads the user-defined geometries, properties and connectivity
            #to generate an assembly. The information is stored in the titan object
            titan = read_geometry(configParser, options)
            #Generate the volume mesh and compute the inertial properties
            for assembly in titan.assembly:
                assembly.generate_inner_domain(write = options.pato.flag, output_folder = options.output_folder)
                assembly.compute_mass_properties()
                if options.pato.flag:
                    for obj in assembly.objects:
                        if obj.pato.flag:
                            GMSH.generate_PATO_domain(obj, output_folder = options.output_folder)                          
                            bloom.generate_PATO_mesh(options, obj.global_ID, bloom = obj.bloom)
                            pato.initialize(options, obj)

                    #for each object, define connectivity to connected objects for heat conduction between objects
                    pato.identify_object_connections(assembly)
            options.save_mesh(titan)

        #Computes the quaternion and cartesian for the initial position
        for assembly in titan.assembly:
            assembly.trajectory = copy.deepcopy(trajectory)
            dynamics.compute_quaternion(assembly)
            dynamics.compute_cartesian(assembly, options)

            if options.wrap_propagator:
                from Uncertainty.UT import setupAssembly
                setupAssembly(assembly,options)
            
        #Reads the Initial pitch/yaw/roll 
        read_initial_conditions(titan, options, configParser)
        
        options.save_state(titan)
        output.generate_volume(titan = titan, options = options)

    if options.collision.flag:
        for assembly in titan.assembly: collision.generate_collision_mesh(assembly, options)
        collision.generate_collision_handler(titan, options)

    ### if options.FENICS:
    ###     fenics = TITAN.FENICS()
    ### else:
    ###     fenics = None

    return options, titan