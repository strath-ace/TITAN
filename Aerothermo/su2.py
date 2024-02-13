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
from Geometry import gmsh as GMSH
from Geometry import assembly
from Geometry import mesh
from Dynamics import frames
from Aerothermo import bloom, amg
from copy import deepcopy
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import subprocess
import os

class Solver():
    """ Class Solver

        A class to store the solver information.
        The class in the su2.py file is hardcoded to work with SU2.
    """

    def __init__(self, restart, su2, freestream):

        #: [str] Solver (EULER, NAVIER-STOKES, NEMO_EULER, NEMO_NAVIER_STOKES)
        self.solver = 'SOLVER = '+ su2.solver

        #: [str] Turbulence Model (Default = NONE)
        self.kind_turb_model = "KIND_TURB_MODEL = NONE"

        #: [str] Restart boolean (if YES, the CFD simulation will restart from previous solution)
        self.restart = "RESTART_SOL = "
        if restart: self.restart += "YES"
        else: self.restart += "NO"
        self.read_binary = "READ_BINARY_RESTART = NO"

        if su2.solver == 'NEMO_NAVIER_STOKES' or su2.solver == 'NEMO_EULER':

            #: [str] Fluid Model (Fluid model = MUTATIONPP -> Uses the Mutationpp Library)
            self.fluid_model = 'FLUID_MODEL= MUTATIONPP'

            #: [str] Gas Model (Gas to be used in the simulation)
            self.gas_model = 'GAS_MODEL= air_5'

            #Hardcoded for the NRLMSISE00 database

            N  = str(np.round(abs(np.sum([mass for mass, index in zip(freestream.percent_mass[0], freestream.species_index) if index in ['N']])),5))
            O  = str(np.round(abs(np.sum([mass for mass, index in zip(freestream.percent_mass[0], freestream.species_index) if index in ['O']])),5))
            NO = str(np.round(abs(np.sum([mass for mass, index in zip(freestream.percent_mass[0], freestream.species_index) if index in ['NO']])),5))

            N2= str(np.round(abs(np.sum([mass for mass, index in zip(freestream.percent_mass[0], freestream.species_index) if index in ['N2','He','Ar','H']])),5))
            O2= str(np.round(abs(np.sum([mass for mass, index in zip(freestream.percent_mass[0], freestream.species_index) if index in ['O2']])),5))
            
            #: [str] Gas Composition
            self.gas_composition = 'GAS_COMPOSITION= (' + N + ','  + O + ',' + NO + ',' + N2 + ',' + O2 + ')'

            #: [str] Transport Coefficient
            self.transport_coeff = 'TRANSPORT_COEFF_MODEL = CHAPMANN-ENSKOG'

class Solver_Freestream_Conditions():
    """ Class Solver Freestream Conditions

        A class to store the freestream conditions used in the CFD simulation
        The class in the su2.py file is hardcoded to work with SU2.
    """

    def __init__(self,freestream):
        #: [str] Initialization option to be used to compute the freestream (Default = TD_CONDITIONS)
        self.init_option = "INIT_OPTION = TD_CONDITIONS"

        #: [str] Mach number
        self.mach = "MACH_NUMBER = " + str(freestream.mach)
        
        #: [str] Freestream Temperature in K
        self.temperature = "FREESTREAM_TEMPERATURE = " + str(freestream.temperature)

        #: [str] Freestream Pressure in Pa
        self.pressure = "FREESTREAM_PRESSURE = " + str(freestream.pressure)

        #: [str] Angle of attack in deg
        self.aoa = "AOA= 180"

class Solver_Reference_Value():
    """ Class Solver Reference value

        A class to store the reference values for the coefficient and moment computation
        The class in the su2.py file is hardcoded to work with SU2.
    """

    def __init__(self):
        #: [str] x-coordinate to which the moment is computed
        self.origin_moment_x = "REF_ORIGIN_MOMENT_X = 0.0"

        #: [str] y-coordinate to which the moment is computed
        self.origin_moment_y = "REF_ORIGIN_MOMENT_Y = 0.0"
        
        #: [str] z-coordinate to which the moment is computed
        self.origin_moment_z = "REF_ORIGIN_MOMENT_Z = 0.0"

        #: [str] Length of reference
        self.ref_length = "REF_LENGTH = 1.0"

        #: [str] Area of reference
        self.ref_area = "REF_AREA = 1.0"

class Solver_BC():
    """ Class Solver Boundary conditions

        A class to store the applied boundary conditions
        The class in the su2.py file is hardcoded to work with SU2.
    """

    def __init__(self, assembly, su2):

        #: [str] Farfield Marker
        self.farfield = "MARKER_FAR = (Farfield)"

        if su2.solver == 'EULER' or su2.solver == 'NEMO_EULER':
            #: [str] Euler Marker
            self.euler = "MARKER_EULER = ("
            for i in range(1,len(assembly)+1):
                self.euler += "Body_"+ str(i) 
                if i != len(assembly): self.euler +=","
            self.euler += ")"

        if su2.solver == 'NAVIER_STOKES' or su2.solver == 'NEMO_NAVIER_STOKES':
            #: [str] Isothermal Marker
            self.iso = "MARKER_ISOTHERMAL = ("
            for i in range(1,len(assembly)+1):
                self.iso += "Body_"+ str(i) + ', 300'
                if i != len(assembly): self.iso +=","
            self.iso += ")"
        
        #: [str] Outlet Marker
        self.outlet = "MARKER_OUTLET = (Outlet, 1)"

        #: [str] Plot Marker
        self.plot = "MARKER_PLOTTING = ("
        for i in range(1,len(assembly)+1):
            self.plot += "Body_"+ str(i) 
            if i != len(assembly): self.plot +=","
        self.plot += ")"

        #: [str] Monitoring Marker
        self.monitor = "MARKER_MONITORING = ("
        for i in range(1,len(assembly)+1):
            self.monitor += "Body_"+ str(i) 
            if i != len(assembly): self.monitor +=","
        self.monitor += ")"

class Solver_Numerical_Method():
    """ Class Solver Numerical Method

        A class to store the solver numerical methods
        The class in the su2.py file is hardcoded to work with SU2.
    """

    def __init__ (self, su2):
        #[str] Method to compute the gradients (Default = WEIGHTED_LEAST_SQUARES)
        self.grad = "NUM_METHOD_GRAD = WEIGHTED_LEAST_SQUARES"

        #[str] CFL Number
        self.cfl = "CFL_NUMBER = " + str(su2.cfl)
        
        #[str] Adaptive CFL boolean (Default is NO)
        self.cfl_adapt = "CFL_ADAPT = NO"

        #[str] Maximum number of iterations
        self.iter = "ITER = " + str(su2.iters)

class Flow_Numerical_Method():
    """ Class Flow Numerical Method

        A class to store the flow numerical methods
        The class in the su2.py file is hardcoded to work with SU2.
    """

    def __init__(self, su2):
        #: [str] Convective method (AUSM, AUSMPLUSUP2)
        self.conv_method = "CONV_NUM_METHOD_FLOW = "+ su2.conv_method

        #: [str] MUSCL activation boolean (Default = YES)
        self.muscl = "MUSCL_FLOW = " + su2.muscl

        #: [str] Limiter method (Default = VENKATAKRISHNAN_WANG)
        self.limiter = "SLOPE_LIMITER_FLOW = VENKATAKRISHNAN_WANG"

        #: [str] Limiter coefficiet (Default = 0.01)
        self.limiter_coeff = "VENKAT_LIMITER_COEFF = 0.01"
        
        #: [str] Time discretization (Default = EULER_EXPLICIT)
        self.time = "TIME_DISCRE_FLOW = EULER_EXPLICIT"

class Solver_Convergence():
    """ Class Solver convergence

        A class to store the convergence criteria
        The class in the su2.py file is hardcoded to work with SU2.
    """

    def __init__(self):
        #: [str] Fields to look for convergence
        self.field = "CONV_FIELD= (LIFT, DRAG)"

        #: [str] Minimum residual for convergence
        self.res_min = "CONV_RESIDUAL_MINVAL= -15"

        #: [str] Start iteration for the convergence assessment
        self.start_iter = "CONV_STARTITER= 10"

        #: [str] Number of elements to be used in the Cauchy convergence window
        self.cauchy_elems = "CONV_CAUCHY_ELEMS= 100"

        #: [str] Residual for convergence using the Cauchy Window
        self.cauchy_eps = "CONV_CAUCHY_EPS= 1E-9"

class Solver_Input_Output():
    """ Class Solver Input Output

        A class to store the IO information.
        The class in the su2.py file is hardcoded to work with SU2.
    """

    def __init__(self,it, iteration, output_folder, cluster_tag):
        #: [str] Name of the mesh to be used in the simulation
        self.mesh_filename = "MESH_FILENAME= "+output_folder+"/CFD_Grid/Domain_"+str(it)+".su2"
        
        #: [str] Mesh format (Default = SU2)
        self.mesh_format = "MESH_FORMAT= SU2"

        #: [str] Solution filename to write
        self.solution_output = "SOLUTION_FILENAME= "+output_folder+"/CFD_sol/restart_flow_"+ str(it) + ".csv"

        #: [str] Solution format
        self.tabular_format = "TABULAR_FORMAT= CSV"
        
        #: [str] Generated output files
        self.output_files = "OUTPUT_FILES= (RESTART_ASCII, PARAVIEW, SURFACE_PARAVIEW)"

        #: [str] Solution filename to read
        self.solution_input = "RESTART_FILENAME = "+output_folder+"/CFD_sol/restart_flow_"+ str(it) + ".csv"

        #: [str] Name of the volume solution filename to write the simulation data
        self.output_vol = "VOLUME_FILENAME= "+output_folder+"/CFD_sol/flow_"+ str(iteration) + '_' + str(it) + '_cluster_'+str(cluster_tag)

        #: [str] Name of the surface solution filename to write the simulation data
        self.output_surf = "SURFACE_FILENAME= "+output_folder+"/CFD_sol/surface_flow_"+ str(iteration) + '_' + str(it) + '_cluster_'+str(cluster_tag)

        #: [str] Frequency for the output file generation
        self.output_freq = "OUTPUT_WRT_FREQ= 500"

        #: [str] Screen output
        self.screen = "SCREEN_OUTPUT= (INNER_ITER, WALL_TIME, FORCE_X, FORCE_Y, FORCE_Z, MOMENT_X, MOMENT_Y, MOMENT_Z)"

class SU2_Config():
    """ Class SU2 Configuration

        A class to store all the information required write the SU2 configuration file
        The class in the su2.py file is hardcoded to work with SU2.
    """

    def __init__(self,freestream, assembly, restart, it, iteration, su2, options, cluster_tag):
        #: [str] Name of the configuration file
        self.name = "Config.cfg"

        #:[Solver] Object of class Solver
        self.solver=Solver(restart, su2,freestream)

        #:[Solver_Freestream_Conditions] Object of class Solver_Freestream_Conditions
        self.free_cond=Solver_Freestream_Conditions(freestream)

        #:[Solver_BC] Object of class Solver_BC
        self.bc = Solver_BC(assembly, su2)

        #:[Solver_Reference_Value] Object of class Solver_Reference_Value
        self.ref = Solver_Reference_Value()

        #:[Solver_Numerical_Method] Object of class Solver_Numerical_Method
        self.num = Solver_Numerical_Method(su2)

        #:[Flow_Numerical_Method] Object of class Flow_Numerical_Method
        self.flow =Flow_Numerical_Method(su2)

        #:[Solver_Convergence] Object of class Solver_Convergence
        self.convergence = Solver_Convergence()

        #:[Solver_Input_Output] Object of class Solver_Input_Output
        self.inout = Solver_Input_Output(it, iteration, options.output_folder, cluster_tag)

def write_SU2_config(freestream, assembly, restart, it, iteration, su2, options, cluster_tag, input_grid, output_grid = "", interpolation = False, bloom = False, interp_to_BL = False):
    """
    Write the SU2 configuration file

    Generates a configuration file to run a SU2 CFD simulation according to the position of the object and the user-defined parameters.

    Parameters
    ----------
    freestream: Freestream
        Object of class Freestream
    assembly: Assembly_list
        Object of class Assembly_list
    restart: bool
        Boolean value to indicate if CFD simulation is restarting from previous solution
    it: int
        Value of adaptive iteration
    iteration: int
        Value of time iteration
    su2: CFD
        Object of class CFD
    options: Options
        Object of class Options
    cluster_tag: int
        Value of the cluster tag number for simulation parallelization
    input_grid: str
        Name of the input mesh file
    output_grid: str
        Name of the output file
    """

    #Creates an object of class SU2_Config
    SU2_config = SU2_Config(freestream, assembly, restart, it, iteration,  su2, options, cluster_tag)

    SU2_config.inout.mesh_filename = 'MESH_FILENAME= '+options.output_folder +'/CFD_Grid/'+input_grid

    with open(options.output_folder + '/CFD_sol/'+ SU2_config.name, 'w') as f:

        #Writes the configuration parameters into a file to be used for the CFD simulation

        f.write('% Solver Settings \n')
        for attr, value in vars(SU2_config.solver).items(): f.write(value + '\n')
        f.write('\n')

        f.write('%  Free-stream Conditions \n')
        for attr, value in vars(SU2_config.free_cond).items(): f.write(value + '\n')
        f.write('\n')

        f.write('%  Reference Settings \n')
        for attr, value in vars(SU2_config.ref).items(): f.write(value + '\n')
        f.write('\n')

        f.write('%  Boundary Conditions Settings \n')
        for attr, value in vars(SU2_config.bc).items(): f.write(value + '\n')
        f.write('\n')

        f.write('%  Solver Numerical Methods Settings \n')
        for attr, value in vars(SU2_config.num).items(): f.write(value + '\n')
        f.write('\n')

        f.write('%  Flow Numerical Methods Settings \n')
        for attr, value in vars(SU2_config.flow).items(): f.write(value + '\n')
        f.write('\n')

        f.write('%  Solver Convergence Settings \n')
        for attr, value in vars(SU2_config.convergence).items(): f.write(value + '\n')
        f.write('\n')

        f.write('%  Solver Input Output Settings \n')
        for attr, value in vars(SU2_config.inout).items(): f.write(value + '\n')
        f.write('\n')

    f.close()
    pass

def retrieve_index(SU2_type):
    """
    Retrieve index to retrieve solution fields

    Returns the index to read the correct fields in the solution file, according to the user-specified solver

    Parameters
    ----------
    SU2_type: str
        Solver used in the CFD simulation

    Returns
    -------
    index: np.array()
        Array of index with the position of solution fields of interest
    """
    #TODO: Instead of hardcoded with SU2 indexes, needs to read name of field and retrieve value
    if SU2_type == 'EULER':
        index = np.array([(0,1,3,4)], dtype = [('Density', 'i4'),('Momentum', 'i4'),('Pressure', 'i4'),('Temperature', 'i4')])
        
    if SU2_type == 'NEMO_EULER':
        index = np.array([(13)], dtype = [('Pressure', 'i4')])

    if SU2_type == 'NAVIER_STOKES':
        index = np.array([(0,1,3,4,8,9)], dtype = [('Density', 'i4'),('Momentum', 'i4'),('Pressure', 'i4'),('Temperature', 'i4'), ('Skin_Friction_Coefficient', 'i4'), ('Heat_Flux' , 'i4')])
        
    if SU2_type == 'NEMO_NAVIER_STOKES':
        index = np.array([(13,19,20)], dtype = [('Pressure', 'i4'), ('Skin_Friction_Coefficient', 'i4'), ('Heat_Flux' , 'i4')])

    return index

def read_vtk_from_su2_v2(filename, assembly_coords, idx_inv,  options):
    """
    Read the VTK file solution

    Reads and retrieves the solution stored in the VTK file format

    Parameters
    ----------
    filename: str
        Name and location of the VTK solution file
    assembly_coords: np.array()
        Coordinates of the mesh nodes
    idx_inv: np.array
        Sort indexing such that the VTK retrieved solution corresponds to the stored mesh nodes positioning
    options: Options
        Object of class Options


    Returns
    -------
    aerothermo: Aerothermo
        object of class Aerothermo
    """

    #Initializes the Aerothermo Object
    aerothermo = assembly.Aerothermo(len(assembly_coords))
    
    #Retrieve the index to read the correct solution fields
    index = retrieve_index(options.cfd.solver)

    #Open the VTK solution file
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()

    npoints = reader.GetNumberOfPoints()
    narrays = reader.GetNumberOfPointArrays()
    
    data = reader.GetOutput()
    coords = vtk_to_numpy(data.GetPoints().GetData())

    #sorts the solution by the correspondent coordinates of the nodes
    coords_sorted , idx_sim= np.unique(coords, axis = 0, return_index = True)
    
    #Missing the different densities from NEMO
    #Retrieves the solution fields and stores them in the aerothermo object.
    if ('Density' in index.dtype.names): aerothermo.density = vtk_to_numpy(data.GetPointData().GetArray(index['Density'][0]))[idx_sim][idx_inv]
    if ('Temperature' in index.dtype.names): aerothermo.temperature = vtk_to_numpy(data.GetPointData().GetArray(index['Temperature'][0]))[idx_sim][idx_inv]
    #if ('Momentum' in index.dtype.names): aerothermo.momentum = vtk_to_numpy(data.GetPointData().GetArray(index['Momentum'][0]))[idx_sim][idx_inv][:,None]
    if ('Pressure' in index.dtype.names): aerothermo.pressure = vtk_to_numpy(data.GetPointData().GetArray(index['Pressure'][0]))[idx_sim][idx_inv]
    if ('Skin_Friction_Coefficient' in index.dtype.names): aerothermo.shear = vtk_to_numpy(data.GetPointData().GetArray(index['Skin_Friction_Coefficient'][0]))[idx_sim][idx_inv]
    if ('Heat_Flux' in index.dtype.names): aerothermo.heatflux = vtk_to_numpy(data.GetPointData().GetArray(index['Heat_Flux'][0]))[idx_sim][idx_inv]

    return aerothermo

def split_aerothermo(total_aerothermo, assembly):
    """
    Split the solution into the different assemblies used in the CFD simulation

    Parameters
    ----------
    total_aerothermo: Aerothermo
        Object of class Aerothermo
    assembly:List_Assembly
        Object of class List_Assembly
    """

    aerothermo_list = []

    last_node = 0
    first_node = 0

    #Loop through all the assembly objects
    for it in range(len(assembly)):

        nodes = np.copy(assembly[it].cfd_mesh.nodes[assembly[it].cfd_mesh.edges])

        nodes.shape = (-1,3)
        nodes = np.unique(nodes, axis = 0)

        last_node += len(nodes)    
        cfd_nodes_sorted, idx_inv = np.unique(assembly[it].cfd_mesh.nodes, axis = 0, return_inverse = True)
        node_index, node_mask = mesh.create_index(cfd_nodes_sorted, nodes)

        #Store the solution into the correspondent assembly
        for field in [field for field in dir(assembly[it].aerothermo) if not field.startswith('__')]:
            if field == 'density':      assembly[it].aerothermo.density[node_index]      = total_aerothermo.density     [first_node:last_node]; assembly[it].aerothermo.density = assembly[it].aerothermo.density[idx_inv]
            if field == 'temperature':  assembly[it].aerothermo.temperature[node_index]  = total_aerothermo.temperature [first_node:last_node]; assembly[it].aerothermo.temperature = assembly[it].aerothermo.temperature[idx_inv]
           #if field == 'momentum':     assembly[it].aerothermo.momentum[node_index]     = total_aerothermo.momentum    [first_node:last_node]
            if field == 'pressure':     assembly[it].aerothermo.pressure[node_index]     = total_aerothermo.pressure    [first_node:last_node]; assembly[it].aerothermo.pressure = assembly[it].aerothermo.pressure[idx_inv]
            if field == 'skinfriction': assembly[it].aerothermo.skinfriction[node_index] = total_aerothermo.shear[first_node:last_node]; assembly[it].aerothermo.shear = assembly[it].aerothermo.shear[idx_inv]
            if field == 'heatflux':     assembly[it].aerothermo.heatflux[node_index]     = total_aerothermo.heatflux    [first_node:last_node]; assembly[it].aerothermo.heatflux = assembly[it].aerothermo.heatflux[idx_inv]

        first_node = last_node

def run_SU2(n, options):
    """
    Calls the SU2 executable and run the simulation

    Parameters
    ----------
    n: int
        Number of cores
    options: Options
        Object of class Options
    """

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([path+'/Executables/mpirun','-np', str(n), path+'/Executables/SU2_CFD',options.output_folder +'/CFD_sol/Config.cfg'], text = True)

def generate_BL(assembly, options, it, cluster_tag):
    """
    Generates a Boundary Layer

    Parameters
    ----------
    assembly: List_Assembly
        Object of class List_Assembly
    options: Options
        Object of class Options
    it: int
        Value of adaptive iteration
    cluster_tag: int
        Value of Cluster tag
    """

    if options.bloom.flag:
        bloom.generate_BL(it, options, num_obj = len(assembly), bloom = options.bloom, input_grid ='Domain_'+str(it)+'_cluster_'+str(cluster_tag) , output_grid = 'Domain_'+str(it)+'_cluster_'+str(cluster_tag)) #grid name without .SU2

    
def adapt_mesh(assembly, options, it, cluster_tag):
    """
    Anisotropically adapts the mesh

    Parameters
    ----------
    assembly: List_Assembly
        Object of class List_Assembly
    options: Options
        Object of class Options
    it: int
        Value of adaptive iteration
    cluster_tag: int
        Value of Cluster tag
    """

    if options.amg.flag:
        amg.adapt_mesh(options.amg, options, j = it, num_obj = len(assembly),  input_grid = 'Domain_'+str(it)+'_cluster_'+str(cluster_tag), output_grid = 'Domain_'+str(it+1)+'_cluster_'+str(cluster_tag)) #Output without .su2


def compute_cfd_aerothermo(assembly_list, options, cluster_tag = 0):
    """
    Compute the aerothermodynamic properties using the CFD software

    Parameters
    ----------
    assembly_list: List_Assembly
        Object of class List_Assembly
    options: Options
        Object of class Options
    cluster_tag: int
        Value of Cluster tag
    """
    
    #TODO:
    # ---> size ref should also be in the options config file

    num_obj = len(assembly_list)
    iteration = options.current_iter
    su2 = options.cfd 
    amg = options.amg 
    bloom = options.bloom
    free = options.freestream

    n = options.cfd.cores
    if options.amg.flag: 
        adapt_iter = options.cfd.adapt_iter
    else:
        adapt_iter = 0
    aerothermo = []

    #Choose index of the object with lower altitude
    altitude = 1E10
    restart = False

    for index,assembly in enumerate(assembly_list):
        if assembly.trajectory.altitude < altitude:
            altitude = assembly.trajectory.altitude
            it = index
            lref = assembly.Lref
    

    free = assembly_list[it].freestream
    #TODO options for ref_size_surf

    #Convert from Body->ECEF and ECEF-> Wind
    #Translate the mesh to match the Center of Mass of the lowest assembly
    assembly_windframe = deepcopy(assembly_list)

    pos = assembly_list[it].position
    
    for i, assembly in enumerate(assembly_windframe):

        R_B_ECEF = Rot.from_quat(assembly.quaternion)

        assembly.cfd_mesh.nodes -= assembly.COG
        assembly.cfd_mesh.nodes = R_B_ECEF.apply(assembly.cfd_mesh.nodes)

        #Translate to the ECEF position
        assembly.cfd_mesh.nodes += np.array(assembly.position-pos)        

        R_ECEF_NED = frames.R_NED_ECEF(lat = assembly.trajectory.latitude, lon = assembly.trajectory.longitude).inv()
        R_NED_W = frames.R_W_NED(ha = assembly.trajectory.chi, fpa = assembly.trajectory.gamma).inv()

        R_ECEF_W = R_NED_W*R_ECEF_NED 
        assembly.cfd_mesh.nodes = (R_ECEF_W).apply(assembly.cfd_mesh.nodes)

        assembly.cfd_mesh.xmin = np.min(assembly.cfd_mesh.nodes , axis = 0)
        assembly.cfd_mesh.xmax = np.max(assembly.cfd_mesh.nodes , axis = 0)

    #Automatically generates the CFD domain
    GMSH.generate_cfd_domain(assembly_windframe, 3, ref_size_surf = 0.5, ref_size_far = 0.5 , output_folder = options.output_folder, output_grid = 'Domain_'+str(0)+'_cluster_'+str(cluster_tag)+'.su2')
    
    #Generate the Boundary Layer (if flag = True)
    generate_BL(assembly_list, options, 0, cluster_tag)

    #Writes the configuration file
    it = 0
    config = write_SU2_config(free, assembly_list, restart, it, iteration, su2, options, cluster_tag, input_grid = 'Domain_'+str(it)+'_cluster_'+str(cluster_tag)+'.su2',bloom=False)
    
    #Runs SU2 simulaton
    run_SU2(n, options)

    #Anisotropically adapts the mesh and runs SU2 until reaches the maximum numbe of adaptive iterations
    if options.amg.flag:
        for it in range(options.cfd.adapt_iter):
            restart = True
            adapt_mesh(assembly_list, options, it, cluster_tag)
            config = write_SU2_config(free, assembly_list, restart, it+1, iteration, su2, options, cluster_tag, input_grid = 'Domain_'+str(it+1)+'_cluster_'+str(cluster_tag)+'.su2',bloom=False)
            run_SU2(n, options)

    assembly_nodes = np.array([])
    assembly_facets = np.array([], dtype = int)

    for assembly in assembly_list:
        nodes = assembly.cfd_mesh.nodes[assembly.cfd_mesh.edges]
        nodes.shape = (-1,3)
        nodes = np.unique(nodes, axis = 0)
        assembly_nodes = np.append(assembly_nodes,nodes)

    assembly_nodes.shape = (-1,3)
    assembly_nodes,idx_inv = np.unique(assembly_nodes, axis = 0, return_inverse = True)
    
    #Reads the solution file and stores into the different assemblies
    total_aerothermo = read_vtk_from_su2_v2(options.output_folder+'/CFD_sol/surface_flow_'+str(iteration)+'_'+str(adapt_iter)+'_cluster_'+str(cluster_tag)+'.vtu', assembly_nodes, idx_inv, options)
    split_aerothermo(total_aerothermo, assembly_list)