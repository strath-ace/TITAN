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
import math
import pandas as pd
from copy import deepcopy
from scipy.spatial.transform import Rotation as Rot
import meshio
from pathlib import Path
import subprocess
import os
import vtk
from Geometry import assembly as Assembly
import trimesh
from vtk.util.numpy_support import vtk_to_numpy
from Geometry import mesh
from Geometry import assembly
from Dynamics import frames
import glob
from Geometry import mesh as Mesh
import shutil
import os
import subprocess
from scipy.spatial import KDTree

class Initialization():

    """ Class Initialization

        A class to store the initialization information.
        The class in the sparta.py file is hardcoded to work with SPARTA.
    """

    def __init__(self):
        rng = np.random.default_rng()
        
        #: [int] Define random seed number 
        self.seed = 'seed \t\t' + str(rng.integers(low=11111, high=99999))
        
        #: [str] Define 3-dimensional problem(default, hardcoded definition)
        self.dim = 'dimension \t\t3'
        
        #: [str] Define gridcut operation and place maximum surfaces by cell
        self.gridcut = 'global \t\t gridcut 0.0 comm/sort yes surfmax 100000'

        #: [str] Boundary condition string for SPARTA: open in x, y, z directions
        self.bc = 'boundary \t\to o o'


class Problem_Definition():

    """ Class Problem_Definition

        A class to store the problem defined information.
        The class in the sparta.py file is hardcoded to work with SPARTA.
    """

    def __init__(self, options, dsmc, freestream):

        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        species = ' '.join(dsmc.sp_present)

        #: [str] Define SPARTA domain box with calculated dimensions based on vehicle characteristic length
        self.box = 'create_box\t\t ' + str(dsmc.domain[0][0]) + ' ' + str(dsmc.domain[0][1]) + ' ' +  str(dsmc.domain[1][0]) + ' ' + str(dsmc.domain[1][1]) +  ' ' +str(dsmc.domain[2][0]) + ' ' + str(dsmc.domain[2][1])
        
        #: [int] Create SPARTA grid with defined number of cells per dimension and initialize grid hierarchy for later refinement 
        # levels = maximum depth of refinement
        # self.grid = 'create_grid\t\t ' + str(dsmc.grid[0]) + ' ' + str(dsmc.grid[1]) + ' ' + str(dsmc.grid[2]) + ' levels 3 subset 2*3 * * * 2 2 2'
        self.grid = 'create_grid\t\t ' + str(dsmc.grid[0]) + ' ' + str(dsmc.grid[1]) + ' ' + str(dsmc.grid[2]) + ' levels ' + str(dsmc.level) + ' subset 2*' + str(dsmc.level) + ' * * *  2 2 2'
        
        #: [str] Write balance grid command to rebalance grid across processors 
        self.balance_grid = 'balance_grid\t\t rcb cell'
        
        #: [str] Write fbalance command to redistribute particles across processors
        self.f_balance = 'fix\t\t fbalance balance 1000 1.1 rcb part'
        
        #: [int] Define random number density and fnum
        self.global_particles = 'global\t\t nrho ' + str(dsmc.nrho) + ' fnum ' + str(dsmc.fnum) 

        #: [str] Write air species used, with location of species details
        self.species = 'species\t\t ' + path + '/Executables/air.species ' + str(species)
        
        #: [int] Define free-stream conditions 
        self.mixture = 'mixture\t\t air ' + str(species) + ' vstream ' + str(freestream.velocity) + ' ' + str(0) + ' ' + str(0) + ' temp ' + str(round(freestream.temperature,2))

        #: [int] Define N2 species concentration
        if "N2" in dsmc.sp_present: self.frac_N2 = 'mixture\t\t air N2 frac ' + str(dsmc.sp_frac[0])
        
        #: [int] Define O2 species concentration
        if "O2" in dsmc.sp_present: self.frac_O2 = 'mixture\t\t air O2 frac ' + str(dsmc.sp_frac[1])
        
        #: [int] Define N species concentration
        if "N" in dsmc.sp_present: self.frac_N = 'mixture\t\t air N frac ' + str(dsmc.sp_frac[2])
        
        #: [int] Define O species concentration
        if "O" in dsmc.sp_present: self.frac_O = 'mixture\t\t air O frac ' + str(dsmc.sp_frac[3])       


class Geometry_Definition():

    """ Class Geometry_Definition

        A class to store the geometry-related information.
        The class in the sparta.py file is hardcoded to work with SPARTA.
    """

    def __init__(self, options, dsmc, assembly_list, output_folder, surface_filename):

        for i, assembly in enumerate(assembly_list):

            #: [str] Read surface mesh from file and assign to a named surface group
            self.read_surf = 'read_surf\t\t ' + str(output_folder) + '/DSMC_sol/' + str(surface_filename[i]) + '.data group ' + str(surface_filename[i])
            
            #: [str] Write a compound surface file (with points) for visualization/postprocessing
            self.write_surf = 'write_surf\t\t ' + str(output_folder) + '/DSMC_sol/' + f'compound_{surface_filename[i]}.data points yes'

            for j, obj in enumerate(assembly.objects):
                
                name = f'{surface_filename[i]}_obj_{j}'
                
                #: [str] Assign surface ID range to a named group representing a physical object
                self.__dict__[name] = 'group\t\t ' + str(name) + ' surf id ' + str(int(dsmc.obj_store[i][j][0])) + ':' + str(int(dsmc.obj_store[i][j][1]))


class Settings():

    """ Class Settings

        A class to store the simulation settings for SPARTA.
        The class in the sparta.py file is hardcoded to work with SPARTA.
    """

    def __init__(self, options, dsmc, assembly_list, surface_filename):

        # SPARTA uses integer IDs to define surface collision models.
        # `obj_count` tracks a cumulative ID for each object in the simulation.
        obj_count = 1

        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        for i, assembly in enumerate(assembly_list):


            for j, obj in enumerate(assembly.objects):
                
                name_collide = f'{surface_filename[i]}_obj_{j}'      # Group name (matches group defined earlier)
                collide = f'collide_{surface_filename[i]}_obj_{j}'   # Unique collision setting name
                
                #: [str] Define how particles interact with this surface (diffuse + temperature + accommodation coefficient)
                self.__dict__[collide] = 'surf_collide\t\t ' + str(obj_count) + ' diffuse ' + str(round(obj.temperature, 4)) + ' ' + str(dsmc.acc)
                
                #: [str] Assign this collision model to the corresponding group
                self.__dict__[name_collide] = 'surf_modify\t\t ' + str(name_collide) +   ' collide ' + str(obj_count)
                
                obj_count += 1



        #: [str] Global collision model setup: uses Variable Hard Sphere (VHS) model with air species [CHECK THIS MODEL]
        self.collision_model = 'collide\t\t vss air ' +  path + '/Executables/air.vhs'
        
        #: [str] Modify vibration treatment in collisions to use smoothing (improves numerical stability)
        self.collision_mod = 'collide_modify\t\t vibrate smooth'
        
        #: [str] Emit particles from xlo boundary face 
        self.bcs = 'fix\t\t in emit/face air xlo ylo yhi zlo zhi'
        
        #: [str] Set the global timestep value
        self.dt = 'timestep \t\t ' + str(dsmc.dt)
        
        #: [str] Output statistics every N timesteps
        self.stats = 'stats \t\t ' + str(dsmc.stats_freq)
        
        #: [str] Define which statistics are printed
        self.stats_style = 'stats_style \t\t step cpu np nattempt ncoll nscoll nscheck maxlevel'
        
        #: [str] Format style for printed statistics
        self.stats_format = 'stats_modify \t\t format float %1.3e'

        
class Unsteady_Simulation():

    """ Class Unsteady_Simulation

        A class to store the unsteady simulation steps for SPARTA.
        The class in the sparta.py file is hardcoded to work with SPARTA.
    """

    # Define unsteady simulation, including grid adaptation and initial simulation period to reach steady simulation:

    def __init__(self, options, dsmc, output_folder, surface_filename, grid_filename):

        #: [str] Write the initial, unadapted grid to file
        self.write_grid = 'write_grid\t\t ' + str(output_folder) + '/DSMC_Grid/' + f'initial_{grid_filename}'

        #: [str] Perform grid refinement based on surface proximity (static, not time-evolving)
        # NOTE: This uses only the first surface group for now; may require generalization for multiple groups
        # NEED TO UNDERSTAND HOW THIS COULD BE DONE DIFFERENTLY - IF A COMMAND CAN BE USED FOR EACH ASSEMBLY 
        self.static_adapt = 'adapt_grid\t\t all refine surf ' + str(surface_filename[0]) + ' ' + str(dsmc.adapt_surf_tr) + ' iterate 2 maxlevel ' + str(dsmc.level - 1) 

        #: [str] Output grid after static adaptation
        self.write_grid = 'write_grid\t\t ' + str(output_folder) + '/DSMC_Grid/' + f'static_adapt_{grid_filename}'

        #: [str] Run the main simulation with statically refined grid
        self.run_unsteady_0 = f'run\t\t {dsmc.run_unsteady}'

        #: [str] Compute number density (n), mass density (nrho), and temperature (temp) per grid cell
        self.compute_grid = 'compute\t\t adapt_compute grid all air n nrho temp'

        #: [str] Time-average the computed properties over Nadapt timesteps
        self.adapt_fix = f'fix\t\t adapt_avg ave/grid all 10 10 {dsmc.Nadapt} c_adapt_compute[*]'
        
        #: [str] Adapt the grid based on averaged number density — triggers coarsen/refine when values fall outside range
        self.active_adapt = f'fix\t\t active_adapt adapt {dsmc.Nadapt} all coarsen refine value f_adapt_avg[1] ' + str(dsmc.nrho * 0.2) + ' ' + str(dsmc.nrho * 0.05)

        #: [str] Run the simulation for a defined number of timesteps during adaptation
        self.run_adapt = f'run\t\t {dsmc.run_adapt}'

        #: [str] Remove adaptive compute and fixes to reset for main run
        self.uncompute_grid = 'uncompute\t\t adapt_compute'
        self.unfix_adapt = 'unfix\t\t active_adapt'
        self.unfix_adaptcompute = 'unfix\t\t adapt_avg'

        #: [str] Output the adapted grid before main simulation begins
        self.final_grid_write = 'write_grid\t\t ' + str(output_folder) + '/DSMC_Grid/' + f'dynamic_adapted_{grid_filename}'

        #: [str] Run the main simulation with final grid
        self.run_unsteady_1 = f'run\t\t {dsmc.run_unsteady}'


class Sampling_Simulation():

    """ Class Sampling_Simulation

        A class to store the sampling simulation steps for SPARTA.
        The class in the sparta.py file is hardcoded to work with SPARTA.
    """

    # Define sampling simulation, including computes, averaging (fixes) for results:

    def __init__(self, options, dsmc, assembly_list, output_folder, surface_filename):
        
        self.compute_surf = []
        self.fix_surf = []
        self.dump_surf = []

        for i, assembly in enumerate(assembly_list):
            
            #: [str] Compute surface quantities (force, heat flux, energy modes, etc.) for a surface group
            self.compute_surf.append(f'compute\t\t 10{i} surf {surface_filename[i]} air mflux fx fy fz press shx shy shz etot erot evib ke')
            
            #: [str] Time-average the computed surface properties over user-defined intervals
            self.fix_surf.append(f'fix\t\t 10{i} ave/surf {surface_filename[i]} {dsmc.Nevery} {dsmc.Nrepeat} {dsmc.Ntotal} c_10{i}[*]')
            
            #: [str] Dump surface-averaged results to file, including force and heat flux components
            self.dump_surf.append(f'dump\t\t 10{i}_dump surf {surface_filename[i]} {dsmc.Ntotal} ' + str(output_folder) + '/DSMC_sol/' + f'results.surf_{surface_filename[i]}.* id f_10{i}[*]')

        #: [str] Compute flow field properties per grid cell (velocity, energy, temperature)
        self.compute_gridflow = 'compute\t\t grid_all grid all air n nrho u v w ke temp erot trot evib tvib'
        
        #: [str] Compute thermal-only quantities (temperature and pressure)
        self.compute_thermalflow = 'compute\t\t thermal_all thermal/grid all air temp press'
        
        #: [str] Time-average the above grid quantities over user-defined intervals
        self.fix_gridflow = f'fix\t\t grid_fix ave/grid all {dsmc.Nevery} {dsmc.Nrepeat} {dsmc.Ntotal} c_grid_all[*] c_thermal_all[*]'
        
        #: [str] Compute mean free path using number density and temperature from averaged data
        self.compute_fow = f'compute\t\t mean_free lambda/grid f_grid_fix[2] f_grid_fix[12] N2 kx'
        
        if dsmc.grid_results:
            #: [str] Dump all grid-averaged results to file (cell-centered values + mean free path)
            self.dump_flow = f'dump\t\t all_dump grid all {dsmc.Ntotal} ' +  str(output_folder) + '/DSMC_Grid/' + f'results.grid_{surface_filename[0]}.* xc yc zc id f_grid_fix[*] c_mean_free[*]'

        if dsmc.write_restart:

            # [str] Write restart file for safety twice during the simulation run
            self.write_restart = f'write_restart \t\t  {output_folder}/DSMC_sol/restart_file_{0}.*'

            restart_ideal = dsmc.run_sample / 2
            restart_safe = (restart_ideal // dsmc.Ntotal) * dsmc.Ntotal

            #: [str] Run simulation for sampling duration
            self.run_sample_0 = f'run\t\t {int(restart_safe)}'

            # [str] Write restart file for safety twice during the simulation run
            self.write_restart_0 = f'write_restart \t\t  {output_folder}/DSMC_sol/restart_file_{int(restart_safe)}.*'

            #: [str] Run simulation for sampling duration
            self.run_sample_1 = f'run\t\t {dsmc.run_sample - int(restart_safe)}'

            # [str] Write restart file for safety twice during the simulation run
            self.write_restart_1 = f'write_restart \t\t  {output_folder}/DSMC_sol/restart_file_{dsmc.run_sample}.*'

        else: 
            #: [str] Run simulation for sampling duration
            self.run_sample = f'run\t\t {dsmc.run_sample}'

        #: [str] Clean-up: unfix/undump/uncompute
        self.uncompute_gridflow = f'uncompute\t\t grid_all'
        self.uncompute_thermalflow = f'uncompute\t\t thermal_all'
        self.unfix_gridflow = f'unfix\t\t grid_fix'
        self.uncompute_flow = f'uncompute\t\t mean_free'
        self.undump_flow = f'undump\t\t all_dump'

        self.uncompute_surf = []
        self.unfix_surf = []
        self.undump_surf = []

        for i, assembly in enumerate(assembly_list):

            #: [str] Remove surface sampling components
            self.uncompute_surf.append(f'uncompute\t\t 10{i}')
            self.unfix_surf.append(f'unfix\t\t 10{i}')
            self.undump_surf.append(f'undump\t\t 10{i}_dump')
                    

class SPARTA_Config():

    def __init__(self, options, dsmc, free, assembly_list, output_folder, sparta_config, surface_filename, grid_filename):
        
        #: [str] Name of the configuration file
        self.name = sparta_config

        #:[Initialization] Object of class Initialization
        self.base=Initialization()

        #:[ProblemDef_Properties] Object of class ProblemDef_Properties
        self.prob_def=Problem_Definition(options, dsmc, free)

        #:[ProblemDef_Geo] Object of class ProblemDef_Geo
        self.geo_def = Geometry_Definition(options, dsmc, assembly_list, output_folder, surface_filename)

        #:[Settings] Object of class Settings
        self.run_settings =Settings(options, dsmc, assembly_list, surface_filename)

        #:[Unsteady_Simulation] Object of class Unsteady_Simulation
        self.unsteady_run =Unsteady_Simulation(options, dsmc, output_folder, surface_filename, grid_filename)

        #:[Sampling_Simulation] Object of class Sampling_Simulation
        self.sample_run =Sampling_Simulation(options, dsmc, assembly_list, output_folder, surface_filename)


def write_sparta_config(options, dsmc, freestream, assembly_list, output_folder = '', sparta_config = 'in.config', 
                        surface_filename = ['assembly_0_cluster_0'], grid_filename = 'grid_cluster_0.txt'):

    SPARTA_config = SPARTA_Config(options, dsmc, freestream, assembly_list, output_folder, sparta_config, surface_filename, grid_filename)

    with open(output_folder + '/DSMC_sol/' + sparta_config, 'w') as f:

        f.write('# Initialization \n\n')
        for attr, value in vars(SPARTA_config.base).items(): f.write(value + '\n')
        f.write('\n\n')

        f.write('# Problem Definition \n\n')
        for attr, value in vars(SPARTA_config.prob_def).items(): f.write(value + '\n')
        f.write('\n\n')

        f.write('# Geometry Definition \n\n')
        for attr, value in vars(SPARTA_config.geo_def).items(): f.write(value + '\n')
        f.write('\n\n')

        f.write('# Settings \n\n')
        for attr, value in vars(SPARTA_config.run_settings).items(): f.write(value + '\n')
        f.write('\n\n')

        f.write('# Unsteady Simulation for Grid Adaptation \n\n')
        for attr, value in vars(SPARTA_config.unsteady_run).items(): f.write(value + '\n')
        f.write('\n\n') 

        f.write('# Sampling Simulation for Results \n\n')
        for attr, value in vars(SPARTA_config.sample_run).items():
            if isinstance(value, list):
                for line in value:
                    f.write(line + '\n')
            else:
                f.write(value + '\n')

        f.write('\n\n') 

        # End Simulation
        f.write('quit \n')


def create_sparta_geo(options, dsmc, assembly_list, output_folder = '', surface_filename = ['assembly_0_cluster_0']):

    """ Function create_sparta_geo

        A function to generate sparta appropriate geometry file.
        Geometries are computed for each time step and asssembly in the TITAN simulation. 
    """

    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for it, assembly in enumerate(assembly_list):

        # Generate stl file of assembly with internal surfaces removed by using 'cfd mesh'
        unique, idx, inv, counts = np.unique(assembly.cfd_mesh.nodes, axis = 0, return_index = True, return_inverse = True, return_counts = True)
        cells = [("triangle", assembly.cfd_mesh.facets)]
        trimesh = meshio.Mesh(assembly.cfd_mesh.nodes, cells)
        trimesh.write(output_folder + '/DSMC_Grid/' + surface_filename[it] + '.stl', file_format = "stl")
        print(f'[INFO] stl file written to: {output_folder}/DSMC_Grid/{surface_filename[it]}.stl')

        subprocess.run(["python3", path  + "/Executables/stl2surf.py", (f"{path}/{output_folder}/DSMC_Grid/{surface_filename[it]}.stl"), f"{path}/{output_folder}/DSMC_sol/{surface_filename[it]}.data"], cwd = path)
        print(f'[INFO] data file written to: {path}/{output_folder}/DSMC_sol/{surface_filename[it]}.data')
        
        ## Split facets across assembly objects to define individual object temperatures in sparta config
        # Define nodes and facets of mesh
        nodes = assembly.cfd_mesh.nodes
        facet = assembly.cfd_mesh.facets
        store = np.zeros(shape=(len(assembly.objects), 2))

        # Iterate across each object, checking mesh of object against facets in total mesh
        for i, obj in enumerate(assembly.objects):
            obj_mesh = obj.mesh.nodes
            save = []
        
            for j, fa in enumerate(facet):
                fx = fa[0]
                fy = fa[1]
                fz = fa[2]

                x = nodes[fx].tolist()
                y = nodes[fy].tolist()
                z = nodes[fz].tolist()

                mx = set(np.where(np.equal(obj_mesh, x)[:,0])[0]) & set(np.where(np.equal(obj_mesh, x)[:,1])[0]) & set(np.where(np.equal(obj_mesh, x)[:,2])[0])
                my = set(np.where(np.equal(obj_mesh, y)[:,0])[0]) & set(np.where(np.equal(obj_mesh, y)[:,1])[0]) & set(np.where(np.equal(obj_mesh, y)[:,2])[0])
                mz = set(np.where(np.equal(obj_mesh, z)[:,0])[0]) & set(np.where(np.equal(obj_mesh, z)[:,1])[0]) & set(np.where(np.equal(obj_mesh, z)[:,2])[0])

                if (len(mx) & len(my) & len(mz)) > 0: save.append(j+1)

            store[i] = [save[0], save[-1]]

        dsmc.obj_store.append(store.tolist())


def compute_dsmc_properties(dsmc, free, assembly_list):

    """ Function compute_dsmc_properties

        A function to compute related dsmc properties from the TITAN time step free-stream data.
        Sets domain size, grid size, number density, particle scaling (fnum), species list, and time step.
    """

    def convert_density2numberDensity(percent_mole,species,density):

        """ Function convert_density2numberDensity

            A function to convert mass density to number density using Avogadro's law.
            Uses hardcoded molar masses for each species.
        """

        Avo = 6.022169e23               #Avogrados number 
        mN2 = 28.01340/1E3;             #molar mass of nitrogen molecule, kg/mole
        mO2 = 31.99880/1E3;             #molar mass of oxigen molecule,   kg/mole
        mO = mO2/2.0;                   #molar mass of oxigen atom,       kg/mole
        mN = mN2/2.0;                   #molar mass of Nitrogen atom,     kg/mole
        mAr = 39.9480/1E3;              #molar mass of Argon molecule,    kg/mole
        mHe = 4.0026020/1E3;            #molar mass of helium molecule,   kg/mole
        mH= 1.007940/1E3;               #molar mass of Hydrogen molecule, kg/mole
        ninf = 0

        for specie in species:
            if specie == "N2":    ninf += (mN2*percent_mole[0][0])
            if specie == "O2":    ninf += (mO2*percent_mole[0][1])
            if specie == "O" :    ninf += (mO*percent_mole[0][2])
            if specie == "N" :    ninf += (mN*percent_mole[0][3])
            if specie == "Ar":    ninf += (mAr*percent_mole[0][4])
            if specie == "He":    ninf += (mHe*percent_mole[0][5])
            if specie == "H" :    ninf += (mH*percent_mole[0][6])
                
        return (Avo/ninf)*density


    # === PROBLEM DEFINITION ===
    # Define domain using front and back scaling (hardcoded currently)
    dom_fac = [2,3]
    xmin, xmax = np.zeros(3), np.zeros(3)
    lref = 0
    
    # Get mesh bounds and reference length across all assemblies
    for assembly in assembly_list:

        xmin = np.minimum(xmin, assembly.mesh.min)
        xmax = np.maximum(xmax, assembly.mesh.max)
        lref = max(lref, assembly.Lref)

    # Define domain size from scaling, bounds, and reference length
    xlo = math.floor(xmin[0] - lref * dom_fac[0])
    xhi = math.ceil(xmax[0] + lref * dom_fac[1])
    ylo = math.floor(xmin[1] - lref * dom_fac[0])
    yhi = math.ceil(xmax[1] + lref * dom_fac[0])
    zlo = math.floor(xmin[2] - lref * dom_fac[0])
    zhi = math.ceil(xmax[2] + lref * dom_fac[0])

    dsmc.domain = np.array([[xlo, xhi], [ylo, yhi], [zlo, zhi]])

    # === GRID SIZE ===
    # Minimum grid cell size = 1/3 of mean free path for numerical stability
    # Gives very refined grids - we won't use for now and instead will focus on grid refinement strategy
    def safe_grid_size(lo, hi, mfp): return max(1, math.ceil(abs(lo - hi) / (mfp / 3)))
    dsmc.grid = [safe_grid_size(xlo, xhi, free.mfp),
                 safe_grid_size(ylo, yhi, free.mfp),
                 safe_grid_size(zlo, zhi, free.mfp),]

    domain_x = abs(xhi - xlo)
    domain_y = abs(yhi - ylo)
    domain_z = abs(zhi - zlo)
    print('free.mfp = ', free.mfp)
    if free.mfp > lref: 
        coarse_cell_size = free.mfp/3
        refined_cell_size = lref/10
    else:
        coarse_cell_size = lref/10
        refined_cell_size = free.mfp/3


    # coarse_cell_size = lref / 10  # Or tune to suit size of domain and resources
    # coarse_cell_size = free.mfp * 2  # Or tune to suit size of domain and resources

    Nx = math.ceil(domain_x / coarse_cell_size)
    Ny = math.ceil(domain_y / coarse_cell_size)
    Nz = math.ceil(domain_z / coarse_cell_size)

    if Nx <= 1: Nx = 2
    if Ny <= 1: Ny = 2
    if Nz <= 1: Nz = 2

    dsmc.grid = [Nx, Ny, Nz]

    print('grid =', dsmc.grid)

    # refined_cell_size = free.mfp / 3
    # refined_cell_size = max(free.mfp / 3, coarse_cell_size / 4)

    # print('refined_cell_size = ',refined_cell_size)
    # print('coarse_cell_size = ',coarse_cell_size)

    dsmc.level = math.ceil(math.log2(coarse_cell_size / refined_cell_size))
    print('dsmc.levels = ', dsmc.level)

    if dsmc.level < 0: dsmc.level = 2
    elif dsmc.level > 5: dsmc.level = 5
    if free.mfp > lref: dsmc.level = 2
    dsmc.adapt_surf_tr = lref*0.2

    print('dsmc.levels = ', dsmc.level)
    print('Kn = ', free.knudsen)


    # === NUMBER DENSITY ===
    dsmc.nrho = convert_density2numberDensity(free.percent_mole, free.species_index, free.density)

    # === PARTICLE CALCULATIONS ===
    volume = abs(xhi - xlo) * abs(yhi - ylo) * abs(zhi - zlo)
    total_cells = dsmc.grid[0] * dsmc.grid[1] * dsmc.grid[2]

    # ratio of real atoms or molecules with simulation particles in DSMC
    dsmc.fnum = (dsmc.nrho * volume) / (total_cells * dsmc.ppc)


    # === SPECIES DEFINITION ===
    # Species fractions hardcoded for NRLMSISE00
    N2 = abs(np.around(1 - (np.around(free.percent_mass[0][1], 5) + np.around(free.percent_mass[0][5], 5) + np.around(free.percent_mass[0][2], 5)), 5))
    O2 = abs(np.around(free.percent_mass[0][1], 5))
    N  = abs(np.around(free.percent_mass[0][5], 5))
    O  = abs(np.around(free.percent_mass[0][2], 5))

    dsmc.sp_frac = [N2, O2, N, O]
    species = ['N2', 'O2', 'N', 'O']

    # Remove any zero-contribution species
    dsmc.sp_present = [sp for sp, frac in zip(species, dsmc.sp_frac) if frac > 0]

    # === TIME STEP CALCULATION ===
    # Time step must be small enough to resolve collisions and advection
    k = 1.38064852e-23  # Boltzmann constant [J/K]
    mass = free.density / dsmc.nrho             # Mass per particle
    c_mean = math.sqrt((8 * k * free.temperature) / (math.pi * mass))  # Mean thermal speed

    # Check to use mean or most probable speed
    # Most Probable speed
    #c = math.sqrt((k * T_inf)/(mass))

    MCT = free.mfp / c_mean                     # Mean collision time
    MTT = free.mfp / (3 * free.velocity)        # Mean transit time (conservative)

    # Conservative timestep: minimum of MCT, MTT divided by safety factor - this potentially will prevent sufficient collisions occuring - using something
    # less conservative instead (original dt equation is from TITAN private branch ~2022)
    # dsmc.dt = min(MCT, MTT) / 10
    dsmc.dt = MCT/ 3


def run_SPARTA(options, dsmc, assembly_list, output_folder = '', sparta_config = 'in.config', sparta_log = 'sparta_log.log'):
    
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Define paths input/log files
    input_file = os.path.join(output_folder + '/DSMC_sol/', sparta_config)
    log_file = os.path.join(output_folder + '/DSMC_sol/', sparta_log)

    # print('RUNNING SPARTA:', "mpirun", "-np", str(dsmc.cores), path+'/Executables/spa_openmpi', "-in", input_file, "-log", log_file)
    print('RUNNING SPARTA:', "mpirun", "-np", str(dsmc.cores), path+'/Executables/spa_openmpi', "-in", input_file)

    # Run SPARTA
    subprocess.run(["mpirun", "-np", str(dsmc.cores), path+'/Executables/spa_openmpi', "-in", input_file, "-log", log_file], text=True)


def mapping_facet_COG(facet_COG, vtk_COG):

        A = facet_COG
        B = vtk_COG

        tree = KDTree(B)
        
        # Find the nearest point in B for each point in A
        distances, indices = tree.query(A)
            
        # If you need the indices as a list
        mapping = list(indices)

        mapping = np.array(mapping)

        return mapping


def read_vtu_from_sparta(filename, assembly_item, assembly_coords, idx_inv,  it):

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


    # Initializes the Aerothermo object with empty arrays of the right size
    aerothermo = assembly.Aerothermo(len(assembly_item.mesh.facets))
    
    # Open the VTK solution file and extract point coordinates
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    npoints = reader.GetNumberOfPoints()
    narrays = reader.GetNumberOfPointArrays()    
    coords = vtk_to_numpy(reader.GetOutput().GetPoints().GetData())
    data = reader.GetOutput().GetCellData()

    # Converts output cell data to point data for TITAN 
    # converter = vtk.vtkCellDataToPointData()
    # converter.ProcessAllArraysOn()
    # converter.SetInputConnection(reader.GetOutputPort())
    # converter.Update()
    # data = converter.GetOutput().GetCellData() #.GetPointData()

    vtk_cell_centers = vtk.vtkCellCenters()
    vtk_cell_centers.SetInputData(reader.GetOutput())
    vtk_cell_centers.Update()
    vtk_cell_centers_data = vtk_cell_centers.GetOutput()
    vtk_COG = vtk_to_numpy(vtk_cell_centers_data.GetPoints().GetData())
    
    mapping = mapping_facet_COG(assembly_item.mesh.facet_COG, vtk_COG)

    # Identify unique node coordinates and get indices for sorting
    # coords_sorted , idx_sim= np.unique(coords, axis = 0, return_index = True)

    # Extract SPARTA surface quantities by field name (SPARTA naming convention)
    array_P_name = f'f_10{it}[5]'  # Pressure
    array_Sx_name = f'f_10{it}[6]'  # Shear x
    array_H_name = f'f_10{it}[9]'  # Total energy (etot)

    # Assign fields to Aerothermo object using sorted and mapped indexing
    aerothermo.pressure = vtk_to_numpy(data.GetArray(array_P_name))[mapping]
    aerothermo.heatflux = vtk_to_numpy(data.GetArray(array_H_name))[mapping]
    aerothermo.shear[:,0] = vtk_to_numpy(data.GetArray(array_Sx_name))[mapping]
    aerothermo.shear[:,1] = vtk_to_numpy(data.GetArray(f'f_10{str(it)}[7]'))[mapping]
    aerothermo.shear[:,2] = vtk_to_numpy(data.GetArray(f'f_10{str(it)}[8]'))[mapping]

    assembly_item.aerothermo.pressure = aerothermo.pressure
    assembly_item.aerothermo.heatflux = aerothermo.heatflux
    assembly_item.aerothermo.shear[:,0] = aerothermo.shear[:,0] 
    assembly_item.aerothermo.shear[:,1] = aerothermo.shear[:,1] 
    assembly_item.aerothermo.shear[:,2] = aerothermo.shear[:,2] 

    assembly_item.aerothermo_cfd = aerothermo

    return aerothermo



def get_env_path(env_name):
    try:
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c", "import os; print(os.environ['CONDA_PREFIX'])"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("Error:", e.stderr)
        return None


def PostProcess_SPARTA(dsmc, assembly_list, output_folder = '', surface_filename = ['assembly_0_cluster_0'], grid_filename = 'grid_cluster_0.txt'):

    """
    Function to postprocess the SPARTA simulation surface output

    Parameters
    ----------
    dsmc: dsmc data
        Object of class dsmc
    assembly:List_Assembly
        Object of class List_Assembly
    """

    # Gather all unique surface nodes across assemblies
    assembly_nodes = np.array([])
    assembly_facets = np.array([], dtype = int)

    for assembly in assembly_list:
        nodes = assembly.cfd_mesh.nodes[assembly.cfd_mesh.edges]
        nodes.shape = (-1,3)
        nodes = np.unique(nodes, axis = 0)
        assembly_nodes = np.append(assembly_nodes,nodes)

    assembly_nodes.shape = (-1,3)
    assembly_nodes, idx_inv = np.unique(assembly_nodes, axis = 0, return_inverse = True)

    
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    env = {
        "PATH": f"{dsmc.paraview_conda_path}/bin",
        "LD_LIBRARY_PATH": f"{dsmc.paraview_conda_path}/lib",
        "PYTHONNOUSERSITE": "1",  # Avoid loading from user site-packages
        "PYTHONPATH": "",         # Kill path bleed
        "CONDA_PREFIX": dsmc.paraview_conda_path,
        "HOME": os.environ["HOME"],}

    pvpython = f"{dsmc.paraview_conda_path}/bin/pvpython"


    # Postprocess each assembly's surface data using Paraview script (surface postprocess only)
    if dsmc.grid_results: 

        basename = f"grid_{grid_filename}"
        
        args = [
            pvpython,
            f"{path}/Executables/grid2paraview.py",
            f"{path}/{output_folder}/DSMC_Grid/grid.txt",
            basename,
            "-r", f"{path}/{output_folder}/DSMC_Grid/results.grid_{grid_filename}.*"]

        subprocess.run(args, env=env, cwd = f'{path}/{output_folder}/DSMC_Grid')


    for it, assembly in enumerate(assembly_list):

        
        basename = f"surf_{surface_filename[it]}"

        args = [
            pvpython,
            f"{path}/Executables/surf2paraview.py",
            f"{path}/{output_folder}/DSMC_sol/compound_{surface_filename[it]}.data",
            basename,
            "-r", f"{path}/{output_folder}/DSMC_sol/results.surf_{surface_filename[it]}.*"]

        subprocess.run(args, env=env, cwd = f'{path}/{output_folder}/DSMC_sol')


    # Grid post-processing can be heavy and hence, it is skipped inside TITAN loop
    print('SPARTA-DSMC simulation: DONE')

    # Determine most recent timestep result for each assembly
    for i, _assembly in enumerate(assembly_list):

        list_of_files = glob.glob(f'{output_folder}/DSMC_sol/surf_{surface_filename[i]}/*') # * means all if need specific format then *.csv
        filename = max(list_of_files, key=os.path.getmtime)

        # Read each individual assembly (written to separate files by SPARTA)
        read_vtu_from_sparta(filename, _assembly, assembly_nodes, idx_inv,  i)

        # Consider deleting the excess files here and saving only the latest surface result for future reference

    print('Post-processing of SPARTA-DSMC simulation is completed!')


def write_postprocess_gridtxt(dsmc, output_folder, grid_filename):

    """
    Function to postprocess the SPARTA simulation output

    Parameters
    ----------
    dsmc: dsmc data
        Object of class dsmc
    NOTE: We aren't postprocessing grid results because of computational cost so not actively used. 
    """

    # check working directory definition
    with open(f"{output_folder}/DSMC_Grid/grid.txt", 'w') as f:

        print(dsmc.domain)
        xlo, xhi = dsmc.domain[0]
        ylo, yhi = dsmc.domain[1]
        zlo, zhi = dsmc.domain[2]

        f.write('dimension\t\t 3\n')
        f.write(f'create_box\t\t{xlo} {xhi}\t{ylo} {yhi}\t{zlo} {zhi}\n')
        f.write(f'read_grid\t\t {output_folder}/DSMC_Grid/dynamic_adapted_{grid_filename}\n')

    f.close()


def compute_dsmc_aerothermo(titan, options, cluster_tag = 0):

    """
    Function to set-up, execute and postprocess SPARTA DSMC simulation

    Parameters
    ----------
    options: Options
        Object of class Options
    assembly:List_Assembly
        Object of class List_Assembly
    """

    assembly_list = titan.assembly

    iteration = options.current_iter
    iteration = 0
    dsmc = options.dsmc 


    # --- Define naming convention of files/folders for SPARTA simulation ---
    surface_filelist = []
    for it, assembly in enumerate(assembly_list):

        # Write filename for geometries (consider making this an input instead of dsmc.surf_name)
        surface_filelist.append(f'assembly_{it}_cluster_{cluster_tag}_iter_{iteration}')

    config_filename = f"in.config_cluster_{cluster_tag}_iter_{iteration}"
    grid_filename = f"grid_cluster_{cluster_tag}_iter_{iteration}.txt"
    log_filename = f"sparta_log_cluster_{cluster_tag}_iter_{iteration}.log"

    print('CONFIG_NAME = ', config_filename)
    # --- Find the lowest-altitude assembly (used for simulation settings) ---
    lowest_alt = 1E10
    for idx, assembly in enumerate(assembly_list):
        if assembly.trajectory.altitude < lowest_alt:
            lowest_alt = assembly.trajectory.altitude
            it = idx
            lref = assembly.Lref  # Not currently used

    free = assembly_list[it].freestream

    
    for i, assembly in enumerate(assembly_list):

        mesh = trimesh.Trimesh()
        for obj in assembly.objects:
            mesh += trimesh.Trimesh(vertices = obj.mesh.nodes, faces = obj.mesh.facets) 

        COG = np.round(np.sum(mesh.vertices[mesh.faces], axis = 1)/3,5)

        faces_tuple = [tuple(f) for f in COG]
        count_faces_dict = pd.Series(faces_tuple).value_counts()
        mask = [count_faces_dict[f] == 1 for f in faces_tuple]

        mesh = trimesh.Trimesh(vertices = mesh.vertices, faces = mesh.faces[mask])

        assembly.cfd_mesh.nodes = mesh.vertices
        assembly.cfd_mesh.facets = mesh.faces
        assembly.cfd_mesh.edges, assembly.cfd_mesh.facet_edges = Mesh.map_edges_connectivity(assembly.cfd_mesh.facets)
        #print(assembly.cfd_mesh.facets.shape)


    # --- Convert to wind frame (ECEF → Wind) centered on lowest object ---
    # assembly_windframe = deepcopy(assembly_list)
    assembly_windframe = Assembly.copy_assembly(assembly_list, options)

    ref_pos = assembly_list[it].position

    for i, assembly in enumerate(assembly_windframe):

        # Rotate from body frame to ECEF
        R_B_ECEF = Rot.from_quat(assembly.quaternion)
        assembly.cfd_mesh.nodes -= assembly.COG
        assembly.cfd_mesh.nodes = R_B_ECEF.apply(assembly.cfd_mesh.nodes)

        # Translate to common center (lowest assembly position)
        assembly.cfd_mesh.nodes += np.array(assembly.position - ref_pos)

        # Rotate from ECEF → NED → Wind frame
        R_ECEF_NED = frames.R_NED_ECEF(lat=assembly.trajectory.latitude, lon=assembly.trajectory.longitude).inv()
        R_NED_W = frames.R_W_NED(ha=assembly.trajectory.chi, fpa=assembly.trajectory.gamma).inv()
        R_ECEF_W = R_NED_W * R_ECEF_NED

        assembly.cfd_mesh.nodes = R_ECEF_W.apply(assembly.cfd_mesh.nodes)

        # Update mesh bounding box
        assembly.cfd_mesh.xmin = np.min(assembly.cfd_mesh.nodes, axis=0)
        assembly.cfd_mesh.xmax = np.max(assembly.cfd_mesh.nodes, axis=0)


    if options.current_iter%options.save_freq == 0:
        options.save_state(titan, titan.iter)     


    
    # Check to see if SPARTA configuration file has been written - if not - then proceed
    expected_file = options.output_folder + '/DSMC_sol/' + config_filename

    if not os.path.exists(expected_file):

        # --- Generate input files ---
        # Create sparta geometry for each assembly in the simulation:
        create_sparta_geo(options, dsmc, assembly_list, output_folder = options.output_folder, surface_filename = surface_filelist)

        # Calculate DSMC properties needed based off of altitude of lowest object:
        compute_dsmc_properties(dsmc, free, assembly_list)

        # Creat configuration file for sparta
        write_sparta_config(options, dsmc, free, assembly_list, output_folder = options.output_folder, sparta_config = config_filename, 
                            surface_filename = surface_filelist, grid_filename = grid_filename)

        if dsmc.grid_results: 

            write_postprocess_gridtxt(dsmc, output_folder, grid_filename)

        # if the mode is automatic we proceed with the SPARTA simulation within the TITAN framework
        if options.dsmc.mode == 'automatic':

            # --- Run SPARTA ---

            # Run SPARTA simulation:
            run_SPARTA(options, dsmc, assembly_list, output_folder = options.output_folder, sparta_config = config_filename, sparta_log = log_filename)

        # Otherwise we save the TITAN state here for the previous iteration (when we re-enter the simulation it will run initial trajectory again)
        else:

            # save titan state here (previous state)
            print(iteration)
            options.save_state(titan, iteration-1)

            print('---------------------------------------------------------')
            print(f"[INFO] SPARTA is operating in manual mode. Input files for timestep {iteration} have been generated:\n"
                    f"        Geometry files: {', '.join([name + '.data' for name in surface_filelist])}\n"
                    f"        Configuration file: {config_filename}\n"
                    f"        These files can be found in: {options.output_folder + '/DSMC_sol'}\n\n"
                    f"        Please run the SPARTA simulation externally using these files.\n"
                    f"        Once complete, place the results files (e.g., "
                    f"{', '.join(['results.surf_' + name + '.*' for name in surface_filelist])})\n"
                    f"        and written geometry files (e.g., "
                    f"{', '.join(['compound_' + name + '.data' for name in surface_filelist])}) in:\n"
                    f"        {options.output_folder + '/DSMC_sol'}\n\n"
                    f"        To continue the TITAN simulation, restart with:\n"
                    f"        Load_Mesh = True, Load_State = True")
            print('---------------------------------------------------------')

            exit()


    # --- Postprocess surface output ---

    # Post process following simulation:
    PostProcess_SPARTA(dsmc, assembly_list, output_folder = options.output_folder, surface_filename = surface_filelist, grid_filename = grid_filename)

    # if we are running in 'manual' mode then we need to check to see if the solution file exists in the expected location
    # otherwise we run generation of associated files for simulation and give message to user

    # we need to save state of titan simulation to restart with 

    # we check to see if the file is there - if it is then we postprocess 

