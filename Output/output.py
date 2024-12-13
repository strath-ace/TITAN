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
import pandas as pd
import numpy as np
import os
import meshio
from pathlib import Path

def write_output_data(titan, options):

    n_assembly = len(titan.assembly)

    if n_assembly == 1:
    
        width_max = titan.assembly[0].Lref
    else:
    
        width = np.zeros((n_assembly, n_assembly))
        for i in range(n_assembly):
            for j in range(n_assembly):
                width_vector = np.subtract(titan.assembly[i].position,titan.assembly[j].position)
                width[i,j] = np.linalg.norm(width_vector)

        width_max = max([max(l) for l in width])      


    df = pd.DataFrame()
    df['Time'] = [titan.time]
    df['Width'] = [width_max]
    df = df.round(decimals = 12)
    df.to_csv(options.output_folder + '/Data/'+ 'data_width.csv', mode='a' ,header=not os.path.exists(options.output_folder + '/Data/data_width.csv'), index = False)

    
    df = pd.DataFrame()

    for assembly in titan.assembly:

        df['Time'] = [titan.time]
        df['Iter'] = [titan.iter]
        df['Assembly_ID']   = [assembly.id]
        df['Mass'] = [assembly.mass]

        #Trajectory Details
        df['Altitude']       = [assembly.trajectory.altitude]
        df['Distance']       = [assembly.distance_travelled]
        df['Velocity']       = [assembly.trajectory.velocity]
        df['FlightPathAngle'] = [assembly.trajectory.gamma*180/np.pi]
        df['HeadingAngle']   = [assembly.trajectory.chi*180/np.pi]
        df['Latitude']       = [assembly.trajectory.latitude*180/np.pi]
        df['Longitude']      = [assembly.trajectory.longitude*180/np.pi]
        df['AngleAttack']   =  [assembly.aoa*180/np.pi]
        df['AngleSideslip'] =  [assembly.slip*180/np.pi]

        #Position and Velocity in the ECEF frame
        df['ECEF_X']  = [assembly.position[0]]
        df['ECEF_Y']  = [assembly.position[1]]
        df['ECEF_Z']  = [assembly.position[2]]
        df['ECEF_vU'] = [assembly.velocity[0]]
        df['ECEF_vV'] = [assembly.velocity[1]]
        df['ECEF_vW'] = [assembly.velocity[2]]

        #Center of mass position in the Body Frame
        df['BODY_COM_X']  = [assembly.COG[0]]
        df['BODY_COM_Y']  = [assembly.COG[1]]
        df['BODY_COM_Z']  = [assembly.COG[2]]

        #Forces and Moments in the Body frame
        df['Aero_Fx_B'] = [assembly.body_force.force[0]]
        df['Aero_Fy_B'] = [assembly.body_force.force[1]]
        df['Aero_Fz_B'] = [assembly.body_force.force[2]]
        df['Aero_Mx_B'] = [assembly.body_force.moment[0]]
        df['Aero_My_B'] = [assembly.body_force.moment[1]]
        df['Aero_Mz_B'] = [assembly.body_force.moment[2]]
    
        #Forces in the Wind Frame
        df['Lift'] =          [assembly.wind_force.lift]
        df['Drag'] =          [assembly.wind_force.drag]
        df['Crosswind'] =     [assembly.wind_force.crosswind]

        #Inertial properties
        df['Mass'] = [assembly.mass]
        df['Inertia_xx'] = [assembly.inertia[0,0]]
        df['Inertia_xy'] = [assembly.inertia[0,1]]
        df['Inertia_xz'] = [assembly.inertia[0,2]]
        df['Inertia_yy'] = [assembly.inertia[1,1]]
        df['Inertia_yz'] = [assembly.inertia[1,2]]
        df['Inertia_zz'] = [assembly.inertia[2,2]]

        #Attitude properties
        df['Roll'] =     [assembly.roll*180/np.pi]
        df['Pitch'] =    [assembly.pitch*180/np.pi]
        df['Yaw'] =      [assembly.yaw*180/np.pi]
        df['VelRoll'] =  [assembly.roll_vel]
        df['VelPitch'] = [assembly.pitch_vel]
        df['VelYaw'] =   [assembly.yaw_vel]

        #Quaternion Body -> ECEF frame        
        df['Quat_w']   = [assembly.quaternion[3]]
        df['Quat_x']   = [assembly.quaternion[0]]
        df['Quat_y']   = [assembly.quaternion[1]]
        df['Quat_z']   = [assembly.quaternion[2]]

        #Quaternion Body -> ECEF frame of the previous iteration, for emissions post-processing   
        df['Quat_prev_w']   = [assembly.quaternion_prev[3]]
        df['Quat_prev_x']   = [assembly.quaternion_prev[0]]
        df['Quat_prev_y']   = [assembly.quaternion_prev[1]]
        df['Quat_prev_z']   = [assembly.quaternion_prev[2]]

        #Freestream properties
        df['Mach'] = [assembly.freestream.mach]
        df['Speedsound'] = [assembly.freestream.sound]
        df['Density'] = [assembly.freestream.density]
        df['Temperature'] = [assembly.freestream.temperature]
        df['Pressure'] = [assembly.freestream.pressure]
        df['SpecificHeatRatio'] = [assembly.freestream.gamma]

        for specie, pct in zip(assembly.freestream.species_index, assembly.freestream.percent_mass[0]) :
            df[specie+"_mass_pct"] = [pct]

        #Stagnation properties
        try:
            df['Qstag'] = [assembly.aerothermo.qconvstag]
            df['Qradstag'] = [assembly.aerothermo.qradstag]

#            df['Qconvstag'] = [assembly.aerothermo.qstag]
#            df['Qradstag'] = [assembly.aerothermo.qradstag]
#            df['Qstag'] = [assembly.aerothermo.qradstag]
        except:
            pass

        try:
            df['Pstag'] = [assembly.freestream.P1_s]
            df['Tstag'] = [assembly.freestream.T1_s]
            df['Rhostag'] = [assembly.freestream.rho_s]
        except:
            pass

        #Reference Dimensionsal constants
        df["Aref"] = [assembly.Aref]
        df["Lref"] = [assembly.Lref]

        df = df.round(decimals = 12)
        df.to_csv(options.output_folder + '/Data/'+ 'data.csv', mode='a' ,header=not os.path.exists(options.output_folder + '/Data/data.csv'), index = False)

    df = pd.DataFrame()
    for assembly in titan.assembly:
        df["Time"] = [titan.time]
        df["Iter"] = [titan.iter]
        df["Assembly_ID"] = [assembly.id]
        for obj in assembly.objects:
            df["Obj_name"] = [obj.name]
            #df["Temperature"] = [obj.temperature]
            df["Density"] = [obj.material.density]
            df["Photons_second"] = [obj.photons]
            df["Mass"] = [obj.mass]
            if options.pato.flag:
                df["MaxTemperature"] = [max(obj.pato.temperature)]
                print('obj:', obj.global_ID, ' max temp:', max(obj.pato.temperature))
            df["Max_stress"] = [obj.max_stress]
            df["Yield_stress"] = [obj.yield_stress]
            df["Parent_id"] = [obj.parent_id]
            #df["Parent_part"] = [obj.parent_part]
            
            df = df.round(decimals = 6)
            df.to_csv(options.output_folder + '/Data/'+ 'data_assembly.csv', mode='a' ,header=not os.path.exists(options.output_folder + '/Data/data_assembly.csv'), index = False)

def generate_surface_solution(titan, options, iter_value, folder = 'Surface_solution'):
    points = np.array([])
    facets = np.array([])
    pressure = np.array([])
    shear = np.array([])
    heatflux = np.array([])
    hf_cond = np.array([])
    radius = np.array([])
    ellipse = np.array([])
    cellID = np.array([])
    emissive_power = np.array([])
    theta = np.array([])
    he = np.array([])
    hw = np.array([])
    Te = np.array([])
    mDotVapor = np.array([])
    mDotMelt = np.array([])


    for assembly in titan.assembly:
        points = assembly.mesh.nodes - assembly.mesh.surface_displacement
        facets = assembly.mesh.facets
        pressure = assembly.aerothermo.pressure
        heatflux = assembly.aerothermo.heatflux
        shear = assembly.aerothermo.shear
        displacement = assembly.mesh.surface_displacement
        radius = assembly.mesh.facet_radius
        ellipse = assembly.inside_shock
        temperature  = assembly.aerothermo.temperature
        emissive_power = assembly.emissive_power
        theta = assembly.aerothermo.theta
        he = assembly.aerothermo.he
        hw = assembly.aerothermo.hw
        Te = assembly.aerothermo.Te
        mDotVapor = np.zeros(len(assembly.mesh.facets))
        mDotMelt  = np.zeros(len(assembly.mesh.facets))
        if options.thermal.ablation_mode.lower() == 'pato' and options.pato.Ta_bc == 'ablation':
            mDotVapor = assembly.mDotVapor
            mDotMelt = assembly.mDotMelt
        #hf_cond = assembly.hf_cond

        for cellid in range(len(assembly.mesh.facets)):
            cellID = np.append(cellID, cellid)

        
        cells = {"triangle": facets}

        cell_data = { "Pressure": [pressure],
                      "Heatflux": [heatflux],
                      "Temperature": [temperature],
                      #"Shear": [shear],
                      #"Radius": [radius],
                      #"CellID": [cellID], #uncommenting this actually crashes the code
                      "Emissive power": [emissive_power],
                      #"Heat conduction": [hf_cond],
                      "Theta": [theta],
                      "He": [he],
                      "Hw": [hw],
                      "Te": [Te],
                      "mDotVapor": [mDotVapor],
                      "mDotMelt": [mDotMelt],
                    }

        point_data = { "Displacement": displacement,
                      # "Ellipse": ellipse,
                     }

        trimesh = meshio.Mesh(points,
                              cells=cells,
                              point_data = point_data,
                              cell_data = cell_data)

        folder_path = options.output_folder+'/' + folder + '/ID_'+str(assembly.id)
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        vol_mesh_filepath = f"{folder_path}/solution_iter_{str(iter_value).zfill(3)}.xdmf"
        meshio.write(vol_mesh_filepath, trimesh, file_format="xdmf")

def generate_surface_solution_emissions(titan, options, iter_value, folder = 'Surface_solution'):

    points = np.array([])
    facets = np.array([])
    cellID = np.array([])
    heatflux = np.array([])
    temperature = np.array([])
    temperature_gas = np.array([])
    blackbody_emissions_OI_surf  = np.array([])
    blackbody_emissions_AlI_surf = np.array([])
    atomic_emissions_OI_surf     = np.array([])
    atomic_emissions_AlI_surf    = np.array([])


    for assembly in titan.assembly:
        points = assembly.mesh.nodes - assembly.mesh.surface_displacement
        facets = assembly.mesh.facets
        heatflux = assembly.aerothermo.heatflux
        temperature  = assembly.aerothermo.temperature
        temperature_gas  = assembly.aerothermo.Te
        blackbody_emissions_OI_surf   = assembly.blackbody_emissions_OI_surf
        blackbody_emissions_AlI_surf  = assembly.blackbody_emissions_AlI_surf
        atomic_emissions_OI_surf      = assembly.atomic_emissions_OI_surf
        atomic_emissions_AlI_surf     = assembly.atomic_emissions_AlI_surf

        for cellid in range(len(assembly.mesh.facets)):
            cellID = np.append(cellID, cellid)

        
        cells = {"triangle": facets}

        cell_data = { "Heatflux":                    [heatflux],
                      "Temperature":                 [temperature],
                      "Temperature equilibrium gas": [temperature_gas],
                      "blackbody_emissions_OI":  [blackbody_emissions_OI_surf],
                      "blackbody_emissions_AlI": [blackbody_emissions_AlI_surf],
                      "atomic_emissions_OI":     [atomic_emissions_OI_surf],
                      "atomic_emissions_AlI":    [atomic_emissions_AlI_surf],
                    }

        trimesh = meshio.Mesh(points,
                              cells=cells,
                              cell_data = cell_data)

        folder_path = options.output_folder+'/' + folder + '/ID_'+str(assembly.id)
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        vol_mesh_filepath = f"{folder_path}/solution_iter_{str(iter_value).zfill(3)}.xdmf"
        meshio.write(vol_mesh_filepath, trimesh, file_format="xdmf")


def generate_surface_solution_object(obj, quantity, options, iter_value, folder = 'Surface_solution'):

    points = np.array([])
    facets = np.array([])

    points = obj.mesh.nodes
    facets = obj.mesh.facets
    
    cells = {"triangle": facets}

    cell_data = { "quantity": [quantity],
                }

    trimesh = meshio.Mesh(points,
                          cells=cells,
                          cell_data = cell_data)

    folder_path = options.output_folder+'/' + folder
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    vol_mesh_filepath = f"{folder_path}/obj_{str(obj.global_ID)}_solution_iter_{str(iter_value).zfill(3)}.xdmf"
    meshio.write(vol_mesh_filepath, trimesh, file_format="xdmf")

#Generate volume for FENICS
def generate_volume(titan, options):
    for assembly in titan.assembly: 

        cells = [
            ("tetra", assembly.mesh.vol_elements) ]

        trimesh = meshio.Mesh(
            assembly.mesh.vol_coords,
            cells={"tetra": assembly.mesh.vol_elements},
            cell_data={"Vol_tags": [assembly.mesh.vol_tag]},
        )
        
        folder_path = options.output_folder+'/Surface_solution/ID_'+str(assembly.id)
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        vol_mesh_filepath = f"{folder_path}/volume.xdmf"        
        meshio.write(vol_mesh_filepath, trimesh, file_format = "xdmf")

# Show DIsplacement and Von Mises for 3D mesh (Not surface mesh)
def generate_volume_solution(titan, options):
    points = np.array([])
    tetra = np.array([])
    displacement = np.array([])
    vonMises = np.array([])

    for assembly in titan.assembly:
        points = assembly.mesh.vol_coords - assembly.mesh.volume_displacement
        tetra = assembly.mesh.vol_elements
        displacement = assembly.mesh.volume_displacement
        
        try:
            vonMises = assembly.mesh.volume_vonMises
        except:
            vonMises = np.zeros(len(assembly.mesh.vol_elements))

        cells = {"tetra": tetra}

        point_data = {"Displacement": displacement}

        cell_data = {"VonMises": [vonMises]}

        trimesh = meshio.Mesh(points,
                              cells=cells,
                              point_data = point_data,
                              cell_data = cell_data)

        folder_path = options.output_folder+'/Volume_solution/ID_'+str(assembly.id)
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        vol_mesh_filepath = f"{folder_path}/volume_iter_{str(titan.iter).zfill(3)}.vtk"
        meshio.write(vol_mesh_filepath, trimesh, file_format="vtk")

def TITAN_information():
    print(f"""                                                                                                                   
       ________  ______  ________   ______   __    __ 
      /        |/      |/        | /      \ /  \  /  |
      $$$$$$$$/ $$$$$$/ $$$$$$$$/ /$$$$$$  |$$  \ $$ |
         $$ |     $$ |     $$ |   $$ |__$$ |$$$  \$$ |
         $$ |     $$ |     $$ |   $$    $$ |$$$$  $$ |
         $$ |     $$ |     $$ |   $$$$$$$$ |$$ $$ $$ |
         $$ |    _$$ |_    $$ |   $$ |  $$ |$$ |$$$$ |
         $$ |   / $$   |   $$ |   $$ |  $$ |$$ | $$$ |
         $$/    $$$$$$/    $$/    $$/   $$/ $$/   $$/                                                                                                             
    """)

    print(f"""
        ###############################################
        # TITAN tool is still under heavy development #
        ###############################################

        Authors: Fábio Morgado, Julie Graham, Sai Peddakotla, Catarina Garbacz, Marco Fossati and contributors
        Contact: fabio.pereira-morgado@strath.ac.uk
        Github:  https://github.com/strath-ace/TITAN
        Version: 0.1
        Release date: 2 February 2023
        """)

def options_information(options):
    print(f"""
        ##########################
        # Simulation Information #
        ##########################

        Output folder: {options.output_folder}
        Maximum number of iterations: {options.iters}
        Fidelity level: {options.fidelity}
        Structural dynamics flag: {options.structural_dynamics}
        Ablation flag: {options.thermal.ablation}  
        Time-step: {options.dynamics.time_step}
        Planet: {options.planet.name.upper()}

        ##########################
        # Freestream Information #
        ##########################

        Method for freestream computation: {options.freestream.method}
        Atmospheric model: {options.freestream.model}
           """)
        
def iteration(titan, options):
    print(f"""  Iteration {titan.iter+1} of {options.iters} """)