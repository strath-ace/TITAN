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
"""output module."""
import pandas as pd
import numpy as np
import os
import meshio
from pathlib import Path
from scipy.spatial.transform import Rotation
from ..Dynamics.frames import R_ECEF_from_ECI
from datetime import timedelta


def write_output_data(titan, options, smooth=False):
    """Append simulation state and assembly properties to output data files.

:param titan: TITAN simulation object.
:type titan: object
:param options: Options or configuration object.
:type options: Options
:param smooth: Whether the output should be marked as smoothed data.
:type smooth: Any"""

    df = pd.DataFrame()
    data = {}
    for assembly in titan.assembly:

        data['Time'] = [titan.time]
        data['Iter'] = [titan.iter]
        data['Assembly_ID']   = [assembly.id]
        data['Mass'] = [assembly.mass]

        #Trajectory Details
        data['Altitude']       = [assembly.trajectory.altitude]
        data['Distance']       = [assembly.distance_travelled]
        data['Velocity']       = [assembly.trajectory.velocity]
        data['FlightPathAngle'] = [assembly.trajectory.gamma*180/np.pi]
        data['HeadingAngle']   = [assembly.trajectory.chi*180/np.pi]
        data['Latitude']       = [assembly.trajectory.latitude*180/np.pi]
        data['Longitude']      = [assembly.trajectory.longitude*180/np.pi]
        data['AngleAttack']   =  [assembly.aoa*180/np.pi]
        data['AngleSideslip'] =  [assembly.slip*180/np.pi]

        #Position and Velocity in the ECEF frame
        data['ECEF_X']  = [assembly.position[0]]
        data['ECEF_Y']  = [assembly.position[1]]
        data['ECEF_Z']  = [assembly.position[2]]
        data['ECEF_vU'] = [assembly.velocity[0]]
        data['ECEF_vV'] = [assembly.velocity[1]]
        data['ECEF_vW'] = [assembly.velocity[2]]

        #Center of mass position in the Body Frame
        data['BODY_COM_X']  = [assembly.COG[0]]
        data['BODY_COM_Y']  = [assembly.COG[1]]
        data['BODY_COM_Z']  = [assembly.COG[2]]

        #Forces and Moments in the Body frame
        data['Aero_Fx_B'] = [assembly.body_force.force[0]]
        data['Aero_Fy_B'] = [assembly.body_force.force[1]]
        data['Aero_Fz_B'] = [assembly.body_force.force[2]]
        data['Aero_Mx_B'] = [assembly.body_force.moment[0]]
        data['Aero_My_B'] = [assembly.body_force.moment[1]]
        data['Aero_Mz_B'] = [assembly.body_force.moment[2]]
    
        #Forces in the Wind Frame
        data['Lift'] =          [assembly.wind_force.lift]
        data['Drag'] =          [assembly.wind_force.drag]
        data['Crosswind'] =     [assembly.wind_force.crosswind]

        #Inertial properties
        data['Mass'] = [assembly.mass]
        data['Inertia_xx'] = [assembly.inertia[0,0]]
        data['Inertia_xy'] = [assembly.inertia[0,1]]
        data['Inertia_xz'] = [assembly.inertia[0,2]]
        data['Inertia_yy'] = [assembly.inertia[1,1]]
        data['Inertia_yz'] = [assembly.inertia[1,2]]
        data['Inertia_zz'] = [assembly.inertia[2,2]]
        data['Kinetic_energy'] = assembly.mass * assembly.trajectory.velocity**2

        #Attitude properties
        data['Roll'] =     [assembly.roll*180/np.pi]
        data['Pitch'] =    [assembly.pitch*180/np.pi]
        data['Yaw'] =      [assembly.yaw*180/np.pi]
        data['distRoll'] = [assembly.unmodded_angles[0]*180/np.pi]
        data['distPitch'] = [assembly.unmodded_angles[1]*180/np.pi]
        data['distYaw'] = [assembly.unmodded_angles[2]*180/np.pi] 
        data['VelRoll'] =  [assembly.roll_vel*180/np.pi]
        data['VelPitch'] = [assembly.pitch_vel*180/np.pi]
        data['VelYaw'] =   [assembly.yaw_vel*180/np.pi]
        
        omega= np.array([assembly.roll_vel,assembly.pitch_vel,assembly.yaw_vel])
        angular_momentum = assembly.inertia @ omega

        data['magnitudeOmega'] = np.linalg.norm(omega)*180/np.pi
        data['angularMomentum_x'] = angular_momentum[0]
        data['angularMomentum_y'] = angular_momentum[1]
        data['angularMomentum_z'] = angular_momentum[2]
        data['magnitudeAngularMomentum'] = np.linalg.norm(angular_momentum)

        #Quaternion Body -> ECEF frame        
        data['Quat_w']   = [assembly.quaternion[3]]
        data['Quat_x']   = [assembly.quaternion[0]]
        data['Quat_y']   = [assembly.quaternion[1]]
        data['Quat_z']   = [assembly.quaternion[2]]

        #Quaternion Body -> ECEF frame of the previous iteration, for emissions post-processing   
        # data['Quat_prev_w']   = [assembly.quaternion_prev[3]]
        # data['Quat_prev_x']   = [assembly.quaternion_prev[0]]
        # data['Quat_prev_y']   = [assembly.quaternion_prev[1]]
        # data['Quat_prev_z']   = [assembly.quaternion_prev[2]]

        #Freestream properties
        data['Mach'] = [assembly.freestream.mach]
        data['Speedsound'] = [assembly.freestream.sound]
        data['Density'] = [assembly.freestream.density]
        data['Temperature'] = [assembly.freestream.temperature]
        data['Pressure'] = [assembly.freestream.pressure]
        data['SpecificHeatRatio'] = [assembly.freestream.gamma]
        data['Qint'] = [np.sum(assembly.aerothermo.heatflux*assembly.mesh.facet_area)]
        data['Qmax'] = [max(assembly.aerothermo.heatflux)]
        data['Tmax'] = [max(assembly.aerothermo.temperature)]
        data['knudsen'] = [assembly.freestream.knudsen]
        dyn_pressure = 0.5 * assembly.freestream.density * assembly.trajectory.velocity**2
        if assembly.Aref is not None:
            data['C_D'] = assembly.wind_force.drag / (dyn_pressure * assembly.Aref)
            data['C_L'] = assembly.wind_force.lift / (dyn_pressure * assembly.Aref)
            data['C_Xwind'] = assembly.wind_force.crosswind / (dyn_pressure * assembly.Aref)
            if options.aerothermo.recalculate_Aref:
                data['Aproj'] = assembly.aerothermo.proj_area
                data['C_D_instant'] = assembly.wind_force.drag / (dyn_pressure * assembly.aerothermo.proj_area)
                data['C_L_instant'] = assembly.wind_force.lift / (dyn_pressure * assembly.aerothermo.proj_area)
                data['C_Xwind_instant'] = assembly.wind_force.crosswind / (dyn_pressure * assembly.aerothermo.proj_area)

        if options.write_inertial_output:
            R_ECI_from_B =  R_ECEF_from_ECI(titan.epoch + timedelta(seconds=titan.time)).inv() * Rotation.from_quat(assembly.quaternion)
            ECI_quat = R_ECI_from_B.as_quat()

            data['Quat_w_ECI'] = [ECI_quat[3]]
            data['Quat_x_ECI'] = [ECI_quat[0]]
            data['Quat_y_ECI'] = [ECI_quat[1]]
            data['Quat_z_ECI'] = [ECI_quat[2]]

            ECI_Aero_F = R_ECI_from_B.apply(assembly.body_force.force)
            data['Aero_Fx_ECI'] = [ECI_Aero_F[0]]
            data['Aero_Fy_ECI'] = [ECI_Aero_F[1]]
            data['Aero_Fz_ECI'] = [ECI_Aero_F[2]]
            yaw_ECI, pitch_ECI, roll_ECI = R_ECI_from_B.as_euler('ZYX',degrees=True)
            data['Roll_ECI'] = [roll_ECI]
            data['Pitch_ECI'] = [pitch_ECI]
            data['Yaw_ECI'] = [yaw_ECI]

        data['gravgrad_x'] = [assembly.body_force.gravgrad_moment[0]]
        data['gravgrad_y'] = [assembly.body_force.gravgrad_moment[1]]
        data['gravgrad_z'] = [assembly.body_force.gravgrad_moment[2]]
        
        for specie, pct in zip(assembly.freestream.species_index, assembly.freestream.mass_fraction[0]) :
            data[specie+"_mass_pct"] = [pct]
        if options.thermal.ablation_mode=='byproducts':
            just_emissions = True #TODO turn this into a proper option
            columns = ['Time','Iter','Assembly_ID', 'Mass', 'Tmax', 'Altitude', 'mDotMelt']
            [columns.append('emission_'+species) for species in list(assembly.byproducts.species)]
            if not just_emissions:
                [columns.append('mass_'+species) for species in list(assembly.byproducts.species)]
                [columns.append('mean_rho_'+species) for species in list(assembly.byproducts.species)]
                [columns.append('mean_mf_'+species) for species in list(assembly.byproducts.species)]
            data_by = [titan.time, titan.iter, assembly.id, assembly.mass, max(assembly.aerothermo.temperature), 
                    assembly.trajectory.altitude, np.sum(assembly.mDotMelt)*titan.delta_t]
            [data_by.append(np.sum(assembly.byproducts.emission[species])) for species in list(assembly.byproducts.species)]

            if not just_emissions:
                [data_by.append(np.sum(assembly.byproducts.mass[species])) for species in list(assembly.byproducts.species)]
                [data_by.append(np.mean(assembly.byproducts.rho[species])) for species in list(assembly.byproducts.species)]
                [data_by.append(np.mean(assembly.byproducts.mf[species]))  for species in list(assembly.byproducts.species)]
            
            df_by = pd.DataFrame(data=[data_by],columns=[columns])

            header = False if os.path.exists(options.output_folder+'/Data/data_byproducts.csv') else True
            df_by.to_csv(options.output_folder+'/Data/data_byproducts.csv',mode='a',header=header,index=False)
        #Stagnation properties
        try:
            data['Qstag'] = [assembly.aerothermo.qconvstag]
            data['Qradstag'] = [assembly.aerothermo.qradstag]
        except:
            pass

        try:
            data['Pstag'] = [assembly.freestream.P1_s]
            data['Tstag'] = [assembly.freestream.T1_s]
            data['Rhostag'] = [assembly.freestream.rho_s]
        except:
            pass

        #Reference Dimensionsal constants
        data["Aref"] = [assembly.Aref]
        data["Lref"] = [assembly.Lref]
        df =  pd.DataFrame.from_dict(data)
        temp = {}
        mass = {}
        if options.write_object_properties:
            for i, obj in enumerate(assembly.objects):
                temp["Temperature_obj_"+str(i)] = [obj.temperature]
                mass["Mass_obj_"+str(i)] = [obj.mass]
            
            df_temp = pd.DataFrame.from_dict(temp)
            df_mass = pd.DataFrame.from_dict(mass)
            df = pd.concat([df, df_temp], axis = 1)
            df = pd.concat([df, df_mass], axis = 1)

        df = df.round(decimals = 12)
        if options.time_fidelity>0.0 and smooth:
            df.to_csv(options.output_folder + '/Data/'+ 'data_smooth.csv', mode='a' ,header=not os.path.exists(options.output_folder + '/Data/data_smooth.csv'), index = False)
            return
        elif not smooth:
            df.to_csv(options.output_folder + '/Data/'+ 'data.csv', mode='a' ,header=not os.path.exists(options.output_folder + '/Data/data.csv'), index = False)

    assem_data = {}
    for assembly in titan.assembly:
        assem_data["Time"] = [titan.time]
        assem_data["Iter"] = [titan.iter]
        assem_data["Assembly_ID"] = [assembly.id]
        for obj in assembly.objects:
            assem_data["Obj_name"] = [obj.name]
            assem_data["Density"] = [obj.material.density]
            assem_data["Mass"] = [obj.mass]
            if options.pato.flag:
                assem_data["MaxTemperature"] = [max(obj.pato.temperature)]
                print('obj:', obj.global_ID, ' max temp:', max(obj.pato.temperature))
            if options.thermal.ablation_mode == "0d":
                assem_data["Temperature"] = [obj.temperature]
            assem_data["Max_stress"] = [obj.max_stress]
            assem_data["Yield_stress"] = [obj.yield_stress]
            assem_data["Parent_id"] = [obj.parent_id]
            df = pd.DataFrame.from_dict(assem_data)        
            df = df.round(decimals = 6)
            df.to_csv(options.output_folder + '/Data/'+ 'data_assembly.csv', mode='a' ,header=not os.path.exists(options.output_folder + '/Data/data_assembly.csv'), index = False)

def write_to_series(data_array,columns,filename):
    """Append tabular values to a CSV file, writing its header when needed.

:param data_array: Rows of values to serialize.
:type data_array: Any
:param columns: Column names corresponding to the data values.
:type columns: Any
:param filename: Destination CSV path.
:type filename: str"""
    import pandas as pd
    import os
    data=pd.DataFrame(data_array,columns=columns)
    doHeader = False if os.path.exists(filename) else True
    data.to_csv(filename,mode='a',index=False,header=doHeader)

def generate_surface_solution(titan, options, iter_value, folder = 'Surface_solution'):
    """Write the assemblies' surface fields for a selected iteration.

:param titan: TITAN simulation object.
:type titan: object
:param options: Options or configuration object.
:type options: Options
:param iter_value: Iteration represented by the surface solution.
:type iter_value: Any
:param folder: Output subdirectory name.
:type folder: Any"""
    if not options.write_surface_solutions: return

    solutions = [create_surface_solution(_assembly,options) for _assembly in titan.assembly]
    [write_surface_solution(options, solution, _assembly.id, iter_value) for solution, _assembly in zip(solutions, titan.assembly)]


## The following funcs split generate_surface_solution() into separate create, update and
# write functions. At the moment this is something of a messy repeat but in future having
# these be separate could give a minor speedup, see the dense solution pipeline for an
#  example of this  

def create_surface_solution(assembly, options):
    """Documentation for the function.
:param titan: TITAN simulation object.
:type titan: object
:param options: Options or configuration object.
:type options: Options
:return: Return value.
:rtype: Any"""
    points = np.array([])
    facets = np.array([])
    pressure = np.array([])
    shear = np.array([])
    heatflux = np.array([])
    #hf_cond = np.array([])
    #radius = np.array([])
    #ellipse = np.array([])
    #cellID = np.array([])
    #emissive_power = np.array([])
    theta = np.array([])
    #he = np.array([])
    #hw = np.array([])
    #Te = np.array([])
    mDotVapor = np.array([])
    mDotMelt = np.array([])
    debug_alpha = np.array([])



    points = assembly.mesh.nodes - assembly.mesh.surface_displacement
    facets = assembly.mesh.facets
    pressure = assembly.aerothermo.pressure
    heatflux = assembly.aerothermo.heatflux
    shear = assembly.aerothermo.shear
    displacement = assembly.mesh.surface_displacement
    # radius = assembly.mesh.facet_radius
    # ellipse = assembly.inside_shock
    temperature  = assembly.aerothermo.temperature
    # emissive_power = assembly.emissive_power
    theta = assembly.aerothermo.theta
    # he = assembly.aerothermo.he
    # hw = assembly.aerothermo.hw
    # Te = assembly.aerothermo.Te
    debug_alpha = assembly.aerothermo.debug_alpha
    cell_data = {}


    if options.thermal.ablation_mode.lower() == 'pato' and options.pato.Ta_bc == 'ablation':
        mDotVapor = np.zeros(len(assembly.mesh.facets))
        mDotMelt  = np.zeros(len(assembly.mesh.facets))
        mDotVapor = assembly.mDotVapor
        mDotMelt = assembly.mDotMelt
    elif options.thermal.ablation_mode.lower() == 'byproducts':
        for byproduct in assembly.byproducts.species:
            cell_data['rho_'+byproduct] = np.array([assembly.byproducts.rho[byproduct]])
            cell_data['mf_'+byproduct] = np.array([assembly.byproducts.mf[byproduct]])
            cell_data['mass_'+byproduct] = np.array([assembly.byproducts.mass[byproduct]])
            cell_data['emission_'+byproduct] = np.array([assembly.byproducts.emission[byproduct]])
        mDotMelt = np.zeros(len(assembly.mesh.facets))
        mDotMelt = assembly.mDotMelt
    #hf_cond = assembly.hf_cond

    #cellID = np.arange(len(assembly.mesh.facets))
    # for cellid in range(len(assembly.mesh.facets)):
    #     cellID = np.append(cellID, cellid)

    
    cells = {"triangle": facets}

    cell_data["pressure"] = [pressure]
    cell_data["heatflux"] = [heatflux]
    cell_data["temperature"] = [temperature]
    cell_data["shear"] = [shear]
    cell_data["theta"] =  [theta]
    cell_data["debug_alpha"] = [debug_alpha]
                    #"Enthalpy BLE": [he],
                    #"Enthalpy Wall": [hw],
                    #"Temperatue BLE": [Te],
                #}
    # I don't believe He, Hw and Te are functional at present
    if options.thermal.ablation:
        if options.thermal.ablation_mode == 'PATO':
            cell_data["mDotVapor"] = [mDotVapor]
            cell_data["mDotMelt"]  = [mDotMelt]
        elif options.thermal.ablation_mode.lower() == 'byproducts':
            cell_data["mDotMelt"]  = [mDotMelt]
    point_data = { "displacement": displacement}

    trimesh = meshio.Mesh(points,
                            cells=cells,
                            point_data = point_data,
                            cell_data = cell_data)
    
    return trimesh

def update_surface_solution(assembly,options,solution,overwrite=None):
    """Documentation for the function.
:param titan: TITAN simulation object.
:type titan: object
:param options: Options or configuration object.
:type options: Options
:param solutions: Value for solutions.
:type solutions: Any
:param overwrite: Value for overwrite.
:type overwrite: Any
:return: Return value.
:rtype: Any"""
    points = np.array([])
    facets = np.array([])
    pressure = np.array([])
    shear = np.array([])
    heatflux = np.array([])
    #hf_cond = np.array([])
    #radius = np.array([])
    #ellipse = np.array([])
    #cellID = np.array([])
    #emissive_power = np.array([])
    theta = np.array([])
    #he = np.array([])
    #hw = np.array([])
    #Te = np.array([])
    mDotVapor = np.array([])
    mDotMelt = np.array([])
    debug_alpha = np.array([])

        # points = assembly.mesh.nodes - assembly.mesh.surface_displacement
    # facets = assembly.mesh.facets
    if overwrite is not None:
        pressure = overwrite['pressure'][:]
        heatflux = overwrite['heatflux'][:]
        shear = overwrite['shear'][:]
        
        temperature = overwrite['temperature'][:]
        theta = overwrite['theta'][:]
        debug_alpha = overwrite['debug_alpha'][:]
    else:
        pressure = [assembly.aerothermo.pressure]
        heatflux = [assembly.aerothermo.heatflux]
        shear = [assembly.aerothermo.shear]
        # radius = assembly.mesh.facet_radius
        # ellipse = assembly.inside_shock
        temperature  = [assembly.aerothermo.temperature]
        # emissive_power = assembly.emissive_power
        theta = [assembly.aerothermo.theta]
        # he = assembly.aerothermo.he
        # hw = assembly.aerothermo.hw
        # Te = assembly.aerothermo.Te
        debug_alpha = [assembly.aerothermo.debug_alpha]
    displacement = assembly.mesh.surface_displacement

    if options.thermal.ablation_mode.lower() == 'pato' and options.pato.Ta_bc == 'ablation':
        mDotVapor = np.zeros(len(assembly.mesh.facets))
        mDotMelt  = np.zeros(len(assembly.mesh.facets))
        if overwrite is not None:
            mDotVapor = overwrite['mDotVapor'][:]
            mDotMelt = overwrite['mDotMelt'][:]
        else:
            mDotVapor = assembly.mDotVapor
            mDotMelt  = assembly.mDotMelt
        solution.cell_data["mDotVapor"][:] = [mDotVapor]
        solution.cell_data["mDotMelt"][:]  = [mDotMelt]

    solution.cell_data["pressure"][:] = pressure
    solution.cell_data["heatflux"][:] = heatflux
    solution.cell_data["temperature"][:] = temperature
    solution.cell_data["shear"][:] = shear
    solution.cell_data["theta"][:] = theta
    solution.point_data["displacement"][:] = displacement
    solution.cell_data["debug_alpha"][:] = debug_alpha
    return solution

def write_surface_solution(options,solution,assembly_id,iter_value,folder='Surface_solution'):
    """Documentation for the function.

    :param options: Options or configuration object.
    :type options: Options
    :param solutions: Value for solutions.
    :type solutions: Any
    :param IDS: Value for ids.
    :type IDS: Any
    :param iter_value: Value for iter value.
    :type iter_value: Any
    :param folder: Value for folder.
    :type folder: Any"""
    if options.write_surface_solutions:
        folder_path = options.output_folder+'/' + folder + '/ID_'+str(assembly_id)
        Path(folder_path).mkdir(parents=True, exist_ok=True)

        vol_mesh_filepath = f"{folder_path}/solution_iter_{str(iter_value).zfill(3)}.xdmf"
        meshio.write(vol_mesh_filepath, solution, file_format="xdmf")

def generate_surface_solution_emissions(titan, options, iter_value, folder = 'Surface_solution'):
    """Documentation for the function.
:param titan: TITAN simulation object.
:type titan: object
:param options: Options or configuration object.
:type options: Options
:param iter_value: Value for iter value.
:type iter_value: Any
:param folder: Value for folder.
:type folder: Any"""

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
    """Documentation for the function.
:param obj: Value for obj.
:type obj: Any
:param quantity: Value for quantity.
:type quantity: Any
:param options: Options or configuration object.
:type options: Options
:param iter_value: Value for iter value.
:type iter_value: Any
:param folder: Value for folder.
:type folder: Any"""

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
    """Documentation for the function.
:param titan: TITAN simulation object.
:type titan: object
:param options: Options or configuration object.
:type options: Options"""
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
    """Documentation for the function.
:param titan: TITAN simulation object.
:type titan: object
:param options: Options or configuration object.
:type options: Options"""
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
    """TITAN_information."""
    # print(f"""                                                                                                                   
    #    ________  ______  ________   ______   __    __ 
    #   /        |/      |/        | /      \\ /  \\  /  |
    #   $$$$$$$$/ $$$$$$/ $$$$$$$$/ /$$$$$$  |$$  \\ $$ |
    #      $$ |     $$ |     $$ |   $$ |__$$ |$$$  \\$$ |
    #      $$ |     $$ |     $$ |   $$    $$ |$$$$  $$ |
    #      $$ |     $$ |     $$ |   $$$$$$$$ |$$ $$ $$ |
    #      $$ |    _$$ |_    $$ |   $$ |  $$ |$$ |$$$$ |
    #      $$ |   / $$   |   $$ |   $$ |  $$ |$$ | $$$ |
    #      $$/    $$$$$$/    $$/    $$/   $$/ $$/   $$/                                                                                                             
    # """)
    asc = '\n\n'
    with open(os.path.expandvars('$TITAN_PATH/titan/Output/titan_logo_terminal'),'r') as f:

        for line in f: asc+=line
    print(asc)
    print(f"""
        ###############################################
        # TITAN tool is still under heavy development #
        ###############################################

        Authors: Fábio Morgado, Julie Graham, Catarina Garbacz, Tommy Williamson, Marco Fossati and contributors
        Contact: tommy.williamson@strath.ac.uk
        Github:  https://github.com/strath-ace/TITAN
        Version: 0.2
        Release date: TBD...
        """)

def options_information(options):
    """Documentation for the function.
:param options: Options or configuration object.
:type options: Options"""
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
        
def iteration(titan, options, show_flow_solves = False):
    """Documentation for the function.
:param titan: TITAN simulation object.
:type titan: object
:param options: Options or configuration object.
:type options: Options
:param show_flow_solves: Value for show flow solves.
:type show_flow_solves: Any"""
    print(f"""  Iteration {titan.iter+1} of {options.iters} """, flush=True)
    if titan.iter>0 and hasattr(titan,'nfeval') and show_flow_solves: 
        print('Total of {} flow solves\n'.format(titan.nfeval))
