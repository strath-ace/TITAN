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
from Geometry.assembly import create_assembly_flag, Assembly
from Geometry.mesh import compute_new_volume_v2, map_surf_to_tetra
from copy import deepcopy
from scipy.spatial.transform import Rotation as Rot
from Output import output
import pymap3d

def demise_components(titan, i, joints_id, options): 
    """
    Computes the inertial forces in the Body Frame

    This functions computes the inertial forces that will be used for the Structurla dynamics

    Parameters
    ----------
    titan: Assembly_list
        Object of class Assembly_list
    assembly_pos: array
        Array containing the index position of the assemblies that will undergo fragmentation
    joints_id: array
        Array containing the index of the joints that demised (index in relation to each assembly that will undergo fragmentation), to be removed from the simulation
    options: Options
        Object of class Options
    """

    titan.assembly[i].temp_ids = np.arange(len(titan.assembly[i].objects)) + 1        

    COG = titan.assembly[i].COG
    angle = np.array([titan.assembly[i].roll, titan.assembly[i].pitch, titan.assembly[i].yaw])
    angle_vel = np.array([titan.assembly[i].roll_vel, titan.assembly[i].pitch_vel, titan.assembly[i].yaw_vel])

    connectivity = titan.assembly[i].connectivity
    index = np.zeros(len(connectivity), dtype = bool)

    # It's a single assembly with only one object, it will not have any connectivity with other objects

    for id in joints_id:
        index += (connectivity[:,0] == id) + (connectivity[:,1] == id) + (connectivity[:,2] == id)

    connectivity = connectivity[~index]

    #Change conectivity here to match the objects vector in assembly
    Flags = np.array([], dtype = int)

    for j in range(len(connectivity)):
        if connectivity[j,2] == 0:
            Flags = np.append(Flags, [connectivity[j,0],connectivity[j,1]])
        else:
            Flags = np.append(Flags, np.array([connectivity[j,0],connectivity[j,2]]))
            Flags = np.append(Flags, np.array([connectivity[j,2],connectivity[j,1]]))

    Flags.shape = (int(len(Flags)/2) ,2)

    #Remove demised objects
    mask_delete = []

    for enum_obj, obj in enumerate(titan.assembly[i].objects):
        if obj.mass <= 0 or titan.assembly[i].trajectory.altitude <= 0:
            mask_delete.append(enum_obj)
            Flags[Flags >= (enum_obj+1)] -= 1

    if len(mask_delete) > 0:
        titan.assembly[i].objects = np.delete(titan.assembly[i].objects,mask_delete)
        titan.assembly[i].temp_ids = np.delete(titan.assembly[i].temp_ids,mask_delete)

    assembly_flag = create_assembly_flag(titan.assembly[i].objects, Flags)

    for j in range(len(assembly_flag)):
        for it in range(len(assembly_flag[j])):
            if assembly_flag[j][it]: 
                print(titan.assembly[i].objects[it].name, it)
       
        titan.assembly.append(Assembly(titan.assembly[i].objects[assembly_flag[j]], titan.id))
        titan.id += 1

        temp_ids = titan.assembly[i].temp_ids[assembly_flag[j]]

        connectivity_assembly = np.zeros(connectivity.shape, dtype = bool)
        id_objs = np.array(range(1,len(assembly_flag[j])+1))[assembly_flag[j]]

        for id in id_objs:
            connectivity_assembly += (connectivity == id)
        titan.assembly[-1].connectivity = connectivity[(np.sum(connectivity_assembly, axis = 1) >= 2)]
        titan.assembly[-1].connectivity.shape = (-1)

        for k1 in range(len(titan.assembly[-1].connectivity)):
            for k2 in range(len(titan.assembly[-1].objects)):
                if titan.assembly[-1].connectivity[k1] == temp_ids[k2]:
                    titan.assembly[-1].connectivity[k1] = k2+1

        titan.assembly[-1].connectivity.shape = (-1,3)

        #Uses GMSH again to compute the inner domain of the new assembly
        #titan.assembly[-1].generate_inner_domain(write = False, output_folder = options.output_folder)
        #titan.assembly[-1].compute_mass_properties()
        #output.generate_volume(titan = titan, options = options)

        compute_new_volume_v2(titan.assembly[i].mesh, titan.assembly[-1].mesh, titan.assembly[-1].objects)
        titan.assembly[-1].mesh.index_surf_tetra = map_surf_to_tetra(titan.assembly[-1].mesh)
        output.generate_volume(titan = titan, options = options)

        titan.assembly[-1].compute_mass_properties()

        titan.assembly[-1].roll  = angle[0]
        titan.assembly[-1].pitch = angle[1]
        titan.assembly[-1].yaw   = angle[2]

        #Vector of COM difference
        dx = titan.assembly[-1].COG - titan.assembly[i].COG

        #Vector from body frame to ECEF frame
        R_B_ECEF = Rot.from_quat(titan.assembly[i].quaternion)
        dx_ECEF = R_B_ECEF.apply(dx)
        angle_vel_ECEF = R_B_ECEF.apply(angle_vel)

        titan.assembly[-1].position = np.copy(titan.assembly[i].position) + dx_ECEF
        titan.assembly[-1].velocity = np.copy(titan.assembly[i].velocity) + np.cross(angle_vel_ECEF,dx_ECEF)
        
        titan.assembly[-1].roll_vel  = angle_vel[0]
        titan.assembly[-1].pitch_vel = angle_vel[1]
        titan.assembly[-1].yaw_vel   = angle_vel[2]

        titan.assembly[-1].trajectory = deepcopy(titan.assembly[i].trajectory)
        titan.assembly[-1].trajectory.dyPrev = None
        titan.assembly[-1].quaternion = deepcopy(titan.assembly[i].quaternion)

        #Compute the trajectory and angular quantities
        [latitude, longitude, altitude] = pymap3d.ecef2geodetic(titan.assembly[-1].position[0], titan.assembly[-1].position[1], titan.assembly[-1].position[2], ell = None,deg = False);

        titan.assembly[-1].trajectory.latitude = latitude
        titan.assembly[-1].trajectory.longitude = longitude
        titan.assembly[-1].trajectory.altitude = altitude

        [vEast, vNorth, vUp] = pymap3d.uvw2enu(titan.assembly[-1].velocity[0], titan.assembly[-1].velocity[1], titan.assembly[-1].velocity[2], latitude, longitude, deg=False)

        titan.assembly[-1].trajectory.gamma = np.arcsin(np.dot(titan.assembly[-1].position, titan.assembly[-1].velocity)/(np.linalg.norm(titan.assembly[-1].position)*np.linalg.norm(titan.assembly[-1].velocity)))
        titan.assembly[-1].trajectory.chi = np.arctan2(vEast,vNorth)

        #ECEF_2_B
        [Vx_B, Vy_B, Vz_B] =  Rot.from_quat(titan.assembly[-1].quaternion).inv().apply(titan.assembly[-1].velocity)
        titan.assembly[-1].trajectory.velocity = np.linalg.norm([Vx_B, Vy_B, Vz_B])
        
        titan.assembly[-1].aoa = titan.assembly[i].aoa
        titan.assembly[-1].slip = titan.assembly[i].slip

def fragmentation(titan, options):

    """
    Check if components meet the specified criteria to be removed from the simulation. 
    At the moment, only altitude, iteration number, time and total ablation are specified.

    Parameters
    ----------
    titan: Assembly_list
        Object of class Assembly_list
    options: Options
        Object of class Options
    """

    assembly_id = np.array([], dtype = int)
    lenght_assembly = len(titan.assembly)

    for it in range(lenght_assembly):
        objs_id = np.array([], dtype = int)

        for _id, obj in enumerate(titan.assembly[it].objects):

            if obj.type == "Joint":

                if obj.trigger_type.lower() == 'altitude' and titan.assembly[it].trajectory.altitude <= obj.trigger_value:

                    print ('Altitude Fragmentation occured ')
                    objs_id = np.append(objs_id, _id)
                
                elif obj.trigger_type.lower() == 'temperature' and obj.temperature >= obj.trigger_value:

                    print ('Demisable joint: Thermal fragmentation activated! ')
                    objs_id = np.append(objs_id, _id)

                elif obj.trigger_type.lower() == 'iteration' and titan.iter >= obj.trigger_value:

                    print ('Iteration Fragmentation occured ')
                    objs_id = np.append(objs_id, _id)
                
                elif obj.trigger_type.lower() == 'time' and titan.time >= obj.trigger_value:

                    print ('Time Fragmentation occured ')
                    objs_id = np.append(objs_id, _id)

                """    
                elif obj.trigger_type == 'Stress' and assembly.trajectory.stress_ratio >= 0.:
                    if (assembly_id == it).any() == False: assembly_id = np.append(assembly_id, it)
                    print ('Stress Fragmentation occured in %s'%(assembly.trajectory.max_stress_obj))
                    print ('max stress vol id: ', assembly.trajectory.max_stress_ratio_vol_id)
                    if assembly.trajectory.max_stress_ratio_vol_id == obj.id:
                        joints_id = np.append(joints_id, k)
                        # assembly_id = np.append(assembly_id, it)
                        # trajectory[it].stress_ratio = -1
                """

            if obj.mass <= 0:
                print ('Mass demise occured')
                objs_id = np.append(objs_id, _id)
                        
            elif titan.assembly[it].trajectory.altitude <= 0:
                print ('Object reached ground')
                objs_id = np.append(objs_id, _id)

        objs_id = np.unique(objs_id)+1

        if len(objs_id) != 0:
            if len(titan.assembly[it].objects) != 1: demise_components(titan, it, objs_id, options)
            assembly_id = np.append(assembly_id, it)
            
    titan.assembly = np.delete(titan.assembly,assembly_id).tolist()