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
from Geometry.mesh import compute_new_volume_v2, map_surf_to_tetra, check_tetra_in_surface, compute_surface_from_tetra, update_volume_displacement
from copy import deepcopy
from scipy.spatial.transform import Rotation as Rot
from Output import output
import pymap3d
import open3d as o3d
import trimesh
from Geometry.component import Component
from collections import defaultdict
from Dynamics import collision
from Thermal import pato

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
    distance_travelled = titan.assembly[i].distance_travelled

    connectivity = titan.assembly[i].connectivity
    index = np.zeros(len(connectivity), dtype = bool)

    # It's a single assembly with only one object, it will not have any connectivity with other objects
    for id in joints_id:
        index += (connectivity[:,0] == id) + (connectivity[:,1] == id) + (connectivity[:,2] == id)

    connectivity = np.copy(connectivity[~index])

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

    aux = 0
    for enum_obj, obj in enumerate(titan.assembly[i].objects):
        if obj.mass <= 0 or titan.assembly[i].trajectory.altitude <= 0:
            mask_delete.append(enum_obj)
            Flags[Flags >= (enum_obj+1-aux)] -= 1
            #connectivity[connectivity >= (enum_obj+1-aux)] -= 1
            aux += 1

    if len(mask_delete) > 0:
        titan.assembly[i].objects = np.delete(titan.assembly[i].objects,mask_delete)
        titan.assembly[i].temp_ids = np.delete(titan.assembly[i].temp_ids,mask_delete)

    assembly_flag = create_assembly_flag(titan.assembly[i].objects, Flags)
    
    update_volume_displacement(titan.assembly[i].mesh, - titan.assembly[i].mesh.volume_displacement)
    
    for j in range(len(assembly_flag)):
        titan.assembly.append(Assembly(titan.assembly[i].objects[assembly_flag[j]], titan.id, options = options))
        for obj in titan.assembly[-1].objects:
            obj.parent_id = titan.assembly[i].id

        titan.id += 1

        """
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
        """

        #New computation of connectivity

        objs_old_ids = np.array([obj.id for obj in titan.assembly[-1].objects])
        objs_new_ids = np.arange(1,len(titan.assembly[-1].objects)+1)

        dict_aux = {0:0}
        for old, new in zip(objs_old_ids, objs_new_ids):
            dict_aux[old] = new

        new_connectivity = []

        for con in connectivity:
            if con[0] in objs_old_ids or con[1] in objs_old_ids:
                new_connectivity.append(con)

        new_connectivity = np.array(new_connectivity)

        new_connectivity.shape = -1

        for index_con in range(len(new_connectivity)):
            new_connectivity[index_con] = dict_aux[new_connectivity[index_con]]

        new_connectivity.shape = (-1,3)


        titan.assembly[-1].connectivity = np.array(new_connectivity)

        #Uses GMSH again to compute the inner domain of the new assembly
        #titan.assembly[-1].generate_inner_domain(write = False, output_folder = options.output_folder)
        #titan.assembly[-1].compute_mass_properties()
        #output.generate_volume(titan = titan, options = options)

        compute_new_volume_v2(titan.assembly[i].mesh, titan.assembly[-1].mesh, titan.assembly[-1].objects)
        titan.assembly[-1].mesh.index_surf_tetra = map_surf_to_tetra(titan.assembly[-1].mesh.vol_coords, titan.assembly[-1].mesh.vol_elements)

        titan.assembly[-1].compute_mass_properties()

        titan.assembly[-1].roll  = angle[0]
        titan.assembly[-1].pitch = angle[1]
        titan.assembly[-1].yaw   = angle[2]

        titan.assembly[-1].roll_vel_last = deepcopy(titan.assembly[i].roll_vel)
        titan.assembly[-1].pitch_vel_last = deepcopy(titan.assembly[i].pitch_vel)
        titan.assembly[-1].yaw_vel_last = deepcopy(titan.assembly[i].yaw_vel)


        #Vector of COM difference
        dx = titan.assembly[-1].COG - titan.assembly[i].COG

        #Vector from body frame to ECEF frame
        R_B_ECEF = Rot.from_quat(titan.assembly[i].quaternion)
        dx_ECEF = R_B_ECEF.apply(dx)
        angle_vel_ECEF = R_B_ECEF.apply(angle_vel)

        titan.assembly[-1].position = np.copy(titan.assembly[i].position) + dx_ECEF
        titan.assembly[-1].velocity = np.copy(titan.assembly[i].velocity) + np.cross(angle_vel_ECEF,dx_ECEF)
        
        titan.assembly[-1].position_nlast = deepcopy(titan.assembly[-1].position)
        titan.assembly[-1].velocity_nlast = deepcopy(titan.assembly[-1].velocity)

        titan.assembly[-1].roll_vel  = angle_vel[0]
        titan.assembly[-1].pitch_vel = angle_vel[1]
        titan.assembly[-1].yaw_vel   = angle_vel[2]

        titan.assembly[-1].trajectory = deepcopy(titan.assembly[i].trajectory)
        titan.assembly[-1].trajectory.dyPrev = None
        titan.assembly[-1].quaternion = deepcopy(titan.assembly[i].quaternion)

        #Compute the trajectory and angular quantities
        [latitude, longitude, altitude] = pymap3d.ecef2geodetic(titan.assembly[-1].position[0], titan.assembly[-1].position[1], titan.assembly[-1].position[2],
                                        ell=pymap3d.Ellipsoid(semimajor_axis = options.planet.ellipsoid()['a'], semiminor_axis = options.planet.ellipsoid()['b']),
                                        deg = False);
        titan.assembly[-1].trajectory.latitude = latitude
        titan.assembly[-1].trajectory.longitude = longitude
        titan.assembly[-1].trajectory.altitude = altitude
        titan.assembly[-1].distance_travelled = distance_travelled 

        [vEast, vNorth, vUp] = pymap3d.uvw2enu(titan.assembly[-1].velocity[0], titan.assembly[-1].velocity[1], titan.assembly[-1].velocity[2], latitude, longitude, deg=False)

        titan.assembly[-1].trajectory.gamma = np.arcsin(np.dot(titan.assembly[-1].position, titan.assembly[-1].velocity)/(np.linalg.norm(titan.assembly[-1].position)*np.linalg.norm(titan.assembly[-1].velocity)))
        titan.assembly[-1].trajectory.chi = np.arctan2(vEast,vNorth)

        #ECEF_2_B
        [Vx_B, Vy_B, Vz_B] =  Rot.from_quat(titan.assembly[-1].quaternion).inv().apply(titan.assembly[-1].velocity)
        titan.assembly[-1].trajectory.velocity = np.linalg.norm([Vx_B, Vy_B, Vz_B])
        
        titan.assembly[-1].aoa = titan.assembly[i].aoa
        titan.assembly[-1].slip = titan.assembly[i].slip

        titan.post_event_iter = 0
        from Dynamics.propagation import construct_state_vector
        construct_state_vector(titan.assembly[-1])
        titan.assembly[-1].unmodded_angles = titan.assembly[i].unmodded_angles


def check_breakup_v2(titan, options):

    #ROUTINE to check if the mesh has split.

    #Loop all the assemblies
    for assembly in titan.assembly:
        if assembly.ablation_mode != 'tetra': continue

        tri_mesh = o3d.geometry.TriangleMesh()
        tri_mesh.vertices = o3d.utility.Vector3dVector(assembly.mesh.nodes)
        tri_mesh.triangles = o3d.utility.Vector3iVector(assembly.mesh.facets)
        
        #Compute the surface clusters
        #Function that clusters connected triangles, i.e., triangles that are connected via edges are assigned
        #the same cluster index. This function returns an array that contains the cluster index per triangle,
        #a second array contains the number of triangles per cluster, and a third vector contains the surface area per cluster.
        triangle_clusters, cluster_n_triangles, cluster_area = (tri_mesh.cluster_connected_triangles())
        triangle_clusters = np.asarray(triangle_clusters)

        #Check how many clusters are there
        cluster_ids = set(triangle_clusters)
        
        #If there is just one cluster, fragmentation did not occurr, so we skip the assembly
        if len(cluster_ids) != 1:

            #Retrieve only the elements that have not been ablated
            vol_elements = assembly.mesh.vol_elements[assembly.mesh.vol_mass != 0]
            vol_tag = assembly.mesh.vol_tag[assembly.mesh.vol_mass != 0]

            last_id = max([obj.id for obj in assembly.objects])
            
            ###
            # This section checks which tetra belong to each computed cluster
            # 
            # The function check_tetra_in_surface uses raycast to determine if the tetra is inside or
            # outside the cluster. There is still no good function open-source that we can use, therefore
            # we have a temporary fix 
            ###
            index_list = []
            for id in cluster_ids:

                #Check which tetras are inside the cluster
                #
                #This function needs some debugging as it's not constantly returning the same number
                #Most likely we need to create our own function in C++
                index_list.append(check_tetra_in_surface(np.asarray(tri_mesh.triangles)[triangle_clusters==id],
                                       np.asarray(tri_mesh.vertices),
                                       vol_elements,
                                       assembly.mesh.vol_coords))

            #Section to fix the tetras that have not been allocated, by searching where the neighbour tetras
            #belong to (which cluster)

            index_list = np.array(index_list)
            missing_tetra = np.sum(index_list , axis = 0).astype(bool)
            missing_tetra = np.invert(missing_tetra)

            index_missing_tetra = np.where(missing_tetra)[0]

            c = assembly.mesh.vol_coords
            map_tetra = map_surf_to_tetra(c, vol_elements)
            for index in index_missing_tetra:
                t = vol_elements[index]

                f1 = np.round((c[t[0]] + c[t[1]] + c[t[2]])/3,5).astype(str)
                f2 = np.round((c[t[0]] + c[t[1]] + c[t[3]])/3,5).astype(str)
                f3 = np.round((c[t[0]] + c[t[2]] + c[t[3]])/3,5).astype(str)
                f4 = np.round((c[t[1]] + c[t[2]] + c[t[3]])/3,5).astype(str)

                #Convert to concatenated strings to obtain key maps
                f1 = str(np.char.add(np.char.add(f1[0],f1[1]),f1[2]))
                f2 = str(np.char.add(np.char.add(f2[0],f2[1]),f2[2]))
                f3 = str(np.char.add(np.char.add(f3[0],f3[1]),f3[2]))
                f4 = str(np.char.add(np.char.add(f4[0],f4[1]),f4[2]))

                for i,row in enumerate(index_list):
                    try:
                        if any(row[map_tetra[f1]]):
                            index_list[i,[map_tetra[f1]]] = True;
                            break;
                    except: pass
                    try:
                        if any(row[map_tetra[f2]]):
                            index_list[i,[map_tetra[f2]]] = True;
                            break;
                    except: pass
                    try:
                        if any(row[map_tetra[f3]]):
                            index_list[i,[map_tetra[f3]]] = True;
                            break;
                    except: pass
                    try:
                        if any(row[map_tetra[f4]]):
                            index_list[i,[map_tetra[f4]]] = True;
                            break;
                    except: pass

            index_list = np.array(index_list)
            missing_tetra = np.sum(index_list , axis = 0).astype(bool)
            missing_tetra = np.invert(missing_tetra)
            print("Final missing tetras: ", sum(missing_tetra))

            # At this stage, we have the division of tetras between the different clusters
            # 
            # There are 2 scenarios: 
            #
            # 1- The tetras for a cluster only belong to one objects
            # 2- The tetras for a cluster belong to several objects (In this case, we need to check for connectivity)

            for index in index_list:
                objs_ids = list(set(assembly.mesh.vol_tag[assembly.mesh.vol_mass != 0][index]))

                #Scenario 1:
                if len(objs_ids)==1:

                    obj_id = objs_ids[0]
                    v0,v1,v2 = compute_surface_from_tetra(assembly.mesh.vol_coords, vol_elements[index])

                    if len(v0) < 4: 
                        continue

                    new_component = Component(filename = None, file_type = "Primitive", id = last_id + 1, material = assembly.objects[obj_id-1].material.name,
                        v0 = v0,
                        v1 = v1,
                        v2 = v2,
                        trigger_type = assembly.objects[obj_id-1].trigger_type,
                        trigger_value = assembly.objects[obj_id-1].trigger_value,
                        parent_id = assembly.id, parent_part = assembly.objects[obj_id-1].parent_part ,options = options)

                    last_id += 1

                    assembly.mesh.vol_tag[np.where(assembly.mesh.vol_mass != 0)[0][index]] = last_id
                    assembly.objects = np.append(assembly.objects,new_component)
                    assembly.connectivity = np.append(assembly.connectivity, np.array([objs_ids[0], last_id, 0], dtype =int)).reshape((-1,3))
                
                #Scenario 2:
                else:

                    # Create dictionary to map the old components with the new components
                    d = defaultdict(list)

                    #Loop the different ids of the object
                    for obj_id in objs_ids:
                    
                        #Checking if any tetra of the object ablated here. If not, we do not need to change anything here
                        #Here we just need to to a simple check:
                        if len(vol_elements[vol_tag == obj_id]) == len(vol_elements[index][vol_tag[index] == obj_id]):
                            d[obj_id].append(obj_id)
                            continue
                        #if the check fails, the obj_id has separated

                        #Generate the new components
                        v0,v1,v2 = compute_surface_from_tetra(assembly.mesh.vol_coords, vol_elements[index][vol_tag[index] == obj_id])

                        if len(v0) < 4: 
                            continue

                        new_component = Component(filename = None, file_type = "Primitive", id = last_id + 1, material = assembly.objects[obj_id-1].material.name,
                                v0 = v0,
                                v1 = v1,
                                v2 = v2,
                                trigger_type = assembly.objects[obj_id-1].trigger_type,
                                trigger_value = assembly.objects[obj_id-1].trigger_value,
                                parent_id = assembly.id, parent_part = assembly.objects[obj_id-1].parent_part, options = options)

                        last_id += 1

                        assembly.mesh.vol_tag[np.where(assembly.mesh.vol_mass != 0)[0][index][vol_tag[index] == obj_id]] = last_id
                        assembly.objects = np.append(assembly.objects,new_component)
                        
                        d[obj_id].append(last_id)

                    #Check the connectivity:
                    connectivity = np.copy(assembly.connectivity)

                    for row in connectivity:
                        new_connect = np.array([0,0,0], dtype = int)

                        #if prim 1 and prim 2 are in dict.keys()
                        if row[0] in d.keys() and row[1] in d.keys():
                            new_connect[0] = d[row[0]][0]
                            new_connect[1] = d[row[1]][0]
                            try:
                                new_connect[2] = d[row[2]][0]
                            except:
                                new_connect[2] = 0

                        elif row[0] in d.keys() and row[2] in d.keys():
                            new_connect[0] = d[row[0]][0]
                            new_connect[1] = d[row[2]][0]
                            try:
                                new_connect[2] = d[row[2]][0]
                            except:
                                new_connect[2] = 0

                        elif row[1] in d.keys() and row[2] in d.keys():
                            new_connect[0] = d[row[1]][0]
                            new_connect[1] = d[row[2]][0]
                            try:
                                new_connect[2] = d[row[2]][0]
                            except:
                                new_connect[2] = 0

                        if (new_connect != np.array([0,0,0])).all():
                            assembly.connectivity = np.append(assembly.connectivity, new_connect).reshape((-1,3))


            #If the mass of a newly generated object is inferior to the imposed threshold, demise it so we do not
            #generate too many debris
            for obj in assembly.objects:
                if np.sum(assembly.mesh.vol_mass[assembly.mesh.vol_tag == obj.id]) <= 0.05:
                    assembly.mesh.vol_density[assembly.mesh.vol_tag == obj.id] = 0
            
            assembly.compute_mass_properties()
            for obj in assembly.objects: print(obj.mass)


def check_breakup(titan, options):

    #ROUTINE to check if the mesh has split.
    # If True, change the density to 0 and recompute the mass of the singular component.
    # For now, only performed for assembly with multiple components

    for assembly in titan.assembly:

        mesh = assembly.mesh

        tri_mesh = o3d.geometry.TriangleMesh()
        tri_mesh.vertices = o3d.utility.Vector3dVector(mesh.nodes)
        tri_mesh.triangles = o3d.utility.Vector3iVector(mesh.facets)
        
        triangle_clusters, cluster_n_triangles, cluster_area = (tri_mesh.cluster_connected_triangles())
        
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_ids = set(triangle_clusters)
        
        #No breakup occurred
        if len(cluster_ids) != 1:

            last_id = max([obj.id for obj in assembly.objects])
            
            #Loop the geometrical clusters
            for id in cluster_ids:

                #Check which tetras are inside the cluster
                #
                #This function needs some debugging as it's not constantly returning the same number
                #Most likely we need to create our own function in C++
                #
                #For this to work, we need to pass the entirity of the assembly tetras to the new objects
                index = check_tetra_in_surface(np.asarray(tri_mesh.triangles)[triangle_clusters==id],
                                       np.asarray(tri_mesh.vertices),
                                       mesh.vol_elements,
                                       mesh.vol_coords)
            
                objs_ids = list(set(mesh.vol_tag[index]))

                #If its only One 
                #AND IS NOT CONNECTED TO ANYTHING
                if len(objs_ids)==1:
                    v0,v1,v2 = compute_surface_from_tetra(assembly.mesh.vol_coords, assembly.mesh.vol_elements[index])

                    new_component = Component(filename = None, file_type = "Primitive", id = last_id + 1, material = 'Unittest_demise',
                        v0 = v0,
                        v1 = v1,
                        v2 = v2, 
                        parent_id = objs_ids[0])

                    last_id += 1
                    assembly.mesh.vol_tag[index] = last_id

                    assembly.objects = np.append(assembly.objects,new_component)

                    assembly.connectivity = np.append(assembly.connectivity, np.array([objs_ids[0], last_id, 0], dtype =int)).reshape((-1,3))
                
                #TODO
                #If mass smaller than 0.05 kg, remove it instead of dealing with it for the remaining iterations 
                #if np.sum(mesh.vol_mass[index]) < 0.05:
                #    mesh.vol_density[mesh.vol_tag == obj.id] = 0
                #trimesh.Trimesh(vertices = tri_mesh.vertices, faces = np.asarray(tri_mesh.triangles)[triangle_clusters==id]).show()

            for obj in assembly.objects:
                if np.sum(assembly.mesh.vol_mass[assembly.mesh.vol_tag == obj.id]) <= 0.05:
                    mesh.vol_density[mesh.vol_tag == obj.id] = 0
            
            assembly.compute_mass_properties()
            for obj in assembly.objects: print(obj.mass)


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

    #if options.ablation_mode.lower() == "tetra":
    
    #Check for tetra ablation
    check_breakup_v2(titan, options)

    assembly_id = np.array([], dtype = int)
    length_assembly = len(titan.assembly)

    fragmentation_flag = False
    
    for it in range(length_assembly):
        objs_id = np.array([], dtype = int)
        primitive_separation = False
        if titan.assembly[it].freestream.mach <= options.dynamics.ignore_mach and titan.assembly[it].freestream.mach>0:
            print('Low Mach fragmentation occured for assembly {} at Ma={}'.format(it,titan.assembly[it].freestream.mach))
            objs_id = np.array([i for i in range(len(titan.assembly[it].objects))])
        elif titan.assembly[it].mass <= options.dynamics.ignore_mass:
            print('Low Mass fragmentation occured for assembly {} at {}kg'.format(it,titan.assembly[it].mass))
            objs_id = np.array([i for i in range(len(titan.assembly[it].objects))])
        for _id, obj in enumerate(titan.assembly[it].objects):

            if obj.type == "Joint":
                if obj.trigger_type.lower() == 'altitude' and titan.assembly[it].trajectory.altitude <= obj.trigger_value:

                    print ('Joint altitude Fragmentation occured ')
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

                elif obj.trigger_type.lower() == 'joint':
                    if len(titan.assembly[it].objects) <= 2:
                        objs_id = np.append(objs_id, _id)

            if obj.type == "Primitive":

                if obj.trigger_type.lower() == 'altitude' and titan.assembly[it].trajectory.altitude <= obj.trigger_value:

                    print ('Primitive altitude Fragmentation occured ')
                    con_delete = []

                    for index, con in enumerate(titan.assembly[it].connectivity):
                        if con[2] == 0:
                            con_delete.append(index)
                    
                    titan.assembly[it].connectivity = np.delete(titan.assembly[it].connectivity, con_delete, axis = 0)
                    titan.assembly[it].shape = (-1,3)

                    primitive_separation = True

                    #Removing the trigger once activated
                    obj.trigger_type = ""

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

            if obj.mass <= 0 or len(obj.mesh.nodes) <= 3:
                print ('Mass demise occured for object:', obj.name)
                objs_id = np.append(objs_id, _id)
                        
            elif titan.assembly[it].trajectory.altitude <= 0:
                print ('Object reached ground')
                objs_id = np.append(objs_id, _id)

        objs_id = np.unique(objs_id)+1

        if len(objs_id) != 0 or primitive_separation:
            fragmentation_flag = True
            if len(titan.assembly[it].objects) != 1:
                demise_components(titan, it, objs_id, options)
            assembly_id = np.append(assembly_id, it)
            
    titan.assembly = np.delete(titan.assembly,assembly_id).tolist()

    if fragmentation_flag:
        for assembly in titan.assembly:
            assembly.rearrange_ids()

        if options.collision.flag and len(assembly_id) != 0:
            for assembly in titan.assembly: collision.generate_collision_mesh(assembly, options)
            collision.generate_collision_handler(titan, options)
        
            if length_assembly < len(titan.assembly): 
                options.time_counter = options.collision.post_fragmentation_iters

        output.generate_volume(titan = titan, options = options)


    if options.thermal.ablation and options.pato.flag:
        for assembly in titan.assembly:
            pato.identify_object_connections(assembly)
