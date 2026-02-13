import numpy as np
from scipy.spatial.transform import Rotation as Rot
import os
from meshio import Mesh
# This function returns a boolean decision of whether to do a raytracing calc on the assembly
# based upon its enclosure state, which is equivalent to ~is_enclosed 
def check_enclosure(assembly_list,options, assembly_index,debug_iter=0):
    # Assembly of interest
    AoI = assembly_list[assembly_index]
    enclosure_list = np.array([component.enclosure for component in AoI.objects])
    # If any part of the assembly is unenclosed we must simulate it
    if np.any(enclosure_list>=0): return True
    do_raytrace = False
    enclosures = np.unique(enclosure_list)
    for enclosed in enclosures:
        found_assembly = False
        for i_assem, _assembly in enumerate(assembly_list):
            if not i_assem==assembly_index:
                if -1*enclosed in _assembly.enclosure_AABB.keys():
                    # If enclosure is split across multiple assemblies the enclosed is exposed  to the airstream
                    if found_assembly: 
                        print('Enclosure fragmented! Breaking connection')
                        for component in AoI.objects: component.enclosure = 0
                        return True
                    else: found_assembly = True

                    enclosure_objs = np.array([component.enclosure for component in _assembly.objects])
                    n_objs = len(np.where(enclosure_objs==-1*enclosed)[0])

                    # If the enclosure has been broken the enclosed is exposed to the airstream
                    if n_objs<_assembly.enclosure_component_num[-1*enclosed]: 
                        print('Enclosure fragmented! Breaking connection')
                        for component in AoI.objects: component.enclosure = 0
                        return True
                    
                    debug_meshes = None #['closed_{}_{}'.format(assembly_index,debug_iter),'closure_{}_{}'.format(i_assem,debug_iter-5)]
                    do_raytrace = enclosure_mesh_check(AoI.mesh,AoI.COG,_assembly.enclosure_AABB[-1*enclosed],AoI.quaternion,
                                                       _assembly.quaternion, _assembly.position-AoI.position, debug_meshes)
    return do_raytrace
# Build collection of Axis-Aligned Bounding Boxes (AABBs) for each enclosure id,
# a necessary but not sufficient criterion for something being inside something else
def build_enclosure_AABB(assembly):
    AABB_dict = {}
    for component in assembly.objects:
        if not component.enclosure==0:
            comp_AABB = [np.zeros(3),np.zeros(3)]
            comp_AABB[0] = np.min(component.mesh.nodes, axis=0)-assembly.COG
            comp_AABB[1] = np.max(component.mesh.nodes, axis=0)-assembly.COG
            if not component.enclosure in AABB_dict.keys(): AABB_dict[component.enclosure] = comp_AABB                
            else:
                for i_ax in range(3):
                    if comp_AABB[0][i_ax]<AABB_dict[component.enclosure][0][i_ax]: 
                        AABB_dict[component.enclosure][0][i_ax] = comp_AABB[0][i_ax] 
                    if comp_AABB[1][i_ax]>AABB_dict[component.enclosure][1][i_ax]: 
                        AABB_dict[component.enclosure][1][i_ax] = comp_AABB[1][i_ax]
    return AABB_dict
# Need also to log the number components in an enclosure to check if enclosure is still whole
def build_enclosure_num(assembly):
    num = {}
    for component in assembly.objects:
        if component.enclosure>0:
            if not component.enclosure in num.keys(): 
                num[component.enclosure] = 1
            else: num[component.enclosure]+=1
    return num

def transform_AABB(AABB, quaternion_BODY, translation_CoG, translation_ECEF, translation_DATUM):
    from Dynamics.propagation import quaternion_to_matrix
    output_AABB = [np.zeros(3),np.zeros(3)]
    box_array = np.array([[AABB[0][0],AABB[0][1],AABB[0][2],1], 
                          [AABB[0][0],AABB[0][1],AABB[1][2],1],
                          [AABB[0][0],AABB[1][1],AABB[0][2],1],
                          [AABB[1][0],AABB[0][1],AABB[0][2],1],
                          [AABB[0][0],AABB[1][1],AABB[1][2],1],
                          [AABB[1][0],AABB[0][1],AABB[1][2],1],
                          [AABB[1][0],AABB[1][1],AABB[0][2],1],
                          [AABB[1][0],AABB[1][1],AABB[1][2],1]])
    mat_CoG  = np.eye(4)
    mat_ECEF = np.eye(4)
    mat_DATUM = np.eye(4)
    mat_ECEF[:-1,-1] = translation_ECEF
    mat_DATUM[:-1,-1] = translation_DATUM
    mat_BODY = quaternion_to_matrix(quaternion_BODY)
    transform =  mat_DATUM @ mat_ECEF @ mat_BODY# @ mat_CoG
    transform_AABB = box_array @ transform.T
    output_AABB[0] = np.min(transform_AABB,axis=0)[:-1]
    output_AABB[1] = np.max(transform_AABB,axis=0)[:-1]
    return output_AABB

def debug_AABB_mesh(AABB, filename=None):
    box_array = np.array([[AABB[0][0],AABB[0][1],AABB[0][2]], 
                         [AABB[0][0],AABB[0][1],AABB[1][2]],
                         [AABB[0][0],AABB[1][1],AABB[0][2]],
                         [AABB[1][0],AABB[0][1],AABB[0][2]],
                         [AABB[0][0],AABB[1][1],AABB[1][2]],
                         [AABB[1][0],AABB[0][1],AABB[1][2]],
                         [AABB[1][0],AABB[1][1],AABB[0][2]],
                         [AABB[1][0],AABB[1][1],AABB[1][2]]])

    faces = np.array([
        # bottom (z = zmin): 0,2,3,6
        [0,2,3], [2,6,3],
        # top (z = zmax): 1,4,5,7
        [1,5,4], [4,5,7],
        # front (y = ymin): 0,1,3,5
        [0,1,3], [1,5,3],
        # back (y = ymax): 2,4,6,7
        [2,6,4], [4,6,7],
        # left (x = xmin): 0,2,1,4
        [0,1,2], [1,4,2],
        # right (x = xmax): 3,5,6,7
        [3,6,5], [5,6,7],
    ])
    aabb_mesh = Mesh(points= box_array,cells = {"triangle": faces})
    if not os.path.exists('./debug_AABB'): os.mkdir('./debug_AABB')
    aabb_mesh.write('./debug_AABB/{}.stl'.format(filename))

def enclosure_mesh_check(enclosed_mesh, enclosed_COG, enclosure_AABB_BODY, quat_enclosed, quat_enclosure, dist, debug_meshes=None):
    ## Firstly must transform enclosed to enclosure body frame...
    R_B_ECEF_closed  = np.eye(4)
    R_ECEF_B_closure = np.eye(4)
    translate_COG    = np.eye(4)
    translate_dist   = np.eye(4)
    
    R_B_ECEF_closed[:-1,:-1]  = Rot.from_quat(quat_enclosed).as_matrix()
    R_ECEF_B_closure[:-1,:-1] = Rot.from_quat(quat_enclosure).inv().as_matrix()
    translate_COG[:-1,-1] = -enclosed_COG
    translate_dist[:-1,-1] = dist
    transform = R_ECEF_B_closure @ translate_dist @ R_B_ECEF_closed @ translate_COG
    transformed_nodes = np.hstack([enclosed_mesh.nodes, np.ones([np.shape(enclosed_mesh.nodes)[0],1])]) @ transform.T

    closed_AABB = [np.min(transformed_nodes,axis=0)[:-1],np.max(transformed_nodes,axis=0)[:-1]]

    if debug_meshes is not None:
        debug_AABB_mesh(closed_AABB,filename=debug_meshes[0])
        debug_AABB_mesh(enclosure_AABB_BODY,filename=debug_meshes[1])

    if np.any(closed_AABB[0]<enclosure_AABB_BODY[0]): 
        print('Assembly has left enclosure!')
        return True
    if np.any(closed_AABB[1]>enclosure_AABB_BODY[1]): 
        print('Assembly has left enclosure!')
        return True
    return False