import gmsh
import numpy as np
import pathlib
import subprocess
from Geometry.gmsh_api import mesh_Settings, object_grid, physical_surf_from_list
from Explosion.split_stl import stl_split
from scipy.spatial import ConvexHull

## Mesh api, modified from the inner domain, does not need to be as robust to assembly configuration as it will only be called on objects
def generate_fragment_meshes(mesh, obj, voronoi, write = False, output_folder = '', output_filename = '', ref_obj_override=None):
    # Most of this is the same as gmsh_api but tweaked to use OCC as the factory
    gmsh.initialize()
    mesh_Settings(gmsh)
    gmsh.option.setNumber("Geometry.Tolerance", 1e-6)
    gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-6)
    gmsh.option.setNumber("Geometry.OCCScaling", 1.0)
    ref_size = 0.15
    ref_joint = 0.5
    ref_panel = 0.05
    factory = gmsh.model.occ
    if ref_obj_override is not None:
        ref_size = ref_obj_override
    elif obj.type.lower() == 'joint':
        ref_size = ref_joint
    elif "panel" in obj.name:
        ref_size = ref_panel

    density_elem = []
    tag_elem = []

    init_ref_surf = 1
    surf_ref_init = 1
    ref_phys_surface = 1

    
    gmsh.model.mesh.createGeometry()
    #Change refs to refine joints as well
    ref = np.ones(len(mesh.nodes))*ref_size



    node_ref_init, edge_ref_init, surf_ref_init = object_grid(gmsh, mesh.nodes, mesh.edges, mesh.facet_edges, ref, factory=factory)


    out = factory.addSurfaceLoop(np.arange(1, surf_ref_init))
    if obj.inner_mesh:
        raise NotImplementedError # Some bizarre issues exist with inner stls
        # Can circumvent by combining into one 
        node_ref_end , edge_ref_end, surf_ref_end = object_grid(gmsh,obj.inner_mesh.nodes, obj.inner_mesh.edges, 
                                                                obj.inner_mesh.facet_edges, ref,node_ref_init, 
                                                                edge_ref_init, surf_ref_init, factory=factory)
        hole = factory.addSurfaceLoop(np.arange(surf_ref_init+1, surf_ref_end))
        vol_tag = factory.addVolume([out,hole])
        
        node_ref_init = node_ref_end
        edge_ref_init = edge_ref_end
        surf_ref_init = surf_ref_end

    else:
        vol_tag = factory.addVolume([out])
    
    obj.vol_tag = vol_tag
    vol_map = add_voronoi_hedra(gmsh, voronoi, node_ref_init, edge_ref_init, surf_ref_init, vol_tag+1)
    vol_map = np.array(vol_map)
    factory.synchronize()
    hedra_tags = vol_map[np.where(vol_map>0)[0]]

    gmsh.write(output_folder+"/voronoi_domain.geo_unrolled")
    #factory.removeAllDuplicates()
    print('Performing fragment boolean, this may take time!')
    fragments = list(set(factory.getEntities(3))-set((3,obj.vol_tag)))
    #volume_boundary = gmsh.model.getBoundary([(3, obj.vol_tag)], recursive=False, oriented=False)
    #fragments_boundary = list(set(gmsh.model.getBoundary([(3, hed) for hed in hedra_tags],recursive=False,oriented=False)))
    for frag in fragments:
        remove = False if not frag == fragments[-1] else True
        inter = factory.intersect([(3,obj.vol_tag)],[frag],removeObject=remove)
    #inter = factory.intersect(fragments_boundary, volume_boundary)
    factory.remove([(3, obj.vol_tag)],recursive=True)
    factory.synchronize()
    surfs = set(factory.getEntities(2))
    vols = factory.getEntities(3)
    boundaries = set([b for b in gmsh.model.getBoundary(vols,oriented=False, recursive=False) if b[0]==2])
    to_remove = list(surfs - boundaries)

    factory.remove(to_remove, recursive=True)
    factory.synchronize()
    
    frags = factory.getEntities(3)
    for i_frag, frag in enumerate(frags):
        surfs = [s[1] for s in gmsh.model.getBoundary([frag],recursive=False,oriented=False)]
        physical_surf_from_list(gmsh, surfs, i_frag+1, 'frag_{}'.format(i_frag+1))

    gmsh.write(output_folder+"/fragments_domain.geo_unrolled")
    gmsh.model.mesh.generate(2)
    gmsh.write(output_folder+"/fragments.stl")
    gmsh.finalize()
    fragment_location = str(pathlib.Path(output_folder).resolve())
    stl_split(fragment_location,fragment_location+'/fragments.stl')
    pathlib.Path(output_folder+"/fragments.stl").unlink()



def add_voronoi_hedra(gmsh, voronoi, node_ref = 1 , edge_ref = 1, surf_ref = 1, vol_ref = 1):
    factory = gmsh.model.occ
    vol_ref_map = []
    hulls = []
    vor_nodes = None

    for i_region, region in enumerate(voronoi.regions):
        # Ignore infinite voronoi regions, these correspond to remnant fragments from after boolean ops
        if not region or -1 in region: 
            vol_ref_map.append(0)
            hulls.append(None)
            continue
        # All voronoi cells are convex polyhedra, can recover faces and edges from just points
        nodes = voronoi.vertices[region]
        hull = ConvexHull(nodes)
        hulls.append(hull)
        print('Hull volume of {}'.format(hull.volume))
        if hull.volume<1e-8:
            print('Region {} found to be low volume ({})'.format(i_region, hull.volume))
            vol_ref_map.append(0)
            hulls.append(None)
            continue
        if vor_nodes is None: vor_nodes = hull.points
        else: vor_nodes = np.vstack([vor_nodes, hull.points])
        # Scipy ConvexHull gives us simplices, can compute necessary edges and faces reasonably easily
        vol_ref_map.append(1)
    
    vert_map = {}
    edge_map = {}
    surf_map = {}
    boundary_maps = {}
    # Add all necessary hull nodes
    vor_nodes = np.unique(vor_nodes, axis=0)
    for i_vert, vert in enumerate(vor_nodes): vert_map[i_vert] = factory.addPoint(*vert)

    for i_region, region in enumerate(voronoi.regions):
        if not vol_ref_map[i_region]: continue
        print('Adding region {}/{}'.format(i_region, len(voronoi.regions)))
        hull = hulls[i_region]
        ## Process involves getting the correct windings for each triangular facet of the convex hull
        centroid = np.mean(hull.points, axis=0)
        boundary_maps[i_region] = []
        for i_tri, tri in enumerate(hull.simplices):
            vert_ids = [np.where(hull.points[tri][i_vert] == vor_nodes)[0][0] for i_vert in range(3)]
            vert_refs = [vert_map[v_id] for v_id in vert_ids]
            v0, v1, v2 = vor_nodes[vert_ids]
            normal = np.cross(v0-v2, v0-v1)
            winding_check = np.dot(normal,v0-centroid)
            if winding_check<0: vert_loop = [vert_ids[0], vert_ids[1], vert_ids[2]]
            else: vert_loop = [vert_ids[0], vert_ids[2], vert_ids[1]]

            vert_loop.append(vert_loop[0]) # Close vert loop
            # Add edges to gmsh
            edge_loop = [(vert_loop[i_vert], vert_loop[i_vert+1]) for i_vert in range(len(vert_loop[:-1]))]
            for edge in edge_loop:
                edge_list = list(edge_map.keys())
                in_edge_list = edge in edge_list
                reverse_in_edge_list = edge[::-1] in edge_list
                if (not in_edge_list) and (not reverse_in_edge_list): 
                    edge_map[edge] = factory.addLine(vert_map[edge[0]],vert_map[edge[1]])
                elif reverse_in_edge_list: edge_map[edge] = -edge_map[edge[::-1]]

            ## This is all a bit weird and hacky, open to better solutions
            vert_loop.append(vert_loop[1]) # This gives all (correctly wound) orderings of verts
            surf_list = list(surf_map.keys()) # Already existing surfs in str format
    
            str_vert_loop = ','.join([str(i) for i in vert_loop[:-2]]) # Convert target winding into a string
            in_surf_list = np.where([str_vert_loop in surf_loop for surf_loop in surf_list])[0] # Search for that string in surf list
            in_surf_list = in_surf_list[0] if len(in_surf_list)>0 else False
            
            str_reverse_surf = ','.join([str(vert_loop[0]), str(vert_loop[2]), str(vert_loop[1])]) # Convert reverse winding to str
            reverse_in_surf_list = np.where([str_reverse_surf in surf_loop for surf_loop in surf_list])[0]
            reverse_in_surf_list = reverse_in_surf_list[0] if len(reverse_in_surf_list)>0 else False

            str_full_loop = ','.join([str(i) for i in vert_loop])
            if (not in_surf_list) and (not reverse_in_surf_list):
                loop_ref = factory.addCurveLoop([edge_map[edge] for edge in edge_loop])
                surf_map[str_full_loop] = factory.addPlaneSurface([loop_ref])
                
            elif reverse_in_surf_list: surf_map[str_full_loop] = surf_map[surf_list[reverse_in_surf_list]]
            elif in_surf_list: surf_map[str_full_loop] = surf_map[surf_list[in_surf_list]]
            boundary_maps[i_region].append(surf_map[str_full_loop])
        surf_loop_ref = factory.addSurfaceLoop(boundary_maps[i_region])
        vol_ref_map[i_region] = factory.addVolume([surf_loop_ref])

    return vol_ref_map



    

        
