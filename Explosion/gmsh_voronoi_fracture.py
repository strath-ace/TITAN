import gmsh
import numpy as np
from Geometry.gmsh_api import mesh_Settings, object_grid, object_physical
from scipy.spatial import ConvexHull

## Mesh api, modified from the inner domain, does not need to be as robust to assembly configuration as it will only be called on objects
def generate_inner_domain(mesh, obj, voronoi, write = False, output_folder = '', output_filename = '', ref_obj_override=None):
    gmsh.initialize()
    mesh_Settings(gmsh)

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

    #map_objects = dict()
    
    gmsh.model.mesh.createGeometry()
    #Change refs to refine joints as well
    ref = np.ones(len(mesh.nodes))*ref_size



    node_ref_init, edge_ref_init, surf_ref_init = object_grid(gmsh, mesh.nodes, mesh.edges, mesh.facet_edges, ref, factory=factory)
    init_ref_surf, ref_phys_surface = object_physical(gmsh, init_ref_surf, surf_ref_init, ref_phys_surface, 'top', factory=factory)


    out = factory.addSurfaceLoop(np.arange(1, surf_ref_init))
    #surf_ref_init = 3*edge_ref_init - surf_ref_init-1
    if obj.inner_mesh:
        raise NotImplementedError
        node_ref_end , edge_ref_end, surf_ref_end = object_grid(gmsh,obj.inner_mesh.nodes, obj.inner_mesh.edges, 
                                                                obj.inner_mesh.facet_edges, ref,node_ref_init, 
                                                                edge_ref_init, surf_ref_init)
        #assembly.objects[i].inner_node_index = np.array(range(node_ref_init-1, node_ref_end-1))
        hole = gmsh.model.geo.addSurfaceLoop(range(surf_ref_init, surf_ref_end-1))
        init_ref_surf, ref_phys_surface = object_physical(gmsh, surf_ref_init, surf_ref_end, ref_phys_surface, 'bottom')
        vol_tag = gmsh.model.geo.addVolume([out,hole])
        
        node_ref_init = node_ref_end
        edge_ref_init = edge_ref_end
        surf_ref_init = surf_ref_end

    else:
        vol_tag = factory.addVolume([out])
    
    obj.vol_tag = vol_tag
    node_ref_end, edge_ref_end, surf_ref_end, vol_tag_end, vol_map = add_voronoi_hedra(gmsh, voronoi, node_ref_init, edge_ref_init, 
                                                                              surf_ref_init, vol_tag+1, factory=factory)
    gmsh.model.geo.synchronize()
    new_frag_tags, map_obj, map_hedra = gmsh.model.occ.fragment([(3,obj.vol_tag)], [(3,hedra_tag) for hedra_tag in np.arange(vol_tag+1,vol_tag_end)])

    gmsh.model.occ.synchronize()

    ref_phys_volume = gmsh.model.geo.addPhysicalGroup(3, [vol_tag])
    gmsh.model.setPhysicalName(3, ref_phys_volume, str(i+1))  

    gmsh.model.geo.synchronize()
    gmsh.write("fragments_domain.geo_unrolled")
    gmsh.model.mesh.generate(3)

    if False:
        gmsh.fltk.initialize()
        while gmsh.fltk.isAvailable() and checkForEvent():
            gmsh.fltk.wait()
   
    entities = gmsh.model.getEntities()

    if write: 
        gmsh.write(output_folder +'/Volume/'+'%s_%s_fragments.vtk'%(output_filename, obj.name.split('/')[-1].split('.')[0]))

    gmsh.finalize()

def add_voronoi_hedra(gmsh, voronoi, node_ref = 1 , edge_ref = 1, surf_ref = 1, vol_ref = 1, debug_sphere_rad = None, factory = None):
    if factory is None: factory = gmsh.model.geo 
    vol_ref_map = []

    if debug_sphere_rad is not None: 
        gmsh.model.occ.add_sphere(0,0,0,debug_sphere_rad)
        node_ref+=2
        edge_ref+=1
        surf_ref+=1
        vol_ref+=1
    i_region = 0
    fragment_surf_refs = 1
    for region in voronoi.regions:
        # Ignore infinite voronoi regions, these correspond to remnant fragments from after boolean ops
        i_region+=1
        if not region or -1 in region: 
            vol_ref_map.append(-1)
            continue
        print('Adding region {}/{}'.format(i_region, len(voronoi.regions)))
        # All voronoi cells are convex polyhedra, can recover faces and edges from just points
        nodes = voronoi.vertices[region]
        hull = ConvexHull(nodes)

        # Scipy ConvexHull gives us simplices, can compute necessary edges and faces reasonably easily
        edges = []
        facets_edges = []

        for simp in hull.simplices:
            simplex = [[simp[0], simp[1]],
                       [simp[1], simp[2]],
                       [simp[2], simp[0]]]
            
            facet_edges = []
            for edge in simplex:
                edge_in_list = (edge in edges)
                reverse_edge_in_list = (edge[::-1] in edges)

                if (not edge_in_list) and (not reverse_edge_in_list): 
                    edges.append(edge)

                if reverse_edge_in_list: 
                    facet_edges.append(-(edges.index(edge[::-1])+1))
                else: facet_edges.append(edges.index(edge)+1)

            facets_edges.append(facet_edges)
        ref = np.ones(hull.points.shape[0])
        n_surfs = np.array(facet_edges).shape[0]
        surf_dev = surf_ref
        node_ref, edge_ref, surf_ref = object_grid(gmsh=gmsh, nodes=nodes, edges=np.array(edges), facet_edges=np.array(facets_edges), 
                                                     ref=ref, node_ref=node_ref, edge_ref=edge_ref, surf_ref=surf_ref, factory=factory)
        # For baffling reasons the surfs reference works 
        vol_ref = factory.addSurfaceLoop(np.arange(surf_dev, surf_ref), vol_ref)
        factory.addVolume([vol_ref])
        vol_ref_map.append(vol_ref)
        vol_ref+=1
    return node_ref, edge_ref, surf_ref, vol_ref, vol_ref_map