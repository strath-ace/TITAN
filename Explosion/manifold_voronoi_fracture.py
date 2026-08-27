from scipy.spatial import ConvexHull
import numpy as np
import trimesh, manifold3d
import glob, subprocess, pathlib
import pandas as pd
from scipy import stats

def bb_check(bb_candidate, bb_to_query, use_any=True):
    if bb_to_query is None:  return True
    bb_candidate = np.array(bb_candidate)
    bb_to_query = np.array(bb_to_query)
    dimension = bb_candidate.shape[1]
    check = np.any if use_any else np.all
    is_in = np.full(dimension, False)
    for dim in range(dimension):
        is_in[dim] = check(np.logical_and(bb_candidate[:,dim] < bb_to_query[1,dim],
                                          bb_candidate[:,dim] > bb_to_query[0,dim]))
    return np.all(is_in)

def generate_fragment_meshes(stl, voronoi, write_dir, extrude=3e-2, verbose=False, nucleus=None, rng=None, n=64):
    base_mesh = trimesh.load_mesh(stl)
    manifold_mesh = manifold3d.Mesh(vert_properties=np.array(base_mesh.vertices),tri_verts=np.array(base_mesh.faces))
    base_manifold = manifold3d.Manifold(manifold_mesh)
    bool_list = [base_manifold]
    if verbose: print('Building prisms')
    if voronoi is not None:
        prisms = build_voro_prisms(voronoi, base_mesh.bounding_box.bounds, extrude=extrude)
    else:
        if rng is None: rng = np.random.default_rng()
        if nucleus is None: 
            distris = [stats.uniform() for _ in range(3)]
            for d in distris: d.random_state = rng
        else:
            focus = (nucleus - base_mesh.bounding_box.bounds[0]) / (base_mesh.bounding_box.bounds[1] - base_mesh.bounding_box.bounds[0])
            distris = []
            for axis in range(3):
                scale = 0.2
                a, b = (0 - focus[axis]) / scale, (1 - focus[axis]) / scale
                distris.append(stats.truncnorm(loc=focus[axis], scale=scale, a=a, b=b))
                distris[-1].random_state=rng
            print(nucleus, focus, base_mesh.bounding_box.bounds)
        prisms = orthogonal_planes(base_mesh.bounding_box.bounds, rng, extrude=extrude,  n=n, distris=distris)
    prism_tool = manifold3d.Manifold.batch_boolean(prisms, manifold3d.OpType.Add).to_mesh()
    trimesh.Trimesh(vertices=prism_tool.vert_properties, faces=prism_tool.tri_verts).export('Planes.stl')
    [bool_list.append(prism) for prism in prisms]
    if verbose: print('Performing boolean')
    # for i_prism, prism in enumerate(prisms[1:]):
    #     if i_prism % np.floor(0.1*len(prisms)) == 0 and verbose:
    #         print('...',np.round(100*i_prism/len(prisms)),'%')
    #     bool_mani = bool_mani + prism
    bool_mani = manifold3d.Manifold.batch_boolean(bool_list, manifold3d.OpType.Subtract)
    if verbose: print('Converting manifold to mesh')
    #bool_mani = bool_mani + bool_mani
    

    mesh = bool_mani.set_tolerance(1e-5).to_mesh()# bool_mani.to_mesh()

    bool_mesh = trimesh.Trimesh(vertices=mesh.vert_properties, faces=mesh.tri_verts)
    bool_mesh.export('prisms0.stl')
    splits = bool_mesh.split()
    n_frags = 0
    n_fails = 0
    for split in splits:
        if np.abs(split.volume)>5e-7 and split.is_watertight:
            split.export('{}/frag_{}.stl'.format(write_dir, n_frags))
            n_frags+=1
        else:
            continue
            split.export('{}/fail_{}.stl'.format(write_dir, n_fails))
            n_fails += 1

def build_voro_prisms(voronoi, bb = None, extrude=1e-2, use_trimesh=True):
    prisms = [] # Final built tri-prisms to use in boolean
    surf_list = [] # Already existing surfs in str format
    for i_region, region in enumerate(voronoi.regions):
        ## We can't create an infinite volume
        if not region or -1 in region: continue

        ## No point in creating volumes with no intersecting nodes
        nodes = voronoi.vertices[region]
        if not bb_check(nodes, bb): continue

        ## Finally if volume is degenerate we also skip
        hull = ConvexHull(nodes)
        if hull.volume<1e-8: continue
        
        centroid = np.mean(nodes,axis=0)
        v_ids = []
        normals = {}
        v_normals = {}
        v_loops = {}
        for i_tri, tri in enumerate(hull.simplices):
            v0, v1, v2 = hull.points[tri]
            normals[i_tri] = np.cross(v0-v2, v0-v1)
            normals[i_tri] /= np.linalg.norm(normals[i_tri])
            
        for i_tri, tri in enumerate(hull.simplices):
            verts = hull.points[tri]
            vert_ids = [np.where(np.all(voronoi.vertices==vert,axis=1))[0][0] for vert in verts]
            v0, v1, v2 = verts
            face_centroid = (v0 + v1 + v2) / 3.0
            winding_check = np.dot(normals[i_tri],face_centroid-centroid)
            if winding_check > 0: 
                vert_loop = [vert_ids[0], vert_ids[1], vert_ids[2]]
                
            else: 
                vert_loop = [vert_ids[0], vert_ids[2], vert_ids[1]]
                #normals[i_tri] *= -1
                #
            v_loops[i_tri] = vert_loop
                # Each vert is connected to at least 3 triangles
        for i_vert, vert_id in enumerate(hull.points):
            vert_normal = np.zeros(3)
            tri_adjacent_to_vert = np.where([i_vert in simp for simp in hull.simplices])[0]
            for adj_tri in tri_adjacent_to_vert:
                vert_normal += normals[adj_tri]
            vert_normal *= 1 / np.linalg.norm(vert_normal)
            v_normals[i_vert] = vert_normal


        for i_tri, tri in enumerate(hull.simplices):
            vert_loop = v_loops[i_tri]
            verts = hull.points[tri]
            ## This is all a bit weird and hacky, open to better solutions
            ## Essentially we define a surface by its vert winding, encoded as a string
            ## Then all vert orderings can be found as substrings in either the base surface or its reverse
            vert_loop.append(vert_loop[0]) # Close vert loop
            vert_loop.append(vert_loop[1]) # This gives all (correctly wound) orderings of verts
    
            str_vert_loop = ','+','.join([str(i) for i in vert_loop[:-2]])+',' # Convert target winding into a string
            in_surf_list = np.where([str_vert_loop in surf_loop for surf_loop in surf_list])[0] # Search for that string in surf list
            in_surf_list = in_surf_list[0] if len(in_surf_list)>0 else False
            
            str_reverse_surf = ','+','.join([str(vert_loop[0]), str(vert_loop[2]), str(vert_loop[1])])+',' # Convert reverse winding to str
            reverse_in_surf_list = np.where([str_reverse_surf in surf_loop for surf_loop in surf_list])[0]
            reverse_in_surf_list = reverse_in_surf_list[0] if len(reverse_in_surf_list)>0 else False

            str_full_loop = ','+','.join([str(i) for i in vert_loop])+','

            if (not in_surf_list) and (not reverse_in_surf_list):



                vert_normals = [v_normals[tri[0]],v_normals[tri[1]],v_normals[tri[2]]]
                if extrude>0:
                # For each tri we symmetrically extrude it to make a triangular prism
                    prism_verts, prism_faces = extrude_prism_from_verts(verts, extrude, normals[i_tri], vert_normals, debug_trimesh=True)
                    prism = manifold3d.Manifold(manifold3d.Mesh(vert_properties=prism_verts, 
                                                            tri_verts=prism_faces))#trimesh.Trimesh(vertices=prism_verts, faces=prism_faces)
                else:
                    prism = manifold3d.Manifold(manifold3d.Mesh(vert_properties=verts, tri_verts=np.array([[0,1,2]])))
                prisms.append(prism)
                # if (not prism.is_volume) or (not prism.is_watertight) or (not prism.is_winding_consistent):
                #     print('Error on prism {}\n Volume : {}\n Watertight : {}\n Winding : {}'.format(len(prisms),
                #                                                                                     prism.is_volume,
                #                                                                                     prism.is_watertight,
                #                                                                                     prism.is_winding_consistent))
                #     prism.export('Error_'+str(len(prisms))+'.stl')
               
                surf_list.append(str_full_loop)
    return prisms

def orthogonal_planes(bounding_box, rng, n=50, extrude=1e-2, distris = None):
    if distris is None: raise Exception('Must provide 3 distributions!')
    prisms = []
    normals = np.eye(3)
    normals[1] *= -1

    for i_plane in range(n):
        axis = rng.choice(3)
        print('Selecting from domain {} for axis {} (nucleus at {})'.format(distris[axis].support(),axis, distris[axis].mean()))
        height = bounding_box[0][axis]+distris[axis].rvs()*(bounding_box[1][axis]-bounding_box[0][axis])#
        print('Selected point {}'.format(height))
        quad = np.zeros([4,3])
        quad[:,axis] = height*np.ones(4)
        flip = False
        for i_ax in range(3):
            if not i_ax==axis and not flip:
                quad[0,i_ax] = bounding_box[0][i_ax]
                quad[1,i_ax] = bounding_box[0][i_ax]
                quad[2,i_ax] = bounding_box[1][i_ax]
                quad[3,i_ax] = bounding_box[1][i_ax]
                flip = True
            elif not i_ax==axis and flip:
                quad[0,i_ax] = bounding_box[0][i_ax]
                quad[1,i_ax] = bounding_box[1][i_ax]
                quad[2,i_ax] = bounding_box[0][i_ax]
                quad[3,i_ax] = bounding_box[1][i_ax]
        prism_verts, prism_faces = extrude_prism_from_verts(quad[:3,:], 
                                                            extrude, 
                                                            normal=normals[axis],
                                                            vert_normals=[normals[axis] for _ in range(3)], 
                                                            debug_trimesh=True)
        prisms.append(manifold3d.Manifold(manifold3d.Mesh(vert_properties=prism_verts, 
                                                            tri_verts=prism_faces)))
        
        prism_verts, prism_faces = extrude_prism_from_verts(quad[:0:-1,:], 
                                                            extrude, 
                                                            normal=normals[axis],
                                                            vert_normals=[normals[axis] for _ in range(3)])
        prisms.append(manifold3d.Manifold(manifold3d.Mesh(vert_properties=prism_verts, 
                                                            tri_verts=prism_faces)))
    return prisms

def intersect_aabb_with_plane(aabb, plane_normal, plane_offset):

    aabb_points = np.array([[aabb[0][0], aabb[0][1], aabb[0][2]],
                            [aabb[0][0], aabb[1][1], aabb[0][2]],
                            [aabb[0][0], aabb[0][1], aabb[1][2]],
                            [aabb[0][0], aabb[1][1], aabb[1][2]],
                            [aabb[1][0], aabb[0][1], aabb[0][2]],
                            [aabb[1][0], aabb[1][1], aabb[0][2]],
                            [aabb[1][0], aabb[0][1], aabb[1][2]],
                            [aabb[1][0], aabb[1][1], aabb[1][2]],
                            ])
    
    ## Traverse the aabb through its edges caputring points that intersect the plane
    edges = [[0,1],[0,2],[1,3],[2,3],
             [0,4],[4,6],[2,6],
             [1,5],[3,7],[5,7],
             [4,5],[6,7]]
    intersects = []
    for edge in edges:
        min = aabb_points[edges[0]]
        max = aabb_points[edges[1]]

        min_dist = np.dot(plane_normal,min)+plane_offset
        max_dist = np.dot(plane_normal,max)+plane_offset

        if np.sign(min_dist)==np.sign(max_dist): 
            if min_dist==0 and max_dist==0: continue
            intersects.append(min+min_dist)

        else: pass

    
def extrude_prism_from_verts(verts, base_extrude_height, normal, vert_normals = None, bias = 0.5, scale=1.0, debug_trimesh=False):
    ## Make sure our normals are well-aligned
    aligned_0 = np.dot(normal, vert_normals[0])
    aligned_1 = np.dot(normal, vert_normals[1])
    aligned_2 = np.dot(normal, vert_normals[2])

    if aligned_0<0: vert_normals[0] *= -1
    if aligned_1<0: vert_normals[1] *= -1
    if aligned_2<0: vert_normals[2] *= -1

    
    # Outside face verts
    v_o = np.zeros_like(verts)
    extrude_height = base_extrude_height#/scale
    for i_vert, v_pos in enumerate(verts):
        v_o[i_vert,:] = v_pos+bias*extrude_height*vert_normals[i_vert]

    # Inside face verts
    v_i = np.zeros_like(verts)

    for i_vert, v_pos in enumerate(verts):
        v_i[i_vert,:] = v_pos-(1-bias)*extrude_height*vert_normals[i_vert]
        # if i_vert==0: ## Hack to break degeneracy
        #     v_i[i_vert,0] += 1e-7
    ## Collate verts
    v = np.vstack([v_o,v_i])
    faces = np.array([[0, 2, 1],  # Top Face
                      [3, 4, 5],  # Bottom Face
                      [3, 0, 1],  # Upper 0-1 Face
                      [3, 1, 4],  # Lower 0-1 Face
                      [4, 1, 2],  # Upper 1-2 Face
                      [4, 2, 5],  # Lower 1-2 Face
                      [5, 2, 0],  # Upper 2-0 Face
                      [5, 0, 3]]) # Lower 2-0 Face
    
    # if np.dot(normal, vert_normals[0])<0: # Normal flip
    #     for f in faces: f[[1,2]] = f[2:0:-1]
    if not scale==1:
        centroid = np.mean(v, axis=0)
        r_v = v - centroid
        r_v *= scale

        v = r_v + centroid
        #assert np.isclose(np.linalg.norm(v[0,:]-v[3,:]),base_extrude_height)
    
    if debug_trimesh:
        mesh = trimesh.Trimesh(vertices=v, faces=faces)
        mesh.export('debug.stl')
    return v, faces

def mesh_check(folder, density, target_volume=None, threshold=10.0, delete_bad=False, quiet=True):
    all_valid = True
    if density==0: density = 2700
    for frag in glob.glob("{}/*.stl".format(folder)):
        if quiet:
            proc = subprocess.run(['blender','-b','-P','./Explosion/stl_verify_blender.py','--',
                                    '--file={}'.format(frag),
                                    '-d={}'.format(density),
                                    '--stats='+folder+'/stats.csv',
                                    '-m={}'.format(1e-4),
                                    '-o'], stdout=subprocess.DEVNULL)
        else:
            proc = subprocess.run(['blender','-b','-P','./Explosion/stl_verify_blender.py','--',
                                    '--file={}'.format(frag),
                                    '-d={}'.format(density),
                                    '--stats='+folder+'/stats.csv',
                                    '-m={}'.format(1e-4),
                                    '-o'])
            
        if proc.returncode==1:
            if not delete_bad: pass#return False
            else: pathlib.Path(frag).unlink()

    if target_volume is not None and threshold>0:
        fragment_volume = np.sum(pd.read_csv(folder+'/stats.csv')['volume'].to_numpy())
        pct_error = 100 * fragment_volume/target_volume
        pct_error = 100 - pct_error if pct_error<100 else pct_error - 100
        if pct_error>threshold: all_valid = False
        if not quiet: print('Percentage error of {}%'.format(pct_error))
    return all_valid
