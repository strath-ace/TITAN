#
# Copyright (c) 2026 TITAN Contributors (cf. AUTHORS.md).
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
"""The raytracer used to compute aerothermodynamic properties"""
import numpy as np
import trimesh, pathlib
from scipy.spatial.transform import Rotation as Rot, RigidTransform as Trans
from ..Dynamics.frames import R_W_from_B
try:
    from trimesh.ray.ray_pyembree import RayMeshIntersector
except:
    print('PyEmbree/Embreex library not set up')
    from trimesh.ray.ray_triangle import RayMeshIntersector

def trace(assembly_group : list, n : int, options, output_rays : str = None):
    """Calls the ray-tracing engine to assign the aero_index parameter across an assembly ensemble

    :param assembly_group: List of assembly objects.
    :type assembly_group: list
    :param n: Number of subdivision levels.
    :type n: int
    :param options: Options object.
    :type options: Options
    :param output_rays: Whether to output debug ray geometry, either "leading" or "trailing".
    :type output_rays: str, optional"""

    # Prefilter our raytracing by flow-facing facets
    theta = [0]
    v0 = [[0,0,0]]
    v1 = [[0,0,0]]
    v2 = [[0,0,0]]
    pointers = [0]
    flow_dirs = [[0,0,0]]
    for i_assem, _assembly in enumerate(assembly_group):
        assem_body_flow_vec = -Rot.from_quat(_assembly.quaternion).inv().apply(_assembly.velocity)
        assem_body_flow_vec /= np.linalg.norm(assem_body_flow_vec)
        facet_normals = _assembly.mesh.facet_normal
        length_normals = np.linalg.norm(facet_normals, axis = 1, ord = 2)
        _assembly.aerothermo.theta = np.pi/2 - np.arccos(np.clip(np.sum(- assem_body_flow_vec * facet_normals/length_normals[:,None] , axis = 1), -1.0, 1.0))
        _assembly.freestream.per_facet_mach = np.full_like(length_normals,_assembly.freestream.mach)
        theta = np.hstack([theta,_assembly.aerothermo.theta])
        v0 = np.vstack([v0, _assembly.mesh.v0])
        v1 = np.vstack([v1,_assembly.mesh.v1])
        v2 = np.vstack([v2, _assembly.mesh.v2])
        pointers.append(pointers[-1]+len(length_normals))
        flow_dirs = np.vstack([flow_dirs,[_assembly.velocity for _ in range(len(length_normals))]])
    
    theta     = theta[1:]
    v0        = v0[1:,:]
    v1        = v1[1:,:]
    v2        = v2[1:,:]
    flow_dirs = flow_dirs[1:,:]
    filtered_facet_indices = np.where(theta>0)[0]
    
    flow_dirs = flow_dirs[filtered_facet_indices,:]

    base_assembly = assembly_group[0]
    meshlist = []

    main_Translate_ECEF = trimesh.transformations.translation_matrix(-_assembly.position)
    main_Translate_CoG  = trimesh.transformations.translation_matrix(Rot.from_quat(base_assembly.quaternion).apply(_assembly.COG))
    flow_dirs  = -flow_dirs
    flow_len   = np.linalg.norm(flow_dirs, axis = 1)
    flow_dirs /= flow_len[:,np.newaxis]

    for i_assem, _assembly in enumerate(assembly_group):
        new_mesh = trimesh.Trimesh(vertices=_assembly.mesh.nodes, faces=_assembly.mesh.facets)
        quaternion = np.append([_assembly.quaternion[3]], _assembly.quaternion[0:3])
        R_ECEF_from_B = trimesh.transformations.quaternion_matrix(quaternion)
        Translate_COG = trimesh.transformations.translation_matrix(-_assembly.COG)
        Translate_ECEF = trimesh.transformations.translation_matrix(_assembly.position)
        #
        Matrix = main_Translate_CoG@main_Translate_ECEF@Translate_ECEF@R_ECEF_from_B@Translate_COG
        new_mesh.apply_transform(Matrix)
        TransMatrix = Trans.from_matrix(Matrix)
        v0[pointers[i_assem]:pointers[i_assem+1],:] = TransMatrix.apply(
            v0[pointers[i_assem]:pointers[i_assem+1],:]
            )
        v1[pointers[i_assem]:pointers[i_assem+1],:] = TransMatrix.apply(
            v1[pointers[i_assem]:pointers[i_assem+1],:]
            )
        v2[pointers[i_assem]:pointers[i_assem+1],:] = TransMatrix.apply(
            v2[pointers[i_assem]:pointers[i_assem+1],:]
            )
        meshlist.append(new_mesh)

    
    mesh = trimesh.util.concatenate(meshlist)

    ray = RayMeshIntersector(mesh)

    facet_centroids = edge_subdivision(v0[filtered_facet_indices], 
                                       v1[filtered_facet_indices], 
                                       v2[filtered_facet_indices], n)
    for _ in range(n):
        flow_dirs = np.repeat(flow_dirs,4, axis=0)

    ray_origins = facet_centroids - 1e-4*flow_dirs
    ray_directions = -flow_dirs

    ray_directions.shape = (-1,3)
    facet_sees_flow =  np.zeros_like(theta, dtype=np.int16)

    match output_rays:
        case 'leading':
            ray_ends   = facet_centroids - 10*flow_dirs
            ray_starts = facet_centroids
        case 'trailing':
            ray_ends   = facet_centroids + 10*flow_dirs 
            ray_starts = facet_centroids
        case 'body':
            R_B_from_ECEF = Rot.from_quat(base_assembly.quaternion).inv()
            ray_ends = R_B_from_ECEF.apply(facet_centroids -10*flow_dirs)
            ray_starts = R_B_from_ECEF.apply(facet_centroids)

    if output_rays is not None:
        if not pathlib.Path(options.output_folder+'/Rays').exists():
            pathlib.Path(options.output_folder+'/Rays').mkdir(parents=True)
        if not hasattr(options, 'n_debug'): options.n_debug = 0
        else: options.n_debug+=1
        mesh.export(options.output_folder+'/Rays/debug_{}.stl'.format(options.n_debug))
        write_rays_to_vtk(options.output_folder+'/Rays/debug_rays_{}.vtk'.format(options.n_debug),ray_starts, ray_ends)

    hits  = ~ray.intersects_any(ray_origins = ray_origins, ray_directions = ray_directions)
    hits = hits.reshape([-1, 4**n])
    hits = np.sum(hits, axis = 1)
    facet_sees_flow[filtered_facet_indices] = hits

    for i_assembly, _assembly in enumerate(assembly_group):

        per_assem_see_flow = facet_sees_flow[pointers[i_assembly]:pointers[i_assembly+1]]
        _assembly.aerothermo.partial_factor = per_assem_see_flow/(4**n)

        per_assem_see_flow = np.arange(len(_assembly.mesh.facets))[per_assem_see_flow != 0]


        if options.aerothermo.recalculate_Aref or _assembly.Aref is None:
            _assembly.aerothermo.proj_area = 0
            proj_facet_areas = _assembly.mesh.facet_area[per_assem_see_flow] * np.sin(_assembly.aerothermo.theta[per_assem_see_flow])
            _assembly.aerothermo.proj_area=np.sum(proj_facet_areas)
        
            if _assembly.Aref is None:
                _assembly.Aref = _assembly.aerothermo.proj_area

        if options.aerothermo.recalculate_Lref:
            transform = Trans.from_rotation(R_W_from_B(_assembly.aoa, _assembly.slip))*Trans.from_translation(-_assembly.COG)
            projected_verts = transform.apply(np.vstack([_assembly.mesh.v0, _assembly.mesh.v1, _assembly.mesh.v2]))
            _assembly.Lref = np.linalg.norm(np.max(projected_verts[:,1:], axis=0) - np.min(projected_verts[:,1:], axis=0))

        _assembly.aero_index = per_assem_see_flow

def edge_subdivision(v0 : np.ndarray,v1 : np.ndarray,v2 : np.ndarray, n : int) -> np.ndarray:
    """Each subdivision level divides the triangle into 4 parts with equal areas

    :param v0: Array of first vertex positions.
    :type v0: np.ndarray
    :param v1: Array of second vertex positions.
    :type v1: np.ndarray
    :param v2: Array of third vertex positions.
    :type v2: np.ndarray
    :param n: Number of subdivision levels.
    :type n: int
    :return: Array of centroid positions.
    :rtype: np.ndarray"""

    def COG_subdivision(v0,v1,v2, COG, start, n, i = 1):
        """Recursively subdivides a triangle based upon centroid

        :param v0: Array of first vertex positions.
        :type v0: np.ndarray
        :param v1: Array of second vertex positions.
        :type v1: np.ndarray
        :param v2: Array of third vertex positions.
        :type v2: np.ndarray
        :param COG: Array of centroid positions.
        :type COG: np.ndarray
        :param start: Starting index for the current subdivision.
        :type start: int
        :param n: Number of subdivision levels.
        :type n: int
        :param i: Current subdivision level.
        :type i: int
        :return: Resultant subdivision level.
        :rtype: int"""
    
        v0v1 = (v0 + v1) / 2.0
        v0v2 = (v0 + v2) / 2.0
        v1v2 = (v1 + v2) / 2.0
    
        if i == n:
    
            COG[start+0::4**n,:] = (v0v1 + v0v2 + v0)/3.0
            COG[start+1::4**n,:] = (v0v1 + v1v2 + v1)/3.0
            COG[start+2::4**n,:] = (v0v2 + v1v2 + v2)/3.0
            COG[start+3::4**n,:] = (v0v1 + v0v2 + v1v2)/3.0
    
            return start + 4
    
        else:
            start = COG_subdivision(v0v1,v0v2, v0, COG, start, n, i+1)
            start = COG_subdivision(v0v1,v1, v1v2, COG, start, n, i+1)
            start = COG_subdivision(v0v2,v1v2, v2, COG, start, n, i+1)
            start = COG_subdivision(v0v1,v1v2, v0v2, COG, start, n, i+1)


    if n == 0:
        COG = (v0+v1+v2)/3.0

    else:
        COG = np.zeros((len(v0)*4**n,3))
        COG_subdivision(v0,v1,v2,COG, 0, n)

    return COG

def write_rays_to_vtk(filename : str, origins : np.ndarray, ends : np.ndarray):
    """Write ray segments to a legacy VTK PolyData file.

    :param filename: Path to the relevant file.
    :type filename: str
    :param origins: Value for origins.
    :type origins: Any
    :param ends: Value for ends.
    :type ends: Any"""
    origins = np.asarray(origins)
    ends = np.asarray(ends)

    assert origins.shape == ends.shape
    N = origins.shape[0]

    # Create point list (each ray = 2 points)
    points = np.vstack([origins, ends])
    
    # Connectivity: each ray is a 2-point polyline
    # VTK polyline format:  "2 i j"
    lines = []
    for i in range(N):
        start = i
        end = i + N
        lines.append(f"2 {start} {end}")

    with open(filename, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Rays\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")

        # Write points
        f.write(f"POINTS {2*N} float\n")
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

        # Write lines
        f.write(f"LINES {N} {3*N}\n")
        for line in lines:
            f.write(line + "\n")

    print(f"[write_rays_to_vtk] Wrote {N} rays to {filename}")