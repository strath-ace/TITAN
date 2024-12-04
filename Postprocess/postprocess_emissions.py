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

from Thermal import thermal
from Output import output
import numpy as np
import pandas as pd
import pickle
import os
import glob
from scipy.spatial.transform import Rotation as Rot
from Aerothermo import aerothermo as Aerothermo
from Aerothermo import switch as Switch
import vg
from copy import deepcopy
from Dynamics import euler, frames
import sympy
from sympy import sqrt,tan
import meshio
from pathlib import Path
from joblib import Parallel, delayed
from scipy.spatial import KDTree


def postprocess_emissions(options):


	print('Computing emissions ...')

	path = options.output_folder+'/Data/*'
	search_string = 'thermal'
	
	# Get a list of all files in the folder
	files = glob.glob(path)
	
	# Iterate and delete each file that contains the search_string
	for file in files:
		#print(os.path.basename(file))
		if os.path.isfile(file) and search_string in os.path.basename(file):
			os.remove(file)

	data = pd.read_csv(options.output_folder+'/Data/data.csv', index_col = False)

	iter_interval = np.unique(data['Iter'].to_numpy())

	if options.radiation.spectral_freq%options.save_freq != 0:
		print('No available solutions for the chosen frequency.');exit()
	
	for iter_value in range(1, max(iter_interval)+2, options.radiation.spectral_freq):
		iter_value = int(iter_value)
		titan = read_state(options, iter_value)
		line_of_sight(titan, options, iter_value); exit()
		emissions(titan, options)
		#output.generate_surface_solution(titan = titan, options = options, folder = 'Postprocess_emissions')

def read_state(options, i = 0):
    """
    Load state of the TITAN object for the given iteration

    Returns
    -------
    titan: Assembly_list
        Object of class Assembly_list
    """

    print("Reading state Assembly_State_.p, iter:", i)

    infile = open(options.output_folder + '/Restart/'+ 'Assembly_State_'+str(i)+'_.p','rb')
    titan = pickle.load(infile)
    infile.close()

    return titan

def emissions(titan, options):

	blackbody_emissions = np.zeros(len(options.radiation.wavelengths))
	particle_emissions  = np.zeros(len(options.radiation.wavelengths))

	phi   = options.radiation.phi
	theta = options.radiation.theta

	#define viewpoint on the basis of angles
	viewpoint = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

	for assembly in titan.assembly:

		R_B_ECEF = Rot.from_quat(assembly.quaternion_prev)

		#Transform assembly from body to ECEF frame
		assembly.mesh.facet_normal = -R_B_ECEF.apply(assembly.mesh.facet_normal)

		#retrive index of facets seen from viewpoint
		index = Aerothermo.ray_trace(assembly, viewpoint)

		#calculate angle of facets seen from viewpoint, relative to view direction
		vec1 = -viewpoint
		vec2 = np.array(assembly.mesh.facet_normal[index])
		angle = vg.angle(vec1, vec2) #degrees

		if options.radiation.spectral:
			blackbody_emissions = thermal.compute_black_body_spectral_emissions(assembly, options.radiation.wavelengths, index, angle, blackbody_emissions)

			print('blackbody_emissions:', blackbody_emissions)

		if options.radiation.spectral and options.thermal.ablation and options.radiation.particle_emissions and options.pato.flag:
			thermal.compute_particle_spectral_emissions(assembly, options.radiation.wavelengths, index, angle, particle_emissions)

def line_of_sight(titan, options, iteration):

	print('Calculating line-of-sight ...')

	titan_windframe = deepcopy(titan)

	for index, assembly in enumerate(titan_windframe.assembly):

		R_B_ECEF = Rot.from_quat(assembly.quaternion_prev)

		lat = assembly.trajectory.latitude
		lon = assembly.trajectory.longitude
		chi = assembly.trajectory.chi
		gamma = assembly.trajectory.gamma

		R_ECEF_NED = frames.R_NED_ECEF(lat = lat, lon = lon).inv()
		R_NED_W = frames.R_W_NED(ha = chi, fpa = gamma).inv()
		R_ECEF_W = R_NED_W*R_ECEF_NED

		M = assembly.freestream.mach
		theta = 0.0001

		p = np.where(assembly.aerothermo.theta*180/np.pi > 1e-3)[0]

		for index_object,obj in enumerate(assembly.objects):

			print("\nObject:", obj.name)

			obj.mesh.nodes = R_B_ECEF.apply(obj.mesh.nodes)
			obj.mesh.nodes = (R_ECEF_W).apply(obj.mesh.nodes)

			obj.mesh.facet_normal =   R_B_ECEF.apply(obj.mesh.facet_normal)
			obj.mesh.facet_normal = (R_ECEF_W).apply(obj.mesh.facet_normal)

			obj.mesh.facet_COG =   R_B_ECEF.apply(obj.mesh.facet_COG)
			obj.mesh.facet_COG = (R_ECEF_W).apply(obj.mesh.facet_COG)

			obj.LOS = np.zeros(len(obj.mesh.facets))

			min_coords = np.min(obj.mesh.nodes, axis = 0)
			max_coords = np.max(obj.mesh.nodes, axis = 0)         

			#Creation of the virtual Sphere
			center = np.zeros((3))
			center[1:] = (min_coords[1:]+max_coords[1:])/2.0
			center[0] = max_coords[0]

			dist_center = np.linalg.norm(obj.mesh.nodes[:,1:] - center[1:], axis = 1)
			radius = np.max(dist_center)

			xmax = np.max(obj.mesh.nodes , axis = 0)
			xmin = np.min(obj.mesh.nodes , axis = 0)

			Lref = np.max(xmax-xmin)

			#Compute billig formula and retrieve the bodies that are inside the computed shock envelopes
			Switch.sphere_surface(radius, center, index, index_object, titan.iter, assembly, options)
			billig_points, billig_facets = compute_billig(M, theta, center, radius, index, Lref, index_object, titan.iter, assembly.freestream, options)

			index_aero_obj = np.intersect1d(p, obj.facet_index)

			index_aero_obj = np.where(np.isin(obj.facet_index, index_aero_obj))[0]

			obj.LOS[index_aero_obj] = compute_shock_distance(obj, index_aero_obj, billig_points, billig_facets)

			output.generate_surface_solution_object(obj, obj.LOS, options, iter_value = iteration, folder = 'Postprocess_emissions')

#def intersect_line_triangle(line_point, line_dir, triangle):
#    """
#    Compute the intersection of a line with a triangle in 3D space.
#
#    Parameters:
#        line_point (numpy.ndarray): A point on the line (1x3).
#        line_dir (numpy.ndarray): Direction vector of the line (1x3).
#        triangle (numpy.ndarray): 3x3 array of triangle vertices.
#
#    Returns:
#        tuple: (intersection_point, distance), or (None, None) if no intersection.
#    """
#    EPSILON = 1e-8
#    v0, v1, v2 = triangle  # Triangle vertices
#    edge1 = v1 - v0
#    edge2 = v2 - v0
#    h = np.cross(line_dir, edge2)
#    a = np.dot(edge1, h)
#
#    if -EPSILON < a < EPSILON:
#        return None, None  # Line is parallel to the triangle
#
#    f = 1.0 / a
#    s = line_point - v0
#    u = f * np.dot(s, h)
#
#    if u < 0.0 or u > 1.0:
#        return None, None  # Intersection lies outside the triangle
#
#    q = np.cross(s, edge1)
#    v = f * np.dot(line_dir, q)
#
#    if v < 0.0 or u + v > 1.0:
#        return None, None  # Intersection lies outside the triangle
#
#    t = f * np.dot(edge2, q)
#
#    if t > EPSILON:
#        intersection_point = line_point + t * line_dir
#        distance = np.linalg.norm(intersection_point - line_point)
#        return intersection_point, distance
#
#    return None, None  # No intersection
#
#def compute_distance_for_facet(cog, normal, coord, triangles):
#    """
#    Compute the distance for a single facet to the nearest intersection with triangles.
#
#    Parameters:
#        cog (numpy.ndarray): Center of gravity of the facet.
#        normal (numpy.ndarray): Normal vector of the facet.
#        coord (numpy.ndarray): nx3 array of vertices of the first surface.
#        triangles (list of tuples): List of triangles (each as a tuple of three vertex indices).
#
#    Returns:
#        float: Distance to the nearest intersection, or float('inf') if no intersection.
#    """
#    # Normalize the normal vector
#    normal_unit = normal / np.linalg.norm(normal)
#
#    for triangle_indices in triangles:
#        triangle = coord[list(triangle_indices)]
#        intersection, distance = intersect_line_triangle(cog, normal_unit, triangle)
#
#        if intersection is not None:
#            return distance  # Return distance immediately if intersection is found
#
#    return float('inf')  # No intersection found
#
#def compute_shock_distance(obj, index, coord, triangles, parallel=True):
#    """
#    Computes the shock distances for facets of a second surface to the first surface (triangular facets).
#
#    Parameters:
#        obj: Object containing mesh information.
#        index (numpy.ndarray): Indices of the facets to compute distances for.
#        coord (numpy.ndarray): nx3 array of vertices of the first surface.
#        triangles (list of tuples): List of triangles (each as a tuple of three vertex indices).
#        parallel (bool): Whether to use parallelization for the computation.
#
#    Returns:
#        numpy.ndarray: m-element array of distances from the facet centers to the intersection points.
#    """
#    print("Calculating shock distance ...")
#
#    facet_COG = obj.mesh.facet_COG[index]
#    facet_normal = obj.mesh.facet_normal[index]
#
#    if parallel:
#        # Use parallel computation
#        distances = Parallel(n_jobs=-1)(
#            delayed(compute_distance_for_facet)(cog, normal, coord, triangles)
#            for cog, normal in zip(facet_COG, facet_normal)
#        )
#    else:
#        # Sequential computation with early exit
#        distances = np.zeros(facet_COG.shape[0])
#        for i, (cog, normal) in enumerate(zip(facet_COG, facet_normal)):
#            distances[i] = compute_distance_for_facet(cog, normal, coord, triangles)
#
#    return np.array(distances)


def precompute_bvh(coord, triangles):
    """
    Precomputes a BVH using PyEmbree.

    Parameters:
        coord (numpy.ndarray): nx3 array of vertices.
        triangles (numpy.ndarray): kx3 array of triangle indices.

    Returns:
        Scene: PyEmbree scene containing the BVH.
    """
    # Create a triangle mesh
    mesh = TriangleMesh(vertices=coord, indices=triangles)

    # Create and return an Embree scene
    scene = Scene()
    scene.add_geometry(mesh)
    scene.commit()
    return scene

def compute_ray_intersection(scene, cog, normal):
    """
    Computes the intersection point of a ray with the surface using BVH.

    Parameters:
        scene (Scene): PyEmbree scene containing the BVH.
        cog (numpy.ndarray): 1x3 center of gravity of the facet.
        normal (numpy.ndarray): 1x3 normal vector of the facet.

    Returns:
        float: Distance to the intersection point, or float('inf') if none.
    """
    # Normalize the normal vector
    normal_unit = normal / np.linalg.norm(normal)

    # Define the ray
    ray = {
        "origin": cog,
        "direction": normal_unit,
        "tnear": 0.0,  # Start of the ray
        "tfar": float("inf"),  # Max range
        "flags": 0,
    }

    # Cast the ray into the scene
    hit = scene.run(ray)

    if hit["geomID"] != -1:  # Valid intersection
        return hit["tfar"]  # Distance to the intersection
    return float("inf")  # No intersection

def compute_shock_distances_embree(obj, index, coord, triangles, parallel=True):
    """
    Computes shock distances for facets using BVH-based ray casting.

    Parameters:
        obj: Object containing mesh information.
        index (numpy.ndarray): Indices of the facets to compute distances for.
        coord (numpy.ndarray): nx3 array of vertices.
        triangles (numpy.ndarray): kx3 array of triangle indices.
        parallel (bool): Whether to use parallel computation.

    Returns:
        numpy.ndarray: Array of distances for each facet.
    """
    print("Calculating shock distances using BVH...")

    # Precompute BVH
    scene = precompute_bvh(coord, triangles)

    # Extract facets' center of gravity and normals
    facet_COG = obj.mesh.facet_COG[index]
    facet_normal = obj.mesh.facet_normal[index]

    if parallel:
        # Parallel computation using joblib
        distances = Parallel(n_jobs=-1)(
            delayed(compute_ray_intersection)(scene, cog, normal)
            for cog, normal in zip(facet_COG, facet_normal)
        )
    else:
        # Sequential computation
        distances = [
            compute_ray_intersection(scene, cog, normal)
            for cog, normal in zip(facet_COG, facet_normal)
        ]

    return np.array(distances)

def compute_shock_distance(obj, index, coord, triangles):
    """
    Computes shock distances using Trimesh's built-in ray tracing.

    Parameters:
        obj: Object containing mesh information.
        index (numpy.ndarray): Indices of the facets to compute distances for.
        coord (numpy.ndarray): nx3 array of vertices.
        triangles (numpy.ndarray): kx3 array of triangle indices.

    Returns:
        numpy.ndarray: Array of distances for each facet.
    """
    from trimesh import Trimesh

    print("Calculating shock distances using Trimesh...")
    mesh = Trimesh(vertices=coord, faces=triangles)

    # Extract facets' center of gravity and normals
    facet_COG = obj.mesh.facet_COG[index]
    facet_normal = obj.mesh.facet_normal[index]

    distances = []
    for cog, normal in zip(facet_COG, facet_normal):
        # Ray origins and directions
        ray_origins = np.array([cog])
        ray_directions = np.array([normal / np.linalg.norm(normal)])

        # Perform ray intersection
        locations, _, _ = mesh.ray.intersects_location(ray_origins, ray_directions)
        if len(locations) > 0:
            distance = np.linalg.norm(locations[0] - cog)
        else:
            distance = float("inf")
        distances.append(distance)

    return np.array(distances)


def compute_billig(M,theta, center, sphere_radius, index_assembly, Lref, index_object, i, freestream, options):
    """
    Computation of the shock envelope using the Billing formula

    if the object is inside the shock envelope generated by an upstream body, the framework will use the high-fidelity methodology to compute the aero
    thermodynamics. Else, it will use low-fidelity methodology

    Parameters
    ----------
    M: float
        Freestream Mach number
    theta: float
        Shockwave inclination angle
    center: np.array()
        Coordinates of the sphere center
    sphere_radius: float
        Radius of the sphere
    index_assembly: int
        Index of the assembly producing the shockwave
    assembly: List_Assembly
        Object of List_Assembly
    list_assembly: np.array()
        Index of the remaining assemblies to check if they are inside or outside the shock envelope

    Returns
    -------
    computational_domain_bodies: List
        List of bodies inside the shock envelope
    
    """

    print("Calculating Billig ...")

    delta = sphere_radius*(0.143*np.exp(3.24/(M**2)))
    Rc = sphere_radius*(1.143*np.exp(0.54/(M-1)**1.2))

    x_coord = np.array([])
    y_coord = np.array([])
    z_coord = np.array([])
    cells = []

    r = sympy.Symbol("r")
    x_limit = 1.2*Lref

    #Blast Wave implementation here
    #x_limit = Switch.compute_blast_wave_limit(sphere_radius, freestream, options) - delta - sphere_radius

    exp = 1*(sphere_radius+delta-Rc*(1/tan(theta))**2*(sqrt(1+(r**2)*(tan(theta)**2)/(Rc**2))-1))+x_limit
    sol = sympy.solve(exp)

    num_points_r = 15 #50
    r = float(abs(sol[0]))
    r = np.linspace(0,r,num_points_r)

    num_points = 15 #36
    angle = np.linspace(0,2*np.pi,num_points+1)[0:-1]


    for index,_r in enumerate(r):
        x = 1*(sphere_radius+delta-Rc*(1/np.tan(theta))**2*(np.sqrt(1+(_r**2)*(np.tan(theta)**2)/(Rc**2))-1))
        y = _r * np.sin(angle)
        z = _r * np.cos(angle)

        x_coord = np.append(x_coord,np.repeat(x,num_points)+center[0])
        y_coord = np.append(y_coord,y+center[1]) 
        z_coord = np.append(z_coord,z+center[2])

        if index == 0:
            continue
        
        for num in range(num_points):
             # Indices for the current and previous layers
             curr_layer_start = num_points * index
             prev_layer_start = num_points * (index - 1)
             
             # Current and next indices in the circular arrangement
             curr_idx = curr_layer_start + num
             next_curr_idx = curr_layer_start + (num + 1) % num_points
             prev_idx = prev_layer_start + num
             next_prev_idx = prev_layer_start + (num + 1) % num_points
             
             # Add two triangles for each quadrilateral
             cells.append([prev_idx, curr_idx, next_prev_idx])
             cells.append([curr_idx, next_curr_idx, next_prev_idx])

    cells = np.array(cells)
    cells.shape = (-1,3)
    coord = np.stack([x_coord,y_coord,z_coord], axis = -1)

    trimesh = meshio.Mesh(coord, cells = {"triangle": cells})

    folder_path = options.output_folder+'/Surface_solution'
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    vol_mesh_filepath = f"{folder_path}/Billig_{i}_{index_object}.vtk"
    meshio.write(vol_mesh_filepath, trimesh)

    return coord, cells
