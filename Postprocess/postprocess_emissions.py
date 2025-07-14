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
from Dynamics import frames
import sympy
from sympy import sqrt, tan
import meshio
from pathlib import Path


def postprocess_emissions(options):

    print('Computing emissions ...')

    path = options.output_folder + '/Data/*'
    search_string = 'emissions'

    # Get a list of all files in the folder
    files = glob.glob(path)

    # Iterate and delete each file that contains the search_string
    #for file in files:
    #    if os.path.isfile(file) and search_string in os.path.basename(file):
    #        os.remove(file)

    data = pd.read_csv(options.output_folder + '/Data/data.csv', index_col=False)

    iter_interval = np.unique(data['Iter'].to_numpy())

    if options.radiation.spectral_freq % options.save_freq != 0:
        print('No available solutions for the chosen frequency.')
        exit()

    for iter_value in range(options.radiation.spectral_freq, max(iter_interval) + 1, options.radiation.spectral_freq):
        iter_value = int(iter_value)
        titan = read_state(options, iter_value)
        #view_direction(titan, options)
        #line_of_sight(titan, options, iter_value)
        #element_gas_densities(titan)
        emissions(titan, options, iter_value)
        #output.generate_surface_solution_emissions(titan=titan, options=options, folder='Postprocess_emissions', iter_value = iter_value)

def view_direction(titan, options):

    print('\nDefining view direction ...')

    phi = options.radiation.phi
    theta = options.radiation.theta

    # define viewpoint on the basis of angles
    viewpoint = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    print('View direction in the ECEF frame:', viewpoint)

    for assembly in titan.assembly:

        R_B_ECEF = Rot.from_quat(assembly.quaternion_prev)

        # Transform assembly from body to ECEF frame
        assembly.mesh.facet_normal = R_B_ECEF.apply(assembly.mesh.facet_normal)
        assembly.mesh.nodes        = R_B_ECEF.apply(assembly.mesh.nodes)
        assembly.mesh.v0           = R_B_ECEF.apply(assembly.mesh.v0)
        assembly.mesh.v1           = R_B_ECEF.apply(assembly.mesh.v1)
        assembly.mesh.v2           = R_B_ECEF.apply(assembly.mesh.v2)

        # Retrieve index of facets seen from viewpoint
        assembly.index_blackbody = Aerothermo.ray_trace(assembly, viewpoint)
        assembly.index_atomic = np.intersect1d(assembly.index_blackbody, assembly.aero_index)

        # Calculate angle of facets seen from viewpoint, relative to view direction
        assembly.angle_blackbody[assembly.index_blackbody] = vg.angle(-viewpoint, assembly.mesh.facet_normal[assembly.index_blackbody]) # degrees
        assembly.angle_atomic[assembly.index_atomic]    = vg.angle(-viewpoint, assembly.mesh.facet_normal[assembly.index_atomic]) # degrees

        points = assembly.mesh.nodes - assembly.mesh.surface_displacement
        facets = assembly.mesh.facets
        heatflux = assembly.aerothermo.heatflux

#        assembly.index_atomic_debug = np.zeros(len(facets))
#        assembly.index_bb_debug = np.zeros(len(facets))
#
#        assembly.index_atomic_debug[assembly.index_atomic] = 1
#        assembly.index_bb_debug[assembly.index_blackbody] = 1
#
#        cellID = np.array([])
#        for cellid in range(len(assembly.mesh.facets)):
#            cellID = np.append(cellID, cellid)
#
#        
#        cells = {"triangle": facets}
#
#        cell_data = { "Heatflux":                    [heatflux],
#        "index_at":                    [assembly.index_atomic_debug],
#        "index_bb":                    [assembly.index_bb_debug],
#
#                    }
#
#        trimesh = meshio.Mesh(points,
#                              cells=cells,
#                              cell_data = cell_data)
#
#        folder_path = options.output_folder+'/' + 'Postprocess_emissions' + '/ID_'+str(assembly.id)
#        Path(folder_path).mkdir(parents=True, exist_ok=True)
#
#        vol_mesh_filepath = f"{folder_path}/solution_iter_debug.xdmf"
#        meshio.write(vol_mesh_filepath, trimesh, file_format="xdmf")

def element_gas_densities(titan):

    for assembly in titan.assembly:

        assembly.aerothermo.rhoe_i = np.zeros((len(assembly.mesh.facets), len(assembly.aerothermo.ce_i[0])))

        if assembly.freestream.mach > 1.1:

            for obj in assembly.objects:
                assembly.LOS[obj.facet_index] = obj.LOS

            gas_volume = assembly.mesh.facet_area * assembly.LOS
            gas_mass = assembly.aerothermo.rhoe * gas_volume
            # before adding melted material
            species_mass = assembly.aerothermo.ce_i * gas_mass[:, np.newaxis]
            # add melted material mass
            #species_mass[:, -1] = assembly.mMelt
            gas_mass = species_mass.sum(axis=1)
            assembly.aerothermo.ce_i = species_mass / gas_mass[:, np.newaxis]
            assembly.aerothermo.ce_i[np.isnan(assembly.aerothermo.ce_i)] = 0
            assembly.aerothermo.ce_i[np.isinf(assembly.aerothermo.ce_i)] = 0
            # updated gas density after ablation
            # Avoid division by zero for gas_volume
            assembly.updated_gas_density = np.where(gas_volume != 0, gas_mass / gas_volume, 0)
            assembly.aerothermo.rhoe_i = np.where( assembly.updated_gas_density[:, np.newaxis] != 0, assembly.aerothermo.ce_i * assembly.updated_gas_density[:, np.newaxis], 0 )

def read_state(options, i=0):
    """
    Load state of the TITAN object for the given iteration

    Returns
    -------
    titan: Assembly_list
        Object of class Assembly_list
    """
    print("\n-------------------------------------------------")
    print("\n\nPost-processing iteration:", i)

    infile = open(options.output_folder + '/Restart/' + 'Assembly_State_' + str(i) + '_.p', 'rb')
    titan = pickle.load(infile)
    infile.close()

    return titan

def emissions(titan, options, iter_value):

    wavelengths_OI    = [777.194e-9, 777.417e-9, 777.539e-9]
    wavelengths_AlI_1 = [394.40058e-9]
    wavelengths_AlI_2 = [396.152e-9]

    for assembly in titan.assembly:

        print('\nAssembly:', assembly.id)

        if options.radiation.spectral:
            print('\nComputing spectral blackbody emissions for O I wavelengths ...')
            assembly.blackbody_emissions_OI, assembly.blackbody_emissions_OI_surf = thermal.compute_black_body_spectral_emissions(
                assembly, wavelengths_OI
            )

            print('assembly.blackbody_emissions_OI:', assembly.blackbody_emissions_OI)

            print('\nComputing spectral blackbody emissions for Al I wavelength 1 ...')
            assembly.blackbody_emissions_AlI_1, assembly.blackbody_emissions_AlI_1_surf = thermal.compute_black_body_spectral_emissions(
                assembly, wavelengths_AlI_1
            )

            print('assembly.blackbody_emissions_AlI_1:', assembly.blackbody_emissions_AlI_1)

            print('\nComputing spectral blackbody emissions for Al I wavelength 2 ...')
            assembly.blackbody_emissions_AlI_2, assembly.blackbody_emissions_AlI_2_surf = thermal.compute_black_body_spectral_emissions(
                assembly, wavelengths_AlI_2
            )   

            print('assembly.blackbody_emissions_AlI_2:', assembly.blackbody_emissions_AlI_2)

#        if options.radiation.spectral and options.thermal.ablation and options.radiation.particle_emissions and options.pato.flag:
#            assembly.OI_atomic_emissions, assembly.atomic_emissions_OI_surf = thermal.compute_particle_spectral_emissions_OI(
#                assembly, wavelengths_OI
#            )
#
#            print('assembly.OI_atomic_emissions:', assembly.OI_atomic_emissions)
#
#            assembly.AlI_1_atomic_emissions, assembly.atomic_emissions_AlI_1_surf = thermal.compute_particle_spectral_emissions_AlI(
#                assembly, wavelengths_AlI, 0
#            )
#
#            print('assembly.AlI_1_atomic_emissions:', assembly.AlI_1_atomic_emissions)
#
#            assembly.AlI_2_atomic_emissions, assembly.atomic_emissions_AlI_2_surf = thermal.compute_particle_spectral_emissions_AlI(
#                assembly, wavelengths_AlI, 1
#            )
#
#            print('assembly.AlI_2_atomic_emissions:', assembly.AlI_2_atomic_emissions)


        #output.generate_surface_solution_emissions(titan=titan, options=options, folder='Postprocess_emissions', iter_value = titan.iter)

#        d = {
#            'Assembly_ID': [assembly.id],
#            'OI_emissions_blackbody':  [np.sum(assembly.blackbody_emissions_OI)],
#            'AlI_1_emissions_blackbody': [np.sum(assembly.blackbody_emissions_AlI_1)],
#            'AlI_2_emissions_blackbody': [np.sum(assembly.blackbody_emissions_AlI_2)],
#            'OI_emissions_atomic':     [np.sum(assembly.OI_atomic_emissions)],
#            'AlI_1_emissions_atomic':    [np.sum(assembly.AlI_1_atomic_emissions)],
#            'AlI_2_emissions_atomic':    [np.sum(assembly.AlI_2_atomic_emissions)],
#        }

        d = {
            'Assembly_ID': [assembly.id],
            'OI_emissions_blackbody':    [np.sum(assembly.blackbody_emissions_OI)],
            'AlI_1_emissions_blackbody': [np.sum(assembly.blackbody_emissions_AlI_1)],
            'AlI_2_emissions_blackbody': [np.sum(assembly.blackbody_emissions_AlI_2)],
            'OI_emissions_atomic':       [0],
            'AlI_1_emissions_atomic':    [0],
            'AlI_2_emissions_atomic':    [0],
        }

        df = pd.DataFrame(data=d)

        df.to_csv(
            options.output_folder + '/Data/' + 'emissions_' + str(iter_value) + '.csv',
            mode='a',
            header=not os.path.exists(
                options.output_folder + '/Data/' + 'emissions_' + str(iter_value) + '.csv'
            ),
            index=False,
        )

def line_of_sight(titan, options, iteration):

    print('Calculating line-of-sight ...')

    for assembly in titan.assembly:
        for obj in assembly.objects:
            obj.LOS = np.zeros(len(obj.mesh.facets))


    titan_windframe = deepcopy(titan)

    for index, assembly in enumerate(titan_windframe.assembly):

        M = assembly.freestream.mach

        if M > 1.1:

            R_B_ECEF = Rot.from_quat(assembly.quaternion_prev)

            lat = assembly.trajectory.latitude
            lon = assembly.trajectory.longitude
            chi = assembly.trajectory.chi
            gamma = assembly.trajectory.gamma

            R_ECEF_NED = frames.R_NED_ECEF(lat=lat, lon=lon).inv()
            R_NED_W = frames.R_W_NED(ha=chi, fpa=gamma).inv()
            R_ECEF_W = R_NED_W * R_ECEF_NED

            
            theta = 0.0001

            #p = np.intersect1d(assembly.index_atomic, np.where(assembly.aerothermo.theta > 0.001)[0])
            p = np.where(assembly.aerothermo.theta > 0.001)

            for index_object, obj in enumerate(assembly.objects):

                print("\nObject:", obj.name)

                obj.mesh.nodes = R_B_ECEF.apply(obj.mesh.nodes)
                obj.mesh.nodes = (R_ECEF_W).apply(obj.mesh.nodes)

                obj.mesh.facet_normal = R_B_ECEF.apply(obj.mesh.facet_normal)
                obj.mesh.facet_normal = (R_ECEF_W).apply(obj.mesh.facet_normal)

                obj.mesh.facet_COG = R_B_ECEF.apply(obj.mesh.facet_COG)
                obj.mesh.facet_COG = (R_ECEF_W).apply(obj.mesh.facet_COG)

                min_coords = np.min(obj.mesh.nodes, axis=0)
                max_coords = np.max(obj.mesh.nodes, axis=0)

                # Creation of the virtual Sphere
                center = np.zeros((3))
                center[1:] = (min_coords[1:] + max_coords[1:]) / 2.0
                center[0] = max_coords[0]

                dist_center = np.linalg.norm(obj.mesh.nodes[:, 1:] - center[1:], axis=1)
                radius = np.max(dist_center)

                xmax = np.max(obj.mesh.nodes, axis=0)
                xmin = np.min(obj.mesh.nodes, axis=0)

                Lref = np.max(xmax - xmin)

                # Compute billig formula and retrieve the bodies that are inside the computed shock envelopes
                Switch.sphere_surface(radius, center, index, index_object, titan.iter, assembly, options)
                billig_points, billig_facets = compute_billig(
                    M, theta, center, radius, index, Lref, obj.global_ID, titan.iter, assembly.freestream, options
                )

                index_aero_obj = np.intersect1d(p, obj.facet_index)

                index_aero_obj = np.where(np.isin(obj.facet_index, index_aero_obj))[0]

                obj.LOS[index_aero_obj] = compute_shock_distance(obj, index_aero_obj, billig_points, billig_facets)

                output.generate_surface_solution_object(
                    obj, obj.LOS, options, iter_value=iteration, folder='Postprocess_emissions'
                )

    for assembly, assembly_wf in zip(titan.assembly, titan_windframe.assembly):
        for obj, obj_wf in zip(assembly.objects, assembly_wf.objects):
            obj.LOS = obj_wf.LOS

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

    print('M:',M)

    x_coord = np.array([])
    y_coord = np.array([])
    z_coord = np.array([])
    cells = []

    r = sympy.Symbol("r")
    x_limit = 10*Lref

    #Blast Wave implementation here
    #x_limit = Switch.compute_blast_wave_limit(sphere_radius, freestream, options) - delta - sphere_radius

    exp = 1*(sphere_radius+delta-Rc*(1/tan(theta))**2*(sqrt(1+(r**2)*(tan(theta)**2)/(Rc**2))-1))+x_limit
    print('delta:',delta)
    print('Rc:',Rc)
    print('theta:',theta)
    print('r:',r)
    print('x_limit:',x_limit)
    print('sphere_radius:', sphere_radius)
    print('exp:',exp)
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

    print('Generating Billig .vtk solution, iteration:', i, ' object global_ID:', index_object)

    vol_mesh_filepath = f"{folder_path}/Billig_{i}_{index_object}.vtk"
    meshio.write(vol_mesh_filepath, trimesh)

    return coord, cells
