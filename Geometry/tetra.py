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
from scipy.spatial import ConvexHull
from concurrent.futures import ThreadPoolExecutor

#def inertia_tetra(v0,v1,v2,v3,vol, COG, rho):    
#
#    det = 6*vol
#    inertia = np.zeros((3,3))
#
#    x = np.stack((v0[:,0] - COG[0],v1[:,0] - COG[0],v2[:,0] - COG[0],v3[:,0] - COG[0]), axis=-1)
#    y = np.stack((v0[:,1] - COG[1],v1[:,1] - COG[1],v2[:,1] - COG[1],v3[:,1] - COG[1]), axis=-1)
#    z = np.stack((v0[:,2] - COG[2],v1[:,2] - COG[2],v2[:,2] - COG[2],v3[:,2] - COG[2]), axis=-1)
#
#    inertia[0,0] = np.sum(rho*det*(  y[:,0]*y[:,0] + y[:,0]*y[:,1] + y[:,1]*y[:,1] + y[:,0]*y[:,2] + y[:,1]*y[:,2] 
#                                   + y[:,2]*y[:,2] + y[:,0]*y[:,3] + y[:,1]*y[:,3] + y[:,2]*y[:,3] + y[:,3]*y[:,3]
#                                   + z[:,0]*z[:,0] + z[:,0]*z[:,1] + z[:,1]*z[:,1] + z[:,0]*z[:,2] + z[:,1]*z[:,2] 
#                                   + z[:,2]*z[:,2] + z[:,0]*z[:,3] + z[:,1]*z[:,3] + z[:,2]*z[:,3] + z[:,3]*z[:,3])/60.0)
#    
#    inertia[1,1] = np.sum(rho*det* ( x[:,0]*x[:,0] + x[:,0]*x[:,1] + x[:,1]*x[:,1] + x[:,0]*x[:,2] + x[:,1]*x[:,2] 
#                                   + x[:,2]*x[:,2] + x[:,0]*x[:,3] + x[:,1]*x[:,3] + x[:,2]*x[:,3] + x[:,3]*x[:,3]
#                                   + z[:,0]*z[:,0] + z[:,0]*z[:,1] + z[:,1]*z[:,1] + z[:,0]*z[:,2] + z[:,1]*z[:,2] 
#                                   + z[:,2]*z[:,2] + z[:,0]*z[:,3] + z[:,1]*z[:,3] + z[:,2]*z[:,3] + z[:,3]*z[:,3])/60.0)
#
#
#    inertia[2,2] = np.sum(rho*det* ( y[:,0]*y[:,0] + y[:,0]*y[:,1] + y[:,1]*y[:,1] + y[:,0]*y[:,2] + y[:,1]*y[:,2] 
#                                   + y[:,2]*y[:,2] + y[:,0]*y[:,3] + y[:,1]*y[:,3] + y[:,2]*y[:,3] + y[:,3]*y[:,3]
#                                   + x[:,0]*x[:,0] + x[:,0]*x[:,1] + x[:,1]*x[:,1] + x[:,0]*x[:,2] + x[:,1]*x[:,2] 
#                                   + x[:,2]*x[:,2] + x[:,0]*x[:,3] + x[:,1]*x[:,3] + x[:,2]*x[:,3] + x[:,3]*x[:,3])/60.0)
#
#
#    inertia[0,1] =np.sum(-rho*det*(2*x[:,0]*z[:,0] +   x[:,1]*z[:,0] +   x[:,2]*z[:,0] +   x[:,3]*z[:,0] 
#                                   + x[:,0]*z[:,1] + 2*x[:,1]*z[:,1] +   x[:,2]*z[:,1] +   x[:,3]*z[:,1]
#                                   + x[:,0]*z[:,2] +   x[:,1]*z[:,2] + 2*x[:,2]*z[:,2] +   x[:,3]*z[:,2]
#                                   + x[:,0]*z[:,3] +   x[:,1]*z[:,3] +   x[:,2]*z[:,3] + 2*x[:,3]*z[:,3])/120.0)
#
#        
#    inertia[1,0] = inertia[0,1]
#
#
#    inertia[0,2] = np.sum(-rho*det*(2*x[:,0]*y[:,0] +   x[:,1]*y[:,0] +   x[:,2]*y[:,0] +   x[:,3]*y[:,0] 
#                                   +  x[:,0]*y[:,1] + 2*x[:,1]*y[:,1] +   x[:,2]*y[:,1] +   x[:,3]*y[:,1]
#                                   +  x[:,0]*y[:,2] +   x[:,1]*y[:,2] + 2*x[:,2]*y[:,2] +   x[:,3]*y[:,2]
#                                   +  x[:,0]*y[:,3] +   x[:,1]*y[:,3] +   x[:,2]*y[:,3] + 2*x[:,3]*y[:,3])/120.0)
#
#    inertia[2,0] = inertia[0,2]
#
#
#    inertia[1,2] = np.sum(-rho*det*(2*y[:,0]*z[:,0] +   y[:,1]*z[:,0] +   y[:,2]*z[:,0] +   y[:,3]*z[:,0] 
#                                   +  y[:,0]*z[:,1] + 2*y[:,1]*z[:,1] +   y[:,2]*z[:,1] +   y[:,3]*z[:,1]
#                                   +  y[:,0]*z[:,2] +   y[:,1]*z[:,2] + 2*y[:,2]*z[:,2] +   y[:,3]*z[:,2]
#                                   +  y[:,0]*z[:,3] +   y[:,1]*z[:,3] +   y[:,2]*z[:,3] + 2*y[:,3]*z[:,3])/120.0)
#
#
#    inertia[2,1] = inertia[1,2]
#
#    return inertia

def inertia_tetra(v0,v1,v2,v3,vol, COG, rho):    

    det = 6*vol
    n = len(vol)
    inertia = np.zeros((n,3,3))
    final_inertia = np.zeros((3,3))

    x = np.stack((v0[:,0] - COG[0],v1[:,0] - COG[0],v2[:,0] - COG[0],v3[:,0] - COG[0]), axis=-1)
    y = np.stack((v0[:,1] - COG[1],v1[:,1] - COG[1],v2[:,1] - COG[1],v3[:,1] - COG[1]), axis=-1)
    z = np.stack((v0[:,2] - COG[2],v1[:,2] - COG[2],v2[:,2] - COG[2],v3[:,2] - COG[2]), axis=-1)

    inertia[:,0,0] = rho*det*(  y[:,0]*y[:,0] + y[:,0]*y[:,1] + y[:,1]*y[:,1] + y[:,0]*y[:,2] + y[:,1]*y[:,2] 
                                   + y[:,2]*y[:,2] + y[:,0]*y[:,3] + y[:,1]*y[:,3] + y[:,2]*y[:,3] + y[:,3]*y[:,3]
                                   + z[:,0]*z[:,0] + z[:,0]*z[:,1] + z[:,1]*z[:,1] + z[:,0]*z[:,2] + z[:,1]*z[:,2] 
                                   + z[:,2]*z[:,2] + z[:,0]*z[:,3] + z[:,1]*z[:,3] + z[:,2]*z[:,3] + z[:,3]*z[:,3])/60.0
    
    inertia[:,1,1] = rho*det* ( x[:,0]*x[:,0] + x[:,0]*x[:,1] + x[:,1]*x[:,1] + x[:,0]*x[:,2] + x[:,1]*x[:,2] 
                                   + x[:,2]*x[:,2] + x[:,0]*x[:,3] + x[:,1]*x[:,3] + x[:,2]*x[:,3] + x[:,3]*x[:,3]
                                   + z[:,0]*z[:,0] + z[:,0]*z[:,1] + z[:,1]*z[:,1] + z[:,0]*z[:,2] + z[:,1]*z[:,2] 
                                   + z[:,2]*z[:,2] + z[:,0]*z[:,3] + z[:,1]*z[:,3] + z[:,2]*z[:,3] + z[:,3]*z[:,3])/60.0


    inertia[:,2,2] = rho*det* ( y[:,0]*y[:,0] + y[:,0]*y[:,1] + y[:,1]*y[:,1] + y[:,0]*y[:,2] + y[:,1]*y[:,2] 
                                   + y[:,2]*y[:,2] + y[:,0]*y[:,3] + y[:,1]*y[:,3] + y[:,2]*y[:,3] + y[:,3]*y[:,3]
                                   + x[:,0]*x[:,0] + x[:,0]*x[:,1] + x[:,1]*x[:,1] + x[:,0]*x[:,2] + x[:,1]*x[:,2] 
                                   + x[:,2]*x[:,2] + x[:,0]*x[:,3] + x[:,1]*x[:,3] + x[:,2]*x[:,3] + x[:,3]*x[:,3])/60.0


    inertia[:,0,1] =-rho*det*(2*x[:,0]*z[:,0] +   x[:,1]*z[:,0] +   x[:,2]*z[:,0] +   x[:,3]*z[:,0] 
                                   + x[:,0]*z[:,1] + 2*x[:,1]*z[:,1] +   x[:,2]*z[:,1] +   x[:,3]*z[:,1]
                                   + x[:,0]*z[:,2] +   x[:,1]*z[:,2] + 2*x[:,2]*z[:,2] +   x[:,3]*z[:,2]
                                   + x[:,0]*z[:,3] +   x[:,1]*z[:,3] +   x[:,2]*z[:,3] + 2*x[:,3]*z[:,3])/120.0

        
    inertia[:,1,0] = inertia[:,0,1]


    inertia[:,0,2] = -rho*det*(2*x[:,0]*y[:,0] +   x[:,1]*y[:,0] +   x[:,2]*y[:,0] +   x[:,3]*y[:,0] 
                                   +  x[:,0]*y[:,1] + 2*x[:,1]*y[:,1] +   x[:,2]*y[:,1] +   x[:,3]*y[:,1]
                                   +  x[:,0]*y[:,2] +   x[:,1]*y[:,2] + 2*x[:,2]*y[:,2] +   x[:,3]*y[:,2]
                                   +  x[:,0]*y[:,3] +   x[:,1]*y[:,3] +   x[:,2]*y[:,3] + 2*x[:,3]*y[:,3])/120.0

    inertia[:,2,0] = inertia[:,0,2]


    inertia[:,1,2] = -rho*det*(2*y[:,0]*z[:,0] +   y[:,1]*z[:,0] +   y[:,2]*z[:,0] +   y[:,3]*z[:,0] 
                                   +  y[:,0]*z[:,1] + 2*y[:,1]*z[:,1] +   y[:,2]*z[:,1] +   y[:,3]*z[:,1]
                                   +  y[:,0]*z[:,2] +   y[:,1]*z[:,2] + 2*y[:,2]*z[:,2] +   y[:,3]*z[:,2]
                                   +  y[:,0]*z[:,3] +   y[:,1]*z[:,3] +   y[:,2]*z[:,3] + 2*y[:,3]*z[:,3])/120.0


    inertia[:,2,1] = inertia[:,1,2]

    final_inertia = np.sum(inertia, axis = 0)

    print('tetra 1st total inertia xx:', inertia[0,0,0])
    print('tetra final_inertia:', final_inertia);exit()


    return inertia

#def inertia_tetra_new(v0, v1, v2, v3, vol, local_com, rho):    
#    rho = 1000
#    det = 6 * vol
#    det = det[:, np.newaxis]
#    inertia = np.zeros((3, 3))
#
#    # Compute relative coordinates to local center of mass
#    x = np.stack((v0[:, 0] - local_com[:,0], v1[:, 0] - local_com[:,0], v2[:, 0] - local_com[:,0], v3[:, 0] - local_com[:,0]), axis=-1)
#    y = np.stack((v0[:, 1] - local_com[:,1], v1[:, 1] - local_com[:,1], v2[:, 1] - local_com[:,1], v3[:, 1] - local_com[:,1]), axis=-1)
#    z = np.stack((v0[:, 2] - local_com[:,2], v1[:, 2] - local_com[:,2], v2[:, 2] - local_com[:,2], v3[:, 2] - local_com[:,2]), axis=-1)
#
#    print('rho:', np.shape(rho))
#    print('det:', np.shape(det))
#    print('y:', np.shape(y))
#    print('z:', np.shape(z))
#
#    # Inertia tensor components
#    inertia[0, 0] = np.sum(rho * det * (y**2 + z**2) / 60.0)
#    inertia[1, 1] = np.sum(rho * det * (x**2 + z**2) / 60.0)
#    inertia[2, 2] = np.sum(rho * det * (x**2 + y**2) / 60.0)
#
#    inertia[0, 1] = np.sum(-rho * det * (x * y) / 120.0)
#    inertia[1, 0] = inertia[0, 1]
#    
#    inertia[0, 2] = np.sum(-rho * det * (x * z) / 120.0)
#    inertia[2, 0] = inertia[0, 2]
#    
#    inertia[1, 2] = np.sum(-rho * det * (y * z) / 120.0)
#    inertia[2, 1] = inertia[1, 2]
#
#    return inertia

def inertia_tetra_new(v0,v1,v2,v3,vol, local_com, rho, global_com):  

    rho = 1000  

    det = 6*vol
    det = det[:, np.newaxis]
    inertia = np.zeros((3,3))

    x = np.stack((v0[:,0] - local_com[:,0],v1[:,0] - local_com[:,0],v2[:,0] - local_com[:,0],v3[:,0] - local_com[:,0]), axis=-1)
    y = np.stack((v0[:,1] - local_com[:,1],v1[:,1] - local_com[:,1],v2[:,1] - local_com[:,1],v3[:,1] - local_com[:,1]), axis=-1)
    z = np.stack((v0[:,2] - local_com[:,2],v1[:,2] - local_com[:,2],v2[:,2] - local_com[:,2],v3[:,2] - local_com[:,2]), axis=-1)

    inertia_local = np.zeros((len(v0), 3, 3))  # Shape (n, 3, 3)

    # Calculate relative coordinates of each vertex relative to the local center of mass
    x = np.stack((v0[:, 0] - local_com[:, 0], v1[:, 0] - local_com[:, 0], v2[:, 0] - local_com[:, 0], v3[:, 0] - local_com[:, 0]), axis=-1)
    y = np.stack((v0[:, 1] - local_com[:, 1], v1[:, 1] - local_com[:, 1], v2[:, 1] - local_com[:, 1], v3[:, 1] - local_com[:, 1]), axis=-1)
    z = np.stack((v0[:, 2] - local_com[:, 2], v1[:, 2] - local_com[:, 2], v2[:, 2] - local_com[:, 2], v3[:, 2] - local_com[:, 2]), axis=-1)

    # Compute inertia tensor for each tetrahedron (summing over vertices, not tetrahedra)
    inertia_local[:, 0, 0] = rho * np.sum(det * (y**2 + z**2) / 60.0, axis=-1)
    inertia_local[:, 1, 1] = rho * np.sum(det * (x**2 + z**2) / 60.0, axis=-1)
    inertia_local[:, 2, 2] = rho * np.sum(det * (x**2 + y**2) / 60.0, axis=-1)

    inertia_local[:, 0, 1] = rho * np.sum(-det * (x * y) / 120.0, axis=-1)
    inertia_local[:, 1, 0] = inertia_local[:, 0, 1]

    inertia_local[:, 0, 2] = rho * np.sum(-det * (x * z) / 120.0, axis=-1)
    inertia_local[:, 2, 0] = inertia_local[:, 0, 2]

    inertia_local[:, 1, 2] = rho * np.sum(-det * (y * z) / 120.0, axis=-1)
    inertia_local[:, 2, 1] = inertia_local[:, 1, 2]

    total_local_inertia = np.sum(inertia_local, axis=0)


    print('total_local_inertia:', total_local_inertia)

    inertia_global = translate_inertia(inertia_local, rho * vol, local_com, global_com)

    total_inertia = np.sum(inertia_global, axis=0)

    return total_inertia

def translate_inertia(inertia_local, mass, local_com, global_com):
    # Vector from each local center of mass to the global center of mass
    d = local_com - global_com[np.newaxis, :]  # Shape (n, 3)
    d_squared = np.sum(d**2, axis=1)  # Shape (n,)

    # Apply parallel axis theorem for each tetrahedron
    translation_inertia = mass[:, np.newaxis, np.newaxis] * (
        d_squared[:, np.newaxis, np.newaxis] * np.eye(3) - np.einsum('ij,ik->ijk', d, d)
    )

    # Return the translated inertia tensor for each tetrahedron
    return inertia_local + translation_inertia

#def translate_inertia(inertia_local, mass, local_com, global_com):
#    d = local_com - global_com[np.newaxis, :]  # Vector from local to global center of mass
#    d_squared = np.dot(d, d)
#    translation_inertia = mass * (d_squared * np.eye(3) - np.outer(d, d))
#    return inertia_local + translation_inertia



# Function to calculate the volume of a convex hull for a set of points
def volume_from_convex_hull(points):
    hull = ConvexHull(points)
    return hull.volume

# Function to calculate volume for one polyhedron at a time
def compute_single_volume(i, v0, v1, v2, v3):
    points = np.vstack([v0[i], v1[i], v2[i], v3[i]])
    return volume_from_convex_hull(points)

# Function to calculate volumes for all polyhedra using parallel processing
def vol_tetra(v0, v1, v2, v3):
    n = len(v0)  # Number of polyhedra
    volumes = np.zeros(n)  # Initialize an array to store the volumes
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_single_volume, i, v0, v1, v2, v3) for i in range(n)]
        volumes = np.array([f.result() for f in futures])
    
    return volumes