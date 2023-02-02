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

def inertia_tetra(v0,v1,v2,v3,vol, COG, rho):    

    det = 6*vol
    inertia = np.zeros((3,3))

    x = np.stack((v0[:,0] - COG[0],v1[:,0] - COG[0],v2[:,0] - COG[0],v3[:,0] - COG[0]), axis=-1)
    y = np.stack((v0[:,1] - COG[1],v1[:,1] - COG[1],v2[:,1] - COG[1],v3[:,1] - COG[1]), axis=-1)
    z = np.stack((v0[:,2] - COG[2],v1[:,2] - COG[2],v2[:,2] - COG[2],v3[:,2] - COG[2]), axis=-1)

    inertia[0,0] = np.sum(rho*det*(  y[:,0]*y[:,0] + y[:,0]*y[:,1] + y[:,1]*y[:,1] + y[:,0]*y[:,2] + y[:,1]*y[:,2] 
                                   + y[:,2]*y[:,2] + y[:,0]*y[:,3] + y[:,1]*y[:,3] + y[:,2]*y[:,3] + y[:,3]*y[:,3]
                                   + z[:,0]*z[:,0] + z[:,0]*z[:,1] + z[:,1]*z[:,1] + z[:,0]*z[:,2] + z[:,1]*z[:,2] 
                                   + z[:,2]*z[:,2] + z[:,0]*z[:,3] + z[:,1]*z[:,3] + z[:,2]*z[:,3] + z[:,3]*z[:,3])/60.0)
    
    inertia[1,1] = np.sum(rho*det* ( x[:,0]*x[:,0] + x[:,0]*x[:,1] + x[:,1]*x[:,1] + x[:,0]*x[:,2] + x[:,1]*x[:,2] 
                                   + x[:,2]*x[:,2] + x[:,0]*x[:,3] + x[:,1]*x[:,3] + x[:,2]*x[:,3] + x[:,3]*x[:,3]
                                   + z[:,0]*z[:,0] + z[:,0]*z[:,1] + z[:,1]*z[:,1] + z[:,0]*z[:,2] + z[:,1]*z[:,2] 
                                   + z[:,2]*z[:,2] + z[:,0]*z[:,3] + z[:,1]*z[:,3] + z[:,2]*z[:,3] + z[:,3]*z[:,3])/60.0)


    inertia[2,2] = np.sum(rho*det* ( y[:,0]*y[:,0] + y[:,0]*y[:,1] + y[:,1]*y[:,1] + y[:,0]*y[:,2] + y[:,1]*y[:,2] 
                                   + y[:,2]*y[:,2] + y[:,0]*y[:,3] + y[:,1]*y[:,3] + y[:,2]*y[:,3] + y[:,3]*y[:,3]
                                   + x[:,0]*x[:,0] + x[:,0]*x[:,1] + x[:,1]*x[:,1] + x[:,0]*x[:,2] + x[:,1]*x[:,2] 
                                   + x[:,2]*x[:,2] + x[:,0]*x[:,3] + x[:,1]*x[:,3] + x[:,2]*x[:,3] + x[:,3]*x[:,3])/60.0)


    inertia[0,1] =np.sum(-rho*det*(2*x[:,0]*z[:,0] +   x[:,1]*z[:,0] +   x[:,2]*z[:,0] +   x[:,3]*z[:,0] 
                                   + x[:,0]*z[:,1] + 2*x[:,1]*z[:,1] +   x[:,2]*z[:,1] +   x[:,3]*z[:,1]
                                   + x[:,0]*z[:,2] +   x[:,1]*z[:,2] + 2*x[:,2]*z[:,2] +   x[:,3]*z[:,2]
                                   + x[:,0]*z[:,3] +   x[:,1]*z[:,3] +   x[:,2]*z[:,3] + 2*x[:,3]*z[:,3])/120.0)

        
    inertia[1,0] = inertia[0,1]


    inertia[0,2] = np.sum(-rho*det*(2*x[:,0]*y[:,0] +   x[:,1]*y[:,0] +   x[:,2]*y[:,0] +   x[:,3]*y[:,0] 
                                   +  x[:,0]*y[:,1] + 2*x[:,1]*y[:,1] +   x[:,2]*y[:,1] +   x[:,3]*y[:,1]
                                   +  x[:,0]*y[:,2] +   x[:,1]*y[:,2] + 2*x[:,2]*y[:,2] +   x[:,3]*y[:,2]
                                   +  x[:,0]*y[:,3] +   x[:,1]*y[:,3] +   x[:,2]*y[:,3] + 2*x[:,3]*y[:,3])/120.0)

    inertia[2,0] = inertia[0,2]


    inertia[1,2] = np.sum(-rho*det*(2*y[:,0]*z[:,0] +   y[:,1]*z[:,0] +   y[:,2]*z[:,0] +   y[:,3]*z[:,0] 
                                   +  y[:,0]*z[:,1] + 2*y[:,1]*z[:,1] +   y[:,2]*z[:,1] +   y[:,3]*z[:,1]
                                   +  y[:,0]*z[:,2] +   y[:,1]*z[:,2] + 2*y[:,2]*z[:,2] +   y[:,3]*z[:,2]
                                   +  y[:,0]*z[:,3] +   y[:,1]*z[:,3] +   y[:,2]*z[:,3] + 2*y[:,3]*z[:,3])/120.0)


    inertia[2,1] = inertia[1,2]

    return inertia

def vol_tetra(v0,v1,v2,v3):

    a1 = np.linalg.norm(v1-v0, axis = 1)**2
    a2 = np.linalg.norm(v2-v0, axis = 1)**2
    a3 = np.linalg.norm(v3-v0, axis = 1)**2
    a4 = np.linalg.norm(v2-v1, axis = 1)**2
    a5 = np.linalg.norm(v3-v2, axis = 1)**2
    a6 = np.linalg.norm(v1-v3, axis = 1)**2

    Vol = 1.0/144.0 * (a1 * a5 * (a2 + a3 + a4 + a6 - a1 - a5)
                    +  a2 * a6 * (a1 + a3 + a4 + a5 - a2 - a6)
                    +  a3 * a4 * (a1 + a2 + a5 + a6 - a3 - a4)
                    -  a1 * a2 * a4  - a2 * a3 * a5 - a1 * a3 * a6 - a4 * a5 * a6)

    return np.sqrt(Vol)