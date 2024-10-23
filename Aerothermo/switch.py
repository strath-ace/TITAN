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
from Aerothermo import aerothermo, su2
import meshio
from copy import deepcopy
from scipy.spatial.transform import Rotation as Rot
import numpy as np
from Dynamics import frames
from sympy import sqrt,tan
import sympy
from pathlib import Path
from Geometry import assembly as Assembly
import os

def sphere_surface(radius, center, num_assembly, num_object, i, assembly, options):
    """    
    Creation of a virtual Sphere

    The function created a virtual sphere that will be used  in assessing the shock envelope through the billing formula

    Parameters
    ----------
    radius: int
        Radius of the sphere
    center: np.array()
        Sphere center coordinates
    """

    num_points = 20
    theta_list = np.linspace(0,np.pi,num_points)
    phi_list = np.linspace(0,2*np.pi,num_points)

    x_coord = np.array([])
    y_coord = np.array([])
    z_coord = np.array([])
    cells = []

    for index,theta in enumerate(theta_list):

        x = radius*np.sin(theta)*np.cos(phi_list)
        y = radius*np.sin(theta)*np.sin(phi_list)
        z = np.repeat(radius*np.cos(theta),num_points)

        x_coord = np.append(x_coord,x+center[0])
        y_coord = np.append(y_coord,y+center[1]) 
        z_coord = np.append(z_coord,z+center[2]) 

        if index == 0: continue

        for num in range(num_points):
            if num == 0: cells.append([num_points*(index-1) + num_points-1, num_points*index + num_points-1,  num_points*index + num, num_points*(index-1) + num])
            else:
                cells.append([num_points*(index-1) + num, num_points*index + num, num_points*index + num -1, num_points*(index-1) + num -1])

    cells = np.array(cells)
    cells.shape = (-1,4)
    coord = np.stack([x_coord,y_coord,z_coord], axis = -1)

    trimesh = meshio.Mesh(coord, cells = {"quad": cells})

    folder_path = options.output_folder+'/Surface_solution/ID_'+str(assembly.id)
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    vol_mesh_filepath = f"{folder_path}/Sphere_{i}_{num_assembly}_{num_object}.vtk"
    meshio.write(vol_mesh_filepath, trimesh)


def compute_aerothermo(titan, options):
    """
    Aerothermo computation using a multi-fidelity approach (i.e. can use both low- and high-fidelity methodology)

    The function uses the Billig formula to assess the shock envelope criteria, used to determine wether to use low- or high-fidelity methods

    Parameters
    ----------
    titan: List_Assembly
        Object of class List_Assembly
    options: Options
        Object of class Options
    """

    #The following block is the same as in su2.py
    #The purpose is to pass the mesh from body into wind frame
    altitude = 1E10

    for index,assembly in enumerate(titan.assembly):
        if assembly.trajectory.altitude < altitude:
            altitude = assembly.trajectory.altitude
            it = index
            lref = assembly.Lref

    free = titan.assembly[it].freestream

    #Convert from Body->ECEF and ECEF-> Wind
    #Translate the mesh to match the Center of Mass of the lowest assembly

    assembly_windframe = Assembly.copy_assembly(titan.assembly, options)
    #assembly_windframe = deepcopy(titan.assembly)
    
    pos = titan.assembly[it].position
    
    for i, assembly in enumerate(assembly_windframe):

        R_B_ECEF = Rot.from_quat(assembly.quaternion)

        assembly.mesh.nodes -= assembly.COG
        assembly.mesh.nodes = R_B_ECEF.apply(assembly.mesh.nodes)

        #Translate to the ECEF position
        assembly.mesh.nodes += np.array(assembly.position-pos)        

        R_ECEF_NED = frames.R_NED_ECEF(lat = assembly.trajectory.latitude, lon = assembly.trajectory.longitude).inv()
        R_NED_W = frames.R_W_NED(ha = assembly.trajectory.chi, fpa = assembly.trajectory.gamma).inv()


        #R_ECEF_B = Rot.from_quat(assembly.quaternion).inv()
        #R_B_NED =   frames.R_B_NED(roll = assembly.roll, pitch = assembly.pitch, yaw = assembly.yaw) 
        #R_NED_W = frames.R_W_NED(ha = assembly.trajectory.chi, fpa = assembly.trajectory.gamma).inv()
        R_ECEF_W = R_NED_W*R_ECEF_NED

#       mesh[i].points = (R_ECEF_W).apply(mesh[i].points)
#       R_ECEF_W = R_NED_W*R_ECEF_NED 

        assembly.mesh.nodes = (R_ECEF_W).apply(assembly.mesh.nodes)

        assembly.mesh.xmin = np.min(assembly.mesh.nodes , axis = 0)
        assembly.mesh.xmax = np.max(assembly.mesh.nodes , axis = 0)

    computational_domain_tag = 0

    #Initialization of the computational tags
    for assembly in assembly_windframe:
        assembly.computational_domain_tag = 0
        assembly.inside_shock = np.zeros((len(assembly.mesh.nodes)))

    computational_domains = []

    #Loop the assembly list
    for index,assembly in enumerate(assembly_windframe):
        M = assembly.freestream.mach
        theta = 0.0001

        #list of bodies afected by the shock
        assembly_shock_list = []

        #if assembly.computational_domain_tag != 0: continue
        for index_object,obj in enumerate(assembly.objects):
            min_coords = np.min(assembly.mesh.nodes[obj.node_index], axis = 0)
            max_coords = np.max(assembly.mesh.nodes[obj.node_index], axis = 0)         

            #Creation of the virtual Sphere
            center = np.zeros((3))
            center[1:] = (min_coords[1:]+max_coords[1:])/2.0
            center[0] = max_coords[0]

            dist_center = np.linalg.norm(assembly.mesh.nodes[obj.node_index][:,1:] - center[1:], axis = 1)
            radius = np.max(dist_center)

            list_assembly = list(range(len(assembly_windframe)))
            list_assembly.remove(index)

            #Compute billig formula and retrieve the bodies that are inside the computed shock envelopes
            sphere_surface(radius, center, index, index_object, titan.iter, assembly, options)
            computational_domain_bodies = compute_billig(M, theta, center, radius, index, np.array(assembly_windframe), list_assembly, index_object, titan.iter, titan.assembly, assembly.freestream, options)
            assembly_shock_list += computational_domain_bodies

        computational_domain_tag += 1

        #Get an unique list of assemblies
        assembly_shock_list = list(set(assembly_shock_list))
        computational_domains.append(assembly_shock_list)
    
    #Run 3 times just to fill the list properly
    #TODO find a better algorithm
    for _ in range(3):
        for list_body in computational_domains:
            for it in range(len(computational_domains)):
                if [i for i in list_body if i in computational_domains[it]]:
                    list_body += [i for i in computational_domains[it] if i not in list_body]

    for list_body in computational_domains:
        list_body = list_body.sort()

    new_computational_domains = []

    #Clean the list
    for elem in computational_domains:
        if elem not in new_computational_domains:
            new_computational_domains.append(elem)
    
    for it,elem in enumerate(new_computational_domains):
        if len(elem) == 1:
            titan.assembly[elem[0]].computational_domains = 0
        else:
            for assembly_index in elem:
                titan.assembly[assembly_index].computational_domains = it+1
   
    tag_list = []

    for assembly in titan.assembly:
        tag_list.append(assembly.computational_domains)

    tag_set = list(set(tag_list))
    indexes = [it for it in range(len(tag_list))]

    titan.assembly = np.array(titan.assembly)

    #Depending on the tag, compute the aerothermodynamics for both low- and high-fidelity cases
    for tag_num,tag in enumerate(tag_set):
        index = [it for it in range(len(tag_list)) if  tag_list[it] == tag]

        if len(index) >= 1:
            if tag == 0:
                aerothermo.compute_low_fidelity_aerothermo(titan.assembly[index], options)

            else:
                su2.compute_cfd_aerothermo(titan.assembly[index],titan, options, tag)

    titan.assembly= list(titan.assembly)



def compute_billig(M,theta, center, sphere_radius, index_assembly, assembly, list_assembly, index_object, i, true_assembly, freestream, options):
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

    delta = sphere_radius*(0.143*np.exp(3.24/(M**2)))
    Rc = sphere_radius*(1.143*np.exp(0.54/(M-1)**1.2))

    x_coord = np.array([])
    y_coord = np.array([])
    z_coord = np.array([])
    cells = []

    r = sympy.Symbol("r")
    #x_limit = 20*assembly[index_assembly].Lref

    #Blast Wave implementation here
    x_limit = compute_blast_wave_limit(sphere_radius, freestream, options) - delta - sphere_radius

    # Sympy gives the following symbolic solution for Billig's equation, can use directly
    r = -Rc * sqrt(-1 + (sphere_radius * tan(theta) ** 2 + Rc + delta * tan(theta) ** 2 + x_limit * tan(theta) ** 2) ** 2 / Rc ** 2) / tan(theta)
    r = float(abs(r))

    #exp = 1*(sphere_radius+delta-Rc*(1/tan(theta))**2*(sqrt(1+(r**2)*(tan(theta)**2)/(Rc**2))-1))+x_limit
    #sol = sympy.solve(exp)

    #r = float(abs(sol[0]))
    r = np.linspace(0,r,50)

    num_points = 36
    angle = np.linspace(0,2*np.pi,num_points+1)[0:-1]


    for index,_r in enumerate(r):
        x = 1*(sphere_radius+delta-Rc*(1/np.tan(theta))**2*(np.sqrt(1+(_r**2)*(np.tan(theta)**2)/(Rc**2))-1))
        y = _r * np.sin(angle)
        z = _r * np.cos(angle)

        x_coord = np.append(x_coord,np.repeat(x,num_points)+center[0])
        y_coord = np.append(y_coord,y+center[1]) 
        z_coord = np.append(z_coord,z+center[2]) 

        if index == 0: continue

        for num in range(num_points):
            if num == 0: cells.append([num_points*(index-1) + num_points-1, num_points*index + num_points-1,  num_points*index + num, num_points*(index-1) + num])
            else:
                cells.append([num_points*(index-1) + num, num_points*index + num, num_points*index + num -1, num_points*(index-1) + num -1])

    cells = np.array(cells)
    cells.shape = (-1,4)
    coord = np.stack([x_coord,y_coord,z_coord], axis = -1)

    trimesh = meshio.Mesh(coord, cells = {"quad": cells})

    folder_path = options.output_folder+'/Surface_solution/ID_'+str(assembly[index_assembly].id)
    Path(folder_path).mkdir(parents=True, exist_ok=True)

    vol_mesh_filepath = f"{folder_path}/Billig_{i}_{index_assembly}_{index_object}.vtk"
    meshio.write(vol_mesh_filepath, trimesh)

    computational_domain_bodies = [index_assembly]

    for index in list_assembly:

        _assembly = assembly[index]

        x_assembly = _assembly.mesh.nodes[:,0]-center[0]
        y_assembly = _assembly.mesh.nodes[:,1]-center[1]
        z_assembly = _assembly.mesh.nodes[:,2]-center[2]

        r_assembly = np.sqrt(y_assembly**2 + z_assembly**2)

        inside_ellipse = ((-x_assembly - Rc/np.tan(theta)**2*(np.sqrt(1+(r_assembly**2)*(np.tan(theta)**2)/(Rc**2))-1) >= -(sphere_radius+delta))) * (-x_assembly <= x_limit)# * (x_assembly >= -sphere_radius)
        #(sphere_radius+delta-Rc*(1/np.tan(theta))**2*(np.sqrt(1+(_r**2)*(np.tan(theta)**2)/(Rc**2))-1))

        true_assembly[index].inside_shock += np.zeros(len(x_assembly))+inside_ellipse

        if inside_ellipse.any():
            computational_domain_bodies.append(index)

    if options.write_solutions==False and i>=2:
        if os.path.exists(f"{folder_path}/Billig_{i-2}_{index_assembly}_{index_object}.vtk"):
            billig_path = Path(f"{folder_path}/Billig_{i-2}_{index_assembly}_{index_object}.vtk")
            sphere_path = Path(f"{folder_path}/Sphere_{i-2}_{index_assembly}_{index_object}.vtk")
            billig_path.unlink()
            sphere_path.unlink()
        return computational_domain_bodies



def compute_blast_wave_limit(radius, freestream, options):

    gamma = freestream.gamma
    velocity = freestream.velocity
    M = freestream.mach
    density = freestream.density
    eta_0 = 1.004 #From Laurence thesis for gamma = 1.4
    ratio = 1.0   # r/Rs

    theta = 0.0001
    delta = radius*(0.143*np.exp(3.24/(M**2)))
    Rc = radius*(1.143*np.exp(0.54/(M-1)**1.2))

    """
    k1 = -(gamma-1.0)/(gamma)
    k2 = -2.0/(2.0 - gamma)
    k3 = 1.0/gamma
    k4 = -k2

    def find_u_nd(gamma, value):
        min_value = (gamma+1.0)/(2*gamma)+ 1E-16
        max_value = 1.0+1E-16
        max_iters = 100

        mid_value = (min_value+max_value)/2.0

        for _ in range(max_iters):
            u_nd = mid_value
            result = u_nd*((2*gamma*u_nd - (gamma+1))/(gamma-1))**k1*(gamma+1.0-gamma*u_nd)

            if result > value:
                min_value = mid_value
                mid_value = (min_value+max_value)/2.0
            else:
                max_value = mid_value
                mid_value = (min_value+max_value)/2.0
        return u_nd

    u_nd = find_u_nd(gamma,ratio**-2)
    rho_nd = ((gamma+1.0-2.0*u_nd)/(gamma-1.0))**k2*((2*gamma*u_nd - (gamma+1))/(gamma-1))**k3*(gamma+1.0-gamma*u_nd)**k4
    p_nd = ((u_nd**2*(gamma + 1.0 - 2*u_nd))/(2*gamma*u_nd-(gamma+1)))*rho_nd
    """

    Ps = freestream.pressure #Pressure at the shock is equal to freestream pressure (shock lost strenght)

    x = density*velocity**2/Ps*(eta_0 * ((2*radius)**2*np.pi*1*0.88/8)**0.25)**2/(2*(gamma+1.0))

    #Shock radius as function of x distance

    Rs = eta_0*((2*radius)**2*np.pi*0.88/8)**0.25*np.sqrt(x)

#    print(Rs, x, density*velocity**2/x**2*Rs**2/(2*(gamma+1.0)))
#    print(density*velocity**2/x**2*Rs**2/(2*(gamma+1.0)))
#    print(freestream.pressure)

    """
    #Use of Billig Rs predicts 2 times the x value than the Blast wave method

    x_min = radius
    x_guess = radius*10
    x_max = radius*300

    r = sympy.Symbol("r")

    while abs(x_max - x_guess) > 0.01:

        exp = 1*(-Rc*(1/tan(theta))**2*(sqrt(1+(r**2)*(tan(theta)**2)/(Rc**2))-1))+x_guess
        sol = sympy.solve(exp)
        R_guess = float(abs(sol[0]))
        Ps = density*velocity**2/x_guess**2*R_guess**2/(2*(gamma+1.0))

        if Ps > freestream.pressure:
            x_min = x_guess
            x_guess = (x_min+x_max)/2.0


        elif Ps < freestream.pressure:
            x_max = x_guess
            x_guess = (x_min+x_max)/2.0

        print(x_guess, Ps, (eta_0 * ((2*radius)**2*np.pi*1*0.88/8)**0.25*x_guess**0.5))

        #x_guess = density*velocity**2/Ps*Rs/(2*(gamma+1.0))

    exit(print(Rs,x))    
    print()

    """

    return x