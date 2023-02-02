#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
"""
Created on Fri Feb  4 14:38:06 2022

@author: dee
"""


import numpy as np
import meshio
from Structural.FENICS.fe_functions_v1 import run_FE, vonMises, maxvonMises, run_FE_MPI
import pickle
# from fe_functions import run_FE, run_FE_MPI, vonMises, maxvonMises

try:
    from dolfin import *
except ModuleNotFoundError:
    pass

try:    
    parameters["linear_algebra_backend"] = "PETSc"
except:
    pass


def generate_xdmf_mesh(vtk_vol_mesh_filename, xdmf_mesh_filename):
    mesh = meshio.read(vtk_vol_mesh_filename)
    elems = mesh.cells[-1].data
    coords = mesh.points
    
    cells = [("tetra", elems)]
    meshio.write_points_cells(xdmf_mesh_filename,coords,cells)
    
def save_force_xdmf(force_func, mesh, filename = 'force.xdmf', verbose = False):
    if verbose: print('Saving force xdmf...', flush = True)
    f_out = XDMFFile(mesh.mpi_comm(), filename)
    f_out.write_checkpoint(force_func, "force", 0, XDMFFile.Encoding.HDF5, False)
    f_out.close()
    if verbose: print('Saved force xdmf', flush = True)
    
def create_bcs(subdomains, CG_VFS, assembly, verbose = False):
    bcs_1 = []
    bcs_2 = []
    if verbose: print ('Volume BC dict: ', vol_bc_dict, flush = True)
    for obj in assembly.objects:
#        vol_id, bc_id = vol_bc_id['Vol id'], vol_bc_id['BC id']
        vol_id = obj.id
        bc_id  = obj.fenics_bc_id
        if bc_id != '-1':
            if bc_id == '1':
                for j in range(3):
                    bcs_1.append(DirichletBC(CG_VFS.sub(j), 0, subdomains, vol_id))
            else:
                for j in range(3):
                    bcs_2.append(DirichletBC(CG_VFS.sub(j), 0, subdomains, vol_id))
                
    return bcs_1, bcs_2

def generate_subdomains(mesh,filename, save_subdomain = False, subdomains_filename = '', subdomains_3d_filename='', verbose = False):
    if verbose: print ('Generating subdomains from cell data...', flush = True)
    mvc = MeshValueCollection("size_t", mesh, 3)
    with XDMFFile(filename) as infile:
        infile.read(mvc, "Vol_tags")
    subdomains_3d = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    
    subdomains = MeshFunction("size_t", mesh,2, 0)
    subdomains_1d = MeshFunction("size_t", mesh,0, 0)
    for cell in cells(mesh):
        facets = cell.entities(2)
        for facet in facets:
                subdomains.array()[facet] = subdomains_3d.array()[cell.index()]
    if save_subdomain:
        save_subdomains(mesh, subdomains, subdomains_filename)
        save_subdomains(mesh, subdomains_3d, subdomains_3d_filename)
    if verbose: print ('Created subdomains', flush = True)
    return subdomains, subdomains_3d

def load_subdomains(mesh, subdomains_filename, subdomains_3d_filename, verbose = False):
    if verbose: print ('Loading subdomain xdmf...', flush = True)
    mvc = MeshValueCollection("size_t", mesh, 2)
    mvc_3d = MeshValueCollection("size_t", mesh, 3)
    with XDMFFile(subdomains_filename) as infile: 
        infile.read(mvc)
    
    with XDMFFile(subdomains_3d_filename) as infile: 
        infile.read(mvc_3d)
    subdomains = MeshFunction('size_t', mesh, mvc)
    subdomains_3d = MeshFunction('size_t', mesh, mvc_3d)
    if verbose: print ('Loaded subdomain xdmf', flush = True)
    return subdomains, subdomains_3d


def save_subdomains(mesh, subdomains, subdomains_filename):
    xdmf = XDMFFile(mesh.mpi_comm(), subdomains_filename)
    xdmf.write(subdomains)
    xdmf.close()


def run_fenics(forces, num_surf_points, vol_bc_dict, vol_mesh_filename = 'Benchmark.xdmf',  
               iteration = 0, verbose = False, case = 'benchmark', save_subdomains = False, 
               MPI = False, num_MPI_cores = 6, rotation = [0,0,0], E = 68e9, yield_stress = 100e6,
               output_folder = 'TITAN_sol', regen_subdomains = False, save_displacement = False,
               save_vonMises = False, assembly = [], options = [], inertial_forces = []):

    # Load mesh
    mesh_fenics = Mesh()
    vol_mesh_filename = options.output_folder+"/Surface_solution/ID_"+str(assembly.id)+"/volume.xdmf"

    if verbose: print ('Loading volume mesh', flush = True)
    with XDMFFile(vol_mesh_filename) as infile:
        infile.read(mesh_fenics)
    if verbose: print ('LOADED MESH', flush = True)
    # Create CG function space
    CG_FS = FunctionSpace(mesh_fenics, "CG", 1)
    DG_FS = FunctionSpace(mesh_fenics, 'DG', 0)
    CG_VFS = VectorFunctionSpace(mesh_fenics, "CG", 1)
    subdomains_filename = options.output_folder +"/Surface_solution/ID_"+str(assembly.id)+'/subdomains.xdmf'
    subdomains_3d_filename = options.output_folder +"/Surface_solution/ID_"+str(assembly.id)+"/subdomains_3d.xdmf"

    if verbose: print ('Computing subdomains...', flush = True)
        
    if MPI: save_subdomains = True

    if regen_subdomains:
        if verbose: print ('Regenerating subdomains', flush = True)
        subdomains, subdomains_3d = generate_subdomains(mesh_fenics,vol_mesh_filename, True, subdomains_filename, subdomains_3d_filename, verbose = verbose)
    else:
        if verbose: print('Reloading subdomain xdmfs', flush = True)
        subdomains, subdomains_3d = load_subdomains(mesh_fenics, subdomains_filename, subdomains_3d_filename, verbose = verbose)

    ds = Measure("ds")(subdomain_data=subdomains, domain=mesh_fenics)
    v2d_CG_FS = vertex_to_dof_map(CG_FS)
    
    d2v_CG_FS = dof_to_vertex_map(CG_FS)
    v2d_CG_VFS = vertex_to_dof_map(CG_VFS)
    
    force_func = Function(CG_VFS)
    force_x_func = Function(CG_FS)
    force_y_func = Function(CG_FS)
    force_z_func = Function(CG_FS)
    
    force_x_func_arr = force_x_func.vector()[:]
    force_y_func_arr = force_x_func_arr.copy()
    force_z_func_arr = force_x_func_arr.copy()
    
    force_x_func_arr[0:num_surf_points] = forces[0]
    force_y_func_arr[0:num_surf_points] = forces[1]
    force_z_func_arr[0:num_surf_points] = forces[2]
    
    force_x_func_arr += inertial_forces[:,0]
    force_y_func_arr += inertial_forces[:,1]
    force_z_func_arr += inertial_forces[:,2]

    force_x_func.vector()[:] = force_x_func_arr[d2v_CG_FS]
    
    force_func_arr = force_func.vector()[v2d_CG_VFS].reshape(-1,3)
    force_func_arr[:,0] = force_x_func_arr[d2v_CG_FS]
    force_func_arr[:,1] = force_y_func_arr[d2v_CG_FS]
    force_func_arr[:,2] = force_z_func_arr[d2v_CG_FS]
    
    force_func_arr = force_func_arr.reshape(-1,)

    force_func.vector()[:] = np.require(force_func_arr,requirements = 'C')

    u1 = Function(CG_VFS)
    u2 = Function(CG_VFS)
    u_tot = Function(CG_VFS)

    if not MPI:
        bcs_1, bcs_2 = create_bcs(subdomains, CG_VFS, verbose = verbose, assembly = assembly)
        u_tot = run_FE([bcs_1, bcs_2],[u1,u2,u_tot], CG_VFS, force_func, ds, 
                        monitor_convergence = True, verbose = verbose, E = E)
    """
    else:
        if verbose:
            print ('Using MPI to solve FE', flush = True)
        
        force_filename = output_folder + 'force.xdmf'
        save_force_xdmf(force_func, mesh_fenics, filename = force_filename, verbose = verbose)
        with open(output_folder + 'vol_bc_dict.pkl', 'wb') as f:
            pickle.dump(vol_bc_dict, f)
            
        u_tot = run_FE_MPI([u1,u2, u_tot],num_MPI_cores, vol_mesh_filename,
                            subdomains_filename, force_filename, rotation, 
                            E = E, output_folder = output_folder)
    """

    print ('displacement: ', np.max(u_tot.vector()[:]), flush = True)

    von_Mises = vonMises(u_tot, DG_FS, E, project_type = 'lumped')
    vM_arr = von_Mises.vector()[:]
    
    save_displacement = True
    save_vonMises = True

    if save_displacement:
        File(options.output_folder +"/Surface_solution/ID_"+str(assembly.id) + '/displacement_%i.pvd'%(iteration)) << u_tot
    if save_vonMises:
        File(options.output_folder +"/Surface_solution/ID_"+str(assembly.id) + '/vonMises_%i.pvd'%(iteration)) << von_Mises

    volume_ids = []
    for obj in assembly.objects:
        volume_ids.append(obj.id) 

    #yield_stress = 1e6 #200 MPa, aluminium

    max_vm_dict = maxvonMises(von_Mises, subdomains_3d.array(), volume_ids, yield_stress)    
    disp_arr = u_tot.vector()[v2d_CG_VFS].reshape(-1,3)
    
    surf_displacement = disp_arr[0:num_surf_points]
    surf_vM = vM_arr[0:num_surf_points]

    force_arr = force_func.vector()[v2d_CG_VFS].reshape(-1,3)
    surf_force = force_arr[0:num_surf_points]
    
    max_x_disp, max_y_disp, max_z_disp = np.max(np.abs(disp_arr[:,0])), np.max(np.abs(disp_arr[:,1])), np.max(np.abs(disp_arr[:,2]))
    max_displacements = [max_x_disp, max_y_disp, max_z_disp]

    return surf_displacement, surf_vM, max_vm_dict, surf_force, disp_arr, max_displacements, vM_arr



######### DEBUGGING CODE ##########
# import time
# # forces = np.load('../UnitTests/tmp_force.npy', allow_pickle=False)
# # num_surf_points = forces[0].shape[0]
# case = 'ATV'

# t1 = time.time()
# run_fenics([], 0, vol_mesh_filename = '../UnitTests/benchmark/Vol_mesh_id_1.xdmf',case = case, verbose=True, save_subdomains=False, MPI= False)
# # run_fenics(forces, num_surf_points, vol_mesh_filename = '../UnitTests/benchmark.xdmf',case = case, verbose=True, save_subdomains=False, MPI= False)
# # surf_displacement, surf_vM, stress_ratio = run_fenics(forces, num_surf_points, vol_mesh_filename = '../UnitTests/benchmark.xdmf',case = case, verbose=True, save_subdomains=False, MPI= False)
# # print (stress_ratio)
# print (time.time() - t1)
