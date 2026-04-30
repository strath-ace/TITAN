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
Created on Fri Feb  4 14:39:58 2022

@author: dee
"""
import os 
import subprocess
import numpy as np


try:
    from dolfin import *
except ModuleNotFoundError:
    pass

try:
    parameters["linear_algebra_backend"] = "PETSc"
except:
    pass

# Elasticity parameters
# E = 68e9




def build_nullspace(V, x):
    """Function to build null space for 3D elasticity"""

    # Create list of vectors for null space
    nullspace_basis = [x.copy() for i in range(6)]

    # Build translational null space basis
    V.sub(0).dofmap().set(nullspace_basis[0], 1.0);
    V.sub(1).dofmap().set(nullspace_basis[1], 1.0);
    V.sub(2).dofmap().set(nullspace_basis[2], 1.0);

    # Build rotational null space basis
    V.sub(0).set_x(nullspace_basis[3], -1.0, 1);
    V.sub(1).set_x(nullspace_basis[3],  1.0, 0);
    V.sub(0).set_x(nullspace_basis[4],  1.0, 2);
    V.sub(2).set_x(nullspace_basis[4], -1.0, 0);
    V.sub(2).set_x(nullspace_basis[5],  1.0, 1);
    V.sub(1).set_x(nullspace_basis[5], -1.0, 2);

    for x in nullspace_basis:
        x.apply("insert")

    # Create vector space basis and orthogonalize
    basis = VectorSpaceBasis(nullspace_basis)
    basis.orthonormalize()

    return basis


def sigma(v, E):
    nu = 0.3
    mu = E/(2.0*(1.0 + nu))
    lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))
    return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(len(v))

def lumpedProject(main_FS,f):
    vv = TestFunction(main_FS)
    lhs = assemble(inner(Constant(1.0),vv)*dx)
    rhs = assemble(inner(f,vv)*dx)
    uu = Function(main_FS)
    as_backend_type(uu.vector())\
        .vec().pointwiseDivide(as_backend_type(rhs).vec(),
                               as_backend_type(lhs).vec())
    return uu



# def run_FE(bcs, CG_VFS, force_func, ds, monitor_convergence = False, verbose = False):
#     # Define variational problem
#     u = TrialFunction(CG_VFS)
#     v = TestFunction(CG_VFS)
#     a = inner(sigma(u), grad(v))*dx
#     L = inner(force_func, v)*dx
#     # L = inner(force_func, v)*ds(2)
#     # L = dot(f, v) * ds(1)
#     # L = dot(force_func, v) * ds(force_surf_id)
    
#     if verbose:
#         print ('Assembling FE system', flush = True)
#     # Assemble system, applying boundary conditions and preserving
#     # symmetry)
#     A, b = assemble_system(a, L, bcs)
    
#     # Create solution function
#     displacement = Function(CG_VFS)
    
#     # Create near null space basis (required for smoothed aggregation
#     # AMG). The solution vector is passed so that it can be copied to
#     # generate compatible vectors for the nullspace.
#     if verbose:
#         print ('Building null space', flush = True)
#     null_space = build_nullspace(CG_VFS, displacement.vector())
    
#     if verbose:
#         print ('Setting PETSc parameters', flush = True)
#     # Attach near nullspace to matrix
#     as_backend_type(A).set_near_nullspace(null_space)
    
#     # Create PETSC smoothed aggregation AMG preconditioner and attach near
#     # null space
#     pc = PETScPreconditioner("petsc_amg")
    
#     # Use Chebyshev smoothing for multigrid
#     PETScOptions.set("mg_levels_ksp_type", "chebyshev")
#     PETScOptions.set("mg_levels_pc_type", "jacobi")
    
#     # Improve estimate of eigenvalues for Chebyshev smoothing
#     PETScOptions.set("mg_levels_esteig_ksp_type", "cg")
#     PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)
    
#     # Create CG Krylov solver and turn convergence monitoring on
#     solver = PETScKrylovSolver("cg", pc)
#     solver.parameters["monitor_convergence"] = monitor_convergence
    
#     # Set matrix operator
#     solver.set_operator(A);
#     if verbose:
#         print ('Solving FE', flush = True)
#     # Compute solution
#     solver.solve(displacement.vector(), b);
    
#     return displacement


def  run_FE(bcs, disp_funcs, CG_VFS, force_func, ds, monitor_convergence = False,
           verbose = False, E = 68e9):
    

    # Define variational problem
    u = TrialFunction(CG_VFS)
    v = TestFunction(CG_VFS)
    a = inner(sigma(u, E), grad(v))*dx
    L = inner(force_func, v)*dx

    if verbose:
        print ('Assembling FE system', flush = True)
    A, b = assemble_system(a, L, bcs[0])
    
    # Create solution function
    u1, u2, displacement = disp_funcs
    
    if verbose:
        print ('Building null space', flush = True)
    null_space = build_nullspace(CG_VFS, u1.vector())
    
    if verbose:
        print ('Setting PETSc parameters', flush = True)
    as_backend_type(A).set_near_nullspace(null_space)
    
    pc = PETScPreconditioner("petsc_amg")
    
    PETScOptions.set("mg_levels_ksp_type", "chebyshev")
    PETScOptions.set("mg_levels_pc_type", "jacobi")
    PETScOptions.set("mg_levels_esteig_ksp_type", "cg")
    PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)
    solver = PETScKrylovSolver("cg", pc)
    solver.parameters["monitor_convergence"] = monitor_convergence
    
    solver.set_operator(A);
    if verbose:
        print ('Solving FE', flush = True)
    solver.solve(u1.vector(), b);
    
    A, b = assemble_system(a, L, bcs[1])
    
    if verbose:
        print ('Building null space', flush = True)
    null_space = build_nullspace(CG_VFS, u2.vector())
    
    if verbose:
        print ('Setting PETSc parameters', flush = True)
    as_backend_type(A).set_near_nullspace(null_space)
    
    pc = PETScPreconditioner("petsc_amg")
    
    PETScOptions.set("mg_levels_ksp_type", "chebyshev")
    PETScOptions.set("mg_levels_pc_type", "jacobi")
    PETScOptions.set("mg_levels_esteig_ksp_type", "cg")
    PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)
    solver = PETScKrylovSolver("cg", pc)
    solver.parameters["monitor_convergence"] = monitor_convergence
    
    solver.set_operator(A);
    if verbose:
        print ('Solving FE', flush = True)
    solver.solve(u2.vector(), b);
    
    # solver.set_operator(A);
    # solver.solve(u2.vector(), b);
    # File('/Users/dee/Desktop/tmp/u1.pvd') << u1
    # File('/Users/dee/Desktop/tmp/u2.pvd') << u2

    displacement.vector()[:] = u1.vector()[:] + u2.vector()[:]
     
    return displacement


def call_MPI(num_mpi_cores, mesh_filename, subdomain_filename, force_filename,
             E, nu, rotation, verbose = False, output_folder = 'TITAN_sol'):
    # cwd = os.getcwd()
    cwd = '../FEniCS_library/'
    rx, ry, rz = rotation
    # print ('Calling MPI with %i cores'%(num_mpi_cores), flush = True)
    if verbose:
        print ('Calling MPI', flush = True)
    subprocess.run(["mpirun", "-np", "%i" % (num_mpi_cores), "python3", "FE_MPI.py",
                        "%s" % (mesh_filename), \
                        "%s" % (subdomain_filename), "%s" % (force_filename), \
                        "%.4f" % (E), "%.4f" % (nu), '%.2f'%(rx), '%.2f'%(ry),
                        '%.2f'%(rz), "%s"%(output_folder)
                        ], cwd=cwd)
    if verbose:
        print ('MPI complete', flush = True)

def run_FE_MPI(displacement_func_list, num_mpi_cores, mesh_filename,
               subdomain_filename, force_filename, rotation, verbose = False,
               E = 68e9, nu = 0.3, output_folder = 'TITAN_sol'):
    
    call_MPI(num_mpi_cores, mesh_filename, subdomain_filename, force_filename,
             E, nu, rotation, verbose = verbose, output_folder = output_folder)

    with XDMFFile(output_folder + '/u1.xdmf') as f_in:
        f_in.read_checkpoint(displacement_func_list[0],"displacement",0) 
    
    with XDMFFile(output_folder + '/u2.xdmf') as f_in:
        f_in.read_checkpoint(displacement_func_list[1],"displacement",0) 
        
    displacement_func_list[2].vector()[:] = (displacement_func_list[0].vector()[:] + displacement_func_list[1].vector()[:])
    
    return displacement_func_list[2]
    

def vonMises(displacement, DG_FS, E, project_type = 'normal'):
    
    s = sigma(displacement, E) - (1./3)*tr(sigma(displacement, E))*Identity(3)  # deviatoric stress
    von_Mises = sqrt(3./2*inner(s, s))
    if project_type == 'normal':
        von_Mises = project(von_Mises, DG_FS, solver_type = 'mumps')
    elif project_type == 'lumped':
        von_Mises = lumpedProject(DG_FS,von_Mises)

    return von_Mises

def maxvonMises(vonMises_func, subdom_arr, vol_ids, yield_stress):
    # use subdomain array to extract stress in specific part
    vm_arr = vonMises_func.vector()[:]
    # print(vm_arr)
    max_vms = []
    stress_dict = {}
    for vol_id in vol_ids:
        idxs = np.where(subdom_arr == vol_id)[0]
        max_vm = np.max(vm_arr[idxs])
        max_vms.append(max_vm)
        stress_dict[vol_id] = {}
        stress_dict[vol_id]['Max vm'] = (max_vm)
        stress_dict[vol_id]['Stress ratio'] = (max_vm/yield_stress) - 1
        
    # return max_vms, vol_ids
    return stress_dict


