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
Created on Fri Mar 25 12:51:51 2022

@author: dee
"""
import numpy as np
import sys
import time
import pickle
from dolfin import *

t0 = time.time()

parameters["linear_algebra_backend"] = "PETSc"

mesh_filename = sys.argv[1]
subdomain_filename = (sys.argv[2])
force_filename = (sys.argv[3])
E = float(sys.argv[4])
nu = float(sys.argv[5])
rot_x = float(sys.argv[6])
rot_y = float(sys.argv[7])
rot_z = float(sys.argv[8])
output_folder = sys.argv[9]



# print ('Loading mesh', flush = True)
mesh = Mesh()
with XDMFFile(mesh_filename) as infile:
    infile.read(mesh)

DG_FS = FunctionSpace(mesh, "DG", 0)
CG_FS = FunctionSpace(mesh, "CG", 1)
CG_VFS = VectorFunctionSpace(mesh, "CG", 1)

mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile(subdomain_filename) as infile:
    infile.read(mvc)
subdomains = MeshFunction('size_t', mesh, mvc)

ds = Measure("ds")(subdomain_data=subdomains, domain=mesh)


with open(output_folder + 'vol_bc_dict.pkl', 'rb') as f:
    vol_bc_dict = pickle.load(f)
    
bcs_1 = []
bcs_2 = []
for obj, vol_bc_id in vol_bc_dict.items():
    vol_id, bc_id = vol_bc_id['Vol id'], vol_bc_id['BC id']
    if bc_id != '-1':
        if bc_id == '1':
            for j in range(3):
                bcs_1.append(DirichletBC(CG_VFS.sub(j), 0, subdomains, vol_id))
        else:
            for j in range(3):
                bcs_2.append(DirichletBC(CG_VFS.sub(j), 0, subdomains, vol_id))
            

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



force = Function(CG_VFS)
with XDMFFile(force_filename) as f_in:
    f_in.read_checkpoint(force,"force",0)

# Elasticity parameters
nu = 0.3
mu = E/(2.0*(1.0 + nu))
lmbda = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

# Stress computation
def sigma(v):
    return 2.0*mu*sym(grad(v)) + lmbda*tr(sym(grad(v)))*Identity(len(v))

# Define variational problem
u = TrialFunction(CG_VFS)
v = TestFunction(CG_VFS)
a = inner(sigma(u), grad(v))*dx
L = inner(force, v)*dx

print ('Assembling FE system', flush = True)
# Assemble system, applying boundary conditions and preserving
# symmetry)
A, b = assemble_system(a, L, bcs_1)

# Create solution function
u1 = Function(CG_VFS)
u2 = Function(CG_VFS)

# Create near null space basis (required for smoothed aggregation
# AMG). The solution vector is passed so that it can be copied to
# generate compatible vectors for the nullspace.
print ('Building null space', flush = True)
null_space = build_nullspace(CG_VFS, u1.vector())

print ('Setting PETSc parameters', flush = True)
# Attach near nullspace to matrix
as_backend_type(A).set_near_nullspace(null_space)

# Create PETSC smoothed aggregation AMG preconditioner and attach near
# null space
pc = PETScPreconditioner("petsc_amg")

# Use Chebyshev smoothing for multigrid
PETScOptions.set("mg_levels_ksp_type", "chebyshev")
PETScOptions.set("mg_levels_pc_type", "jacobi")

# Improve estimate of eigenvalues for Chebyshev smoothing
PETScOptions.set("mg_levels_esteig_ksp_type", "cg")
PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)

# Create CG Krylov solver and turn convergence monitoring on
solver = PETScKrylovSolver("cg", pc)
solver.parameters["monitor_convergence"] = False

# Set matrix operator
solver.set_operator(A);

# Compute solution
solver.solve(u1.vector(), b);


A, b = assemble_system(a, L, bcs_2)

# Create near null space basis (required for smoothed aggregation
# AMG). The solution vector is passed so that it can be copied to
# generate compatible vectors for the nullspace.
print ('Building null space', flush = True)
null_space = build_nullspace(CG_VFS, u2.vector())

print ('Setting PETSc parameters', flush = True)
# Attach near nullspace to matrix
as_backend_type(A).set_near_nullspace(null_space)

# Create PETSC smoothed aggregation AMG preconditioner and attach near
# null space
pc = PETScPreconditioner("petsc_amg")

# Use Chebyshev smoothing for multigrid
PETScOptions.set("mg_levels_ksp_type", "chebyshev")
PETScOptions.set("mg_levels_pc_type", "jacobi")

# Improve estimate of eigenvalues for Chebyshev smoothing
PETScOptions.set("mg_levels_esteig_ksp_type", "cg")
PETScOptions.set("mg_levels_ksp_chebyshev_esteig_steps", 50)

# Create CG Krylov solver and turn convergence monitoring on
solver = PETScKrylovSolver("cg", pc)
solver.parameters["monitor_convergence"] = False

# Set matrix operator
solver.set_operator(A);

# Compute solution
solver.solve(u2.vector(), b);

f_out = XDMFFile(mesh.mpi_comm(), output_folder + 'u1.xdmf')
f_out.write_checkpoint(u1, "displacement", 0, XDMFFile.Encoding.HDF5, False)
f_out.close()

f_out = XDMFFile(mesh.mpi_comm(),  output_folder + 'u2.xdmf')
f_out.write_checkpoint(u2, "displacement", 0, XDMFFile.Encoding.HDF5, False)
f_out.close()
