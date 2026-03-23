import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import os

def solve_part(mesh_path, is_cube=False):
    file_name = os.path.basename(mesh_path)
    print(f"--- Analyzing: {file_name} ---")
    
    with io.XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    # 1. Physics & Material
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    E, nu = 69e9, 0.33
    mu = fem.Constant(domain, E / (2.0 * (1.0 + nu)))
    lmbda = fem.Constant(domain, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    # 2. Boundary Conditions
    coords = domain.geometry.x
    z_min, z_max = np.min(coords[:, 2]), np.max(coords[:, 2])
    
    # Anchor logic: If it's the cube, bolt the back face. 
    # If it's a flap, we assume it's "held" at its origin (simplified for the meeting)
    fdim = domain.topology.dim - 1
    if is_cube:
        facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[2], z_min, atol=1e-3))
    else:
        # For flaps/hinges, we lock the side closest to the cube center
        facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], -0.05, atol=0.2))

    bc_dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(fem.Constant(domain, default_scalar_type((0, 0, 0))), bc_dofs, V)

 # ==========================================
    # 3. FIXED WIND LOAD (Hitting the Flat Faces)
    # ==========================================
    # Calculate the total thickness of the part
    z_thickness = z_max - z_min
    
    # Grab any surface that exists in the top 5% of the part 
    # (This guarantees we capture the entire broad flat face, not just an edge)
    def front_face(x):
        return x[2] > (z_max - (0.05 * z_thickness))
        
    load_facets = mesh.locate_entities_boundary(domain, fdim, front_face)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mesh.meshtags(domain, fdim, load_facets, 1))
    
    # THE WIND VECTOR: (X, Y, Z)
    # If the flat faces point UP (+Z), the wind should push DOWN (-Z).
    # If your flaps are actually oriented along the Y-axis, change this to (0, -25000, 0)
    T = fem.Constant(domain, default_scalar_type((0.0, 0.0, -25000.0)))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    def sigma(u): return lmbda * ufl.div(u) * ufl.Identity(3) + 2 * mu * ufl.sym(ufl.grad(u))

    a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx
    L = ufl.dot(T, v) * ds(1)

    # 4. Solve
    problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix=f"solve_{file_name.split('.')[0]}_")
    uh = problem.solve()

    # 5. FIXED Von Mises Stress Calculation
    s = sigma(uh) - (1./3) * ufl.tr(sigma(uh)) * ufl.Identity(3)
    von_Mises = ufl.sqrt(3./2 * ufl.inner(s, s))
    
    V_stress = fem.functionspace(domain, ("Lagrange", 1))
    # FIX: Removed the () from interpolation_points
    stress_expr = fem.Expression(von_Mises, V_stress.element.interpolation_points)
    vm_field = fem.Function(V_stress)
    vm_field.interpolate(stress_expr)

    # 6. Save
    out_name = mesh_path.replace(".xdmf", "_results.xdmf")
    with io.XDMFFile(domain.comm, out_name, "w") as xdmf:
        xdmf.write_mesh(domain)
        uh.name = "Displacement"
        vm_field.name = "VonMises_Stress"
        xdmf.write_function(uh)
        xdmf.write_function(vm_field)
    print(f"  -> Results saved to {out_name}")

if __name__ == "__main__":
    geometry_dir = "Geometry"
    # Process the Cube first
    solve_part(os.path.join(geometry_dir, "cubesatCube_3d.xdmf"), is_cube=True)
    
    # Then process all flaps and hinges
    for part in os.listdir(geometry_dir):
        if part.endswith("_3d.xdmf") and "Cube" not in part:
            solve_part(os.path.join(geometry_dir, part))

    print("\n>>> TOTAL ASSEMBLY ANALYSIS COMPLETE <<<")