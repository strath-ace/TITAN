import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem  # Fixed import for modern FEniCSx
import ufl

def run_titan_simulation():
    # ==========================================
    # 1. LOAD THE UNIVERSE
    # ==========================================
    mesh_file = "Geometry/cubesatCube_3d.xdmf"
    print(f"Loading {mesh_file} into FEniCSx...")
    
    with io.XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    # ==========================================
    # 2. DEFINE THE MATERIAL (Aerospace Aluminum)
    # ==========================================
    E = 69e9      # 69 GPa
    nu = 0.33     
    mu = E / (2.0 * (1.0 + nu))
    lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # ==========================================
    # 3. SET UP THE PHYSICS SPACE
    # ==========================================
    # Fixed: Vector Function Space with shape (3,) for 3D displacement
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

    # Define the "Anchor" (The back of the cube)
    # NOTE: If your cube is centered at 0, you might need -0.05 here!
    def back_face(x):
        return np.isclose(x[2], 0.0, atol=1e-3)
    
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, back_face)
    bc_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Fixed: Correct Rank for Dirichlet BC using fem.Constant vector
    u_zero = fem.Constant(domain, default_scalar_type((0.0, 0.0, 0.0)))
    bc = fem.dirichletbc(u_zero, bc_dofs, V)

    # ==========================================
    # 4. APPLY TITAN'S WIND (The Load)
    # ==========================================
    # Define the surface where wind hits (Front face)
    def front_face(x):
        # Assumes front face is at the other end of the cube
        return np.isclose(x[2], 0.1, atol=1e-3)

    # Create a boundary measure for the load
    front_facets = mesh.locate_entities_boundary(domain, fdim, front_face)
    mt = mesh.meshtags(domain, fdim, front_facets, 1)
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt)

    # 10 kPa hitting the front (pushing in -Z direction)
    wind_pressure = 10000.0 
    T = fem.Constant(domain, default_scalar_type((0.0, 0.0, -wind_pressure)))

    # ==========================================
    # 5. THE MATH (Linear Elasticity)
    # ==========================================
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    def epsilon(u):
        return ufl.sym(ufl.grad(u))

    def sigma(u):
        return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(T, v) * ds(1)  # Apply wind only to front face (subdomain 1)

# ==========================================
    # 6. SOLVE THE SYSTEM
    # ==========================================
    print("Solving the elasticity equations... hold on to your helmet.")
    
    # Added petsc_options_prefix argument
    problem = LinearProblem(
        a, L, bcs=[bc], 
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="titan_solve_"
    )
    uh = problem.solve()

 # ==========================================
    # 7. EXPORT RESULTS (XDMF version)
    # ==========================================
    print("Simulation complete! Exporting results to XDMF...")
    with io.XDMFFile(domain.comm, "titan_results.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)
    print("SUCCESS: Open 'titan_results.xdmf' in ParaView.")

if __name__ == "__main__":
    run_titan_simulation()