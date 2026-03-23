import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl

def run_global_sim():
    mesh_file = "titan_full_system.xdmf"
    print(f"Loading fully welded assembly: {mesh_file}...")
    
    with io.XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    # 1. Physics & Material (Aluminum 6061)
    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    E, nu = 69e9, 0.33
    mu = fem.Constant(domain, E / (2.0 * (1.0 + nu)))
    lmbda = fem.Constant(domain, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    coords = domain.geometry.x
    z_min, z_max = np.min(coords[:, 2]), np.max(coords[:, 2])
    z_thickness = z_max - z_min

    # 2. Anchor the Assembly (Bolt the back of the cube)
    fdim = domain.topology.dim - 1
    anchor_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[2], z_min, atol=1e-3))
    bc_dofs = fem.locate_dofs_topological(V, fdim, anchor_facets)
    bc = fem.dirichletbc(fem.Constant(domain, default_scalar_type((0, 0, 0))), bc_dofs, V)

    # 3. Apply Titan Wind (Pushing the top 5% of the geometry - the broad flat faces)
    load_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: x[2] > (z_max - (0.05 * z_thickness)))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mesh.meshtags(domain, fdim, load_facets, 1))
    
    # 25 kPa pushing down
    T = fem.Constant(domain, default_scalar_type((0, 0, -25000))) 

    # 4. Elasticity Math
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    def sigma(u): return lmbda * ufl.div(u) * ufl.Identity(3) + 2 * mu * ufl.sym(ufl.grad(u))

    a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx
    L = ufl.dot(T, v) * ds(1)

    # 5. Solve Global Assembly
    print("Solving global structural displacement... this may take a moment.")
    problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="global_")
    uh = problem.solve()

    # 6. Calculate Stress
    s = sigma(uh) - (1./3) * ufl.tr(sigma(uh)) * ufl.Identity(3)
    von_Mises = ufl.sqrt(3./2 * ufl.inner(s, s))
    
    V_stress = fem.functionspace(domain, ("Lagrange", 1))
    stress_expr = fem.Expression(von_Mises, V_stress.element.interpolation_points)
    vm_field = fem.Function(V_stress)
    vm_field.interpolate(stress_expr)

    # 7. Export ONE master file
    with io.XDMFFile(domain.comm, "titan_global_results.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        uh.name = "Displacement"
        vm_field.name = "VonMises_Stress"
        xdmf.write_function(uh)
        xdmf.write_function(vm_field)

    print("\n>>> GLOBAL SIMULATION COMPLETE <<<")
    print("Open 'titan_global_results.xdmf' in ParaView.")

if __name__ == "__main__":
    run_global_sim()