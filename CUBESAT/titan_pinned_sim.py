import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl

def run_pinned_sim_reversed():
    mesh_file = "titan_perfect_assembly.xdmf"
    print(f"Loading perfect geometry: {mesh_file}...")
    
    with io.XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")

    V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    E, nu = 69e9, 0.33
    mu = fem.Constant(domain, E / (2.0 * (1.0 + nu)))
    lmbda = fem.Constant(domain, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

    coords = domain.geometry.x
    x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
    fdim = domain.topology.dim - 1

# ==========================================
    # 1. ANCHORING (Standard)
    # ==========================================
    # Anchor the BACK of the Cube (highest X coordinate)
    cube_anchor = mesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], x_max, atol=1e-3))

    def hinge_pins(x):
        r2 = 0.015**2 # 1.5cm radius bolt
        p1 = (x[0] + 0.305)**2 + (x[1] + 0.045)**2 + (x[2])**2 < r2
        p2 = (x[0] + 0.305)**2 + (x[1] - 0.045)**2 + (x[2])**2 < r2
        p3 = (x[0] + 0.305)**2 + (x[1])**2 + (x[2] + 0.045)**2 < r2
        p4 = (x[0] + 0.305)**2 + (x[1])**2 + (x[2] - 0.045)**2 < r2
        return p1 | p2 | p3 | p4

    flap_anchors = mesh.locate_entities_boundary(domain, fdim, hinge_pins)
    
    all_anchors = np.unique(np.concatenate([cube_anchor, flap_anchors]))
    bc_dofs = fem.locate_dofs_topological(V, fdim, all_anchors)
    bc = fem.dirichletbc(fem.Constant(domain, default_scalar_type((0.0, 0.0, 0.0))), bc_dofs, V)

    # ==========================================
    # 2. WIND LOAD (The "Pull" Trick)
    # ==========================================
    # Grab the same successful flap faces from the forward sim (lowest X coordinates)
    load_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: x[0] < (x_min + 0.002))
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mesh.meshtags(domain, fdim, load_facets, 1))
    
    # Change the wind vector to pull in the -X direction
    T = fem.Constant(domain, default_scalar_type((-25000.0, 0.0, 0.0)))
      

    # ==========================================
    # 3. MATH & SOLVE
    # ==========================================
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    def sigma(u): return lmbda * ufl.div(u) * ufl.Identity(3) + 2 * mu * ufl.sym(ufl.grad(u))

    a = ufl.inner(sigma(u), ufl.sym(ufl.grad(v))) * ufl.dx
    L = ufl.dot(T, v) * ds(1)

    print("Solving reversed aerodynamic load...")
    problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="pinned_rev_")
    uh = problem.solve()

    # Calculate Stress
    s = sigma(uh) - (1./3) * ufl.tr(sigma(uh)) * ufl.Identity(3)
    von_Mises = ufl.sqrt(3./2 * ufl.inner(s, s))
    
    V_stress = fem.functionspace(domain, ("Lagrange", 1))
    stress_expr = fem.Expression(von_Mises, V_stress.element.interpolation_points)
    vm_field = fem.Function(V_stress)
    vm_field.interpolate(stress_expr)

    # ==========================================
    # 4. CALCULATE DISPLACEMENT
    # ==========================================
    disp_array = uh.x.array.reshape((-1, 3))
    max_disp_meters = np.max(np.linalg.norm(disp_array, axis=1))
    max_disp_mm = max_disp_meters * 1000.0

    # Output & Save
    with io.XDMFFile(domain.comm, "titan_reversed_presentation.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        uh.name = "Displacement"
        vm_field.name = "VonMises_Stress"
        xdmf.write_function(uh)
        xdmf.write_function(vm_field)

    print("\n--- REVERSED STRUCTURAL RESULTS ---")
    print(f"Max Von Mises Stress: {np.max(vm_field.x.array):.2e} Pa")
    print(f"Max Flap Deflection:  {max_disp_meters:.6f} meters ({max_disp_mm:.2f} mm)")
    print("-----------------------------------\n")
    print("Open 'titan_reversed_presentation.xdmf' in ParaView.")

if __name__ == "__main__":
    run_pinned_sim_reversed()