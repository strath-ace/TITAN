import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl

class CubeSatStressModule:
    def __init__(self, mesh_filename="titan_perfect_assembly.xdmf"):
        self.comm = MPI.COMM_WORLD
        
        # 1. Load the Perfect Multi-Part Mesh
        if self.comm.rank == 0:
            print(f"   [FEniCSx] Loading CubeSat assembly from {mesh_filename}...")
            
        with io.XDMFFile(self.comm, mesh_filename, "r") as xdmf:
            self.domain = xdmf.read_mesh(name="Grid")

        self.gdim = self.domain.geometry.dim

        # 2. Material Properties (Aluminum 6061)
        E = 69e9 
        nu = 0.33
        self.mu = fem.Constant(self.domain, default_scalar_type(E / (2.0 * (1.0 + nu))))
        self.lmbda = fem.Constant(self.domain, default_scalar_type(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))))

        # Function Spaces
        self.V = fem.functionspace(self.domain, ("Lagrange", 1, (self.gdim,)))
        self.Q_plot = fem.functionspace(self.domain, ("Lagrange", 1)) # For Stress & Pressure

        # --- VIRTUAL PINNING (Boundary Conditions) ---
        coords = self.domain.geometry.x
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        fdim = self.domain.topology.dim - 1

        # Anchor 1: Back of the Cube
        cube_anchor = mesh.locate_entities_boundary(self.domain, fdim, lambda x: np.isclose(x[0], x_max, atol=1e-3))

        # Anchor 2: The 4 Hinge Origins
        def hinge_pins(x):
            r2 = 0.015**2 # 1.5cm radius bolt
            p1 = (x[0] + 0.305)**2 + (x[1] + 0.045)**2 + (x[2])**2 < r2
            p2 = (x[0] + 0.305)**2 + (x[1] - 0.045)**2 + (x[2])**2 < r2
            p3 = (x[0] + 0.305)**2 + (x[1])**2 + (x[2] + 0.045)**2 < r2
            p4 = (x[0] + 0.305)**2 + (x[1])**2 + (x[2] - 0.045)**2 < r2
            return p1 | p2 | p3 | p4

        flap_anchors = mesh.locate_entities_boundary(self.domain, fdim, hinge_pins)
        all_anchors = np.unique(np.concatenate([cube_anchor, flap_anchors]))
        bc_dofs = fem.locate_dofs_topological(self.V, fdim, all_anchors)
        self.bc = fem.dirichletbc(np.zeros(3, dtype=PETSc.ScalarType), bc_dofs, self.V)

        # --- AERODYNAMIC LOAD (The "Pull" Vector) ---
        # Using the strict 2mm tolerance to ensure perfect symmetry across all 4 flaps
        load_facets = mesh.locate_entities_boundary(self.domain, fdim, lambda x: np.isclose(x[0], x_min, atol=0.002))
        facet_tags = mesh.meshtags(self.domain, fdim, load_facets, 1)
        ds = ufl.Measure("ds", domain=self.domain, subdomain_data=facet_tags)

        # Visualization logic: Find nodes on the front face to paint them with pressure
        self.dofs_load_plot = fem.locate_dofs_topological(self.Q_plot, fdim, load_facets)

        # Dynamic Pressure Constant (starts at 0)
        self.P_val = fem.Constant(self.domain, PETSc.ScalarType(0.0))
        # Vector acting along X axis
        T = ufl.as_vector((self.P_val, 0.0, 0.0))

        # --- WEAK FORM ---
        u, v = ufl.TrialFunction(self.V), ufl.TestFunction(self.V)
        def epsilon(u): return ufl.sym(ufl.grad(u))
        def sigma(u): return self.lmbda * ufl.div(u) * ufl.Identity(3) + 2 * self.mu * epsilon(u)
        
        a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        L = ufl.dot(T, v) * ds(1)

        # --- SOLVER SETUP ---
        self.u_sol = fem.Function(self.V, name="Displacement")
        # Direct LU solver for rock-solid stability on multi-part assemblies
        self.problem = LinearProblem(a, L, bcs=[self.bc], u=self.u_sol, 
                                     petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, 
                                     petsc_options_prefix="titan_")

        # --- EXPORT SETUP ---
        self.xdmf = io.XDMFFile(self.comm, "titan_coupled_reentry.xdmf", "w")
        self.xdmf.write_mesh(self.domain)
        
        self.vm_plot = fem.Function(self.Q_plot, name="VonMises_Stress")
        self.p_plot = fem.Function(self.Q_plot, name="Applied_Pressure")

    def solve_step(self, pressure_pa, time_s):
        """Called by TITAN at every time step."""
        # 1. Update the Physics Load
        # We enforce a negative value so it pulls the flaps forward (-X direction)
        self.P_val.value = -abs(pressure_pa)
        
        # 2. Solve Elasticity
        self.problem.solve()
        
        # 3. Calculate Von Mises Stress
        s = 2.0 * self.mu * ufl.sym(ufl.grad(self.u_sol)) + self.lmbda * ufl.tr(ufl.sym(ufl.grad(self.u_sol))) * ufl.Identity(self.gdim)
        s_dev = s - (1./3) * ufl.tr(s) * ufl.Identity(self.gdim)
        von_mises = ufl.sqrt(3./2 * ufl.inner(s_dev, s_dev))
        self.vm_plot.interpolate(fem.Expression(von_mises, self.Q_plot.element.interpolation_points))
        
        # 4. Update Pressure Visualization Field
        self.p_plot.x.array[:] = 0.0
        self.p_plot.x.array[self.dofs_load_plot] = pressure_pa
        
        # 5. Write Time Step to XDMF
        self.xdmf.write_function(self.u_sol, time_s)
        self.xdmf.write_function(self.vm_plot, time_s)
        self.xdmf.write_function(self.p_plot, time_s)
        
        # 6. Terminal Output for the Engineer
        if self.comm.rank == 0:
            disp_array = self.u_sol.x.array.reshape((-1, 3))
            max_disp_mm = np.max(np.linalg.norm(disp_array, axis=1)) * 1000.0
            max_stress_mpa = np.max(self.vm_plot.x.array) / 1e6
            
            status = "OK" if max_stress_mpa < 276.0 else "YIELD!"
            print(f"   [FEniCSx] t={time_s:5.1f}s | P_dyn={pressure_pa:7.1f} Pa | Stress={max_stress_mpa:5.1f} MPa | Deflection={max_disp_mm:5.2f} mm | [{status}]")

    def close(self):
        self.xdmf.close()
        if self.comm.rank == 0:
            print("   [FEniCSx] Trajectory complete. Results saved to 'titan_coupled_reentry.xdmf'.")