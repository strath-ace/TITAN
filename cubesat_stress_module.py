import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl

class CubeSatStressModule:
    def __init__(self, mesh_filename="cubesat_volumetric_60deg_up.xdmf"):
        self.comm = MPI.COMM_WORLD
        
        # 1. Load Mesh
        if self.comm.rank == 0:
            print(f"   [FEniCS] Loading CubeSat mesh from {mesh_filename}...")
            
        with io.XDMFFile(self.comm, mesh_filename, "r") as xdmf:
            self.domain = xdmf.read_mesh(name="Grid")

        self.gdim = self.domain.geometry.dim
        fdim = self.domain.topology.dim - 1

        # 2. Define Materials & Spaces (Aluminum 6061)
        E = 69e9  
        nu = 0.33
        self.mu = fem.Constant(self.domain, E / (2.0 * (1.0 + nu)))
        self.lmbda = fem.Constant(self.domain, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))

        self.V = fem.functionspace(self.domain, ("Lagrange", 2, (self.gdim,)))
        self.V_plot = fem.functionspace(self.domain, ("Lagrange", 1, (self.gdim,)))
        self.Q_plot = fem.functionspace(self.domain, ("Lagrange", 1))

        # ==========================================
        # 3. BOUNDARY CONDITIONS (Pin the Nose)
        # ==========================================
        def front_face(x):
            return np.isclose(x[2], -0.15, atol=1e-3)
            
        front_boundaries = mesh.locate_entities_boundary(self.domain, fdim, front_face)
        dofs_front = fem.locate_dofs_topological(self.V, fdim, front_boundaries)
        
        # The nose is bolted in place (0 displacement)
        u_zero = np.zeros(self.gdim, dtype=PETSc.ScalarType)
        self.bc = fem.dirichletbc(u_zero, dofs_front, self.V)

        # ==========================================
        # 4. WINDWARD AERODYNAMIC LOAD
        # ==========================================
        self.P_val = fem.Constant(self.domain, PETSc.ScalarType(0.0))
        n = ufl.FacetNormal(self.domain)
        
        # UFL Magic: Check if the surface normal is pointing downwards (into the wind)
        # If n[2] is less than 0, the face is pointing down. 
        windward_filter = ufl.conditional(ufl.lt(n[2], -0.01), 1.0, 0.0)
        
        # Apply to the ENTIRE boundary of the satellite
        ds = ufl.Measure("ds", domain=self.domain)

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        
        a = ufl.inner(2.0 * self.mu * ufl.sym(ufl.grad(u)) + self.lmbda * ufl.tr(ufl.sym(ufl.grad(u))) * ufl.Identity(self.gdim), ufl.grad(v)) * ufl.dx
        
        # The load now perfectly pushes inward (-n) ONLY on the windward faces
        L = ufl.dot(-self.P_val * windward_filter * n, v) * ds

        # ==========================================
        # 5. SOLVER SETUP
        # ==========================================
        self.u_sol = fem.Function(self.V, name="Displacement")
        
        solver_options = {"ksp_type": "cg", "pc_type": "gamg", "ksp_rtol": 1e-6}
        self.problem = LinearProblem(
            a, L, 
            bcs=[self.bc], 
            u=self.u_sol, 
            petsc_options=solver_options, 
            petsc_options_prefix="cubesat_solver_"
        )
        
        self.xdmf = io.XDMFFile(self.comm, "cubesat_coupled_results.xdmf", "w")
        self.xdmf.write_mesh(self.domain)
        
        self.u_plot = fem.Function(self.V_plot, name="Displacement")
        self.vm_plot = fem.Function(self.Q_plot, name="von_Mises")
        self.p_plot = fem.Function(self.Q_plot, name="Applied_Pressure")

    def solve_step(self, pressure_pa, time_s):
        self.P_val.value = pressure_pa
        self.problem.solve()
        
        s = 2.0 * self.mu * ufl.sym(ufl.grad(self.u_sol)) + self.lmbda * ufl.tr(ufl.sym(ufl.grad(self.u_sol))) * ufl.Identity(self.gdim)
        s_dev = s - (1./3) * ufl.tr(s) * ufl.Identity(self.gdim)
        von_mises = ufl.sqrt(3./2 * ufl.inner(s_dev, s_dev))

        self.u_plot.interpolate(self.u_sol)
        self.vm_plot.interpolate(fem.Expression(von_mises, self.Q_plot.element.interpolation_points))
        
        # --- UPDATE VISUALIZATION FOR PARAVIEW ---
        def windward_nodes(x):
            # Visually paint the nose and the flaps red in ParaView
            return np.logical_or(np.isclose(x[2], -0.15, atol=1e-2), x[2] >= 0.13)
            
        dofs_wind = fem.locate_dofs_geometrical(self.Q_plot, windward_nodes)
        self.p_plot.x.array[:] = 0.0
        self.p_plot.x.array[dofs_wind] = pressure_pa
        
        self.xdmf.write_function(self.u_plot, time_s)
        self.xdmf.write_function(self.vm_plot, time_s)
        self.xdmf.write_function(self.p_plot, time_s)
        
        if self.comm.rank == 0:
            max_disp = np.max(np.abs(self.u_plot.x.array))
            print(f"   [FEniCS] t={time_s:5.1f}s | P={pressure_pa:.4e} Pa | Max Deflection: {max_disp*1000:.4e} mm")

    def close(self):
        self.xdmf.close()