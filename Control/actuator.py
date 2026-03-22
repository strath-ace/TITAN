# Control/actuator.py
from Geometry.component import Component
import numpy as np


class ControlSurface(Component):

    def __init__(
        self,
        *args,
        hinge_origin=(0, 0, 0),   # LOCAL coords in STL frame
        hinge_axis=(0, 0, 1),     # LOCAL axis
        deflection=0.0,           # radians (config will pass float)
        deflection_limits=(-90*np.pi/180, 90*np.pi/180),
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Stored baseline nodes (local STL coords). Assembly will later overwrite
        # self.baseline_nodes with the *global* baseline nodes.
        self.baseline_nodes = self.mesh.nodes.copy()

        # Local hinge definition (Assembly converts these to global)
        self.hinge_origin_local = np.asarray(hinge_origin, float)
        self.hinge_axis_local   = np.asarray(hinge_axis, float)
        self.hinge_axis_local  /= np.linalg.norm(self.hinge_axis_local)

        # Will be replaced by Assembly with global versions
        self.hinge_origin = self.hinge_origin_local.copy()
        self.hinge_axis   = self.hinge_axis_local.copy()

        # Deflection limits are in radians
        self.deflection_limits = deflection_limits
        self.deflection        = float(deflection)


    # -----------------------------------------------------------
    # deflection setter
    # -----------------------------------------------------------
    def set_deflection(self, angle_radians: float):
        """Set deflection (in radians), clipped to limits."""
        self.deflection = float(np.clip(angle_radians, *self.deflection_limits))


    # -----------------------------------------------------------
    # rotation matrix
    # -----------------------------------------------------------
    @staticmethod
    def _rotation_matrix(axis, angle):
        axis = np.asarray(axis, float)
        axis /= np.linalg.norm(axis)
        ux, uy, uz = axis
        c, s = np.cos(angle), np.sin(angle)
        C = 1 - c
        return np.array([
            [c + ux*ux*C,     ux*uy*C - uz*s, ux*uz*C + uy*s],
            [uy*ux*C + uz*s,  c + uy*uy*C,    uy*uz*C - ux*s],
            [uz*ux*C - uy*s,  uz*uy*C + ux*s, c + uz*uz*C]
        ])


    # -----------------------------------------------------------
    # RIGID rotation from baseline global geometry
    # -----------------------------------------------------------
    def update_geometry(self, global_mesh, original_mesh):
        idx = self.node_index
        nodes0 = original_mesh[idx]

        hinge = np.asarray(self.hinge_origin, dtype=float).reshape(3,)
        axis  = np.asarray(self.hinge_axis, dtype=float).reshape(3,)
        axis /= np.linalg.norm(axis)

        angle = float(self.deflection)
        R = self._rotation_matrix(axis, angle)

        rotated = (nodes0 - hinge) @ R.T + hinge
        global_mesh.nodes[idx] = rotated
