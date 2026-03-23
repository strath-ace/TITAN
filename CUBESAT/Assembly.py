import trimesh
import numpy as np
import os

# Now that we've renamed the folder, no extra space is needed
path = "Geometry"

def load_p(name):
    # This joins the path safely for Mac/Linux
    p = os.path.join(path, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Could not find: {p}")
    return trimesh.load(p)

# 1. Load Parts
cube = load_p('cubesatCube.stl')
# Hinges
h_l = load_p('cubesatHinge_L.stl')
h_r = load_p('cubesatHinge_R.stl')
h_t = load_p('cubesatHinge_T.stl')
h_b = load_p('cubesatHinge_B.stl')
# Flaps
f_l = load_p('cubesatFlap_L.stl')
f_r = load_p('cubesatFlap_R.stl')
f_t = load_p('cubesatFlap_T.stl')
f_b = load_p('cubesatFlap_B.stl')

# 2. Apply Transformations
angle = np.radians(-60)

# Flap_L: Axis (0,0,1), Origin (-.3, -.05, 0)
f_l.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1], [-.3, -.05, 0]))

# Flap_R: Axis (0,0,-1), Origin (-.3, .05, 0)
f_r.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, -1], [-.3, .05, 0]))

# Flap_T: Axis (0,-1,0), Origin (-.3, 0, -.05)
f_t.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, -1, 0], [-.3, 0, -.05]))

# Flap_B: Axis (0,1,0), Origin (-.3, 0, .05)
f_b.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 1, 0], [-.3, 0, .05]))

# 3. Combine into a single assembly
# We merge them so Gmsh can mesh them as one connected structural unit
full_list = [cube, h_l, h_r, h_t, h_b, f_l, f_r, f_t, f_b]
assembly = trimesh.util.concatenate(full_list)

# 4. Export
assembly.export('cubesat_assembly_final.stl')
print("--- SUCCESS: cubesat_assembly_final.stl created ---")