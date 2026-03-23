import trimesh
import os

# 1. Load the original intersecting assembly
if not os.path.exists("cubesat_assembly_final.stl"):
    print("Error: cubesat_assembly_final.stl not found.")
else:
    mesh = trimesh.load("cubesat_assembly_final.stl")

    print("Step 1: Voxelizing at 1mm resolution...")
    voxels = mesh.voxelized(pitch=0.001)

    print("Step 2: Filling internal voids...")
    voxels.fill()

    print("Step 3: Extracting surface via Marching Cubes...")
    manifold_mesh = voxels.marching_cubes

    print("Step 4: Applying Laplacian Smoothing...")
    # Explicitly using the smoothing module to avoid AttributeErrors
    trimesh.smoothing.filter_laplacian(manifold_mesh, iterations=10)

    # 5. Export the 'Healed & Smooth' version
    manifold_mesh.export("cubesat_assembly_smooth.stl")
    print("--- SUCCESS: cubesat_assembly_smooth.stl created ---")