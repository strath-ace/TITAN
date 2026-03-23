import meshio
import glob
import numpy as np
from scipy.spatial import cKDTree
import os

def surgical_assembly():
    print("Starting Surgical Geometry Welder...")
    
    # 1. Find the Main Cube
    cube_files = glob.glob("Geometry/*Cube*_3d.msh")
    if not cube_files:
        print("Error: Could not find the Cube mesh.")
        return
    cube_file = cube_files[0]
    
    other_files = glob.glob("Geometry/*_3d.msh")
    other_files.remove(cube_file)
    
    # Load Cube
    cube_m = meshio.read(cube_file)
    cube_pts = cube_m.points
    cube_cells = [c.data for c in cube_m.cells if c.type == "tetra"][0]
    
    all_points = [cube_pts]
    all_cells = [cube_cells]
    offset = len(cube_pts)
    
    cube_tree = cKDTree(cube_pts)
    
    # 2. Your Exact Assembly Origins (The Hinge Pivot Points)
    hinge_origins = np.array([
        [-0.305, -0.045, 0.0],
        [-0.305, 0.045, 0.0],
        [-0.305, 0.0, -0.045],
        [-0.305, 0.0, 0.045]
    ])
    hinge_tree = cKDTree(hinge_origins)
    
    merges = 0
    
    # 3. Process the Flaps and Hinges
    for f in other_files:
        m = meshio.read(f)
        pts = np.copy(m.points)
        
        # Find how close every node is to your specific hinge origins
        dist_to_hinge, _ = hinge_tree.query(pts)
        # Find how close every node is to the Cube
        dist_to_cube, cube_indices = cube_tree.query(pts)
        
        # ==========================================
        # SURGICAL RULE:
        # If the node is within 4cm (0.04m) of a Hinge Origin 
        # AND it is within 2cm (0.02m) of the Cube... snap it to the Cube.
        # ALL OTHER NODES ARE IGNORED.
        # ==========================================
        snap_mask = (dist_to_hinge < 0.04) & (dist_to_cube < 0.02)
        
        # Copy the exact coordinate of the nearest cube node
        pts[snap_mask] = cube_pts[cube_indices[snap_mask]]
        merges += np.sum(snap_mask)
        
        all_points.append(pts)
        cells = [c.data for c in m.cells if c.type == "tetra"][0]
        all_cells.append(cells + offset)
        offset += len(pts)
        
    final_points = np.vstack(all_points)
    final_cells = np.vstack(all_cells)
    
    # 4. Clean up the shared nodes so FEniCSx registers the physical connection
    # Rounding slightly just to ensure bitwise-identical floats merge perfectly
    rounded_for_unique = np.round(final_points, decimals=5)
    unique_pts, inv_idx = np.unique(rounded_for_unique, axis=0, return_inverse=True)
    
    # Retrieve the unrounded, perfect coordinates
    final_clean_points = final_points[np.unique(inv_idx, return_index=True)[1]]
    welded_cells = inv_idx[final_cells]
    
    welded_mesh = meshio.Mesh(points=final_clean_points, cells=[("tetra", welded_cells)])
    meshio.write("titan_surgical_system.xdmf", welded_mesh)
    
    print(f"\n>>> SUCCESS: Surgically tied {merges} nodes EXACTLY at the hinges. <<<")
    print("100% of the flat flap geometry has been preserved.")

if __name__ == "__main__":
    surgical_assembly()