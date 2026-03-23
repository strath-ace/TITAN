import meshio
import glob
import numpy as np

def weld_assembly():
    print("Loading all component meshes...")
    # Grab all the 3D meshes we generated earlier
    files = glob.glob("Geometry/*_3d.msh")
    
    if not files:
        print("Error: No .msh files found in Geometry folder.")
        return

    all_points = []
    all_cells = []
    offset = 0
    
    # Stack all points and tetrahedrons together
    for f in files:
        m = meshio.read(f)
        all_points.append(m.points)
        for c in m.cells:
            if c.type == "tetra":
                all_cells.append(c.data + offset)
                break
        offset += len(m.points)
        
    raw_points = np.vstack(all_points)
    raw_cells = np.vstack(all_cells)
    
    print(f"Total separate vertices loaded: {len(raw_points)}")
    
    # THE WELDING MAGIC:
    # Round coordinates to 3 decimal places (1mm grid).
    # This forces nodes on the flaps and hinges that are touching to become the EXACT same node.
    rounded = np.round(raw_points, decimals=3)
    unique_pts, inv_idx = np.unique(rounded, axis=0, return_inverse=True)
    
    # Rebuild the mesh using only the shared/welded nodes
    final_points = raw_points[np.unique(inv_idx, return_index=True)[1]]
    welded_cells = inv_idx[raw_cells]
    
    welded_mesh = meshio.Mesh(points=final_points, cells=[("tetra", welded_cells)])
    meshio.write("titan_full_system.xdmf", welded_mesh)
    
    print(f"\n>>> SUCCESS: Welded {len(files)} parts into ONE solid assembly! <<<")
    print(f"Reduced {len(raw_points)} isolated nodes down to {len(unique_pts)} physically connected nodes.")

if __name__ == "__main__":
    weld_assembly()