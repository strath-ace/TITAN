import meshio
import glob
import numpy as np

def pack_perfect_assembly():
    print("Packing separate components into one container...")
    files = glob.glob("Geometry/*_3d.msh")
    
    all_points = []
    all_cells = []
    offset = 0
    
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
    
    # Notice we are NOT welding or snapping anything. 
    # Just putting the perfect geometries into one box.
    merged = meshio.Mesh(points=raw_points, cells=[("tetra", raw_cells)])
    meshio.write("titan_perfect_assembly.xdmf", merged)
    print("SUCCESS: Packed assembly saved to 'titan_perfect_assembly.xdmf'.")

if __name__ == "__main__":
    pack_perfect_assembly()