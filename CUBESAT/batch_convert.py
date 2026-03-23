import meshio
import os
import glob

def convert_msh_to_xdmf(msh_path):
    # Create the output filename
    xdmf_path = msh_path.replace(".msh", ".xdmf")
    file_name = os.path.basename(msh_path)
    
    print(f"Filtering and converting {file_name}...")

    try:
        # 1. Read the messy Gmsh file
        msh = meshio.read(msh_path)

        # 2. Hunt down ONLY the 3D tetrahedra
        tetra_cells = None
        for cell in msh.cells:
            if cell.type == "tetra":
                tetra_cells = cell.data
                break
        
        if tetra_cells is None:
            print(f"  -> WARNING: No tetrahedra found in {file_name}. Skipping.")
            return

        # 3. Create a brand new, clean mesh containing ONLY the 3D volume
        clean_mesh = meshio.Mesh(
            points=msh.points,
            cells=[("tetra", tetra_cells)]
        )

        # 4. Save it as an XDMF file
        meshio.write(xdmf_path, clean_mesh)
        print(f"  -> SUCCESS: Saved as {os.path.basename(xdmf_path)}")
        
    except Exception as e:
        print(f"  -> Failed to convert {file_name}: {e}")

if __name__ == "__main__":
    target_folder = "Geometry"
    
    if not os.path.exists(target_folder):
        print(f"Error: Could not find the folder '{target_folder}'.")
    else:
        # Find all the .msh files we just generated
        msh_files = glob.glob(os.path.join(target_folder, "*_3d.msh"))
        
        if not msh_files:
            print(f"No .msh files found in {target_folder}/.")
        else:
            print(f"Found {len(msh_files)} meshes. Starting batch conversion...\n")
            for msh in msh_files:
                convert_msh_to_xdmf(msh)
            print("\n--- BATCH XDMF CONVERSION COMPLETE ---")