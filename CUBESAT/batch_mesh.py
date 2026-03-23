import gmsh
import os
import glob

def mesh_individual_part(stl_path):
    msh_path = stl_path.replace(".stl", "_3d.msh")
    file_name = os.path.basename(stl_path)
    
    print(f"\n--- Processing: {file_name} ---")
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0) 
    gmsh.model.add(file_name)
    
    try:
        gmsh.merge(stl_path)
        
        # 1. Rebuild geometry
        gmsh.model.mesh.classifySurfaces(40 * 3.14159 / 180., True, True, 3.14159)
        gmsh.model.mesh.createGeometry()
        
        # 2. Force Volume Creation
        surfaces = gmsh.model.getEntities(2)
        surface_tags = [t[1] for t in surfaces]
        
        if surface_tags:
            loop = gmsh.model.geo.addSurfaceLoop(surface_tags)
            gmsh.model.geo.addVolume([loop])
            gmsh.model.geo.synchronize()
        else:
            print(f"  -> WARNING: No surfaces found in {file_name}")
            gmsh.finalize()
            return

        # ==========================================
        # 3. HIGH-RESOLUTION MESH SETTINGS
        # ==========================================
        # Drop the base sizes down (e.g., 1mm minimum, 4mm maximum)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0025) 
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.008) 
        
        # Turn on smart curvature refinement for the hinges
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
        # Tell it to use at least 20 elements around a full circle
        gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 20)
        
        gmsh.option.setNumber("Mesh.Algorithm3D", 1) 

        # 4. Generate 3D
        gmsh.model.mesh.generate(3)
        
        # 5. Save
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(msh_path)
        
        # 6. Verify
        num_tets = gmsh.model.mesh.getElements(3)[1]
        if len(num_tets) > 0:
            print(f"  -> SUCCESS: Created {len(num_tets[0])} Highly Refined Tetrahedrons!")
        else:
            print(f"  -> FAILED: 0 Tetrahedrons generated.")
            
    except Exception as e:
        print(f"  -> Meshing failed with error: {e}")
        
    gmsh.finalize()

if __name__ == "__main__":
    target_folder = "Geometry"
    
    if not os.path.exists(target_folder):
        print(f"Error: Could not find the folder '{target_folder}'.")
    else:
        stl_files = glob.glob(os.path.join(target_folder, "*.stl"))
        
        if not stl_files:
            print(f"No STL files found in {target_folder}/.")
        else:
            print(f"Found {len(stl_files)} STL files. Starting high-res batch mesher...")
            for stl in stl_files:
                mesh_individual_part(stl)
            print("\n--- HIGH-RES BATCH MESHING COMPLETE ---")