import trimesh
import gmsh
import os

def mesh_exact_stl(stl_file, output_msh):
    print(f"Loading exact geometry from {stl_file}...")
    
    if not os.path.exists(stl_file):
        print("Error: File not found.")
        return

    # 1. Crack open the STL and find every single separated part
    mesh = trimesh.load(stl_file, force='mesh')
    parts = mesh.split(only_watertight=False)
    print(f"Detected {len(parts)} individual components (cube, flaps, hinges, etc.).")

    gmsh.initialize()
    gmsh.model.add("Titan_Exact_Geometry")

    # 2. Process every component one by one
    for i, part in enumerate(parts):
        temp_file = f"temp_component_{i}.stl"
        
        # Clean up this specific component just in case
        part.remove_degenerate_faces()
        part.export(temp_file)
        
        # Track which surfaces exist BEFORE we add the new part
        surfs_before = [s[1] for s in gmsh.model.getEntities(2)]
        
        # Merge the new component and rebuild its geometry
        gmsh.merge(temp_file)
        gmsh.model.mesh.classifySurfaces(40 * 3.14159 / 180., True, True, 3.14159)
        gmsh.model.mesh.createGeometry()
        
        # Track which surfaces exist AFTER we add the new part
        surfs_after = [s[1] for s in gmsh.model.getEntities(2)]
        
        # The difference is our new component's "skin"
        new_surfs = list(set(surfs_after) - set(surfs_before))
        
        # Force Gmsh to recognize this exact skin as a solid 3D volume
        if new_surfs:
            try:
                loop = gmsh.model.geo.addSurfaceLoop(new_surfs)
                gmsh.model.geo.addVolume([loop])
                print(f" -> Successfully wrapped Volume around component {i+1}/{len(parts)}")
            except Exception as e:
                print(f" -> Could not close Volume for component {i+1}: {e}")
                
        os.remove(temp_file) # Clean up temp file

    gmsh.model.geo.synchronize()

    # 3. Generate the exact mesh
    print("\nGenerating comprehensive 3D Volumetric Mesh...")
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.005) 
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.015)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)

    try:
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write(output_msh)
        
        num_tets = gmsh.model.mesh.getElements(3)[1]
        if num_tets:
            total_tets = sum(len(t) for t in num_tets)
            print(f"\n>>> SUCCESS: Exact STL meshed! {total_tets} Tetrahedrons generated across all {len(parts)} components. <<<")
        else:
            print("\n>>> FAILED: 0 Tetrahedrons generated. <<<")
            
    except Exception as e:
        print(f"Meshing failed: {e}")

    gmsh.finalize()

if __name__ == "__main__":
    mesh_exact_stl("cubesat.stl", "cubesat_exact_3d.msh")